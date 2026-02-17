"""Train the GridEncoder (Stream A belief encoder).

Loss:
  1. Reconstruction: predict next agent_pos for each action (MSE)
  2. Auxiliary: predict goal direction from belief (CrossEntropy, 5 classes)

The encoder must learn a belief that is useful for downstream scoring,
not just copy the input.  Predicting next-position forces it to encode
obstacle layout; predicting goal direction forces position awareness.

Usage:
    python -m train.train_a --data train/data/grid_expert.json --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grid_encoder import GridEncoder, encode_grid_observation, GRID_MAX


ACTIONS = ("up", "down", "left", "right")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


class ReconstructionHead(nn.Module):
    """Predict next_pos_delta for each action from belief."""

    def __init__(self, belief_dim: int = 32, n_actions: int = 4):
        super().__init__()
        # For each action, predict (dx, dy) delta
        self.heads = nn.ModuleList([
            nn.Linear(belief_dim, 2) for _ in range(n_actions)
        ])

    def forward(self, belief: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """belief: (B, 32), action_idx: (B,) -> (B, 2)"""
        batch_size = belief.size(0)
        # Gather predictions for the taken action
        all_preds = torch.stack([h(belief) for h in self.heads], dim=1)  # (B, 4, 2)
        idx = action_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)     # (B, 1, 2)
        return all_preds.gather(1, idx).squeeze(1)                       # (B, 2)


class GoalDirectionHead(nn.Module):
    """Predict goal direction (5 classes: up/down/left/right/at_goal)."""

    def __init__(self, belief_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(belief_dim, 5)

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.fc(belief)


def goal_direction_label(agent_pos, goal_pos, width, height):
    """Compute goal direction class: 0=up, 1=down, 2=left, 3=right, 4=at_goal.

    When goal_pos is (-1, -1), we still use the hidden goal for training
    (we know it from the expert data).  Returns the dominant direction.
    """
    ax, ay = agent_pos
    gx, gy = goal_pos

    if ax == gx and ay == gy:
        return 4  # at_goal

    dx = gx - ax
    dy = gy - ay

    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 2  # right or left
    else:
        return 1 if dy > 0 else 0  # down or up


def prepare_data(data_path: str):
    """Load JSON transitions and prepare tensors."""
    with open(data_path) as f:
        transitions = json.load(f)

    obs_tensors = []
    action_tensors = []
    delta_tensors = []
    direction_tensors = []

    for tr in transitions:
        obs = encode_grid_observation(
            width=tr["width"],
            height=tr["height"],
            agent_pos=tuple(tr["agent_pos"]),
            goal_pos=tuple(tr["goal_pos"]),
            obstacles=[tuple(o) for o in tr["obstacles"]],
            hint=tr.get("hint"),
        )
        obs_tensors.append(obs)

        action_idx = ACTION_TO_IDX.get(tr["action"], 0)
        action_tensors.append(action_idx)

        # Position delta (normalised)
        norm = float(max(tr["width"], tr["height"], 1))
        dx = (tr["next_agent_pos"][0] - tr["agent_pos"][0]) / norm
        dy = (tr["next_agent_pos"][1] - tr["agent_pos"][1]) / norm
        delta_tensors.append([dx, dy])

        # Goal direction (using agent_pos relative to grid center as proxy
        # when goal is hidden, since we want general direction awareness)
        goal_x = tr["goal_pos"][0] if tr["goal_pos"][0] >= 0 else tr["width"] // 2
        goal_y = tr["goal_pos"][1] if tr["goal_pos"][1] >= 0 else tr["height"] // 2
        direction = goal_direction_label(
            tuple(tr["agent_pos"]), (goal_x, goal_y),
            tr["width"], tr["height"],
        )
        direction_tensors.append(direction)

    X = torch.stack(obs_tensors)
    actions = torch.tensor(action_tensors, dtype=torch.long)
    deltas = torch.tensor(delta_tensors, dtype=torch.float32)
    directions = torch.tensor(direction_tensors, dtype=torch.long)

    return X, actions, deltas, directions


def train(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    belief_dim: int = 32,
    hidden: int = 64,
    checkpoint_path: str = "train/checkpoints/grid_encoder.pt",
):
    print(f"Loading data from {data_path}...")
    X, actions, deltas, directions = prepare_data(data_path)
    print(f"  {X.shape[0]} samples, obs_dim={X.shape[1]}")

    # Train/val split (90/10)
    n = X.shape[0]
    n_val = max(1, n // 10)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = TensorDataset(X[train_idx], actions[train_idx],
                             deltas[train_idx], directions[train_idx])
    val_ds = TensorDataset(X[val_idx], actions[val_idx],
                           deltas[val_idx], directions[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Models
    encoder = GridEncoder(obs_dim=X.shape[1], belief_dim=belief_dim, hidden=hidden)
    recon_head = ReconstructionHead(belief_dim)
    direction_head = GoalDirectionHead(belief_dim)

    params = (
        list(encoder.parameters())
        + list(recon_head.parameters())
        + list(direction_head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        encoder.train()
        recon_head.train()
        direction_head.train()

        train_loss_sum = 0.0
        for batch_x, batch_a, batch_d, batch_dir in train_dl:
            belief = encoder(batch_x)

            pred_delta = recon_head(belief, batch_a)
            loss_recon = mse_loss(pred_delta, batch_d)

            pred_dir = direction_head(belief)
            loss_dir = ce_loss(pred_dir, batch_dir)

            loss = loss_recon + 0.5 * loss_dir

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)

        # --- Validate ---
        encoder.eval()
        recon_head.eval()
        direction_head.eval()

        val_loss_sum = 0.0
        val_dir_correct = 0
        val_n = 0

        with torch.no_grad():
            for batch_x, batch_a, batch_d, batch_dir in val_dl:
                belief = encoder(batch_x)

                pred_delta = recon_head(belief, batch_a)
                loss_recon = mse_loss(pred_delta, batch_d)

                pred_dir = direction_head(belief)
                loss_dir = ce_loss(pred_dir, batch_dir)

                val_loss_sum += (loss_recon + 0.5 * loss_dir).item() * batch_x.size(0)
                val_dir_correct += (pred_dir.argmax(1) == batch_dir).sum().item()
                val_n += batch_x.size(0)

        train_loss = train_loss_sum / len(train_ds)
        val_loss = val_loss_sum / val_n
        dir_acc = val_dir_correct / val_n

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"dir_acc={dir_acc:.1%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = Path(checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), ckpt_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Print model size
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params:,}")

    return encoder


def main():
    parser = argparse.ArgumentParser(description="Train GridEncoder (Stream A)")
    parser.add_argument("--data", type=str, default="train/data/grid_expert.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--belief-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--checkpoint", type=str,
                        default="train/checkpoints/grid_encoder.pt")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        belief_dim=args.belief_dim,
        hidden=args.hidden,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()

"""Train the ActionValueNet for Neural Agent C.

Trains on BFS-labelled action values to learn obstacle-aware scoring.

Loss: MSE(predicted_score, bfs_label)

The network learns to replicate the BFS oracle's action values, which
account for obstacles.  On obstacle-free grids, BFS = Manhattan, so the
network doesn't contradict the baseline.  On obstacle-rich grids, it
learns the detour-aware scores that Manhattan misses.

Usage:
    python -m train.train_c --data train/data/expert_c.json --epochs 100
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

from models.action_value_net import ActionValueNet, extract_features


def prepare_data(data_path: str):
    """Load JSON samples and prepare feature tensors + labels."""
    with open(data_path) as f:
        samples = json.load(f)

    features = []
    labels = []

    for s in samples:
        feat = extract_features(
            agent_pos=tuple(s["agent_pos"]),
            next_pos=tuple(s["next_pos"]),
            goal_pos=tuple(s["goal_pos"]),
            obstacles=[tuple(o) for o in s["obstacles"]],
            width=s["width"],
            height=s["height"],
        )
        features.append(feat)
        labels.append(s["bfs_label"])

    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return X, y


def train(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 64,
    checkpoint_path: str = "train/checkpoints/action_value_net.pt",
):
    print(f"Loading data from {data_path}...")
    X, y = prepare_data(data_path)
    print(f"  {X.shape[0]} samples, feature_dim={X.shape[1]}")

    # Stats
    disagree_mask = torch.abs(y.squeeze() - 0.0) > 0.01  # non-trivial labels
    print(f"  Non-zero labels: {disagree_mask.sum().item()} ({disagree_mask.float().mean():.1%})")

    # Train/val split (90/10)
    n = X.shape[0]
    n_val = max(1, n // 10)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    net = ActionValueNet(input_dim=X.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        # --- Train ---
        net.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in train_dl:
            pred = net(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_x.size(0)

        # --- Validate ---
        net.eval()
        val_loss_sum = 0.0
        val_correct_sign = 0
        val_n = 0

        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                pred = net(batch_x)
                val_loss_sum += loss_fn(pred, batch_y).item() * batch_x.size(0)

                # Sign accuracy: does predicted sign match BFS label sign?
                pred_sign = (pred > 0.01).float() - (pred < -0.01).float()
                true_sign = (batch_y > 0.01).float() - (batch_y < -0.01).float()
                val_correct_sign += (pred_sign == true_sign).sum().item()
                val_n += batch_x.size(0)

        train_loss = train_loss_sum / len(train_ds)
        val_loss = val_loss_sum / val_n
        sign_acc = val_correct_sign / val_n

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"sign_acc={sign_acc:.1%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_path = Path(checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), ckpt_path)

    print(f"\nBest val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Checkpoint saved to: {checkpoint_path}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"ActionValueNet params: {n_params:,}")

    return net


def main():
    parser = argparse.ArgumentParser(description="Train ActionValueNet (Neural C)")
    parser.add_argument("--data", type=str, default="train/data/expert_c.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--checkpoint", type=str,
                        default="train/checkpoints/action_value_net.pt")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()

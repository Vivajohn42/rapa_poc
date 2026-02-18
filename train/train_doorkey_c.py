"""Train DoorKeyActionValueNet on BFS-labelled data.

Same training loop as train_c.py: MSE loss, 90/10 train/val split,
Adam optimizer, sign accuracy tracking.

Usage:
    python -m train.train_doorkey_c --data train/data/expert_doorkey.json --epochs 100
    python -m train.train_doorkey_c --epochs 50 --lr 0.0005  # custom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.doorkey_action_value_net import (
    DoorKeyActionValueNet,
    extract_doorkey_features,
)


def prepare_data(data_path: str):
    """Load JSON samples and prepare feature tensors + labels."""
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    print(f"  {len(data)} samples loaded")

    features_list = []
    labels_list = []

    for sample in data:
        feat = extract_doorkey_features(
            agent_pos=tuple(sample["agent_pos"]),
            agent_dir=sample["agent_dir"],
            next_pos=tuple(sample["next_pos"]),
            next_dir=sample["next_dir"],
            target_pos=tuple(sample["target_pos"]),
            obstacles=[tuple(o) for o in sample["obstacles"]],
            width=sample["width"],
            height=sample["height"],
            phase=sample["phase"],
            carrying_key=sample["carrying_key"],
        )
        features_list.append(feat)
        labels_list.append(sample["bfs_label"])

    X = torch.stack(features_list)
    y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    print(f"  Features: {X.shape}, Labels: {y.shape}")
    print(f"  Label range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Label mean: {y.mean():.4f}, std: {y.std():.4f}")

    return X, y


def train(
    data_path: str = "train/data/expert_doorkey.json",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 64,
    checkpoint_path: str = "train/checkpoints/doorkey_action_value_net.pt",
):
    """Train DoorKeyActionValueNet on BFS labels."""
    X, y = prepare_data(data_path)

    # 90/10 train/val split
    n = len(X)
    n_val = max(n // 10, 1)
    n_train = n - n_val

    perm = torch.randperm(n)
    X, y = X[perm], y[perm]

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print(f"\nTrain: {n_train}, Val: {n_val}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    net = DoorKeyActionValueNet(hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"DoorKeyActionValueNet params: {n_params:,}")
    print(f"Training for {epochs} epochs...\n")

    best_val_loss = float("inf")
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Train
        net.train()
        train_loss = 0.0
        n_batches = 0
        for bx, by in train_dl:
            pred = net(bx)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate
        net.eval()
        val_loss = 0.0
        n_correct_sign = 0
        n_total = 0
        n_val_batches = 0
        with torch.no_grad():
            for bx, by in val_dl:
                pred = net(bx)
                val_loss += loss_fn(pred, by).item()
                n_val_batches += 1

                # Sign accuracy: does predicted sign match label sign?
                pred_pos = pred > 0.01
                label_pos = by > 0.01
                n_correct_sign += (pred_pos == label_pos).sum().item()
                n_total += by.numel()

        val_loss /= max(n_val_batches, 1)
        sign_acc = n_correct_sign / max(n_total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), checkpoint_path)
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  "
                  f"sign_acc={sign_acc:.3f}{marker}")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"DoorKeyActionValueNet params: {n_params:,}")

    return net


def main():
    parser = argparse.ArgumentParser(
        description="Train DoorKeyActionValueNet")
    parser.add_argument("--data", type=str,
                        default="train/data/expert_doorkey.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--checkpoint", type=str,
                        default="train/checkpoints/doorkey_action_value_net.pt")
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

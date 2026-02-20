"""Train DirectionPriorNet on C-behavior data (L2→L1 distillation).

Same training loop pattern as train_doorkey_c.py: MSE loss, 90/10 split,
Adam optimizer, sign accuracy tracking.

The DirectionPriorNet learns to predict C's navigation scores using only
L1-level features (no target, no phase), enabling compressed-L2 operation.

Usage:
    python -m train.train_doorkey_b --data train/data/c_behavior_doorkey.json --epochs 100
    python -m train.train_doorkey_b --data train/data/c_behavior_doorkey.pt --epochs 100
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

from models.direction_prior_net import (
    DirectionPriorNet,
    extract_l1_features,
)


def _process_sample(sample: dict):
    """Extract L1 features and C-score label from a single sample dict."""
    feat = extract_l1_features(
        agent_pos=tuple(sample["agent_pos"]),
        agent_dir=sample["agent_dir"],
        next_pos=tuple(sample["next_pos"]),
        next_dir=sample["next_dir"],
        obstacles=[tuple(o) for o in sample["obstacles"]],
        width=sample["width"],
        height=sample["height"],
        carrying_key=sample["carrying_key"],
    )
    return feat, sample["c_score"]


def prepare_data(data_path: str, max_samples: int = 0):
    """Load samples and prepare feature tensors + labels."""
    print(f"Loading data from {data_path}...")

    if data_path.endswith(".pt"):
        saved = torch.load(data_path, weights_only=True)
        X, y = saved["X"], saved["y"]
        if max_samples > 0 and len(X) > max_samples:
            perm = torch.randperm(len(X))[:max_samples]
            X, y = X[perm], y[perm]
        print(f"  {len(X)} samples loaded from .pt file")
    else:
        with open(data_path) as f:
            data = json.load(f)

        if max_samples > 0 and len(data) > max_samples:
            import random
            rng = random.Random(42)
            data = rng.sample(data, max_samples)

        print(f"  {len(data)} raw samples, extracting L1 features...")
        features_list = []
        labels_list = []
        for i, sample in enumerate(data):
            feat, label = _process_sample(sample)
            features_list.append(feat)
            labels_list.append(label)
            if (i + 1) % 500000 == 0:
                print(f"  {i + 1}/{len(data)} processed...")

        X = torch.stack(features_list)
        y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        print(f"  {len(X)} samples loaded")

    print(f"  Features: {X.shape}, Labels: {y.shape}")
    print(f"  Label range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Label mean: {y.mean():.4f}, std: {y.std():.4f}")

    return X, y


def train(
    data_path: str = "train/data/c_behavior_doorkey.json",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 64,
    checkpoint_path: str = "train/checkpoints/direction_prior_net.pt",
    max_samples: int = 0,
):
    """Train DirectionPriorNet on C-behavior labels."""
    X, y = prepare_data(data_path, max_samples=max_samples)

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

    net = DirectionPriorNet(hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"DirectionPriorNet params: {n_params:,}")
    print(f"Training for {epochs} epochs...\n")

    best_val_loss = float("inf")
    best_sign_acc = 0.0
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
            best_sign_acc = sign_acc
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
    print(f"Best sign accuracy: {best_sign_acc:.3f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"DirectionPriorNet params: {n_params:,}")

    return net


def main():
    parser = argparse.ArgumentParser(
        description="Train DirectionPriorNet (L2->L1 distillation)")
    parser.add_argument("--data", type=str,
                        default="train/data/c_behavior_doorkey.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--checkpoint", type=str,
                        default="train/checkpoints/direction_prior_net.pt")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to use (0=all)")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

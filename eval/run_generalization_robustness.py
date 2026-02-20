"""Experiment A: Multi-Seed Robustness for DoorKey Generalization.

Tests whether the 16x16 generalization result is robust across different
training seeds, not dependent on a lucky train/eval seed.

Pipeline per train seed (42, 123, 777):
  1. Collect expert data with --no16 distribution (3000 configs)
  2. Train DoorKeyActionValueNet with identical hyperparams
  3. Eval on 16x16 with 10 eval seeds (1000-1009), 20 episodes each

Result: 3x10 matrix of SR values. Mean, std, min/max per train seed.

Pass criteria: all train seeds >= 95% mean SR with std < 3%.

Usage:
    python eval/run_generalization_robustness.py           # full
    python eval/run_generalization_robustness.py --quick    # fewer eval seeds
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eval.run_generalization_eval import (
    _load_net,
    _max_steps,
    run_neural_c_episode,
)
from models.doorkey_action_value_net import (
    DoorKeyActionValueNet,
    extract_doorkey_features,
)
from train.collect_expert_doorkey import (
    collect_doorkey_samples,
    random_doorkey_config,
)
from train.train_doorkey_c import train as train_doorkey_c

# ── Configuration ─────────────────────────────────────────────────
TRAIN_SEEDS = [42, 123, 777]
N_CONFIGS = 3000
EVAL_SEEDS_FULL = list(range(1000, 1010))   # 10 eval seeds
EVAL_SEEDS_QUICK = list(range(1000, 1005))  # 5 eval seeds
EPISODES_PER_EVAL_SEED_FULL = 20
EPISODES_PER_EVAL_SEED_QUICK = 10
EVAL_SIZE = 16

DATA_DIR = Path(__file__).resolve().parent.parent / "train" / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "train" / "checkpoints"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Training hyperparams (identical for all seeds)
TRAIN_EPOCHS = 100
TRAIN_BATCH = 128
TRAIN_LR = 1e-3
TRAIN_HIDDEN = 64


def _data_path(seed: int) -> Path:
    return DATA_DIR / f"expert_doorkey_no16_s{seed}.json"


def _pt_path(seed: int) -> Path:
    return DATA_DIR / f"expert_doorkey_no16_s{seed}.pt"


def _ckpt_path(seed: int) -> Path:
    return CKPT_DIR / f"doorkey_action_value_net_no16_s{seed}.pt"


# ── Step 1: Collect data ──────────────────────────────────────────

def _ensure_pt(json_path: Path, pt_path: Path) -> None:
    """Extract features from JSON and save as .pt if not already done."""
    if pt_path.exists():
        return
    print(f"  Extracting features from {json_path.name}...")
    with open(json_path) as f:
        data = json.load(f)
    features_list = []
    labels_list = []
    for i, s in enumerate(data):
        feat = extract_doorkey_features(
            agent_pos=tuple(s["agent_pos"]),
            agent_dir=s["agent_dir"],
            next_pos=tuple(s["next_pos"]),
            next_dir=s["next_dir"],
            target_pos=tuple(s["target_pos"]),
            obstacles=[tuple(o) for o in s["obstacles"]],
            width=s["width"], height=s["height"],
            phase=s["phase"], carrying_key=s["carrying_key"],
        )
        features_list.append(feat)
        labels_list.append(s["bfs_label"])
        if (i + 1) % 500000 == 0:
            print(f"    {i + 1}/{len(data)} features extracted...")
    X = torch.stack(features_list)
    y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
    torch.save({"X": X, "y": y}, pt_path)
    print(f"  Features saved -> {pt_path.name} ({X.shape[0]} x {X.shape[1]})")


def collect_data(seed: int) -> Path:
    """Collect BFS-labelled expert data with no16 distribution."""
    out = _data_path(seed)
    if out.exists():
        print(f"  [SKIP] Data exists: {out.name}")
        _ensure_pt(out, _pt_path(seed))
        return out

    from env.doorkey import DoorKeyEnv

    print(f"  Collecting {N_CONFIGS} configs (seed={seed}, no16)...")
    rng = random.Random(seed)
    all_samples: List[Dict] = []
    n_disagree = 0

    for ep in range(N_CONFIGS):
        cfg = random_doorkey_config(rng, no16=True)
        try:
            env = DoorKeyEnv(**cfg)
        except Exception:
            continue
        samples = collect_doorkey_samples(env, 30, rng)
        n_disagree += sum(1 for s in samples if s["disagree"])
        all_samples.extend(samples)
        if (ep + 1) % 500 == 0:
            print(f"    {ep + 1}/{N_CONFIGS} configs, {len(all_samples)} samples")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_samples, f)
    print(f"  Saved {len(all_samples)} samples -> {out.name}")

    # Extract .pt for fast training
    _ensure_pt(out, _pt_path(seed))

    return out


# ── Step 2: Train ─────────────────────────────────────────────────

def train_model(seed: int) -> Path:
    """Train DoorKeyActionValueNet on no16 data with given seed."""
    ckpt = _ckpt_path(seed)
    if ckpt.exists():
        print(f"  [SKIP] Checkpoint exists: {ckpt.name}")
        return ckpt

    # Prefer .pt for speed
    data = str(_pt_path(seed)) if _pt_path(seed).exists() else str(_data_path(seed))
    print(f"  Training (seed={seed}, epochs={TRAIN_EPOCHS})...")

    train_doorkey_c(
        data_path=data,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH,
        lr=TRAIN_LR,
        hidden=TRAIN_HIDDEN,
        checkpoint_path=str(ckpt),
        seed=seed,
    )
    print(f"  Checkpoint saved -> {ckpt.name}")
    return ckpt


# ── Step 3: Eval ──────────────────────────────────────────────────

def eval_checkpoint(
    ckpt_path: Path,
    eval_seeds: List[int],
    eps_per_seed: int,
) -> List[Dict]:
    """Eval one checkpoint on 16x16 with multiple eval seeds.

    Returns list of dicts: {train_seed, eval_seed, episode_idx, success, steps}
    """
    net = _load_net(DoorKeyActionValueNet, ckpt_path)
    if net is None:
        print(f"  [ERROR] Could not load {ckpt_path}")
        return []

    results = []
    for eval_seed in eval_seeds:
        for i in range(eps_per_seed):
            ep_seed = eval_seed * 10000 + i  # unique seed per episode
            r = run_neural_c_episode(EVAL_SIZE, seed=ep_seed, value_net=net)
            results.append({
                "eval_seed": eval_seed,
                "episode_idx": i,
                "success": r["success"],
                "steps": r["steps"],
            })
    return results


# ── Main ──────────────────────────────────────────────────────────

def main() -> bool:
    parser = argparse.ArgumentParser(
        description="Experiment A: Multi-Seed Robustness")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer eval seeds and episodes")
    args = parser.parse_args()

    quick = args.quick
    eval_seeds = EVAL_SEEDS_QUICK if quick else EVAL_SEEDS_FULL
    eps_per_seed = EPISODES_PER_EVAL_SEED_QUICK if quick else EPISODES_PER_EVAL_SEED_FULL

    print("=" * 78)
    print("  Experiment A: Multi-Seed Generalization Robustness")
    print(f"  Train seeds: {TRAIN_SEEDS}")
    print(f"  Eval seeds: {eval_seeds[0]}..{eval_seeds[-1]} "
          f"({len(eval_seeds)} seeds x {eps_per_seed} eps)")
    print(f"  Eval size: {EVAL_SIZE}x{EVAL_SIZE}")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print("=" * 78)

    t0 = time.time()

    # ── Pipeline per train seed ───────────────────────────────────
    all_results: Dict[int, List[Dict]] = {}  # train_seed -> results

    for train_seed in TRAIN_SEEDS:
        print(f"\n--- Train Seed {train_seed} ---")

        # 1. Collect
        collect_data(train_seed)

        # 2. Train
        ckpt = train_model(train_seed)

        # 3. Eval
        print(f"  Evaluating on {EVAL_SIZE}x{EVAL_SIZE} "
              f"({len(eval_seeds)} eval seeds x {eps_per_seed} eps)...")
        results = eval_checkpoint(ckpt, eval_seeds, eps_per_seed)
        all_results[train_seed] = results

        # Per-eval-seed SR
        for es in eval_seeds:
            es_results = [r for r in results if r["eval_seed"] == es]
            sr = sum(1 for r in es_results if r["success"]) / max(len(es_results), 1)
            print(f"    eval_seed={es}: SR={sr:.0%} "
                  f"({sum(1 for r in es_results if r['success'])}/{len(es_results)})")

    elapsed = time.time() - t0

    # ── Results table ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)

    seed_stats: Dict[int, Dict] = {}

    for train_seed in TRAIN_SEEDS:
        results = all_results[train_seed]
        # Per-eval-seed SR
        per_seed_sr = []
        for es in eval_seeds:
            es_results = [r for r in results if r["eval_seed"] == es]
            sr = sum(1 for r in es_results if r["success"]) / max(len(es_results), 1)
            per_seed_sr.append(sr)

        mean_sr = sum(per_seed_sr) / len(per_seed_sr)
        std_sr = math.sqrt(
            sum((s - mean_sr) ** 2 for s in per_seed_sr) / len(per_seed_sr)
        )
        min_sr = min(per_seed_sr)
        max_sr = max(per_seed_sr)

        seed_stats[train_seed] = {
            "mean": mean_sr,
            "std": std_sr,
            "min": min_sr,
            "max": max_sr,
            "per_seed": per_seed_sr,
        }

    # Header
    print(f"\n  {'Train Seed':>12s} | {'Mean SR':>8s} | {'Std':>6s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}")

    for train_seed in TRAIN_SEEDS:
        s = seed_stats[train_seed]
        print(f"  {train_seed:>12d} | {s['mean']:>7.1%} | {s['std']:>5.1%} | "
              f"{s['min']:>5.1%} | {s['max']:>5.1%}")

    # Overall
    means = [seed_stats[s]["mean"] for s in TRAIN_SEEDS]
    overall_range = max(means) - min(means)
    print(f"\n  Cross-seed range: {overall_range:.1%}")

    # ── Assertions ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  ASSERTIONS")
    print("=" * 78)

    checks = []

    # 1-3: Each seed >= 95% mean SR
    for i, train_seed in enumerate(TRAIN_SEEDS):
        s = seed_stats[train_seed]
        p = s["mean"] >= 0.95
        checks.append(p)
        print(f"\n  [{'PASS' if p else 'FAIL'}] {i + 1}. "
              f"SR(seed={train_seed}) >= 95%: {s['mean']:.1%}")

    # 4-6: Each seed std < 3%
    for i, train_seed in enumerate(TRAIN_SEEDS):
        s = seed_stats[train_seed]
        p = s["std"] < 0.03
        checks.append(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {i + 4}. "
              f"std(seed={train_seed}) < 3%: {s['std']:.1%}")

    # 7: Cross-seed range < 5pp
    p7 = overall_range < 0.05
    checks.append(p7)
    print(f"  [{'PASS' if p7 else 'FAIL'}] 7. "
          f"cross-seed range < 5pp: {overall_range:.1%}")

    # ── CSV ───────────────────────────────────────────────────────
    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RUNS_DIR / f"generalization_robustness_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_seed", "eval_seed", "episode_idx",
                         "success", "steps"])
        for train_seed in TRAIN_SEEDS:
            for r in all_results[train_seed]:
                writer.writerow([
                    train_seed,
                    r["eval_seed"],
                    r["episode_idx"],
                    int(r["success"]),
                    r["steps"],
                ])
    print(f"\n  CSV: {csv_path}")

    all_pass = all(checks)
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({sum(checks)}/{len(checks)})")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print("=" * 78)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

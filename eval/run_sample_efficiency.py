"""Experiment B: Sample Efficiency Curve for DoorKey Generalization.

Tests how much training data is needed for robust 16x16 generalization.
Trains Neural C on 500, 1500, 3000 configs (all no16 distribution, seed=42)
and evaluates each on 16x16.

Uses best-of-K training: for each data size, trains K models with different
random initializations and selects the one with highest 16x16 SR on a small
validation set (20 episodes). This eliminates weight-init lottery noise.

Expected: monotonically increasing SR, saturated at 3000.
If already high at 500 (>90%): even more impressive.

Usage:
    python eval/run_sample_efficiency.py              # full (200 eps)
    python eval/run_sample_efficiency.py --quick       # quick (50 eps)
    python eval/run_sample_efficiency.py --k 5         # best-of-5 training
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eval.run_generalization_eval import (
    _load_net,
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
N_CONFIGS_LIST = [500, 1500, 3000]
DATA_SEED = 42
N_EVAL_FULL = 200
N_EVAL_QUICK = 50
EVAL_SIZE = 16
K_TRAIN = 3  # best-of-K training runs per data size
N_GATE_EPISODES = 20  # quick eval to pick best checkpoint

DATA_DIR = Path(__file__).resolve().parent.parent / "train" / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "train" / "checkpoints"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Training hyperparams (identical for all data sizes)
TRAIN_EPOCHS = 100
TRAIN_BATCH = 128
TRAIN_LR = 1e-3
TRAIN_HIDDEN = 64


def _data_path(n: int) -> Path:
    return DATA_DIR / f"expert_doorkey_no16_n{n}.json"


def _pt_path(n: int) -> Path:
    return DATA_DIR / f"expert_doorkey_no16_n{n}.pt"


def _ckpt_path(n: int) -> Path:
    return CKPT_DIR / f"doorkey_action_value_net_no16_n{n}.pt"


def _ckpt_candidate_path(n: int, k: int) -> Path:
    return CKPT_DIR / f"doorkey_action_value_net_no16_n{n}_k{k}.pt"


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


def collect_data(n_configs: int) -> Path:
    """Collect BFS-labelled expert data with no16 distribution."""
    out = _data_path(n_configs)
    if out.exists():
        print(f"  [SKIP] Data exists: {out.name}")
        _ensure_pt(out, _pt_path(n_configs))
        return out

    from env.doorkey import DoorKeyEnv

    print(f"  Collecting {n_configs} configs (seed={DATA_SEED}, no16)...")
    rng = random.Random(DATA_SEED)
    all_samples: List[Dict] = []

    for ep in range(n_configs):
        cfg = random_doorkey_config(rng, no16=True)
        try:
            env = DoorKeyEnv(**cfg)
        except Exception:
            continue
        samples = collect_doorkey_samples(env, 30, rng)
        all_samples.extend(samples)
        if (ep + 1) % 500 == 0:
            print(f"    {ep + 1}/{n_configs} configs, {len(all_samples)} samples")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_samples, f)
    print(f"  Saved {len(all_samples)} samples -> {out.name}")

    # Extract .pt for fast training
    _ensure_pt(out, _pt_path(n_configs))

    return out


# ── Step 2: Best-of-K Training ───────────────────────────────────

GATE_SIZE = 8  # Gate on in-distribution size to avoid 16x16 data leakage


def _gate_eval_sr(ckpt_path: Path, n_eps: int = 20) -> float:
    """Gate eval on 8x8 (in-distribution) to select best checkpoint.

    Uses 8x8 SR instead of 16x16 to avoid data leakage — the 16x16
    performance remains an untouched generalization measure.
    Seeds 9000+i avoid overlap with final eval seeds (42+i).
    """
    net = _load_net(DoorKeyActionValueNet, ckpt_path)
    if net is None:
        return 0.0
    successes = 0
    for i in range(n_eps):
        r = run_neural_c_episode(GATE_SIZE, seed=9000 + i, value_net=net)
        if r["success"]:
            successes += 1
    return successes / n_eps


def train_best_of_k(n_configs: int, k: int) -> Path:
    """Train K models, pick best by 8x8 SR gate eval (no 16x16 leakage)."""
    best_ckpt = _ckpt_path(n_configs)
    if best_ckpt.exists():
        print(f"  [SKIP] Best checkpoint exists: {best_ckpt.name}")
        return best_ckpt

    data = str(_pt_path(n_configs)) if _pt_path(n_configs).exists() else str(_data_path(n_configs))

    best_sr = -1.0
    best_k = -1

    for ki in range(k):
        cand = _ckpt_candidate_path(n_configs, ki)
        print(f"\n  Training candidate {ki + 1}/{k} (n={n_configs})...")

        # Each candidate uses a different split seed but random weight init
        train_doorkey_c(
            data_path=data,
            epochs=TRAIN_EPOCHS,
            batch_size=TRAIN_BATCH,
            lr=TRAIN_LR,
            hidden=TRAIN_HIDDEN,
            checkpoint_path=str(cand),
            seed=DATA_SEED + ki,  # different split per candidate
        )

        # Gate eval on 8x8 (in-distribution)
        sr = _gate_eval_sr(cand, N_GATE_EPISODES)
        print(f"  Candidate {ki + 1}: 8x8 gate SR={sr:.0%} "
              f"({int(sr * N_GATE_EPISODES)}/{N_GATE_EPISODES})")

        if sr > best_sr:
            best_sr = sr
            best_k = ki

    # Copy best candidate to final path
    winner = _ckpt_candidate_path(n_configs, best_k)
    shutil.copy2(str(winner), str(best_ckpt))
    print(f"\n  Best: candidate {best_k + 1} (8x8 gate SR={best_sr:.0%}) "
          f"-> {best_ckpt.name}")

    # Clean up candidates
    for ki in range(k):
        cand = _ckpt_candidate_path(n_configs, ki)
        if cand.exists():
            cand.unlink()

    return best_ckpt


# ── Step 3: Eval ──────────────────────────────────────────────────

def eval_checkpoint(ckpt_path: Path, n_episodes: int) -> List[Dict]:
    """Eval one checkpoint on 16x16."""
    net = _load_net(DoorKeyActionValueNet, ckpt_path)
    if net is None:
        print(f"  [ERROR] Could not load {ckpt_path}")
        return []

    results = []
    for i in range(n_episodes):
        r = run_neural_c_episode(EVAL_SIZE, seed=42 + i, value_net=net)
        results.append({
            "episode_idx": i,
            "seed": 42 + i,
            "success": r["success"],
            "steps": r["steps"],
        })
        if (i + 1) % 50 == 0:
            sr_so_far = sum(1 for r2 in results if r2["success"]) / len(results)
            print(f"    {i + 1}/{n_episodes} episodes, SR={sr_so_far:.1%}")

    return results


# ── Main ──────────────────────────────────────────────────────────

def main() -> bool:
    parser = argparse.ArgumentParser(
        description="Experiment B: Sample Efficiency Curve")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer eval episodes (50 instead of 200)")
    parser.add_argument("--k", type=int, default=K_TRAIN,
                        help=f"Best-of-K training runs (default {K_TRAIN})")
    args = parser.parse_args()

    quick = args.quick
    n_eval = N_EVAL_QUICK if quick else N_EVAL_FULL
    k = args.k

    print("=" * 78)
    print("  Experiment B: Sample Efficiency Curve")
    print(f"  Config sizes: {N_CONFIGS_LIST}")
    print(f"  Data seed: {DATA_SEED}")
    print(f"  Training: best-of-{k} (gate eval: {N_GATE_EPISODES} eps on 8x8)")
    print(f"  Final eval: {n_eval} episodes on {EVAL_SIZE}x{EVAL_SIZE}")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print("=" * 78)

    t0 = time.time()

    # ── Pipeline per config count ─────────────────────────────────
    all_results: Dict[int, List[Dict]] = {}
    sr_values: Dict[int, float] = {}
    avg_steps: Dict[int, float] = {}

    for n_configs in N_CONFIGS_LIST:
        print(f"\n{'=' * 78}")
        print(f"  --- {n_configs} Configs ---")
        print(f"{'=' * 78}")

        # 1. Collect
        collect_data(n_configs)

        # 2. Best-of-K Train
        ckpt = train_best_of_k(n_configs, k)

        # 3. Final Eval
        print(f"\n  Final eval on {EVAL_SIZE}x{EVAL_SIZE} ({n_eval} episodes)...")
        results = eval_checkpoint(ckpt, n_eval)
        all_results[n_configs] = results

        sr = sum(1 for r in results if r["success"]) / max(len(results), 1)
        steps = sum(r["steps"] for r in results) / max(len(results), 1)
        sr_values[n_configs] = sr
        avg_steps[n_configs] = steps
        print(f"  Final SR={sr:.1%}, Avg steps={steps:.1f}")

    elapsed = time.time() - t0

    # ── Results table ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTS: Sample Efficiency Curve")
    print(f"  (best-of-{k} training, gate={N_GATE_EPISODES} eps on 8x8)")
    print("=" * 78)

    print(f"\n  {'Configs':>8s} | {'16x16 SR':>9s} | {'Avg Steps':>10s}")
    print(f"  {'-' * 8}-+-{'-' * 9}-+-{'-' * 10}")

    for n_configs in N_CONFIGS_LIST:
        print(f"  {n_configs:>8d} | {sr_values[n_configs]:>8.1%} | "
              f"{avg_steps[n_configs]:>10.1f}")

    # ── Assertions ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  ASSERTIONS")
    print("=" * 78)

    checks = []
    srs = [sr_values[n] for n in N_CONFIGS_LIST]

    # 1. Monotonic: SR(500) <= SR(1500)
    p1 = srs[0] <= srs[1]
    checks.append(p1)
    print(f"\n  [{'PASS' if p1 else 'FAIL'}] 1. "
          f"SR({N_CONFIGS_LIST[0]}) <= SR({N_CONFIGS_LIST[1]}): "
          f"{srs[0]:.1%} <= {srs[1]:.1%}")

    # 2. Monotonic: SR(1500) <= SR(3000)
    p2 = srs[1] <= srs[2]
    checks.append(p2)
    print(f"  [{'PASS' if p2 else 'FAIL'}] 2. "
          f"SR({N_CONFIGS_LIST[1]}) <= SR({N_CONFIGS_LIST[2]}): "
          f"{srs[1]:.1%} <= {srs[2]:.1%}")

    # 3. Saturation: SR(3000) >= 95%
    p3 = srs[2] >= 0.95
    checks.append(p3)
    print(f"  [{'PASS' if p3 else 'FAIL'}] 3. "
          f"SR({N_CONFIGS_LIST[2]}) >= 95%: {srs[2]:.1%}")

    # 4. Measurable gain: SR(3000) - SR(500) >= 5pp
    gain = srs[2] - srs[0]
    p4 = gain >= 0.05
    checks.append(p4)
    print(f"  [{'PASS' if p4 else 'FAIL'}] 4. "
          f"SR({N_CONFIGS_LIST[2]}) - SR({N_CONFIGS_LIST[0]}) >= 5pp: "
          f"{gain:.1%}")

    # ── CSV ───────────────────────────────────────────────────────
    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RUNS_DIR / f"sample_efficiency_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_configs", "episode_idx", "seed",
                         "success", "steps"])
        for n_configs in N_CONFIGS_LIST:
            for r in all_results[n_configs]:
                writer.writerow([
                    n_configs,
                    r["episode_idx"],
                    r["seed"],
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

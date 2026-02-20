"""Diagnostic: Seed-Scheme vs Model Fragility.

Tests two hypotheses for the SR drop on 16x16:
  H1 (Eval-Shift): New eval seeds generate harder layouts
  H2 (Model fragility): Model doesn't generalize stably

Tests:
  A) Original seeds (42+i for i=0..199) — should match old 100% if H1
  B) Three seed schemes with difficulty metrics:
     - low seeds:  0..199
     - mid seeds:  1000..1199
     - high seeds: 10000000..10000199  (the "robustness" scheme)

For each layout, logs BFS difficulty metric:
  optimal path length = BFS(start->key) + BFS(key->door) + BFS(door->goal)

Usage:
    python eval/run_seed_diagnostic.py
    python eval/run_seed_diagnostic.py --n 50  # quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env.doorkey import DoorKeyEnv
from eval.run_generalization_eval import (
    _load_net,
    run_neural_c_episode,
)
from models.doorkey_action_value_net import DoorKeyActionValueNet
from train.bfs_expert_doorkey import bfs_distance_map

# ── Checkpoint ────────────────────────────────────────────────────
CKPT_DIR = Path(__file__).resolve().parent.parent / "train" / "checkpoints"

# Use the original no16 checkpoint (from previous session)
CKPT_ORIGINAL = CKPT_DIR / "doorkey_action_value_net_no16.pt"
# Also try the s42 checkpoint from robustness experiment
CKPT_S42 = CKPT_DIR / "doorkey_action_value_net_no16_s42.pt"


def compute_bfs_difficulty(size: int, seed: int) -> Optional[Dict]:
    """Compute BFS-optimal path length for a DoorKey layout.

    Returns dict with difficulty metrics, or None if layout is invalid.
    """
    try:
        env = DoorKeyEnv(size=size, seed=seed)
        obs = env.reset()
    except Exception:
        return None

    width, height = obs.width, obs.height
    key_pos = obs.key_pos
    door_pos = obs.door_pos
    goal_pos = obs.goal_pos if obs.goal_pos != (-1, -1) else (width - 2, height - 2)
    agent_pos = obs.agent_pos

    if key_pos is None or door_pos is None:
        return None

    base_walls = set(obs.obstacles)
    if door_pos is not None:
        base_walls.discard(door_pos)

    # Phase 1: start -> key (door is obstacle)
    walls_p1 = base_walls | {door_pos}
    dist_to_key = bfs_distance_map(key_pos, walls_p1, width, height)
    d1 = dist_to_key.get(agent_pos, 999)

    # Phase 2: key -> door (door removed)
    dist_to_door = bfs_distance_map(door_pos, base_walls, width, height)
    d2 = dist_to_door.get(key_pos, 999)

    # Phase 3: door -> goal (door open)
    dist_to_goal = bfs_distance_map(goal_pos, base_walls, width, height)
    d3 = dist_to_goal.get(door_pos, 999)

    total = d1 + d2 + d3

    # Obstacle density
    n_cells = (width - 2) * (height - 2)  # interior cells
    n_obstacles = len(base_walls)
    obstacle_density = n_obstacles / max(n_cells, 1)

    return {
        "bfs_start_key": d1,
        "bfs_key_door": d2,
        "bfs_door_goal": d3,
        "bfs_total": total,
        "obstacle_density": obstacle_density,
        "n_obstacles": n_obstacles,
    }


def run_seed_scheme(
    name: str,
    seeds: List[int],
    value_net: DoorKeyActionValueNet,
    size: int = 16,
) -> Tuple[List[Dict], float]:
    """Run episodes on a seed scheme, return (results, SR)."""
    results = []
    for seed in seeds:
        diff = compute_bfs_difficulty(size, seed)
        r = run_neural_c_episode(size, seed=seed, value_net=value_net)
        r["seed"] = seed
        if diff:
            r.update(diff)
        else:
            r["bfs_total"] = 999
        results.append(r)

    sr = sum(1 for r in results if r["success"]) / max(len(results), 1)
    return results, sr


def main():
    parser = argparse.ArgumentParser(description="Seed Diagnostic")
    parser.add_argument("--n", type=int, default=200,
                        help="Episodes per scheme")
    args = parser.parse_args()
    n = args.n

    # Load checkpoint
    ckpt = CKPT_ORIGINAL if CKPT_ORIGINAL.exists() else CKPT_S42
    print(f"Loading checkpoint: {ckpt.name}")
    net = _load_net(DoorKeyActionValueNet, ckpt)
    if net is None:
        print("ERROR: No checkpoint found")
        sys.exit(1)

    print("=" * 78)
    print("  Seed Diagnostic: H1 (eval-shift) vs H2 (model fragility)")
    print(f"  Checkpoint: {ckpt.name}")
    print(f"  Episodes per scheme: {n}")
    print("=" * 78)

    # ── Seed schemes ──────────────────────────────────────────────
    schemes = {
        "original (42+i)":      [42 + i for i in range(n)],
        "low (0+i)":            [i for i in range(n)],
        "mid (1000+i)":         [1000 + i for i in range(n)],
        "high (10M+i)":         [10_000_000 + i for i in range(n)],
    }

    all_results: Dict[str, Tuple[List[Dict], float]] = {}

    for scheme_name, seeds in schemes.items():
        print(f"\n--- {scheme_name} ---")
        results, sr = run_seed_scheme(scheme_name, seeds, net)
        all_results[scheme_name] = (results, sr)

        # Difficulty stats
        bfs_totals = [r["bfs_total"] for r in results if r["bfs_total"] < 999]
        if bfs_totals:
            bfs_totals.sort()
            median = bfs_totals[len(bfs_totals) // 2]
            mean_d = sum(bfs_totals) / len(bfs_totals)
            min_d = min(bfs_totals)
            max_d = max(bfs_totals)
            print(f"  SR: {sr:.1%}  |  BFS difficulty: "
                  f"median={median}, mean={mean_d:.1f}, "
                  f"range=[{min_d}, {max_d}]")

            # SR by difficulty decile
            n_dec = max(len(bfs_totals) // 5, 1)
            sorted_results = sorted(
                [r for r in results if r["bfs_total"] < 999],
                key=lambda r: r["bfs_total"])
            print(f"  SR by difficulty quintile:")
            for q in range(5):
                chunk = sorted_results[q * n_dec:(q + 1) * n_dec]
                if chunk:
                    q_sr = sum(1 for r in chunk if r["success"]) / len(chunk)
                    q_min = min(r["bfs_total"] for r in chunk)
                    q_max = max(r["bfs_total"] for r in chunk)
                    print(f"    Q{q + 1} (BFS {q_min}-{q_max}): "
                          f"SR={q_sr:.0%} ({sum(1 for r in chunk if r['success'])}/{len(chunk)})")
        else:
            print(f"  SR: {sr:.1%}  |  No valid BFS data")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    print(f"\n  {'Scheme':>20s} | {'SR':>6s} | {'BFS median':>10s} | {'BFS mean':>10s}")
    print(f"  {'-' * 20}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}")

    for scheme_name in schemes:
        results, sr = all_results[scheme_name]
        bfs = [r["bfs_total"] for r in results if r["bfs_total"] < 999]
        if bfs:
            bfs.sort()
            median = bfs[len(bfs) // 2]
            mean_d = sum(bfs) / len(bfs)
        else:
            median = mean_d = 0
        print(f"  {scheme_name:>20s} | {sr:>5.1%} | {median:>10d} | {mean_d:>10.1f}")

    # ── Diagnosis ─────────────────────────────────────────────────
    print("\n  DIAGNOSIS:")
    orig_sr = all_results["original (42+i)"][1]
    high_sr = all_results["high (10M+i)"][1]

    if orig_sr >= 0.90 and high_sr < 0.70:
        print("  -> H1 CONFIRMED: Eval-shift. Original seeds are easy,")
        print("     high seeds generate harder layouts.")
        print("     Action: Use difficulty-matched seed ranges, or report")
        print("     SR vs BFS-difficulty curve instead of raw SR.")
    elif orig_sr < 0.70:
        print("  -> H2 LIKELY: Model fragility. Even original seeds are low.")
        print("     Action: Check training convergence, model capacity.")
    else:
        print("  -> BOTH SCHEMES HIGH: Model is robust across seed ranges.")

    print("=" * 78)


if __name__ == "__main__":
    main()

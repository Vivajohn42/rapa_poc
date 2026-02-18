"""Collect BFS-labelled action-value data for training Neural DoorKey C.

For each (pos, dir, action) triple and each DoorKey phase, computes:
  - BFS label: effective_distance(now) - effective_distance(after_action)
  - Heuristic label: manhattan+turns(now) - manhattan+turns(after_action)

Only navigation actions (turn_left, turn_right, forward) are labelled.
Pickup/toggle remain deterministic in Neural DoorKey C.

Usage:
    python -m train.collect_expert_doorkey --episodes 3000 --out train/data/expert_doorkey.json
    python -m train.collect_expert_doorkey --episodes 500 --out train/data/expert_doorkey.json  # quick
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey import DOOR_OPEN, DoorKeyEnv
from train.bfs_expert_doorkey import (
    UNREACHABLE,
    NAV_ACTIONS,
    bfs_distance_map,
    compute_next_state,
    effective_distance,
    heuristic_distance,
)

# Phase targets: phase_name -> which position is the target
PHASES = ["FIND_KEY", "OPEN_DOOR", "REACH_GOAL"]


def collect_doorkey_samples(
    env: DoorKeyEnv,
    n_positions: int = 30,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Collect BFS-labelled samples from one DoorKey configuration.

    For each phase (FIND_KEY, OPEN_DOOR, REACH_GOAL):
    1. Determine target (key_pos, door_pos, goal_pos)
    2. Compute BFS distance map from target
    3. Sample random (pos, dir) pairs
    4. Label each navigation action
    """
    if rng is None:
        rng = random.Random()

    obs = env.reset()
    width, height = obs.width, obs.height

    # Extract positions from privileged access
    key_pos = obs.key_pos
    door_pos = obs.door_pos
    goal_pos = obs.goal_pos if obs.goal_pos != (-1, -1) else (width - 2, height - 2)

    # Base obstacles (walls only — door handled per-phase)
    base_walls = set(obs.obstacles)
    # Door might already be in obstacles list if locked/closed
    if door_pos is not None:
        base_walls.discard(door_pos)
    # Now base_walls = walls only

    if key_pos is None or door_pos is None:
        return []  # Unusual env state

    # Phase configs: (target, obstacles_for_bfs, carrying_key)
    phase_configs = {
        "FIND_KEY": (
            key_pos,
            base_walls | {door_pos},  # door is obstacle
            False,
        ),
        "OPEN_DOOR": (
            door_pos,
            base_walls,  # door REMOVED from obstacles for BFS
            True,
        ),
        "REACH_GOAL": (
            goal_pos,
            base_walls,  # door is open, not obstacle
            True,
        ),
    }

    all_samples: List[Dict[str, Any]] = []

    for phase, (target, obstacles_for_bfs, carrying_key) in phase_configs.items():
        if target is None:
            continue

        # BFS distance map from target (one BFS per phase)
        dist_map = bfs_distance_map(target, obstacles_for_bfs, width, height)
        if not dist_map:
            continue

        # Reachable positions
        reachable = list(dist_map.keys())
        if not reachable:
            continue

        # Sampling bias: 50% near-wall, 25% near key/door, 25% random
        near_wall = []
        for wx, wy in base_walls:
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                pos = (wx + dx, wy + dy)
                if pos in dist_map:
                    near_wall.append(pos)

        near_objects = []
        for obj_pos in [key_pos, door_pos, goal_pos]:
            if obj_pos is None:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]:
                pos = (obj_pos[0] + dx, obj_pos[1] + dy)
                if pos in dist_map:
                    near_objects.append(pos)

        n_wall = min(n_positions // 2, len(near_wall))
        n_obj = min(n_positions // 4, len(near_objects))
        n_rand = n_positions - n_wall - n_obj

        sampled = set()
        if near_wall:
            sampled.update(rng.sample(near_wall, min(n_wall, len(near_wall))))
        if near_objects:
            remaining_obj = [p for p in near_objects if p not in sampled]
            if remaining_obj:
                sampled.update(rng.sample(
                    remaining_obj, min(n_obj, len(remaining_obj))))
        remaining = [p for p in reachable if p not in sampled]
        if remaining:
            sampled.update(rng.sample(remaining, min(n_rand, len(remaining))))

        obstacles_list = [list(o) for o in obstacles_for_bfs]

        for pos in sampled:
            # For each of 4 directions
            for direction in range(4):
                d_now_bfs = effective_distance(
                    pos, direction, target,
                    obstacles_for_bfs, width, height)

                if d_now_bfs >= UNREACHABLE:
                    continue

                d_now_heur = heuristic_distance(pos, direction, target)

                for action in NAV_ACTIONS:
                    next_pos, next_dir = compute_next_state(
                        pos, direction, action,
                        obstacles_for_bfs, width, height,
                        door_pos=door_pos if phase != "FIND_KEY" else None,
                        door_open=(phase == "REACH_GOAL"),
                    )

                    d_next_bfs = effective_distance(
                        next_pos, next_dir, target,
                        obstacles_for_bfs, width, height)

                    if d_next_bfs >= UNREACHABLE:
                        bfs_label = -2.0
                    else:
                        bfs_label = d_now_bfs - d_next_bfs

                    d_next_heur = heuristic_distance(
                        next_pos, next_dir, target)
                    heur_label = d_now_heur - d_next_heur

                    all_samples.append({
                        "width": width,
                        "height": height,
                        "agent_pos": list(pos),
                        "agent_dir": direction,
                        "next_pos": list(next_pos),
                        "next_dir": next_dir,
                        "target_pos": list(target),
                        "obstacles": obstacles_list,
                        "action": action,
                        "phase": phase,
                        "carrying_key": carrying_key,
                        "bfs_label": round(bfs_label, 4),
                        "heuristic_label": round(heur_label, 4),
                        "disagree": abs(bfs_label - heur_label) > 0.01,
                    })

    return all_samples


def random_doorkey_config(rng: random.Random) -> Dict[str, Any]:
    """Generate a random DoorKey config (size + seed)."""
    r = rng.random()
    if r < 0.10:
        size = 5
    elif r < 0.70:
        size = 6
    elif r < 1.00:
        size = 8
    seed = rng.randint(0, 999999)
    return {"size": size, "seed": seed}


def main():
    parser = argparse.ArgumentParser(
        description="Collect BFS-labelled DoorKey C data")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of DoorKey configs to sample")
    parser.add_argument("--positions-per-grid", type=int, default=30,
                        help="Random positions per grid config per phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="train/data/expert_doorkey.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples: List[Dict[str, Any]] = []
    n_disagree = 0

    print(f"Collecting from {args.episodes} DoorKey configs, "
          f"{args.positions_per_grid} positions each...")

    for ep in range(args.episodes):
        cfg = random_doorkey_config(rng)
        try:
            env = DoorKeyEnv(**cfg)
        except Exception:
            continue  # Skip invalid configs

        samples = collect_doorkey_samples(
            env, args.positions_per_grid, rng)
        n_disagree += sum(1 for s in samples if s["disagree"])
        all_samples.extend(samples)

        if (ep + 1) % 500 == 0:
            print(f"  {ep + 1}/{args.episodes} configs, "
                  f"{len(all_samples)} samples, "
                  f"disagree={n_disagree}/{len(all_samples)} "
                  f"({n_disagree / max(len(all_samples), 1):.1%})")

    # Phase distribution
    phase_counts = {}
    for s in all_samples:
        phase_counts[s["phase"]] = phase_counts.get(s["phase"], 0) + 1

    # Ensure output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_samples, f)

    print(f"\nDone: {len(all_samples)} samples from {args.episodes} configs")
    print(f"Disagree rate: {n_disagree}/{len(all_samples)} "
          f"({n_disagree / max(len(all_samples), 1):.1%})")
    print(f"Phase distribution: {phase_counts}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

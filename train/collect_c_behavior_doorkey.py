"""Collect C-behavior data for training DirectionPriorNet (L2→L1 distillation).

For each (pos, dir, action) triple, computes:
  - C's BFS-based navigation score (from DoorKeyAgentC.choose_action)
  - L1-level features (59 dims): NO target, phase, or heuristic_delta

This captures C's full goal-conditioned navigation knowledge and distills it
into features that B-level processing can compute without C's goal context.

The resulting DirectionPriorNet learns to approximate C's scoring using only
local obstacle awareness + agent state, enabling compressed-L2 operation.

Usage:
    python -m train.collect_c_behavior_doorkey --episodes 2000 --out train/data/c_behavior_doorkey.json
    python -m train.collect_c_behavior_doorkey --episodes 2000 --out train/data/c_behavior_doorkey.json --pt
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
from agents.doorkey_agent_b import DoorKeyAgentB
from agents.doorkey_agent_c import DoorKeyAgentC, ACTIONS, DIR_VEC, _bfs_distance, UNREACHABLE
from state.schema import ZA
from train.bfs_expert_doorkey import (
    NAV_ACTIONS,
    bfs_distance_map,
    compute_next_state,
)

# Phase targets and obstacle configs (same as collect_expert_doorkey.py)
PHASES = ["FIND_KEY", "OPEN_DOOR", "REACH_GOAL"]


def collect_c_behavior_samples(
    env: DoorKeyEnv,
    n_positions: int = 30,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Collect C's navigation scores from one DoorKey configuration.

    For each phase:
    1. Set up DoorKeyAgentC + DoorKeyAgentB with correct phase context
    2. Sample random (pos, dir) pairs
    3. For each nav action, call B.predict_next → C.choose_action → score
    4. Record L1-level features + C's score as label
    """
    if rng is None:
        rng = random.Random()

    obs = env.reset()
    width, height = obs.width, obs.height

    key_pos = obs.key_pos
    door_pos = obs.door_pos
    goal_pos = obs.goal_pos if obs.goal_pos != (-1, -1) else (width - 2, height - 2)

    base_walls = set(obs.obstacles)
    if door_pos is not None:
        base_walls.discard(door_pos)

    if key_pos is None or door_pos is None:
        return []

    # Phase configs: (target, obstacles_for_bfs, carrying_key, door_open)
    phase_configs = {
        "FIND_KEY": (key_pos, base_walls | {door_pos}, False, False),
        "OPEN_DOOR": (door_pos, base_walls, True, False),
        "REACH_GOAL": (goal_pos, base_walls, True, True),
    }

    all_samples: List[Dict[str, Any]] = []

    for phase, (target, obstacles, carrying_key, door_open_flag) in phase_configs.items():
        if target is None:
            continue

        # BFS distance map for sampling reachable positions
        dist_map = bfs_distance_map(target, obstacles, width, height)
        if not dist_map:
            continue

        reachable = list(dist_map.keys())
        if not reachable:
            continue

        # Set up C + B for this phase
        c = DoorKeyAgentC(goal_mode="seek")
        c.phase = phase
        c.key_pos = key_pos
        c.door_pos = door_pos
        c.carrying_key = carrying_key
        c.door_open = door_open_flag
        c.goal.target = target

        b = DoorKeyAgentB(door_pos=door_pos, door_open=door_open_flag)

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

        sampled: set = set()
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

        obstacles_list = [list(o) for o in obstacles]

        for pos in sampled:
            for direction in range(4):
                # Build ZA for this position/direction
                zA = ZA(
                    width=width,
                    height=height,
                    agent_pos=pos,
                    goal_pos=goal_pos,
                    obstacles=list(obstacles),
                    hint=None,
                    direction=direction,
                )

                # C's memory must have target for full scoring
                memory = {"target": list(target)}
                if carrying_key:
                    memory["has_key"] = True
                if phase != "FIND_KEY":
                    memory["phase"] = phase.lower()

                # Call C's choose_action for full scored list
                _, scored = c.choose_action(
                    zA, b.predict_next, memory=memory,
                    tie_break_delta=0.25,
                )

                # Extract per-action scores
                score_map = {a: s for a, s in scored}

                for action in NAV_ACTIONS:
                    # Compute next state after this action
                    next_pos, next_dir = compute_next_state(
                        pos, direction, action,
                        obstacles, width, height,
                        door_pos=door_pos if phase != "FIND_KEY" else None,
                        door_open=door_open_flag,
                    )

                    all_samples.append({
                        "width": width,
                        "height": height,
                        "agent_pos": list(pos),
                        "agent_dir": direction,
                        "next_pos": list(next_pos),
                        "next_dir": next_dir,
                        "obstacles": obstacles_list,
                        "carrying_key": carrying_key,
                        # Label: C's navigation score
                        "c_score": round(score_map.get(action, 0.0), 4),
                        # Metadata (not used as features, just for analysis)
                        "phase": phase,
                        "action": action,
                    })

    return all_samples


def random_doorkey_config(rng: random.Random, *, no16: bool = False) -> Dict[str, Any]:
    """Generate a random DoorKey config.

    Default distribution: 5% size=5, 35% size=6, 30% size=8, 30% size=16.
    With no16=True:       10% size=5, 50% size=6, 40% size=8, 0% size=16.
    """
    r = rng.random()
    if no16:
        if r < 0.10:
            size = 5
        elif r < 0.60:
            size = 6
        else:
            size = 8
    else:
        if r < 0.05:
            size = 5
        elif r < 0.40:
            size = 6
        elif r < 0.70:
            size = 8
        else:
            size = 16
    seed = rng.randint(0, 999999)
    return {"size": size, "seed": seed}


def main():
    parser = argparse.ArgumentParser(
        description="Collect C-behavior DoorKey data for DirectionPriorNet")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of DoorKey configs to sample")
    parser.add_argument("--positions-per-grid", type=int, default=30,
                        help="Random positions per grid config per phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="train/data/c_behavior_doorkey.json")
    parser.add_argument("--pt", action="store_true",
                        help="Also save pre-extracted L1 features as .pt file")
    parser.add_argument("--no16", action="store_true",
                        help="Exclude 16x16 grids (10/50/40/0%% distribution)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples: List[Dict[str, Any]] = []

    dist_label = "10/50/40/0% (no 16x16)" if args.no16 else "5/35/30/30%"
    print(f"Collecting C-behavior from {args.episodes} DoorKey configs, "
          f"{args.positions_per_grid} positions each... [{dist_label}]")

    for ep in range(args.episodes):
        cfg = random_doorkey_config(rng, no16=args.no16)
        try:
            env = DoorKeyEnv(**cfg)
        except Exception:
            continue

        samples = collect_c_behavior_samples(
            env, args.positions_per_grid, rng)
        all_samples.extend(samples)

        if (ep + 1) % 500 == 0:
            print(f"  {ep + 1}/{args.episodes} configs, "
                  f"{len(all_samples)} samples")

    # Phase distribution
    phase_counts: Dict[str, int] = {}
    for s in all_samples:
        phase_counts[s["phase"]] = phase_counts.get(s["phase"], 0) + 1

    # Score distribution
    scores = [s["c_score"] for s in all_samples]
    pos_scores = [s for s in scores if s > 0]
    neg_scores = [s for s in scores if s < 0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_samples, f)

    print(f"\nDone: {len(all_samples)} samples from {args.episodes} configs")
    print(f"Phase distribution: {phase_counts}")
    print(f"Score stats: mean={sum(scores)/len(scores):.3f}, "
          f"positive={len(pos_scores)}, negative={len(neg_scores)}")
    print(f"Saved to: {out_path}")

    if args.pt:
        import torch
        from models.direction_prior_net import extract_l1_features

        print(f"\nExtracting L1 features for {len(all_samples)} samples...")
        features_list = []
        labels_list = []
        for i, s in enumerate(all_samples):
            feat = extract_l1_features(
                agent_pos=tuple(s["agent_pos"]),
                agent_dir=s["agent_dir"],
                next_pos=tuple(s["next_pos"]),
                next_dir=s["next_dir"],
                obstacles=[tuple(o) for o in s["obstacles"]],
                width=s["width"],
                height=s["height"],
                carrying_key=s["carrying_key"],
            )
            features_list.append(feat)
            labels_list.append(s["c_score"])
            if (i + 1) % 500000 == 0:
                print(f"  {i + 1}/{len(all_samples)} features extracted...")

        X = torch.stack(features_list)
        y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        pt_path = out_path.with_suffix(".pt")
        torch.save({"X": X, "y": y}, pt_path)
        print(f"L1 features saved to: {pt_path} "
              f"({X.shape[0]} samples, {X.shape[1]} dims)")


if __name__ == "__main__":
    main()

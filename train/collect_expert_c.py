"""Collect BFS-labelled action-value data for training Neural C.

For each (state, action) pair, computes both:
  - Manhattan label: manhattan(now, goal) - manhattan(next, goal)
  - BFS label:       bfs_dist(now, goal) - bfs_dist(next, goal)

The interesting training examples are where these two DISAGREE —
i.e., obstacles force detours that Manhattan cannot see.

Uses bfs_distance_map() for efficiency: one BFS per goal per grid config,
then look up distances per step.

Usage:
    python -m train.collect_expert_c --episodes 5000 --out train/data/expert_c.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld, GoalDef, HintCellDef
from train.bfs_expert import bfs_distance_map, UNREACHABLE
from models.action_value_net import extract_features, manhattan

ACTIONS = ("up", "down", "left", "right")


def compute_next_pos(
    pos: Tuple[int, int],
    action: str,
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Tuple[int, int]:
    """Compute next position after action (mirrors AgentB logic)."""
    x, y = pos
    moves = {
        "up": (x, y - 1),
        "down": (x, y + 1),
        "left": (x - 1, y),
        "right": (x + 1, y),
    }
    nx, ny = moves.get(action, (x, y))
    if not (0 <= nx < width and 0 <= ny < height):
        return pos
    if (nx, ny) in obstacles:
        return pos
    return (nx, ny)


def random_grid_config(rng: random.Random) -> dict:
    """Generate diverse grid configs, biased toward obstacle-rich grids."""
    r = rng.random()
    if r < 0.05:
        # Small, easy
        w = h = rng.randint(5, 7)
        n_obs = rng.randint(0, 2)
    elif r < 0.35:
        # Medium
        w = h = rng.randint(8, 10)
        n_obs = rng.randint(5, 12)
    elif r < 0.70:
        # Large, obstacle-rich (where neural advantage matters)
        w = h = rng.randint(12, 15)
        n_obs = rng.randint(12, 25)
    else:
        # Very obstacle-rich medium
        w = h = rng.randint(8, 12)
        n_obs = rng.randint(10, 20)

    goals = [
        GoalDef("A", (w - 1, h - 1)),
        GoalDef("B", (w - 1, 0)),
    ]
    hint_cells = [HintCellDef(pos=(0, h - 1))]

    return dict(
        width=w, height=h,
        goals=goals, hint_cells=hint_cells,
        obstacles=[], n_random_obstacles=n_obs,
        seed=rng.randint(0, 999999),
    )


def collect_grid_samples(
    env: GridWorld,
    n_random_states: int = 20,
    rng: random.Random = None,
) -> List[Dict[str, Any]]:
    """Collect labelled samples from a single grid configuration.

    Instead of running full episodes, we sample random reachable positions
    and compute BFS labels for all 4 actions.  This is more data-efficient
    and ensures diverse state coverage (episodes are biased toward the path).
    """
    if rng is None:
        rng = random.Random()

    width, height = env.width, env.height
    obstacle_set = set(map(tuple, env.obstacles))
    goal_pos = env.true_goal_pos

    # BFS distance map from goal (one BFS for entire grid)
    dist_map = bfs_distance_map(goal_pos, obstacle_set, width, height)

    if not dist_map:
        return []  # goal unreachable (shouldn't happen)

    # Find all reachable positions
    reachable = list(dist_map.keys())
    if not reachable:
        return []

    # Sample random positions (+ include some near-obstacle positions)
    near_obstacle = []
    for ox, oy in obstacle_set:
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            pos = (ox + dx, oy + dy)
            if pos in dist_map:
                near_obstacle.append(pos)

    # Priority: 50% near-obstacle, 50% random reachable
    n_near = min(n_random_states // 2, len(near_obstacle))
    n_rand = n_random_states - n_near

    sampled = set()
    if near_obstacle:
        sampled.update(rng.sample(near_obstacle, min(n_near, len(near_obstacle))))
    if reachable:
        remaining = [p for p in reachable if p not in sampled]
        sampled.update(rng.sample(remaining, min(n_rand, len(remaining))))

    samples = []
    obstacles_list = list(obstacle_set)

    for pos in sampled:
        d_now = dist_map.get(pos, UNREACHABLE)
        if d_now >= UNREACHABLE:
            continue

        for action in ACTIONS:
            next_pos = compute_next_pos(pos, action, obstacle_set, width, height)
            d_next = dist_map.get(next_pos, UNREACHABLE)

            # BFS label
            if d_next >= UNREACHABLE:
                bfs_label = -1.0  # moving to unreachable = bad
            else:
                bfs_label = float(d_now - d_next)

            # Manhattan label
            manh_label = float(
                manhattan(pos, goal_pos) - manhattan(next_pos, goal_pos)
            )

            # Feature vector (stored as metadata for reconstruction)
            samples.append({
                "width": width,
                "height": height,
                "agent_pos": list(pos),
                "next_pos": list(next_pos),
                "goal_pos": list(goal_pos),
                "obstacles": [list(o) for o in obstacles_list],
                "action": action,
                "bfs_label": bfs_label,
                "manhattan_label": manh_label,
                "disagree": abs(bfs_label - manh_label) > 0.01,
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect BFS-labelled C data")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of grid configurations to sample")
    parser.add_argument("--states-per-grid", type=int, default=20,
                        help="Random states per grid config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="train/data/expert_c.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples = []
    n_disagree = 0

    print(f"Collecting from {args.episodes} grid configs, "
          f"{args.states_per_grid} states each...")

    for ep in range(args.episodes):
        cfg = random_grid_config(rng)
        env = GridWorld(**cfg)

        samples = collect_grid_samples(env, args.states_per_grid, rng)
        n_disagree += sum(1 for s in samples if s["disagree"])
        all_samples.extend(samples)

        if (ep + 1) % 1000 == 0:
            print(f"  {ep + 1}/{args.episodes} configs, "
                  f"{len(all_samples)} samples, "
                  f"disagree={n_disagree}/{len(all_samples)} "
                  f"({n_disagree / len(all_samples):.1%})")

    # Ensure output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_samples, f)

    print(f"\nDone: {len(all_samples)} samples from {args.episodes} configs")
    print(f"Disagree rate: {n_disagree}/{len(all_samples)} "
          f"({n_disagree / len(all_samples):.1%})")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

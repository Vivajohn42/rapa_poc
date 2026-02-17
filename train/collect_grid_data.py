"""Collect expert trajectories from deterministic agents on GridWorld.

Runs Agent A/B/C directly (no kernel overhead) on randomly configured
grids and records (observation, action, next_observation) tuples.
Agent C uses the true goal position for seeking — this is the expert policy.

Usage:
    python -m train.collect_grid_data --episodes 2000 --out train/data/grid_expert.json

Output format (JSON):
    [
      {
        "width": 10, "height": 10,
        "agent_pos": [3, 2], "goal_pos": [-1, -1],
        "obstacles": [[1,1], [2,3], ...],
        "hint": null,
        "action": "right",
        "next_agent_pos": [4, 2],
        "reward": -0.01, "done": false
      },
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld, GoalDef, HintCellDef
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec


def random_grid_config(rng: random.Random) -> dict:
    """Generate a random grid configuration for diverse training data."""
    # Mix of sizes: 60% medium (8-10), 30% large (12-15), 10% small (5-7)
    r = rng.random()
    if r < 0.1:
        w = h = rng.randint(5, 7)
        n_obs = rng.randint(0, 3)
    elif r < 0.7:
        w = h = rng.randint(8, 10)
        n_obs = rng.randint(3, 10)
    else:
        w = h = rng.randint(12, 15)
        n_obs = rng.randint(8, 20)

    # 2 goals (standard)
    goals = [
        GoalDef("A", (w - 1, h - 1)),
        GoalDef("B", (w - 1, 0)),
    ]

    hint_pos = (0, h - 1)
    hint_cells = [HintCellDef(pos=hint_pos)]

    return dict(
        width=w,
        height=h,
        goals=goals,
        hint_cells=hint_cells,
        obstacles=[],
        n_random_obstacles=n_obs,
        seed=rng.randint(0, 999999),
    )


def collect_episode(
    env: GridWorld,
    max_steps: int = 150,
) -> List[Dict[str, Any]]:
    """Run one episode using A/B/C agents directly (expert policy)."""
    obs = env.reset()

    agent_a = AgentA()
    agent_b = AgentB()
    # Expert policy: C knows the true goal position
    agent_c = AgentC(GoalSpec(mode="seek", target=env.true_goal_pos))

    transitions = []

    for t in range(max_steps):
        zA = agent_a.infer_zA(obs)

        # Agent C chooses action using B's forward model
        action, scored = agent_c.choose_action(
            zA, agent_b.predict_next,
        )

        next_obs, reward, done = env.step(action)

        transitions.append({
            "width": env.width,
            "height": env.height,
            "agent_pos": list(zA.agent_pos),
            "goal_pos": list(zA.goal_pos),
            "obstacles": [list(o) for o in zA.obstacles],
            "hint": zA.hint,
            "action": action,
            "next_agent_pos": list(next_obs.agent_pos),
            "reward": reward,
            "done": done,
        })

        if done:
            break

        obs = next_obs

    return transitions


def main():
    parser = argparse.ArgumentParser(description="Collect GridWorld expert data")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="train/data/grid_expert.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_transitions = []
    successes = 0

    print(f"Collecting {args.episodes} episodes...")

    for ep in range(args.episodes):
        cfg = random_grid_config(rng)
        env = GridWorld(**cfg)

        transitions = collect_episode(env, max_steps=args.max_steps)
        all_transitions.extend(transitions)

        if transitions and transitions[-1]["done"]:
            successes += 1

        if (ep + 1) % 200 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes, "
                  f"{len(all_transitions)} transitions, "
                  f"SR={successes / (ep + 1):.1%}")

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_transitions, f)

    print(f"\nDone: {len(all_transitions)} transitions from {args.episodes} episodes")
    print(f"Success rate: {successes / args.episodes:.1%}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

"""Collect SFT training data from deterministic agents playing GridWorld.

Runs episodes with det. Agents A/B/C/D, captures every D-invocation tick as
a (prompt, completion) pair in the exact format that AgentDLLM uses.

Output: JSON lines file with {prompt, completion, metadata} per sample.

Usage:
    cd rapa_mvp
    python -m train.collect_sft_data --episodes 500 --out train/data/sft_gridworld.jsonl
    python -m train.collect_sft_data --episodes 500 --out train/data/sft_gridworld.jsonl --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from env.gridworld import GridWorld
from kernel.kernel import MvpKernel
from router.deconstruct import deconstruct_d_to_c
from state.schema import ZA


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_prediction(zA: ZA, action: str, zA_next: ZA, goal_pos: Tuple[int, int]) -> Tuple[str, str]:
    """Build prediction text + predict:* tag from B's next-state."""
    if zA_next.agent_pos == zA.agent_pos:
        pred = f"Agent will be blocked at {zA.agent_pos} (wall or boundary)."
        tag = "predict:blocked"
    elif manhattan(zA_next.agent_pos, goal_pos) < manhattan(zA.agent_pos, goal_pos):
        pred = f"Agent moves closer to goal, reaching {zA_next.agent_pos}."
        tag = "predict:progress"
    else:
        pred = f"Agent moves to {zA_next.agent_pos}, further from the goal."
        tag = "predict:detour"
    return pred, tag


def build_prompt_completion(
    events: list,
    goal_mode: str,
    action: str | None,
    zA: ZA | None,
    zA_next: ZA | None,
    goal_pos: Tuple[int, int],
    last_n: int = 5,
) -> dict:
    """Build one SFT training sample from events + D's output."""
    slice_events = events[-last_n:] if events else []

    # === PROMPT (matches AgentDLLM system prompt — NO angle brackets) ===
    system = (
        "You are a narrative/meaning module.\n"
        "RULES:\n"
        "- Use ONLY the FACTS provided.\n"
        "- Do NOT invent positions/actions/rewards/hints.\n"
        "- Output EXACTLY two lines, for example:\n"
        "NARRATIVE: The agent moved right and discovered a hint.\n"
        "TAGS: movement, hint:A, goal:seek\n"
    )

    facts = [
        f"t={e['t']} pos={e['pos']} action={e['action']} "
        f"reward={e['reward']} done={e['done']} hint={e['hint']}"
        for e in slice_events
    ]
    user = "FACTS:\n" + "\n".join(facts) + f"\nMODE={goal_mode}\n"

    if action is not None:
        user += f"ACTION_TAKEN={action}\n"
    if zA_next is not None:
        user += f"NEXT_POS={zA_next.agent_pos}\n"

    prompt = system + "\n" + user

    # === COMPLETION (from deterministic D + B prediction) ===
    # Narrative from det D logic
    if not slice_events:
        narrative = "No events recorded."
        tags = ["empty"]
        prediction = ""
        pred_tag = ""
    else:
        start = slice_events[0]["pos"]
        end = slice_events[-1]["pos"]
        steps = len(slice_events)
        total_reward = sum(e["reward"] for e in slice_events)

        # Build narrative
        actions_str = ", ".join(e["action"] for e in slice_events)
        positions = [e["pos"] for e in slice_events]
        unique_pos = len(set(str(p) for p in positions))

        if unique_pos == 1:
            narrative = f"Agent stayed at {start} for {steps} steps in {goal_mode} mode, unable to make progress."
        else:
            narrative = (
                f"Agent moved from {start} to {end} over {steps} steps "
                f"in {goal_mode} mode, taking actions: {actions_str}."
            )

        # Build tags
        tags = [f"goal:{goal_mode}"]
        if unique_pos == 1:
            tags.append("stability:stuck")
        else:
            tags.append("stability:moving")

        # Hint tag
        hint = None
        for e in reversed(slice_events):
            if e["hint"] is not None:
                hint = e["hint"]
                break
        if hint:
            tags.append(f"hint:{hint}")

        # Prediction
        prediction = ""
        pred_tag = ""
        if action and zA and zA_next and goal_pos != (-1, -1):
            prediction, pred_tag = build_prediction(zA, action, zA_next, goal_pos)
            tags.append(pred_tag)

    # Format completion
    completion = f"NARRATIVE: {narrative}\n"
    if prediction:
        completion += f"PREDICTION: {prediction}\n"
    completion += f"TAGS: {', '.join(tags)}\n"

    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "goal_mode": goal_mode,
            "action": action,
            "n_events": len(slice_events),
        },
    }


def collect_gridworld_episodes(
    n_episodes: int = 500,
    seed_start: int = 0,
    max_steps: int = 50,
    width: int = 5,
    height: int = 5,
) -> List[dict]:
    """Run GridWorld episodes and collect SFT samples."""
    samples = []

    for ep in range(n_episodes):
        seed = seed_start + ep
        env = GridWorld(width=width, height=height, seed=seed)

        agent_a = AgentA()
        agent_b = AgentB()
        agent_c = AgentC(goal=GoalSpec(mode="seek", target=(-1, -1)))
        agent_d = AgentD()

        kernel = MvpKernel(
            agent_a=agent_a, agent_b=agent_b,
            agent_c=agent_c, agent_d=agent_d,
            deconstruct_fn=deconstruct_d_to_c,
        )

        goal_map = env.goal_positions
        kernel.reset_episode(goal_mode="seek", episode_id=f"sft_ep{ep:04d}")
        if hasattr(kernel, '_memory_manager') and kernel._memory_manager:
            kernel._memory_manager.goal_map = goal_map

        obs = env.reset()
        events = []

        for t in range(max_steps):
            zA = agent_a.infer_zA(obs)

            result = kernel.tick(t, obs, done=False)
            action = result.action

            # Record event
            events.append({
                "t": t,
                "pos": tuple(zA.agent_pos),
                "action": action,
                "reward": 0.0,
                "done": False,
                "hint": zA.hint,
            })

            # If D was activated this tick, capture a sample
            if result.d_activated:
                zA_next = agent_b.predict_next(zA, action)
                # Find current goal target (might be unknown)
                target = kernel._zC.memory.get("target", (-1, -1)) if kernel._zC else (-1, -1)
                if target is None:
                    target = (-1, -1)

                sample = build_prompt_completion(
                    events=events,
                    goal_mode="seek",
                    action=action,
                    zA=zA,
                    zA_next=zA_next,
                    goal_pos=target,
                    last_n=5,
                )
                samples.append(sample)

            obs_next, reward, done = env.step(action)
            # Update last event with actual reward
            events[-1]["reward"] = reward

            kernel.observe_reward(reward)
            obs = obs_next

            if done:
                events[-1]["done"] = True
                kernel.tick(t + 1, obs, done=True)
                break

        if (ep + 1) % 100 == 0:
            print(f"  Episodes: {ep + 1}/{n_episodes}, samples so far: {len(samples)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect SFT data from GridWorld")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--out", type=str, default="train/data/sft_gridworld.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    args = parser.parse_args()

    print(f"Collecting SFT data: {args.episodes} episodes, seed={args.seed}")
    print(f"Grid: {args.width}x{args.height}, max_steps={args.max_steps}")

    samples = collect_gridworld_episodes(
        n_episodes=args.episodes,
        seed_start=args.seed,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
    )

    # Write JSONL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(samples)} samples written to {out_path}")
    # Stats
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / max(len(samples), 1)
    avg_completion_len = sum(len(s["completion"]) for s in samples) / max(len(samples), 1)
    print(f"  Avg prompt chars: {avg_prompt_len:.0f}")
    print(f"  Avg completion chars: {avg_completion_len:.0f}")


if __name__ == "__main__":
    main()

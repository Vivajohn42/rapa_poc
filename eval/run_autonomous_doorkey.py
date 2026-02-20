"""RAPA Autonomous DoorKey — Learning without labels.

The agent learns to solve DoorKey purely from experience over ~100 episodes.
No BFS-expert, no labels, no LLM.

Three new components:
  - ObjectMemory (A-Level): Ego-view world model
  - EventPatternD (D-Level): Learns task sequence from experience
  - AutonomousDoorKeyAgentC (C-Level): Frontier + known-grid BFS

5 Assertions:
  1. D learns constraints or partial hypothesis within 30 episodes
  2. SR > 50% within first 50 episodes
  3. SR > 80% in last 20 episodes (size=6)
  4. D learns correct order [KEY_PICKED_UP, DOOR_OPENED, GOAL_REACHED]
  5. No BFS-Expert/Labels imported (honesty check)

Usage:
    python eval/run_autonomous_doorkey.py
    python eval/run_autonomous_doorkey.py --n 30 --size 6
    python eval/run_autonomous_doorkey.py --n 100 --size 8
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey import DoorKeyEnv
from agents.doorkey_agent_a import DoorKeyAgentA
from agents.doorkey_agent_b import DoorKeyAgentB
from agents.autonomous_doorkey_agent_c import AutonomousDoorKeyAgentC
from agents.event_pattern_d import EventPatternD, DoorKeyEventType
from agents.object_memory import ObjectMemory
from kernel.kernel import MvpKernel
from router.deconstruct_doorkey import deconstruct_doorkey_d_to_c


def _derive_phase(om: ObjectMemory) -> str:
    """Derive DoorKey phase from ObjectMemory state."""
    if om.door_open:
        return "REACH_GOAL"
    elif om.carrying_key:
        return "OPEN_DOOR"
    return "FIND_KEY"


def _recent_sr(results: List[Dict], window: int) -> float:
    """Compute success rate over the last `window` results."""
    if len(results) < window:
        return 0.0
    recent = results[-window:]
    return sum(1 for r in recent if r["success"]) / len(recent)


def run_autonomous(
    n_episodes: int = 100,
    size: int = 6,
    max_steps: int = 200,
    seed_base: int = 42,
    stagnation_window: int = 5,
    verbose: bool = False,
) -> Tuple[List[Dict], EventPatternD]:
    """Run the autonomous DoorKey learning experiment.

    Returns (results_list, event_d) for assertion checking.
    """
    # Persistent D across all episodes
    event_d = EventPatternD()
    results: List[Dict] = []

    print(f"\nRAPY Autonomous DoorKey: {n_episodes} episodes, "
          f"size={size}, max_steps={max_steps}\n")

    t0 = time.time()

    for ep in range(n_episodes):
        # ── Per-episode setup ──
        env = DoorKeyEnv(size=size, seed=seed_base + ep)
        obs = env.reset()
        obj_mem = ObjectMemory(grid_size=size)

        agent_a = DoorKeyAgentA()
        agent_b = DoorKeyAgentB()
        agent_c = AutonomousDoorKeyAgentC(goal_mode="seek")
        agent_c.set_object_memory(obj_mem)
        event_d.set_object_memory(obj_mem)
        event_d.reset_episode()

        # Default goal_pos for deconstruction fallback
        # (goal is typically at (size-2, size-2) but we don't assume this)
        kernel = MvpKernel(
            agent_a=agent_a,
            agent_b=agent_b,
            agent_c=agent_c,
            agent_d=event_d,
            goal_map=None,  # no privileged goal_map
            enable_governance=True,
            deconstruct_fn=deconstruct_doorkey_d_to_c,
            fallback_actions=["turn_left", "turn_right", "forward",
                              "pickup", "toggle"],
        )
        kernel.reset_episode(goal_mode="seek",
                             episode_id=f"auto_{ep}")

        done = False
        reward = 0.0
        step_count = 0

        for t in range(max_steps):
            # 1. ObjectMemory scans ego-view (BEFORE kernel.tick)
            obj_mem.update(env._env.unwrapped)

            # 2. Sync C and B with ObjectMemory state
            phase = _derive_phase(obj_mem)
            agent_c.phase = phase
            agent_c.key_pos = obj_mem.key_pos
            agent_c.door_pos = obj_mem.door_pos
            agent_c.carrying_key = obj_mem.carrying_key
            agent_c.door_open = obj_mem.door_open
            agent_b.update_door_state(obj_mem.door_pos, obj_mem.door_open)

            # 3. Kernel tick
            result = kernel.tick(t, obs, done=False)

            # 4. Environment step
            obs, reward, done = env.step(result.action)
            kernel.observe_reward(reward)
            step_count = t + 1

            if done:
                # Final ObjectMemory update + episode-end tick
                obj_mem.update(env._env.unwrapped)
                kernel.tick(t + 1, obs, done=True)
                break

        # 5. Episode end: D learns
        success = done and reward > 0
        event_d.end_episode(success=success, steps=step_count)

        ep_info = {
            "episode": ep,
            "success": success,
            "steps": step_count,
            "reward": round(reward, 4),
            "has_hypothesis": event_d.has_hypothesis,
            "seq_known": event_d.success_sequence is not None,
            "n_constraints": len(event_d.negative_constraints),
            "n_partial": len(event_d.partial_hypotheses),
        }
        results.append(ep_info)

        if verbose or ep < 10 or ep % 10 == 9:
            sr_so_far = sum(1 for r in results if r["success"]) / len(results)
            hyp = ("SEQ" if ep_info["seq_known"]
                   else f"P={ep_info['n_partial']},C={ep_info['n_constraints']}"
                   if ep_info["has_hypothesis"]
                   else "none")
            status = "OK" if success else "FAIL"
            print(f"  ep {ep:3d}: {status:4s}  steps={step_count:3d}  "
                  f"SR={sr_so_far:.0%}  hyp={hyp}")

        # 6. Stagnation check every 5 episodes
        if ((ep + 1) % stagnation_window == 0
                and ep >= 2 * stagnation_window):
            recent_sr = _recent_sr(results, stagnation_window)
            prev_sr = _recent_sr(
                results[:-stagnation_window], stagnation_window)
            if recent_sr <= prev_sr + 0.05:
                event_d.reflect()
                if verbose:
                    print(f"    -> reflect() triggered (SR {prev_sr:.0%}"
                          f" -> {recent_sr:.0%})")

        env.close()

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s "
          f"({elapsed / n_episodes:.2f}s per episode)")

    return results, event_d


def run_assertions(
    results: List[Dict],
    event_d: EventPatternD,
    size: int,
) -> bool:
    """Run the 5 assertions and print results."""
    print("\n" + "=" * 60)
    print("ASSERTIONS:")
    print("=" * 60)
    checks = []

    # ── Assertion 1: D learns constraints or hypothesis within 30 eps ──
    hypothesis_at = None
    for r in results:
        if r["has_hypothesis"]:
            hypothesis_at = r["episode"]
            break
    p1 = hypothesis_at is not None and hypothesis_at < 30
    checks.append(p1)
    print(f"  [{'PASS' if p1 else 'FAIL'}] 1. D learns hypothesis "
          f"within 30 eps: ep={hypothesis_at}")

    # ── Assertion 2: SR > 50% within first 50 episodes ──
    first_50 = results[:min(50, len(results))]
    sr_50 = (sum(1 for r in first_50 if r["success"]) / len(first_50)
             if first_50 else 0.0)
    p2 = sr_50 > 0.50
    checks.append(p2)
    print(f"  [{'PASS' if p2 else 'FAIL'}] 2. SR > 50% in first 50 eps: "
          f"{sr_50:.1%}")

    # ── Assertion 3: SR > 80% in last 20 episodes ──
    last_20 = results[-20:] if len(results) >= 20 else results
    sr_last = (sum(1 for r in last_20 if r["success"]) / len(last_20)
               if last_20 else 0.0)
    threshold = 0.80 if size <= 6 else 0.60
    p3 = sr_last > threshold
    checks.append(p3)
    print(f"  [{'PASS' if p3 else 'FAIL'}] 3. SR > {threshold:.0%} in "
          f"last 20 eps: {sr_last:.1%}")

    # ── Assertion 4: D learns correct order ──
    correct_order = [
        DoorKeyEventType.KEY_PICKED_UP,
        DoorKeyEventType.DOOR_OPENED,
        DoorKeyEventType.GOAL_REACHED,
    ]
    p4 = event_d.success_sequence == correct_order
    seq_str = ([e.name for e in event_d.success_sequence]
               if event_d.success_sequence else "None")
    checks.append(p4)
    print(f"  [{'PASS' if p4 else 'FAIL'}] 4. Correct order "
          f"[KEY->DOOR->GOAL]: {seq_str}")

    # ── Assertion 5: No BFS-Expert/Labels imported ──
    p5 = True
    forbidden = ["bfs_expert", "expert_doorkey", "collect_expert",
                 "neural_doorkey_agent_c", "doorkey_action_value_net"]
    for mod_name in ["agents.object_memory",
                     "agents.event_pattern_d",
                     "agents.autonomous_doorkey_agent_c"]:
        try:
            mod = importlib.import_module(mod_name)
            source = inspect.getsource(mod)
            for frag in forbidden:
                if frag in source:
                    p5 = False
                    print(f"    !! Found '{frag}' in {mod_name}")
        except Exception as e:
            print(f"    !! Could not inspect {mod_name}: {e}")
    checks.append(p5)
    print(f"  [{'PASS' if p5 else 'FAIL'}] 5. No BFS-Expert/Labels imported")

    # ── Summary ──
    n_pass = sum(checks)
    all_pass = all(checks)
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({n_pass}/{len(checks)})")

    return all_pass


def save_csv(results: List[Dict], size: int) -> Path:
    """Save results to CSV."""
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"autonomous_doorkey_{size}x{size}_{ts}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"  Results saved to {path}")
    return path


def main() -> bool:
    parser = argparse.ArgumentParser(
        description="RAPA Autonomous DoorKey — Learning without labels")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of episodes (default: 100)")
    parser.add_argument("--size", type=int, default=6,
                        help="Grid size (default: 6)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every episode")
    args = parser.parse_args()

    results, event_d = run_autonomous(
        n_episodes=args.n,
        size=args.size,
        max_steps=args.max_steps,
        seed_base=args.seed,
        verbose=args.verbose,
    )

    # Print learning summary
    print("\n" + "-" * 60)
    print("LEARNING SUMMARY:")
    print(f"  Episodes: {len(results)}")
    print(f"  Success sequence: "
          f"{[e.name for e in event_d.success_sequence] if event_d.success_sequence else 'None'}")
    print(f"  Partial hypotheses: {event_d.partial_hypotheses}")
    print(f"  Negative constraints: {event_d.negative_constraints}")
    total_sr = sum(1 for r in results if r["success"]) / len(results)
    print(f"  Overall SR: {total_sr:.1%}")

    # Learning curve (10-episode windows)
    print("\n  Learning Curve (10-ep windows):")
    for i in range(0, len(results), 10):
        window = results[i:i + 10]
        w_sr = sum(1 for r in window if r["success"]) / len(window)
        print(f"    ep {i:3d}-{i + len(window) - 1:3d}: SR={w_sr:.0%}")

    save_csv(results, args.size)
    return run_assertions(results, event_d, args.size)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

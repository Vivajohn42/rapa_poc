"""DoorKey D-essentiality ablation: with_d vs no_d vs random.

Tests whether D (narrative + phase tracking) is essential for DoorKey.
Expected: with_d ~100%, no_d ~0%, random <5%.

D is essential because:
  - Only D's deconstruction sets zC.memory["target"]
  - Without target, C cannot score pickup/toggle actions
  - Without pickup/toggle, agent cannot pick up key or open door

Usage:
    python eval/run_doorkey_ablation.py
    python eval/run_doorkey_ablation.py --n 50 --size 6
    python eval/run_doorkey_ablation.py --size 5 --n 10  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey_adapter import DoorKeyAdapter
from kernel.kernel import MvpKernel


def run_episode(
    variant: str,
    size: int = 6,
    seed: int = 42,
    max_steps: int = 200,
) -> Dict:
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_steps)
    obs = adapter.reset()

    if variant == "random":
        rng = random.Random(seed)
        actions = adapter.available_actions()
        done = False
        reward = 0.0
        steps = 0
        for t in range(max_steps):
            action = rng.choice(actions)
            obs, reward, done = adapter.step(action)
            steps = t + 1
            if done:
                break
        return {
            "variant": variant, "seed": seed, "size": size,
            "success": done and reward > 0, "steps": steps,
            "reward": round(reward, 4),
        }

    A, B, C, D = adapter.make_agents(variant=variant)
    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=adapter.get_goal_map(),
        enable_governance=True,
        deconstruct_fn=adapter.get_deconstruct_fn(),
        fallback_actions=adapter.available_actions(),
    )
    kernel.reset_episode(goal_mode="seek",
                         episode_id=f"dk_{variant}_{seed}")

    done = False
    reward = 0.0
    for t in range(max_steps):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        obs, reward, done = adapter.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    return {
        "variant": variant, "seed": seed, "size": size,
        "success": done and reward > 0, "steps": t + 1,
        "reward": round(reward, 4),
    }


def main() -> bool:
    parser = argparse.ArgumentParser(
        description="DoorKey D-essentiality ablation")
    parser.add_argument("--n", type=int, default=30,
                        help="Episodes per variant")
    parser.add_argument("--size", type=int, default=6,
                        help="Grid size (5 or 6)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 64)
    print(f"  DoorKey-{args.size}x{args.size} D-Essentiality Ablation")
    print(f"  Episodes per variant: {args.n}")
    print("=" * 64)

    all_results: List[Dict] = []
    for variant in ["with_d", "no_d", "random"]:
        for i in range(args.n):
            r = run_episode(variant, size=args.size,
                            seed=42 + i, max_steps=args.max_steps)
            all_results.append(r)

        vr = [r for r in all_results if r["variant"] == variant]
        sr = sum(1 for r in vr if r["success"]) / len(vr)
        avg_steps = sum(r["steps"] for r in vr) / len(vr)
        avg_reward = sum(r["reward"] for r in vr) / len(vr)
        print(f"  {variant:<10s}: SR={sr:.1%}  "
              f"avg_steps={avg_steps:.1f}  avg_reward={avg_reward:.3f}")

    # ── Assertions ──────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  ASSERTIONS")
    print("=" * 64)

    wd = [r for r in all_results if r["variant"] == "with_d"]
    nd = [r for r in all_results if r["variant"] == "no_d"]
    rd = [r for r in all_results if r["variant"] == "random"]

    sr_wd = sum(1 for r in wd if r["success"]) / len(wd)
    sr_nd = sum(1 for r in nd if r["success"]) / len(nd)
    sr_rd = sum(1 for r in rd if r["success"]) / len(rd)

    checks = []

    p1 = sr_wd >= 0.90
    checks.append(p1)
    print(f"  [{'PASS' if p1 else 'FAIL'}] 1. with_d SR >= 90%: "
          f"{sr_wd:.1%}")

    p2 = sr_wd > sr_nd
    checks.append(p2)
    print(f"  [{'PASS' if p2 else 'FAIL'}] 2. with_d > no_d: "
          f"{sr_wd:.1%} > {sr_nd:.1%}")

    p3 = sr_wd > sr_rd
    checks.append(p3)
    print(f"  [{'PASS' if p3 else 'FAIL'}] 3. with_d > random: "
          f"{sr_wd:.1%} > {sr_rd:.1%}")

    p4 = (sr_wd - sr_nd) >= 0.40
    checks.append(p4)
    print(f"  [{'PASS' if p4 else 'FAIL'}] 4. D-advantage >= 40pp: "
          f"{sr_wd - sr_nd:.1%}")

    avg_steps_wd = sum(r["steps"] for r in wd) / len(wd)
    p5 = avg_steps_wd <= 50
    checks.append(p5)
    print(f"  [{'PASS' if p5 else 'FAIL'}] 5. with_d avg_steps <= 50: "
          f"{avg_steps_wd:.1f}")

    # ── CSV output ──────────────────────────────────────────────
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"doorkey_ablation_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "seed", "size",
                           "success", "steps", "reward"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  CSV: {csv_path}")

    all_pass = all(checks)
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

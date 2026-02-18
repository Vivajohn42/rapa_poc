"""Neural DoorKey C evaluation: det_c vs neural_c vs neural_c_no_d vs random.

Tests whether a BFS-trained DoorKeyActionValueNet matches or exceeds the
deterministic BFS+turn heuristic on DoorKey grids.

4 variants × multiple grid sizes × N episodes:

  det_c         — DoorKeyAgentC + DoorKeyAgentD  (baseline)
  neural_c      — NeuralDoorKeyAgentC + DoorKeyAgentD  (hybrid scoring)
  neural_c_no_d — NeuralDoorKeyAgentC, no D  (D-essentiality check)
  random        — Random action selection  (chance baseline)

Usage:
    python eval/run_neural_doorkey_eval.py --n 50 --sizes 6,8
    python eval/run_neural_doorkey_eval.py --n 10 --sizes 6       # smoke
    python eval/run_neural_doorkey_eval.py --n 100 --sizes 6,8,16 # full
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env.doorkey_adapter import DoorKeyAdapter
from kernel.kernel import MvpKernel
from models.doorkey_action_value_net import DoorKeyActionValueNet

VARIANTS = ["det_c", "neural_c", "neural_c_no_d", "random"]


def _load_value_net() -> DoorKeyActionValueNet:
    """Load trained DoorKeyActionValueNet checkpoint."""
    ckpt_path = Path("train/checkpoints/doorkey_action_value_net.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Run: python -m train.train_doorkey_c first."
        )
    net = DoorKeyActionValueNet()
    net.load_state_dict(torch.load(ckpt_path, weights_only=True))
    net.eval()
    return net


def _max_steps_for_size(size: int) -> int:
    """Return appropriate max_steps for a grid size."""
    if size <= 6:
        return 200
    elif size <= 8:
        return 400
    else:
        return 1200


def run_episode(
    variant: str,
    size: int,
    seed: int,
    max_steps: int,
    value_net: Optional[DoorKeyActionValueNet] = None,
) -> Dict:
    """Run one DoorKey episode with given variant."""
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_steps)
    obs = adapter.reset()

    # Random baseline
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

    # Kernel-based variants
    A, B, C, D = adapter.make_agents(
        variant=variant, value_net=value_net,
    )
    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=adapter.get_goal_map(),
        enable_governance=True,
        deconstruct_fn=adapter.get_deconstruct_fn(),
        fallback_actions=adapter.available_actions(),
    )
    kernel.reset_episode(
        goal_mode="seek",
        episode_id=f"dk_{variant}_{size}_{seed}",
    )

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
        description="Neural DoorKey C Evaluation")
    parser.add_argument("--n", type=int, default=50,
                        help="Episodes per variant per size")
    parser.add_argument("--sizes", type=str, default="6,8",
                        help="Grid sizes (comma-separated)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print("=" * 70)
    print(f"  Neural DoorKey C — Evaluation")
    print(f"  Sizes: {sizes} | Episodes per cell: {args.n}")
    print("=" * 70)

    # Load value net
    print("\nLoading DoorKeyActionValueNet checkpoint...")
    value_net = _load_value_net()
    n_params = sum(p.numel() for p in value_net.parameters())
    print(f"  DoorKeyActionValueNet: {n_params:,} params\n")

    all_results: List[Dict] = []

    for size in sizes:
        max_steps = args.max_steps or _max_steps_for_size(size)
        print(f"  DoorKey-{size}x{size}  (max_steps={max_steps})")
        print(f"  {'-' * 50}")

        for variant in VARIANTS:
            successes = 0
            total_steps = 0
            for i in range(args.n):
                r = run_episode(
                    variant=variant, size=size, seed=42 + i,
                    max_steps=max_steps, value_net=value_net,
                )
                all_results.append(r)
                if r["success"]:
                    successes += 1
                total_steps += r["steps"]

            sr = successes / args.n
            avg_steps = total_steps / args.n
            print(f"    {variant:<18s}: SR={sr:>5.1%}  "
                  f"avg_steps={avg_steps:>6.1f}  ({successes}/{args.n})")

        print()

    # ── Assertions ──────────────────────────────────────────────
    print("=" * 70)
    print("  ASSERTIONS")
    print("=" * 70)

    checks = []

    def _sr(variant: str, size: int) -> float:
        sub = [r for r in all_results
               if r["variant"] == variant and r["size"] == size]
        if not sub:
            return 0.0
        return sum(1 for r in sub if r["success"]) / len(sub)

    def _avg_steps(variant: str, size: int) -> float:
        sub = [r for r in all_results
               if r["variant"] == variant and r["size"] == size]
        if not sub:
            return 999.0
        return sum(r["steps"] for r in sub) / len(sub)

    # Smallest size (typically 6) assertions
    s0 = sizes[0]

    # 1. neural_c SR >= 90% on smallest size
    sr_nc = _sr("neural_c", s0)
    p1 = sr_nc >= 0.90
    checks.append(p1)
    print(f"  [{'PASS' if p1 else 'FAIL'}] 1. neural_c SR >= 90% on "
          f"{s0}x{s0}: {sr_nc:.1%}")

    # 2. neural_c >= det_c - 5pp on smallest size (parity)
    sr_dc = _sr("det_c", s0)
    p2 = sr_nc >= sr_dc - 0.05
    checks.append(p2)
    print(f"  [{'PASS' if p2 else 'FAIL'}] 2. neural_c >= det_c - 5pp on "
          f"{s0}x{s0}: {sr_nc:.1%} vs {sr_dc:.1%}")

    # 3. neural_c >= det_c on largest size (generalization)
    if len(sizes) > 1:
        s_big = sizes[-1]
        sr_nc_big = _sr("neural_c", s_big)
        sr_dc_big = _sr("det_c", s_big)
        p3 = sr_nc_big >= sr_dc_big
        checks.append(p3)
        print(f"  [{'PASS' if p3 else 'FAIL'}] 3. neural_c >= det_c on "
              f"{s_big}x{s_big}: {sr_nc_big:.1%} vs {sr_dc_big:.1%}")
    else:
        print(f"  [SKIP] 3. Only one size — skipping generalization check")

    # 4. neural_c_no_d SR == 0% (D-essentiality)
    sr_ncnd = _sr("neural_c_no_d", s0)
    p4 = sr_ncnd == 0.0
    checks.append(p4)
    print(f"  [{'PASS' if p4 else 'FAIL'}] 4. neural_c_no_d SR == 0% on "
          f"{s0}x{s0}: {sr_ncnd:.1%}")

    # 5. D-advantage >= 40pp (neural_c - neural_c_no_d)
    d_advantage = sr_nc - sr_ncnd
    p5 = d_advantage >= 0.40
    checks.append(p5)
    print(f"  [{'PASS' if p5 else 'FAIL'}] 5. D-advantage >= 40pp: "
          f"{d_advantage:.1%}")

    # 6. neural_c avg_steps <= 50 on smallest size
    avg_nc = _avg_steps("neural_c", s0)
    p6 = avg_nc <= 50
    checks.append(p6)
    print(f"  [{'PASS' if p6 else 'FAIL'}] 6. neural_c avg_steps <= 50 on "
          f"{s0}x{s0}: {avg_nc:.1f}")

    # 7. random SR < 15% on smallest size
    sr_rand = _sr("random", s0)
    p7 = sr_rand < 0.15
    checks.append(p7)
    print(f"  [{'PASS' if p7 else 'FAIL'}] 7. random SR < 15% on "
          f"{s0}x{s0}: {sr_rand:.1%}")

    # ── CSV output ──────────────────────────────────────────────
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"neural_doorkey_eval_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "seed", "size",
                           "success", "steps", "reward"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  CSV: {csv_path}")

    all_pass = all(checks)
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({sum(checks)}/{len(checks)})")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

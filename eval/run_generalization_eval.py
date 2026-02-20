"""Generalization Evaluation — Neural nets trained without 16x16.

Tests whether neural networks trained only on 5x5/6x6/8x8 grids
generalize to unseen 16x16 grids. Compares:

  original    — trained on 5/35/30/30% (5,6,8,16)
  no16        — trained on 10/50/40/0% (5,6,8 only)

Both DoorKeyActionValueNet (Neural C) and DirectionPriorNet (Neural B)
are tested.

Usage:
    python eval/run_generalization_eval.py --n 200
    python eval/run_generalization_eval.py --n 50   # quick check
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
from models.direction_prior_net import DirectionPriorNet


# ── Checkpoint paths ──────────────────────────────────────────────
CKPT_DIR = Path(__file__).resolve().parent.parent / "train" / "checkpoints"

NEURAL_C_CKPTS = {
    "original": CKPT_DIR / "doorkey_action_value_net.pt",
    "no16":     CKPT_DIR / "doorkey_action_value_net_no16.pt",
}

NEURAL_B_CKPTS = {
    "original": CKPT_DIR / "direction_prior_net.pt",
    "no16":     CKPT_DIR / "direction_prior_net_no16.pt",
}


def _load_net(cls, path: Path):
    """Load a net checkpoint, return None if not found."""
    if not path.exists():
        return None
    net = cls()
    net.load_state_dict(torch.load(str(path), weights_only=True))
    net.eval()
    return net


def _max_steps(size: int) -> int:
    if size <= 6:
        return 200
    elif size <= 8:
        return 400
    return 1200


# ── Neural C evaluation ──────────────────────────────────────────

def run_neural_c_episode(
    size: int,
    seed: int,
    value_net: DoorKeyActionValueNet,
) -> Dict:
    """Run one DoorKey episode with NeuralDoorKeyAgentC."""
    max_s = _max_steps(size)
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_s)
    obs = adapter.reset()

    A, B, C, D = adapter.make_agents(variant="neural_c", value_net=value_net)
    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=adapter.get_goal_map(),
        enable_governance=True,
        deconstruct_fn=adapter.get_deconstruct_fn(),
        fallback_actions=adapter.available_actions(),
    )
    kernel.reset_episode(goal_mode="seek", episode_id=f"gen_nc_{size}_{seed}")

    done = False
    reward = 0.0
    for t in range(max_s):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        obs, reward, done = adapter.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    return {
        "success": done and reward > 0,
        "steps": t + 1,
    }


# ── Neural B evaluation (active mode) ────────────────────────────

def run_neural_b_episode(
    size: int,
    seed: int,
    prior_net: DirectionPriorNet,
) -> Dict:
    """Run one DoorKey episode with active compression + neural prior."""
    max_s = _max_steps(size)
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_s)
    obs = adapter.reset()

    A, B, C, D = adapter.make_agents(variant="with_d")
    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=adapter.get_goal_map(),
        enable_governance=True,
        deconstruct_fn=adapter.get_deconstruct_fn(),
        fallback_actions=adapter.available_actions(),
        use_unified_memory=True,
        active_compression=True,
    )
    kernel.set_direction_prior_net(prior_net)
    kernel.reset_episode(goal_mode="seek", episode_id=f"gen_nb_{size}_{seed}")

    done = False
    reward = 0.0
    c_comp = 0
    for t in range(max_s):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        if result.c_compressed:
            c_comp += 1
        obs, reward, done = adapter.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    return {
        "success": done and reward > 0,
        "steps": t + 1,
        "c_compressed_ticks": c_comp,
    }


def _sr(results: List[Dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> bool:
    parser = argparse.ArgumentParser(description="Generalization Eval")
    parser.add_argument("--n", type=int, default=200,
                        help="Episodes per cell")
    args = parser.parse_args()

    n = args.n
    sizes = [6, 8, 16]

    print("=" * 78)
    print("  Neural Generalization Evaluation - Trained Without 16x16")
    print(f"  Episodes per cell: {n}")
    print("=" * 78)

    # ══════════════════════════════════════════════════════════════
    # Part 1: Neural C (DoorKeyActionValueNet)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-" * 78)
    print("  Part 1: Neural C (DoorKeyActionValueNet)")
    print("-" * 78)

    c_nets = {}
    for label, path in NEURAL_C_CKPTS.items():
        net = _load_net(DoorKeyActionValueNet, path)
        if net is None:
            print(f"  [SKIP] {label}: checkpoint not found at {path}")
        else:
            n_params = sum(p.numel() for p in net.parameters())
            print(f"  Loaded {label}: {n_params:,} params")
            c_nets[label] = net

    c_results: Dict[str, Dict[int, List[Dict]]] = {}  # label -> size -> results

    for label, net in c_nets.items():
        c_results[label] = {}
        for size in sizes:
            results = []
            for i in range(n):
                r = run_neural_c_episode(size, seed=42 + i, value_net=net)
                results.append(r)
            c_results[label][size] = results

    if c_nets:
        print(f"\n  {'':15s}", end="")
        for size in sizes:
            print(f"  {size}x{size:>2d}  ", end="")
        print()
        print(f"  {'':15s}", end="")
        for _ in sizes:
            print(f"  ------", end="")
        print()

        for label in c_nets:
            print(f"  {label:15s}", end="")
            for size in sizes:
                sr = _sr(c_results[label][size])
                print(f"  {sr:5.1%} ", end="")
            print()

    # ══════════════════════════════════════════════════════════════
    # Part 2: Neural B (DirectionPriorNet, active mode)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-" * 78)
    print("  Part 2: Neural B (DirectionPriorNet, active mode)")
    print("-" * 78)

    b_nets = {}
    for label, path in NEURAL_B_CKPTS.items():
        net = _load_net(DirectionPriorNet, path)
        if net is None:
            print(f"  [SKIP] {label}: checkpoint not found at {path}")
        else:
            n_params = sum(p.numel() for p in net.parameters())
            print(f"  Loaded {label}: {n_params:,} params")
            b_nets[label] = net

    b_results: Dict[str, Dict[int, List[Dict]]] = {}

    for label, net in b_nets.items():
        b_results[label] = {}
        for size in sizes:
            results = []
            for i in range(n):
                r = run_neural_b_episode(size, seed=42 + i, prior_net=net)
                results.append(r)
            b_results[label][size] = results

    if b_nets:
        print(f"\n  {'':15s}", end="")
        for size in sizes:
            print(f"  {size}x{size:>2d}  ", end="")
        print()
        print(f"  {'':15s}", end="")
        for _ in sizes:
            print(f"  ------", end="")
        print()

        for label in b_nets:
            print(f"  {label:15s}", end="")
            for size in sizes:
                sr = _sr(b_results[label][size])
                print(f"  {sr:5.1%} ", end="")
            print()

        # C-compressed ticks comparison
        print(f"\n  C-compressed %:")
        for label in b_nets:
            print(f"  {label:15s}", end="")
            for size in sizes:
                rr = b_results[label][size]
                total_ticks = sum(r["steps"] for r in rr)
                c_comp = sum(r["c_compressed_ticks"] for r in rr)
                pct = c_comp / max(total_ticks, 1) * 100
                print(f"  {pct:5.1f}%", end="")
            print()

    # ══════════════════════════════════════════════════════════════
    # Assertions
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  ASSERTIONS")
    print("=" * 78)

    checks = []

    # Neural C assertions
    if "no16" in c_nets:
        # 1. no16 SR >= 90% on 6x6
        sr_c_no16_6 = _sr(c_results["no16"][6])
        p1 = sr_c_no16_6 >= 0.90
        checks.append(p1)
        print(f"\n  [{'PASS' if p1 else 'FAIL'}] 1. Neural C (no16) SR >= 90% on 6x6: "
              f"{sr_c_no16_6:.1%}")

        # 2. no16 SR >= 90% on 8x8
        sr_c_no16_8 = _sr(c_results["no16"][8])
        p2 = sr_c_no16_8 >= 0.90
        checks.append(p2)
        print(f"  [{'PASS' if p2 else 'FAIL'}] 2. Neural C (no16) SR >= 90% on 8x8: "
              f"{sr_c_no16_8:.1%}")

        # 3. no16 SR >= 85% on 16x16 (generalization!)
        sr_c_no16_16 = _sr(c_results["no16"][16])
        p3 = sr_c_no16_16 >= 0.85
        checks.append(p3)
        print(f"  [{'PASS' if p3 else 'FAIL'}] 3. Neural C (no16) SR >= 85% on 16x16: "
              f"{sr_c_no16_16:.1%}")

        # 4. no16 within 10pp of original on 16x16
        if "original" in c_nets:
            sr_c_orig_16 = _sr(c_results["original"][16])
            diff = sr_c_orig_16 - sr_c_no16_16
            p4 = diff <= 0.10
            checks.append(p4)
            print(f"  [{'PASS' if p4 else 'FAIL'}] 4. Neural C gap <= 10pp on 16x16: "
                  f"orig={sr_c_orig_16:.1%} no16={sr_c_no16_16:.1%} "
                  f"gap={diff:.1%}")

    # Neural B assertions
    if "no16" in b_nets:
        # 5. no16 SR >= 90% on 6x6
        sr_b_no16_6 = _sr(b_results["no16"][6])
        p5 = sr_b_no16_6 >= 0.90
        checks.append(p5)
        print(f"\n  [{'PASS' if p5 else 'FAIL'}] 5. Neural B (no16) SR >= 90% on 6x6: "
              f"{sr_b_no16_6:.1%}")

        # 6. no16 SR >= 90% on 16x16 (active mode with analytical fallback)
        sr_b_no16_16 = _sr(b_results["no16"][16])
        p6 = sr_b_no16_16 >= 0.90
        checks.append(p6)
        print(f"  [{'PASS' if p6 else 'FAIL'}] 6. Neural B (no16) SR >= 90% on 16x16: "
              f"{sr_b_no16_16:.1%}")

        # 7. no16 within 5pp of original on 16x16
        if "original" in b_nets:
            sr_b_orig_16 = _sr(b_results["original"][16])
            diff_b = sr_b_orig_16 - sr_b_no16_16
            p7 = diff_b <= 0.05
            checks.append(p7)
            print(f"  [{'PASS' if p7 else 'FAIL'}] 7. Neural B gap <= 5pp on 16x16: "
                  f"orig={sr_b_orig_16:.1%} no16={sr_b_no16_16:.1%} "
                  f"gap={diff_b:.1%}")

    # ── CSV ───────────────────────────────────────────────────────
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"generalization_eval_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["net_type", "checkpoint", "size", "seed",
                         "success", "steps", "c_compressed_ticks"])
        for label in c_nets:
            for size in sizes:
                for i, r in enumerate(c_results[label][size]):
                    writer.writerow([
                        "neural_c", label, size, 42 + i,
                        int(r["success"]), r["steps"], "",
                    ])
        for label in b_nets:
            for size in sizes:
                for i, r in enumerate(b_results[label][size]):
                    writer.writerow([
                        "neural_b", label, size, 42 + i,
                        int(r["success"]), r["steps"],
                        r.get("c_compressed_ticks", ""),
                    ])
    print(f"\n  CSV: {csv_path}")

    all_pass = all(checks) if checks else False
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({sum(checks)}/{len(checks)})")
    print("=" * 78)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

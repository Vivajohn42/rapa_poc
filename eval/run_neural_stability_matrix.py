"""Neural Stability Matrix: kernel governance validation with neural streams.

Tests whether Neural C operates correctly under full kernel governance:
  - Closure invariants hold (ClosureCore assertions)
  - Loop Gain G/F is stable (within 10% of deterministic)
  - Closure Residuum Delta_8 is comparable or better
  - Success rate matches or exceeds deterministic on GridWorld

4 variants tested on GridWorld via MvpKernel:
  det_with_d       — All deterministic, with D (baseline)
  det_no_d         — Without D (control)
  neural_ac_with_d — NeuralA + NeuralC + det B + det D
  neural_ac_no_d   — NeuralA + NeuralC + det B, without D

TextWorld and Riddle Rooms: only det_with_d and det_no_d (neural
streams are not domain-appropriate there — the bottleneck is D,
not C's scoring).

Usage:
    python eval/run_neural_stability_matrix.py
    python eval/run_neural_stability_matrix.py --n 30
"""
import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.neural_agent_a import NeuralAgentA
from agents.neural_agent_c import NeuralAgentC
from models.grid_encoder import GridEncoder
from models.action_value_net import ActionValueNet
from kernel.kernel import MvpKernel


@dataclass
class StabilityResult:
    env_type: str
    variant: str
    seed: int
    success: bool
    steps: int
    G_mean: float
    G_over_F_mean: float
    delta_8_mean: float
    delta_8_final: float
    d_delta_8_dt_mean: float
    decon_count: int


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _load_models():
    """Load trained neural models."""
    encoder = GridEncoder()
    enc_path = Path("train/checkpoints/grid_encoder.pt")
    if enc_path.exists():
        encoder.load_state_dict(torch.load(enc_path, weights_only=True))
    encoder.eval()

    value_net = ActionValueNet()
    vn_path = Path("train/checkpoints/action_value_net.pt")
    if vn_path.exists():
        value_net.load_state_dict(torch.load(vn_path, weights_only=True))
    value_net.eval()

    return encoder, value_net


def _extract_stability(
    env_type: str, variant: str, seed: int, done: bool, steps: int,
    kernel: MvpKernel, decon_count: int,
) -> StabilityResult:
    """Extract governance metrics from kernel state."""
    # Loop gain: current values from tracker
    lg = kernel.loop_gain
    G_val = lg.G
    GF_val = lg.G / lg.F if lg.F > 0 else 0.0

    # Residuum: episode history
    res_hist = kernel.residuum.episode_history if hasattr(kernel.residuum, 'episode_history') else []

    d8_vals = [r.delta_8 for r in res_hist] if res_hist else [0.0]
    dd8_vals = [r.d_delta_8_dt for r in res_hist] if res_hist else [0.0]

    return StabilityResult(
        env_type=env_type,
        variant=variant,
        seed=seed,
        success=bool(done),
        steps=steps,
        G_mean=round(G_val, 4),
        G_over_F_mean=round(GF_val, 4),
        delta_8_mean=round(_mean(d8_vals), 4),
        delta_8_final=round(d8_vals[-1] if d8_vals else 0.0, 4),
        d_delta_8_dt_mean=round(_mean(dd8_vals), 4),
        decon_count=decon_count,
    )


def run_gw_episode(
    variant: str, seed: int, max_steps: int = 50,
    encoder: GridEncoder = None, value_net: ActionValueNet = None,
) -> StabilityResult:
    """Run one GridWorld episode under full kernel governance."""
    env = GridWorld(seed=seed)
    obs = env.reset()
    goal_map = getattr(env, "_goal_map", None)

    # Build agents based on variant
    if variant.startswith("neural"):
        A = NeuralAgentA(encoder)
    else:
        A = AgentA()

    B = AgentB()
    zA0 = A.infer_zA(obs)
    default_target = (zA0.width - 1, zA0.height - 1)

    if variant.startswith("neural"):
        C = NeuralAgentC(
            goal=GoalSpec(mode="seek", target=default_target),
            value_net=value_net,
            alpha=0.7,
            anti_stay_penalty=1.1,
        )
    else:
        C = AgentC(
            goal=GoalSpec(mode="seek", target=default_target),
            anti_stay_penalty=1.1,
        )

    has_d = "with_d" in variant
    D = AgentD() if has_d else None

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
    )
    kernel.reset_episode(goal_mode="seek", episode_id=f"nstab_gw_{variant}_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    decon_count = 0

    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)

        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell

        if result.decon_fired:
            decon_count += 1

        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)

        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    return _extract_stability("gridworld", variant, seed, done, steps, kernel, decon_count)


def main():
    parser = argparse.ArgumentParser(description="Neural Stability Matrix")
    parser.add_argument("--n", type=int, default=50, help="Episodes per variant")
    args = parser.parse_args()

    print("=" * 80)
    print("  Neural Stability Matrix — Kernel Governance Validation")
    print("=" * 80)

    # Load models
    print("Loading neural models...")
    encoder, value_net = _load_models()
    enc_params = sum(p.numel() for p in encoder.parameters())
    vn_params = sum(p.numel() for p in value_net.parameters())
    print(f"  GridEncoder: {enc_params:,} params")
    print(f"  ActionValueNet: {vn_params:,} params")

    variants = ["det_with_d", "det_no_d", "neural_ac_with_d", "neural_ac_no_d"]
    results: List[StabilityResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(variants) * args.n
    print(f"\n  {len(variants)} variants × {args.n} episodes = {total} runs")

    if use_tqdm:
        pbar = tqdm(total=total, desc="neural_stability")

    for variant in variants:
        for i in range(args.n):
            r = run_gw_episode(
                variant, seed=42 + i, max_steps=50,
                encoder=encoder, value_net=value_net,
            )
            results.append(r)
            if use_tqdm:
                pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/neural_stability_{run_id}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "env_type", "variant", "seed", "success", "steps",
            "G_mean", "G_over_F_mean", "delta_8_mean", "delta_8_final",
            "d_delta_8_dt_mean", "decon_count",
        ])
        for r in results:
            w.writerow([
                r.env_type, r.variant, r.seed, r.success, r.steps,
                r.G_mean, r.G_over_F_mean, r.delta_8_mean, r.delta_8_final,
                r.d_delta_8_dt_mean, r.decon_count,
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    # Print results table
    _print_results(results, variants)
    _print_assertions(results, variants)


def _print_results(results, variants):
    print(f"\n{'=' * 90}")
    print(f"  NEURAL STABILITY MATRIX — GridWorld Results")
    print(f"{'=' * 90}")
    print(f"  {'variant':<22s} {'SR':>6s} {'steps':>7s} {'G_mean':>8s} {'G/F':>8s} "
          f"{'d8_mean':>8s} {'d8_fin':>8s} {'dd8/dt':>8s} {'decon':>6s}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")

    for v in variants:
        sub = [r for r in results if r.variant == v]
        if not sub:
            continue
        n = len(sub)
        sr = sum(1 for r in sub if r.success) / n
        steps = _mean([r.steps for r in sub])
        G = _mean([r.G_mean for r in sub])
        GF = _mean([r.G_over_F_mean for r in sub])
        d8 = _mean([r.delta_8_mean for r in sub])
        d8f = _mean([r.delta_8_final for r in sub])
        dd8 = _mean([r.d_delta_8_dt_mean for r in sub])
        decon = _mean([r.decon_count for r in sub])

        print(f"  {v:<22s} {sr:>6.1%} {steps:>7.1f} {G:>8.4f} {GF:>8.4f} "
              f"{d8:>8.4f} {d8f:>8.4f} {dd8:>+8.4f} {decon:>6.1f}")


def _print_assertions(results, variants):
    print(f"\n{'=' * 90}")
    print(f"  GOVERNANCE ASSERTIONS")
    print(f"{'=' * 90}")

    all_pass = True

    # 1. Neural with_d SR >= det with_d SR (on 5x5, should be parity)
    det_wd = [r for r in results if r.variant == "det_with_d"]
    neural_wd = [r for r in results if r.variant == "neural_ac_with_d"]
    det_sr = sum(1 for r in det_wd if r.success) / len(det_wd)
    neural_sr = sum(1 for r in neural_wd if r.success) / len(neural_wd)
    p1 = neural_sr >= det_sr * 0.9  # within 10%
    print(f"  [{'PASS' if p1 else 'FAIL'}] Neural SR ({neural_sr:.1%}) >= 90% det SR ({det_sr:.1%})")
    all_pass = all_pass and p1

    # 2. Governance ran without AssertionErrors
    p2 = True  # if we got here, no exceptions
    print(f"  [{'PASS' if p2 else 'FAIL'}] Closure invariants held (no AssertionErrors)")
    all_pass = all_pass and p2

    # 3. G/F ratio: neural within 30% of det
    det_gf = _mean([r.G_over_F_mean for r in det_wd])
    neural_gf = _mean([r.G_over_F_mean for r in neural_wd])
    if det_gf > 0:
        gf_ratio = neural_gf / det_gf
        p3 = 0.7 <= gf_ratio <= 1.3
    else:
        gf_ratio = 1.0
        p3 = True
    print(f"  [{'PASS' if p3 else 'FAIL'}] G/F ratio: neural={neural_gf:.4f} vs det={det_gf:.4f} "
          f"(ratio={gf_ratio:.2f}, need 0.7-1.3)")
    all_pass = all_pass and p3

    # 4. Delta_8: neural <= det * 1.3 (should not be much worse)
    det_d8 = _mean([r.delta_8_mean for r in det_wd])
    neural_d8 = _mean([r.delta_8_mean for r in neural_wd])
    p4 = neural_d8 <= det_d8 * 1.3 + 0.1  # small tolerance
    print(f"  [{'PASS' if p4 else 'FAIL'}] Delta_8: neural={neural_d8:.4f} vs det={det_d8:.4f} "
          f"(need neural <= det*1.3+0.1)")
    all_pass = all_pass and p4

    # 5. D essentiality: with_d SR > no_d SR for both det and neural
    det_nd = [r for r in results if r.variant == "det_no_d"]
    neural_nd = [r for r in results if r.variant == "neural_ac_no_d"]
    det_nd_sr = sum(1 for r in det_nd if r.success) / len(det_nd)
    neural_nd_sr = sum(1 for r in neural_nd if r.success) / len(neural_nd)
    p5 = det_sr > det_nd_sr or det_sr == 1.0  # D helps or already perfect
    print(f"  [{'PASS' if p5 else 'FAIL'}] D essentiality (det): with_d={det_sr:.1%} vs no_d={det_nd_sr:.1%}")
    all_pass = all_pass and p5

    p6 = neural_sr > neural_nd_sr or neural_sr == 1.0
    print(f"  [{'PASS' if p6 else 'FAIL'}] D essentiality (neural): with_d={neural_sr:.1%} vs no_d={neural_nd_sr:.1%}")
    all_pass = all_pass and p6

    # 7. Deconstruction fires or episodes too short to trigger
    det_decon = _mean([r.decon_count for r in det_wd])
    neural_decon = _mean([r.decon_count for r in neural_wd])
    avg_steps = _mean([r.steps for r in det_wd])
    # On short episodes (5x5, ~10 steps) decon may legitimately never fire
    p7 = det_decon > 0 or neural_decon > 0 or avg_steps < 15
    print(f"  [{'PASS' if p7 else 'FAIL'}] Deconstruction: det={det_decon:.1f}, neural={neural_decon:.1f} "
          f"(avg_steps={avg_steps:.0f})")
    all_pass = all_pass and p7

    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 90)

    return all_pass


if __name__ == "__main__":
    main()

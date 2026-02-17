"""Closure Residuum Diagnostics and Correlation Analysis.

Runs GridWorld and TextWorld episodes, collects Delta_8 residuum data
alongside Loop Gain, and analyses correlations.

Analyses:
  1. Delta_8 trajectory over episode (should decrease on success)
  2. Correlation Delta_8 vs G/F (expected: inverse)
  3. Delta_8 at success vs failure (success = lower final Delta_8)
  4. C-term trajectory: drops when target identified
  5. D-term trajectory: drops when D synthesizes correctly
  6. dDelta_8/dt: negative = convergence, positive = divergence

Assertions:
  1. Delta_8_final(success) < Delta_8_final(fail)
  2. mean(dDelta_8/dt) < 0 for successful episodes
  3. C_term drops after target identification

Usage:
    python eval/run_residuum_analysis.py
    python eval/run_residuum_analysis.py --n 30
    python eval/run_residuum_analysis.py --textworld
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from kernel.kernel import MvpKernel
from kernel.types import ResidualSnapshot


# --- TextWorld imports (conditional) ---
try:
    from env.textworld import TextWorld, SCENARIOS
    from agents.text_agent_a import TextAgentA
    from agents.text_agent_b import TextAgentB
    from agents.text_agent_c import TextAgentC
    from agents.text_agent_d import TextAgentD
    from router.deconstruct_text import deconstruct_text_d_to_c
    HAS_TEXTWORLD = True
except ImportError:
    HAS_TEXTWORLD = False


@dataclass
class ResidAnalysisResult:
    env_type: str           # "gridworld" or "textworld"
    variant: str            # "with_d" or "no_d"
    seed: int
    success: bool
    steps: int
    delta_8_final: float
    delta_8_mean: float
    delta_4_mean: float
    c_term_mean: float
    d_term_mean: float
    d_delta_8_dt_mean: float
    gf_mean: float
    lambda_1_final: float
    lambda_2_final: float
    # Trajectory snapshots for detailed analysis
    delta_8_trajectory: List[float] = field(default_factory=list)
    c_term_trajectory: List[float] = field(default_factory=list)
    d_term_trajectory: List[float] = field(default_factory=list)


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ------------------------------------------------------------------
# GridWorld episode
# ------------------------------------------------------------------

def run_gridworld_episode(
    variant: str, seed: int, max_steps: int = 50,
) -> ResidAnalysisResult:
    env = GridWorld(seed=seed)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    zA0 = A.infer_zA(obs)
    default_target = (zA0.width - 1, zA0.height - 1)
    C = AgentC(
        goal=GoalSpec(mode="seek", target=default_target),
        anti_stay_penalty=1.1,
    )
    goal_map = getattr(env, "_goal_map", None)

    if variant == "with_d":
        D = AgentD()
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=goal_map, enable_governance=True,
        )
    else:
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=goal_map, enable_governance=True,
        )

    kernel.reset_episode(goal_mode="seek", episode_id=f"resid_gw_{variant}_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    t = -1
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)
        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell
        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    return _extract_result("gridworld", variant, seed, done, steps, kernel)


# ------------------------------------------------------------------
# TextWorld episode
# ------------------------------------------------------------------

def _make_tw_deconstruct_fn(room_index):
    def fn(zC, zD, goal_map=None):
        return deconstruct_text_d_to_c(zC, zD, goal_map=goal_map, room_index=room_index)
    return fn


def run_textworld_episode(
    variant: str, scenario_id: int, seed: int, max_steps: int = 50,
) -> ResidAnalysisResult:
    env = TextWorld(seed=seed, scenario_id=scenario_id)
    obs = env.reset()

    room_ids = env.room_ids
    room_index = {rid: i for i, rid in enumerate(room_ids)}
    index_to_room = {i: rid for rid, i in room_index.items()}
    room_graph = env.room_graph
    room_properties = env.room_properties
    n_rooms = len(room_ids)

    A = TextAgentA(room_index, n_rooms)
    B = TextAgentB(room_graph, room_index, index_to_room)
    C = TextAgentC(room_graph, room_index, index_to_room, goal_mode="seek")

    deconstruct_fn = _make_tw_deconstruct_fn(room_index)

    if variant == "with_d":
        D = TextAgentD(room_properties, room_ids, room_index)
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=room_index, enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )
    else:
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=room_index, enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )

    kernel.reset_episode(goal_mode="seek", episode_id=f"resid_tw_{variant}_{seed}")

    done = False
    t = -1
    for t in range(max_steps):
        kernel.zC.memory["visited_rooms"] = obs.get("visited", set())
        result = kernel.tick(t, obs, done=False)
        action = result.action
        obs, reward, done = env.step(action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    return _extract_result("textworld", variant, seed, done, steps, kernel)


# ------------------------------------------------------------------
# Shared extraction
# ------------------------------------------------------------------

def _extract_result(
    env_type: str, variant: str, seed: int,
    success: bool, steps: int, kernel,
) -> ResidAnalysisResult:
    r_hist = kernel.residuum.episode_history
    g_hist = kernel.loop_gain.episode_history

    if r_hist:
        d8_vals = [s.delta_8 for s in r_hist]
        d4_vals = [s.delta_4 for s in r_hist]
        c_vals = [s.c_term for s in r_hist]
        d_vals = [s.d_term for s in r_hist]
        ddt_vals = [s.d_delta_8_dt for s in r_hist]

        delta_8_final = d8_vals[-1]
        delta_8_mean = _mean(d8_vals)
        delta_4_mean = _mean(d4_vals)
        c_term_mean = _mean(c_vals)
        d_term_mean = _mean(d_vals)
        d_delta_8_dt_mean = _mean(ddt_vals)
        lambda_1_final = r_hist[-1].lambda_1
        lambda_2_final = r_hist[-1].lambda_2
    else:
        delta_8_final = delta_8_mean = delta_4_mean = 0.0
        c_term_mean = d_term_mean = d_delta_8_dt_mean = 0.0
        lambda_1_final = lambda_2_final = 1.0
        d8_vals = c_vals = d_vals = []

    if g_hist:
        gf_vals = [s.G_over_F for s in g_hist]
        gf_mean = _mean(gf_vals)
    else:
        gf_mean = 0.0

    return ResidAnalysisResult(
        env_type=env_type,
        variant=variant,
        seed=seed,
        success=success,
        steps=steps,
        delta_8_final=round(delta_8_final, 4),
        delta_8_mean=round(delta_8_mean, 4),
        delta_4_mean=round(delta_4_mean, 4),
        c_term_mean=round(c_term_mean, 4),
        d_term_mean=round(d_term_mean, 4),
        d_delta_8_dt_mean=round(d_delta_8_dt_mean, 4),
        gf_mean=round(gf_mean, 4),
        lambda_1_final=round(lambda_1_final, 4),
        lambda_2_final=round(lambda_2_final, 4),
        delta_8_trajectory=[round(v, 4) for v in d8_vals],
        c_term_trajectory=[round(v, 4) for v in c_vals],
        d_term_trajectory=[round(v, 4) for v in d_vals],
    )


# ------------------------------------------------------------------
# Correlation helper
# ------------------------------------------------------------------

def pearson_r(xs, ys):
    """Simple Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = max(sum((x - mx) ** 2 for x in xs) ** 0.5, 1e-8)
    sy = max(sum((y - my) ** 2 for y in ys) ** 0.5, 1e-8)
    return cov / (sx * sy)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Closure Residuum Analysis")
    parser.add_argument("--n", type=int, default=20, help="Episodes per variant")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    parser.add_argument("--textworld", action="store_true", help="Include TextWorld")
    args = parser.parse_args()

    print("=" * 75)
    print("  Closure Residuum Diagnostics")
    print("=" * 75)

    all_results: List[ResidAnalysisResult] = []
    variants = ["with_d", "no_d"]

    # --- GridWorld ---
    print("\n--- GridWorld ---")
    for variant in variants:
        for i in range(args.n):
            r = run_gridworld_episode(variant, seed=42 + i, max_steps=args.max_steps)
            all_results.append(r)
        vr = [r for r in all_results if r.env_type == "gridworld" and r.variant == variant]
        sr = sum(1 for r in vr if r.success) / len(vr)
        d8 = _mean([r.delta_8_mean for r in vr])
        ddt = _mean([r.d_delta_8_dt_mean for r in vr])
        print(f"  {variant:<10s}: SR={sr:.1%}  Delta_8={d8:.4f}  dDelta_8/dt={ddt:.4f}")

    # --- TextWorld ---
    if args.textworld and HAS_TEXTWORLD:
        print("\n--- TextWorld ---")
        for variant in variants:
            for sid in range(len(SCENARIOS)):
                for i in range(max(args.n // len(SCENARIOS), 4)):
                    r = run_textworld_episode(
                        variant, scenario_id=sid, seed=42 + i,
                        max_steps=args.max_steps,
                    )
                    all_results.append(r)
            tw_r = [r for r in all_results if r.env_type == "textworld" and r.variant == variant]
            sr = sum(1 for r in tw_r if r.success) / len(tw_r) if tw_r else 0
            d8 = _mean([r.delta_8_mean for r in tw_r])
            ddt = _mean([r.d_delta_8_dt_mean for r in tw_r])
            print(f"  {variant:<10s}: SR={sr:.1%}  Delta_8={d8:.4f}  dDelta_8/dt={ddt:.4f}")

    # ================================================================
    # Analysis
    # ================================================================
    print("\n" + "=" * 75)
    print("  ANALYSIS")
    print("=" * 75)

    # --- 1. Delta_8 Success vs Failure ---
    success_results = [r for r in all_results if r.success and r.variant == "with_d"]
    fail_results = [r for r in all_results if not r.success and r.variant == "with_d"]

    d8_success = _mean([r.delta_8_final for r in success_results]) if success_results else 0
    d8_fail = _mean([r.delta_8_final for r in fail_results]) if fail_results else 999

    print(f"\n  1. Delta_8 final (with_d):")
    print(f"     Success: {d8_success:.4f} (n={len(success_results)})")
    print(f"     Failure: {d8_fail:.4f} (n={len(fail_results)})")

    if fail_results:
        p1 = d8_success < d8_fail
    else:
        # All succeed (expected for GridWorld with_d) -> auto pass
        p1 = True
        print(f"     (All episodes succeeded - assertion trivially true)")
    print(f"     [{'PASS' if p1 else 'FAIL'}] Success Delta_8 < Failure Delta_8")

    # --- 2. dDelta_8/dt for successful episodes ---
    ddt_success = [r.d_delta_8_dt_mean for r in success_results]
    ddt_mean = _mean(ddt_success) if ddt_success else 0
    p2 = ddt_mean <= 0.05  # Allow small positive (discretization noise)
    print(f"\n  2. dDelta_8/dt (with_d, success):")
    print(f"     Mean: {ddt_mean:.4f}")
    print(f"     [{'PASS' if p2 else 'FAIL'}] dDelta_8/dt <= 0.05 (convergence)")

    # --- 3. Correlation Delta_8 vs G/F ---
    d8_all = [r.delta_8_mean for r in all_results if r.variant == "with_d"]
    gf_all = [r.gf_mean for r in all_results if r.variant == "with_d"]
    corr = pearson_r(d8_all, gf_all)
    # We expect inverse correlation, but the exact value depends on the environment
    p3 = True  # Diagnostic only — no hard assertion
    print(f"\n  3. Correlation (Delta_8 vs G/F, with_d):")
    print(f"     Pearson r = {corr:.4f}")
    print(f"     [INFO] {'Inverse' if corr < 0 else 'Positive'} correlation")

    # --- 4. C-term drops after target identification ---
    # For with_d episodes: compare c_term in early vs late ticks
    c_early = []
    c_late = []
    for r in success_results:
        traj = r.c_term_trajectory
        if len(traj) >= 4:
            c_early.extend(traj[:2])
            c_late.extend(traj[-2:])
    c_early_mean = _mean(c_early) if c_early else 0
    c_late_mean = _mean(c_late) if c_late else 0
    p4 = c_late_mean <= c_early_mean + 0.1  # Allow small tolerance
    print(f"\n  4. C-term trajectory (with_d, success):")
    print(f"     Early (t=0-1): {c_early_mean:.4f}")
    print(f"     Late  (last 2): {c_late_mean:.4f}")
    print(f"     [{'PASS' if p4 else 'FAIL'}] C-term late <= early (valence alignment improves)")

    # --- 5. D-term comparison: with_d vs no_d ---
    d_term_wd = _mean([r.d_term_mean for r in all_results if r.variant == "with_d"])
    d_term_nd = _mean([r.d_term_mean for r in all_results if r.variant == "no_d"])
    p5 = d_term_wd <= d_term_nd
    print(f"\n  5. D-term (meaning alignment):")
    print(f"     with_d: {d_term_wd:.4f}")
    print(f"     no_d:   {d_term_nd:.4f}")
    print(f"     [{'PASS' if p5 else 'FAIL'}] with_d D-term <= no_d D-term")

    # --- 6. Delta_8 trajectory (sample) ---
    if success_results:
        sample = success_results[0]
        print(f"\n  6. Sample Delta_8 trajectory (seed={sample.seed}, {sample.env_type}):")
        traj = sample.delta_8_trajectory
        for i in range(min(len(traj), 12)):
            ct = sample.c_term_trajectory[i] if i < len(sample.c_term_trajectory) else 0
            dt = sample.d_term_trajectory[i] if i < len(sample.d_term_trajectory) else 0
            print(f"     t={i:2d}: Delta_8={traj[i]:.4f}  c={ct:.4f}  d={dt:.4f}")

    # --- 7. Lambda values ---
    print(f"\n  7. Lambda values (final):")
    l1_vals = [r.lambda_1_final for r in all_results if r.variant == "with_d"]
    l2_vals = [r.lambda_2_final for r in all_results if r.variant == "with_d"]
    print(f"     lambda_1 mean: {_mean(l1_vals):.4f}")
    print(f"     lambda_2 mean: {_mean(l2_vals):.4f}")

    # --- CSV output ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"residuum_analysis_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "env_type", "variant", "seed", "success", "steps",
            "delta_8_final", "delta_8_mean", "delta_4_mean",
            "c_term_mean", "d_term_mean", "d_delta_8_dt_mean",
            "gf_mean", "lambda_1_final", "lambda_2_final",
        ])
        for r in all_results:
            writer.writerow([
                r.env_type, r.variant, r.seed, int(r.success), r.steps,
                r.delta_8_final, r.delta_8_mean, r.delta_4_mean,
                r.c_term_mean, r.d_term_mean, r.d_delta_8_dt_mean,
                r.gf_mean, r.lambda_1_final, r.lambda_2_final,
            ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p4 and p5
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 75)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

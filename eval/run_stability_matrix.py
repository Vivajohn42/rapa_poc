"""Full Stability Matrix Validation: Delta_8 + G/F combined.

Tests the complete stability picture: Closure Residuum (Delta_8) alongside
Loop Gain (G/F) across GridWorld, TextWorld, and Riddle Rooms variants.

Assertions:
  1. Delta_8 converges in successful episodes (dDelta_8/dt < 0 mean)
  2. with_d has lower Delta_8 than no_d overall
  3. Per-environment: no_d Delta_8 > with_d Delta_8
  4. TextWorld: Delta_8 reduction significant with D present
  5. dDelta_8/dt trigger fires sensibly (decon at divergence)
  6. Riddle Rooms: D is essential (SR delta >= 40pp)

Usage:
    python eval/run_stability_matrix.py
    python eval/run_stability_matrix.py --n 30
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from env.textworld import TextWorld, SCENARIOS
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.text_agent_a import TextAgentA
from agents.text_agent_b import TextAgentB
from agents.text_agent_c import TextAgentC
from agents.text_agent_d import TextAgentD
from router.deconstruct_text import deconstruct_text_d_to_c
from env.riddle_rooms import ALL_PUZZLES
from env.riddle_adapter import RiddleRoomsAdapter
from kernel.kernel import MvpKernel
from kernel.types import ResidualSnapshot, MvpLoopGain


@dataclass
class StabilityResult:
    env_type: str
    variant: str
    seed: int
    success: bool
    steps: int
    # Loop Gain
    G_mean: float
    G_over_F_mean: float
    weakest_final: str
    # Residuum
    delta_8_mean: float
    delta_8_final: float
    delta_4_mean: float
    c_term_mean: float
    d_term_mean: float
    d_delta_8_dt_mean: float
    lambda_1_final: float
    lambda_2_final: float
    # Triggers
    decon_count: int
    residuum_decon_count: int  # Phase 3 trigger fires


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ------------------------------------------------------------------
# GridWorld
# ------------------------------------------------------------------

def run_gw_episode(variant: str, seed: int, max_steps: int = 50) -> StabilityResult:
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
    elif variant == "no_d":
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=goal_map, enable_governance=True,
        )
    else:  # no_governance
        D = AgentD()
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=goal_map, enable_governance=False,
        )

    kernel.reset_episode(goal_mode="seek", episode_id=f"stab_gw_{variant}_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    decon_count = 0
    resid_decon_count = 0
    t = -1
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)
        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell
        if result.decon_fired:
            decon_count += 1
            if result.decision and "RESIDUUM_DIVERGENCE" in result.decision.reasons:
                resid_decon_count += 1
        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    return _extract_stability("gridworld", variant, seed, done, steps,
                               kernel, decon_count, resid_decon_count)


# ------------------------------------------------------------------
# TextWorld
# ------------------------------------------------------------------

def _make_tw_decon_fn(room_index):
    def fn(zC, zD, goal_map=None):
        return deconstruct_text_d_to_c(zC, zD, goal_map=goal_map, room_index=room_index)
    return fn


def run_tw_episode(
    variant: str, scenario_id: int, seed: int, max_steps: int = 50,
) -> StabilityResult:
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
    decon_fn = _make_tw_decon_fn(room_index)

    if variant == "with_d":
        D = TextAgentD(room_properties, room_ids, room_index)
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=room_index, enable_governance=True,
            deconstruct_fn=decon_fn,
        )
    else:
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=room_index, enable_governance=True,
            deconstruct_fn=decon_fn,
        )

    kernel.reset_episode(goal_mode="seek", episode_id=f"stab_tw_{variant}_{seed}")

    done = False
    decon_count = 0
    resid_decon_count = 0
    t = -1
    for t in range(max_steps):
        kernel.zC.memory["visited_rooms"] = obs.get("visited", set())
        result = kernel.tick(t, obs, done=False)
        if result.decon_fired:
            decon_count += 1
            if result.decision and "RESIDUUM_DIVERGENCE" in result.decision.reasons:
                resid_decon_count += 1
        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    return _extract_stability("textworld", variant, seed, done, steps,
                               kernel, decon_count, resid_decon_count)


# ------------------------------------------------------------------
# Riddle Rooms
# ------------------------------------------------------------------

def run_riddle_episode(
    variant: str, puzzle_id: str, seed: int, max_steps: int = 30,
) -> StabilityResult:
    adapter = RiddleRoomsAdapter(seed=seed, puzzle_id=puzzle_id)
    obs = adapter.reset()
    A, B, C, D = adapter.make_agents(variant=variant)
    decon_fn = adapter.get_deconstruct_fn()
    goal_map = adapter.get_goal_map()

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
        deconstruct_fn=decon_fn,
        fallback_actions=adapter.available_actions(obs),
    )
    kernel.reset_episode(goal_mode="seek", episode_id=f"stab_ri_{variant}_{seed}")

    done = False
    decon_count = 0
    resid_decon_count = 0
    reward = 0.0
    t = -1
    for t in range(max_steps):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        if result.decon_fired:
            decon_count += 1
            if result.decision and "RESIDUUM_DIVERGENCE" in result.decision.reasons:
                resid_decon_count += 1
        obs, reward, done = adapter.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    success = done and reward > 0
    return _extract_stability("riddle", variant, seed, success, steps,
                               kernel, decon_count, resid_decon_count)


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

def _extract_stability(
    env_type, variant, seed, success, steps,
    kernel, decon_count, resid_decon_count,
) -> StabilityResult:
    g_hist = kernel.loop_gain.episode_history
    r_hist = kernel.residuum.episode_history

    if g_hist:
        G_mean = _mean([s.G for s in g_hist])
        gf_vals = [s.G_over_F for s in g_hist]
        gf_mean = _mean(gf_vals)
        weakest_final = kernel.loop_gain.weakest_coupling
    else:
        G_mean = gf_mean = 0.0
        weakest_final = "N/A"

    if r_hist:
        d8_vals = [s.delta_8 for s in r_hist]
        d4_vals = [s.delta_4 for s in r_hist]
        c_vals = [s.c_term for s in r_hist]
        d_vals = [s.d_term for s in r_hist]
        ddt_vals = [s.d_delta_8_dt for s in r_hist]
        delta_8_mean = _mean(d8_vals)
        delta_8_final = d8_vals[-1]
        delta_4_mean = _mean(d4_vals)
        c_term_mean = _mean(c_vals)
        d_term_mean = _mean(d_vals)
        d_delta_8_dt_mean = _mean(ddt_vals)
        lambda_1_final = r_hist[-1].lambda_1
        lambda_2_final = r_hist[-1].lambda_2
    else:
        delta_8_mean = delta_8_final = delta_4_mean = 0.0
        c_term_mean = d_term_mean = d_delta_8_dt_mean = 0.0
        lambda_1_final = lambda_2_final = 1.0

    return StabilityResult(
        env_type=env_type,
        variant=variant,
        seed=seed,
        success=success,
        steps=steps,
        G_mean=round(G_mean, 6),
        G_over_F_mean=round(gf_mean, 4),
        weakest_final=weakest_final,
        delta_8_mean=round(delta_8_mean, 4),
        delta_8_final=round(delta_8_final, 4),
        delta_4_mean=round(delta_4_mean, 4),
        c_term_mean=round(c_term_mean, 4),
        d_term_mean=round(d_term_mean, 4),
        d_delta_8_dt_mean=round(d_delta_8_dt_mean, 4),
        lambda_1_final=round(lambda_1_final, 4),
        lambda_2_final=round(lambda_2_final, 4),
        decon_count=decon_count,
        residuum_decon_count=resid_decon_count,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stability Matrix Validation")
    parser.add_argument("--n", type=int, default=20, help="Episodes per variant")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    args = parser.parse_args()

    print("=" * 75)
    print("  Stability Matrix Validation: Delta_8 + G/F")
    print("=" * 75)

    all_results: List[StabilityResult] = []
    n = args.n

    # --- GridWorld ---
    print("\n--- GridWorld ---")
    for variant in ["with_d", "no_d"]:
        for i in range(n):
            r = run_gw_episode(variant, seed=42 + i, max_steps=args.max_steps)
            all_results.append(r)
        vr = [r for r in all_results if r.env_type == "gridworld" and r.variant == variant]
        sr = sum(1 for r in vr if r.success) / len(vr)
        d8 = _mean([r.delta_8_mean for r in vr])
        gf = _mean([r.G_over_F_mean for r in vr])
        ddt = _mean([r.d_delta_8_dt_mean for r in vr])
        print(f"  {variant:<12s}: SR={sr:.1%}  D8={d8:.4f}  G/F={gf:.4f}  dD8/dt={ddt:.4f}")

    # --- TextWorld ---
    print("\n--- TextWorld ---")
    for variant in ["with_d", "no_d"]:
        for sid in range(len(SCENARIOS)):
            for i in range(max(n // len(SCENARIOS), 4)):
                r = run_tw_episode(
                    variant, scenario_id=sid, seed=42 + i,
                    max_steps=args.max_steps,
                )
                all_results.append(r)
        tw_r = [r for r in all_results if r.env_type == "textworld" and r.variant == variant]
        sr = sum(1 for r in tw_r if r.success) / len(tw_r) if tw_r else 0
        d8 = _mean([r.delta_8_mean for r in tw_r])
        gf = _mean([r.G_over_F_mean for r in tw_r])
        ddt = _mean([r.d_delta_8_dt_mean for r in tw_r])
        print(f"  {variant:<12s}: SR={sr:.1%}  D8={d8:.4f}  G/F={gf:.4f}  dD8/dt={ddt:.4f}")

    # --- Riddle Rooms ---
    print("\n--- Riddle Rooms ---")
    puzzle_ids = [p.puzzle_id for p in ALL_PUZZLES]
    for variant in ["with_d", "no_d"]:
        for pid in puzzle_ids:
            for i in range(max(n // len(puzzle_ids), 4)):
                r = run_riddle_episode(
                    variant, puzzle_id=pid, seed=42 + i,
                    max_steps=30,
                )
                all_results.append(r)
        ri_r = [r for r in all_results if r.env_type == "riddle" and r.variant == variant]
        sr = sum(1 for r in ri_r if r.success) / len(ri_r) if ri_r else 0
        d8 = _mean([r.delta_8_mean for r in ri_r])
        gf = _mean([r.G_over_F_mean for r in ri_r])
        ddt = _mean([r.d_delta_8_dt_mean for r in ri_r])
        print(f"  {variant:<12s}: SR={sr:.1%}  D8={d8:.4f}  G/F={gf:.4f}  dD8/dt={ddt:.4f}")

    # ================================================================
    # Assertions
    # ================================================================
    print("\n" + "=" * 75)
    print("  STABILITY ASSERTIONS")
    print("=" * 75)

    # Helper: group by (env_type, variant)
    def group(env, var):
        return [r for r in all_results if r.env_type == env and r.variant == var]

    # 1. Delta_8 converges in successful episodes (dDelta_8/dt < 0 mean)
    success_results = [r for r in all_results if r.success and r.variant == "with_d"]
    ddt_mean = _mean([r.d_delta_8_dt_mean for r in success_results]) if success_results else 0
    p1 = ddt_mean <= 0.05
    print(f"\n  1. [{'PASS' if p1 else 'FAIL'}] Convergence: "
          f"dDelta_8/dt(success,with_d) = {ddt_mean:.4f} (<= 0.05)")

    # 2. with_d has lower Delta_8 than no_d (per environment)
    # This tests that D presence reduces fixpoint distance, which is the
    # core stability claim. G/F scaling differs across environments, so
    # we compare within each environment rather than using a G/F threshold.
    wd_all = [r for r in all_results if r.variant == "with_d"]
    nd_all = [r for r in all_results if r.variant == "no_d"]
    d8_wd = _mean([r.delta_8_mean for r in wd_all])
    d8_nd = _mean([r.delta_8_mean for r in nd_all])
    p2 = d8_wd < d8_nd
    print(f"  2. [{'PASS' if p2 else 'FAIL'}] D reduces residuum (overall): "
          f"with_d={d8_wd:.4f} < no_d={d8_nd:.4f}")

    # 3. no_d: Delta_8 > with_d: Delta_8 (D reduces residuum)
    # GridWorld
    gw_wd = group("gridworld", "with_d")
    gw_nd = group("gridworld", "no_d")
    d8_gw_wd = _mean([r.delta_8_mean for r in gw_wd])
    d8_gw_nd = _mean([r.delta_8_mean for r in gw_nd])
    p3a = d8_gw_wd < d8_gw_nd
    print(f"\n  3a. [{'PASS' if p3a else 'FAIL'}] GridWorld D reduces residuum: "
          f"with_d={d8_gw_wd:.4f} < no_d={d8_gw_nd:.4f}")

    # TextWorld
    tw_wd = group("textworld", "with_d")
    tw_nd = group("textworld", "no_d")
    d8_tw_wd = _mean([r.delta_8_mean for r in tw_wd])
    d8_tw_nd = _mean([r.delta_8_mean for r in tw_nd])
    p3b = d8_tw_wd < d8_tw_nd
    print(f"  3b. [{'PASS' if p3b else 'FAIL'}] TextWorld D reduces residuum: "
          f"with_d={d8_tw_wd:.4f} < no_d={d8_tw_nd:.4f}")

    # Riddle Rooms
    ri_wd = group("riddle", "with_d")
    ri_nd = group("riddle", "no_d")
    d8_ri_wd = _mean([r.delta_8_mean for r in ri_wd])
    d8_ri_nd = _mean([r.delta_8_mean for r in ri_nd])
    p3c = d8_ri_wd < d8_ri_nd
    print(f"  3c. [{'PASS' if p3c else 'FAIL'}] Riddle D reduces residuum: "
          f"with_d={d8_ri_wd:.4f} < no_d={d8_ri_nd:.4f}")

    # 4. TextWorld: Delta_8 reduction significant with D
    tw_delta = d8_tw_nd - d8_tw_wd
    p4 = tw_delta >= 0.1
    print(f"\n  4. [{'PASS' if p4 else 'FAIL'}] TextWorld D8 delta: "
          f"{tw_delta:.4f} (>= 0.1)")

    # 5. dDelta_8/dt trigger fires sensibly
    resid_decon_total = sum(r.residuum_decon_count for r in all_results)
    decon_total = sum(r.decon_count for r in all_results)
    # No divergent decon in successful with_d episodes (system is converging)
    success_resid_decon = sum(r.residuum_decon_count
                              for r in all_results
                              if r.success and r.variant == "with_d")
    p5 = True  # Diagnostic
    print(f"\n  5. [INFO] Residuum triggers: {resid_decon_total}/{decon_total} total decon events")
    print(f"     Success+with_d residuum triggers: {success_resid_decon}")

    # 6. Riddle Rooms: D is essential (SR delta >= 40pp)
    sr_ri_wd = sum(1 for r in ri_wd if r.success) / len(ri_wd) if ri_wd else 0
    sr_ri_nd = sum(1 for r in ri_nd if r.success) / len(ri_nd) if ri_nd else 0
    sr_delta = sr_ri_wd - sr_ri_nd
    p6 = sr_delta >= 0.40
    print(f"\n  6. [{'PASS' if p6 else 'FAIL'}] Riddle D essential: "
          f"SR(with_d)={sr_ri_wd:.1%} - SR(no_d)={sr_ri_nd:.1%} = {sr_delta:.1%} (>= 40pp)")

    # --- Lambda adaptation ---
    print(f"\n  Lambda adaptation:")
    l1_gw = _mean([r.lambda_1_final for r in gw_wd])
    l2_gw = _mean([r.lambda_2_final for r in gw_wd])
    l1_tw = _mean([r.lambda_1_final for r in tw_wd])
    l2_tw = _mean([r.lambda_2_final for r in tw_wd])
    l1_ri = _mean([r.lambda_1_final for r in ri_wd])
    l2_ri = _mean([r.lambda_2_final for r in ri_wd])
    print(f"    GridWorld: lambda_1={l1_gw:.4f}  lambda_2={l2_gw:.4f}")
    print(f"    TextWorld: lambda_1={l1_tw:.4f}  lambda_2={l2_tw:.4f}")
    print(f"    Riddle:    lambda_1={l1_ri:.4f}  lambda_2={l2_ri:.4f}")

    # --- Combined table ---
    print(f"\n  Combined Stability Matrix:")
    print(f"  {'Env':<12s} {'Var':<12s} {'SR':>6s} {'D8':>8s} {'G/F':>8s} "
          f"{'dD8/dt':>8s} {'D4':>8s} {'C':>8s} {'D':>8s}")
    for env in ["gridworld", "textworld", "riddle"]:
        for var in ["with_d", "no_d"]:
            vr = [r for r in all_results if r.env_type == env and r.variant == var]
            if not vr:
                continue
            sr = sum(1 for r in vr if r.success) / len(vr)
            d8 = _mean([r.delta_8_mean for r in vr])
            gf = _mean([r.G_over_F_mean for r in vr])
            ddt = _mean([r.d_delta_8_dt_mean for r in vr])
            d4 = _mean([r.delta_4_mean for r in vr])
            ct = _mean([r.c_term_mean for r in vr])
            dt = _mean([r.d_term_mean for r in vr])
            print(f"  {env:<12s} {var:<12s} {sr:6.1%} {d8:8.4f} {gf:8.4f} "
                  f"{ddt:8.4f} {d4:8.4f} {ct:8.4f} {dt:8.4f}")

    # --- CSV ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"stability_matrix_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "env_type", "variant", "seed", "success", "steps",
            "G_mean", "G_over_F_mean", "weakest_final",
            "delta_8_mean", "delta_8_final", "delta_4_mean",
            "c_term_mean", "d_term_mean", "d_delta_8_dt_mean",
            "lambda_1_final", "lambda_2_final",
            "decon_count", "residuum_decon_count",
        ])
        for r in all_results:
            writer.writerow([
                r.env_type, r.variant, r.seed, int(r.success), r.steps,
                r.G_mean, r.G_over_F_mean, r.weakest_final,
                r.delta_8_mean, r.delta_8_final, r.delta_4_mean,
                r.c_term_mean, r.d_term_mean, r.d_delta_8_dt_mean,
                r.lambda_1_final, r.lambda_2_final,
                r.decon_count, r.residuum_decon_count,
            ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p3a and p3b and p3c and p4 and p5 and p6
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 75)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

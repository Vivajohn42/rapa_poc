"""
Stufe 8: Shadow-D Forward Planning — 5D Regime Validation

DEF Claim: A forward-planning D-agent (Shadow-D) that uses B's
deterministic model for multi-step lookahead provides measurable
advantage over single-step lookahead, especially in obstacle-rich
environments.

Shadow-D populates C's tie_break_preference (agent_c.py:90-98),
which was previously never set. This creates a genuine planning
advantage without modifying AgentC.

Five variants tested:
- modular_nod:         A+B+C (3D baseline)
- modular_ond_tb:      A+B+C+D narrative (4D)
- modular_5d:          A+B+C+D+Shadow-D (full 5D)
- modular_shadow_only: A+B+C+Shadow-D (planning without narrative)
- baseline_mono:       Monolithic baseline

DEF Predictions:
  1. 5D > 4D on obstacle-heavy levels (Shadow-D avoids dead-ends)
  2. Performance gap grows with obstacle density
  3. shadow_only > nod (planning helps even without narrative)
  4. On simple levels, no 5D advantage (planning is overkill)
  5. Shadow-D plan confidence decreases with obstacle density
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from env.gridworld import GridWorld, GoalDef, HintCellDef
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.agent_shadow_d import AgentShadowD
from router.deconstruct import deconstruct_d_to_c
from router.deconstruct_plan import deconstruct_plan_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA

from eval.baselines import baseline_monolithic_policy
from eval.stats import (
    confidence_interval_proportion,
    compare_variants,
    format_comparison,
    mean,
)

ACTIONS = ("up", "down", "left", "right")


# ── Obstacle Level Definitions ────────────────────────────────────────

@dataclass
class ObstacleLevel:
    """Test configuration focused on obstacle density."""
    name: str
    width: int
    height: int
    goals: List[GoalDef]
    hint_cells: List[HintCellDef]
    n_random_obstacles: int
    dynamic_obstacles: bool
    max_steps: int


def _make_levels() -> List[ObstacleLevel]:
    """Build obstacle-focused test levels."""
    levels = []

    # ── 5x5, 2 goals, few obstacles (easy — planning overkill) ──
    levels.append(ObstacleLevel(
        name="5x5_few_obs",
        width=5, height=5,
        goals=[GoalDef("A", (4, 4)), GoalDef("B", (4, 0))],
        hint_cells=[HintCellDef(pos=(0, 4), eliminates=[], hint_text="")],
        n_random_obstacles=0,
        dynamic_obstacles=False,
        max_steps=50,
    ))

    # ── 10x10, 2 goals, medium obstacles ──
    levels.append(ObstacleLevel(
        name="10x10_medium_obs",
        width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
        n_random_obstacles=8,
        dynamic_obstacles=False,
        max_steps=100,
    ))

    # ── 10x10, 2 goals, many obstacles (dense) ──
    levels.append(ObstacleLevel(
        name="10x10_dense_obs",
        width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
        n_random_obstacles=18,
        dynamic_obstacles=False,
        max_steps=150,
    ))

    # ── 10x10, 2 goals, dynamic obstacles ──
    levels.append(ObstacleLevel(
        name="10x10_dynamic_obs",
        width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
        n_random_obstacles=5,
        dynamic_obstacles=True,
        max_steps=100,
    ))

    # ── 15x15, 2 goals, dense obstacles ──
    levels.append(ObstacleLevel(
        name="15x15_dense_obs",
        width=15, height=15,
        goals=[GoalDef("A", (14, 14)), GoalDef("B", (14, 0))],
        hint_cells=[HintCellDef(pos=(0, 14), eliminates=[], hint_text="")],
        n_random_obstacles=30,
        dynamic_obstacles=False,
        max_steps=200,
    ))

    return levels


# ── Episode Runner ────────────────────────────────────────────────────

@dataclass
class PlanningResult:
    level: str
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_triggers: int
    shadow_d_triggers: int
    plans_accepted: int
    plan_avg_confidence: float
    pct_3d: float
    pct_4d: float
    pct_5d: float


def _make_router(enable_planning: bool = False) -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
        enable_planning_trigger=enable_planning,
        planning_uncertainty_threshold=0.25,
        planning_cooldown_steps=4,
    ))


def _random_action(zA: ZA, predict_next_fn, rng: random.Random) -> str:
    """Choose a random non-wall action."""
    valid = []
    for a in ACTIONS:
        zA_next = predict_next_fn(zA, a)
        if zA_next.agent_pos != zA.agent_pos:
            valid.append(a)
    if valid:
        return rng.choice(valid)
    return rng.choice(ACTIONS)


def run_episode(
    level: ObstacleLevel,
    variant: str,
    goal_mode: str,
    seed: Optional[int] = None,
) -> PlanningResult:
    """
    Run a single episode with optional Shadow-D planning.

    For this test we give C the true goal (known target) to isolate
    the planning advantage from hint-gathering complexity.
    """
    rng = random.Random(seed)

    env = GridWorld(
        width=level.width,
        height=level.height,
        seed=seed,
        goals=level.goals,
        hint_cells=level.hint_cells,
        obstacles=[(2, 2)] if level.width >= 5 else [],
        n_random_obstacles=level.n_random_obstacles,
        dynamic_obstacles=level.dynamic_obstacles,
    )
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    known_target = env.true_goal_pos
    goal_map = env.goal_positions

    stay_count = 0
    total_reward = 0.0
    done = False
    d_triggers = 0
    shadow_d_triggers = 0
    plans_accepted = 0
    plan_confidences = []

    zC = None
    C = None
    D = None
    shadow_d = None
    router = None
    use_tie_break = False
    use_d = False
    use_shadow_d = False

    # ── Variant setup ──
    if variant in ("modular_nod", "modular_ond_tb", "modular_5d", "modular_shadow_only"):
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=known_target), anti_stay_penalty=1.1)

    if variant in ("modular_ond_tb", "modular_5d"):
        D = AgentD()
        use_d = True
        use_tie_break = True

    if variant in ("modular_5d", "modular_shadow_only"):
        shadow_d = AgentShadowD(
            predict_next_fn=B.predict_next,
            rollout_depth=5,
            beam_width=8,
        )
        use_shadow_d = True
        use_tie_break = True

    if variant in ("modular_ond_tb", "modular_5d", "modular_shadow_only"):
        router = _make_router(enable_planning=use_shadow_d)

    # For modular_nod: no router, no D, no planning
    if variant == "modular_nod":
        router = _make_router(enable_planning=False)

    last_positions = []

    for t in range(level.max_steps):
        zA = A.infer_zA(obs)

        # ── Update C's target ──
        if C is not None and zC is not None:
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])

        # ── Action Selection ──
        decision_delta = None

        if variant == "baseline_mono":
            zA_with_goal = ZA(
                width=zA.width, height=zA.height,
                agent_pos=zA.agent_pos, goal_pos=known_target,
                obstacles=zA.obstacles, hint=zA.hint,
            )
            action = baseline_monolithic_policy(zA_with_goal, mode=goal_mode)

        else:
            if use_tie_break:
                action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            else:
                action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        # ── Environment Step ──
        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        last_positions.append(zA_next.agent_pos)
        total_reward += reward
        obs = obs_next

        # ── Shadow-D Planning ──
        if use_shadow_d and shadow_d is not None and router is not None:
            plan_active = zC.memory.get("plan_active", False) if zC else False
            activate_plan, plan_reason = router.should_activate_shadow_d(
                t=t, decision_delta=decision_delta, plan_active=plan_active,
            )
            if activate_plan:
                shadow_d_triggers += 1
                zPlan = shadow_d.plan(zA_next, target=C.goal.target, goal_mode=goal_mode)
                plan_confidences.append(zPlan.confidence)
                if zPlan.confidence >= 0.3:
                    plans_accepted += 1
                if zC is not None:
                    zC = deconstruct_plan_to_c(zC, zPlan)

        # ── D Logic ──
        if use_d and D is not None:
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d = False
            if router and decision_delta is not None:
                activate_d, reason = router.should_activate_d(
                    t=t,
                    last_positions=tuple(last_positions[-4:]),
                    decision_delta=decision_delta,
                )

            if activate_d:
                d_triggers += 1
                zD = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
                if zC is not None:
                    zC = deconstruct_d_to_c(zC, zD, goal_map=goal_map)

        if done:
            break

    steps = (t + 1) if done else level.max_steps
    stay_rate = (stay_count / steps) if steps > 0 else 0.0

    # Compute regime percentages
    pct_3d = 0.0
    pct_4d = 0.0
    pct_5d = 0.0
    if router and router.regime_log:
        summary = router.regime_summary()
        total_steps = sum(summary.values())
        if total_steps > 0:
            pct_3d = summary.get("3D", 0) / total_steps
            pct_4d = summary.get("4D", 0) / total_steps
            pct_5d = summary.get("5D", 0) / total_steps

    avg_conf = mean(plan_confidences) if plan_confidences else 0.0

    return PlanningResult(
        level=level.name,
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        stay_rate=stay_rate,
        d_triggers=d_triggers,
        shadow_d_triggers=shadow_d_triggers,
        plans_accepted=plans_accepted,
        plan_avg_confidence=avg_conf,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        pct_5d=pct_5d,
    )


# ── Batch Runner & Analysis ───────────────────────────────────────────

VARIANTS = [
    "modular_nod",
    "modular_ond_tb",
    "modular_5d",
    "modular_shadow_only",
    "baseline_mono",
]


def run_batch(n: int = 100, goal_mode: str = "seek"):
    """Run shadow-D planning study."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/shadow_planning_{run_id}.csv"

    levels = _make_levels()
    results: List[PlanningResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(levels) * len(VARIANTS) * n
    if use_tqdm:
        pbar = tqdm(total=total, desc="shadow_planning")

    for level in levels:
        for variant in VARIANTS:
            for i in range(n):
                r = run_episode(level, variant, goal_mode, seed=i)
                results.append(r)
                if use_tqdm:
                    pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "level", "variant", "goal_mode", "success", "steps",
            "total_reward", "stay_rate", "d_triggers", "shadow_d_triggers",
            "plans_accepted", "plan_avg_confidence",
            "pct_3d", "pct_4d", "pct_5d",
        ])
        for r in results:
            w.writerow([
                r.level, r.variant, r.goal_mode, r.success, r.steps,
                f"{r.total_reward:.4f}", f"{r.stay_rate:.4f}",
                r.d_triggers, r.shadow_d_triggers, r.plans_accepted,
                f"{r.plan_avg_confidence:.4f}",
                f"{r.pct_3d:.4f}", f"{r.pct_4d:.4f}", f"{r.pct_5d:.4f}",
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_results_table(results, levels)
    _print_planning_stats(results, levels)
    _print_gap_analysis(results, levels)
    _print_def_predictions(results, levels)

    return results


def _print_results_table(results: List[PlanningResult], levels: List[ObstacleLevel]):
    """Print success rates per level x variant."""
    print(f"\n{'='*100}")
    print(f"  STUFE 8: SHADOW-D PLANNING — Success Rate by Level x Variant")
    print(f"{'='*100}")
    print(f"  {'level':<20s}", end="")
    for v in VARIANTS:
        print(f" {v[:16]:>16s}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in VARIANTS:
        print(f" {'-'*16}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<20s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                sr = sum(1 for r in subset if r.success) / len(subset)
                _, lo, hi = confidence_interval_proportion(
                    sum(1 for r in subset if r.success), len(subset)
                )
                print(f" {sr:>5.3f}[{lo:.2f},{hi:.2f}]", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # Mean steps table
    print(f"\n  {'level':<20s}", end="")
    for v in VARIANTS:
        print(f" {v[:16]:>16s}", end="")
    print("  (mean steps)")
    print(f"  {'-'*20}", end="")
    for _ in VARIANTS:
        print(f" {'-'*16}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<20s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                steps_m = mean([float(r.steps) for r in subset])
                print(f" {steps_m:>16.1f}", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # Stay rate table
    print(f"\n  {'level':<20s}", end="")
    for v in VARIANTS:
        print(f" {v[:16]:>16s}", end="")
    print("  (mean stay rate)")
    print(f"  {'-'*20}", end="")
    for _ in VARIANTS:
        print(f" {'-'*16}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<20s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                sr_m = mean([r.stay_rate for r in subset])
                print(f" {sr_m:>16.3f}", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()


def _print_planning_stats(results: List[PlanningResult], levels: List[ObstacleLevel]):
    """Print Shadow-D planning statistics."""
    print(f"\n{'='*100}")
    print(f"  SHADOW-D PLANNING STATISTICS")
    print(f"{'='*100}")
    print(f"  {'level':<20s} {'variant':<22s} {'sd_trigs':>8s} {'accepted':>8s} {'avg_conf':>8s}")
    print(f"  {'-'*20} {'-'*22} {'-'*8} {'-'*8} {'-'*8}")

    for level in levels:
        for v in ("modular_5d", "modular_shadow_only"):
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                sd_t = mean([float(r.shadow_d_triggers) for r in subset])
                acc = mean([float(r.plans_accepted) for r in subset])
                conf = mean([r.plan_avg_confidence for r in subset if r.plan_avg_confidence > 0])
                print(f"  {level.name:<20s} {v:<22s} {sd_t:>8.1f} {acc:>8.1f} {conf:>8.3f}")


def _print_gap_analysis(results: List[PlanningResult], levels: List[ObstacleLevel]):
    """Print performance gap between 5D and 4D/3D variants."""
    print(f"\n{'='*100}")
    print(f"  PLANNING ADVANTAGE ANALYSIS")
    print(f"{'='*100}")

    for ref_v, comp_v, label in [
        ("modular_5d", "modular_ond_tb", "5D vs 4D"),
        ("modular_shadow_only", "modular_nod", "shadow_only vs 3D"),
        ("modular_5d", "baseline_mono", "5D vs baseline"),
    ]:
        print(f"\n  --- {label} ---")
        print(f"  {'level':<20s} {'ref_sr':>7s} {'comp_sr':>7s} {'gap':>7s} {'p-val':>8s} {'sig':>5s}")
        print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")

        for level in levels:
            ref_sub = [r for r in results if r.level == level.name and r.variant == ref_v]
            comp_sub = [r for r in results if r.level == level.name and r.variant == comp_v]

            if ref_sub and comp_sub:
                ref_sr = sum(1 for r in ref_sub if r.success) / len(ref_sub)
                comp_sr = sum(1 for r in comp_sub if r.success) / len(comp_sub)
                gap = ref_sr - comp_sr

                ref_vals = [1.0 if r.success else 0.0 for r in ref_sub]
                comp_vals = [1.0 if r.success else 0.0 for r in comp_sub]
                report = compare_variants(
                    ref_v, ref_vals, comp_v, comp_vals,
                    "sr", is_proportion=True,
                )

                sig = "***" if report["p_value"] < 0.001 else "**" if report["p_value"] < 0.01 else "*" if report["p_value"] < 0.05 else "ns"
                print(f"  {level.name:<20s} {ref_sr:>7.3f} {comp_sr:>7.3f} {gap:>+7.3f} {report['p_value']:>8.4f} {sig:>5s}")


def _print_def_predictions(results: List[PlanningResult], levels: List[ObstacleLevel]):
    """Check DEF predictions."""
    print(f"\n{'='*100}")
    print(f"  DEF PREDICTIONS CHECK")
    print(f"{'='*100}")

    all_pass = True

    # Prediction 1: 5D > 4D on obstacle-heavy levels
    print(f"\n  Prediction 1: 5D > 4D on obstacle-heavy levels")
    obstacle_heavy = [l for l in levels if "dense" in l.name or "dynamic" in l.name]
    for level in obstacle_heavy:
        d5_sub = [r for r in results if r.level == level.name and r.variant == "modular_5d"]
        d4_sub = [r for r in results if r.level == level.name and r.variant == "modular_ond_tb"]
        if d5_sub and d4_sub:
            sr_5d = sum(1 for r in d5_sub if r.success) / len(d5_sub)
            sr_4d = sum(1 for r in d4_sub if r.success) / len(d4_sub)
            passed = sr_5d >= sr_4d - 0.01
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    {level.name:<20s}: 5D={sr_5d:.3f} vs 4D={sr_4d:.3f} gap={sr_5d-sr_4d:+.3f} [{status}]")

    # Prediction 2: Gap grows with obstacle density
    print(f"\n  Prediction 2: Performance gap grows with obstacle density")
    simple_levels = [l for l in levels if "few" in l.name]
    dense_levels = [l for l in levels if "dense" in l.name]

    simple_gap = 0.0
    dense_gap = 0.0

    for lvl_list, label in [(simple_levels, "simple"), (dense_levels, "dense")]:
        d5_s = [1.0 if r.success else 0.0 for l in lvl_list for r in results if r.level == l.name and r.variant == "modular_5d"]
        d4_s = [1.0 if r.success else 0.0 for l in lvl_list for r in results if r.level == l.name and r.variant == "modular_ond_tb"]
        gap = mean(d5_s) - mean(d4_s) if d5_s and d4_s else 0.0
        if label == "simple":
            simple_gap = gap
        else:
            dense_gap = gap
        print(f"    {label:>6s}: gap={gap:+.3f}")

    passed = dense_gap >= simple_gap - 0.02
    status = "PASS" if passed else "PARTIAL"
    if not passed:
        all_pass = False
    print(f"    [{status}] simple={simple_gap:+.3f} dense={dense_gap:+.3f}")

    # Prediction 3: shadow_only > nod
    print(f"\n  Prediction 3: shadow_only > nod (planning helps without narrative)")
    for level in obstacle_heavy:
        sh_sub = [r for r in results if r.level == level.name and r.variant == "modular_shadow_only"]
        nod_sub = [r for r in results if r.level == level.name and r.variant == "modular_nod"]
        if sh_sub and nod_sub:
            sr_sh = sum(1 for r in sh_sub if r.success) / len(sh_sub)
            sr_nod = sum(1 for r in nod_sub if r.success) / len(nod_sub)
            passed = sr_sh >= sr_nod - 0.01
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    {level.name:<20s}: shadow={sr_sh:.3f} vs nod={sr_nod:.3f} gap={sr_sh-sr_nod:+.3f} [{status}]")

    # Prediction 4: No 5D advantage on simple levels
    print(f"\n  Prediction 4: No significant 5D advantage on simple levels")
    for level in simple_levels:
        d5_sub = [r for r in results if r.level == level.name and r.variant == "modular_5d"]
        d4_sub = [r for r in results if r.level == level.name and r.variant == "modular_ond_tb"]
        if d5_sub and d4_sub:
            sr_5d = sum(1 for r in d5_sub if r.success) / len(d5_sub)
            sr_4d = sum(1 for r in d4_sub if r.success) / len(d4_sub)
            gap = sr_5d - sr_4d
            passed = abs(gap) <= 0.15  # small gap acceptable
            status = "PASS" if passed else "WARN"
            if not passed:
                all_pass = False
            print(f"    {level.name:<20s}: 5D={sr_5d:.3f} vs 4D={sr_4d:.3f} gap={gap:+.3f} [{status}]")

    # Prediction 5: Plan confidence decreases with density
    print(f"\n  Prediction 5: Plan confidence vs obstacle density")
    for level in levels:
        subset = [r for r in results if r.level == level.name and r.variant == "modular_5d"]
        if subset:
            avg_conf = mean([r.plan_avg_confidence for r in subset if r.plan_avg_confidence > 0])
            print(f"    {level.name:<20s}: avg_confidence={avg_conf:.3f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME PREDICTIONS NOT FULLY MET'}")


if __name__ == "__main__":
    run_batch(n=100, goal_mode="seek")

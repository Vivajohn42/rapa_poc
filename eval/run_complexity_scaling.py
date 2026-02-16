"""
Stufe 3: Complexity Scaling — DEF Regime Justification

DEF Claim: Higher-dimensional regimes (3D, 4D) justify their cost at higher
task complexity. The performance gap between full modular (A+B+C+D) and
simpler variants GROWS with complexity.

Complexity axes:
  - Grid size: 5x5, 10x10, 15x15
  - Number of goals: 2, 3, 4
  - Obstacles: few, many, dynamic

This script runs the same variant set across complexity levels and measures
success rate, steps, and reward. The key metric is the PERFORMANCE GAP
between modular_ond_tb (4D) and baseline_mono / ab_only.
"""

import sys
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld, GoalDef, HintCellDef
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA

from eval.baselines import baseline_monolithic_policy
from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    compare_variants,
    format_comparison,
    mean,
)

ACTIONS = ("up", "down", "left", "right")


# ── Complexity Level Definitions ──────────────────────────────────────

@dataclass
class ComplexityLevel:
    """Defines one complexity configuration."""
    name: str
    width: int
    height: int
    goals: List[GoalDef]
    hint_cells: List[HintCellDef]
    n_random_obstacles: int
    dynamic_obstacles: bool
    max_steps: int


def _make_levels() -> List[ComplexityLevel]:
    """Build the matrix of complexity levels."""
    levels = []

    # ── 5x5, 2 goals (baseline, same as original) ──
    levels.append(ComplexityLevel(
        name="5x5_2g",
        width=5, height=5,
        goals=[GoalDef("A", (4, 4)), GoalDef("B", (4, 0))],
        hint_cells=[HintCellDef(pos=(0, 4), eliminates=[], hint_text="")],
        n_random_obstacles=0,
        dynamic_obstacles=False,
        max_steps=50,
    ))

    # ── 10x10, 2 goals, more obstacles ──
    levels.append(ComplexityLevel(
        name="10x10_2g",
        width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
        n_random_obstacles=8,
        dynamic_obstacles=False,
        max_steps=100,
    ))

    # ── 10x10, 4 goals, 2 hints needed ──
    levels.append(ComplexityLevel(
        name="10x10_4g",
        width=10, height=10,
        goals=[
            GoalDef("A", (9, 9)), GoalDef("B", (9, 0)),
            GoalDef("C", (0, 9)), GoalDef("D", (5, 5)),
        ],
        hint_cells=[
            HintCellDef(pos=(0, 5), group_a=["A", "B"], group_b=["C", "D"]),
            HintCellDef(pos=(5, 0), group_a=["A", "C"], group_b=["B", "D"]),
        ],
        n_random_obstacles=8,
        dynamic_obstacles=False,
        max_steps=150,
    ))

    # ── 15x15, 2 goals, many obstacles ──
    levels.append(ComplexityLevel(
        name="15x15_2g",
        width=15, height=15,
        goals=[GoalDef("A", (14, 14)), GoalDef("B", (14, 0))],
        hint_cells=[HintCellDef(pos=(0, 14), eliminates=[], hint_text="")],
        n_random_obstacles=20,
        dynamic_obstacles=False,
        max_steps=200,
    ))

    # ── 15x15, 4 goals, 2 hints, many obstacles ──
    levels.append(ComplexityLevel(
        name="15x15_4g",
        width=15, height=15,
        goals=[
            GoalDef("A", (14, 14)), GoalDef("B", (14, 0)),
            GoalDef("C", (0, 14)), GoalDef("D", (7, 7)),
        ],
        hint_cells=[
            HintCellDef(pos=(0, 7), group_a=["A", "B"], group_b=["C", "D"]),
            HintCellDef(pos=(7, 0), group_a=["A", "C"], group_b=["B", "D"]),
        ],
        n_random_obstacles=20,
        dynamic_obstacles=False,
        max_steps=200,
    ))

    # ── 10x10, 2 goals, dynamic obstacles ──
    levels.append(ComplexityLevel(
        name="10x10_2g_dyn",
        width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
        n_random_obstacles=5,
        dynamic_obstacles=True,
        max_steps=100,
    ))

    return levels


# ── Episode Runner ────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    level: str
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_triggers: int
    hints_collected: int


def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8
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
    level: ComplexityLevel,
    variant: str,
    goal_mode: str,
    seed: Optional[int] = None,
) -> EpisodeResult:
    """
    Run a single episode at a given complexity level.

    For the complexity scaling test we use the TRUE goal (known target)
    for C-based variants, same as stream_isolation. This isolates the
    question of whether streams help at higher complexity.

    For multi-goal levels: we still give C the true target directly
    (the hidden-goal hint-gathering question is tested separately in
    the multi-goal variant below).
    """
    rng = random.Random(seed)

    # Build environment from complexity level
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
    hints_collected = 0

    zC = None
    C = None
    D = None
    router = None
    use_tie_break = False

    if variant in ("modular_nod", "modular_ond", "modular_ond_tb"):
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=known_target), anti_stay_penalty=1.1)
        use_tie_break = (variant == "modular_ond_tb")

    if variant in ("modular_ond", "modular_ond_tb"):
        D = AgentD()
        router = _make_router()

    for t in range(level.max_steps):
        zA = A.infer_zA(obs)

        # ── Action Selection ──────────────────────────────────────
        decision_delta = None

        if variant == "baseline_mono":
            zA_with_goal = ZA(
                width=zA.width, height=zA.height,
                agent_pos=zA.agent_pos, goal_pos=known_target,
                obstacles=zA.obstacles, hint=zA.hint,
            )
            action = baseline_monolithic_policy(zA_with_goal, mode=goal_mode)

        elif variant == "ab_only":
            action = _random_action(zA, B.predict_next, rng)

        else:
            # modular variants with C
            if zC and "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])

            if use_tie_break:
                action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            else:
                action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)

            decision_delta = scored[0][1] - scored[1][1]

        # ── Environment Step ──────────────────────────────────────
        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        # Track hint collection
        if zA_next.hint is not None:
            hints_collected += 1

        total_reward += reward
        obs = obs_next

        # ── D Logic ───────────────────────────────────────────────
        if D is not None:
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d = False
            if router and decision_delta is not None:
                activate_d, reason = router.should_activate_d(
                    t=t,
                    last_positions=(zA_next.agent_pos,),
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

    return EpisodeResult(
        level=level.name,
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        stay_rate=stay_rate,
        d_triggers=d_triggers,
        hints_collected=hints_collected,
    )


# ── Multi-Goal Episode (hidden goal, must collect hints) ──────────────

def run_episode_multi_goal(
    level: ComplexityLevel,
    variant: str,
    goal_mode: str,
    seed: Optional[int] = None,
) -> EpisodeResult:
    """
    Multi-goal variant: goal is HIDDEN and agent must collect hints to
    narrow down candidates. This tests D's narrative integration capacity.

    Strategy for C-based variants:
    - Initially navigate toward first hint cell
    - After collecting hints, deconstruct updates target if narrowed to 1
    - If target still unknown, navigate to next hint cell
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

    goal_map = env.goal_positions
    # Default target: center of all goal positions
    default_target = (level.width - 1, level.height - 1)

    stay_count = 0
    total_reward = 0.0
    done = False
    d_triggers = 0
    hints_collected = 0

    zC = None
    C = None
    D = None
    router = None
    use_tie_break = False
    hint_cell_queue = [h.pos for h in level.hint_cells]

    if variant in ("modular_nod", "modular_ond", "modular_ond_tb"):
        # Start by navigating to first hint cell
        initial_target = hint_cell_queue[0] if hint_cell_queue else default_target
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=initial_target), anti_stay_penalty=1.1)
        use_tie_break = (variant == "modular_ond_tb")

    if variant in ("modular_ond", "modular_ond_tb"):
        D = AgentD()
        router = _make_router()

    hint_idx = 0  # which hint cell to navigate to next

    for t in range(level.max_steps):
        zA = A.infer_zA(obs)

        # ── Update C's target based on current knowledge ──────────
        if C is not None and zC is not None:
            if "target" in zC.memory and zC.memory["target"] is not None:
                # We know the goal — navigate to it
                C.goal.target = tuple(zC.memory["target"])
            elif hint_idx < len(hint_cell_queue):
                # Navigate to next hint cell
                C.goal.target = hint_cell_queue[hint_idx]
                # If we're at the hint cell, advance to next
                if zA.agent_pos == hint_cell_queue[hint_idx]:
                    hint_idx += 1
                    if hint_idx < len(hint_cell_queue):
                        C.goal.target = hint_cell_queue[hint_idx]
                    else:
                        C.goal.target = default_target
            else:
                C.goal.target = default_target

        # ── Action Selection ──────────────────────────────────────
        decision_delta = None

        if variant == "baseline_mono":
            # Baseline doesn't know the goal — navigate to default
            target = default_target
            if zC and "target" in zC.memory and zC.memory["target"] is not None:
                target = tuple(zC.memory["target"])
            zA_with_goal = ZA(
                width=zA.width, height=zA.height,
                agent_pos=zA.agent_pos, goal_pos=target,
                obstacles=zA.obstacles, hint=zA.hint,
            )
            action = baseline_monolithic_policy(zA_with_goal, mode=goal_mode)

        elif variant == "ab_only":
            action = _random_action(zA, B.predict_next, rng)

        else:
            if use_tie_break:
                action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            else:
                action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        # ── Environment Step ──────────────────────────────────────
        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        # ── Hint Processing ───────────────────────────────────────
        if zA_next.hint is not None:
            hints_collected += 1
            # For D-based variants, D processes the hint
            if D is not None:
                D.observe_step(t=t, zA=zA_next, action="hint", reward=0.0, done=False)
                zD_hint = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)
                if zC is not None:
                    zC = deconstruct_d_to_c(zC, zD_hint, goal_map=goal_map)
                d_triggers += 1
            elif zC is not None:
                # For modular_nod: process hint directly via a minimal ZD
                from state.schema import ZD
                fake_tags = [f"hint:{zA_next.hint}"] if len(zA_next.hint) == 1 else [zA_next.hint]
                zD_hint = ZD(narrative="hint", meaning_tags=fake_tags, length_chars=4, grounding_violations=0)
                zC = deconstruct_d_to_c(zC, zD_hint, goal_map=goal_map)

        total_reward += reward
        obs = obs_next

        # ── D Logic (router-gated) ────────────────────────────────
        if D is not None:
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d = False
            if router and decision_delta is not None:
                activate_d, reason = router.should_activate_d(
                    t=t, last_positions=(zA_next.agent_pos,), decision_delta=decision_delta,
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

    return EpisodeResult(
        level=level.name,
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        stay_rate=stay_rate,
        d_triggers=d_triggers,
        hints_collected=hints_collected,
    )


# ── Batch Runner & Analysis ───────────────────────────────────────────

VARIANTS = [
    "baseline_mono",
    "modular_nod",
    "modular_ond_tb",
    "ab_only",
]


def run_batch(n: int = 100, goal_mode: str = "seek"):
    """Run complexity scaling study."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/complexity_scaling_{run_id}.csv"

    levels = _make_levels()
    results: List[EpisodeResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(levels) * len(VARIANTS) * n
    if use_tqdm:
        pbar = tqdm(total=total, desc="complexity_scaling")

    for level in levels:
        for variant in VARIANTS:
            for i in range(n):
                # Multi-goal levels use hidden-goal episode runner
                n_goals = len(level.goals)
                if n_goals > 2:
                    r = run_episode_multi_goal(level, variant, goal_mode, seed=i)
                else:
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
            "total_reward", "stay_rate", "d_triggers", "hints_collected",
        ])
        for r in results:
            w.writerow([
                r.level, r.variant, r.goal_mode, r.success, r.steps,
                f"{r.total_reward:.4f}", f"{r.stay_rate:.4f}",
                r.d_triggers, r.hints_collected,
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_scaling_table(results, levels)
    _print_gap_analysis(results, levels)

    return results


def _print_scaling_table(results: List[EpisodeResult], levels: List[ComplexityLevel]):
    """Print success rates per level x variant."""
    print(f"\n{'='*90}")
    print(f"  COMPLEXITY SCALING — Success Rate by Level x Variant")
    print(f"{'='*90}")
    print(f"  {'level':<18s}", end="")
    for v in VARIANTS:
        print(f" {v:>16s}", end="")
    print()
    print(f"  {'-'*18}", end="")
    for _ in VARIANTS:
        print(f" {'-'*16}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<18s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                sr = sum(1 for r in subset if r.success) / len(subset)
                n = len(subset)
                _, lo, hi = confidence_interval_proportion(
                    sum(1 for r in subset if r.success), n
                )
                print(f" {sr:>5.3f}[{lo:.2f},{hi:.2f}]", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # Steps table
    print(f"\n  {'level':<18s}", end="")
    for v in VARIANTS:
        print(f" {v:>16s}", end="")
    print("  (mean steps)")
    print(f"  {'-'*18}", end="")
    for _ in VARIANTS:
        print(f" {'-'*16}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<18s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                steps_m = mean([float(r.steps) for r in subset])
                print(f" {steps_m:>16.1f}", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()


def _print_gap_analysis(results: List[EpisodeResult], levels: List[ComplexityLevel]):
    """
    Print the performance gap analysis: does the gap between modular and
    simpler variants grow with complexity?
    """
    print(f"\n{'='*90}")
    print(f"  PERFORMANCE GAP ANALYSIS (DEF prediction: gap grows with complexity)")
    print(f"{'='*90}")

    ref_variant = "modular_ond_tb"  # 4D full system

    for compare_v in ["baseline_mono", "ab_only"]:
        print(f"\n  --- {ref_variant} vs {compare_v} ---")
        print(f"  {'level':<18s} {'4D_sr':>7s} {compare_v[:8]+'_sr':>10s} {'gap':>7s} {'direction':>10s}")
        print(f"  {'-'*18} {'-'*7} {'-'*10} {'-'*7} {'-'*10}")

        gaps = []
        for level in levels:
            ref_sub = [r for r in results if r.level == level.name and r.variant == ref_variant]
            comp_sub = [r for r in results if r.level == level.name and r.variant == compare_v]

            if ref_sub and comp_sub:
                ref_sr = sum(1 for r in ref_sub if r.success) / len(ref_sub)
                comp_sr = sum(1 for r in comp_sub if r.success) / len(comp_sub)
                gap = ref_sr - comp_sr
                gaps.append((level.name, gap))

                direction = "4D wins" if gap > 0.01 else ("tie" if abs(gap) <= 0.01 else "4D loses")
                print(f"  {level.name:<18s} {ref_sr:>7.3f} {comp_sr:>10.3f} {gap:>+7.3f} {direction:>10s}")

        # Check if gap is growing
        if len(gaps) >= 2:
            first_gap = gaps[0][1]
            last_gap = gaps[-1][1]
            if last_gap > first_gap + 0.01:
                print(f"  [PASS] Gap grows from {first_gap:+.3f} to {last_gap:+.3f} with complexity")
            elif abs(last_gap - first_gap) <= 0.01:
                print(f"  [PARTIAL] Gap stable ({first_gap:+.3f} -> {last_gap:+.3f})")
            else:
                print(f"  [WARN] Gap shrinks ({first_gap:+.3f} -> {last_gap:+.3f})")

    # Statistical comparison for the hardest level
    hardest = levels[-2]  # 15x15_4g (hardest non-dynamic)
    print(f"\n  --- Statistical comparison at hardest level ({hardest.name}) ---")
    for compare_v in ["baseline_mono", "ab_only"]:
        ref_sub = [r for r in results if r.level == hardest.name and r.variant == ref_variant]
        comp_sub = [r for r in results if r.level == hardest.name and r.variant == compare_v]
        if ref_sub and comp_sub:
            ref_vals = [1.0 if r.success else 0.0 for r in ref_sub]
            comp_vals = [1.0 if r.success else 0.0 for r in comp_sub]
            report = compare_variants(
                ref_variant, ref_vals, compare_v, comp_vals,
                "success_rate", is_proportion=True,
            )
            print(f"  {format_comparison(report)}")


if __name__ == "__main__":
    run_batch(n=100, goal_mode="seek")

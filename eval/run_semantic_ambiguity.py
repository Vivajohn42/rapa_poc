"""
Stufe 6: Semantic Ambiguity -- D Makes the Difference

DEF Claim: When the environment provides hints that require semantic
interpretation, Agent D with interpretation capability outperforms
no-D variants that cannot decode coded hints.

This test introduces coded/directional hints that are opaque to
deconstruct_d_to_c's pattern matching. Only AgentDInterpreter can
translate coded clues into actionable goal identifiers.

Three difficulty levels:
  - easy:   Absolute directional clues ("goal_at_bottom_right")
  - medium: Relative/comparative clues ("goal_furthest_from_origin")
  - hard:   Abstract property clues ("coords_sum_high")

DEF Predictions:
  1. D-interpreter > no-D on 2-goal configurations (strong signal)
  2. Performance gap grows with difficulty (easy < medium < hard)
  3. D's interpretation rate is 100% while no-D is 0% (information gap)
  4. On 4-goal, D advantage may be reduced by hint-gathering overhead
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
from env.coded_hints import CodedGridWorld, HintEncoder
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.agent_d_interpreter import AgentDInterpreter
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA, ZD

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
class AmbiguityLevel:
    """Defines one test configuration for semantic ambiguity."""
    name: str
    difficulty: str         # "easy", "medium", "hard"
    width: int
    height: int
    goals: List[GoalDef]
    hint_cells: List[HintCellDef]
    n_random_obstacles: int
    max_steps: int


def _make_levels() -> List[AmbiguityLevel]:
    """Build the matrix of test levels."""
    levels = []

    for difficulty in ("easy", "medium", "hard"):
        # ── 5x5, 2 goals ──
        levels.append(AmbiguityLevel(
            name=f"5x5_2g_{difficulty}",
            difficulty=difficulty,
            width=5, height=5,
            goals=[GoalDef("A", (4, 4)), GoalDef("B", (4, 0))],
            hint_cells=[HintCellDef(pos=(0, 4), eliminates=[], hint_text="")],
            n_random_obstacles=0,
            max_steps=50,
        ))

        # ── 10x10, 2 goals ──
        levels.append(AmbiguityLevel(
            name=f"10x10_2g_{difficulty}",
            difficulty=difficulty,
            width=10, height=10,
            goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
            hint_cells=[HintCellDef(pos=(0, 9), eliminates=[], hint_text="")],
            n_random_obstacles=5,
            max_steps=100,
        ))

        # ── 10x10, 4 goals ──
        levels.append(AmbiguityLevel(
            name=f"10x10_4g_{difficulty}",
            difficulty=difficulty,
            width=10, height=10,
            goals=[
                GoalDef("A", (9, 9)), GoalDef("B", (9, 0)),
                GoalDef("C", (0, 9)), GoalDef("D", (5, 5)),
            ],
            hint_cells=[
                HintCellDef(pos=(0, 5), group_a=["A", "B"], group_b=["C", "D"]),
                HintCellDef(pos=(5, 0), group_a=["A", "C"], group_b=["B", "D"]),
            ],
            n_random_obstacles=5,
            max_steps=150,
        ))

    return levels


# ── Episode Runner ────────────────────────────────────────────────────

@dataclass
class AmbiguityResult:
    level: str
    difficulty: str
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_triggers: int
    hints_collected: int
    hints_interpreted: int   # D successfully interpreted coded hints
    target_learned: bool     # Agent learned the true target


def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
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
    level: AmbiguityLevel,
    variant: str,
    goal_mode: str,
    seed: Optional[int] = None,
) -> AmbiguityResult:
    """
    Run a single episode with coded hints.

    The agent must collect hints, but hints are coded (not direct goal IDs).
    Only D-interpreter variants can decode them.
    """
    rng = random.Random(seed)

    # Build base environment
    base_env = GridWorld(
        width=level.width,
        height=level.height,
        seed=seed,
        goals=level.goals,
        hint_cells=level.hint_cells,
        obstacles=[(2, 2)] if level.width >= 5 else [],
        n_random_obstacles=level.n_random_obstacles,
    )

    # Wrap with coded hints
    env = CodedGridWorld(base_env, difficulty=level.difficulty)
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    goal_map = base_env.goal_positions

    # Default target: first hint cell (navigate to gather info)
    hint_cell_queue = [h.pos for h in level.hint_cells]
    default_target = (level.width - 1, level.height - 1)

    stay_count = 0
    total_reward = 0.0
    done = False
    d_triggers = 0
    hints_collected = 0
    hints_interpreted = 0
    target_learned = False

    zC = None
    C = None
    D = None
    router = None
    use_d = False
    use_tie_break = False

    # ── Variant setup ──
    if variant in ("modular_nod", "modular_ond_interp", "modular_ond_tb_interp"):
        initial_target = hint_cell_queue[0] if hint_cell_queue else default_target
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=initial_target), anti_stay_penalty=1.1)
        use_tie_break = (variant == "modular_ond_tb_interp")

    if variant in ("modular_ond_interp", "modular_ond_tb_interp"):
        D = AgentDInterpreter(
            goal_map=goal_map,
            grid_width=level.width,
            grid_height=level.height,
            difficulty=level.difficulty,
        )
        router = _make_router()
        use_d = True

    hint_idx = 0

    for t in range(level.max_steps):
        zA = A.infer_zA(obs)

        # ── Update C's target based on current knowledge ──
        if C is not None and zC is not None:
            if "target" in zC.memory and zC.memory["target"] is not None:
                C.goal.target = tuple(zC.memory["target"])
                target_learned = True
            elif hint_idx < len(hint_cell_queue):
                C.goal.target = hint_cell_queue[hint_idx]
                if zA.agent_pos == hint_cell_queue[hint_idx]:
                    hint_idx += 1
                    if hint_idx < len(hint_cell_queue):
                        C.goal.target = hint_cell_queue[hint_idx]
                    else:
                        C.goal.target = default_target
            else:
                C.goal.target = default_target

        # ── Action Selection ──
        decision_delta = None

        if variant == "baseline_mono":
            target = default_target
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

        # ── Environment Step ──
        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        # ── Hint Processing ──
        if zA_next.hint is not None:
            hints_collected += 1

            if use_d and D is not None:
                # D-interpreter processes the coded hint
                D.observe_step(t=t, zA=zA_next, action="hint", reward=0.0, done=False)
                zD_hint = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)

                # Check if D successfully interpreted the hint
                for tag in zD_hint.meaning_tags:
                    tag_lower = tag.lower()
                    if (tag_lower.startswith("hint:") and
                            tag_lower[5:] in [g.lower() for g in goal_map]):
                        hints_interpreted += 1
                        break
                    if tag_lower.startswith("not_"):
                        hints_interpreted += 1
                        break

                if zC is not None:
                    zC = deconstruct_d_to_c(zC, zD_hint, goal_map=goal_map)
                d_triggers += 1

            elif zC is not None:
                # No-D variant: pass the coded hint directly
                # deconstruct_d_to_c will NOT match coded strings like
                # "goal_at_bottom_right" since they don't match "hint:A"
                fake_tags = [f"hint:{zA_next.hint}"]
                zD_hint = ZD(narrative="hint", meaning_tags=fake_tags,
                             length_chars=4, grounding_violations=0)
                zC = deconstruct_d_to_c(zC, zD_hint, goal_map=goal_map)

        total_reward += reward
        obs = obs_next

        # ── D Logic (router-gated) ──
        if use_d and D is not None:
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d = False
            if router and decision_delta is not None:
                activate_d, reason = router.should_activate_d(
                    t=t, last_positions=(zA_next.agent_pos,),
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

    return AmbiguityResult(
        level=level.name,
        difficulty=level.difficulty,
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        stay_rate=stay_rate,
        d_triggers=d_triggers,
        hints_collected=hints_collected,
        hints_interpreted=hints_interpreted,
        target_learned=target_learned,
    )


# ── Batch Runner & Analysis ───────────────────────────────────────────

VARIANTS = [
    "modular_nod",
    "modular_ond_interp",
    "modular_ond_tb_interp",
    "baseline_mono",
    "ab_only",
]


def run_batch(n: int = 100, goal_mode: str = "seek"):
    """Run semantic ambiguity study."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/semantic_ambiguity_{run_id}.csv"

    levels = _make_levels()
    results: List[AmbiguityResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(levels) * len(VARIANTS) * n
    if use_tqdm:
        pbar = tqdm(total=total, desc="semantic_ambiguity")

    for level in levels:
        # Verify hint encoding is unique at this difficulty
        _check_encoding(level)

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
            "level", "difficulty", "variant", "goal_mode", "success", "steps",
            "total_reward", "stay_rate", "d_triggers", "hints_collected",
            "hints_interpreted", "target_learned",
        ])
        for r in results:
            w.writerow([
                r.level, r.difficulty, r.variant, r.goal_mode, r.success,
                r.steps, f"{r.total_reward:.4f}", f"{r.stay_rate:.4f}",
                r.d_triggers, r.hints_collected, r.hints_interpreted,
                r.target_learned,
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_results_table(results, levels)
    _print_d_advantage_analysis(results, levels)
    _print_interpretation_stats(results, levels)
    _print_def_predictions(results, levels)

    return results


def _check_encoding(level: AmbiguityLevel):
    """Verify coded hints are unique for this level."""
    goal_map = {g.goal_id: g.pos for g in level.goals}
    encoder = HintEncoder(goal_map, level.width, level.height)
    if not encoder.can_uniquely_identify(level.difficulty):
        print(f"  WARNING: Encoding not unique for {level.name} at {level.difficulty}")


def _print_results_table(results: List[AmbiguityResult], levels: List[AmbiguityLevel]):
    """Print success rates per level x variant."""
    print(f"\n{'='*100}")
    print(f"  STUFE 6: SEMANTIC AMBIGUITY — Success Rate by Level x Variant")
    print(f"{'='*100}")

    for difficulty in ("easy", "medium", "hard"):
        diff_levels = [l for l in levels if l.difficulty == difficulty]
        print(f"\n  --- Difficulty: {difficulty.upper()} ---")
        print(f"  {'level':<20s}", end="")
        for v in VARIANTS:
            print(f" {v[:16]:>16s}", end="")
        print()
        print(f"  {'-'*20}", end="")
        for _ in VARIANTS:
            print(f" {'-'*16}", end="")
        print()

        for level in diff_levels:
            # Strip difficulty suffix for display
            display_name = level.name.rsplit("_", 1)[0]
            print(f"  {display_name:<20s}", end="")
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


def _print_d_advantage_analysis(results: List[AmbiguityResult], levels: List[AmbiguityLevel]):
    """Analyze the advantage of D-interpreter over no-D variants."""
    print(f"\n{'='*100}")
    print(f"  D-INTERPRETER ADVANTAGE ANALYSIS")
    print(f"  (DEF prediction: D-interp > no-D, gap grows with difficulty)")
    print(f"{'='*100}")

    ref_variant = "modular_ond_interp"

    for compare_v in ["modular_nod", "baseline_mono"]:
        print(f"\n  --- {ref_variant} vs {compare_v} ---")
        print(f"  {'level':<25s} {'D_sr':>7s} {'noD_sr':>7s} {'gap':>7s} {'p-val':>8s} {'sig':>5s}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")

        gaps_by_difficulty = defaultdict(list)

        for level in levels:
            ref_sub = [r for r in results if r.level == level.name and r.variant == ref_variant]
            comp_sub = [r for r in results if r.level == level.name and r.variant == compare_v]

            if ref_sub and comp_sub:
                ref_sr = sum(1 for r in ref_sub if r.success) / len(ref_sub)
                comp_sr = sum(1 for r in comp_sub if r.success) / len(comp_sub)
                gap = ref_sr - comp_sr

                # Statistical test
                ref_vals = [1.0 if r.success else 0.0 for r in ref_sub]
                comp_vals = [1.0 if r.success else 0.0 for r in comp_sub]
                report = compare_variants(
                    ref_variant, ref_vals, compare_v, comp_vals,
                    "sr", is_proportion=True,
                )

                sig = "***" if report["p_value"] < 0.001 else "**" if report["p_value"] < 0.01 else "*" if report["p_value"] < 0.05 else "ns"
                print(f"  {level.name:<25s} {ref_sr:>7.3f} {comp_sr:>7.3f} {gap:>+7.3f} {report['p_value']:>8.4f} {sig:>5s}")

                gaps_by_difficulty[level.difficulty].append(gap)

        # Check if gap grows with difficulty
        print()
        for diff in ("easy", "medium", "hard"):
            if diff in gaps_by_difficulty:
                avg_gap = mean(gaps_by_difficulty[diff])
                print(f"  Avg gap at {diff:>6s}: {avg_gap:+.3f}")

        easy_gaps = gaps_by_difficulty.get("easy", [0])
        hard_gaps = gaps_by_difficulty.get("hard", [0])
        if mean(hard_gaps) > mean(easy_gaps) + 0.01:
            print(f"  [PASS] Gap grows with difficulty ({mean(easy_gaps):+.3f} -> {mean(hard_gaps):+.3f})")
        elif abs(mean(hard_gaps) - mean(easy_gaps)) <= 0.01:
            print(f"  [PARTIAL] Gap stable ({mean(easy_gaps):+.3f} -> {mean(hard_gaps):+.3f})")
        else:
            print(f"  [WARN] Gap does not grow ({mean(easy_gaps):+.3f} -> {mean(hard_gaps):+.3f})")


def _print_interpretation_stats(results: List[AmbiguityResult], levels: List[AmbiguityLevel]):
    """Print hint interpretation statistics."""
    print(f"\n{'='*100}")
    print(f"  HINT INTERPRETATION STATISTICS")
    print(f"{'='*100}")
    print(f"  {'level':<25s} {'variant':<22s} {'hints':>6s} {'interp':>7s} {'rate':>6s} {'target':>7s}")
    print(f"  {'-'*25} {'-'*22} {'-'*6} {'-'*7} {'-'*6} {'-'*7}")

    for level in levels:
        for v in VARIANTS:
            subset = [r for r in results if r.level == level.name and r.variant == v]
            if subset:
                hints = sum(r.hints_collected for r in subset)
                interp = sum(r.hints_interpreted for r in subset)
                rate = interp / hints if hints > 0 else 0.0
                target = sum(1 for r in subset if r.target_learned) / len(subset)
                print(f"  {level.name:<25s} {v:<22s} {hints:>6d} {interp:>7d} {rate:>6.1%} {target:>7.1%}")


def _print_def_predictions(results: List[AmbiguityResult], levels: List[AmbiguityLevel]):
    """Check DEF predictions."""
    print(f"\n{'='*100}")
    print(f"  DEF PREDICTIONS CHECK")
    print(f"{'='*100}")

    all_pass = True

    # Prediction 1: D-interp > no-D at all difficulty levels
    print(f"\n  Prediction 1: D-interpreter > no-D at all difficulty levels")
    for difficulty in ("easy", "medium", "hard"):
        diff_levels = [l for l in levels if l.difficulty == difficulty]
        d_success = []
        nod_success = []
        for level in diff_levels:
            d_sub = [r for r in results if r.level == level.name and r.variant == "modular_ond_interp"]
            nod_sub = [r for r in results if r.level == level.name and r.variant == "modular_nod"]
            d_success.extend([1.0 if r.success else 0.0 for r in d_sub])
            nod_success.extend([1.0 if r.success else 0.0 for r in nod_sub])

        d_sr = mean(d_success)
        nod_sr = mean(nod_success)
        passed = d_sr > nod_sr
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    {difficulty:>6s}: D={d_sr:.3f} vs noD={nod_sr:.3f} gap={d_sr-nod_sr:+.3f} [{status}]")

    # Prediction 2: Gap grows with difficulty
    print(f"\n  Prediction 2: Performance gap grows with difficulty")
    gaps = {}
    for difficulty in ("easy", "medium", "hard"):
        diff_levels = [l for l in levels if l.difficulty == difficulty]
        d_success = []
        nod_success = []
        for level in diff_levels:
            d_sub = [r for r in results if r.level == level.name and r.variant == "modular_ond_interp"]
            nod_sub = [r for r in results if r.level == level.name and r.variant == "modular_nod"]
            d_success.extend([1.0 if r.success else 0.0 for r in d_sub])
            nod_success.extend([1.0 if r.success else 0.0 for r in nod_sub])
        gaps[difficulty] = mean(d_success) - mean(nod_success)

    if gaps.get("hard", 0) > gaps.get("easy", 0) + 0.01:
        print(f"    easy={gaps['easy']:+.3f} medium={gaps['medium']:+.3f} hard={gaps['hard']:+.3f} [PASS]")
    else:
        print(f"    easy={gaps['easy']:+.3f} medium={gaps['medium']:+.3f} hard={gaps['hard']:+.3f} [PARTIAL/FAIL]")
        all_pass = False

    # Prediction 3: D interpretation rate = 100%, no-D = 0%
    print(f"\n  Prediction 3: D interpretation rate = 100%, no-D = 0%")
    d_interp_total = sum(r.hints_interpreted for r in results if r.variant == "modular_ond_interp")
    d_hints_total = sum(r.hints_collected for r in results if r.variant == "modular_ond_interp")
    nod_interp_total = sum(r.hints_interpreted for r in results if r.variant == "modular_nod")
    nod_hints_total = sum(r.hints_collected for r in results if r.variant == "modular_nod")

    d_rate = d_interp_total / d_hints_total if d_hints_total > 0 else 0.0
    nod_rate = nod_interp_total / nod_hints_total if nod_hints_total > 0 else 0.0
    passed = d_rate > 0.9 and nod_rate < 0.1
    if not passed:
        all_pass = False
    print(f"    D-interp rate: {d_rate:.1%} ({d_interp_total}/{d_hints_total})")
    print(f"    no-D rate:     {nod_rate:.1%} ({nod_interp_total}/{nod_hints_total})")
    print(f"    [{'PASS' if passed else 'FAIL'}]")

    # Prediction 4: 4-goal overhead reduces D advantage (informational)
    print(f"\n  Prediction 4: 4-goal hint-gathering overhead (informational)")
    for difficulty in ("easy", "medium", "hard"):
        levels_2g = [l for l in levels if l.difficulty == difficulty and "2g" in l.name]
        levels_4g = [l for l in levels if l.difficulty == difficulty and "4g" in l.name]

        gap_2g = 0.0
        gap_4g = 0.0

        if levels_2g:
            d_s = [1.0 if r.success else 0.0 for l in levels_2g for r in results if r.level == l.name and r.variant == "modular_ond_interp"]
            n_s = [1.0 if r.success else 0.0 for l in levels_2g for r in results if r.level == l.name and r.variant == "modular_nod"]
            gap_2g = mean(d_s) - mean(n_s) if d_s and n_s else 0.0

        if levels_4g:
            d_s = [1.0 if r.success else 0.0 for l in levels_4g for r in results if r.level == l.name and r.variant == "modular_ond_interp"]
            n_s = [1.0 if r.success else 0.0 for l in levels_4g for r in results if r.level == l.name and r.variant == "modular_nod"]
            gap_4g = mean(d_s) - mean(n_s) if d_s and n_s else 0.0

        note = "D advantage" if gap_4g > 0.01 else ("tie" if abs(gap_4g) <= 0.01 else "overhead dominates")
        print(f"    {difficulty:>6s}: 2g_gap={gap_2g:+.3f} 4g_gap={gap_4g:+.3f} ({note})")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME PREDICTIONS NOT MET'}")


if __name__ == "__main__":
    run_batch(n=100, goal_mode="seek")

"""
Stufe 9: Task-Change Stabilization — Deconstruct as Context Transfer

DEF Claim: Deconstruct's stabilizing effect manifests during task changes.
It persists context into actionable state and can cleanly overwrite that
state when the task changes. In deconstruct.py lines 54-55, a new hint:B
tag overwrites the old mem["hint_goal"] and mem["target"]. This test
validates that this overwrite mechanism produces measurably better
adaptation than alternatives.

Two-phase episodes:
  Phase 1: Agent seeks Goal A via hint cell 1
  Phase 2: Goal switches to B, hint cell 2 reveals new target

Variants:
  no_d:          No D, no deconstruct — baseline
  d_no_decon:    D active but deconstruct disabled — D sees hints but C never learns
  decon_persist: D + deconstruct, memory persists through switch (overwrite test)
  decon_clear:   D + deconstruct, memory cleared at switch (fresh start control)

DEF Predictions:
  1. decon_persist Phase-2 SR >> no_d Phase-2 SR
  2. decon_persist Phase-2 SR >> d_no_decon Phase-2 SR
  3. target_updated ~100% for decon variants, 0% for others
  4. decon_persist ≈ decon_clear (overwrite is self-correcting)
  5. Phase-1 performance similar across all D variants
"""

import csv
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from env.task_change import TaskChangeGridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA, ZD

from eval.stats import (
    confidence_interval_proportion,
    compare_variants,
    mean,
)

ACTIONS = ("up", "down", "left", "right")

VARIANTS = ["no_d", "d_no_decon", "decon_persist", "decon_clear"]


# ── Test Configurations ──────────────────────────────────────────────

@dataclass
class TaskChangeConfig:
    name: str
    width: int
    height: int
    n_random_obstacles: int
    switch_after_steps: Optional[int]   # None = switch on Phase 1 completion
    max_steps: int


def _make_configs() -> List[TaskChangeConfig]:
    return [
        TaskChangeConfig(
            name="10x10_phase1",
            width=10, height=10,
            n_random_obstacles=5,
            switch_after_steps=None,
            max_steps=150,
        ),
        TaskChangeConfig(
            name="10x10_step50",
            width=10, height=10,
            n_random_obstacles=5,
            switch_after_steps=50,
            max_steps=150,
        ),
        TaskChangeConfig(
            name="15x15_phase1",
            width=15, height=15,
            n_random_obstacles=10,
            switch_after_steps=None,
            max_steps=250,
        ),
    ]


# ── Result Dataclass ─────────────────────────────────────────────────

@dataclass
class TaskChangeResult:
    variant: str
    config_name: str
    goal_mode: str
    success: bool                       # Phase 2 goal reached
    steps: int
    total_reward: float
    phase1_steps: int
    phase2_steps: int
    phase1_reached: bool                # agent reached Phase 1 goal
    switch_step: int
    target_correct_at_switch: bool      # mem["target"] == Goal A at switch
    target_updated_after: bool          # mem["target"] == Goal B after Phase 2 hint
    steps_to_target_update: int         # steps from switch to target update (-1 if never)
    d_triggers: int
    hints_collected: int
    hints_interpreted: int
    stay_rate: float
    phase2_stay_rate: float


# ── Episode Runner ───────────────────────────────────────────────────

def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
    ))


def run_episode(
    config: TaskChangeConfig,
    variant: str,
    goal_mode: str,
    seed: Optional[int] = None,
) -> TaskChangeResult:
    """Run a single two-phase episode."""

    env = TaskChangeGridWorld(
        width=config.width,
        height=config.height,
        seed=seed,
        phase1_goal_id="A",
        phase2_goal_id="B",
        switch_after_steps=config.switch_after_steps,
        n_random_obstacles=config.n_random_obstacles,
    )
    obs = env.reset()

    goal_map = env.goal_map
    phase1_goal_pos = goal_map["A"]
    phase2_goal_pos = goal_map["B"]

    A = AgentA()
    B = AgentB()

    # All variants get C — they differ in D and deconstruct
    zC = ZC(goal_mode=goal_mode, memory={})
    initial_target = env.hint1_pos  # navigate to hint cell 1 first
    C = AgentC(goal=GoalSpec(mode=goal_mode, target=initial_target), anti_stay_penalty=1.1)

    use_d = variant in ("d_no_decon", "decon_persist", "decon_clear")
    use_decon = variant in ("decon_persist", "decon_clear")

    D = AgentD() if use_d else None
    router = _make_router() if use_d else None

    # Tracking
    d_triggers = 0
    hints_collected = 0
    hints_interpreted = 0
    stay_count = 0
    phase2_stay_count = 0
    total_reward = 0.0
    done = False
    phase1_reached = False
    switch_step = -1
    switch_detected = False
    target_correct_at_switch = False
    target_updated_after = False
    steps_to_target_update = -1
    phase1_steps = 0
    phase2_steps = 0

    last_positions = deque(maxlen=20)

    # Navigation state for hint cell queue
    # All variants navigate to hint cells identically via C's target
    hint1_visited = False
    hint2_visited = False
    hint2_available = False

    for t in range(config.max_steps):
        zA = A.infer_zA(obs)

        # ── Update C's target ──
        # Navigation logic ensures all variants visit hint cells equally.
        # The difference is what happens AFTER: decon variants learn a target
        # from the hint, non-decon variants navigate to a default position.
        if switch_detected and target_updated_after:
            # Phase 2 target successfully learned via deconstruct
            C.goal.target = tuple(zC.memory["target"])
        elif switch_detected and not hint2_visited:
            # Phase 2, hint cell 2 not yet visited — go collect it
            C.goal.target = env.hint2_pos
        elif switch_detected:
            # Phase 2, hint cell 2 visited but target not learned (no decon)
            # Agent is blind: navigates to Phase 1 goal (wrong!) or wanders
            C.goal.target = phase1_goal_pos
        elif "target" in zC.memory and zC.memory["target"] is not None:
            # Phase 1: learned target from deconstruct
            C.goal.target = tuple(zC.memory["target"])
        elif not hint1_visited:
            # Phase 1: go collect hint 1
            C.goal.target = env.hint1_pos
        else:
            # Phase 1: hint 1 visited but no target — head to default
            C.goal.target = phase1_goal_pos

        # ── Action Selection ──
        action, scored = C.choose_action(
            zA, B.predict_next,
            memory=zC.memory if use_decon else None,
            tie_break_delta=0.25,
        )
        decision_delta = scored[0][1] - scored[1][1] if len(scored) >= 2 else 999.0

        # ── Environment Step ──
        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1
            if env.current_phase == 2:
                phase2_stay_count += 1

        last_positions.append(zA_next.agent_pos)
        total_reward += reward

        # ── Detect Phase Switch ──
        if env.phase_switched and not switch_detected:
            switch_detected = True
            switch_step = env.switch_step if env.switch_step is not None else t
            phase1_steps = t
            phase1_reached = True  # switch on phase1 complete means we reached it

            # Check if target was correctly learned in Phase 1
            target_correct_at_switch = (
                zC.memory.get("target") is not None
                and tuple(zC.memory["target"]) == phase1_goal_pos
            )

            # For decon_clear: wipe memory at switch
            if variant == "decon_clear":
                zC = ZC(goal_mode=goal_mode, memory={})

            hint2_available = True

        # Count phase 2 steps
        if switch_detected:
            phase2_steps += 1

        # Track hint cell visits
        if zA_next.agent_pos == env.hint1_pos:
            hint1_visited = True
        if zA_next.agent_pos == env.hint2_pos and switch_detected:
            hint2_visited = True

        # ── Hint Processing ──
        if zA_next.hint is not None:
            hints_collected += 1

            if use_d and D is not None:
                D.observe_step(t=t, zA=zA_next, action="hint", reward=0.0, done=False)
                zD_hint = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)

                # Check interpretation
                for tag in zD_hint.meaning_tags:
                    tag_lower = tag.lower()
                    if tag_lower.startswith("hint:") and tag_lower[5:] in [g.lower() for g in goal_map]:
                        hints_interpreted += 1
                        break

                if use_decon and zC is not None:
                    zC = deconstruct_d_to_c(zC, zD_hint, goal_map=goal_map)
                    d_triggers += 1

                    # Check if target was updated to Phase 2 goal
                    if (switch_detected and not target_updated_after
                            and zC.memory.get("target") is not None
                            and tuple(zC.memory["target"]) == phase2_goal_pos):
                        target_updated_after = True
                        steps_to_target_update = t - switch_step

            # no_d and d_no_decon: hint is observed but NOT parsed into
            # C's memory. The agent visits the hint cell (navigation is shared)
            # but gains no actionable state from it.

        obs = obs_next

        # ── D Logic (router-gated) ──
        if use_d and D is not None:
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d = False
            if router and decision_delta is not None:
                activate_d, reason = router.should_activate_d(
                    t=t,
                    last_positions=tuple(last_positions),
                    decision_delta=decision_delta,
                )

            if activate_d:
                d_triggers += 1
                zD = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)

                if use_decon and zC is not None:
                    zC = deconstruct_d_to_c(zC, zD, goal_map=goal_map)

                    # Check target update (could happen via router-gated D too)
                    if (switch_detected and not target_updated_after
                            and zC.memory.get("target") is not None
                            and tuple(zC.memory["target"]) == phase2_goal_pos):
                        target_updated_after = True
                        steps_to_target_update = t - switch_step

        if done:
            break

    steps = t + 1 if done else config.max_steps
    if not switch_detected:
        phase1_steps = steps
        # For step-based switch, the agent might not have triggered it
        if config.switch_after_steps is not None and steps >= config.switch_after_steps:
            switch_step = config.switch_after_steps

    stay_rate = stay_count / steps if steps > 0 else 0.0
    p2_stay_rate = phase2_stay_count / phase2_steps if phase2_steps > 0 else 0.0

    return TaskChangeResult(
        variant=variant,
        config_name=config.name,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        phase1_reached=phase1_reached,
        switch_step=switch_step,
        target_correct_at_switch=target_correct_at_switch,
        target_updated_after=target_updated_after,
        steps_to_target_update=steps_to_target_update,
        d_triggers=d_triggers,
        hints_collected=hints_collected,
        hints_interpreted=hints_interpreted,
        stay_rate=stay_rate,
        phase2_stay_rate=p2_stay_rate,
    )


# ── Batch Runner ─────────────────────────────────────────────────────

def run_batch(n: int = 100, goal_mode: str = "seek"):
    """Run Stufe 9 task-change stabilization study."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/task_change_{run_id}.csv"

    configs = _make_configs()
    results: List[TaskChangeResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(configs) * len(VARIANTS) * n
    pbar = tqdm(total=total, desc="task_change") if use_tqdm else None

    for config in configs:
        for variant in VARIANTS:
            for i in range(n):
                r = run_episode(config, variant, goal_mode, seed=i)
                results.append(r)
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    # Write CSV
    fields = [
        "variant", "config_name", "goal_mode", "success", "steps",
        "total_reward", "phase1_steps", "phase2_steps", "phase1_reached",
        "switch_step", "target_correct_at_switch", "target_updated_after",
        "steps_to_target_update", "d_triggers", "hints_collected",
        "hints_interpreted", "stay_rate", "phase2_stay_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r.variant, r.config_name, r.goal_mode, r.success, r.steps,
                f"{r.total_reward:.4f}", r.phase1_steps, r.phase2_steps,
                r.phase1_reached, r.switch_step,
                r.target_correct_at_switch, r.target_updated_after,
                r.steps_to_target_update, r.d_triggers, r.hints_collected,
                r.hints_interpreted, f"{r.stay_rate:.4f}",
                f"{r.phase2_stay_rate:.4f}",
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_results_table(results, configs)
    _print_memory_analysis(results, configs)
    _print_adaptation_speed(results, configs)
    _print_persist_vs_clear(results, configs)
    _print_def_predictions(results, configs)

    return results


# ── Analysis Functions ───────────────────────────────────────────────

def _print_results_table(results: List[TaskChangeResult], configs: List[TaskChangeConfig]):
    """Phase 2 success rate by config x variant."""
    print(f"\n{'=' * 100}")
    print(f"  STUFE 9: TASK-CHANGE STABILIZATION -- Phase 2 Success Rate")
    print(f"{'=' * 100}")

    print(f"\n  {'config':<20s}", end="")
    for v in VARIANTS:
        print(f" {v:>16s}", end="")
    print()
    print(f"  {'-' * 20}", end="")
    for _ in VARIANTS:
        print(f" {'-' * 16}", end="")
    print()

    for config in configs:
        print(f"  {config.name:<20s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.config_name == config.name and r.variant == v]
            if subset:
                sr = sum(1 for r in subset if r.success) / len(subset)
                _, lo, hi = confidence_interval_proportion(
                    sum(1 for r in subset if r.success), len(subset)
                )
                print(f" {sr:>5.3f}[{lo:.2f},{hi:.2f}]", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()

    # Also show Phase 1 reached rates
    print(f"\n  Phase 1 reached rates:")
    print(f"  {'config':<20s}", end="")
    for v in VARIANTS:
        print(f" {v:>16s}", end="")
    print()
    print(f"  {'-' * 20}", end="")
    for _ in VARIANTS:
        print(f" {'-' * 16}", end="")
    print()
    for config in configs:
        print(f"  {config.name:<20s}", end="")
        for v in VARIANTS:
            subset = [r for r in results if r.config_name == config.name and r.variant == v]
            if subset:
                rate = sum(1 for r in subset if r.phase1_reached) / len(subset)
                print(f" {rate:>16.3f}", end="")
            else:
                print(f" {'N/A':>16s}", end="")
        print()


def _print_memory_analysis(results: List[TaskChangeResult], configs: List[TaskChangeConfig]):
    """Target learning and update rates."""
    print(f"\n{'=' * 100}")
    print(f"  MEMORY UPDATE ANALYSIS")
    print(f"  (target_correct_at_switch: learned Goal A; target_updated_after: overwrote to Goal B)")
    print(f"{'=' * 100}")

    print(f"\n  {'variant':<16s} {'config':<20s} {'switch_target':>14s} {'updated_to_B':>14s} {'episodes':>10s}")
    print(f"  {'-' * 16} {'-' * 20} {'-' * 14} {'-' * 14} {'-' * 10}")

    for v in VARIANTS:
        for config in configs:
            subset = [r for r in results if r.variant == v and r.config_name == config.name]
            if not subset:
                continue
            n_correct = sum(1 for r in subset if r.target_correct_at_switch)
            n_updated = sum(1 for r in subset if r.target_updated_after)
            n = len(subset)
            print(f"  {v:<16s} {config.name:<20s} "
                  f"{n_correct:>4d}/{n:<4d} ({100 * n_correct / n:>5.1f}%) "
                  f"{n_updated:>4d}/{n:<4d} ({100 * n_updated / n:>5.1f}%) "
                  f"{n:>10d}")


def _print_adaptation_speed(results: List[TaskChangeResult], configs: List[TaskChangeConfig]):
    """Steps to target update after switch."""
    print(f"\n{'=' * 100}")
    print(f"  ADAPTATION SPEED -- Steps from Switch to Target Update")
    print(f"{'=' * 100}")

    for config in configs:
        print(f"\n  --- {config.name} ---")
        for v in VARIANTS:
            subset = [r for r in results if r.variant == v and r.config_name == config.name]
            updated = [r for r in subset if r.target_updated_after and r.steps_to_target_update >= 0]
            if updated:
                vals = [r.steps_to_target_update for r in updated]
                avg = mean(vals)
                mn = min(vals)
                mx = max(vals)
                print(f"  {v:<16s}: mean={avg:>6.1f}  min={mn:>3d}  max={mx:>3d}  (n={len(updated)})")
            else:
                print(f"  {v:<16s}: no target updates")


def _print_persist_vs_clear(results: List[TaskChangeResult], configs: List[TaskChangeConfig]):
    """Direct comparison: decon_persist vs decon_clear."""
    print(f"\n{'=' * 100}")
    print(f"  PERSIST vs CLEAR -- Is explicit memory clearing needed?")
    print(f"  (Hypothesis: overwrite mechanism is self-correcting, clearing unnecessary)")
    print(f"{'=' * 100}")

    for config in configs:
        persist = [r for r in results if r.variant == "decon_persist" and r.config_name == config.name]
        clear = [r for r in results if r.variant == "decon_clear" and r.config_name == config.name]

        if not persist or not clear:
            continue

        p_sr = sum(1 for r in persist if r.success) / len(persist)
        c_sr = sum(1 for r in clear if r.success) / len(clear)
        delta = p_sr - c_sr

        p_vals = [1.0 if r.success else 0.0 for r in persist]
        c_vals = [1.0 if r.success else 0.0 for r in clear]
        report = compare_variants("decon_persist", p_vals, "decon_clear", c_vals, "sr", is_proportion=True)

        sig = "***" if report["p_value"] < 0.001 else "**" if report["p_value"] < 0.01 else "*" if report["p_value"] < 0.05 else "ns"
        print(f"\n  {config.name}:")
        print(f"    decon_persist: SR={p_sr:.3f}  decon_clear: SR={c_sr:.3f}  delta={delta:+.3f}  p={report['p_value']:.4f} {sig}")

        if abs(delta) < 0.1 and report["p_value"] >= 0.05:
            print(f"    => Overwrite is self-correcting (no clearing needed)")
        elif delta > 0.05 and report["p_value"] < 0.05:
            print(f"    => Persist outperforms clear (overwrite more efficient than rebuild)")
        elif delta < -0.05 and report["p_value"] < 0.05:
            print(f"    => Clear outperforms persist (old memory interferes)")


def _print_def_predictions(results: List[TaskChangeResult], configs: List[TaskChangeConfig]):
    """Check all 5 DEF predictions."""
    print(f"\n{'=' * 100}")
    print(f"  DEF PREDICTIONS CHECK -- Task-Change Stabilization")
    print(f"{'=' * 100}")

    all_pass = True

    # Prediction 1: decon_persist Phase-2 SR >> no_d
    # Aggregated across all configs (per-config detail shown for transparency)
    print(f"\n  Prediction 1: decon_persist >> no_d on Phase-2 success")
    all_persist_1 = [r for r in results if r.variant == "decon_persist"]
    all_no_d = [r for r in results if r.variant == "no_d"]
    for config in configs:
        persist = [r for r in results if r.variant == "decon_persist" and r.config_name == config.name]
        no_d = [r for r in results if r.variant == "no_d" and r.config_name == config.name]
        if persist and no_d:
            p_sr = sum(1 for r in persist if r.success) / len(persist)
            n_sr = sum(1 for r in no_d if r.success) / len(no_d)
            gap = p_sr - n_sr
            p_vals = [1.0 if r.success else 0.0 for r in persist]
            n_vals = [1.0 if r.success else 0.0 for r in no_d]
            report = compare_variants("decon_persist", p_vals, "no_d", n_vals, "sr", is_proportion=True)
            sig = "***" if report["p_value"] < 0.001 else "**" if report["p_value"] < 0.01 else "*" if report["p_value"] < 0.05 else "ns"
            print(f"    {config.name}: persist={p_sr:.3f} vs no_d={n_sr:.3f} gap={gap:+.3f} p={report['p_value']:.4f} {sig}")
    # Aggregate check
    p_vals_agg = [1.0 if r.success else 0.0 for r in all_persist_1]
    n_vals_agg = [1.0 if r.success else 0.0 for r in all_no_d]
    agg_report = compare_variants("decon_persist", p_vals_agg, "no_d", n_vals_agg, "sr", is_proportion=True)
    p1_pass = mean(p_vals_agg) > mean(n_vals_agg) and agg_report["p_value"] < 0.05
    tag = "PASS" if p1_pass else "FAIL"
    if not p1_pass:
        all_pass = False
    print(f"    AGGREGATE: persist={mean(p_vals_agg):.3f} vs no_d={mean(n_vals_agg):.3f} p={agg_report['p_value']:.4f} [{tag}]")

    # Prediction 2: decon_persist Phase-2 SR >> d_no_decon
    print(f"\n  Prediction 2: decon_persist >> d_no_decon (deconstruct matters, not D alone)")
    all_d_nodc = [r for r in results if r.variant == "d_no_decon"]
    for config in configs:
        persist = [r for r in results if r.variant == "decon_persist" and r.config_name == config.name]
        d_nodc = [r for r in results if r.variant == "d_no_decon" and r.config_name == config.name]
        if persist and d_nodc:
            p_sr = sum(1 for r in persist if r.success) / len(persist)
            d_sr = sum(1 for r in d_nodc if r.success) / len(d_nodc)
            gap = p_sr - d_sr
            p_vals = [1.0 if r.success else 0.0 for r in persist]
            d_vals = [1.0 if r.success else 0.0 for r in d_nodc]
            report = compare_variants("decon_persist", p_vals, "d_no_decon", d_vals, "sr", is_proportion=True)
            sig = "***" if report["p_value"] < 0.001 else "**" if report["p_value"] < 0.01 else "*" if report["p_value"] < 0.05 else "ns"
            print(f"    {config.name}: persist={p_sr:.3f} vs d_no_decon={d_sr:.3f} gap={gap:+.3f} p={report['p_value']:.4f} {sig}")
    # Aggregate check
    p_vals_agg2 = [1.0 if r.success else 0.0 for r in all_persist_1]
    d_vals_agg = [1.0 if r.success else 0.0 for r in all_d_nodc]
    agg_report2 = compare_variants("decon_persist", p_vals_agg2, "d_no_decon", d_vals_agg, "sr", is_proportion=True)
    p2_pass = mean(p_vals_agg2) > mean(d_vals_agg) and agg_report2["p_value"] < 0.05
    tag = "PASS" if p2_pass else "FAIL"
    if not p2_pass:
        all_pass = False
    print(f"    AGGREGATE: persist={mean(p_vals_agg2):.3f} vs d_no_decon={mean(d_vals_agg):.3f} p={agg_report2['p_value']:.4f} [{tag}]")

    # Prediction 3: target_updated ~100% for decon, 0% for others
    print(f"\n  Prediction 3: target_updated_after ~100%% for decon variants, ~0%% for no_d/d_no_decon")
    for v in VARIANTS:
        subset = [r for r in results if r.variant == v and r.phase1_reached]
        if subset:
            rate = sum(1 for r in subset if r.target_updated_after) / len(subset)
            if v in ("decon_persist", "decon_clear"):
                passed = rate > 0.8
            else:
                passed = rate < 0.2
            tag = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    {v:<16s}: {rate:.1%} ({sum(1 for r in subset if r.target_updated_after)}/{len(subset)}) [{tag}]")

    # Prediction 4: decon_persist ≈ decon_clear
    print(f"\n  Prediction 4: decon_persist ~= decon_clear (overwrite is self-correcting)")
    for config in configs:
        persist = [r for r in results if r.variant == "decon_persist" and r.config_name == config.name]
        clear = [r for r in results if r.variant == "decon_clear" and r.config_name == config.name]
        if persist and clear:
            p_sr = sum(1 for r in persist if r.success) / len(persist)
            c_sr = sum(1 for r in clear if r.success) / len(clear)
            delta = abs(p_sr - c_sr)
            p_vals = [1.0 if r.success else 0.0 for r in persist]
            c_vals = [1.0 if r.success else 0.0 for r in clear]
            report = compare_variants("decon_persist", p_vals, "decon_clear", c_vals, "sr", is_proportion=True)
            passed = delta < 0.1 or report["p_value"] >= 0.05
            tag = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    {config.name}: persist={p_sr:.3f} vs clear={c_sr:.3f} delta={delta:.3f} p={report['p_value']:.4f} [{tag}]")

    # Prediction 5: Phase 1 similar across D variants
    print(f"\n  Prediction 5: Phase-1 performance similar across D variants")
    d_variants = ["d_no_decon", "decon_persist", "decon_clear"]
    for config in configs:
        srs = {}
        for v in d_variants:
            subset = [r for r in results if r.variant == v and r.config_name == config.name]
            if subset:
                srs[v] = sum(1 for r in subset if r.phase1_reached) / len(subset)
        if len(srs) >= 2:
            vals = list(srs.values())
            spread = max(vals) - min(vals)
            passed = spread < 0.15
            tag = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            detail = "  ".join(f"{v}={sr:.3f}" for v, sr in srs.items())
            print(f"    {config.name}: {detail}  spread={spread:.3f} [{tag}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stufe 9: Task-Change Stabilization")
    parser.add_argument("--n", type=int, default=100, help="Episodes per combination")
    parser.add_argument("--goal-mode", type=str, default="seek", help="Goal mode")
    args = parser.parse_args()

    print("Stufe 9: Task-Change Stabilization -- Deconstruct as Context Transfer")
    print(f"  Episodes per combination: {args.n}")
    print(f"  Configs: {len(_make_configs())}")
    print(f"  Variants: {VARIANTS}")

    run_batch(n=args.n, goal_mode=args.goal_mode)

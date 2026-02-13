"""
Stufe 5: Regime-Transition Validation

DEF Claim: The router correctly transitions between 3D and 4D regimes
based on task difficulty. Simple tasks should stay in 3D (C sufficient);
complex tasks should trigger 4D (D needed for narrative/hint integration).

Task types:
  2D_task: Navigate to VISIBLE goal, no obstacles -> D should never trigger
  3D_task: Navigate to VISIBLE goal with obstacles -> D rarely triggers
  4D_task: Navigate to HIDDEN goal with hints needed -> D frequently triggers

DEF Predictions:
  1. 2D tasks -> almost 100% 3D regime (C handles it easily, no uncertainty)
  2. 4D tasks -> more 4D regime steps (uncertainty triggers D frequently)
  3. Harder tasks -> more regime switches
  4. D's marginal gain is higher in 4D tasks than in 2D tasks
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA

from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    mean,
)


@dataclass
class RegimeResult:
    task_type: str       # "2D_task", "3D_task", "4D_task"
    success: bool
    steps: int
    total_reward: float
    pct_3d: float        # fraction of steps in 3D regime
    pct_4d: float        # fraction of steps in 4D regime
    regime_switches: int  # number of 3D<->4D transitions
    d_triggers: int
    d_trigger_reasons: str  # comma-separated trigger reasons


def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
    ))


def run_2d_task(seed: Optional[int] = None, max_steps: int = 50) -> RegimeResult:
    """
    2D Task: Visible goal, no obstacles, straight-line navigation.
    C should have zero uncertainty -> D never triggers.
    """
    # Minimal 5x5 grid, no obstacles, goal visible
    env = GridWorld(width=5, height=5, seed=seed, obstacles=[])
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
    D = AgentD()
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []

    for t in range(max_steps):
        zA = A.infer_zA(obs)
        action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
        delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)
        total_reward += reward

        D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

        activate, reason = router.should_activate_d(
            t=t, last_positions=(zA_next.agent_pos,), decision_delta=delta,
        )
        if activate:
            trigger_reasons.append(reason)
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return RegimeResult(
        task_type="2D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons) if trigger_reasons else "none",
    )


def run_3d_task(seed: Optional[int] = None, max_steps: int = 50) -> RegimeResult:
    """
    3D Task: Visible goal, obstacles present, agent must route around them.
    C handles most cases, but obstacles may create tie situations -> occasional D.
    """
    env = GridWorld(width=10, height=10, seed=seed, n_random_obstacles=10)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
    D = AgentD()
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []

    from collections import deque
    last_pos = deque(maxlen=20)

    for t in range(max_steps):
        zA = A.infer_zA(obs)
        if "target" in zC.memory:
            C.goal.target = tuple(zC.memory["target"])

        action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
        delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)
        total_reward += reward
        last_pos.append(zA_next.agent_pos)

        D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

        activate, reason = router.should_activate_d(
            t=t, last_positions=tuple(last_pos), decision_delta=delta,
        )
        if activate:
            trigger_reasons.append(reason)
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return RegimeResult(
        task_type="3D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons[:10]) if trigger_reasons else "none",
    )


def run_4d_task(seed: Optional[int] = None, max_steps: int = 100) -> RegimeResult:
    """
    4D Task: Hidden goal, must collect hint, obstacles present.
    Agent must navigate to hint cell, process hint (D+deconstruct), then navigate to goal.
    High uncertainty at start -> D triggers frequently.
    """
    env = GridWorld(width=10, height=10, seed=seed, n_random_obstacles=5)
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    # Start with hint cell as target (goal is hidden)
    hint_target = env.hint_cell
    C = AgentC(goal=GoalSpec(mode="seek", target=hint_target), anti_stay_penalty=1.1)
    D = AgentD()
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []
    goal_learned = False

    from collections import deque
    last_pos = deque(maxlen=20)

    for t in range(max_steps):
        zA = A.infer_zA(obs)

        # Navigate to hint first, then to goal once learned
        if "target" in zC.memory and zC.memory["target"] is not None:
            C.goal.target = tuple(zC.memory["target"])
            goal_learned = True
        elif not goal_learned:
            C.goal.target = hint_target

        action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
        delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)
        total_reward += reward
        last_pos.append(zA_next.agent_pos)

        D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

        # Process hint through D if seen
        if zA_next.hint is not None:
            zD_hint = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=1)
            zC = deconstruct_d_to_c(zC, zD_hint)
            trigger_reasons.append("hint_capture")

        activate, reason = router.should_activate_d(
            t=t, last_positions=tuple(last_pos), decision_delta=delta,
        )
        if activate:
            trigger_reasons.append(reason)
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return RegimeResult(
        task_type="4D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons[:10]) if trigger_reasons else "none",
    )


# ── Batch Runner ──────────────────────────────────────────────────────

def run_batch(n: int = 100):
    """Run regime transition study across all task types."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/regime_transition_{run_id}.csv"

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    task_fns = {
        "2D_task": run_2d_task,
        "3D_task": run_3d_task,
        "4D_task": run_4d_task,
    }

    results: List[RegimeResult] = []
    total = len(task_fns) * n

    if use_tqdm:
        pbar = tqdm(total=total, desc="regime_transition")

    for task_name, fn in task_fns.items():
        for i in range(n):
            r = fn(seed=i)
            results.append(r)
            if use_tqdm:
                pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "task_type", "success", "steps", "total_reward",
            "pct_3d", "pct_4d", "regime_switches", "d_triggers",
            "d_trigger_reasons",
        ])
        for r in results:
            w.writerow([
                r.task_type, r.success, r.steps,
                f"{r.total_reward:.4f}",
                f"{r.pct_3d:.4f}", f"{r.pct_4d:.4f}",
                r.regime_switches, r.d_triggers,
                r.d_trigger_reasons,
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_regime_table(results)
    _print_def_predictions(results)
    _print_trigger_analysis(results)

    return results


def _print_regime_table(results: List[RegimeResult]):
    """Print regime distribution per task type."""
    print(f"\n{'='*90}")
    print(f"  REGIME TRANSITION -- Distribution by Task Type")
    print(f"{'='*90}")
    print(
        f"  {'task_type':<12s} {'sr':>5s} {'steps':>6s} "
        f"{'%3D':>6s} {'%4D':>6s} {'switches':>8s} {'d_trig':>7s}"
    )
    print(
        f"  {'-'*12} {'-'*5} {'-'*6} "
        f"{'-'*6} {'-'*6} {'-'*8} {'-'*7}"
    )

    for task_type in ["2D_task", "3D_task", "4D_task"]:
        subset = [r for r in results if r.task_type == task_type]
        if not subset:
            continue

        sr = sum(1 for r in subset if r.success) / len(subset)
        steps_m = mean([float(r.steps) for r in subset])
        pct_3d_m = mean([r.pct_3d for r in subset])
        pct_4d_m = mean([r.pct_4d for r in subset])
        switches_m = mean([float(r.regime_switches) for r in subset])
        d_trig_m = mean([float(r.d_triggers) for r in subset])

        print(
            f"  {task_type:<12s} {sr:>5.3f} {steps_m:>6.1f} "
            f"{pct_3d_m:>6.3f} {pct_4d_m:>6.3f} {switches_m:>8.1f} {d_trig_m:>7.1f}"
        )


def _print_def_predictions(results: List[RegimeResult]):
    """Validate DEF predictions about regime transitions."""
    print(f"\n{'='*90}")
    print(f"  DEF PREDICTIONS -- Regime Transition")
    print(f"{'='*90}")

    def avg(task_type, field):
        subset = [r for r in results if r.task_type == task_type]
        if not subset:
            return 0.0
        return mean([getattr(r, field) for r in subset])

    # Prediction 1: 2D tasks -> almost all 3D regime
    pct_3d_2d = avg("2D_task", "pct_3d")
    pct_3d_4d = avg("4D_task", "pct_3d")
    print(f"\n  1. '2D tasks stay mostly in 3D regime'")
    print(f"     2D_task %3D: {pct_3d_2d:.3f}")
    print(f"     4D_task %3D: {pct_3d_4d:.3f}")
    if pct_3d_2d > 0.9:
        print(f"     [PASS] 2D tasks have {pct_3d_2d:.1%} of steps in 3D regime")
    elif pct_3d_2d > 0.7:
        print(f"     [PARTIAL] 2D tasks at {pct_3d_2d:.1%} 3D (expected >90%)")
    else:
        print(f"     [WARN] 2D tasks only {pct_3d_2d:.1%} in 3D")

    # Prediction 2: 4D tasks -> more 4D regime
    pct_4d_2d = avg("2D_task", "pct_4d")
    pct_4d_4d = avg("4D_task", "pct_4d")
    print(f"\n  2. '4D tasks use more 4D regime than 2D tasks'")
    print(f"     2D_task %4D: {pct_4d_2d:.3f}")
    print(f"     4D_task %4D: {pct_4d_4d:.3f}")
    if pct_4d_4d > pct_4d_2d:
        print(f"     [PASS] 4D tasks use more D ({pct_4d_4d:.3f} > {pct_4d_2d:.3f})")
    else:
        print(f"     [WARN] Expected 4D tasks to use more D")

    # Prediction 3: Harder tasks -> more regime switches
    sw_2d = avg("2D_task", "regime_switches")
    sw_3d = avg("3D_task", "regime_switches")
    sw_4d = avg("4D_task", "regime_switches")
    print(f"\n  3. 'Harder tasks have more regime switches'")
    print(f"     2D_task switches: {sw_2d:.1f}")
    print(f"     3D_task switches: {sw_3d:.1f}")
    print(f"     4D_task switches: {sw_4d:.1f}")
    if sw_4d > sw_2d:
        print(f"     [PASS] 4D tasks have more switches ({sw_4d:.1f} > {sw_2d:.1f})")
    else:
        print(f"     [PARTIAL] Switch counts similar")

    # Prediction 4: D triggers scale with task difficulty
    dt_2d = avg("2D_task", "d_triggers")
    dt_3d = avg("3D_task", "d_triggers")
    dt_4d = avg("4D_task", "d_triggers")
    print(f"\n  4. 'D triggers increase with task difficulty'")
    print(f"     2D_task D-triggers: {dt_2d:.1f}")
    print(f"     3D_task D-triggers: {dt_3d:.1f}")
    print(f"     4D_task D-triggers: {dt_4d:.1f}")
    if dt_4d > dt_3d > dt_2d:
        print(f"     [PASS] D triggers: 2D({dt_2d:.1f}) < 3D({dt_3d:.1f}) < 4D({dt_4d:.1f})")
    elif dt_4d > dt_2d:
        print(f"     [PARTIAL] 4D > 2D but 3D out of order")
    else:
        print(f"     [WARN] Expected monotonic increase")


def _print_trigger_analysis(results: List[RegimeResult]):
    """Analyze what triggers D in each task type."""
    print(f"\n{'='*90}")
    print(f"  D TRIGGER REASON ANALYSIS")
    print(f"{'='*90}")

    for task_type in ["2D_task", "3D_task", "4D_task"]:
        subset = [r for r in results if r.task_type == task_type]
        if not subset:
            continue

        # Count trigger reasons across all episodes
        reason_counts: Dict[str, int] = defaultdict(int)
        total_triggers = 0
        for r in subset:
            if r.d_trigger_reasons == "none":
                continue
            for reason in r.d_trigger_reasons.split(","):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] += 1
                    total_triggers += 1

        print(f"\n  {task_type} (n={len(subset)}, total_triggers={total_triggers}):")
        if total_triggers == 0:
            print(f"    No D triggers")
            continue

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
            pct = count / total_triggers * 100
            print(f"    {reason:<40s} {count:>5d} ({pct:>5.1f}%)")


if __name__ == "__main__":
    run_batch(n=100)

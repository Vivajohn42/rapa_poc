"""Neural C vs Deterministic C: Head-to-Head Evaluation.

Tests the hypothesis that a BFS-trained ActionValueNet outperforms
Manhattan scoring on obstacle-rich grids.

5 variants × 6 complexity levels × N episodes (default 100):

  det_abcd        — Deterministic A + B + Manhattan C + D
  neural_c        — Det A + B + NeuralC (70% neural, 30% manhattan) + D
  planner_bc      — Det A + B + Manhattan C + PlannerBC + D
  neural_c_planner — Det A + B + NeuralC + PlannerBC + D
  ab_only         — Det A + B only (random valid action)

C uses the TRUE goal (known target) to isolate the scoring question.

Usage:
    python eval/run_neural_vs_deterministic.py --n 100
    python eval/run_neural_vs_deterministic.py --n 50 --quick
"""
import sys
import csv
import random
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from env.gridworld import GridWorld, GoalDef, HintCellDef
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.neural_agent_c import NeuralAgentC
from models.action_value_net import ActionValueNet
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA

from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    compare_variants,
    format_comparison,
    mean,
)

ACTIONS = ("up", "down", "left", "right")


# ── Complexity Levels ──────────────────────────────────────────────────

@dataclass
class ComplexityLevel:
    name: str
    width: int
    height: int
    goals: List[GoalDef]
    hint_cells: List[HintCellDef]
    n_random_obstacles: int
    dynamic_obstacles: bool
    max_steps: int


def _make_levels(quick: bool = False) -> List[ComplexityLevel]:
    levels = []

    # 5x5, 2 goals, no random obstacles (baseline — expect parity)
    levels.append(ComplexityLevel(
        name="5x5_2g", width=5, height=5,
        goals=[GoalDef("A", (4, 4)), GoalDef("B", (4, 0))],
        hint_cells=[HintCellDef(pos=(0, 4))],
        n_random_obstacles=0, dynamic_obstacles=False, max_steps=50,
    ))

    # 10x10, 2 goals, 8 obstacles
    levels.append(ComplexityLevel(
        name="10x10_2g", width=10, height=10,
        goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
        hint_cells=[HintCellDef(pos=(0, 9))],
        n_random_obstacles=8, dynamic_obstacles=False, max_steps=100,
    ))

    if not quick:
        # 15x15, 2 goals, 20 obstacles (key test — expect neural advantage)
        levels.append(ComplexityLevel(
            name="15x15_2g", width=15, height=15,
            goals=[GoalDef("A", (14, 14)), GoalDef("B", (14, 0))],
            hint_cells=[HintCellDef(pos=(0, 14))],
            n_random_obstacles=20, dynamic_obstacles=False, max_steps=200,
        ))

        # 10x10, dynamic obstacles
        levels.append(ComplexityLevel(
            name="10x10_dyn", width=10, height=10,
            goals=[GoalDef("A", (9, 9)), GoalDef("B", (9, 0))],
            hint_cells=[HintCellDef(pos=(0, 9))],
            n_random_obstacles=5, dynamic_obstacles=True, max_steps=100,
        ))

        # 15x15, 4 goals, 20 obstacles (hardest — expect greatest advantage)
        levels.append(ComplexityLevel(
            name="15x15_4g", width=15, height=15,
            goals=[
                GoalDef("A", (14, 14)), GoalDef("B", (14, 0)),
                GoalDef("C", (0, 14)), GoalDef("D", (7, 7)),
            ],
            hint_cells=[
                HintCellDef(pos=(0, 7), group_a=["A", "B"], group_b=["C", "D"]),
                HintCellDef(pos=(7, 0), group_a=["A", "C"], group_b=["B", "D"]),
            ],
            n_random_obstacles=20, dynamic_obstacles=False, max_steps=200,
        ))

    return levels


# ── Shared Setup ───────────────────────────────────────────────────────

def _load_value_net() -> ActionValueNet:
    """Load the trained ActionValueNet checkpoint."""
    ckpt_path = Path("train/checkpoints/action_value_net.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"ActionValueNet checkpoint not found at {ckpt_path}. "
            f"Run: python -m train.train_c first."
        )
    net = ActionValueNet()
    net.load_state_dict(torch.load(ckpt_path, weights_only=True))
    net.eval()
    return net


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
    valid = [a for a in ACTIONS
             if predict_next_fn(zA, a).agent_pos != zA.agent_pos]
    return rng.choice(valid) if valid else rng.choice(ACTIONS)


# ── Episode Result ─────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    level: str
    variant: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float


# ── Episode Runner ─────────────────────────────────────────────────────

def run_episode(
    level: ComplexityLevel,
    variant: str,
    value_net: Optional[ActionValueNet],
    seed: int = 0,
) -> EpisodeResult:
    """Run one episode with given variant and complexity level."""
    rng = random.Random(seed)

    env = GridWorld(
        width=level.width, height=level.height,
        seed=seed, goals=level.goals, hint_cells=level.hint_cells,
        obstacles=[(2, 2)] if level.width >= 5 else [],
        n_random_obstacles=level.n_random_obstacles,
        dynamic_obstacles=level.dynamic_obstacles,
    )
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos
    goal_map = env.goal_positions

    # Set up variant-specific agents
    C = None
    D = None
    router = None
    planner = None
    zC = ZC(goal_mode="seek", memory={})

    if variant == "det_abcd":
        C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
        D = AgentD()
        router = _make_router()

    elif variant == "neural_c":
        C = NeuralAgentC(
            goal=GoalSpec(mode="seek", target=known_target),
            value_net=value_net,
            alpha=0.7,
            anti_stay_penalty=1.1,
        )
        D = AgentD()
        router = _make_router()

    elif variant == "planner_bc":
        C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
        D = AgentD()
        router = _make_router()
        try:
            from agents.planner_bc import PlannerBC
            planner = PlannerBC(predict_next_fn=B.predict_next, rollout_depth=5, beam_width=3)
        except ImportError:
            pass

    elif variant == "neural_c_planner":
        C = NeuralAgentC(
            goal=GoalSpec(mode="seek", target=known_target),
            value_net=value_net,
            alpha=0.7,
            anti_stay_penalty=1.1,
        )
        D = AgentD()
        router = _make_router()
        try:
            from agents.planner_bc import PlannerBC
            planner = PlannerBC(predict_next_fn=B.predict_next, rollout_depth=5, beam_width=3)
        except ImportError:
            pass

    # ab_only: no C, D, or router

    stay_count = 0
    total_reward = 0.0
    done = False

    for t in range(level.max_steps):
        zA = A.infer_zA(obs)

        if variant == "ab_only":
            action = _random_action(zA, B.predict_next, rng)
        else:
            # Update target from memory if hint was processed
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])

            action, scored = C.choose_action(
                zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25,
            )
            decision_delta = scored[0][1] - scored[1][1]

            # D logic (router-gated)
            if D is not None and router is not None:
                D.observe_step(t=t, zA=zA, action=action, reward=0.0, done=False)
                activate_d, _ = router.should_activate_d(
                    t=t, last_positions=(zA.agent_pos,),
                    decision_delta=decision_delta,
                )
                if activate_d:
                    zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
                    zC = deconstruct_d_to_c(zC, zD, goal_map=goal_map)

            # Planner logic
            if planner is not None:
                try:
                    plan_result = planner.plan(zA, C.goal.target, "seek")
                    if plan_result and plan_result.recommended_actions:
                        zC.memory["tie_break_preference"] = plan_result.recommended_actions
                except Exception:
                    pass

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        total_reward += reward
        obs = obs_next

        if done:
            break

    steps = (t + 1) if done else level.max_steps
    stay_rate = stay_count / steps if steps > 0 else 0.0

    return EpisodeResult(
        level=level.name,
        variant=variant,
        success=bool(done),
        steps=steps,
        total_reward=round(total_reward, 4),
        stay_rate=round(stay_rate, 4),
    )


# ── Batch Runner ───────────────────────────────────────────────────────

VARIANTS = ["det_abcd", "neural_c", "planner_bc", "neural_c_planner", "ab_only"]


def run_batch(n: int = 100, quick: bool = False):
    """Run the full neural vs deterministic comparison."""
    print("=" * 80)
    print("  Neural C vs Deterministic C — Head-to-Head Evaluation")
    print("=" * 80)

    # Load value net once
    print("Loading ActionValueNet checkpoint...")
    value_net = _load_value_net()
    n_params = sum(p.numel() for p in value_net.parameters())
    print(f"  ActionValueNet: {n_params:,} params")

    levels = _make_levels(quick=quick)
    variants = VARIANTS

    # Check if PlannerBC is available
    try:
        from agents.planner_bc import PlannerBC
        has_planner = True
        print("  PlannerBC: available")
    except ImportError:
        has_planner = False
        variants = [v for v in variants if "planner" not in v]
        print("  PlannerBC: not available (skipping planner variants)")

    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/neural_vs_det_{run_id}.csv"

    results: List[EpisodeResult] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = len(levels) * len(variants) * n
    print(f"\n  {len(levels)} levels × {len(variants)} variants × {n} episodes = {total} runs")

    if use_tqdm:
        pbar = tqdm(total=total, desc="neural_vs_det")

    for level in levels:
        for variant in variants:
            for i in range(n):
                r = run_episode(level, variant, value_net, seed=i)
                results.append(r)
                if use_tqdm:
                    pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "variant", "success", "steps", "total_reward", "stay_rate"])
        for r in results:
            w.writerow([r.level, r.variant, r.success, r.steps, r.total_reward, r.stay_rate])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_results_table(results, levels, variants)
    _print_statistical_tests(results, levels, variants)

    return results


def _print_results_table(results, levels, variants):
    """Print SR table: level × variant."""
    print(f"\n{'=' * 90}")
    print(f"  SUCCESS RATE by Level × Variant")
    print(f"{'=' * 90}")

    # Header
    print(f"  {'level':<14s}", end="")
    for v in variants:
        print(f" {v:>18s}", end="")
    print()
    print(f"  {'-' * 14}", end="")
    for _ in variants:
        print(f" {'-' * 18}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<14s}", end="")
        for v in variants:
            sub = [r for r in results if r.level == level.name and r.variant == v]
            if sub:
                sr = sum(1 for r in sub if r.success) / len(sub)
                _, lo, hi = confidence_interval_proportion(
                    sum(1 for r in sub if r.success), len(sub),
                )
                print(f" {sr:>5.1%}[{lo:.2f},{hi:.2f}]", end="")
            else:
                print(f" {'N/A':>18s}", end="")
        print()

    # Steps table
    print(f"\n  MEAN STEPS by Level × Variant")
    print(f"  {'level':<14s}", end="")
    for v in variants:
        print(f" {v:>18s}", end="")
    print()
    print(f"  {'-' * 14}", end="")
    for _ in variants:
        print(f" {'-' * 18}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<14s}", end="")
        for v in variants:
            sub = [r for r in results if r.level == level.name and r.variant == v]
            if sub:
                steps_m = mean([float(r.steps) for r in sub])
                _, ci_lo, ci_hi = confidence_interval_95([float(r.steps) for r in sub])
                print(f" {steps_m:>6.1f}±{(ci_hi - ci_lo) / 2:>4.1f}", end="  ")
            else:
                print(f" {'N/A':>18s}", end="")
        print()

    # Stay rate table
    print(f"\n  STAY RATE by Level × Variant")
    print(f"  {'level':<14s}", end="")
    for v in variants:
        print(f" {v:>18s}", end="")
    print()
    print(f"  {'-' * 14}", end="")
    for _ in variants:
        print(f" {'-' * 18}", end="")
    print()

    for level in levels:
        print(f"  {level.name:<14s}", end="")
        for v in variants:
            sub = [r for r in results if r.level == level.name and r.variant == v]
            if sub:
                sr_m = mean([r.stay_rate for r in sub])
                print(f" {sr_m:>18.1%}", end="")
            else:
                print(f" {'N/A':>18s}", end="")
        print()


def _print_statistical_tests(results, levels, variants):
    """Run Mann-Whitney U / Cohen's d on key comparisons."""
    print(f"\n{'=' * 90}")
    print(f"  STATISTICAL COMPARISONS (neural_c vs det_abcd)")
    print(f"{'=' * 90}")

    for level in levels:
        det_sub = [r for r in results if r.level == level.name and r.variant == "det_abcd"]
        neural_sub = [r for r in results if r.level == level.name and r.variant == "neural_c"]

        if not det_sub or not neural_sub:
            continue

        det_sr_vals = [1.0 if r.success else 0.0 for r in det_sub]
        neural_sr_vals = [1.0 if r.success else 0.0 for r in neural_sub]

        report = compare_variants(
            "neural_c", neural_sr_vals,
            "det_abcd", det_sr_vals,
            "success_rate", is_proportion=True,
        )
        print(f"\n  {level.name}:")
        print(f"    {format_comparison(report)}")

        # Also compare steps for successful episodes
        det_steps = [float(r.steps) for r in det_sub if r.success]
        neural_steps = [float(r.steps) for r in neural_sub if r.success]

        if det_steps and neural_steps:
            steps_report = compare_variants(
                "neural_c", neural_steps,
                "det_abcd", det_steps,
                "steps_successful",
            )
            print(f"    {format_comparison(steps_report)}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural C vs Deterministic C evaluation",
    )
    parser.add_argument("--n", type=int, default=100, help="Episodes per cell")
    parser.add_argument("--quick", action="store_true", help="Skip large levels")
    args = parser.parse_args()

    run_batch(n=args.n, quick=args.quick)


if __name__ == "__main__":
    main()

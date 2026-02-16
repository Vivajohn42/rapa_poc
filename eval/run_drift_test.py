"""
Stufe 4: Drift-Test & Deconstruction Validation

DEF Claim: D's narrative drifts over time without stabilization.
Deconstruction (D->C feedback) acts as a stabilizer that prevents
unconstrained narrative drift.

Variants:
  d_always_no_decon   — D called every step, NO deconstruction feedback
  d_always_decon_k5   — D called every step, deconstruction every 5 steps
  d_always_decon_k10  — D called every step, deconstruction every 10 steps
  d_always_decon_k1   — D called every step, deconstruction every step
  d_routed            — D called by router (current production approach)

DEF Predictions:
  1. Without deconstruction, tag flip rate increases over time (drift)
  2. With deconstruction, tag stability remains higher
  3. More frequent deconstruction = more stable
  4. d_routed achieves good stability with fewer D calls (efficiency)
"""

import sys
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Optional, Dict, Any

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZD

from eval.drift_metrics import (
    compute_drift_series,
    drift_summary,
    windowed_tag_stability,
)
from eval.stats import (
    confidence_interval_95,
    mean,
)


VARIANTS = [
    "d_always_no_decon",
    "d_always_decon_k1",
    "d_always_decon_k5",
    "d_always_decon_k10",
    "d_routed",
]


@dataclass
class DriftResult:
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    d_calls: int
    decon_calls: int
    mean_tag_flip_rate: float
    mean_narrative_similarity: float
    total_grounding_violations: int
    mean_narrative_length: float
    tag_stability_w5: float
    drift_trend: float


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
    variant: str,
    goal_mode: str = "seek",
    max_steps: int = 200,
    grid_size: int = 10,
    seed: Optional[int] = None,
) -> DriftResult:
    """
    Run a single drift-test episode.

    Uses a larger grid (10x10) with more steps to give D time to drift.
    The agent always knows the true goal (isolating drift from navigation).
    """
    env = GridWorld(
        width=grid_size, height=grid_size, seed=seed,
        n_random_obstacles=max(1, grid_size // 2),
    )
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode=goal_mode, target=known_target), anti_stay_penalty=1.1)
    D = AgentD()
    zC = ZC(goal_mode=goal_mode, memory={})

    router = _make_router() if variant == "d_routed" else None

    # Parse variant parameters
    decon_k = _parse_decon_k(variant)

    total_reward = 0.0
    done = False
    d_calls = 0
    decon_calls = 0
    zd_series: List[ZD] = []

    for t in range(max_steps):
        zA = A.infer_zA(obs)

        if "target" in zC.memory:
            C.goal.target = tuple(zC.memory["target"])

        action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
        decision_delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)
        total_reward += reward

        D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)
        obs = obs_next

        # ── D invocation logic ────────────────────────────────────
        call_d = False
        call_decon = False

        if variant == "d_routed":
            activate, reason = router.should_activate_d(
                t=t,
                last_positions=(zA_next.agent_pos,),
                decision_delta=decision_delta,
            )
            if activate:
                call_d = True
                call_decon = True  # routed always deconstructs

        elif variant.startswith("d_always"):
            call_d = True
            if decon_k is not None and decon_k > 0 and (t + 1) % decon_k == 0:
                call_decon = True

        if call_d:
            zD = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
            zd_series.append(zD)
            d_calls += 1

        if call_decon and zd_series:
            zC = deconstruct_d_to_c(zC, zd_series[-1])
            decon_calls += 1

        if done:
            break

    steps = (t + 1) if done else max_steps

    # Compute drift metrics
    ds = drift_summary(zd_series)

    return DriftResult(
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        d_calls=d_calls,
        decon_calls=decon_calls,
        mean_tag_flip_rate=ds["mean_tag_flip_rate"],
        mean_narrative_similarity=ds["mean_narrative_similarity"],
        total_grounding_violations=ds["total_grounding_violations"],
        mean_narrative_length=ds["mean_narrative_length"],
        tag_stability_w5=ds["tag_stability_w5"],
        drift_trend=ds["drift_trend"],
    )


def _parse_decon_k(variant: str) -> Optional[int]:
    """Parse deconstruction frequency from variant name."""
    if variant == "d_always_no_decon":
        return None
    elif variant == "d_always_decon_k1":
        return 1
    elif variant == "d_always_decon_k5":
        return 5
    elif variant == "d_always_decon_k10":
        return 10
    elif variant == "d_routed":
        return None  # handled by router
    return None


# ── Batch Runner ──────────────────────────────────────────────────────

def run_batch(n: int = 50, max_steps: int = 200, grid_size: int = 10):
    """Run drift test across all variants."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/drift_test_{run_id}.csv"

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    results: List[DriftResult] = []
    total = len(VARIANTS) * n

    if use_tqdm:
        pbar = tqdm(total=total, desc="drift_test")

    for variant in VARIANTS:
        for i in range(n):
            r = run_episode(variant, "seek", max_steps=max_steps,
                            grid_size=grid_size, seed=i)
            results.append(r)
            if use_tqdm:
                pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "goal_mode", "success", "steps", "total_reward",
            "d_calls", "decon_calls",
            "mean_tag_flip_rate", "mean_narrative_similarity",
            "total_grounding_violations", "mean_narrative_length",
            "tag_stability_w5", "drift_trend",
        ])
        for r in results:
            w.writerow([
                r.variant, r.goal_mode, r.success, r.steps,
                f"{r.total_reward:.4f}", r.d_calls, r.decon_calls,
                f"{r.mean_tag_flip_rate:.4f}", f"{r.mean_narrative_similarity:.4f}",
                r.total_grounding_violations, f"{r.mean_narrative_length:.1f}",
                f"{r.tag_stability_w5:.4f}", f"{r.drift_trend:.4f}",
            ])

    print(f"\nWrote {len(results)} episodes to: {csv_path}")

    _print_drift_table(results)
    _print_def_predictions(results)
    _print_temporal_profile(n, max_steps, grid_size)

    return results


def _print_drift_table(results: List[DriftResult]):
    """Print drift metrics per variant."""
    print(f"\n{'='*100}")
    print(f"  DRIFT TEST -- D Narrative Stability per Variant")
    print(f"{'='*100}")
    print(
        f"  {'variant':<24s} {'d_calls':>7s} {'decon':>5s} "
        f"{'flip_rate':>9s} {'narr_sim':>8s} {'tag_stab':>8s} "
        f"{'drift_tr':>8s} {'sr':>5s} {'steps':>6s}"
    )
    print(
        f"  {'-'*24} {'-'*7} {'-'*5} "
        f"{'-'*9} {'-'*8} {'-'*8} "
        f"{'-'*8} {'-'*5} {'-'*6}"
    )

    for variant in VARIANTS:
        subset = [r for r in results if r.variant == variant]
        if not subset:
            continue

        d_calls_m = mean([float(r.d_calls) for r in subset])
        decon_m = mean([float(r.decon_calls) for r in subset])
        flip_m = mean([r.mean_tag_flip_rate for r in subset])
        sim_m = mean([r.mean_narrative_similarity for r in subset])
        stab_m = mean([r.tag_stability_w5 for r in subset])
        trend_m = mean([r.drift_trend for r in subset])
        sr = sum(1 for r in subset if r.success) / len(subset)
        steps_m = mean([float(r.steps) for r in subset])

        print(
            f"  {variant:<24s} {d_calls_m:>7.1f} {decon_m:>5.1f} "
            f"{flip_m:>9.4f} {sim_m:>8.4f} {stab_m:>8.4f} "
            f"{trend_m:>+8.4f} {sr:>5.3f} {steps_m:>6.1f}"
        )


def _print_def_predictions(results: List[DriftResult]):
    """Validate DEF predictions about drift and deconstruction."""
    print(f"\n{'='*100}")
    print(f"  DEF PREDICTIONS -- Drift & Deconstruction")
    print(f"{'='*100}")

    def avg_metric(variant, field):
        subset = [r for r in results if r.variant == variant]
        if not subset:
            return 0.0
        return mean([getattr(r, field) for r in subset])

    # Prediction 1: Without decon, drift trend is positive (drift increases)
    no_decon_trend = avg_metric("d_always_no_decon", "drift_trend")
    decon_k1_trend = avg_metric("d_always_decon_k1", "drift_trend")
    print(f"\n  1. 'Without deconstruction, drift increases over time'")
    print(f"     no_decon drift_trend:  {no_decon_trend:+.4f}")
    print(f"     decon_k1 drift_trend:  {decon_k1_trend:+.4f}")
    if no_decon_trend > decon_k1_trend:
        print(f"     [PASS] No-decon drifts more ({no_decon_trend:+.4f} > {decon_k1_trend:+.4f})")
    else:
        print(f"     [PARTIAL] Drift trend difference small or reversed")

    # Prediction 2: Deconstruction stabilizes tags
    no_decon_flip = avg_metric("d_always_no_decon", "mean_tag_flip_rate")
    decon_k1_flip = avg_metric("d_always_decon_k1", "mean_tag_flip_rate")
    decon_k5_flip = avg_metric("d_always_decon_k5", "mean_tag_flip_rate")
    print(f"\n  2. 'Deconstruction reduces tag flip rate'")
    print(f"     no_decon flip_rate:  {no_decon_flip:.4f}")
    print(f"     decon_k1 flip_rate:  {decon_k1_flip:.4f}")
    print(f"     decon_k5 flip_rate:  {decon_k5_flip:.4f}")
    if no_decon_flip >= decon_k1_flip:
        print(f"     [PASS] Deconstruction reduces flip rate")
    else:
        print(f"     [PARTIAL] Flip rates similar")

    # Prediction 3: More frequent decon = more stable
    decon_k10_stab = avg_metric("d_always_decon_k10", "tag_stability_w5")
    decon_k5_stab = avg_metric("d_always_decon_k5", "tag_stability_w5")
    decon_k1_stab = avg_metric("d_always_decon_k1", "tag_stability_w5")
    print(f"\n  3. 'More frequent deconstruction = more stable tags'")
    print(f"     decon_k10 tag_stability: {decon_k10_stab:.4f}")
    print(f"     decon_k5 tag_stability:  {decon_k5_stab:.4f}")
    print(f"     decon_k1 tag_stability:  {decon_k1_stab:.4f}")
    if decon_k1_stab >= decon_k5_stab >= decon_k10_stab:
        print(f"     [PASS] Stability increases with decon frequency")
    elif decon_k1_stab >= decon_k10_stab:
        print(f"     [PARTIAL] k1 >= k10 but k5 out of order")
    else:
        print(f"     [WARN] Expected k1 >= k5 >= k10")

    # Prediction 4: Routed D achieves good efficiency
    routed_calls = avg_metric("d_routed", "d_calls")
    always_calls = avg_metric("d_always_decon_k1", "d_calls")
    routed_stab = avg_metric("d_routed", "tag_stability_w5")
    routed_sr = sum(1 for r in results if r.variant == "d_routed" and r.success) / max(1, sum(1 for r in results if r.variant == "d_routed"))
    always_sr = sum(1 for r in results if r.variant == "d_always_decon_k1" and r.success) / max(1, sum(1 for r in results if r.variant == "d_always_decon_k1"))
    print(f"\n  4. 'Routed D achieves good stability with fewer calls'")
    print(f"     d_routed:        {routed_calls:.1f} D-calls, stability={routed_stab:.4f}, SR={routed_sr:.3f}")
    print(f"     d_always_decon_k1: {always_calls:.1f} D-calls, stability={decon_k1_stab:.4f}, SR={always_sr:.3f}")
    if routed_calls < always_calls:
        efficiency = (1 - routed_calls / always_calls) * 100
        print(f"     [PASS] Router uses {efficiency:.0f}% fewer D calls")
    else:
        print(f"     [INFO] Router called D as often as always-D")


def _print_temporal_profile(n: int, max_steps: int, grid_size: int):
    """Run one detailed episode per variant and show temporal drift profile."""
    print(f"\n{'='*100}")
    print(f"  TEMPORAL DRIFT PROFILE (single episode, seed=0, {grid_size}x{grid_size}, {max_steps} steps)")
    print(f"{'='*100}")

    for variant in ["d_always_no_decon", "d_always_decon_k5", "d_routed"]:
        env = GridWorld(
            width=grid_size, height=grid_size, seed=0,
            n_random_obstacles=max(1, grid_size // 2),
        )
        obs = env.reset()
        A = AgentA()
        B = AgentB()
        known_target = env.true_goal_pos
        C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
        D = AgentD()
        zC = ZC(goal_mode="seek", memory={})
        router = _make_router() if variant == "d_routed" else None
        decon_k = _parse_decon_k(variant)

        zd_series: List[ZD] = []
        d_call_steps: List[int] = []

        for t in range(max_steps):
            zA = A.infer_zA(obs)
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])

            action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            delta = scored[0][1] - scored[1][1]

            obs_next, reward, done = env.step(action)
            zA_next = A.infer_zA(obs_next)
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)
            obs = obs_next

            call_d = False
            call_decon = False

            if variant == "d_routed" and router:
                activate, _ = router.should_activate_d(
                    t=t, last_positions=(zA_next.agent_pos,), decision_delta=delta,
                )
                if activate:
                    call_d = True
                    call_decon = True
            elif variant.startswith("d_always"):
                call_d = True
                if decon_k and decon_k > 0 and (t + 1) % decon_k == 0:
                    call_decon = True

            if call_d:
                zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
                zd_series.append(zD)
                d_call_steps.append(t)

            if call_decon and zd_series:
                zC = deconstruct_d_to_c(zC, zd_series[-1])

            if done:
                break

        # Show temporal profile in 5-bucket windows
        print(f"\n  [{variant}] D calls={len(zd_series)}, steps={t+1}")
        if len(zd_series) < 2:
            print(f"    (too few D calls for temporal analysis)")
            continue

        series = compute_drift_series(zd_series)
        flip_rates = series["tag_flip_rates"]
        sims = series["narrative_similarities"]

        # Divide into 5 equal buckets
        n_buckets = min(5, len(flip_rates))
        bucket_size = len(flip_rates) // n_buckets if n_buckets > 0 else 1

        print(f"    {'bucket':>8s} {'flip_rate':>10s} {'narr_sim':>10s} {'n_samples':>10s}")
        for b in range(n_buckets):
            start = b * bucket_size
            end = start + bucket_size if b < n_buckets - 1 else len(flip_rates)
            bucket_flips = flip_rates[start:end]
            bucket_sims = sims[start:end]
            if bucket_flips:
                print(
                    f"    {b+1:>8d} {mean(bucket_flips):>10.4f} "
                    f"{mean(bucket_sims):>10.4f} {len(bucket_flips):>10d}"
                )


if __name__ == "__main__":
    run_batch(n=50, max_steps=200, grid_size=10)

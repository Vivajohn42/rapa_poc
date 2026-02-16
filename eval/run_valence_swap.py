"""
Stufe 1: Valence-Swap Test

DEF Claim: "Same world, inverted goal/reward → zA and zB remain stable;
            zC changes dramatically."

Two complementary test designs:

Part 1 — STATIC STATE ANALYSIS:
  Sample grid positions, compute seek/avoid scores at each.
  Verify that A (observation) and B (predictions) are mode-independent,
  while C (scores) invert.

Part 2 — FORCED-TRAJECTORY PAIRED COMPARISON:
  Run a single "reference" episode. Replay the exact same action sequence
  under SEEK and AVOID scoring. Because positions are identical (same actions),
  we can compare scores step-by-step.

Part 3 — MID-EPISODE VALENCE SWAP:
  Run an episode that switches mode mid-flight. Verify A+B stay stable
  while C inverts immediately at the swap point.
"""

import sys
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from state.schema import ZA

from eval.metrics import (
    ACTIONS,
    score_negation_error,
    ranking_from_scored,
    ranking_inversion_score,
    ab_identity_check,
    spearman_corr,
    compute_symmetry_metrics,
)
from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    mean,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1: Static State Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StaticStateResult:
    """Metrics for a single grid state evaluated under both modes."""
    pos: Tuple[int, int]
    target: Tuple[int, int]
    score_negation_error: float
    ranking_inversion_score: float
    top_action_flipped: bool
    ab_prediction_valid: bool
    seek_scores: Dict[str, float] = field(default_factory=dict)
    avoid_scores: Dict[str, float] = field(default_factory=dict)


def run_static_analysis(
    n_positions: int = 100,
    targets: Optional[List[Tuple[int, int]]] = None,
) -> List[StaticStateResult]:
    """
    For each sampled position × target, compute seek/avoid scores and verify:
      - B predictions are mode-independent (guaranteed by design: B has no mode)
      - Seek and avoid scores are negations of each other
      - Action rankings are inverted
    """
    import random
    rng = random.Random(42)

    if targets is None:
        targets = [(4, 4), (4, 0)]  # Goal A and Goal B

    B = AgentB()
    results = []

    for _ in range(n_positions):
        pos = (rng.randint(0, 4), rng.randint(0, 4))
        if pos in [(2, 2)]:
            continue

        for target in targets:
            zA = ZA(width=5, height=5, agent_pos=pos, goal_pos=(-1, -1), obstacles=[(2, 2)])

            C_seek = AgentC(goal=GoalSpec(mode="seek", target=target), anti_stay_penalty=1.1)
            C_avoid = AgentC(goal=GoalSpec(mode="avoid", target=target), anti_stay_penalty=1.1)

            m = compute_symmetry_metrics(
                zA, B.predict_next, C_seek.score_action, C_avoid.score_action
            )

            # Collect raw scores for CSV
            seek_scores = {}
            avoid_scores = {}
            for a in ACTIONS:
                zA_next = B.predict_next(zA, a)
                seek_scores[a] = C_seek.score_action(zA, zA_next)
                avoid_scores[a] = C_avoid.score_action(zA, zA_next)

            results.append(StaticStateResult(
                pos=pos,
                target=target,
                score_negation_error=m["score_negation_error"],
                ranking_inversion_score=m["ranking_inversion_score"],
                top_action_flipped=m["top_action_flipped"],
                ab_prediction_valid=m["ab_prediction_valid"],
                seek_scores=seek_scores,
                avoid_scores=avoid_scores,
            ))

    return results


def print_static_analysis(results: List[StaticStateResult]):
    """Print aggregate static analysis results."""
    n = len(results)
    neg_errs = [r.score_negation_error for r in results]
    inv_scores = [r.ranking_inversion_score for r in results]
    flipped = sum(1 for r in results if r.top_action_flipped)
    ab_valid = sum(1 for r in results if r.ab_prediction_valid)

    ne_m, ne_lo, ne_hi = confidence_interval_95(neg_errs)
    inv_m, inv_lo, inv_hi = confidence_interval_95(inv_scores)
    flip_p, flip_lo, flip_hi = confidence_interval_proportion(flipped, n)
    ab_p, ab_lo, ab_hi = confidence_interval_proportion(ab_valid, n)

    # Compute Spearman across all scores
    all_seek = []
    all_avoid = []
    for r in results:
        for a in ACTIONS:
            all_seek.append(r.seek_scores[a])
            all_avoid.append(r.avoid_scores[a])
    sp = spearman_corr(all_seek, all_avoid)

    print(f"\n{'='*76}")
    print(f"  PART 1: Static State Analysis")
    print(f"  n={n} state-target pairs")
    print(f"{'='*76}")

    print(f"\n  --- Stream A+B Stability ---")
    print(f"  B predictions valid:     {ab_p:.4f} [{ab_lo:.4f}, {ab_hi:.4f}]  expect 1.0")
    print(f"  (B is mode-independent by design: no mode parameter)")

    print(f"\n  --- Stream C Divergence ---")
    print(f"  Score negation error:    {ne_m:.4f} [{ne_lo:.4f}, {ne_hi:.4f}]  expect 0.0")
    print(f"  Ranking inversion score: {inv_m:.4f} [{inv_lo:.4f}, {inv_hi:.4f}]  expect 1.0")
    print(f"  Top action flipped:      {flip_p:.4f} [{flip_lo:.4f}, {flip_hi:.4f}]  expect ~1.0")
    print(f"  Spearman(seek, avoid):   {sp:.4f}  expect -1.0")

    # Diagnosis: why negation error > 0?
    anti_stay_positions = sum(
        1 for r in results
        for a in ACTIONS
        if r.seek_scores[a] + r.avoid_scores[a] != 0.0
    )
    total_scores = n * len(ACTIONS)
    print(f"\n  --- Diagnosis ---")
    print(f"  Non-zero negation pairs: {anti_stay_positions}/{total_scores}")
    print(f"  (Non-zero is caused by anti_stay_penalty which subtracts")
    print(f"   in both modes instead of negating. This is by design.)")

    # Verdict
    print(f"\n  --- Verdict ---")
    if ab_p == 1.0:
        print(f"  [PASS] Stream B is perfectly mode-independent")
    else:
        print(f"  [FAIL] Stream B has mode-dependent predictions")

    if flip_p >= 0.85:
        print(f"  [PASS] Stream C strongly diverges (top_flip={flip_p:.3f})")
    elif flip_p >= 0.5:
        print(f"  [PARTIAL] Stream C partially diverges (top_flip={flip_p:.3f})")
    else:
        print(f"  [FAIL] Stream C does not diverge (top_flip={flip_p:.3f})")

    if sp < -0.5:
        print(f"  [PASS] Seek/Avoid scores anti-correlated (rho={sp:.3f})")
    else:
        print(f"  [WARN] Seek/Avoid scores not strongly anti-correlated (rho={sp:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2: Forced-Trajectory Paired Comparison
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ForcedTrajectoryResult:
    """Result of replaying a fixed action sequence under both modes."""
    seed: int
    n_steps: int
    # Per-step metrics (aggregated)
    mean_negation_error: float = 0.0
    mean_inversion_score: float = 0.0
    action_flip_rate: float = 0.0
    spearman_corr: float = 0.0
    zB_identity_rate: float = 1.0  # should always be 1.0


def run_forced_trajectory(seed: int, target: Tuple[int, int] = (4, 4), max_steps: int = 30) -> ForcedTrajectoryResult:
    """
    Run a SEEK episode to get a reference trajectory.
    Then replay the EXACT same actions and compute AVOID scores at each step.
    Positions are guaranteed identical because actions are the same.
    """
    env = GridWorld(seed=seed)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    C_seek = AgentC(goal=GoalSpec(mode="seek", target=target), anti_stay_penalty=1.1)

    # Phase 1: Run seek episode, record actions
    actions_taken = []
    for t in range(max_steps):
        zA = A.infer_zA(obs)
        _, scored = C_seek.choose_action(zA, B.predict_next)
        action = scored[0][0]
        actions_taken.append(action)
        obs, reward, done = env.step(action)
        if done:
            break

    # Phase 2: Replay with both scorers
    env_replay = GridWorld(seed=seed)
    obs_replay = env_replay.reset()

    C_seek2 = AgentC(goal=GoalSpec(mode="seek", target=target), anti_stay_penalty=1.1)
    C_avoid = AgentC(goal=GoalSpec(mode="avoid", target=target), anti_stay_penalty=1.1)

    neg_errors = []
    inv_scores = []
    flip_count = 0
    zB_match = 0
    all_seek_s = []
    all_avoid_s = []

    for t, action in enumerate(actions_taken):
        zA = A.infer_zA(obs_replay)

        # Score all actions under both modes
        scored_seek = []
        scored_avoid = []
        b_preds_seek = {}
        b_preds_avoid = {}

        for a in ACTIONS:
            zA_next = B.predict_next(zA, a)
            b_preds_seek[a] = zA_next.agent_pos

            s_seek = C_seek2.score_action(zA, zA_next)
            s_avoid = C_avoid.score_action(zA, zA_next)
            scored_seek.append((a, s_seek))
            scored_avoid.append((a, s_avoid))

            all_seek_s.append(s_seek)
            all_avoid_s.append(s_avoid)

        # B predictions are identical (same function, same input)
        # Verify explicitly
        for a in ACTIONS:
            zA_next2 = B.predict_next(zA, a)
            b_preds_avoid[a] = zA_next2.agent_pos
        if b_preds_seek == b_preds_avoid:
            zB_match += 1

        neg_errors.append(score_negation_error(scored_seek, scored_avoid))

        rank_s = ranking_from_scored(sorted(scored_seek, key=lambda x: x[1], reverse=True))
        rank_a = ranking_from_scored(sorted(scored_avoid, key=lambda x: x[1], reverse=True))
        inv_scores.append(ranking_inversion_score(rank_s, rank_a))

        top_seek = sorted(scored_seek, key=lambda x: x[1], reverse=True)[0][0]
        top_avoid = sorted(scored_avoid, key=lambda x: x[1], reverse=True)[0][0]
        if top_seek != top_avoid:
            flip_count += 1

        obs_replay, _, done = env_replay.step(action)
        if done:
            break

    n = len(actions_taken)
    sp = spearman_corr(all_seek_s, all_avoid_s) if len(all_seek_s) >= 4 else 0.0

    return ForcedTrajectoryResult(
        seed=seed,
        n_steps=n,
        mean_negation_error=mean(neg_errors),
        mean_inversion_score=mean(inv_scores),
        action_flip_rate=flip_count / n if n > 0 else 0.0,
        spearman_corr=sp,
        zB_identity_rate=zB_match / n if n > 0 else 0.0,
    )


def run_forced_trajectory_batch(n_episodes: int = 100, target: Tuple[int, int] = (4, 4), max_steps: int = 30):
    """Run forced-trajectory paired comparisons."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/valence_swap_forced_{run_id}.csv"

    results: List[ForcedTrajectoryResult] = []
    for i in range(n_episodes):
        results.append(run_forced_trajectory(seed=i, target=target, max_steps=max_steps))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "n_steps", "mean_negation_error", "mean_inversion_score",
            "action_flip_rate", "spearman_corr", "zB_identity_rate",
        ])
        for r in results:
            w.writerow([
                r.seed, r.n_steps,
                f"{r.mean_negation_error:.4f}", f"{r.mean_inversion_score:.4f}",
                f"{r.action_flip_rate:.4f}", f"{r.spearman_corr:.4f}",
                f"{r.zB_identity_rate:.4f}",
            ])

    print(f"\nWrote {len(results)} forced-trajectory episodes to: {csv_path}")

    # Aggregates
    ne_vals = [r.mean_negation_error for r in results]
    inv_vals = [r.mean_inversion_score for r in results]
    flip_vals = [r.action_flip_rate for r in results]
    sp_vals = [r.spearman_corr for r in results]
    zB_vals = [r.zB_identity_rate for r in results]

    ne_m, ne_lo, ne_hi = confidence_interval_95(ne_vals)
    inv_m, inv_lo, inv_hi = confidence_interval_95(inv_vals)
    flip_m, flip_lo, flip_hi = confidence_interval_95(flip_vals)
    sp_m, sp_lo, sp_hi = confidence_interval_95(sp_vals)
    zB_m, zB_lo, zB_hi = confidence_interval_95(zB_vals)

    print(f"\n{'='*76}")
    print(f"  PART 2: Forced-Trajectory Paired Comparison")
    print(f"  n={len(results)} episodes (same actions, both scores)")
    print(f"{'='*76}")

    print(f"\n  --- Stream B Stability ---")
    print(f"  zB identity rate:        {zB_m:.4f} [{zB_lo:.4f}, {zB_hi:.4f}]  expect 1.0")

    print(f"\n  --- Stream C Divergence ---")
    print(f"  Score negation error:    {ne_m:.4f} [{ne_lo:.4f}, {ne_hi:.4f}]  expect ~0.0")
    print(f"  Ranking inversion score: {inv_m:.4f} [{inv_lo:.4f}, {inv_hi:.4f}]  expect ~1.0")
    print(f"  Action flip rate:        {flip_m:.4f} [{flip_lo:.4f}, {flip_hi:.4f}]  expect ~1.0")
    print(f"  Spearman(seek, avoid):   {sp_m:.4f} [{sp_lo:.4f}, {sp_hi:.4f}]  expect -1.0")

    # Verdict
    print(f"\n  --- Verdict ---")
    if zB_m == 1.0:
        print(f"  [PASS] Stream B perfectly mode-independent along trajectory")
    else:
        print(f"  [FAIL] Stream B has mode-dependent predictions")

    if flip_m >= 0.75:
        print(f"  [PASS] Stream C strongly diverges (flip={flip_m:.3f})")
    elif flip_m >= 0.5:
        print(f"  [PARTIAL] Stream C partially diverges (flip={flip_m:.3f})")
    else:
        print(f"  [WARN] Low flip rate (flip={flip_m:.3f})")
        print(f"         At boundary/corner positions, seek and avoid may agree")
        print(f"         because all non-wall moves are equally good/bad")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: Mid-Episode Valence Swap
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MidSwapResult:
    """Result of a mid-episode valence swap experiment."""
    seed: int
    swap_at_step: int
    total_steps: int
    swap_pos: Optional[Tuple[int, int]] = None

    # At the swap point
    action_changed_at_swap: bool = False
    negation_error_at_swap: float = 0.0
    inversion_score_at_swap: float = 0.0


def run_mid_swap_episode(
    seed: int, target: Tuple[int, int] = (4, 4),
    swap_at: int = 5, max_steps: int = 30
) -> MidSwapResult:
    """
    Run an episode that starts in SEEK mode and swaps to AVOID at step `swap_at`.
    At the exact swap point, compute scores under both modes for the same state.
    """
    env = GridWorld(seed=seed)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    C_seek = AgentC(goal=GoalSpec(mode="seek", target=target), anti_stay_penalty=1.1)
    C_avoid = AgentC(goal=GoalSpec(mode="avoid", target=target), anti_stay_penalty=1.1)

    result = MidSwapResult(seed=seed, swap_at_step=swap_at, total_steps=0)

    for t in range(max_steps):
        zA = A.infer_zA(obs)
        current_mode = "seek" if t < swap_at else "avoid"
        C_active = C_seek if current_mode == "seek" else C_avoid

        # At swap boundary: compute both modes' scores at this exact state
        if t == swap_at:
            result.swap_pos = zA.agent_pos

            scored_seek = []
            scored_avoid = []
            for a in ACTIONS:
                zA_next = B.predict_next(zA, a)
                scored_seek.append((a, C_seek.score_action(zA, zA_next)))
                scored_avoid.append((a, C_avoid.score_action(zA, zA_next)))

            result.negation_error_at_swap = score_negation_error(scored_seek, scored_avoid)

            rank_s = ranking_from_scored(sorted(scored_seek, key=lambda x: x[1], reverse=True))
            rank_a = ranking_from_scored(sorted(scored_avoid, key=lambda x: x[1], reverse=True))
            result.inversion_score_at_swap = ranking_inversion_score(rank_s, rank_a)

            seek_top = sorted(scored_seek, key=lambda x: x[1], reverse=True)[0][0]
            avoid_top = sorted(scored_avoid, key=lambda x: x[1], reverse=True)[0][0]
            result.action_changed_at_swap = (seek_top != avoid_top)

        # Pick action with current mode
        _, scored = C_active.choose_action(zA, B.predict_next)
        action = scored[0][0]

        obs, reward, done = env.step(action)
        result.total_steps = t + 1
        if done:
            break

    return result


def run_mid_swap_batch(n_episodes: int = 100, swap_at: int = 5, max_steps: int = 30):
    """Run mid-episode valence swap experiments."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/valence_mid_swap_{run_id}.csv"

    results: List[MidSwapResult] = []
    for i in range(n_episodes):
        results.append(run_mid_swap_episode(seed=i, swap_at=swap_at, max_steps=max_steps))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "swap_at_step", "total_steps", "swap_pos",
            "action_changed_at_swap", "negation_error_at_swap", "inversion_score_at_swap",
        ])
        for r in results:
            w.writerow([
                r.seed, r.swap_at_step, r.total_steps, r.swap_pos,
                r.action_changed_at_swap,
                f"{r.negation_error_at_swap:.4f}", f"{r.inversion_score_at_swap:.4f}",
            ])

    print(f"\nWrote {len(results)} mid-swap episodes to: {csv_path}")

    # Aggregates
    action_changed = sum(1 for r in results if r.action_changed_at_swap)
    ne_vals = [r.negation_error_at_swap for r in results]
    inv_vals = [r.inversion_score_at_swap for r in results]

    ne_m, ne_lo, ne_hi = confidence_interval_95(ne_vals)
    inv_m, inv_lo, inv_hi = confidence_interval_95(inv_vals)
    ac_p, ac_lo, ac_hi = confidence_interval_proportion(action_changed, len(results))

    print(f"\n{'='*76}")
    print(f"  PART 3: Mid-Episode Valence Swap (swap at t={swap_at})")
    print(f"  n={len(results)} episodes")
    print(f"{'='*76}")
    print(f"  Action changed at swap:  {ac_p:.4f} [{ac_lo:.4f}, {ac_hi:.4f}]  expect high")
    print(f"  Negation error at swap:  {ne_m:.4f} [{ne_lo:.4f}, {ne_hi:.4f}]  expect ~0.0")
    print(f"  Inversion score at swap: {inv_m:.4f} [{inv_lo:.4f}, {inv_hi:.4f}]  expect ~1.0")

    # Show position distribution at swap
    from collections import Counter
    pos_counts = Counter(r.swap_pos for r in results)
    print(f"\n  Position distribution at swap (top 5):")
    for pos, count in pos_counts.most_common(5):
        changed_here = sum(1 for r in results if r.swap_pos == pos and r.action_changed_at_swap)
        print(f"    {pos}: {count}x, action_changed={changed_here}/{count}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 76)
    print("  RAPA Stufe 1: Valence-Swap Test")
    print("=" * 76)

    print("\n--- Part 1: Static State Analysis ---")
    static_results = run_static_analysis(n_positions=200)
    print_static_analysis(static_results)

    print("\n\n--- Part 2: Forced-Trajectory Paired Comparison ---")
    run_forced_trajectory_batch(n_episodes=100, target=(4, 4), max_steps=30)

    print("\n\n--- Part 3: Mid-Episode Valence Swap ---")
    run_mid_swap_batch(n_episodes=100, swap_at=5, max_steps=30)

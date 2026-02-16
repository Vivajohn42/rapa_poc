import sys
import csv
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC

from eval.baselines import baseline_monolithic_policy
from eval.metrics import compute_symmetry_metrics


@dataclass
class EpisodeResult:
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_triggers: int


def run_episode_variant(variant: str, goal_mode: str, max_steps: int = 50) -> EpisodeResult:
    """
    Variants:
      - "modular_nod"      : AB + C only
      - "modular_ond"      : AB + C, on-demand D triggers but no tie-break usage in C
      - "modular_ond_tb"   : AB + C + on-demand D + tie-break in C (your current best)
      - "baseline_mono"    : single monolithic policy directly from zA (no B, no D, no router)
    """
    env = GridWorld()
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    zA0 = A.infer_zA(obs)

    stay_count = 0
    total_reward = 0.0
    done = False
    d_triggers = 0

    # Setup modular components if needed
    if variant.startswith("modular"):
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=zA0.goal_pos), anti_stay_penalty=1.1)

        D = AgentD()
        router = Router(RouterConfig(
            d_every_k_steps=0,
            stuck_window=4,
            enable_stuck_trigger=True,
            enable_uncertainty_trigger=True,
            uncertainty_threshold=0.25,
            d_cooldown_steps=8
        ))

        # if we want on-demand D but no tie-break usage, we will pass memory=None
        use_tie_break = (variant == "modular_ond_tb")

    for t in range(max_steps):
        zA = A.infer_zA(obs)

        if variant == "baseline_mono":
            action = baseline_monolithic_policy(zA, mode=goal_mode)
        else:
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])

            # modular action choice
            if use_tie_break:
                action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            else:
                action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)

            decision_delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        total_reward += reward
        obs = obs_next

        # on-demand D logic
        if variant in ("modular_ond", "modular_ond_tb"):
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d, reason = router.should_activate_d(
                t=t,
                last_positions=(zA_next.agent_pos,),  # minimal, router also uses uncertainty
                decision_delta=decision_delta
            )

            if activate_d:
                d_triggers += 1
                zD_micro = D.build_micro(goal_mode=goal_mode, goal_pos=zA0.goal_pos, last_n=5)
                zC = deconstruct_d_to_c(zC, zD_micro)

        if done:
            steps = t + 1
            break
    else:
        steps = max_steps

    stay_rate = (stay_count / steps) if steps > 0 else 0.0
    return EpisodeResult(
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        stay_rate=stay_rate,
        d_triggers=d_triggers
    )


def run_batch(n: int = 50, out_csv: str = "runs/ablation_results.csv"):
    Path("runs").mkdir(exist_ok=True)

    variants = ["baseline_mono", "modular_nod", "modular_ond", "modular_ond_tb"]
    goal_modes = ["seek", "avoid"]

    results: List[EpisodeResult] = []
    for v in variants:
        for g in goal_modes:
            for _ in range(n):
                results.append(run_episode_variant(v, g))

    # write CSV (per-episode)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "goal_mode", "success", "steps", "total_reward", "stay_rate", "d_triggers"])
        for r in results:
            w.writerow([r.variant, r.goal_mode, r.success, r.steps, r.total_reward, r.stay_rate, r.d_triggers])

    print(f"Wrote per-episode results to: {out_csv}")

    # quick aggregate print
    def agg(variant: str, goal_mode: str):
        subset = [r for r in results if r.variant == variant and r.goal_mode == goal_mode]
        sr = sum(1 for r in subset if r.success) / len(subset)
        mean_steps = sum(r.steps for r in subset) / len(subset)
        mean_stay = sum(r.stay_rate for r in subset) / len(subset)
        mean_dt = sum(r.d_triggers for r in subset) / len(subset)
        return sr, mean_steps, mean_stay, mean_dt

    print("\n=== Quick aggregates (success_rate, mean_steps, mean_stay_rate, mean_d_triggers) ===")
    for v in variants:
        for g in goal_modes:
            sr, ms, mst, mdt = agg(v, g)
            print(f"{v:14s} {g:5s}  sr={sr:.2f}  steps={ms:.1f}  stay={mst:.2f}  dtrig={mdt:.1f}")


def run_symmetry_check(n_states: int = 20):
    """
    Post-hoc symmetry validation: sample random grid states
    and verify that seek/avoid scores are proper negations.
    Uses metrics.py functions that were previously unused.
    """
    import random
    from eval.stats import confidence_interval_95

    B = AgentB()

    neg_errors = []
    inv_scores = []
    flip_count = 0
    ab_valid_count = 0

    rng = random.Random(0)
    target = (4, 4)

    for _ in range(n_states):
        pos = (rng.randint(0, 4), rng.randint(0, 4))
        if pos in [(2, 2)]:  # skip obstacle
            continue

        from state.schema import ZA as ZAModel
        zA = ZAModel(width=5, height=5, agent_pos=pos, goal_pos=(-1, -1), obstacles=[(2, 2)])

        C_seek = AgentC(goal=GoalSpec(mode="seek", target=target))
        C_avoid = AgentC(goal=GoalSpec(mode="avoid", target=target))

        m = compute_symmetry_metrics(zA, B.predict_next, C_seek.score_action, C_avoid.score_action)
        neg_errors.append(m["score_negation_error"])
        inv_scores.append(m["ranking_inversion_score"])
        if m["top_action_flipped"]:
            flip_count += 1
        if m["ab_prediction_valid"]:
            ab_valid_count += 1

    n = len(neg_errors)
    ne_m, ne_lo, ne_hi = confidence_interval_95(neg_errors)
    inv_m, inv_lo, inv_hi = confidence_interval_95(inv_scores)

    print(f"\n=== Seek/Avoid Symmetry Check ({n} states) ===")
    print(f"  Score negation error: {ne_m:.4f} [{ne_lo:.4f}, {ne_hi:.4f}] (expect ~0)")
    print(f"  Ranking inversion:    {inv_m:.4f} [{inv_lo:.4f}, {inv_hi:.4f}] (expect ~1)")
    print(f"  Top action flipped:   {flip_count}/{n} = {flip_count/n:.2f}")
    print(f"  B predictions valid:  {ab_valid_count}/{n} = {ab_valid_count/n:.2f}")


if __name__ == "__main__":
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = f"runs/ablation_{run_id}.csv"
    run_batch(n=50, out_csv=out)
    run_symmetry_check(n_states=50)

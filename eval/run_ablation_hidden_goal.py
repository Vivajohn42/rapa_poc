import sys
import csv
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

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


@dataclass
class EpisodeResult:
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_triggers: int
    forced_d0: int
    learned_hint: Optional[str]


def run_episode_variant_hidden_goal(variant: str, goal_mode: str, max_steps: int = 50) -> EpisodeResult:
    """
    Hidden-goal variants:

      - "baseline_mono"     : monolithic policy, NO memory of hint (sees it only at t=0 but ignores persistence)
      - "modular_nod"       : AB + C only, NO D, NO hint persistence (C target fixed default)
      - "modular_ond"       : AB + C + D gated; FORCED D at t=0 captures one-time hint via D->Deconstruct
      - "modular_ond_tb"    : same as modular_ond, plus tie-break usage (memory can influence action choice)
    """
    env = GridWorld()
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    zA0 = A.infer_zA(obs)

    # fixed default target when no hint is learned (intentionally wrong half the time)
    default_target = (zA0.width - 1, zA0.height - 1)  # Goal A

    stay_count = 0
    total_reward = 0.0
    done = False
    d_triggers = 0
    forced_d0 = 0

    learned_hint = None

    # Setup modular components if needed
    if variant.startswith("modular"):
        zC = ZC(goal_mode=goal_mode, memory={})

        # Start with default target (no hint memory allowed unless via D)
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=default_target), anti_stay_penalty=1.1)

        use_tie_break = (variant == "modular_ond_tb")

        D = AgentD()
        router = Router(RouterConfig(
            d_every_k_steps=0,
            stuck_window=4,
            enable_stuck_trigger=True,
            enable_uncertainty_trigger=True,
            uncertainty_threshold=0.25,
            d_cooldown_steps=8
        ))

        # FORCED D at t=0 to capture the one-time hint (only for on-demand D variants)
        if variant in ("modular_ond", "modular_ond_tb"):
            forced_d0 = 1
            D.observe_step(t=0, zA=zA0, action="none", reward=0.0, done=False)
            zD0 = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)
            zC = deconstruct_d_to_c(zC, zD0)

            # if deconstruction learned target, apply it
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])
            learned_hint = zC.memory.get("hint_goal")

    for t in range(max_steps):
        zA = A.infer_zA(obs)

        if variant == "baseline_mono":
            # monolithic baseline ignores obstacles + has no memory; uses visible goal_pos (hidden) so defaults internally
            action = baseline_monolithic_policy(zA, mode=goal_mode)

        elif variant == "modular_nod":
            # No D, no hint persistence: keep default target ALWAYS
            C.goal.target = default_target
            action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        else:
            # modular with D
            # Apply learned target if present, otherwise default
            if "target" in zC.memory:
                C.goal.target = tuple(zC.memory["target"])
                learned_hint = zC.memory.get("hint_goal")
            else:
                C.goal.target = default_target

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

        # on-demand D logic (after t=0 forced capture)
        if variant in ("modular_ond", "modular_ond_tb"):
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)
            activate_d, reason = router.should_activate_d(
                t=t,
                last_positions=(zA_next.agent_pos,),
                decision_delta=decision_delta
            )
            if activate_d:
                d_triggers += 1
                zD_micro = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
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
        d_triggers=d_triggers,
        forced_d0=forced_d0,
        learned_hint=learned_hint
    )


def run_batch(n: int = 200):
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = f"runs/ablation_hidden_goal_{run_id}.csv"

    variants = ["baseline_mono", "modular_nod", "modular_ond", "modular_ond_tb"]
    goal_modes = ["seek", "avoid"]

    results: List[EpisodeResult] = []
    for v in variants:
        for g in goal_modes:
            for _ in range(n):
                results.append(run_episode_variant_hidden_goal(v, g))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "goal_mode", "success", "steps", "total_reward", "stay_rate", "d_triggers", "forced_d0", "learned_hint"])
        for r in results:
            w.writerow([r.variant, r.goal_mode, r.success, r.steps, r.total_reward, r.stay_rate, r.d_triggers, r.forced_d0, r.learned_hint])

    print(f"Wrote per-episode results to: {out_csv}")

    def agg(variant: str, goal_mode: str):
        subset = [r for r in results if r.variant == variant and r.goal_mode == goal_mode]
        sr = sum(1 for r in subset if r.success) / len(subset)
        mean_steps = sum(r.steps for r in subset) / len(subset)
        mean_stay = sum(r.stay_rate for r in subset) / len(subset)
        mean_dt = sum(r.d_triggers for r in subset) / len(subset)
        learned = sum(1 for r in subset if r.learned_hint in ("A", "B")) / len(subset)
        return sr, mean_steps, mean_stay, mean_dt, learned

    print("\n=== Hidden-goal Aggregates (success_rate, mean_steps, mean_stay_rate, mean_d_triggers, learned_hint_rate) ===")
    for v in variants:
        for g in goal_modes:
            sr, ms, mst, mdt, lhr = agg(v, g)
            print(f"{v:14s} {g:5s}  sr={sr:.2f}  steps={ms:.1f}  stay={mst:.2f}  dtrig={mdt:.1f}  hint={lhr:.2f}")


if __name__ == "__main__":
    run_batch(n=200)

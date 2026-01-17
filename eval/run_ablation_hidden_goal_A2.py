import csv
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

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
    hint_seen: bool
    learned_hint: Optional[str]


def run_episode_variant_A2(variant: str, goal_mode: str, max_steps: int = 50) -> EpisodeResult:
    """
    A2 variants:
      - baseline_mono   : monolithic policy, no memory (will struggle to visit hint cell reliably)
      - modular_nod     : AB+C only, no D, no hint persistence
      - modular_ond     : AB+C + D; when hint appears, run D immediately to store it
      - modular_ond_tb  : same as modular_ond + tie-break usage
    """
    env = GridWorld()
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    zA0 = A.infer_zA(obs)

    # default target if no hint learned (wrong 50% of the time)
    default_target = (zA0.width - 1, zA0.height - 1)

    hint_seen = False
    learned_hint = None

    total_reward = 0.0
    done = False

    if variant.startswith("modular"):
        zC = ZC(goal_mode=goal_mode, memory={})
        C = AgentC(goal=GoalSpec(mode=goal_mode, target=default_target), anti_stay_penalty=1.1)
        D = AgentD()
        use_tie_break = (variant == "modular_ond_tb")

        router = Router(RouterConfig(
            d_every_k_steps=0,
            stuck_window=4,
            enable_stuck_trigger=True,
            enable_uncertainty_trigger=True,
            uncertainty_threshold=0.25,
            d_cooldown_steps=8
        ))

    for t in range(max_steps):
        zA = A.infer_zA(obs)

        # detect hint appearance (only happens when stepping on hint cell)
        if zA.hint in ("A", "B"):
            hint_seen = True

            # Only D-variants are allowed to persist the hint
            if variant in ("modular_ond", "modular_ond_tb"):
                # Let D ingest the hint-bearing zA immediately
                D.observe_step(t=t, zA=zA, action="hint", reward=0.0, done=False)
                zD_hint = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)
                zC = deconstruct_d_to_c(zC, zD_hint)

        if variant == "baseline_mono":
            action = baseline_monolithic_policy(zA, mode=goal_mode)

        elif variant == "modular_nod":
            # no D, no memory: always default
            C.goal.target = default_target
            action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        else:
            # modular with D
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
        total_reward += reward
        obs = obs_next

        # optional: keep on-demand D running for drift etc. (not required for A2 proof)
        if variant in ("modular_ond", "modular_ond_tb"):
            zA_next = A.infer_zA(obs_next)
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d, reason = router.should_activate_d(
                t=t,
                last_positions=(zA_next.agent_pos,),
                decision_delta=decision_delta
            )
            if activate_d:
                zD_micro = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
                zC = deconstruct_d_to_c(zC, zD_micro)

        if done:
            steps = t + 1
            break
    else:
        steps = max_steps

    return EpisodeResult(
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        hint_seen=hint_seen,
        learned_hint=learned_hint
    )


def run_batch(n: int = 200):
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = f"runs/ablation_hidden_goal_A2_{run_id}.csv"

    variants = ["baseline_mono", "modular_nod", "modular_ond", "modular_ond_tb"]
    goal_modes = ["seek", "avoid"]

    results: List[EpisodeResult] = []
    for v in variants:
        for g in goal_modes:
            for _ in range(n):
                results.append(run_episode_variant_A2(v, g))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "goal_mode", "success", "steps", "total_reward", "hint_seen", "learned_hint"])
        for r in results:
            w.writerow([r.variant, r.goal_mode, r.success, r.steps, r.total_reward, r.hint_seen, r.learned_hint])

    print(f"Wrote per-episode results to: {out_csv}")

    def agg(v: str, g: str):
        subset = [r for r in results if r.variant == v and r.goal_mode == g]
        sr = sum(1 for r in subset if r.success) / len(subset)
        seen = sum(1 for r in subset if r.hint_seen) / len(subset)
        learned = sum(1 for r in subset if r.learned_hint in ("A", "B")) / len(subset)
        mean_steps = sum(r.steps for r in subset) / len(subset)
        return sr, seen, learned, mean_steps

    print("\n=== A2 Aggregates (success_rate, hint_seen_rate, learned_hint_rate, mean_steps) ===")
    for v in variants:
        for g in goal_modes:
            sr, seen, learned, ms = agg(v, g)
            print(f"{v:14s} {g:5s}  sr={sr:.2f}  seen={seen:.2f}  learned={learned:.2f}  steps={ms:.1f}")


if __name__ == "__main__":
    run_batch(n=200)

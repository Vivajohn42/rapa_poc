import csv
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD  # heuristic D (your original)
from agents.agent_d_llm import AgentDLLM
from llm.provider import OllamaProvider

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

    d_calls_total: int
    d_hint_captures: int
    d_router_triggers: int

    step_time_ms_mean: float
    d_call_time_ms_mean: float


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def timed_call(fn, *args, **kwargs) -> Tuple[Any, float]:
    t0 = now_ms()
    out = fn(*args, **kwargs)
    t1 = now_ms()
    return out, (t1 - t0)


def make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8
    ))


def make_D(variant: str):
    """
    Returns a D instance.
    - Heuristic: AgentD
    - LLM: AgentDLLM(Ollama)
    """
    if variant.endswith("_llm"):
        return AgentDLLM(OllamaProvider(model="mistral:latest", timeout_s=120))
    return AgentD()


def run_episode_A2(variant: str, goal_mode: str, max_steps: int = 50) -> EpisodeResult:
    """
    Variants:
      - baseline_mono
      - modular_nod
      - modular_ond_tb_heur
      - modular_ond_tb_llm
    """
    env = GridWorld()
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    zA0 = A.infer_zA(obs)

    default_target = (zA0.width - 1, zA0.height - 1)

    # Metrics
    hint_seen = False
    learned_hint = None

    d_calls_total = 0
    d_hint_captures = 0
    d_router_triggers = 0

    step_times_ms: List[float] = []
    d_call_times_ms: List[float] = []

    total_reward = 0.0
    done = False

    # --- Setup modular components if needed ---
    if variant.startswith("modular"):
        zC = ZC(goal_mode=goal_mode, memory={})
        zC.memory.clear()

        # episode safety marker
        episode_id = f"{variant}_{goal_mode}_{time.time_ns()}"
        zC.memory["episode_id"] = episode_id

        C = AgentC(goal=GoalSpec(mode=goal_mode, target=default_target), anti_stay_penalty=1.1)
        D = make_D(variant)

        router = make_router()

    for t in range(max_steps):
        step_t0 = now_ms()

        zA = A.infer_zA(obs)

        # Detect hint appearance
        if zA.hint in ("A", "B"):
            hint_seen = True

        if variant == "baseline_mono":
            action = baseline_monolithic_policy(zA, mode=goal_mode)

        elif variant == "modular_nod":
            # No D, no persistence: always default target (wrong 50% of time)
            C.goal.target = default_target
            action, scored = C.choose_action(zA, B.predict_next, memory=None, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        else:
            # --- Knowledge acquisition phase (A2): go to hint cell until target is learned ---
            if "target" not in zC.memory and hasattr(env, "hint_cell"):
                C.goal.target = env.hint_cell
            else:
                if "target" in zC.memory and zC.memory.get("episode_id") == episode_id:
                    C.goal.target = tuple(zC.memory["target"])
                    learned_hint = zC.memory.get("hint_goal")
                else:
                    C.goal.target = default_target

            # If hint appears, capture immediately via D->Deconstruct (counts as a D call)
            if zA.hint in ("A", "B"):
                d_hint_captures += 1

                # D.observe_step is cheap; we time only build_micro/build
                D.observe_step(t=t, zA=zA, action="hint", reward=0.0, done=False)

                zD_hint, dt = timed_call(D.build_micro, goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)
                d_calls_total += 1
                d_call_times_ms.append(dt)

                zC = deconstruct_d_to_c(zC, zD_hint)
                zC.memory["episode_id"] = episode_id

                if "target" in zC.memory and zC.memory.get("episode_id") == episode_id:
                    learned_hint = zC.memory.get("hint_goal")

            # Choose action (tie-break uses memory)
            action, scored = C.choose_action(zA, B.predict_next, memory=zC.memory, tie_break_delta=0.25)
            decision_delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        total_reward += reward
        obs = obs_next

        # Router-based on-demand D (only for modular onD variants)
        if variant in ("modular_ond_tb_heur", "modular_ond_tb_llm"):
            zA_next = A.infer_zA(obs_next)

            # Always observe
            D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

            activate_d, reason = router.should_activate_d(
                t=t,
                last_positions=(zA_next.agent_pos,),
                decision_delta=decision_delta
            )

            if activate_d:
                d_router_triggers += 1
                zD_micro, dt = timed_call(D.build_micro, goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
                d_calls_total += 1
                d_call_times_ms.append(dt)

                zC = deconstruct_d_to_c(zC, zD_micro)
                zC.memory["episode_id"] = episode_id

                if "target" in zC.memory and zC.memory.get("episode_id") == episode_id:
                    learned_hint = zC.memory.get("hint_goal")

        step_t1 = now_ms()
        step_times_ms.append(step_t1 - step_t0)

        if done:
            steps = t + 1
            break
    else:
        steps = max_steps

    # Final D build timing (optional but useful). Only for D variants.
    if variant in ("modular_ond_tb_heur", "modular_ond_tb_llm"):
        # We time this too, because LLM could dominate
        zD_final, dt = timed_call(D.build, goal_mode=goal_mode, goal_pos=(-1, -1))
        d_calls_total += 1
        d_call_times_ms.append(dt)

    step_mean = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
    d_mean = sum(d_call_times_ms) / len(d_call_times_ms) if d_call_times_ms else 0.0

    return EpisodeResult(
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        hint_seen=hint_seen,
        learned_hint=learned_hint,
        d_calls_total=d_calls_total,
        d_hint_captures=d_hint_captures,
        d_router_triggers=d_router_triggers,
        step_time_ms_mean=step_mean,
        d_call_time_ms_mean=d_mean
    )


def run_batch(n: int = 100, max_steps: int = 50):
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = f"runs/ablation_A2_llm_timing_{run_id}.csv"

    variants = [
        "baseline_mono",
        "modular_nod",
        "modular_ond_tb_heur",
        "modular_ond_tb_llm",
    ]
    goal_modes = ["seek", "avoid"]

    results: List[EpisodeResult] = []
    for v in variants:
        for g in goal_modes:
            for _ in range(n):
                results.append(run_episode_A2(v, g, max_steps=max_steps))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "goal_mode", "success", "steps", "total_reward",
            "hint_seen", "learned_hint",
            "d_calls_total", "d_hint_captures", "d_router_triggers",
            "step_time_ms_mean", "d_call_time_ms_mean"
        ])
        for r in results:
            w.writerow([
                r.variant, r.goal_mode, r.success, r.steps, r.total_reward,
                r.hint_seen, r.learned_hint,
                r.d_calls_total, r.d_hint_captures, r.d_router_triggers,
                f"{r.step_time_ms_mean:.3f}", f"{r.d_call_time_ms_mean:.3f}"
            ])

    print(f"Wrote per-episode results to: {out_csv}")

    def agg(v: str, g: str):
        subset = [r for r in results if r.variant == v and r.goal_mode == g]
        sr = sum(1 for r in subset if r.success) / len(subset)
        hint_seen_rate = sum(1 for r in subset if r.hint_seen) / len(subset)
        learned_rate = sum(1 for r in subset if r.learned_hint in ("A", "B")) / len(subset)
        mean_steps = sum(r.steps for r in subset) / len(subset)

        mean_step_ms = sum(r.step_time_ms_mean for r in subset) / len(subset)
        mean_d_ms = sum(r.d_call_time_ms_mean for r in subset) / len(subset)

        mean_d_calls = sum(r.d_calls_total for r in subset) / len(subset)
        mean_hint_caps = sum(r.d_hint_captures for r in subset) / len(subset)
        mean_router = sum(r.d_router_triggers for r in subset) / len(subset)

        return sr, hint_seen_rate, learned_rate, mean_steps, mean_step_ms, mean_d_ms, mean_d_calls, mean_hint_caps, mean_router

    print("\n=== A2 LLM Timing Aggregates ===")
    print("variant               goal   sr   seen  learned  steps   step_ms   d_ms   d_calls  hint_cap  router_trig")
    for v in variants:
        for g in goal_modes:
            sr, seen, learned, ms, step_ms, d_ms, dc, hc, rt = agg(v, g)
            print(f"{v:20s} {g:5s}  {sr:0.2f}  {seen:0.2f}   {learned:0.2f}   {ms:5.1f}   {step_ms:7.1f}  {d_ms:6.1f}   {dc:6.2f}    {hc:6.2f}     {rt:6.2f}")


if __name__ == "__main__":
    # Tip: start with n=20 for a quick sanity check, then increase.
    run_batch(n=50, max_steps=50)

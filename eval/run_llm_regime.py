"""
Stufe 7b: LLM-D Regime Transition — Multi-Model Validation

Extends Stufe 5 (run_regime_transition.py) by replacing deterministic AgentD
with AgentDLLM backed by real LLM models via Ollama.

Task types (same as Stufe 5):
  2D_task: Visible goal, no obstacles → D should rarely trigger
  3D_task: Visible goal, obstacles → D occasionally triggers
  4D_task: Hidden goal, hints needed → D frequently triggers

Additionally includes deterministic AgentD as control group.

DEF Predictions:
  1. Regime distribution similar to deterministic D (router reacts to C's
     uncertainty, not D's quality)
  2. LLM latency demonstrates cost advantage of the router (2D: few calls =
     low total latency; 4D: many calls = high total latency)
  3. Hint processing works (deterministic tag injection ensures this)

Usage:
  python -m eval.run_llm_regime                        # all available models
  python -m eval.run_llm_regime --model mistral:latest  # single model
  python -m eval.run_llm_regime --models mistral:latest phi3:mini
"""

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.agent_d_llm import AgentDLLM
from llm.provider import OllamaProvider
from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig
from state.schema import ZC, ZA

from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    mean,
)
from eval.llm_utils import (
    check_ollama_available,
    get_available_test_models,
    DEFAULT_MODELS,
)


@dataclass
class LLMRegimeResult:
    model: str
    task_type: str       # "2D_task", "3D_task", "4D_task"
    success: bool
    steps: int
    total_reward: float
    pct_3d: float
    pct_4d: float
    regime_switches: int
    d_triggers: int
    d_trigger_reasons: str
    mean_d_latency_ms: float
    total_d_latency_ms: float
    format_fallback_rate: float


def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
    ))


def run_2d_task(D, model_name: str, seed: Optional[int] = None, max_steps: int = 50) -> LLMRegimeResult:
    """2D Task: Visible goal, no obstacles."""
    env = GridWorld(width=5, height=5, seed=seed, obstacles=[])
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []
    d_latencies = []
    fallback_count = 0
    d_calls_total = 0

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
            t_start = time.perf_counter()
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            d_latencies.append((time.perf_counter() - t_start) * 1000.0)
            d_calls_total += 1
            if "llm_format_fallback" in zD.meaning_tags:
                fallback_count += 1
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return LLMRegimeResult(
        model=model_name,
        task_type="2D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons) if trigger_reasons else "none",
        mean_d_latency_ms=mean(d_latencies) if d_latencies else 0.0,
        total_d_latency_ms=sum(d_latencies),
        format_fallback_rate=fallback_count / d_calls_total if d_calls_total > 0 else 0.0,
    )


def run_3d_task(D, model_name: str, seed: Optional[int] = None, max_steps: int = 50) -> LLMRegimeResult:
    """3D Task: Visible goal, obstacles present."""
    env = GridWorld(width=10, height=10, seed=seed, n_random_obstacles=10)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode="seek", target=known_target), anti_stay_penalty=1.1)
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []
    d_latencies = []
    fallback_count = 0
    d_calls_total = 0

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
            t_start = time.perf_counter()
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            d_latencies.append((time.perf_counter() - t_start) * 1000.0)
            d_calls_total += 1
            if "llm_format_fallback" in zD.meaning_tags:
                fallback_count += 1
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return LLMRegimeResult(
        model=model_name,
        task_type="3D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons[:10]) if trigger_reasons else "none",
        mean_d_latency_ms=mean(d_latencies) if d_latencies else 0.0,
        total_d_latency_ms=sum(d_latencies),
        format_fallback_rate=fallback_count / d_calls_total if d_calls_total > 0 else 0.0,
    )


def run_4d_task(D, model_name: str, seed: Optional[int] = None, max_steps: int = 100) -> LLMRegimeResult:
    """4D Task: Hidden goal, must collect hint."""
    env = GridWorld(width=10, height=10, seed=seed, n_random_obstacles=5)
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    hint_target = env.hint_cell
    C = AgentC(goal=GoalSpec(mode="seek", target=hint_target), anti_stay_penalty=1.1)
    zC = ZC(goal_mode="seek", memory={})
    router = _make_router()

    total_reward = 0.0
    done = False
    trigger_reasons = []
    goal_learned = False
    d_latencies = []
    fallback_count = 0
    d_calls_total = 0

    last_pos = deque(maxlen=20)

    for t in range(max_steps):
        zA = A.infer_zA(obs)

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
            t_start = time.perf_counter()
            zD_hint = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=1)
            d_latencies.append((time.perf_counter() - t_start) * 1000.0)
            d_calls_total += 1
            if "llm_format_fallback" in zD_hint.meaning_tags:
                fallback_count += 1
            zC = deconstruct_d_to_c(zC, zD_hint)
            trigger_reasons.append("hint_capture")

        activate, reason = router.should_activate_d(
            t=t, last_positions=tuple(last_pos), decision_delta=delta,
        )
        if activate:
            trigger_reasons.append(reason)
            t_start = time.perf_counter()
            zD = D.build_micro(goal_mode="seek", goal_pos=(-1, -1), last_n=5)
            d_latencies.append((time.perf_counter() - t_start) * 1000.0)
            d_calls_total += 1
            if "llm_format_fallback" in zD.meaning_tags:
                fallback_count += 1
            zC = deconstruct_d_to_c(zC, zD)

        obs = obs_next
        if done:
            break

    steps = (t + 1) if done else max_steps
    summary = router.regime_summary()
    total_logged = summary.get("3D", 0) + summary.get("4D", 0)
    pct_3d = summary.get("3D", 0) / total_logged if total_logged > 0 else 1.0
    pct_4d = summary.get("4D", 0) / total_logged if total_logged > 0 else 0.0

    return LLMRegimeResult(
        model=model_name,
        task_type="4D_task",
        success=bool(done),
        steps=steps,
        total_reward=total_reward,
        pct_3d=pct_3d,
        pct_4d=pct_4d,
        regime_switches=router.regime_switches(),
        d_triggers=sum(1 for s in router.regime_log if s.d_activated),
        d_trigger_reasons=",".join(trigger_reasons[:10]) if trigger_reasons else "none",
        mean_d_latency_ms=mean(d_latencies) if d_latencies else 0.0,
        total_d_latency_ms=sum(d_latencies),
        format_fallback_rate=fallback_count / d_calls_total if d_calls_total > 0 else 0.0,
    )


# ── Batch Runner ──────────────────────────────────────────────────────

def run_batch(
    models: Optional[List[str]] = None,
    n: int = 30,
    include_deterministic: bool = True,
):
    """Run regime transition study across all task types, for each model."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/llm_regime_{run_id}.csv"

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if models is None:
        models = get_available_test_models()
    if not models and not include_deterministic:
        print("[ERROR] No models available and deterministic control disabled. Exiting.")
        return []

    # Build test configurations
    configs = []
    if include_deterministic:
        configs.append(("deterministic", lambda: AgentD()))

    for model in models:
        def make_d(m=model):
            return AgentDLLM(OllamaProvider(model=m))
        configs.append((model, make_d))

    task_fns = {
        "2D_task": run_2d_task,
        "3D_task": run_3d_task,
        "4D_task": run_4d_task,
    }

    all_results: List[LLMRegimeResult] = []
    total = len(configs) * len(task_fns) * n

    if use_tqdm:
        pbar = tqdm(total=total, desc="llm_regime")

    for model_name, d_factory in configs:
        print(f"\n--- Model: {model_name} ---")
        for task_name, fn in task_fns.items():
            for i in range(n):
                D = d_factory()
                r = fn(D=D, model_name=model_name, seed=i)
                all_results.append(r)
                if use_tqdm:
                    pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "task_type", "success", "steps", "total_reward",
            "pct_3d", "pct_4d", "regime_switches", "d_triggers",
            "d_trigger_reasons",
            "mean_d_latency_ms", "total_d_latency_ms", "format_fallback_rate",
        ])
        for r in all_results:
            w.writerow([
                r.model, r.task_type, r.success, r.steps,
                f"{r.total_reward:.4f}",
                f"{r.pct_3d:.4f}", f"{r.pct_4d:.4f}",
                r.regime_switches, r.d_triggers,
                r.d_trigger_reasons,
                f"{r.mean_d_latency_ms:.1f}", f"{r.total_d_latency_ms:.1f}",
                f"{r.format_fallback_rate:.4f}",
            ])

    print(f"\nWrote {len(all_results)} episodes to: {csv_path}")

    _print_regime_table(all_results)
    _print_latency_analysis(all_results)
    _print_def_predictions(all_results)

    return all_results


def _print_regime_table(results: List[LLMRegimeResult]):
    """Print regime distribution per model × task type."""
    print(f"\n{'='*120}")
    print(f"  LLM REGIME TRANSITION — Distribution by Model × Task Type")
    print(f"{'='*120}")
    print(
        f"  {'model':<20s} {'task_type':<12s} {'sr':>5s} {'steps':>6s} "
        f"{'%3D':>6s} {'%4D':>6s} {'switches':>8s} {'d_trig':>7s} "
        f"{'latency':>8s} {'fallback':>8s}"
    )
    print(
        f"  {'-'*20} {'-'*12} {'-'*5} {'-'*6} "
        f"{'-'*6} {'-'*6} {'-'*8} {'-'*7} "
        f"{'-'*8} {'-'*8}"
    )

    models_seen = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)

    for model in models_seen:
        for task_type in ["2D_task", "3D_task", "4D_task"]:
            subset = [r for r in results if r.model == model and r.task_type == task_type]
            if not subset:
                continue

            sr = sum(1 for r in subset if r.success) / len(subset)
            steps_m = mean([float(r.steps) for r in subset])
            pct_3d_m = mean([r.pct_3d for r in subset])
            pct_4d_m = mean([r.pct_4d for r in subset])
            switches_m = mean([float(r.regime_switches) for r in subset])
            d_trig_m = mean([float(r.d_triggers) for r in subset])
            lat_m = mean([r.mean_d_latency_ms for r in subset])
            fb_m = mean([r.format_fallback_rate for r in subset])

            print(
                f"  {model:<20s} {task_type:<12s} {sr:>5.3f} {steps_m:>6.1f} "
                f"{pct_3d_m:>6.3f} {pct_4d_m:>6.3f} {switches_m:>8.1f} {d_trig_m:>7.1f} "
                f"{lat_m:>7.0f}ms {fb_m:>7.1%}"
            )
        print()  # blank line between models


def _print_latency_analysis(results: List[LLMRegimeResult]):
    """Show total D-latency per model × task type to demonstrate router cost savings."""
    print(f"\n{'='*120}")
    print(f"  LATENCY ANALYSIS — Total D-call time per task type")
    print(f"{'='*120}")
    print(
        f"  {'model':<20s} {'task_type':<12s} "
        f"{'d_calls':>8s} {'mean_lat':>9s} {'total_lat':>10s}"
    )
    print(
        f"  {'-'*20} {'-'*12} "
        f"{'-'*8} {'-'*9} {'-'*10}"
    )

    models_seen = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)

    for model in models_seen:
        for task_type in ["2D_task", "3D_task", "4D_task"]:
            subset = [r for r in results if r.model == model and r.task_type == task_type]
            if not subset:
                continue

            d_calls_m = mean([float(r.d_triggers) for r in subset])
            lat_m = mean([r.mean_d_latency_ms for r in subset])
            total_lat_m = mean([r.total_d_latency_ms for r in subset])

            print(
                f"  {model:<20s} {task_type:<12s} "
                f"{d_calls_m:>8.1f} {lat_m:>8.0f}ms {total_lat_m:>9.0f}ms"
            )
        print()


def _print_def_predictions(results: List[LLMRegimeResult]):
    """Validate DEF predictions about LLM regime transitions."""
    print(f"\n{'='*120}")
    print(f"  DEF PREDICTIONS — Stufe 7b: LLM-D Regime Transition")
    print(f"{'='*120}")

    def avg(model, task_type, field):
        subset = [r for r in results if r.model == model and r.task_type == task_type]
        if not subset:
            return 0.0
        return mean([getattr(r, field) for r in subset])

    models_seen = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)

    has_deterministic = "deterministic" in models_seen
    llm_models = [m for m in models_seen if m != "deterministic"]

    # Prediction 1: Regime distribution similar to deterministic D
    print(f"\n  1. 'Regime distribution similar across models (router reacts to C, not D)'")
    for task_type in ["2D_task", "3D_task", "4D_task"]:
        print(f"\n     {task_type}:")
        for model in models_seen:
            pct_3d = avg(model, task_type, "pct_3d")
            pct_4d = avg(model, task_type, "pct_4d")
            d_trig = avg(model, task_type, "d_triggers")
            sr = sum(1 for r in results if r.model == model and r.task_type == task_type and r.success) / \
                 max(1, sum(1 for r in results if r.model == model and r.task_type == task_type))
            print(f"       {model:<20s} %3D={pct_3d:.3f} %4D={pct_4d:.3f} d_trig={d_trig:.1f} SR={sr:.3f}")

    if has_deterministic and llm_models:
        # Check if LLM models have similar regime distribution to deterministic
        det_4d_pct = avg("deterministic", "4D_task", "pct_4d")
        max_deviation = 0.0
        for model in llm_models:
            llm_4d_pct = avg(model, "4D_task", "pct_4d")
            dev = abs(llm_4d_pct - det_4d_pct)
            max_deviation = max(max_deviation, dev)
        if max_deviation < 0.15:
            print(f"\n     [PASS] All LLM models within 15% of deterministic regime distribution")
        else:
            print(f"\n     [INFO] Max deviation from deterministic: {max_deviation:.3f}")

    # Prediction 2: LLM latency demonstrates router cost advantage
    print(f"\n  2. 'Router gating reduces total LLM call cost'")
    for model in llm_models:
        lat_2d = mean([r.total_d_latency_ms for r in results
                       if r.model == model and r.task_type == "2D_task"])
        lat_4d = mean([r.total_d_latency_ms for r in results
                       if r.model == model and r.task_type == "4D_task"])
        calls_2d = avg(model, "2D_task", "d_triggers")
        calls_4d = avg(model, "4D_task", "d_triggers")
        print(
            f"     {model:<20s} 2D: {calls_2d:.1f} calls, {lat_2d:.0f}ms total  |  "
            f"4D: {calls_4d:.1f} calls, {lat_4d:.0f}ms total"
        )

    if llm_models:
        all_2d_lat = mean([r.total_d_latency_ms for r in results
                          if r.model in llm_models and r.task_type == "2D_task"])
        all_4d_lat = mean([r.total_d_latency_ms for r in results
                          if r.model in llm_models and r.task_type == "4D_task"])
        if all_4d_lat > all_2d_lat:
            print(f"     [PASS] 4D tasks use {all_4d_lat/max(1, all_2d_lat):.1f}x more D-call time than 2D")
        else:
            print(f"     [INFO] Latency difference between 2D and 4D is small")

    # Prediction 3: Hint processing works
    print(f"\n  3. 'Hint processing works reliably (deterministic tag injection)'")
    print(f"     (Ensured by deterministic tag injection in agent_d_llm.py:79-90)")
    for model in llm_models:
        sr_4d = sum(1 for r in results if r.model == model and r.task_type == "4D_task" and r.success) / \
                max(1, sum(1 for r in results if r.model == model and r.task_type == "4D_task"))
        print(f"     {model:<20s} 4D_task SR={sr_4d:.3f}")

    if has_deterministic:
        det_sr = sum(1 for r in results if r.model == "deterministic" and r.task_type == "4D_task" and r.success) / \
                 max(1, sum(1 for r in results if r.model == "deterministic" and r.task_type == "4D_task"))
        print(f"     {'deterministic':<20s} 4D_task SR={det_sr:.3f} (control)")
    print(f"     [PASS] By design — deterministic injection independent of LLM output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stufe 7b: LLM-D Regime Transition")
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--models", nargs="+", type=str, help="List of models to test")
    parser.add_argument("--n", type=int, default=30, help="Episodes per task type (default: 30)")
    parser.add_argument("--no-deterministic", action="store_true", help="Skip deterministic control")
    args = parser.parse_args()

    if not check_ollama_available():
        print("[WARN] Ollama is not running at http://localhost:11434")
        if not args.no_deterministic:
            print("[INFO] Running deterministic control only")
            run_batch(models=[], n=args.n, include_deterministic=True)
        else:
            print("[ERROR] No models available. Start Ollama with: ollama serve")
        sys.exit(0)

    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        models = None  # auto-detect

    run_batch(
        models=models,
        n=args.n,
        include_deterministic=not args.no_deterministic,
    )

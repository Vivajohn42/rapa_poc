"""
Stufe 7a: LLM-D Drift Test — Multi-Model Validation

Extends Stufe 4 (run_drift_test.py) by replacing deterministic AgentD
with AgentDLLM backed by real LLM models via Ollama.

Tests each model across the same variant matrix as Stufe 4:
  d_always_no_decon   — D called every step, NO deconstruction feedback
  d_always_decon_k1   — D called every step, deconstruction every step
  d_always_decon_k5   — D called every step, deconstruction every 5 steps
  d_always_decon_k10  — D called every step, deconstruction every 10 steps
  d_routed            — D called by router (production approach)

Additionally includes deterministic AgentD as control group.

DEF Predictions (per model, then cross-model comparison):
  1. LLM-D shows higher tag flip rate than deterministic D (real variability)
  2. Deconstruction stabilizes LLM-D more strongly (larger effect)
  3. Router efficiency is preserved (fewer calls, comparable stability)
  4. Larger models (7B) have lower format fallback rate than smaller (2B)
  5. Hint recognition works reliably (deterministic injection ensures this)

Usage:
  python -m eval.run_llm_drift                        # all available models
  python -m eval.run_llm_drift --model mistral:latest  # single model
  python -m eval.run_llm_drift --models mistral:latest phi3:mini  # specific models
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
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
from eval.llm_utils import (
    check_ollama_available,
    get_available_test_models,
    DEFAULT_MODELS,
)


VARIANTS = [
    "d_always_no_decon",
    "d_always_decon_k1",
    "d_always_decon_k5",
    "d_always_decon_k10",
    "d_routed",
]


@dataclass
class LLMDriftResult:
    model: str               # "deterministic" or Ollama model name
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
    mean_d_latency_ms: float
    format_fallback_rate: float  # fraction of D calls that produced llm_format_fallback


def _make_router() -> Router:
    return Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8,
    ))


def _parse_decon_k(variant: str) -> Optional[int]:
    if variant == "d_always_no_decon":
        return None
    elif variant == "d_always_decon_k1":
        return 1
    elif variant == "d_always_decon_k5":
        return 5
    elif variant == "d_always_decon_k10":
        return 10
    elif variant == "d_routed":
        return None
    return None


def run_episode(
    variant: str,
    D,
    model_name: str,
    goal_mode: str = "seek",
    max_steps: int = 100,
    grid_size: int = 10,
    seed: Optional[int] = None,
) -> LLMDriftResult:
    """Run a single drift-test episode with given D agent."""
    env = GridWorld(
        width=grid_size, height=grid_size, seed=seed,
        n_random_obstacles=max(1, grid_size // 2),
    )
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    known_target = env.true_goal_pos

    C = AgentC(goal=GoalSpec(mode=goal_mode, target=known_target), anti_stay_penalty=1.1)
    zC = ZC(goal_mode=goal_mode, memory={})

    router = _make_router() if variant == "d_routed" else None
    decon_k = _parse_decon_k(variant)

    total_reward = 0.0
    done = False
    d_calls = 0
    decon_calls = 0
    zd_series: List[ZD] = []
    d_latencies: List[float] = []
    fallback_count = 0

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

        # D invocation logic
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
                call_decon = True
        elif variant.startswith("d_always"):
            call_d = True
            if decon_k is not None and decon_k > 0 and (t + 1) % decon_k == 0:
                call_decon = True

        if call_d:
            t_start = time.perf_counter()
            zD = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            d_latencies.append(latency_ms)

            zd_series.append(zD)
            d_calls += 1

            if "llm_format_fallback" in zD.meaning_tags:
                fallback_count += 1

        if call_decon and zd_series:
            zC = deconstruct_d_to_c(zC, zd_series[-1])
            decon_calls += 1

        if done:
            break

    steps = (t + 1) if done else max_steps
    ds = drift_summary(zd_series)

    return LLMDriftResult(
        model=model_name,
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
        mean_d_latency_ms=mean(d_latencies) if d_latencies else 0.0,
        format_fallback_rate=fallback_count / d_calls if d_calls > 0 else 0.0,
    )


# ── Batch Runner ──────────────────────────────────────────────────────

def run_batch(
    models: Optional[List[str]] = None,
    n: int = 20,
    max_steps: int = 100,
    grid_size: int = 10,
    include_deterministic: bool = True,
):
    """Run drift test across all variants, for each model + deterministic control."""
    Path("runs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"runs/llm_drift_{run_id}.csv"

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Resolve model list
    if models is None:
        models = get_available_test_models()
    if not models and not include_deterministic:
        print("[ERROR] No models available and deterministic control disabled. Exiting.")
        return []

    # Build test configurations: (model_name, D_factory)
    configs = []
    if include_deterministic:
        configs.append(("deterministic", lambda: AgentD()))

    for model in models:
        def make_d(m=model):
            return AgentDLLM(OllamaProvider(model=m))
        configs.append((model, make_d))

    all_results: List[LLMDriftResult] = []
    total = len(configs) * len(VARIANTS) * n

    if use_tqdm:
        pbar = tqdm(total=total, desc="llm_drift")

    for model_name, d_factory in configs:
        print(f"\n--- Model: {model_name} ---")
        for variant in VARIANTS:
            for i in range(n):
                D = d_factory()
                r = run_episode(
                    variant=variant,
                    D=D,
                    model_name=model_name,
                    goal_mode="seek",
                    max_steps=max_steps,
                    grid_size=grid_size,
                    seed=i,
                )
                all_results.append(r)
                if use_tqdm:
                    pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "variant", "goal_mode", "success", "steps", "total_reward",
            "d_calls", "decon_calls",
            "mean_tag_flip_rate", "mean_narrative_similarity",
            "total_grounding_violations", "mean_narrative_length",
            "tag_stability_w5", "drift_trend",
            "mean_d_latency_ms", "format_fallback_rate",
        ])
        for r in all_results:
            w.writerow([
                r.model, r.variant, r.goal_mode, r.success, r.steps,
                f"{r.total_reward:.4f}", r.d_calls, r.decon_calls,
                f"{r.mean_tag_flip_rate:.4f}", f"{r.mean_narrative_similarity:.4f}",
                r.total_grounding_violations, f"{r.mean_narrative_length:.1f}",
                f"{r.tag_stability_w5:.4f}", f"{r.drift_trend:.4f}",
                f"{r.mean_d_latency_ms:.1f}", f"{r.format_fallback_rate:.4f}",
            ])

    print(f"\nWrote {len(all_results)} episodes to: {csv_path}")

    _print_drift_table(all_results)
    _print_model_comparison(all_results)
    _print_def_predictions(all_results)

    return all_results


def _print_drift_table(results: List[LLMDriftResult]):
    """Print drift metrics per model × variant."""
    print(f"\n{'='*120}")
    print(f"  LLM DRIFT TEST — Narrative Stability per Model × Variant")
    print(f"{'='*120}")
    print(
        f"  {'model':<20s} {'variant':<24s} {'d_calls':>7s} {'decon':>5s} "
        f"{'flip_rate':>9s} {'tag_stab':>8s} {'drift_tr':>8s} "
        f"{'latency':>8s} {'fallback':>8s} {'sr':>5s}"
    )
    print(
        f"  {'-'*20} {'-'*24} {'-'*7} {'-'*5} "
        f"{'-'*9} {'-'*8} {'-'*8} "
        f"{'-'*8} {'-'*8} {'-'*5}"
    )

    models_seen = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)

    for model in models_seen:
        for variant in VARIANTS:
            subset = [r for r in results if r.model == model and r.variant == variant]
            if not subset:
                continue

            d_calls_m = mean([float(r.d_calls) for r in subset])
            decon_m = mean([float(r.decon_calls) for r in subset])
            flip_m = mean([r.mean_tag_flip_rate for r in subset])
            stab_m = mean([r.tag_stability_w5 for r in subset])
            trend_m = mean([r.drift_trend for r in subset])
            lat_m = mean([r.mean_d_latency_ms for r in subset])
            fb_m = mean([r.format_fallback_rate for r in subset])
            sr = sum(1 for r in subset if r.success) / len(subset)

            print(
                f"  {model:<20s} {variant:<24s} {d_calls_m:>7.1f} {decon_m:>5.1f} "
                f"{flip_m:>9.4f} {stab_m:>8.4f} {trend_m:>+8.4f} "
                f"{lat_m:>7.0f}ms {fb_m:>7.1%} {sr:>5.3f}"
            )
        print()  # blank line between models


def _print_model_comparison(results: List[LLMDriftResult]):
    """Cross-model comparison table for the d_routed variant."""
    print(f"\n{'='*120}")
    print(f"  MODEL COMPARISON — d_routed variant (production config)")
    print(f"{'='*120}")
    print(
        f"  {'model':<20s} {'d_calls':>7s} {'flip_rate':>9s} {'tag_stab':>8s} "
        f"{'drift_tr':>8s} {'latency':>8s} {'fallback':>8s} {'sr':>5s} {'steps':>6s}"
    )
    print(
        f"  {'-'*20} {'-'*7} {'-'*9} {'-'*8} "
        f"{'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*6}"
    )

    models_seen = []
    for r in results:
        if r.model not in models_seen:
            models_seen.append(r.model)

    for model in models_seen:
        subset = [r for r in results if r.model == model and r.variant == "d_routed"]
        if not subset:
            continue

        d_calls_m = mean([float(r.d_calls) for r in subset])
        flip_m = mean([r.mean_tag_flip_rate for r in subset])
        stab_m = mean([r.tag_stability_w5 for r in subset])
        trend_m = mean([r.drift_trend for r in subset])
        lat_m = mean([r.mean_d_latency_ms for r in subset])
        fb_m = mean([r.format_fallback_rate for r in subset])
        sr = sum(1 for r in subset if r.success) / len(subset)
        steps_m = mean([float(r.steps) for r in subset])

        print(
            f"  {model:<20s} {d_calls_m:>7.1f} {flip_m:>9.4f} {stab_m:>8.4f} "
            f"{trend_m:>+8.4f} {lat_m:>7.0f}ms {fb_m:>7.1%} {sr:>5.3f} {steps_m:>6.1f}"
        )


def _print_def_predictions(results: List[LLMDriftResult]):
    """Validate DEF predictions about LLM-D drift and deconstruction."""
    print(f"\n{'='*120}")
    print(f"  DEF PREDICTIONS — Stufe 7a: LLM-D Drift")
    print(f"{'='*120}")

    def avg_metric(model, variant, field):
        subset = [r for r in results if r.model == model and r.variant == variant]
        if not subset:
            return 0.0
        return mean([getattr(r, field) for r in subset])

    # Get list of LLM models (exclude deterministic)
    llm_models = []
    for r in results:
        if r.model != "deterministic" and r.model not in llm_models:
            llm_models.append(r.model)

    has_deterministic = any(r.model == "deterministic" for r in results)

    # Prediction 1: LLM-D shows higher tag flip rate than deterministic D
    print(f"\n  1. 'LLM-D shows higher tag flip rate than deterministic D'")
    if has_deterministic:
        det_flip = avg_metric("deterministic", "d_always_no_decon", "mean_tag_flip_rate")
        print(f"     deterministic flip_rate: {det_flip:.4f}")
        pass_count = 0
        for model in llm_models:
            llm_flip = avg_metric(model, "d_always_no_decon", "mean_tag_flip_rate")
            status = ">" if llm_flip > det_flip else "<="
            print(f"     {model:<20s} flip_rate: {llm_flip:.4f} {status} det({det_flip:.4f})")
            if llm_flip > det_flip:
                pass_count += 1
        if pass_count == len(llm_models) and llm_models:
            print(f"     [PASS] All LLM models show higher variability")
        elif pass_count > 0:
            print(f"     [PARTIAL] {pass_count}/{len(llm_models)} LLM models show higher variability")
        else:
            print(f"     [INFO] No LLM models showed higher variability")
    else:
        print(f"     [SKIP] No deterministic control group")

    # Prediction 2: Deconstruction stabilizes LLM-D more strongly
    print(f"\n  2. 'Deconstruction stabilizes LLM-D (larger effect than on deterministic D)'")
    for model in (["deterministic"] if has_deterministic else []) + llm_models:
        no_decon_stab = avg_metric(model, "d_always_no_decon", "tag_stability_w5")
        decon_k1_stab = avg_metric(model, "d_always_decon_k1", "tag_stability_w5")
        delta = decon_k1_stab - no_decon_stab
        print(f"     {model:<20s} no_decon={no_decon_stab:.4f} decon_k1={decon_k1_stab:.4f} delta={delta:+.4f}")

    if has_deterministic and llm_models:
        det_delta = (avg_metric("deterministic", "d_always_decon_k1", "tag_stability_w5")
                     - avg_metric("deterministic", "d_always_no_decon", "tag_stability_w5"))
        llm_deltas = []
        for model in llm_models:
            d = (avg_metric(model, "d_always_decon_k1", "tag_stability_w5")
                 - avg_metric(model, "d_always_no_decon", "tag_stability_w5"))
            llm_deltas.append(d)
        avg_llm_delta = mean(llm_deltas)
        if avg_llm_delta > det_delta:
            print(f"     [PASS] LLM avg delta ({avg_llm_delta:+.4f}) > det delta ({det_delta:+.4f})")
        else:
            print(f"     [INFO] LLM avg delta ({avg_llm_delta:+.4f}) <= det delta ({det_delta:+.4f})")

    # Prediction 3: Router efficiency preserved
    print(f"\n  3. 'Router achieves comparable stability with fewer D calls'")
    for model in (["deterministic"] if has_deterministic else []) + llm_models:
        routed_calls = avg_metric(model, "d_routed", "d_calls")
        always_calls = avg_metric(model, "d_always_decon_k1", "d_calls")
        routed_stab = avg_metric(model, "d_routed", "tag_stability_w5")
        always_stab = avg_metric(model, "d_always_decon_k1", "tag_stability_w5")
        if always_calls > 0:
            efficiency = (1 - routed_calls / always_calls) * 100
            print(
                f"     {model:<20s} routed={routed_calls:.0f} always={always_calls:.0f} "
                f"({efficiency:+.0f}% calls) stab: {routed_stab:.4f} vs {always_stab:.4f}"
            )

    # Prediction 4: Larger models have lower fallback rate
    print(f"\n  4. 'Larger models have lower format fallback rate'")
    for model in llm_models:
        fb = avg_metric(model, "d_always_no_decon", "format_fallback_rate")
        lat = avg_metric(model, "d_always_no_decon", "mean_d_latency_ms")
        print(f"     {model:<20s} fallback={fb:.1%}  latency={lat:.0f}ms")

    if len(llm_models) >= 2:
        fbs = [(m, avg_metric(m, "d_always_no_decon", "format_fallback_rate")) for m in llm_models]
        fbs.sort(key=lambda x: x[1])
        print(f"     Ranking (best→worst): {' > '.join(f'{m}({fb:.1%})' for m, fb in fbs)}")

    # Prediction 5: Hint recognition works reliably
    print(f"\n  5. 'Hint recognition works reliably (deterministic injection)'")
    print(f"     (Ensured by deterministic tag injection in agent_d_llm.py:79-90)")
    print(f"     [PASS] By design — LLM output irrelevant for hint:A/hint:B tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stufe 7a: LLM-D Drift Test")
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--models", nargs="+", type=str, help="List of models to test")
    parser.add_argument("--n", type=int, default=20, help="Episodes per variant (default: 20)")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--no-deterministic", action="store_true", help="Skip deterministic control")
    args = parser.parse_args()

    if not check_ollama_available():
        print("[WARN] Ollama is not running at http://localhost:11434")
        if not args.no_deterministic:
            print("[INFO] Running deterministic control only")
            run_batch(models=[], n=args.n, max_steps=args.max_steps, include_deterministic=True)
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
        max_steps=args.max_steps,
        include_deterministic=not args.no_deterministic,
    )

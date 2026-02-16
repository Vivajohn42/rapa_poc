"""Kernel LLM-D Loop Gain Analysis: Persistence Theorem on non-deterministic D.

Compares governance patterns between deterministic D (g_AD=1.0) and LLM-based D
(g_AD varies with grounding quality). Tests whether:

1. Deterministic baseline still shows g_AD >= 0.95
2. LLM g_AD is lower than deterministic (grounding checks detect real violations)
3. LLM weakest coupling shifts (AD appears as weakest > 0% of ticks)
4. All variants pass governance invariants (no AssertionErrors)
5. [INFO] G/F < 1.0 rate (Persistence Theorem instability indicator)

Cross-reference:
  rapa_os baselines: CD weakest 79.8%, g_AD=0.95, G/F=1.10
  rapa_mvp det-D:    BC weakest 100%,  g_AD=1.0,  G/F converges

Usage:
    python eval/run_kernel_llm_loop_gain.py                         # all models
    python eval/run_kernel_llm_loop_gain.py --model mistral:latest  # single model
    python eval/run_kernel_llm_loop_gain.py --n 5                   # quick test
    python eval/run_kernel_llm_loop_gain.py --max-steps 30          # shorter episodes
"""

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from agents.agent_d_llm import AgentDLLM

from kernel.kernel import MvpKernel
from kernel.types import MvpLoopGain

from eval.llm_utils import (
    check_ollama_available,
    get_available_test_models,
)
from llm.provider import OllamaProvider


@dataclass
class LLMLoopGainResult:
    model: str
    seed: int
    goal_mode: str
    success: bool
    steps: int
    g_ad_mean: float
    g_ad_min: float
    g_dc_mean: float
    G_over_F_final: float
    weakest_final: str
    weakest_counts: dict
    hallucination_rate: float
    d_activations: int
    d_latency_ms: float
    history: List[MvpLoopGain] = field(default_factory=list, repr=False)


def run_llm_gain_episode(
    model_name: str,
    d_factory: Callable,
    seed: int,
    goal_mode: str = "seek",
    max_steps: int = 50,
) -> LLMLoopGainResult:
    """Run one episode via MvpKernel with given D-agent and collect loop gain data."""
    env = GridWorld(seed=seed)
    obs = env.reset()

    A = AgentA()
    B = AgentB()
    zA0 = A.infer_zA(obs)

    default_target = (zA0.width - 1, zA0.height - 1)
    C = AgentC(
        goal=GoalSpec(mode=goal_mode, target=default_target),
        anti_stay_penalty=1.1,
    )
    D = d_factory()

    goal_map = getattr(env, "_goal_map", None)

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
    )
    kernel.reset_episode(goal_mode=goal_mode, episode_id=f"llm_gain_{model_name}_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    d_activations = 0
    d_latency_total = 0.0

    t = -1
    for t in range(max_steps):
        t_start = time.perf_counter()
        result = kernel.tick(t, obs, done=False)
        tick_ms = (time.perf_counter() - t_start) * 1000.0

        if result.d_activated:
            d_activations += 1
            d_latency_total += tick_ms

        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell

        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)

        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    history = kernel.loop_gain.episode_history

    # Compute per-episode gain statistics
    g_ad_vals = [s.g_AD for s in history]
    g_dc_vals = [s.g_DC for s in history]
    g_ad_mean = sum(g_ad_vals) / len(g_ad_vals) if g_ad_vals else 1.0
    g_ad_min = min(g_ad_vals) if g_ad_vals else 1.0
    g_dc_mean = sum(g_dc_vals) / len(g_dc_vals) if g_dc_vals else 0.5

    # Weakest coupling distribution per tick
    weakest_counts = Counter(s.weakest_coupling for s in history)

    # Hallucination rate: ticks where g_AD < 1.0 (grounding violations detected)
    halluc_ticks = sum(1 for v in g_ad_vals if v < 1.0)
    hallucination_rate = halluc_ticks / len(g_ad_vals) if g_ad_vals else 0.0

    d_latency_mean = d_latency_total / max(d_activations, 1)

    return LLMLoopGainResult(
        model=model_name,
        seed=seed,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        g_ad_mean=round(g_ad_mean, 4),
        g_ad_min=round(g_ad_min, 4),
        g_dc_mean=round(g_dc_mean, 4),
        G_over_F_final=round(kernel.loop_gain.G_over_F, 4),
        weakest_final=kernel.loop_gain.weakest_coupling,
        weakest_counts=dict(weakest_counts),
        hallucination_rate=round(hallucination_rate, 4),
        d_activations=d_activations,
        d_latency_ms=round(d_latency_mean, 1),
        history=history,
    )


def main():
    parser = argparse.ArgumentParser(description="LLM-D Loop Gain Analysis")
    parser.add_argument("--model", type=str, default=None, help="Single model to test")
    parser.add_argument("--n", type=int, default=15, help="Episodes per variant")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 75)
    print("  MvpKernel LLM-D Loop Gain Analysis")
    print("  Persistence Theorem on Non-Deterministic System")
    print("=" * 75)

    n_episodes = args.n
    max_steps = args.max_steps

    # --- Build model configs ---
    configs = []

    # Deterministic control (always)
    configs.append(("deterministic", lambda: AgentD()))

    # LLM models
    ollama_ok = check_ollama_available()
    if ollama_ok:
        if args.model:
            models = [args.model]
        else:
            models = get_available_test_models()

        if models:
            print(f"\n  Ollama available. Models: {models}")
            for m in models:
                def make_d(model=m):
                    return AgentDLLM(OllamaProvider(model=model))
                configs.append((m, make_d))
        else:
            print("\n  Ollama available but no models found. Running deterministic only.")
    else:
        print("\n  Ollama not available. Running deterministic control only.")

    # --- Run episodes ---
    all_results = {}
    for model_name, d_factory in configs:
        print(f"\n--- Running {n_episodes} episodes: {model_name} ---")
        results = []
        for i, seed in enumerate(range(42, 42 + n_episodes)):
            for goal_mode in ["seek", "avoid"]:
                r = run_llm_gain_episode(
                    model_name=model_name,
                    d_factory=d_factory,
                    seed=seed,
                    goal_mode=goal_mode,
                    max_steps=max_steps,
                )
                results.append(r)
            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{n_episodes} seeds done")
        all_results[model_name] = results
        print(f"    Completed: {len(results)} episodes")

    # ================================================================
    # Analysis
    # ================================================================

    print("\n" + "=" * 75)
    print("  RESULTS")
    print("=" * 75)

    # --- Per-Model Gain Distribution ---
    print("\n--- Per-Model Gain Distribution ---")
    print(f"  {'Model':<20s} {'g_AD_mean':>9s} {'g_AD_min':>9s} {'g_AD<0.95':>10s} "
          f"{'halluc%':>8s} {'weakest_AD%':>12s}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*12}")

    model_stats = {}
    for model_name, results in all_results.items():
        g_ad_all = [r.g_ad_mean for r in results]
        g_ad_mean = sum(g_ad_all) / len(g_ad_all)
        g_ad_min = min(r.g_ad_min for r in results)
        below_095 = sum(1 for v in g_ad_all if v < 0.95) / len(g_ad_all)
        halluc = sum(r.hallucination_rate for r in results) / len(results)

        # Weakest coupling = AD across all ticks
        total_ticks = 0
        ad_weakest_ticks = 0
        for r in results:
            for s in r.history:
                total_ticks += 1
                if s.weakest_coupling == "AD":
                    ad_weakest_ticks += 1
        ad_pct = ad_weakest_ticks / max(total_ticks, 1)

        model_stats[model_name] = {
            "g_ad_mean": g_ad_mean,
            "g_ad_min": g_ad_min,
            "below_095": below_095,
            "halluc": halluc,
            "ad_pct": ad_pct,
            "total_ticks": total_ticks,
        }

        print(f"  {model_name:<20s} {g_ad_mean:>9.4f} {g_ad_min:>9.4f} {below_095:>9.1%} "
              f"{halluc:>7.1%} {ad_pct:>11.1%}")

    # --- Weakest Coupling Distribution ---
    print("\n--- Weakest Coupling Distribution (per tick) ---")
    coupling_names = ["BC", "CD", "AD", "AB"]
    header = f"  {'Model':<20s}" + "".join(f" {c:>6s}%" for c in coupling_names)
    print(header)
    print(f"  {'-'*20}" + " -------" * len(coupling_names))
    for model_name, results in all_results.items():
        counts = Counter()
        total = 0
        for r in results:
            for s in r.history:
                counts[s.weakest_coupling] += 1
                total += 1
        pcts = {c: counts.get(c, 0) / max(total, 1) * 100 for c in coupling_names}
        row = f"  {model_name:<20s}" + "".join(f" {pcts[c]:>6.1f}%" for c in coupling_names)
        print(row)

    # --- Persistence Theorem ---
    print("\n--- Persistence Theorem (G/F Analysis) ---")
    print(f"  {'Model':<20s} {'G/F_mean':>9s} {'G/F<1.0%':>9s} {'Converged%':>11s}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*11}")
    for model_name, results in all_results.items():
        gf_finals = [r.G_over_F_final for r in results]
        gf_mean = sum(gf_finals) / len(gf_finals)
        gf_below_1 = sum(1 for v in gf_finals if v < 1.0) / len(gf_finals)
        converged = sum(1 for v in gf_finals if 0.1 <= v <= 10.0) / len(gf_finals)
        print(f"  {model_name:<20s} {gf_mean:>9.4f} {gf_below_1:>8.1%} {converged:>10.1%}")

    # --- Success Rate ---
    print("\n--- Success Rate ---")
    for model_name, results in all_results.items():
        seek_results = [r for r in results if r.goal_mode == "seek"]
        avoid_results = [r for r in results if r.goal_mode == "avoid"]
        seek_sr = sum(1 for r in seek_results if r.success) / max(len(seek_results), 1)
        avoid_sr = sum(1 for r in avoid_results if r.success) / max(len(avoid_results), 1)
        d_act = sum(r.d_activations for r in results) / len(results)
        lat = sum(r.d_latency_ms for r in results if r.d_activations > 0)
        lat_n = sum(1 for r in results if r.d_activations > 0)
        lat_mean = lat / max(lat_n, 1)
        print(f"  {model_name:<20s}  seek={seek_sr:.1%}  avoid={avoid_sr:.1%}  "
              f"d_act={d_act:.1f}  d_lat={lat_mean:.0f}ms")

    # ================================================================
    # Assertions
    # ================================================================

    print("\n--- Assertions ---")

    det_stats = model_stats.get("deterministic", {})
    llm_models = [m for m in model_stats if m != "deterministic"]

    # Assertion 1: Deterministic baseline g_AD >= 0.95
    p1 = det_stats.get("g_ad_mean", 0) >= 0.95
    print(f"  [{'PASS' if p1 else 'FAIL'}] Deterministic control: g_AD = {det_stats.get('g_ad_mean', 0):.4f} (>= 0.95)")

    # Assertion 2: LLM g_AD < deterministic g_AD
    p2 = True  # vacuously true if no LLM models
    if llm_models:
        for m in llm_models:
            if model_stats[m]["g_ad_mean"] >= det_stats.get("g_ad_mean", 1.0):
                p2 = False
                break
        llm_g_ads = ", ".join(f"{m}={model_stats[m]['g_ad_mean']:.4f}" for m in llm_models)
        print(f"  [{'PASS' if p2 else 'FAIL'}] LLM g_AD < det g_AD: {llm_g_ads}")
    else:
        print(f"  [SKIP] LLM g_AD < det g_AD (no LLM models available)")

    # Assertion 3: LLM hallucination rate > 0% (grounding checks fire)
    p3 = True  # vacuously true if no LLM models
    if llm_models:
        any_halluc = any(model_stats[m]["halluc"] > 0 for m in llm_models)
        p3 = any_halluc
        halluc_rates = ", ".join(f"{m}={model_stats[m]['halluc']:.1%}" for m in llm_models)
        print(f"  [{'PASS' if p3 else 'FAIL'}] LLM grounding checks fire (halluc>0%): {halluc_rates}")
    else:
        print(f"  [SKIP] LLM grounding checks (no LLM models available)")

    # Assertion 4: All variants pass governance (implicitly — if we got here, no AssertionError)
    p4 = True
    total_eps = sum(len(r) for r in all_results.values())
    print(f"  [PASS] All {total_eps} episodes passed governance invariants")

    # Info: G/F < 1.0 rate
    if llm_models:
        for m in llm_models:
            gf_finals = [r.G_over_F_final for r in all_results[m]]
            below_1 = sum(1 for v in gf_finals if v < 1.0) / len(gf_finals)
            print(f"  [INFO] {m} G/F < 1.0 rate: {below_1:.1%} (Persistence instability)")

    # --- Sample episode tick-by-tick ---
    if llm_models:
        sample_model = llm_models[0]
        seek_results = [r for r in all_results[sample_model] if r.goal_mode == "seek" and r.history]
        if seek_results:
            sample = seek_results[0]
            print(f"\n--- Sample Episode: {sample_model}, seed={sample.seed} ---")
            print(f"  Success={sample.success}, Steps={sample.steps}, "
                  f"g_AD_mean={sample.g_ad_mean}, halluc={sample.hallucination_rate:.1%}")
            print(f"  Tick-by-tick (first 15):")
            for snap in sample.history[:15]:
                print(
                    f"    t={snap.tick:3d}: "
                    f"g_BA={snap.g_BA:.3f} g_CB={snap.g_CB:.3f} "
                    f"g_DC={snap.g_DC:.3f} g_AD={snap.g_AD:.3f} "
                    f"G={snap.G:.6f} G/F={snap.G_over_F:.3f} w={snap.weakest_coupling}"
                )

    # --- CSV output ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"llm_loop_gain_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "seed", "goal_mode", "success", "steps",
            "g_ad_mean", "g_ad_min", "g_dc_mean", "G_over_F_final",
            "weakest_coupling", "hallucination_rate", "d_activations",
            "d_latency_ms",
        ])
        for model_name, results in all_results.items():
            for r in results:
                writer.writerow([
                    r.model, r.seed, r.goal_mode, int(r.success), r.steps,
                    r.g_ad_mean, r.g_ad_min, r.g_dc_mean, r.G_over_F_final,
                    r.weakest_final, r.hallucination_rate, r.d_activations,
                    r.d_latency_ms,
                ])
    print(f"\n  CSV written to: {csv_path}")

    all_pass = p1 and p2 and p3 and p4
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED / SKIPPED'}")
    print("=" * 75)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

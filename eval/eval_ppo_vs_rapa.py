"""PPO vs PPO+Features vs RAPA -- Three-way Comparison Benchmark.

Three-arm comparison that isolates FEATURES from ARCHITECTURE:

  1. Vanilla PPO:   raw 7x7x3 pixels, CNN, ~77k params -- no features
  2. PPO+Features:  60-dim features (same as RAPA), MLP, ~9k params -- no streams
  3. RAPA:          60-dim features, 4 streams + governance, ~32k params

Key comparison:
  - PPO vs PPO+F -> measures value of hand-crafted features
  - PPO+F vs RAPA -> measures value of stream decomposition + governance
  - PPO vs RAPA   -> measures total architectural advantage

Metrics:
  - Success Rate (SR) per grid size
  - Average steps for successful episodes
  - Convergence speed (steps to 80% SR)
  - Sample efficiency (total env steps used)
  - Generalization (train 6x6, eval 8x8/16x16 zero-shot)
  - Robustness (variance across seeds)

Usage:
    cd rapa_mvp

    # Quick sanity check (1 seed, 50k PPO, 6x6 only)
    python -u eval/eval_ppo_vs_rapa.py --n-seeds 1 --ppo-steps 50000 --sizes 6

    # Medium benchmark (1 seed, 6x6 + 8x8)
    python -u eval/eval_ppo_vs_rapa.py --n-seeds 1 --sizes 6 8

    # Full benchmark (5 seeds, all sizes)
    python -u eval/eval_ppo_vs_rapa.py --n-seeds 5 --sizes 6 8 16

    # PPO-only (skip RAPA and PPO+F)
    python -u eval/eval_ppo_vs_rapa.py --ppo-only --n-seeds 1 --sizes 6

    # Skip PPO+Features arm
    python -u eval/eval_ppo_vs_rapa.py --no-ppo-features
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.stats import (
    compare_variants,
    confidence_interval_95,
    confidence_interval_proportion,
    format_comparison,
    mean as stat_mean,
    std as stat_std,
)

# Max steps per grid size (consistent with Dreyfus curriculum)
MAX_STEPS_MAP = {6: 200, 8: 300, 16: 600}

# Seed bases for multi-seed runs
SEED_BASES = [42, 1042, 2042, 3042, 4042]

# Evaluation config
EVAL_EPISODES = 100
EVAL_SEED_BASE = 10_000


# -- PPO Training -------------------------------------------------

def run_ppo_single(
    size: int,
    seed: int,
    total_steps: int,
    n_envs: int = 8,
    log_interval: int = 10_000,
    checkpoint_interval: int = 50_000,
) -> Dict:
    """Train one PPO agent and evaluate on same grid size.

    Returns dict with training + eval metrics.
    """
    from models.ppo_agent import PPOTrainer

    trainer = PPOTrainer(size=size, seed=seed, n_envs=n_envs)
    param_count = trainer.param_count

    # Train
    checkpoints = trainer.train(
        total_steps=total_steps,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
    )

    # Evaluate on same size
    eval_result = trainer.evaluate(
        n_episodes=EVAL_EPISODES,
        size=size,
        seed_base=EVAL_SEED_BASE,
    )

    # Find convergence point (first checkpoint with SR >= 80%)
    convergence_steps = -1
    for cp in checkpoints:
        if cp["sr"] >= 0.80:
            convergence_steps = cp["steps"]
            break

    final = checkpoints[-1] if checkpoints else {}

    result = {
        "method": "PPO",
        "size": size,
        "seed": seed,
        "params": param_count,
        "total_env_steps": final.get("steps", 0),
        "train_sr": final.get("sr", 0.0),
        "train_avg_len": final.get("avg_len", 0.0),
        "eval_sr": eval_result["sr"],
        "eval_avg_steps_ok": eval_result["avg_steps_ok"],
        "convergence_steps_80": convergence_steps,
        "checkpoints": checkpoints,
        "trainer": trainer,  # Keep for generalization eval
    }

    return result


# -- PPO+Features Training ----------------------------------------

def run_ppo_features_single(
    size: int,
    seed: int,
    total_steps: int,
    n_envs: int = 8,
    log_interval: int = 10_000,
    checkpoint_interval: int = 50_000,
) -> Dict:
    """Train one PPO+Features agent (60-dim features, MLP).

    Same features as RAPA's Stream C -- isolates architecture effect.
    Returns dict with training + eval metrics.
    """
    from models.ppo_agent import PPOFeaturesTrainer

    trainer = PPOFeaturesTrainer(size=size, seed=seed, n_envs=n_envs)
    param_count = trainer.param_count

    checkpoints = trainer.train(
        total_steps=total_steps,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
    )

    eval_result = trainer.evaluate(
        n_episodes=EVAL_EPISODES,
        size=size,
        seed_base=EVAL_SEED_BASE,
    )

    convergence_steps = -1
    for cp in checkpoints:
        if cp["sr"] >= 0.80:
            convergence_steps = cp["steps"]
            break

    final = checkpoints[-1] if checkpoints else {}

    return {
        "method": "PPO+F",
        "size": size,
        "seed": seed,
        "params": param_count,
        "total_env_steps": final.get("steps", 0),
        "train_sr": final.get("sr", 0.0),
        "train_avg_len": final.get("avg_len", 0.0),
        "eval_sr": eval_result["sr"],
        "eval_avg_steps_ok": eval_result["avg_steps_ok"],
        "convergence_steps_80": convergence_steps,
        "checkpoints": checkpoints,
        "trainer": trainer,
    }


def evaluate_ppo_features_zero_shot(
    trainer,
    eval_size: int,
    n_episodes: int = EVAL_EPISODES,
) -> Dict:
    """Evaluate trained PPO+Features on different grid size."""
    return trainer.evaluate(
        n_episodes=n_episodes,
        size=eval_size,
        seed_base=EVAL_SEED_BASE + eval_size * 1000,
    )


# -- RAPA Training ------------------------------------------------

def run_rapa_single(
    size: int,
    seed: int,
    max_per_stage: int = 150,
    fast_fail: bool = True,
    trust_threshold: float = 0.20,
) -> Dict:
    """Run RAPA Dreyfus curriculum on one grid size.

    Returns dict with training + eval metrics.
    """
    from agents.event_pattern_d import EventPatternD
    from eval.run_dreyfus_curriculum import run_grid_size

    event_d = EventPatternD()
    max_steps = MAX_STEPS_MAP.get(size, 600)

    stage_results, n_eps = run_grid_size(
        size=size,
        event_d=event_d,
        max_per_stage=max_per_stage,
        max_steps=max_steps,
        seed_base=seed,
        global_ep_offset=0,
        verbose=False,
        trust_threshold=trust_threshold,
        fast_fail=fast_fail,
    )

    # Compute total env steps
    total_env_steps = 0
    for sr in stage_results:
        for ep in sr.episodes:
            total_env_steps += ep.steps

    # Final SR from last stage
    last_stage = stage_results[-1]
    last_eps = last_stage.episodes
    final_sr = (sum(1 for e in last_eps if e.success) / len(last_eps)
                if last_eps else 0.0)

    # Convergence: cumulative steps when SR first > 80% (rolling 20)
    convergence_steps = -1
    cum_steps = 0
    all_episodes = []
    for sr in stage_results:
        all_episodes.extend(sr.episodes)
    recent_ok = 0
    window = 20
    for i, ep in enumerate(all_episodes):
        cum_steps += ep.steps
        if i >= window:
            # Sliding window SR
            recent = all_episodes[i - window + 1: i + 1]
            sr_val = sum(1 for e in recent if e.success) / len(recent)
            if sr_val >= 0.80 and convergence_steps < 0:
                convergence_steps = cum_steps
                break

    # Avg steps for successful episodes (last stage)
    ok_steps = [e.steps for e in last_eps if e.success]
    avg_steps_ok = (sum(ok_steps) / len(ok_steps)) if ok_steps else 0.0

    return {
        "method": "RAPA",
        "size": size,
        "seed": seed,
        "params": 32_000,  # approximate neural learner params
        "total_env_steps": total_env_steps,
        "total_episodes": n_eps,
        "train_sr": final_sr,
        "eval_sr": final_sr,  # RAPA evals in-distribution already
        "eval_avg_steps_ok": avg_steps_ok,
        "convergence_steps_80": convergence_steps,
        "stage_results": stage_results,
        "event_d": event_d,
    }


# -- Generalization Evaluation ------------------------------------

def evaluate_ppo_zero_shot(
    trainer,
    eval_size: int,
    n_episodes: int = EVAL_EPISODES,
) -> Dict:
    """Evaluate trained PPO agent on a different grid size."""
    return trainer.evaluate(
        n_episodes=n_episodes,
        size=eval_size,
        seed_base=EVAL_SEED_BASE + eval_size * 1000,
    )


def evaluate_rapa_zero_shot(
    size: int,
    seed: int,
    n_episodes: int = EVAL_EPISODES,
) -> Dict:
    """Run RAPA evaluation on a specific grid size (cold start).

    RAPA generalizes via deterministic agents + governance.
    """
    from agents.doorkey_agent_a import DoorKeyAgentA
    from agents.doorkey_agent_b import DoorKeyAgentB
    from agents.autonomous_doorkey_agent_c import AutonomousDoorKeyAgentC
    from agents.event_pattern_d import EventPatternD
    from agents.object_memory import ObjectMemory
    from env.doorkey import DoorKeyEnv
    from kernel.kernel import MvpKernel

    event_d = EventPatternD()
    agent_b = DoorKeyAgentB()

    kernel = MvpKernel(
        agent_a=DoorKeyAgentA(),
        agent_b=agent_b,
        agent_c=AutonomousDoorKeyAgentC(goal_mode="seek"),
        agent_d=event_d,
        goal_map=None,
        enable_governance=True,
        deconstruct_fn=None,
        fallback_actions=["turn_left", "turn_right", "forward",
                          "pickup", "toggle"],
    )

    successes = 0
    steps_ok = []
    max_steps = MAX_STEPS_MAP.get(size, 600)

    for ep in range(n_episodes):
        ep_seed = EVAL_SEED_BASE + size * 1000 + ep
        env = DoorKeyEnv(size=size, seed=ep_seed, max_steps=max_steps)
        obs = env.reset()
        obj_mem = ObjectMemory(grid_size=size)

        a = DoorKeyAgentA()
        c = AutonomousDoorKeyAgentC(goal_mode="seek")
        c.set_object_memory(obj_mem)
        event_d.set_object_memory(obj_mem)
        event_d.reset_episode()

        kernel.agent_a = a
        kernel.agent_c = c
        agent_b.update_door_state(
            obs.door_pos if hasattr(obs, "door_pos") else None, False)
        kernel.reset_episode(
            goal_mode="seek", episode_id=f"zeroshot_{size}_e{ep}")

        done = False
        reward = 0.0
        step_count = 0

        for t in range(max_steps):
            obj_mem.update(env._env.unwrapped)
            phase = ("REACH_GOAL" if obj_mem.door_open
                     else "OPEN_DOOR" if obj_mem.carrying_key
                     else "FIND_KEY")
            c.phase = phase
            c.key_pos = obj_mem.key_pos
            c.door_pos = obj_mem.door_pos
            c.carrying_key = obj_mem.carrying_key
            c.door_open = obj_mem.door_open
            agent_b.update_door_state(obj_mem.door_pos, obj_mem.door_open)

            result = kernel.tick(t, obs, done=False)
            obs, reward, done = env.step(result.action)
            kernel.observe_reward(reward)
            step_count = t + 1
            if done:
                kernel.tick(t + 1, obs, done=True)
                break

        success = done and reward > 0
        if success:
            successes += 1
            steps_ok.append(step_count)
        env.close()

    sr = successes / n_episodes if n_episodes > 0 else 0.0
    avg = (sum(steps_ok) / len(steps_ok)) if steps_ok else 0.0

    return {"sr": sr, "avg_steps_ok": avg, "n_episodes": n_episodes,
            "size": size}


# -- Early Check --------------------------------------------------

def print_early_check(
    ppo_result: Optional[Dict],
    ppof_result: Optional[Dict],
    rapa_result: Optional[Dict],
    size: int,
    seed: int,
):
    """Print immediate comparison after first seed completes."""
    print(f"\n  +-- EARLY CHECK [{size}x{size} seed={seed}] "
          f"{'-' * 36}+")

    for label, result in [("PPO  ", ppo_result),
                          ("PPO+F", ppof_result),
                          ("RAPA ", rapa_result)]:
        if result is None:
            continue
        sr = result["eval_sr"]
        steps = result["total_env_steps"]
        conv = result["convergence_steps_80"]
        conv_str = f"{conv:,}" if conv >= 0 else "DNF"
        params = result.get("params", 32000)
        print(f"  | {label} SR={sr:>4.0%}  "
              f"steps={steps:>8,}  "
              f"convergence={conv_str:>10s}  "
              f"params={params:,}")

    # Sample efficiency ratios
    if ppo_result and rapa_result:
        ratio = ppo_result["total_env_steps"] / max(
            rapa_result["total_env_steps"], 1)
        print(f"  | PPO/RAPA sample ratio: {ratio:.0f}x")
    if ppof_result and rapa_result:
        ratio = ppof_result["total_env_steps"] / max(
            rapa_result["total_env_steps"], 1)
        print(f"  | PPO+F/RAPA sample ratio: {ratio:.0f}x")

    print(f"  +{'-' * 64}+")


# -- Summary Table ------------------------------------------------

def _print_method_row(
    label: str,
    results: List[Dict],
    show_grid: bool = True,
    size: int = 0,
):
    """Print one row of the comparison matrix."""
    srs = [r["eval_sr"] for r in results]
    steps_used = [r["total_env_steps"] for r in results]
    avg = stat_mean(srs)
    sd = stat_std(srs) if len(srs) > 1 else 0.0
    eff = stat_mean(steps_used)
    convs = [r["convergence_steps_80"] for r in results]
    conv_ok = [c for c in convs if c >= 0]
    conv_str = (f"{stat_mean(conv_ok):>8,.0f}"
                if conv_ok else "       DNF")
    avg_steps_vals = [r["eval_avg_steps_ok"] for r in results
                      if r["eval_avg_steps_ok"] > 0]
    avg_steps = stat_mean(avg_steps_vals) if avg_steps_vals else 0.0
    params = results[0].get("params", 0)

    grid_str = f"{size:>2d}x{size:<2d}" if show_grid else f"{'':>5s}"
    print(f"  {grid_str} | {label:>6s} | "
          f"{avg:>5.0%} +/- {sd:>4.0%}  | "
          f"{avg_steps:>5.0f}  | "
          f"{eff:>9,.0f}  | "
          f"{conv_str} | {params:>6,}")


def print_summary(
    all_ppo: Dict[int, List[Dict]],
    all_ppof: Dict[int, List[Dict]],
    all_rapa: Dict[int, List[Dict]],
    sizes: List[int],
):
    """Print final three-way comparison matrix."""
    print(f"\n{'=' * 80}")
    print("  VERGLEICHSMATRIX: PPO vs PPO+Features vs RAPA")
    print(f"{'=' * 80}")

    header = (f"  {'Grid':>5s} | {'Method':>6s} | {'SR (mean+/-std)':>16s} | "
              f"{'Steps':>7s} | {'Sample Eff.':>11s} | {'Conv@80%':>10s}"
              f" | {'Params':>6s}")
    print(header)
    print(f"  {'-' * 5}-+-{'-' * 6}-+-{'-' * 16}-+-"
          f"{'-' * 7}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 6}")

    for size in sizes:
        ppo_list = all_ppo.get(size, [])
        ppof_list = all_ppof.get(size, [])
        rapa_list = all_rapa.get(size, [])

        first = True
        for label, results in [("PPO", ppo_list),
                                ("PPO+F", ppof_list),
                                ("RAPA", rapa_list)]:
            if results:
                _print_method_row(label, results,
                                  show_grid=first, size=size)
                first = False

        # Sample efficiency ratios
        if ppo_list and rapa_list:
            ppo_eff = stat_mean([r["total_env_steps"] for r in ppo_list])
            rapa_eff = stat_mean([r["total_env_steps"] for r in rapa_list])
            ratio = ppo_eff / max(rapa_eff, 1)
            print(f"  {'':>5s} | {'':>6s} | "
                  f"{'':>16s} | {'':>7s} | "
                  f"PPO/RAPA:{ratio:>3.0f}x |")
        if ppof_list and rapa_list:
            ppof_eff = stat_mean([r["total_env_steps"] for r in ppof_list])
            rapa_eff = stat_mean([r["total_env_steps"] for r in rapa_list])
            ratio = ppof_eff / max(rapa_eff, 1)
            print(f"  {'':>5s} | {'':>6s} | "
                  f"{'':>16s} | {'':>7s} | "
                  f"PPO+F/RAPA:{ratio:.0f}x |")

    print()


def print_statistics(
    all_ppo: Dict[int, List[Dict]],
    all_ppof: Dict[int, List[Dict]],
    all_rapa: Dict[int, List[Dict]],
    sizes: List[int],
):
    """Print statistical tests for each grid size and each comparison pair."""
    print(f"  STATISTICAL TESTS:")
    print(f"  {'-' * 68}")

    for size in sizes:
        ppo_list = all_ppo.get(size, [])
        ppof_list = all_ppof.get(size, [])
        rapa_list = all_rapa.get(size, [])

        ppo_srs = [r["eval_sr"] for r in ppo_list] if ppo_list else []
        ppof_srs = [r["eval_sr"] for r in ppof_list] if ppof_list else []
        rapa_srs = [r["eval_sr"] for r in rapa_list] if rapa_list else []

        # PPO vs RAPA
        if ppo_srs and rapa_srs:
            report = compare_variants(
                "RAPA", rapa_srs, "PPO", ppo_srs,
                metric_name=f"SR {size}x{size} RAPA-vs-PPO",
            )
            print(f"  {format_comparison(report)}")

        # PPO+F vs RAPA (key comparison: architecture value)
        if ppof_srs and rapa_srs:
            report = compare_variants(
                "RAPA", rapa_srs, "PPO+F", ppof_srs,
                metric_name=f"SR {size}x{size} RAPA-vs-PPO+F",
            )
            print(f"  {format_comparison(report)}")

        # PPO vs PPO+F (feature value)
        if ppo_srs and ppof_srs:
            report = compare_variants(
                "PPO+F", ppof_srs, "PPO", ppo_srs,
                metric_name=f"SR {size}x{size} PPO+F-vs-PPO",
            )
            print(f"  {format_comparison(report)}")

    print()


def print_generalization(gen_results: Dict):
    """Print generalization table (three-way)."""
    if not gen_results:
        return

    print(f"  GENERALIZATION (trained on 6x6):")
    print(f"  {'Grid':>5s} | {'PPO SR':>7s} | {'PPO+F SR':>9s} | "
          f"{'RAPA SR':>8s} | {'RAPA-PPO':>8s} | {'RAPA-PPO+F':>10s}")
    print(f"  {'-' * 5}-+-{'-' * 7}-+-{'-' * 9}-+-"
          f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    for eval_size in sorted(gen_results.keys()):
        g = gen_results[eval_size]
        ppo_sr = g.get("ppo_sr", 0.0)
        ppof_sr = g.get("ppof_sr", 0.0)
        rapa_sr = g.get("rapa_sr", 0.0)
        d1 = rapa_sr - ppo_sr
        d2 = rapa_sr - ppof_sr
        print(f"  {eval_size:>2d}x{eval_size:<2d} | {ppo_sr:>6.0%} | "
              f"{ppof_sr:>8.0%} | "
              f"{rapa_sr:>7.0%} | {d1:>+7.0%} | {d2:>+9.0%}")

    print()


# -- CSV Output ---------------------------------------------------

def save_csv(
    all_ppo: Dict[int, List[Dict]],
    all_ppof: Dict[int, List[Dict]],
    all_rapa: Dict[int, List[Dict]],
    gen_results: Dict,
) -> Path:
    """Save three-way results to CSV."""
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"ppo_vs_rapa_{ts}.csv"

    fieldnames = [
        "method", "grid_size", "seed", "phase",
        "env_steps", "episodes", "sr", "avg_steps_ok",
        "convergence_80pct", "params",
    ]

    rows = []
    all_sizes = sorted(set(
        list(all_ppo.keys()) + list(all_ppof.keys())
        + list(all_rapa.keys())))

    for size in all_sizes:
        for method, results in [("PPO", all_ppo.get(size, [])),
                                 ("PPO+F", all_ppof.get(size, [])),
                                 ("RAPA", all_rapa.get(size, []))]:
            for r in results:
                rows.append({
                    "method": method,
                    "grid_size": size,
                    "seed": r["seed"],
                    "phase": "train",
                    "env_steps": r["total_env_steps"],
                    "episodes": r.get("total_episodes", ""),
                    "sr": round(r["eval_sr"], 4),
                    "avg_steps_ok": round(r["eval_avg_steps_ok"], 1),
                    "convergence_80pct": r["convergence_steps_80"],
                    "params": r.get("params", 0),
                })

    # Generalization rows
    for eval_size, g in gen_results.items():
        for key, method in [("ppo_per_seed", "PPO"),
                             ("ppof_per_seed", "PPO+F"),
                             ("rapa_per_seed", "RAPA")]:
            for seed_sr in g.get(key, []):
                rows.append({
                    "method": method,
                    "grid_size": eval_size,
                    "seed": seed_sr.get("seed", ""),
                    "phase": "generalize_from_6",
                    "env_steps": "",
                    "episodes": EVAL_EPISODES,
                    "sr": round(seed_sr.get("sr", 0.0), 4),
                    "avg_steps_ok": round(
                        seed_sr.get("avg_steps_ok", 0.0), 1),
                    "convergence_80pct": "",
                    "params": "",
                })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Results saved to {path}")
    return path


# -- Benchmark Orchestration --------------------------------------

def run_benchmark(
    sizes: List[int],
    n_seeds: int,
    ppo_total_steps: int,
    rapa_max_per_stage: int,
    ppo_only: bool = False,
    rapa_only: bool = False,
    no_ppo_features: bool = False,
    ppo_log_interval: int = 10_000,
    ppo_checkpoint_interval: int = 50_000,
    n_envs: int = 8,
    trust_threshold: float = 0.20,
):
    """Full three-way PPO vs PPO+Features vs RAPA benchmark."""
    seeds = SEED_BASES[:n_seeds]

    run_ppo = not rapa_only
    run_ppof = not rapa_only and not ppo_only and not no_ppo_features
    run_rapa = not ppo_only

    print(f"\n{'=' * 80}")
    print("  PPO vs PPO+Features vs RAPA Benchmark")
    print(f"  Grid sizes: {sizes}")
    print(f"  Seeds: {n_seeds} ({seeds})")
    print(f"  PPO budget: {ppo_total_steps:,} steps/size")
    print(f"  RAPA: {rapa_max_per_stage} eps/stage")
    arms = []
    if run_ppo:
        arms.append("PPO")
    if run_ppof:
        arms.append("PPO+F")
    if run_rapa:
        arms.append("RAPA")
    print(f"  Arms: {', '.join(arms)}")
    print(f"{'=' * 80}")

    all_ppo: Dict[int, List[Dict]] = {}
    all_ppof: Dict[int, List[Dict]] = {}
    all_rapa: Dict[int, List[Dict]] = {}
    ppo_trainers_6x6: Dict[int, object] = {}
    ppof_trainers_6x6: Dict[int, object] = {}

    t0 = time.time()

    # -- Phase 1: Training --
    print(f"\n-- Phase 1: Training {'-' * 56}")

    for size in sizes:
        print(f"\n  [GRID {size}x{size}]")

        for i, seed in enumerate(seeds):
            # Vanilla PPO
            if run_ppo:
                print(f"\n  --- PPO [{size}x{size}] seed={seed} "
                      f"({i+1}/{n_seeds}) ---")
                ppo_result = run_ppo_single(
                    size=size,
                    seed=seed,
                    total_steps=ppo_total_steps,
                    n_envs=n_envs,
                    log_interval=ppo_log_interval,
                    checkpoint_interval=ppo_checkpoint_interval,
                )
                all_ppo.setdefault(size, []).append(ppo_result)
                if size == 6:
                    ppo_trainers_6x6[seed] = ppo_result["trainer"]
                print(f"  PPO [{size}x{size}] seed={seed} DONE: "
                      f"SR={ppo_result['eval_sr']:.0%}, "
                      f"steps={ppo_result['total_env_steps']:,}")

            # PPO+Features
            if run_ppof:
                print(f"\n  --- PPO+F [{size}x{size}] seed={seed} "
                      f"({i+1}/{n_seeds}) ---")
                ppof_result = run_ppo_features_single(
                    size=size,
                    seed=seed,
                    total_steps=ppo_total_steps,
                    n_envs=n_envs,
                    log_interval=ppo_log_interval,
                    checkpoint_interval=ppo_checkpoint_interval,
                )
                all_ppof.setdefault(size, []).append(ppof_result)
                if size == 6:
                    ppof_trainers_6x6[seed] = ppof_result["trainer"]
                print(f"  PPO+F [{size}x{size}] seed={seed} DONE: "
                      f"SR={ppof_result['eval_sr']:.0%}, "
                      f"steps={ppof_result['total_env_steps']:,}")

            # RAPA
            if run_rapa:
                print(f"\n  --- RAPA [{size}x{size}] seed={seed} "
                      f"({i+1}/{n_seeds}) ---")
                rapa_result = run_rapa_single(
                    size=size,
                    seed=seed,
                    max_per_stage=rapa_max_per_stage,
                    trust_threshold=trust_threshold,
                )
                all_rapa.setdefault(size, []).append(rapa_result)
                print(f"  RAPA [{size}x{size}] seed={seed} DONE: "
                      f"SR={rapa_result['eval_sr']:.0%}, "
                      f"steps={rapa_result['total_env_steps']:,}")

            # Early check after first seed
            if i == 0:
                has_ppo = bool(all_ppo.get(size))
                has_ppof = bool(all_ppof.get(size))
                has_rapa = bool(all_rapa.get(size))
                if (int(has_ppo) + int(has_ppof) + int(has_rapa)) >= 2:
                    print_early_check(
                        all_ppo[size][-1] if has_ppo else None,
                        all_ppof[size][-1] if has_ppof else None,
                        all_rapa[size][-1] if has_rapa else None,
                        size, seed)

    # -- Phase 2: Generalization --
    gen_results: Dict[int, Dict] = {}

    if 6 in sizes and len(sizes) > 1:
        print(f"\n-- Phase 2: Generalization (trained on 6x6) "
              f"{'-' * 34}")

        gen_sizes = [s for s in [8, 16] if s in sizes]

        for eval_size in gen_sizes:
            ppo_gen_per_seed = []
            ppof_gen_per_seed = []
            rapa_gen_per_seed = []

            for seed in seeds:
                # PPO zero-shot
                if run_ppo and seed in ppo_trainers_6x6:
                    trainer = ppo_trainers_6x6[seed]
                    gen_eval = evaluate_ppo_zero_shot(
                        trainer, eval_size, EVAL_EPISODES)
                    ppo_gen_per_seed.append({"seed": seed, **gen_eval})
                    print(f"  PPO  6x6->{eval_size}x{eval_size} "
                          f"seed={seed}: SR={gen_eval['sr']:.0%}")

                # PPO+F zero-shot
                if run_ppof and seed in ppof_trainers_6x6:
                    trainer = ppof_trainers_6x6[seed]
                    gen_eval = evaluate_ppo_features_zero_shot(
                        trainer, eval_size, EVAL_EPISODES)
                    ppof_gen_per_seed.append({"seed": seed, **gen_eval})
                    print(f"  PPO+F 6x6->{eval_size}x{eval_size} "
                          f"seed={seed}: SR={gen_eval['sr']:.0%}")

                # RAPA zero-shot
                if run_rapa:
                    rapa_gen = evaluate_rapa_zero_shot(
                        eval_size, seed, EVAL_EPISODES)
                    rapa_gen_per_seed.append({"seed": seed, **rapa_gen})
                    print(f"  RAPA 6x6->{eval_size}x{eval_size} "
                          f"seed={seed}: SR={rapa_gen['sr']:.0%}")

            ppo_srs = [r["sr"] for r in ppo_gen_per_seed]
            ppof_srs = [r["sr"] for r in ppof_gen_per_seed]
            rapa_srs = [r["sr"] for r in rapa_gen_per_seed]

            gen_results[eval_size] = {
                "ppo_sr": stat_mean(ppo_srs) if ppo_srs else 0.0,
                "ppof_sr": stat_mean(ppof_srs) if ppof_srs else 0.0,
                "rapa_sr": stat_mean(rapa_srs) if rapa_srs else 0.0,
                "ppo_per_seed": ppo_gen_per_seed,
                "ppof_per_seed": ppof_gen_per_seed,
                "rapa_per_seed": rapa_gen_per_seed,
            }

    # Close trainers
    for trainer in ppo_trainers_6x6.values():
        if hasattr(trainer, "close"):
            trainer.close()
    for trainer in ppof_trainers_6x6.values():
        if hasattr(trainer, "close"):
            trainer.close()
    for size_results in all_ppo.values():
        for r in size_results:
            t = r.pop("trainer", None)
            if t is not None and hasattr(t, "close"):
                t.close()
    for size_results in all_ppof.values():
        for r in size_results:
            t = r.pop("trainer", None)
            if t is not None and hasattr(t, "close"):
                t.close()

    elapsed = time.time() - t0

    # -- Phase 3: Results --
    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK COMPLETE: {elapsed:.0f}s elapsed")
    print(f"{'=' * 80}")

    print_summary(all_ppo, all_ppof, all_rapa, sizes)
    if sum([bool(all_ppo), bool(all_ppof), bool(all_rapa)]) >= 2:
        print_statistics(all_ppo, all_ppof, all_rapa, sizes)
    print_generalization(gen_results)
    save_csv(all_ppo, all_ppof, all_rapa, gen_results)


# -- CLI ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PPO vs RAPA Comparison Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 16],
                        help="Grid sizes (default: 6 8 16)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--ppo-steps", type=int, default=500_000,
                        help="PPO training steps per grid (default: 500000)")
    parser.add_argument("--rapa-max-per-stage", type=int, default=150,
                        help="RAPA episodes per stage (default: 150)")
    parser.add_argument("--ppo-only", action="store_true",
                        help="Run PPO only (skip RAPA)")
    parser.add_argument("--rapa-only", action="store_true",
                        help="Run RAPA only (skip PPO and PPO+F)")
    parser.add_argument("--no-ppo-features", action="store_true",
                        help="Skip PPO+Features arm")
    parser.add_argument("--ppo-log-interval", type=int, default=10_000,
                        help="PPO logging interval (default: 10000)")
    parser.add_argument("--ppo-checkpoint-interval", type=int, default=50_000,
                        help="PPO checkpoint interval (default: 50000)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="PPO parallel envs (default: 8)")
    parser.add_argument("--trust-threshold", type=float, default=0.20,
                        help="RAPA trust threshold (default: 0.20)")
    args = parser.parse_args()

    run_benchmark(
        sizes=args.sizes,
        n_seeds=args.n_seeds,
        ppo_total_steps=args.ppo_steps,
        rapa_max_per_stage=args.rapa_max_per_stage,
        ppo_only=args.ppo_only,
        rapa_only=args.rapa_only,
        no_ppo_features=args.no_ppo_features,
        ppo_log_interval=args.ppo_log_interval,
        ppo_checkpoint_interval=args.ppo_checkpoint_interval,
        n_envs=args.n_envs,
        trust_threshold=args.trust_threshold,
    )


if __name__ == "__main__":
    main()

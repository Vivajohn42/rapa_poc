"""
Shared test runner for RAPA ablation studies.

Provides:
- Batch episode execution with configurable variants and goal modes
- CSV output with auto-timestamped filenames
- Aggregation with statistical analysis via eval.stats
- Seed management for reproducibility
- Progress reporting via tqdm (if available)
"""

import csv
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Sequence

from eval.stats import (
    confidence_interval_95,
    confidence_interval_proportion,
    compare_variants,
    format_comparison,
    mean,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class RunConfig:
    """Configuration for a batch ablation run."""
    name: str                         # e.g. "ablation", "hidden_goal", "valence_swap"
    variants: List[str]               # e.g. ["baseline_mono", "modular_nod", ...]
    goal_modes: List[str]             # e.g. ["seek", "avoid"]
    n_episodes: int = 50              # episodes per variant × goal_mode
    max_steps: int = 50
    seed_start: Optional[int] = None  # if set, seeds are seed_start, seed_start+1, ...
    out_dir: str = "runs"
    reference_variant: Optional[str] = None  # variant to compare others against in stats


def run_batch(
    config: RunConfig,
    episode_fn: Callable,
    extra_csv_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a batch of episodes and write results to CSV.

    Args:
        config: RunConfig with variants, goal_modes, n_episodes, etc.
        episode_fn: Callable(variant, goal_mode, max_steps, seed) -> dataclass result.
                    The result must be a dataclass with at least: variant, goal_mode, success, steps.
                    If the callable accepts a `seed` kwarg, it will be passed.
        extra_csv_fields: Additional field names to extract from the result dataclass.
                         If None, all fields from the first result are used.

    Returns:
        Dict with keys: "results", "csv_path", "aggregates"
    """
    Path(config.out_dir).mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = f"{config.out_dir}/{config.name}_{run_id}.csv"

    results = []
    total = len(config.variants) * len(config.goal_modes) * config.n_episodes

    # Determine if episode_fn accepts seed parameter
    import inspect
    sig = inspect.signature(episode_fn)
    accepts_seed = "seed" in sig.parameters

    iterator = range(total)
    if tqdm:
        iterator = tqdm(iterator, desc=config.name, total=total)

    idx = 0
    for v in config.variants:
        for g in config.goal_modes:
            for i in range(config.n_episodes):
                seed = (config.seed_start + idx) if config.seed_start is not None else None

                kwargs = {"variant": v, "goal_mode": g, "max_steps": config.max_steps}
                if accepts_seed and seed is not None:
                    kwargs["seed"] = seed

                result = episode_fn(**kwargs)
                results.append(result)

                if tqdm:
                    next(iter(iterator), None)  # advance tqdm
                idx += 1

    # Determine CSV fields from first result
    if results:
        fields = [f.name for f in dataclasses.fields(results[0])]
    else:
        fields = []

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            row = [getattr(r, f) for f in fields]
            w.writerow(row)

    print(f"\nWrote {len(results)} episode results to: {csv_path}")

    # Aggregation
    aggregates = _aggregate(results, config)

    # Statistical comparisons
    if config.reference_variant:
        _print_comparisons(results, config)

    return {
        "results": results,
        "csv_path": csv_path,
        "aggregates": aggregates,
    }


def _aggregate(results: list, config: RunConfig) -> Dict[str, Dict[str, Any]]:
    """Compute per-variant/goal_mode aggregate statistics."""
    agg = {}

    for v in config.variants:
        for g in config.goal_modes:
            subset = [r for r in results if r.variant == v and r.goal_mode == g]
            if not subset:
                continue

            key = f"{v}/{g}"
            n = len(subset)

            # Success rate with Wilson CI
            successes = sum(1 for r in subset if r.success)
            sr, sr_lo, sr_hi = confidence_interval_proportion(successes, n)

            # Steps with t-CI
            steps_vals = [float(r.steps) for r in subset]
            steps_m, steps_lo, steps_hi = confidence_interval_95(steps_vals)

            # Total reward with t-CI
            reward_vals = [float(r.total_reward) for r in subset]
            reward_m, reward_lo, reward_hi = confidence_interval_95(reward_vals)

            entry = {
                "variant": v,
                "goal_mode": g,
                "n": n,
                "success_rate": sr,
                "success_ci": (sr_lo, sr_hi),
                "steps_mean": steps_m,
                "steps_ci": (steps_lo, steps_hi),
                "reward_mean": reward_m,
                "reward_ci": (reward_lo, reward_hi),
            }

            # Extract additional numeric fields
            sample = subset[0]
            for f in dataclasses.fields(sample):
                if f.name in ("variant", "goal_mode", "success", "steps", "total_reward"):
                    continue
                vals = []
                for r in subset:
                    val = getattr(r, f.name)
                    if isinstance(val, (int, float)):
                        vals.append(float(val))
                    elif isinstance(val, bool):
                        vals.append(1.0 if val else 0.0)
                if vals:
                    m, lo, hi = confidence_interval_95(vals)
                    entry[f"{f.name}_mean"] = m
                    entry[f"{f.name}_ci"] = (lo, hi)

            agg[key] = entry

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Aggregates for: {config.name} (n={config.n_episodes} per cell)")
    print(f"{'='*80}")
    print(f"  {'variant':<22s} {'mode':<6s} {'sr':>6s} {'sr_ci':>16s} {'steps':>7s} {'steps_ci':>16s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*16} {'-'*7} {'-'*16}")

    for key in sorted(agg.keys()):
        a = agg[key]
        print(
            f"  {a['variant']:<22s} {a['goal_mode']:<6s} "
            f"{a['success_rate']:>6.3f} [{a['success_ci'][0]:.3f},{a['success_ci'][1]:.3f}] "
            f"{a['steps_mean']:>7.1f} [{a['steps_ci'][0]:.1f},{a['steps_ci'][1]:.1f}]"
        )

    return agg


def _print_comparisons(results: list, config: RunConfig):
    """Print statistical comparisons against the reference variant."""
    ref = config.reference_variant
    others = [v for v in config.variants if v != ref]

    if not others:
        return

    print(f"\n{'='*80}")
    print(f"  Statistical comparisons vs reference: {ref}")
    print(f"{'='*80}")

    for g in config.goal_modes:
        ref_subset = [r for r in results if r.variant == ref and r.goal_mode == g]
        if not ref_subset:
            continue

        print(f"\n  --- goal_mode = {g} ---")

        for other in others:
            other_subset = [r for r in results if r.variant == other and r.goal_mode == g]
            if not other_subset:
                continue

            # Compare success rate
            ref_sr = [1.0 if r.success else 0.0 for r in ref_subset]
            other_sr = [1.0 if r.success else 0.0 for r in other_subset]
            report = compare_variants(ref, ref_sr, other, other_sr, "success_rate", is_proportion=True)
            print(format_comparison(report))

            # Compare steps
            ref_steps = [float(r.steps) for r in ref_subset]
            other_steps = [float(r.steps) for r in other_subset]
            report = compare_variants(ref, ref_steps, other, other_steps, "steps")
            print(format_comparison(report))

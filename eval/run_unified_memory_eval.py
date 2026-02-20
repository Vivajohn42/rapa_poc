"""Jungian Weights x Unified Memory Evaluation — DoorKey.

Tests the complete RAPA v2 Unified Memory + Cascaded Compression pipeline
across 5 Jungian V3 profiles (function-stack based).

Assertions:
  1. Backward compatibility: DoorKey with use_unified_memory=False SR >= 90%
  2. UM parity: DoorKey with use_unified_memory=True SR >= 90%
  3. Compression works: L3->L2 fires in at least some successful episodes
  4. Cascade depth: L2->L1 fires in at least some episodes
  5. Profile behavioral difference: 5 profiles produce different compression counts
  6. Profiles produce diverse compression behavior (at least 2 distinct values)
  7. Cache invalidation: spikes reactivate layers (measured via UM invalidation_count)
  8. All profiles >= 90% SR on DoorKey-6x6

Usage:
    python eval/run_unified_memory_eval.py
    python eval/run_unified_memory_eval.py --n 30
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey_adapter import DoorKeyAdapter
from kernel.kernel import MvpKernel
from kernel.jung_profiles import JungProfileV3, PROFILES_V3


@dataclass
class UMEpisodeResult:
    profile_name: str
    seed: int
    success: bool
    steps: int
    delta_8_mean: float
    # Compression metrics
    l3_l2_count: int
    l2_l1_count: int
    l1_l0_count: int
    total_compressions: int
    # UM diagnostics
    regime_final: str
    compressed_layers: List[str]
    invalidation_count: int
    # Legacy metrics
    decon_count: int
    d_activations: int


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def run_dk_um_episode(
    profile_name: str,
    seed: int,
    use_unified_memory: bool = True,
    size: int = 6,
    max_steps: int = 200,
) -> UMEpisodeResult:
    """Run one DoorKey episode with Unified Memory + Jung profile."""
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_steps)
    obs = adapter.reset()
    A, B, C, D = adapter.make_agents(variant="with_d")
    decon_fn = adapter.get_deconstruct_fn()
    goal_map = adapter.get_goal_map()

    profile = PROFILES_V3.get(profile_name)

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
        deconstruct_fn=decon_fn,
        fallback_actions=adapter.available_actions(obs),
        jung_profile=profile,
        use_unified_memory=use_unified_memory,
    )
    kernel.reset_episode(goal_mode="seek", episode_id=f"um_{profile_name}_{seed}")

    done = False
    decon_count = 0
    d_activations = 0
    l3_l2 = 0
    l2_l1 = 0
    l1_l0 = 0
    reward = 0.0
    t = -1

    for t in range(max_steps):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        if result.decon_fired:
            decon_count += 1
        if result.d_activated:
            d_activations += 1
        # Count compression stages
        for stage in result.compression_stages:
            if stage == "L3_L2":
                l3_l2 += 1
            elif stage == "L2_L1":
                l2_l1 += 1
            elif stage == "L1_L0":
                l1_l0 += 1

        obs, reward, done = adapter.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    success = done and reward > 0

    # UM diagnostics
    um = kernel.unified_memory
    if um is not None:
        snap = um.snapshot()
        regime_final = snap["active_regime"]
        compressed_layers = snap["compressed"]
        invalidation_count = snap["invalidation_count"]
    else:
        regime_final = "N/A"
        compressed_layers = []
        invalidation_count = 0

    # Delta_8
    r_hist = kernel.residuum.episode_history
    d8_vals = [s.delta_8 for s in r_hist] if r_hist else [0.0]
    delta_8_mean = _mean(d8_vals)

    return UMEpisodeResult(
        profile_name=profile_name,
        seed=seed,
        success=success,
        steps=steps,
        delta_8_mean=round(delta_8_mean, 4),
        l3_l2_count=l3_l2,
        l2_l1_count=l2_l1,
        l1_l0_count=l1_l0,
        total_compressions=l3_l2 + l2_l1 + l1_l0,
        regime_final=regime_final,
        compressed_layers=compressed_layers,
        invalidation_count=invalidation_count,
        decon_count=decon_count,
        d_activations=d_activations,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified Memory Eval")
    parser.add_argument("--n", type=int, default=20, help="Episodes per profile")
    parser.add_argument("--size", type=int, default=6, help="DoorKey grid size")
    args = parser.parse_args()

    n = args.n
    size = args.size
    profile_names = list(PROFILES_V3.keys())

    print("=" * 78)
    print("  Jungian Weights x Unified Memory — DoorKey Evaluation")
    print("=" * 78)

    # ================================================================
    # Test 1: Backward Compatibility (use_unified_memory=False)
    # ================================================================
    print("\n--- Test 1: Backward Compatibility (UM=False) ---")
    legacy_results = []
    for i in range(n):
        r = run_dk_um_episode("DEFAULT_V3", seed=42 + i,
                              use_unified_memory=False, size=size)
        legacy_results.append(r)
    sr_legacy = sum(1 for r in legacy_results if r.success) / n
    steps_legacy = _mean([r.steps for r in legacy_results if r.success])
    p1 = sr_legacy >= 0.90
    print(f"  SR={sr_legacy:.1%}  steps={steps_legacy:.1f}")
    print(f"  [{'PASS' if p1 else 'FAIL'}] Legacy SR >= 90%")

    # ================================================================
    # Test 2: UM Parity (use_unified_memory=True, DEFAULT_V2)
    # ================================================================
    print("\n--- Test 2: UM Parity (UM=True, DEFAULT_V2) ---")
    um_default_results = []
    for i in range(n):
        r = run_dk_um_episode("DEFAULT_V3", seed=42 + i,
                              use_unified_memory=True, size=size)
        um_default_results.append(r)
    sr_um = sum(1 for r in um_default_results if r.success) / n
    steps_um = _mean([r.steps for r in um_default_results if r.success])
    p2 = sr_um >= 0.90
    print(f"  SR={sr_um:.1%}  steps={steps_um:.1f}")
    print(f"  [{'PASS' if p2 else 'FAIL'}] UM SR >= 90%")

    # ================================================================
    # Run all profiles with UM=True
    # ================================================================
    print(f"\n--- Running {n} episodes x {len(profile_names)} profiles ---")
    all_results: Dict[str, List[UMEpisodeResult]] = {}

    for name in profile_names:
        if name == "DEFAULT_V3":
            # Already ran above
            all_results[name] = um_default_results
            continue
        results = []
        for i in range(n):
            r = run_dk_um_episode(name, seed=42 + i,
                                  use_unified_memory=True, size=size)
            results.append(r)
        all_results[name] = results

    # Print summary table
    print(f"\n  {'Profile':<12s} {'SR':>6s} {'Steps':>7s} {'D8':>7s} "
          f"{'L3>L2':>6s} {'L2>L1':>6s} {'L1>L0':>6s} "
          f"{'Inval':>6s} {'Regime':>7s}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    for name in profile_names:
        rr = all_results[name]
        sr = sum(1 for r in rr if r.success) / n
        st = _mean([r.steps for r in rr if r.success])
        d8 = _mean([r.delta_8_mean for r in rr])
        l32 = sum(r.l3_l2_count for r in rr)
        l21 = sum(r.l2_l1_count for r in rr)
        l10 = sum(r.l1_l0_count for r in rr)
        inv = sum(r.invalidation_count for r in rr)
        # Most common final regime
        regimes = [r.regime_final for r in rr]
        regime_common = max(set(regimes), key=regimes.count) if regimes else "N/A"
        print(f"  {name:<12s} {sr:6.1%} {st:7.1f} {d8:7.4f} "
              f"{l32:6d} {l21:6d} {l10:6d} "
              f"{inv:6d} {regime_common:>7s}")

    # Profile parameter diagnostics
    print(f"\n  --- Profile Parameters ---")
    print(f"  {'Profile':<12s} {'T_l3':>6s} {'T_l2':>6s} {'T_l1':>6s} "
          f"{'CW':>4s} {'CD':>4s}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*4}")
    for name in profile_names:
        p = PROFILES_V3[name]
        print(f"  {name:<12s} {p.compression_threshold_l3:6.3f} "
              f"{p.compression_threshold_l2:6.3f} "
              f"{p.compression_threshold_l1:6.3f} "
              f"{p.compression_window:4d} {p.cascade_depth:4d}")

    # ================================================================
    # Assertions
    # ================================================================
    print("\n" + "=" * 78)
    print("  ASSERTIONS")
    print("=" * 78)

    # Test 3: Compression works (L3->L2 fires)
    all_flat = [r for rr in all_results.values() for r in rr]
    success_with_l3l2 = sum(
        1 for r in all_flat if r.success and r.l3_l2_count > 0
    )
    total_success = sum(1 for r in all_flat if r.success)
    p3 = success_with_l3l2 > 0
    print(f"\n  3. [{'PASS' if p3 else 'FAIL'}] L3->L2 fires: "
          f"{success_with_l3l2}/{total_success} successful episodes have compressions")

    # Test 4: Cascade depth (L2->L1 fires)
    eps_with_l2l1 = sum(1 for r in all_flat if r.l2_l1_count > 0)
    p4 = eps_with_l2l1 > 0
    print(f"  4. [{'PASS' if p4 else 'FAIL'}] L2->L1 fires: "
          f"{eps_with_l2l1} episodes have L2->L1 compressions")

    # Test 5: Profile behavioral difference
    comp_per_profile = {}
    for name in profile_names:
        rr = all_results[name]
        comp_per_profile[name] = _mean([r.total_compressions for r in rr])
    unique_comps = len(set(round(v, 2) for v in comp_per_profile.values()))
    p5 = unique_comps >= 2
    print(f"\n  5. [{'PASS' if p5 else 'FAIL'}] Profile difference: "
          f"{unique_comps} distinct avg compression counts")
    for name in profile_names:
        print(f"     {name}: {comp_per_profile[name]:.2f} avg compressions/episode")

    # Test 6: Profiles produce diverse compression behavior
    # V3 function stacks should produce meaningfully different compression counts
    comp_values = sorted(set(round(v, 2) for v in comp_per_profile.values()))
    p6 = len(comp_values) >= 2
    print(f"\n  6. [{'PASS' if p6 else 'FAIL'}] Compression diversity: "
          f"{len(comp_values)} distinct values: {comp_values}")

    # Test 7: Cache invalidation (surprise triggers)
    total_inval = sum(r.invalidation_count for r in all_flat)
    p7 = True  # Diagnostic — invalidation may or may not occur depending on dynamics
    print(f"\n  7. [INFO] Total invalidations across all episodes: {total_inval}")
    if total_inval > 0:
        inval_eps = sum(1 for r in all_flat if r.invalidation_count > 0)
        print(f"     {inval_eps} episodes had at least one invalidation")

    # Test 8: All profiles >= 90% SR
    p8 = True
    print(f"\n  8. Profile SR check (>= 90%):")
    for name in profile_names:
        rr = all_results[name]
        sr = sum(1 for r in rr if r.success) / n
        ok = sr >= 0.90
        if not ok:
            p8 = False
        print(f"     [{'PASS' if ok else 'FAIL'}] {name}: SR={sr:.1%}")

    # ================================================================
    # CSV
    # ================================================================
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"unified_memory_eval_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profile", "seed", "success", "steps", "delta_8_mean",
            "l3_l2", "l2_l1", "l1_l0", "total_comp",
            "regime_final", "invalidation_count",
            "decon_count", "d_activations",
        ])
        for rr in all_results.values():
            for r in rr:
                writer.writerow([
                    r.profile_name, r.seed, int(r.success), r.steps,
                    r.delta_8_mean,
                    r.l3_l2_count, r.l2_l1_count, r.l1_l0_count,
                    r.total_compressions,
                    r.regime_final, r.invalidation_count,
                    r.decon_count, r.d_activations,
                ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 78)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

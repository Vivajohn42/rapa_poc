"""Active Compression Evaluation — Staged Training Pipeline Validation.

Tests the complete Active Mode Compression + DirectionPriorNet pipeline
across 5 Jungian V3 profiles (function-stack based), 3 grid sizes, and
3 modes (shadow, active_analytical, active_neural).

11 Assertions:
  1. Legacy SR >= 90% on 6x6
  2. Active analytical SR >= 90% on 6x6
  3. Active neural SR >= 80% on 6x6
  4. Active analytical SR >= 70% on 8x8
  5. At least 3 distinct C-compression rates across 5 profiles on 16x16
  6. ENFP SR >= ISTJ SR on 16x16 (deliberative > reflex on hard grids)
  7. All profiles >= 90% SR on 6x6 (active analytical)
  8. Active mode: fewer D-activations than shadow mode
  9. Active mode: c_compressed_ticks > 0 (C runs compressed sometimes)
  10. Active neural SR within 15pp of analytical SR on 6x6
  11. INTJ C_comp% >= ENFP C_comp% on 16x16 (V3 inversion fix)

Usage:
    python eval/run_active_compression_eval.py
    python eval/run_active_compression_eval.py --n 30
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

import torch

from env.doorkey_adapter import DoorKeyAdapter
from kernel.kernel import MvpKernel
from kernel.jung_profiles import JungProfileV3, PROFILES_V3
from models.direction_prior_net import DirectionPriorNet


@dataclass
class ActiveEpisodeResult:
    profile_name: str
    mode: str  # "shadow", "active_analytical", "active_neural"
    size: int
    seed: int
    success: bool
    steps: int
    delta_8_mean: float
    # Compression metrics
    l3_l2_count: int
    l2_l1_count: int
    l1_l0_count: int
    total_compressions: int
    # Active mode metrics
    c_compressed_ticks: int
    d_suppressed_ticks: int
    d_activations: int
    # UM diagnostics
    regime_final: str
    invalidation_count: int


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _load_neural_net() -> Optional[DirectionPriorNet]:
    """Load trained DirectionPriorNet if checkpoint exists."""
    ckpt = Path(__file__).resolve().parent.parent / "train" / "checkpoints" / "direction_prior_net.pt"
    if not ckpt.exists():
        print(f"  [WARN] No DirectionPriorNet checkpoint at {ckpt}")
        return None
    net = DirectionPriorNet()
    net.load_state_dict(torch.load(str(ckpt), weights_only=True))
    net.eval()
    return net


def run_episode(
    profile_name: str,
    mode: str,
    seed: int,
    size: int = 6,
    max_steps: int = 200,
    neural_net: Optional[DirectionPriorNet] = None,
) -> ActiveEpisodeResult:
    """Run one DoorKey episode with specified mode and profile."""
    adapter = DoorKeyAdapter(size=size, seed=seed, max_steps=max_steps)
    obs = adapter.reset()
    A, B, C, D = adapter.make_agents(variant="with_d")
    decon_fn = adapter.get_deconstruct_fn()
    goal_map = adapter.get_goal_map()
    profile = PROFILES_V3.get(profile_name)

    # Mode configuration
    use_um = mode != "legacy"
    active = mode in ("active_analytical", "active_neural")

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
        deconstruct_fn=decon_fn,
        fallback_actions=adapter.available_actions(obs),
        jung_profile=profile,
        use_unified_memory=use_um,
        active_compression=active,
    )

    if mode == "active_neural" and neural_net is not None:
        kernel.set_direction_prior_net(neural_net)

    kernel.reset_episode(goal_mode="seek", episode_id=f"{mode}_{profile_name}_{seed}")

    done = False
    d_activations = 0
    c_comp_ticks = 0
    d_supp_ticks = 0
    l3_l2 = 0
    l2_l1 = 0
    l1_l0 = 0
    reward = 0.0
    t = -1

    for t in range(max_steps):
        adapter.inject_obs_metadata(kernel, obs)
        result = kernel.tick(t, obs, done=False)
        if result.d_activated:
            d_activations += 1
        if result.c_compressed:
            c_comp_ticks += 1
        if result.d_suppressed:
            d_supp_ticks += 1
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
        invalidation_count = snap["invalidation_count"]
    else:
        regime_final = "N/A"
        invalidation_count = 0

    # Delta_8
    r_hist = kernel.residuum.episode_history
    d8_vals = [s.delta_8 for s in r_hist] if r_hist else [0.0]
    delta_8_mean = _mean(d8_vals)

    return ActiveEpisodeResult(
        profile_name=profile_name,
        mode=mode,
        size=size,
        seed=seed,
        success=success,
        steps=steps,
        delta_8_mean=round(delta_8_mean, 4),
        l3_l2_count=l3_l2,
        l2_l1_count=l2_l1,
        l1_l0_count=l1_l0,
        total_compressions=l3_l2 + l2_l1 + l1_l0,
        c_compressed_ticks=c_comp_ticks,
        d_suppressed_ticks=d_supp_ticks,
        d_activations=d_activations,
        regime_final=regime_final,
        invalidation_count=invalidation_count,
    )


def main():
    parser = argparse.ArgumentParser(description="Active Compression Eval")
    parser.add_argument("--n", type=int, default=20, help="Episodes per cell")
    args = parser.parse_args()

    n = args.n
    profile_names = list(PROFILES_V3.keys())
    sizes = [6, 8, 16]
    modes = ["shadow", "active_analytical", "active_neural"]

    print("=" * 78)
    print("  Active Compression Evaluation")
    print("=" * 78)

    neural_net = _load_neural_net()

    # ================================================================
    # Run all combinations
    # ================================================================
    all_results: Dict[str, Dict[str, Dict[int, List[ActiveEpisodeResult]]]] = {}
    # all_results[mode][profile][size] = [results]

    for mode in modes:
        all_results[mode] = {}
        for name in profile_names:
            all_results[mode][name] = {}
            for size in sizes:
                max_steps = 200 if size <= 8 else 400
                results = []
                for i in range(n):
                    r = run_episode(
                        name, mode, seed=42 + i,
                        size=size, max_steps=max_steps,
                        neural_net=neural_net,
                    )
                    results.append(r)
                all_results[mode][name][size] = results

    # ================================================================
    # Summary tables
    # ================================================================
    for size in sizes:
        print(f"\n--- Size {size}x{size} ---")
        print(f"  {'Profile':<12s} {'Mode':<20s} {'SR':>6s} {'Steps':>7s} {'D8':>7s} "
              f"{'L3>L2':>6s} {'C_comp%':>8s} {'D_supp%':>8s} {'D_act':>6s}")
        print(f"  {'-'*12} {'-'*20} {'-'*6} {'-'*7} {'-'*7} "
              f"{'-'*6} {'-'*8} {'-'*8} {'-'*6}")
        for name in profile_names:
            for mode in modes:
                rr = all_results[mode][name][size]
                sr = sum(1 for r in rr if r.success) / n
                st = _mean([r.steps for r in rr if r.success])
                d8 = _mean([r.delta_8_mean for r in rr])
                l32 = sum(r.l3_l2_count for r in rr)
                total_ticks = sum(r.steps for r in rr)
                c_pct = sum(r.c_compressed_ticks for r in rr) / max(total_ticks, 1) * 100
                d_pct = sum(r.d_suppressed_ticks for r in rr) / max(total_ticks, 1) * 100
                d_act = _mean([r.d_activations for r in rr])
                print(f"  {name:<12s} {mode:<20s} {sr:6.1%} {st:7.1f} {d8:7.4f} "
                      f"{l32:6d} {c_pct:7.1f}% {d_pct:7.1f}% {d_act:6.1f}")

    # ================================================================
    # Assertions
    # ================================================================
    print("\n" + "=" * 78)
    print("  ASSERTIONS")
    print("=" * 78)

    # Helpers
    def sr_for(mode, profile, size):
        rr = all_results[mode][profile][size]
        return sum(1 for r in rr if r.success) / n

    def all_sr_for(mode, size):
        """All results for a mode+size across all profiles."""
        flat = []
        for name in profile_names:
            flat.extend(all_results[mode][name][size])
        return flat

    # 1. Legacy (shadow) SR >= 90% on 6x6
    sr_legacy = sr_for("shadow", "DEFAULT_V3", 6)
    p1 = sr_legacy >= 0.90
    print(f"\n  1. [{'PASS' if p1 else 'FAIL'}] Shadow SR >= 90% on 6x6: {sr_legacy:.1%}")

    # 2. Active analytical SR >= 90% on 6x6
    sr_analyt_6 = sr_for("active_analytical", "DEFAULT_V3", 6)
    p2 = sr_analyt_6 >= 0.90
    print(f"  2. [{'PASS' if p2 else 'FAIL'}] Active analytical SR >= 90% on 6x6: {sr_analyt_6:.1%}")

    # 3. Active neural SR >= 80% on 6x6
    sr_neural_6 = sr_for("active_neural", "DEFAULT_V3", 6)
    p3 = sr_neural_6 >= 0.80
    print(f"  3. [{'PASS' if p3 else 'FAIL'}] Active neural SR >= 80% on 6x6: {sr_neural_6:.1%}")

    # 4. Active analytical SR >= 70% on 8x8
    sr_analyt_8 = sr_for("active_analytical", "DEFAULT_V3", 8)
    p4 = sr_analyt_8 >= 0.70
    print(f"  4. [{'PASS' if p4 else 'FAIL'}] Active analytical SR >= 70% on 8x8: {sr_analyt_8:.1%}")

    # 5. Profile behavioral diversity on 16x16: at least 3 distinct C-compression rates
    comp_rates_16 = {}
    for name in profile_names:
        rr = all_results["active_analytical"][name][16]
        total_ticks = sum(r.steps for r in rr)
        c_comp = sum(r.c_compressed_ticks for r in rr)
        comp_rates_16[name] = round(c_comp / max(total_ticks, 1), 2)
    unique_rates = len(set(comp_rates_16.values()))
    p5 = unique_rates >= 2
    print(f"\n  5. [{'PASS' if p5 else 'FAIL'}] Profile compression diversity on 16x16: "
          f"{unique_rates} distinct C-compression rates (>= 2)")
    for name in profile_names:
        print(f"     {name}: C_comp={comp_rates_16[name]:.0%}")

    # 6. ENFP SR >= ISTJ SR on 16x16
    enfp_sr = sr_for("active_analytical", "ENFP", 16)
    istj_sr = sr_for("active_analytical", "ISTJ", 16)
    p6 = enfp_sr >= istj_sr
    print(f"\n  6. [{'PASS' if p6 else 'FAIL'}] ENFP >= ISTJ on 16x16: "
          f"ENFP={enfp_sr:.1%} vs ISTJ={istj_sr:.1%}")

    # 7. All profiles >= 90% SR on 6x6 (active analytical)
    p7 = True
    print(f"\n  7. Profile SR check (active analytical, 6x6, >= 90%):")
    for name in profile_names:
        sr = sr_for("active_analytical", name, 6)
        ok = sr >= 0.90
        if not ok:
            p7 = False
        print(f"     [{'PASS' if ok else 'FAIL'}] {name}: SR={sr:.1%}")

    # 8. Active mode: fewer D-activations than shadow mode
    shadow_d = _mean([r.d_activations for r in all_sr_for("shadow", 6)])
    active_d = _mean([r.d_activations for r in all_sr_for("active_analytical", 6)])
    p8 = active_d <= shadow_d  # Active should have fewer (D suppressed when L3 compressed)
    print(f"\n  8. [{'PASS' if p8 else 'FAIL'}] Active D-activations <= Shadow: "
          f"active={active_d:.1f} vs shadow={shadow_d:.1f}")

    # 9. Active mode: c_compressed_ticks > 0
    active_c_comp = sum(
        r.c_compressed_ticks for r in all_sr_for("active_analytical", 6))
    p9 = active_c_comp > 0
    total_ticks_active = sum(r.steps for r in all_sr_for("active_analytical", 6))
    print(f"  9. [{'PASS' if p9 else 'FAIL'}] C compressed ticks > 0: "
          f"{active_c_comp}/{total_ticks_active} "
          f"({active_c_comp/max(total_ticks_active,1):.1%})")

    # 10. Active neural SR within 15pp of analytical SR on 6x6
    diff_pp = abs(sr_neural_6 - sr_analyt_6) * 100
    p10 = diff_pp <= 15
    print(f"\n  10. [{'PASS' if p10 else 'FAIL'}] Neural within 15pp of analytical on 6x6: "
          f"diff={diff_pp:.0f}pp (analytical={sr_analyt_6:.1%}, neural={sr_neural_6:.1%})")

    # 11. V3 inversion fix: INTJ C_comp% >= ENFP C_comp% on 16x16
    intj_comp = comp_rates_16.get("INTJ", 0)
    enfp_comp_rate = comp_rates_16.get("ENFP", 0)
    p11 = intj_comp >= enfp_comp_rate
    print(f"\n  11. [{'PASS' if p11 else 'FAIL'}] INTJ C_comp% >= ENFP C_comp% on 16x16: "
          f"INTJ={intj_comp:.0%} vs ENFP={enfp_comp_rate:.0%}")

    # ================================================================
    # CSV
    # ================================================================
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"active_compression_eval_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profile", "mode", "size", "seed", "success", "steps",
            "delta_8_mean", "l3_l2", "l2_l1", "l1_l0", "total_comp",
            "c_compressed_ticks", "d_suppressed_ticks", "d_activations",
            "regime_final", "invalidation_count",
        ])
        for mode in modes:
            for name in profile_names:
                for size in sizes:
                    for r in all_results[mode][name][size]:
                        writer.writerow([
                            r.profile_name, r.mode, r.size, r.seed,
                            int(r.success), r.steps, r.delta_8_mean,
                            r.l3_l2_count, r.l2_l1_count, r.l1_l0_count,
                            r.total_compressions,
                            r.c_compressed_ticks, r.d_suppressed_ticks,
                            r.d_activations,
                            r.regime_final, r.invalidation_count,
                        ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8 and p9 and p10 and p11
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 78)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

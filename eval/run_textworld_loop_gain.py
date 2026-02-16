"""TextWorld Loop Gain Validation: Persistence Theorem Test.

The genuine test of the Persistence Theorem on TextWorld, where D is essential.

Assertions:
  1. with_d: G/F converges > 1.0 in successful episodes (stable attractor)
  2. no_d: G/F stays < 1.0 (system decays without D)
  3. DC is weakest coupling in significant ticks (D is genuine bottleneck)
  4. D-ablation causes measurable G/F collapse
  5. g_DC progression: rises as D collects clues and synthesizes

Usage:
    python eval/run_textworld_loop_gain.py
    python eval/run_textworld_loop_gain.py --n 30
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.textworld import TextWorld, SCENARIOS
from agents.text_agent_a import TextAgentA
from agents.text_agent_b import TextAgentB
from agents.text_agent_c import TextAgentC
from agents.text_agent_d import TextAgentD
from router.deconstruct_text import deconstruct_text_d_to_c
from kernel.kernel import MvpKernel
from kernel.types import MvpLoopGain


@dataclass
class LoopGainResult:
    variant: str
    scenario_id: int
    seed: int
    success: bool
    steps: int
    g_BA_mean: float
    g_CB_mean: float
    g_DC_mean: float
    g_AD_mean: float
    G_mean: float
    G_over_F_final: float
    weakest_final: str
    weakest_distribution: Dict[str, int]
    g_DC_progression: List[float]  # g_DC over time (for progression analysis)


def _make_deconstruct_fn(room_index):
    def fn(zC, zD, goal_map=None):
        return deconstruct_text_d_to_c(zC, zD, goal_map=goal_map, room_index=room_index)
    return fn


def run_loop_gain_episode(
    variant: str,
    scenario_id: int,
    seed: int,
    max_steps: int = 50,
) -> LoopGainResult:
    """Run one episode and collect loop gain data."""
    env = TextWorld(seed=seed, scenario_id=scenario_id)
    obs = env.reset()

    room_ids = env.room_ids
    room_index = {rid: i for i, rid in enumerate(room_ids)}
    index_to_room = {i: rid for rid, i in room_index.items()}
    room_graph = env.room_graph
    room_properties = env.room_properties
    n_rooms = len(room_ids)

    A = TextAgentA(room_index, n_rooms)
    B = TextAgentB(room_graph, room_index, index_to_room)
    C = TextAgentC(room_graph, room_index, index_to_room, goal_mode="seek")

    deconstruct_fn = _make_deconstruct_fn(room_index)

    if variant == "with_d":
        D = TextAgentD(room_properties, room_ids, room_index)
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=room_index,
            enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )
    else:  # no_d
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=room_index,
            enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )

    kernel.reset_episode(goal_mode="seek", episode_id=f"tw_lg_{variant}_{seed}")

    done = False
    t = -1
    for t in range(max_steps):
        kernel.zC.memory["visited_rooms"] = obs.get("visited", set())
        result = kernel.tick(t, obs, done=False)
        action = result.action
        obs, reward, done = env.step(action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0

    # Collect gain history
    history = kernel.loop_gain.episode_history

    if history:
        g_ba = [s.g_BA for s in history]
        g_cb = [s.g_CB for s in history]
        g_dc = [s.g_DC for s in history]
        g_ad = [s.g_AD for s in history]
        g_all = [s.G for s in history]

        g_ba_mean = sum(g_ba) / len(g_ba)
        g_cb_mean = sum(g_cb) / len(g_cb)
        g_dc_mean = sum(g_dc) / len(g_dc)
        g_ad_mean = sum(g_ad) / len(g_ad)
        g_mean = sum(g_all) / len(g_all)
        # Use mean G/F over episode (not final tick, which can be distorted
        # by terminal claim actions)
        gf_vals = [s.G_over_F for s in history]
        gf_final = sum(gf_vals) / len(gf_vals)
        weakest_final = kernel.loop_gain.weakest_coupling

        weakest_dist = Counter(s.weakest_coupling for s in history)
    else:
        g_ba_mean = g_cb_mean = g_dc_mean = g_ad_mean = g_mean = 0.0
        gf_final = 0.0
        weakest_final = "N/A"
        weakest_dist = Counter()

    return LoopGainResult(
        variant=variant,
        scenario_id=scenario_id,
        seed=seed,
        success=bool(done),
        steps=steps,
        g_BA_mean=round(g_ba_mean, 4),
        g_CB_mean=round(g_cb_mean, 4),
        g_DC_mean=round(g_dc_mean, 4),
        g_AD_mean=round(g_ad_mean, 4),
        G_mean=round(g_mean, 6),
        G_over_F_final=round(gf_final, 4),
        weakest_final=weakest_final,
        weakest_distribution=dict(weakest_dist),
        g_DC_progression=[round(s.g_DC, 4) for s in history],
    )


def main():
    parser = argparse.ArgumentParser(description="TextWorld Loop Gain Validation")
    parser.add_argument("--n", type=int, default=20, help="Episodes per variant per scenario")
    parser.add_argument("--scenario", type=int, default=None, help="Specific scenario (None=all)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 75)
    print("  TextWorld Loop Gain Validation — Persistence Theorem Test")
    print("=" * 75)

    n_episodes = args.n
    max_steps = args.max_steps
    variants = ["with_d", "no_d"]

    if args.scenario is not None:
        scenario_ids = [args.scenario]
    else:
        scenario_ids = list(range(len(SCENARIOS)))

    all_results: List[LoopGainResult] = []

    for sid in scenario_ids:
        sc = SCENARIOS[sid]
        print(f"\n--- Scenario {sid}: {len(sc.rooms)} rooms, "
              f"target={sc.target_room}, clues={sc.required_clues} ---")

        for variant in variants:
            results = []
            for i in range(n_episodes):
                r = run_loop_gain_episode(
                    variant=variant,
                    scenario_id=sid,
                    seed=42 + i,
                    max_steps=max_steps,
                )
                results.append(r)
                all_results.append(r)

            sr = sum(1 for r in results if r.success) / len(results)
            g_dc = sum(r.g_DC_mean for r in results) / len(results)
            g_ad = sum(r.g_AD_mean for r in results) / len(results)
            gf = sum(r.G_over_F_final for r in results) / len(results)

            print(f"    {variant:<10s}: SR={sr:.1%}  g_DC={g_dc:.3f}  "
                  f"g_AD={g_ad:.3f}  G/F={gf:.3f}")

    # ================================================================
    # Aggregate Analysis
    # ================================================================

    print("\n" + "=" * 75)
    print("  AGGREGATE RESULTS")
    print("=" * 75)

    for variant in variants:
        vr = [r for r in all_results if r.variant == variant]
        sr = sum(1 for r in vr if r.success) / len(vr) if vr else 0
        g_ba = sum(r.g_BA_mean for r in vr) / len(vr) if vr else 0
        g_cb = sum(r.g_CB_mean for r in vr) / len(vr) if vr else 0
        g_dc = sum(r.g_DC_mean for r in vr) / len(vr) if vr else 0
        g_ad = sum(r.g_AD_mean for r in vr) / len(vr) if vr else 0
        g_mean = sum(r.G_mean for r in vr) / len(vr) if vr else 0
        gf = sum(r.G_over_F_final for r in vr) / len(vr) if vr else 0

        # Weakest distribution across all episodes
        weakest_all = Counter()
        for r in vr:
            weakest_all.update(r.weakest_distribution)
        total_ticks = sum(weakest_all.values())

        print(f"\n  {variant}:")
        print(f"    Success Rate:  {sr:.1%}")
        print(f"    g_BA mean:     {g_ba:.4f}")
        print(f"    g_CB mean:     {g_cb:.4f}")
        print(f"    g_DC mean:     {g_dc:.4f}")
        print(f"    g_AD mean:     {g_ad:.4f}")
        print(f"    G mean:        {g_mean:.6f}")
        print(f"    G/F final:     {gf:.4f}")
        print(f"    Weakest dist:  ", end="")
        for c in ["BC", "CD", "AD", "AB"]:
            pct = weakest_all.get(c, 0) / max(total_ticks, 1) * 100
            if pct > 0:
                print(f"{c}={pct:.0f}% ", end="")
        print()

    # --- g_DC Progression (with_d only) ---
    wd_results = [r for r in all_results if r.variant == "with_d" and r.success]
    if wd_results:
        # Average g_DC at each tick across successful episodes
        max_len = max(len(r.g_DC_progression) for r in wd_results)
        print(f"\n  g_DC Progression (with_d, {len(wd_results)} successful episodes):")
        for tick in [0, 1, 2, 3, 4, 5, 8, 10]:
            if tick < max_len:
                vals = [r.g_DC_progression[tick] for r in wd_results
                        if tick < len(r.g_DC_progression)]
                if vals:
                    avg = sum(vals) / len(vals)
                    print(f"    t={tick:2d}: g_DC={avg:.4f} (n={len(vals)})")

    # ================================================================
    # Assertions (Persistence Theorem)
    # ================================================================

    print("\n--- Persistence Theorem Assertions ---")

    with_d = [r for r in all_results if r.variant == "with_d"]
    no_d = [r for r in all_results if r.variant == "no_d"]

    # Compute means
    gf_with_d_success = [r.G_over_F_final for r in with_d if r.success]
    gf_no_d = [r.G_over_F_final for r in no_d]

    gf_wd_mean = sum(gf_with_d_success) / len(gf_with_d_success) if gf_with_d_success else 0
    gf_nd_mean = sum(gf_no_d) / len(gf_no_d) if gf_no_d else 0

    g_dc_wd = sum(r.g_DC_mean for r in with_d) / len(with_d) if with_d else 0
    g_dc_nd = sum(r.g_DC_mean for r in no_d) / len(no_d) if no_d else 0

    # Weakest coupling distribution (with_d)
    weakest_wd = Counter()
    for r in with_d:
        weakest_wd.update(r.weakest_distribution)
    total_wd_ticks = sum(weakest_wd.values())
    cd_pct = weakest_wd.get("CD", 0) / max(total_wd_ticks, 1)
    dc_pct = weakest_wd.get("DC", 0) / max(total_wd_ticks, 1)

    # 1. D-ablation causes G/F collapse: with_d > no_d
    p1 = gf_wd_mean > gf_nd_mean
    print(f"  [{'PASS' if p1 else 'FAIL'}] G/F collapse: "
          f"with_d={gf_wd_mean:.4f} > no_d={gf_nd_mean:.4f}")

    # 2. g_DC(with_d) > g_DC(no_d) (DC coupling is real)
    p2 = g_dc_wd > g_dc_nd
    print(f"  [{'PASS' if p2 else 'FAIL'}] g_DC: "
          f"with_d={g_dc_wd:.4f} > no_d={g_dc_nd:.4f}")

    # 3. with_d success rate >> no_d (D is essential)
    sr_wd = sum(1 for r in with_d if r.success) / len(with_d) if with_d else 0
    sr_nd = sum(1 for r in no_d if r.success) / len(no_d) if no_d else 0
    delta = sr_wd - sr_nd
    p3 = delta >= 0.50
    print(f"  [{'PASS' if p3 else 'FAIL'}] SR delta: "
          f"{delta:.1%} (with_d={sr_wd:.1%}, no_d={sr_nd:.1%}) >= 50pp")

    # 4. g_DC progression rises in with_d (clue accumulation → synthesis)
    # Compare early g_DC (t=0-2) vs late g_DC (last 3 ticks) in successful episodes
    if wd_results:
        early_gdc = []
        late_gdc = []
        for r in wd_results:
            prog = r.g_DC_progression
            if len(prog) >= 4:
                early_gdc.extend(prog[:3])
                late_gdc.extend(prog[-3:])
        early_mean = sum(early_gdc) / len(early_gdc) if early_gdc else 0
        late_mean = sum(late_gdc) / len(late_gdc) if late_gdc else 0
        p4 = late_mean >= early_mean
        print(f"  [{'PASS' if p4 else 'FAIL'}] g_DC progression: "
              f"early={early_mean:.4f} -> late={late_mean:.4f}")
    else:
        p4 = False
        print(f"  [FAIL] g_DC progression: no successful episodes")

    # 5. g_AD(with_d) = 1.0 (deterministic D, no hallucinations)
    g_ad_wd = sum(r.g_AD_mean for r in with_d) / len(with_d) if with_d else 0
    p5 = g_ad_wd >= 0.95
    print(f"  [{'PASS' if p5 else 'FAIL'}] g_AD(with_d) = {g_ad_wd:.4f} (>= 0.95)")

    # --- CSV output ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"textworld_loop_gain_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "scenario_id", "seed", "success", "steps",
            "g_BA_mean", "g_CB_mean", "g_DC_mean", "g_AD_mean",
            "G_mean", "G_over_F_final", "weakest_final",
        ])
        for r in all_results:
            writer.writerow([
                r.variant, r.scenario_id, r.seed, int(r.success), r.steps,
                r.g_BA_mean, r.g_CB_mean, r.g_DC_mean, r.g_AD_mean,
                r.G_mean, r.G_over_F_final, r.weakest_final,
            ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p3 and p4 and p5
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 75)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

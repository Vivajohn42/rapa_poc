"""TextWorld D-Ablation Test: Proves D is essential.

Compares three variants on the same TextWorld scenarios:
  with_d:  Full 4D stack (A+B+C+D) via MvpKernel
  no_d:    3D stack (A+B+C, agent_d=None)
  random:  Random exit selection

If D is essential, with_d >> no_d in success rate. This is the critical
validation that TextWorld makes D a genuine architectural component.

Usage:
    python eval/run_textworld_ablation.py
    python eval/run_textworld_ablation.py --n 30
    python eval/run_textworld_ablation.py --scenario 0
"""

import argparse
import csv
import random as stdlib_random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.textworld import TextWorld, SCENARIOS
from agents.text_agent_a import TextAgentA
from agents.text_agent_b import TextAgentB
from agents.text_agent_c import TextAgentC
from agents.text_agent_d import TextAgentD
from router.deconstruct_text import deconstruct_text_d_to_c
from kernel.kernel import MvpKernel

# Optional LLM imports (only needed for --llm flag)
try:
    from agents.text_agent_d_llm import TextAgentDLLM
    from llm.provider import OllamaProvider
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


@dataclass
class AblationResult:
    variant: str
    scenario_id: int
    seed: int
    success: bool
    steps: int
    total_reward: float
    clues_collected: int
    target_identified: bool
    d_activations: int
    g_DC_mean: float
    g_AD_mean: float
    G_over_F_final: float
    weakest_final: str


def _make_deconstruct_fn(room_index):
    """Create a deconstruct function with room_index bound."""
    def fn(zC, zD, goal_map=None):
        return deconstruct_text_d_to_c(zC, zD, goal_map=goal_map, room_index=room_index)
    return fn


def run_textworld_episode(
    variant: str,
    scenario_id: int,
    seed: int,
    max_steps: int = 40,
) -> AblationResult:
    """Run one TextWorld episode with the given variant."""
    env = TextWorld(seed=seed, scenario_id=scenario_id)
    obs = env.reset()

    # Build room mappings
    room_ids = env.room_ids
    room_index = {rid: i for i, rid in enumerate(room_ids)}
    index_to_room = {i: rid for rid, i in room_index.items()}
    room_graph = env.room_graph
    room_properties = env.room_properties
    n_rooms = len(room_ids)

    # Agents
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
    elif variant.startswith("llm_"):
        # LLM-D variant: e.g. "llm_mistral"
        model_name = variant.split("_", 1)[1] + ":latest"
        llm = OllamaProvider(model=model_name)
        D = TextAgentDLLM(llm, room_properties, room_ids, room_index)
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            goal_map=room_index,
            enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )
    elif variant == "no_d":
        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=None,
            goal_map=room_index,
            enable_governance=True,
            deconstruct_fn=deconstruct_fn,
        )
    else:  # random
        kernel = None

    if kernel:
        kernel.reset_episode(goal_mode="seek", episode_id=f"tw_{variant}_{seed}")

    done = False
    total_reward = 0.0
    rng = stdlib_random.Random(seed)

    t = -1
    for t in range(max_steps):
        if variant == "random":
            # Random action from available exits + occasional claim
            exits = obs.get("exits", [])
            # Random agent claims with 1/N probability (N = total rooms)
            if rng.random() < 1.0 / len(env.room_ids):
                action = "claim"
            elif exits:
                action = rng.choice(exits)
            else:
                action = "wait"
            obs, reward, done = env.step(action)
            total_reward += reward
        else:
            # Inject visited rooms into C's memory for exploration heuristic
            kernel.zC.memory["visited_rooms"] = obs.get("visited", set())
            result = kernel.tick(t, obs, done=False)
            action = result.action

            obs, reward, done = env.step(action)
            total_reward += reward
            kernel.observe_reward(reward)

        if done:
            if kernel:
                kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0

    # Collect metrics
    clues_collected = len(env._clues_shown)

    if kernel and kernel.agent_d is not None and hasattr(kernel.agent_d, '_target'):
        target_identified = kernel.agent_d._target is not None
    else:
        target_identified = False

    d_activations = 0
    g_dc_mean = 0.0
    g_ad_mean = 1.0
    gf_final = 0.0
    weakest_final = "N/A"

    if kernel:
        history = kernel.loop_gain.episode_history
        if history:
            g_dc_vals = [s.g_DC for s in history]
            g_ad_vals = [s.g_AD for s in history]
            g_dc_mean = sum(g_dc_vals) / len(g_dc_vals)
            g_ad_mean = sum(g_ad_vals) / len(g_ad_vals)
            gf_final = kernel.loop_gain.G_over_F
            weakest_final = kernel.loop_gain.weakest_coupling
        d_activations = sum(1 for s in history if True)  # approximate

    return AblationResult(
        variant=variant,
        scenario_id=scenario_id,
        seed=seed,
        success=bool(done),
        steps=steps,
        total_reward=round(total_reward, 4),
        clues_collected=clues_collected,
        target_identified=target_identified,
        d_activations=d_activations,
        g_DC_mean=round(g_dc_mean, 4),
        g_AD_mean=round(g_ad_mean, 4),
        G_over_F_final=round(gf_final, 4),
        weakest_final=weakest_final,
    )


def main():
    parser = argparse.ArgumentParser(description="TextWorld D-Ablation Test")
    parser.add_argument("--n", type=int, default=20, help="Episodes per variant per scenario")
    parser.add_argument("--scenario", type=int, default=None, help="Specific scenario (None=all)")
    parser.add_argument("--max-steps", type=int, default=40, help="Max steps per episode")
    parser.add_argument("--llm", action="store_true", help="Include LLM-D variant (requires ollama)")
    parser.add_argument("--llm-model", type=str, default="mistral",
                        help="LLM model name for LLM-D variant (default: mistral)")
    args = parser.parse_args()

    print("=" * 75)
    print("  TextWorld D-Ablation Test")
    print("  Does D make a difference? (It should.)")
    print("=" * 75)

    n_episodes = args.n
    max_steps = args.max_steps
    variants = ["with_d", "no_d", "random"]

    if args.llm:
        if not HAS_LLM:
            print("  ERROR: LLM imports not available")
            sys.exit(1)
        variants.insert(1, f"llm_{args.llm_model}")
        print(f"  LLM-D variant: llm_{args.llm_model}")
        # Fewer episodes for LLM (slower)
        if n_episodes > 10:
            print(f"  (capping LLM episodes to 10 for speed)")

    if args.scenario is not None:
        scenario_ids = [args.scenario]
    else:
        scenario_ids = list(range(len(SCENARIOS)))

    all_results: List[AblationResult] = []

    for sid in scenario_ids:
        scenario = SCENARIOS[sid]
        print(f"\n--- Scenario {sid}: {len(scenario.rooms)} rooms, "
              f"target={scenario.target_room}, clues={scenario.required_clues} ---")

        for variant in variants:
            results = []
            # Cap LLM episodes for speed
            ep_count = min(n_episodes, 10) if variant.startswith("llm_") else n_episodes
            for i in range(ep_count):
                r = run_textworld_episode(
                    variant=variant,
                    scenario_id=sid,
                    seed=42 + i,
                    max_steps=max_steps,
                )
                results.append(r)
                all_results.append(r)

            sr = sum(1 for r in results if r.success) / len(results)
            steps_mean = sum(r.steps for r in results) / len(results)
            clues_mean = sum(r.clues_collected for r in results) / len(results)
            target_rate = sum(1 for r in results if r.target_identified) / len(results)
            print(f"    {variant:<10s}: SR={sr:.1%}  steps={steps_mean:.1f}  "
                  f"clues={clues_mean:.1f}  target_id={target_rate:.1%}")

    # ================================================================
    # Aggregate Analysis
    # ================================================================

    print("\n" + "=" * 75)
    print("  AGGREGATE RESULTS")
    print("=" * 75)

    for variant in variants:
        vr = [r for r in all_results if r.variant == variant]
        sr = sum(1 for r in vr if r.success) / len(vr) if vr else 0
        steps = sum(r.steps for r in vr) / len(vr) if vr else 0
        clues = sum(r.clues_collected for r in vr) / len(vr) if vr else 0
        target_id = sum(1 for r in vr if r.target_identified) / len(vr) if vr else 0
        g_dc = sum(r.g_DC_mean for r in vr) / len(vr) if vr else 0
        g_ad = sum(r.g_AD_mean for r in vr) / len(vr) if vr else 0
        gf = sum(r.G_over_F_final for r in vr) / len(vr) if vr else 0

        print(f"\n  {variant}:")
        print(f"    Success Rate:     {sr:.1%}")
        print(f"    Avg Steps:        {steps:.1f}")
        print(f"    Avg Clues:        {clues:.1f}")
        print(f"    Target ID Rate:   {target_id:.1%}")
        print(f"    g_DC mean:        {g_dc:.4f}")
        print(f"    g_AD mean:        {g_ad:.4f}")
        print(f"    G/F final:        {gf:.4f}")

    # --- Weakest coupling distribution ---
    wd = [r for r in all_results if r.variant == "with_d"]
    if wd:
        weakest_counts = Counter(r.weakest_final for r in wd)
        total = len(wd)
        print(f"\n  Weakest Coupling (with_d): ", end="")
        for c in ["BC", "CD", "AD", "AB"]:
            pct = weakest_counts.get(c, 0) / total * 100
            if pct > 0:
                print(f"{c}={pct:.0f}% ", end="")
        print()

    # ================================================================
    # Assertions
    # ================================================================

    print("\n--- Assertions ---")

    with_d_results = [r for r in all_results if r.variant == "with_d"]
    no_d_results = [r for r in all_results if r.variant == "no_d"]
    random_results = [r for r in all_results if r.variant == "random"]

    sr_with_d = sum(1 for r in with_d_results if r.success) / len(with_d_results) if with_d_results else 0
    sr_no_d = sum(1 for r in no_d_results if r.success) / len(no_d_results) if no_d_results else 0
    sr_random = sum(1 for r in random_results if r.success) / len(random_results) if random_results else 0

    g_dc_with_d = sum(r.g_DC_mean for r in with_d_results) / len(with_d_results) if with_d_results else 0
    g_dc_no_d = sum(r.g_DC_mean for r in no_d_results) / len(no_d_results) if no_d_results else 0

    # 1. with_d success rate >= 70%
    p1 = sr_with_d >= 0.70
    print(f"  [{'PASS' if p1 else 'FAIL'}] with_d SR = {sr_with_d:.1%} (>= 70%)")

    # 2. no_d success rate <= 40%
    p2 = sr_no_d <= 0.40
    print(f"  [{'PASS' if p2 else 'FAIL'}] no_d SR = {sr_no_d:.1%} (<= 40%)")

    # 3. with_d - no_d >= 30 percentage points
    delta = sr_with_d - sr_no_d
    p3 = delta >= 0.30
    print(f"  [{'PASS' if p3 else 'FAIL'}] SR delta = {delta:.1%} (>= 30pp)")

    # 4. g_DC(with_d) > g_DC(no_d)
    p4 = g_dc_with_d > g_dc_no_d
    print(f"  [{'PASS' if p4 else 'FAIL'}] g_DC: with_d={g_dc_with_d:.4f} > no_d={g_dc_no_d:.4f}")

    # --- CSV output ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"textworld_ablation_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "scenario_id", "seed", "success", "steps", "total_reward",
            "clues_collected", "target_identified", "d_activations",
            "g_DC_mean", "g_AD_mean", "G_over_F_final", "weakest_final",
        ])
        for r in all_results:
            writer.writerow([
                r.variant, r.scenario_id, r.seed, int(r.success), r.steps,
                r.total_reward, r.clues_collected, int(r.target_identified),
                r.d_activations, r.g_DC_mean, r.g_AD_mean,
                r.G_over_F_final, r.weakest_final,
            ])
    print(f"\n  CSV: {csv_path}")

    all_pass = p1 and p2 and p3 and p4
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 75)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

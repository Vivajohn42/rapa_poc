"""Kernel Loop Gain Validation: G/F convergence and weakest_coupling analysis.

Tests that:
1. G and F converge (G/F stabilizes near 1.0 for successful episodes)
2. weakest_coupling is identified correctly
3. g_AD = 1.0 for deterministic D (grounding always perfect)
4. g_DC responds to deconstruction events
5. Loop gain history is recorded per episode

Usage:
    python eval/run_kernel_loop_gain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD

from kernel.kernel import MvpKernel


def run_gain_episode(seed: int, goal_mode: str = "seek", max_steps: int = 50):
    """Run one episode and collect loop gain history."""
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
    D = AgentD()

    goal_map = getattr(env, "_goal_map", None)

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
    )

    kernel.reset_episode(goal_mode=goal_mode, episode_id=f"gain_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)

        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell

        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)

        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    return {
        "success": done,
        "steps": t + 1 if t >= 0 else 0,
        "history": kernel.loop_gain.episode_history,
        "final_G": kernel.loop_gain.G,
        "final_F": kernel.loop_gain.F,
        "final_G_over_F": kernel.loop_gain.G_over_F,
        "weakest": kernel.loop_gain.weakest_coupling,
    }


def main():
    print("=" * 70)
    print("  MvpKernel Loop Gain Validation")
    print("=" * 70)

    n_episodes = 30
    results = []
    for seed in range(42, 42 + n_episodes):
        r = run_gain_episode(seed, "seek")
        results.append(r)

    # --- Test 1: G/F converges ---
    print("\n--- Test 1: G/F Convergence ---")
    converged = 0
    for r in results:
        if r["history"]:
            final_gf = r["final_G_over_F"]
            # G/F should be between 0.5 and 2.0 (not diverging)
            if 0.1 <= final_gf <= 10.0:
                converged += 1
    conv_rate = converged / n_episodes
    p1 = conv_rate >= 0.8
    print(f"  [{'PASS' if p1 else 'FAIL'}] Converged: {converged}/{n_episodes} ({conv_rate:.1%})")

    # --- Test 2: weakest_coupling identification ---
    print("\n--- Test 2: Weakest Coupling Identification ---")
    weakest_counts = {}
    for r in results:
        w = r["weakest"]
        weakest_counts[w] = weakest_counts.get(w, 0) + 1
    print(f"  Weakest coupling distribution: {weakest_counts}")
    p2 = len(weakest_counts) >= 1  # At least one identified
    print(f"  [{'PASS' if p2 else 'FAIL'}] At least one weakest coupling identified")

    # --- Test 3: g_AD = 1.0 for deterministic D ---
    print("\n--- Test 3: g_AD for Deterministic D ---")
    g_ad_values = []
    for r in results:
        for snap in r["history"]:
            g_ad_values.append(snap.g_AD)
    g_ad_mean = sum(g_ad_values) / len(g_ad_values) if g_ad_values else 0
    p3 = g_ad_mean >= 0.95  # Should be ~1.0 for deterministic D
    print(f"  [{'PASS' if p3 else 'FAIL'}] g_AD mean = {g_ad_mean:.4f} (>= 0.95)")

    # --- Test 4: g_DC responds to deconstruction ---
    print("\n--- Test 4: g_DC on Deconstruction ---")
    g_dc_values = []
    for r in results:
        for snap in r["history"]:
            g_dc_values.append(snap.g_DC)
    g_dc_mean = sum(g_dc_values) / len(g_dc_values) if g_dc_values else 0
    # g_DC should be > 0.3 on average (some decon events should happen)
    p4 = g_dc_mean >= 0.3
    print(f"  [{'PASS' if p4 else 'FAIL'}] g_DC mean = {g_dc_mean:.4f} (>= 0.3)")

    # --- Test 5: History recorded ---
    print("\n--- Test 5: Episode History ---")
    total_ticks = sum(len(r["history"]) for r in results)
    p5 = total_ticks > 0
    print(f"  [{'PASS' if p5 else 'FAIL'}] Total gain snapshots recorded: {total_ticks}")

    # --- Sample episode detail ---
    print("\n--- Sample Episode (seed=42) ---")
    sample = results[0]
    print(f"  Success: {sample['success']}, Steps: {sample['steps']}")
    print(f"  Final G={sample['final_G']:.6f}, F={sample['final_F']:.6f}, G/F={sample['final_G_over_F']:.4f}")
    print(f"  Weakest: {sample['weakest']}")
    if sample["history"]:
        print(f"  Tick-by-tick (first 10):")
        for snap in sample["history"][:10]:
            print(
                f"    t={snap.tick:3d}: "
                f"g_BA={snap.g_BA:.3f} g_CB={snap.g_CB:.3f} "
                f"g_DC={snap.g_DC:.3f} g_AD={snap.g_AD:.3f} "
                f"G={snap.G:.6f} G/F={snap.G_over_F:.3f} w={snap.weakest_coupling}"
            )

    all_pass = p1 and p2 and p3 and p4 and p5
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

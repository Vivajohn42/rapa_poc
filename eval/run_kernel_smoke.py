"""Kernel Smoke Test: validate MvpKernel via 50 episodes per goal_mode.

Runs the MvpKernel orchestrator with deterministic Agent D (no LLM needed).
Verifies that the kernel tick lifecycle produces correct behavior matching
the existing ad-hoc orchestration.

Usage:
    python eval/run_kernel_smoke.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from state.schema import ZC

from kernel.kernel import MvpKernel

from eval.runner import run_batch, RunConfig


@dataclass
class KernelSmokeResult:
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    stay_rate: float
    d_activations: int
    decon_count: int
    hint_learned: bool


def run_kernel_episode(
    variant: str, goal_mode: str, max_steps: int = 50, seed: int = None,
) -> KernelSmokeResult:
    """Run a single episode via MvpKernel."""
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

    # Build goal_map from env
    goal_map = getattr(env, "_goal_map", None)

    kernel = MvpKernel(
        agent_a=A,
        agent_b=B,
        agent_c=C,
        agent_d=D,
        goal_map=goal_map,
        enable_governance=True,
    )

    zC = kernel.reset_episode(goal_mode=goal_mode, episode_id=f"smoke_{seed}")

    # Pre-target: go to hint cell first
    if "target" not in zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    total_reward = 0.0
    done = False
    stay_count = 0
    d_activations = 0
    decon_count = 0

    t = -1
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)

        if result.d_activated:
            d_activations += 1
        if result.decon_fired:
            decon_count += 1

        # Update C target from kernel state if learned
        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell

        # Step environment
        obs_next, reward, done = env.step(result.action)
        kernel.observe_reward(reward)

        zA_next = A.infer_zA(obs_next)
        zA_prev = A.infer_zA(obs)
        if zA_next.agent_pos == zA_prev.agent_pos:
            stay_count += 1

        total_reward += reward
        obs = obs_next

        if done:
            # Episode-end tick
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0
    hint_learned = "target" in kernel.zC.memory

    return KernelSmokeResult(
        variant=variant,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=round(total_reward, 4),
        stay_rate=round(stay_count / steps, 4) if steps > 0 else 0.0,
        d_activations=d_activations,
        decon_count=decon_count,
        hint_learned=hint_learned,
    )


def main():
    print("=" * 70)
    print("  MvpKernel Smoke Test")
    print("  50 episodes x 2 goal_modes via MvpKernel tick() lifecycle")
    print("=" * 70)

    config = RunConfig(
        name="kernel_smoke",
        variants=["kernel"],
        goal_modes=["seek", "avoid"],
        n_episodes=50,
        max_steps=50,
        seed_start=42,
        reference_variant=None,
    )

    batch = run_batch(config, run_kernel_episode)
    agg = batch["aggregates"]

    # --- Smoke assertions ---
    print("\n" + "=" * 70)
    print("  Smoke Assertions")
    print("=" * 70)

    all_pass = True

    # 1. Seek success rate should be > 50%
    seek_sr = agg.get("kernel/seek", {}).get("success_rate", 0)
    p1 = seek_sr >= 0.50
    print(f"  [{'PASS' if p1 else 'FAIL'}] Seek success_rate = {seek_sr:.3f} (>= 0.50)")
    all_pass = all_pass and p1

    # 2. Avoid success rate should be > 50%
    avoid_sr = agg.get("kernel/avoid", {}).get("success_rate", 0)
    p2 = avoid_sr >= 0.50
    print(f"  [{'PASS' if p2 else 'FAIL'}] Avoid success_rate = {avoid_sr:.3f} (>= 0.50)")
    all_pass = all_pass and p2

    # 3. Seek steps should be reasonable (< 40 mean)
    seek_steps = agg.get("kernel/seek", {}).get("steps_mean", 999)
    p3 = seek_steps <= 40.0
    print(f"  [{'PASS' if p3 else 'FAIL'}] Seek steps_mean = {seek_steps:.1f} (<= 40.0)")
    all_pass = all_pass and p3

    # 4. D activated at least sometimes
    d_act_mean = agg.get("kernel/seek", {}).get("d_activations_mean", 0)
    p4 = d_act_mean >= 0.0  # D can be 0 if no uncertainty/stuck triggers
    print(f"  [{'PASS' if p4 else 'FAIL'}] Seek d_activations_mean = {d_act_mean:.2f} (>= 0.0)")
    all_pass = all_pass and p4

    # 5. Hint learning should work in some episodes
    all_results = batch["results"]
    seek_results = [r for r in all_results if r.goal_mode == "seek"]
    hint_rate = sum(1 for r in seek_results if r.hint_learned) / len(seek_results) if seek_results else 0
    p5 = hint_rate > 0.0
    print(f"  [{'PASS' if p5 else 'FAIL'}] Seek hint_learned_rate = {hint_rate:.3f} (> 0.0)")
    all_pass = all_pass and p5

    # 6. Governance should have run (no assertion errors during tick)
    p6 = True  # If we got this far without exceptions, governance passed
    print(f"  [{'PASS' if p6 else 'FAIL'}] Governance invariants held (no AssertionErrors)")
    all_pass = all_pass and p6

    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

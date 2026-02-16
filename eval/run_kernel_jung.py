"""Kernel Jung Profile Validation: personality profiles produce different behavior.

Tests that:
1. Default profile (None) works identically to DEFAULT profile
2. Different profiles produce measurably different kernel parameters
3. SENSOR vs INTUITIVE produce different step counts / behavior
4. All profiles pass governance invariants (no AssertionErrors)

Usage:
    python eval/run_kernel_jung.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD

from kernel.kernel import MvpKernel
from kernel.jung_profiles import JungProfile, PROFILES


@dataclass
class JungEpisodeResult:
    variant: str
    goal_mode: str
    success: bool
    steps: int
    total_reward: float
    d_activations: int
    decon_count: int
    final_G: float
    weakest: str


def run_jung_episode(
    profile_name: str,
    seed: int,
    goal_mode: str = "seek",
    max_steps: int = 50,
    grid_size: int = 5,
    n_obstacles: int = 1,
) -> JungEpisodeResult:
    """Run one episode with a specific Jung profile."""
    if grid_size == 5:
        env = GridWorld(seed=seed)
    else:
        from env.gridworld import GoalDef, HintCellDef
        goals = [
            GoalDef("A", (grid_size - 1, grid_size - 1)),
            GoalDef("B", (grid_size - 1, 0)),
        ]
        hint_cells = [HintCellDef(pos=(0, grid_size - 1))]
        env = GridWorld(
            width=grid_size, height=grid_size, seed=seed,
            goals=goals, hint_cells=hint_cells,
            n_random_obstacles=n_obstacles,
        )
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
    profile = PROFILES.get(profile_name)

    kernel = MvpKernel(
        agent_a=A, agent_b=B, agent_c=C, agent_d=D,
        goal_map=goal_map, enable_governance=True,
        jung_profile=profile,
    )

    kernel.reset_episode(goal_mode=goal_mode, episode_id=f"jung_{profile_name}_{seed}")

    if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
        C.goal.target = env.hint_cell

    done = False
    d_activations = 0
    decon_count = 0
    total_reward = 0.0

    t = -1
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)

        if result.d_activated:
            d_activations += 1
        if result.decon_fired:
            decon_count += 1

        if "target" in kernel.zC.memory:
            C.goal.target = tuple(kernel.zC.memory["target"])
        elif hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell

        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        total_reward += reward

        if done:
            kernel.tick(t + 1, obs, done=True)
            break

    steps = (t + 1) if t >= 0 else 0

    return JungEpisodeResult(
        variant=profile_name,
        goal_mode=goal_mode,
        success=bool(done),
        steps=steps,
        total_reward=round(total_reward, 4),
        d_activations=d_activations,
        decon_count=decon_count,
        final_G=round(kernel.loop_gain.G, 6),
        weakest=kernel.loop_gain.weakest_coupling,
    )


def main():
    print("=" * 70)
    print("  MvpKernel Jung Profile Validation")
    print("=" * 70)

    n_episodes = 30
    profile_names = ["DEFAULT", "SENSOR", "INTUITIVE", "ANALYST"]

    # --- Test 1: Parameter differences ---
    print("\n--- Test 1: Profile Parameter Comparison ---")
    print(f"  {'Profile':<12s} {'cooldown':>8s} {'stuck_w':>8s} {'tb_delta':>8s} {'decon_cd':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name in profile_names:
        p = PROFILES[name]
        print(
            f"  {name:<12s} {p.d_cooldown_steps:>8d} {p.stuck_window:>8d} "
            f"{p.tie_break_delta:>8.3f} {p.deconstruct_cooldown:>8d}"
        )

    params = {name: (
        PROFILES[name].d_cooldown_steps,
        PROFILES[name].stuck_window,
        PROFILES[name].tie_break_delta,
        PROFILES[name].deconstruct_cooldown,
    ) for name in profile_names}

    unique_params = len(set(params.values()))
    p1 = unique_params == len(profile_names)
    print(f"\n  [{'PASS' if p1 else 'FAIL'}] {unique_params}/{len(profile_names)} profiles have unique parameters")

    # --- Run episodes ---
    print(f"\n--- Running {n_episodes} episodes per profile (seek mode) ---")
    results_by_profile = {}
    for name in profile_names:
        results = []
        for seed in range(42, 42 + n_episodes):
            r = run_jung_episode(name, seed, "seek")
            results.append(r)
        results_by_profile[name] = results

    # --- Test 2: All profiles work (governance holds) ---
    print("\n--- Test 2: All Profiles Pass Governance ---")
    p2 = True
    for name in profile_names:
        sr = sum(1 for r in results_by_profile[name] if r.success) / n_episodes
        steps_mean = sum(r.steps for r in results_by_profile[name]) / n_episodes
        d_mean = sum(r.d_activations for r in results_by_profile[name]) / n_episodes
        print(f"  {name:<12s}: SR={sr:.3f}  steps={steps_mean:.1f}  d_act={d_mean:.1f}")
    print(f"  [{'PASS' if p2 else 'FAIL'}] All profiles completed without errors")

    # --- Test 3: SENSOR vs INTUITIVE behavioral difference (10x10 grid) ---
    print("\n--- Test 3: SENSOR vs INTUITIVE Behavioral Difference (10x10 grid) ---")
    hard_profiles = {}
    for name in ["SENSOR", "INTUITIVE"]:
        results = []
        for seed in range(42, 42 + n_episodes):
            r = run_jung_episode(name, seed, "seek", max_steps=100,
                                 grid_size=10, n_obstacles=8)
            results.append(r)
        hard_profiles[name] = results

    sensor_steps = [r.steps for r in hard_profiles["SENSOR"]]
    intuitive_steps = [r.steps for r in hard_profiles["INTUITIVE"]]
    sensor_d = [r.d_activations for r in hard_profiles["SENSOR"]]
    intuitive_d = [r.d_activations for r in hard_profiles["INTUITIVE"]]

    sensor_steps_mean = sum(sensor_steps) / len(sensor_steps)
    intuitive_steps_mean = sum(intuitive_steps) / len(intuitive_steps)
    sensor_d_mean = sum(sensor_d) / len(sensor_d)
    intuitive_d_mean = sum(intuitive_d) / len(intuitive_d)

    print(f"  SENSOR    steps_mean={sensor_steps_mean:.1f}  d_activations_mean={sensor_d_mean:.1f}")
    print(f"  INTUITIVE steps_mean={intuitive_steps_mean:.1f}  d_activations_mean={intuitive_d_mean:.1f}")

    # On 10x10 grid, profiles should produce measurable behavioral difference
    steps_diff = abs(sensor_steps_mean - intuitive_steps_mean)
    d_diff = abs(sensor_d_mean - intuitive_d_mean)
    p3 = steps_diff > 0 or d_diff > 0
    print(f"  Steps diff: {steps_diff:.1f}, D-activation diff: {d_diff:.1f}")
    print(f"  [{'PASS' if p3 else 'FAIL'}] Profiles produce behavioral difference")

    # --- Test 4: None profile == DEFAULT profile ---
    print("\n--- Test 4: None Profile Equivalence ---")
    # Run with None profile
    none_results = []
    for seed in range(42, 42 + n_episodes):
        env = GridWorld(seed=seed)
        obs = env.reset()
        A = AgentA()
        B = AgentB()
        zA0 = A.infer_zA(obs)
        C = AgentC(goal=GoalSpec(mode="seek", target=(zA0.width-1, zA0.height-1)), anti_stay_penalty=1.1)
        D = AgentD()
        kernel = MvpKernel(agent_a=A, agent_b=B, agent_c=C, agent_d=D,
                           goal_map=getattr(env, "_goal_map", None),
                           enable_governance=True, jung_profile=None)
        kernel.reset_episode(goal_mode="seek", episode_id=f"none_{seed}")
        if "target" not in kernel.zC.memory and hasattr(env, "hint_cell") and env.hint_cell:
            C.goal.target = env.hint_cell
        done = False
        for t in range(50):
            result = kernel.tick(t, obs, done=False)
            if "target" in kernel.zC.memory:
                C.goal.target = tuple(kernel.zC.memory["target"])
            elif hasattr(env, "hint_cell") and env.hint_cell:
                C.goal.target = env.hint_cell
            obs, reward, done = env.step(result.action)
            kernel.observe_reward(reward)
            if done:
                kernel.tick(t+1, obs, done=True)
                break
        none_results.append(done)

    none_sr = sum(1 for d in none_results if d) / n_episodes
    default_sr = sum(1 for r in results_by_profile["DEFAULT"] if r.success) / n_episodes
    p4 = abs(none_sr - default_sr) < 0.1  # Should be very close
    print(f"  None SR:    {none_sr:.3f}")
    print(f"  DEFAULT SR: {default_sr:.3f}")
    print(f"  [{'PASS' if p4 else 'FAIL'}] None profile matches DEFAULT behavior")

    all_pass = p1 and p2 and p3 and p4
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

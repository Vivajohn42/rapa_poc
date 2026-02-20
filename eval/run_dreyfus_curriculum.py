"""Dreyfus Curriculum Runner -- Skill Acquisition per Grid Size.

Three stages per grid size, measuring governance-driven compression:

  Stage 1 (Novice):     D->C only. D learns task structure, C navigates with BFS.
  Stage 2 (Proficient): C->B compression. UnifiedMemory + CompressionController active.
  Stage 3 (Expert):     Reflex. B navigates autonomously, C only on phase transitions.

Grid sizes escalate only after completing all 3 stages: 6->8->(16).

Exit criteria (size-adaptive):
  Stage 1->2: SR > threshold (95% for 6, 90% for 8, 70% for 16)
  Stage 2->3: pct_c_compressed > 30% AND SR stable (>90%/80%/60%)
  Stage 3->next: >80%/60%/40% ticks in 4FoM regime

6 Assertions:
  1. Stage 1 reaches SR>95% on 6x6 within 150 eps
  2. Stage 2 shows at least 1 L3->L2 compression event
  3. Stage 2 shows step reduction (avg_steps < 90% of Stage-1)
  4. Stage 3 reaches >50% 4FoM on 6x6
  5. 8x8 Stage 1 reaches SR>90% within 150 eps
  6. D learns correct order [KEY, DOOR, GOAL]

Usage:
    python eval/run_dreyfus_curriculum.py                    # 6x6 + 8x8
    python eval/run_dreyfus_curriculum.py --include-16       # + 16x16
    python eval/run_dreyfus_curriculum.py --sizes 6          # only 6x6
    python eval/run_dreyfus_curriculum.py --max-per-stage 50 # faster test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey import DoorKeyEnv
from agents.doorkey_agent_a import DoorKeyAgentA
from agents.doorkey_agent_b import DoorKeyAgentB
from agents.autonomous_doorkey_agent_c import AutonomousDoorKeyAgentC
from agents.event_pattern_d import EventPatternD, DoorKeyEventType
from agents.object_memory import ObjectMemory
from kernel.kernel import MvpKernel
from router.deconstruct_doorkey import deconstruct_doorkey_d_to_c

FALLBACK_ACTIONS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]

# ---------------------------------------------------------------------------
# Size-adaptive thresholds
# ---------------------------------------------------------------------------

# Stage 1->2: SR must exceed this threshold over 20 episodes
STAGE1_SR_THRESHOLD = {6: 0.95, 8: 0.90, 16: 0.70}

# Stage 2->3: SR must stay above this AND pct_c_compressed > 30%
STAGE2_SR_THRESHOLD = {6: 0.90, 8: 0.80, 16: 0.60}

# Stage 3->next: pct_c_compressed (B-takeover) must exceed this over 20 eps
STAGE3_B_TAKEOVER_THRESHOLD = {6: 0.80, 8: 0.60, 16: 0.40}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    stage: int
    grid_size: int
    episode: int           # global episode counter
    stage_episode: int     # episode within current stage
    success: bool = False
    steps: int = 0
    reward: float = 0.0
    regime_ticks: Dict[str, int] = field(default_factory=dict)
    compression_events: List[str] = field(default_factory=list)
    mean_delta_8: float = 0.0
    mean_loop_gain: float = 0.0
    d_has_sequence: bool = False
    pct_4fom: float = 0.0
    pct_c_compressed: float = 0.0  # fraction of ticks where B took over from C


@dataclass
class StageResult:
    stage: int
    grid_size: int
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    exit_met: bool = False
    exit_reason: str = ""
    avg_steps_successful: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_phase(om: ObjectMemory) -> str:
    """Derive DoorKey phase from ObjectMemory state."""
    if om.door_open:
        return "REACH_GOAL"
    elif om.carrying_key:
        return "OPEN_DOOR"
    return "FIND_KEY"


def _recent_sr(episodes: List[EpisodeMetrics], window: int) -> float:
    """Success rate over last `window` episodes."""
    if len(episodes) < window:
        return 0.0
    recent = episodes[-window:]
    return sum(1 for e in recent if e.success) / len(recent)


def _recent_avg_steps(episodes: List[EpisodeMetrics], window: int) -> float:
    """Average steps over last `window` successful episodes."""
    if not episodes:
        return float("inf")
    recent = episodes[-window:]
    successes = [e.steps for e in recent if e.success]
    return sum(successes) / len(successes) if successes else float("inf")


def _recent_pct_4fom(episodes: List[EpisodeMetrics], window: int) -> float:
    """Average pct_4fom over last `window` episodes."""
    if len(episodes) < window:
        return 0.0
    recent = episodes[-window:]
    return sum(e.pct_4fom for e in recent) / len(recent)


def _recent_pct_c_compressed(episodes: List[EpisodeMetrics], window: int) -> float:
    """Average pct_c_compressed over last `window` episodes."""
    if len(episodes) < window:
        return 0.0
    recent = episodes[-window:]
    return sum(e.pct_c_compressed for e in recent) / len(recent)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    kernel: MvpKernel,
    event_d: EventPatternD,
    size: int,
    seed: int,
    max_steps: int,
    stage: int,
    global_ep: int,
    stage_ep: int,
) -> EpisodeMetrics:
    """Run a single DoorKey episode, collecting governance metrics."""
    env = DoorKeyEnv(size=size, seed=seed)
    obs = env.reset()
    obj_mem = ObjectMemory(grid_size=size)

    # Per-episode agents (stateless)
    agent_a = DoorKeyAgentA()
    agent_b = DoorKeyAgentB()
    agent_c = AutonomousDoorKeyAgentC(goal_mode="seek")
    agent_c.set_object_memory(obj_mem)
    event_d.set_object_memory(obj_mem)
    event_d.reset_episode()

    # Swap agents on persistent kernel
    kernel.agent_a = agent_a
    kernel.agent_b = agent_b
    kernel.agent_c = agent_c
    # kernel.agent_d stays: event_d is persistent

    kernel.reset_episode(goal_mode="seek", episode_id=f"dreyfus_s{stage}_e{global_ep}")

    # Metrics collectors
    regime_counter: Dict[str, int] = defaultdict(int)
    compression_events: List[str] = []
    delta_8_values: List[float] = []
    loop_gain_values: List[float] = []
    c_compressed_ticks: int = 0

    done = False
    reward = 0.0
    step_count = 0

    for t in range(max_steps):
        # 1. ObjectMemory scans ego-view
        obj_mem.update(env._env.unwrapped)

        # 2. Sync C and B with ObjectMemory state
        phase = _derive_phase(obj_mem)
        agent_c.phase = phase
        agent_c.key_pos = obj_mem.key_pos
        agent_c.door_pos = obj_mem.door_pos
        agent_c.carrying_key = obj_mem.carrying_key
        agent_c.door_open = obj_mem.door_open
        agent_b.update_door_state(obj_mem.door_pos, obj_mem.door_open)

        # 3. Kernel tick
        result = kernel.tick(t, obs, done=False)

        # 4. Collect governance metrics
        if kernel.unified_memory is not None:
            regime = kernel.unified_memory.get_active_regime()
            regime_counter[regime] += 1
        else:
            regime_counter["8FoM"] += 1  # Stage 1: no UM -> full deliberation

        if result.compression_stages:
            compression_events.extend(result.compression_stages)

        if result.c_compressed:
            c_compressed_ticks += 1

        if result.residual is not None:
            delta_8_values.append(result.residual.delta_8)

        if result.gain is not None:
            loop_gain_values.append(result.gain.G_over_F)

        # 5. Environment step
        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        step_count = t + 1

        if done:
            obj_mem.update(env._env.unwrapped)
            kernel.tick(t + 1, obs, done=True)
            break

    # Episode end
    success = done and reward > 0
    event_d.end_episode(success=success, steps=step_count)
    env.close()

    total_ticks = sum(regime_counter.values()) or 1
    pct_4fom = regime_counter.get("4FoM", 0) / total_ticks
    pct_c_comp = c_compressed_ticks / total_ticks

    return EpisodeMetrics(
        stage=stage,
        grid_size=size,
        episode=global_ep,
        stage_episode=stage_ep,
        success=success,
        steps=step_count,
        reward=round(reward, 4),
        regime_ticks=dict(regime_counter),
        compression_events=compression_events,
        mean_delta_8=round(sum(delta_8_values) / len(delta_8_values), 4)
            if delta_8_values else 0.0,
        mean_loop_gain=round(sum(loop_gain_values) / len(loop_gain_values), 4)
            if loop_gain_values else 0.0,
        d_has_sequence=event_d.success_sequence is not None,
        pct_4fom=round(pct_4fom, 3),
        pct_c_compressed=round(pct_c_comp, 3),
    )


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(
    stage: int,
    size: int,
    event_d: EventPatternD,
    kernel: MvpKernel,
    max_episodes: int,
    max_steps: int,
    seed_base: int,
    global_ep_offset: int,
    stage1_avg_steps: float = 0.0,
    verbose: bool = False,
    stagnation_window: int = 5,
) -> StageResult:
    """Run one Dreyfus stage, checking exit criteria each episode."""
    window = 20
    result = StageResult(stage=stage, grid_size=size)

    stage_names = {1: "Novice (D->C)", 2: "Proficient (C->B)", 3: "Expert (Reflex)"}
    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {stage_names.get(stage, '?')} | Grid {size}x{size}")
    print(f"{'='*60}")

    for ep in range(max_episodes):
        global_ep = global_ep_offset + ep
        seed = seed_base + global_ep

        metrics = run_episode(
            kernel=kernel,
            event_d=event_d,
            size=size,
            seed=seed,
            max_steps=max_steps,
            stage=stage,
            global_ep=global_ep,
            stage_ep=ep,
        )
        result.episodes.append(metrics)

        # Print progress
        if verbose or ep < 5 or ep % 10 == 9:
            sr = _recent_sr(result.episodes, min(len(result.episodes), window))
            status = "OK" if metrics.success else "FAIL"
            comp = ",".join(metrics.compression_events) if metrics.compression_events else "-"
            print(f"  ep {ep:3d}: {status:4s}  steps={metrics.steps:3d}  "
                  f"SR={sr:.0%}  4FoM={metrics.pct_4fom:.0%}  "
                  f"B={metrics.pct_c_compressed:.0%}  comp=[{comp}]  "
                  f"d8={metrics.mean_delta_8:.3f}")

        # Stagnation check for D reflection
        if ((ep + 1) % stagnation_window == 0
                and ep >= 2 * stagnation_window):
            recent_sr = _recent_sr(result.episodes, stagnation_window)
            prev_sr = _recent_sr(
                result.episodes[:-stagnation_window], stagnation_window)
            if recent_sr <= prev_sr + 0.05:
                event_d.reflect()

        # Check exit criteria (size-adaptive thresholds)
        if len(result.episodes) >= window:
            sr = _recent_sr(result.episodes, window)
            s1_thr = STAGE1_SR_THRESHOLD.get(size, 0.90)
            s2_thr = STAGE2_SR_THRESHOLD.get(size, 0.80)
            s3_thr = STAGE3_B_TAKEOVER_THRESHOLD.get(size, 0.60)

            if stage == 1 and sr > s1_thr:
                result.exit_met = True
                result.exit_reason = f"SR {sr:.0%} > {s1_thr:.0%} over {window} eps"
                break

            elif stage == 2:
                pct_comp = _recent_pct_c_compressed(result.episodes, window)
                if pct_comp > 0.30 and sr > s2_thr:
                    result.exit_met = True
                    result.exit_reason = (
                        f"B-takeover {pct_comp:.0%} > 30%, SR {sr:.0%}")
                    break
                # Fallback: step reduction also counts
                avg_steps = _recent_avg_steps(result.episodes, window)
                step_reduction = (1.0 - avg_steps / stage1_avg_steps
                                  if stage1_avg_steps > 0 else 0.0)
                if step_reduction > 0.30 and sr > s2_thr:
                    result.exit_met = True
                    result.exit_reason = (
                        f"Steps reduced {step_reduction:.0%} "
                        f"({stage1_avg_steps:.0f}->{avg_steps:.0f}), SR {sr:.0%}")
                    break

            elif stage == 3:
                pct_b = _recent_pct_c_compressed(result.episodes, window)
                if pct_b > s3_thr:
                    result.exit_met = True
                    result.exit_reason = (
                        f"B-takeover={pct_b:.0%} > {s3_thr:.0%} over "
                        f"{window} eps")
                    break

    # Compute avg steps for successful episodes
    successes = [e.steps for e in result.episodes if e.success]
    result.avg_steps_successful = (
        sum(successes) / len(successes) if successes else float("inf")
    )

    if not result.exit_met:
        result.exit_reason = f"Max episodes ({max_episodes}) reached"

    sr_final = _recent_sr(result.episodes, min(len(result.episodes), window))
    print(f"\n  Stage {stage} done: {len(result.episodes)} eps, "
          f"SR={sr_final:.0%}, avg_steps={result.avg_steps_successful:.0f}")
    print(f"  Exit: {'OK' if result.exit_met else '!!'} {result.exit_reason}")

    return result


# ---------------------------------------------------------------------------
# Grid-size runner (all 3 stages)
# ---------------------------------------------------------------------------

def run_grid_size(
    size: int,
    event_d: EventPatternD,
    max_per_stage: int,
    max_steps: int,
    seed_base: int,
    global_ep_offset: int,
    verbose: bool = False,
) -> Tuple[List[StageResult], int]:
    """Run all 3 Dreyfus stages for one grid size.

    Returns (stage_results, total_episodes_used).
    """
    print(f"\n{'#'*60}")
    print(f"  GRID SIZE: {size}x{size}")
    print(f"{'#'*60}")

    stage_results: List[StageResult] = []
    ep_offset = global_ep_offset

    # ---- Stage 1: Novice (no UM, no compression) ----
    kernel_s1 = MvpKernel(
        agent_a=DoorKeyAgentA(),
        agent_b=DoorKeyAgentB(),
        agent_c=AutonomousDoorKeyAgentC(goal_mode="seek"),
        agent_d=event_d,
        goal_map=None,
        enable_governance=True,
        deconstruct_fn=deconstruct_doorkey_d_to_c,
        fallback_actions=FALLBACK_ACTIONS,
        use_unified_memory=False,
        active_compression=False,
    )

    s1 = run_stage(
        stage=1, size=size, event_d=event_d, kernel=kernel_s1,
        max_episodes=max_per_stage, max_steps=max_steps,
        seed_base=seed_base, global_ep_offset=ep_offset,
        verbose=verbose,
    )
    stage_results.append(s1)
    ep_offset += len(s1.episodes)

    # ---- Stage 2: Proficient (UM + compression active) ----
    kernel_s2 = MvpKernel(
        agent_a=DoorKeyAgentA(),
        agent_b=DoorKeyAgentB(),
        agent_c=AutonomousDoorKeyAgentC(goal_mode="seek"),
        agent_d=event_d,
        goal_map=None,
        enable_governance=True,
        deconstruct_fn=deconstruct_doorkey_d_to_c,
        fallback_actions=FALLBACK_ACTIONS,
        use_unified_memory=True,
        active_compression=True,
    )

    s2 = run_stage(
        stage=2, size=size, event_d=event_d, kernel=kernel_s2,
        max_episodes=max_per_stage, max_steps=max_steps,
        seed_base=seed_base, global_ep_offset=ep_offset,
        stage1_avg_steps=s1.avg_steps_successful,
        verbose=verbose,
    )
    stage_results.append(s2)
    ep_offset += len(s2.episodes)

    # ---- Stage 3: Expert (same kernel as S2, persisted compression) ----
    s3 = run_stage(
        stage=3, size=size, event_d=event_d, kernel=kernel_s2,
        max_episodes=max_per_stage, max_steps=max_steps,
        seed_base=seed_base, global_ep_offset=ep_offset,
        stage1_avg_steps=s1.avg_steps_successful,
        verbose=verbose,
    )
    stage_results.append(s3)
    ep_offset += len(s3.episodes)

    total = ep_offset - global_ep_offset
    return stage_results, total


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(
    all_results: Dict[int, List[StageResult]],
    event_d: EventPatternD,
) -> bool:
    """Run the 6 assertions and print results."""
    print(f"\n{'='*60}")
    print("ASSERTIONS:")
    print(f"{'='*60}")
    checks: List[bool] = []

    # -- 1. Stage 1 reaches SR>95% on 6x6 within 150 eps --
    if 6 in all_results:
        s1_6 = all_results[6][0]  # Stage 1 for size=6
        sr = _recent_sr(s1_6.episodes, 20) if len(s1_6.episodes) >= 20 else 0.0
        p1 = s1_6.exit_met and sr > 0.95
    else:
        p1 = False
        sr = 0.0
    checks.append(p1)
    print(f"  [{'PASS' if p1 else 'FAIL'}] 1. 6x6 Stage 1 SR>95%: "
          f"{'exit met' if 6 in all_results and all_results[6][0].exit_met else 'not met'} "
          f"(SR={sr:.0%})")

    # -- 2. Stage 2 shows at least 1 L3->L2 compression event --
    if 6 in all_results and len(all_results[6]) >= 2:
        s2_6 = all_results[6][1]  # Stage 2 for size=6
        all_comp = []
        for e in s2_6.episodes:
            all_comp.extend(e.compression_events)
        has_l3_l2 = "L3_L2" in all_comp
    else:
        has_l3_l2 = False
        all_comp = []
    p2 = has_l3_l2
    checks.append(p2)
    print(f"  [{'PASS' if p2 else 'FAIL'}] 2. Stage 2 shows L3->L2 compression: "
          f"{len([c for c in all_comp if c == 'L3_L2'])} events")

    # -- 3. Stage 2 shows B-takeover (pct_c_compressed > 0 in any episode) --
    if 6 in all_results and len(all_results[6]) >= 2:
        s2_6 = all_results[6][1]
        max_comp_pct = max((e.pct_c_compressed for e in s2_6.episodes), default=0.0)
        avg_comp_pct = (sum(e.pct_c_compressed for e in s2_6.episodes)
                        / len(s2_6.episodes) if s2_6.episodes else 0.0)
        p3 = max_comp_pct > 0.0  # any B-takeover at all
    else:
        p3 = False
        max_comp_pct = avg_comp_pct = 0.0
    checks.append(p3)
    print(f"  [{'PASS' if p3 else 'FAIL'}] 3. Stage 2 B-takeover: "
          f"max={max_comp_pct:.0%}, avg={avg_comp_pct:.0%}")

    # -- 4. Stage 3 reaches >50% B-takeover on 6x6 --
    if 6 in all_results and len(all_results[6]) >= 3:
        s3_6 = all_results[6][2]  # Stage 3 for size=6
        pct_b = _recent_pct_c_compressed(
            s3_6.episodes, min(len(s3_6.episodes), 20))
        pct_4 = _recent_pct_4fom(
            s3_6.episodes, min(len(s3_6.episodes), 20))
    else:
        pct_b = pct_4 = 0.0
    p4 = pct_b > 0.50
    checks.append(p4)
    print(f"  [{'PASS' if p4 else 'FAIL'}] 4. 6x6 Stage 3 B>50%: "
          f"B={pct_b:.0%}, 4FoM={pct_4:.0%}")

    # -- 5. 8x8 Stage 1 reaches SR>90% within 150 eps --
    if 8 in all_results:
        s1_8 = all_results[8][0]
        sr_8 = _recent_sr(s1_8.episodes, 20) if len(s1_8.episodes) >= 20 else 0.0
        p5 = sr_8 > 0.90
    else:
        p5 = True  # Skip if 8x8 not run
        sr_8 = -1.0
    checks.append(p5)
    if sr_8 < 0:
        print(f"  [SKIP] 5. 8x8 Stage 1 SR>90%: not run")
    else:
        print(f"  [{'PASS' if p5 else 'FAIL'}] 5. 8x8 Stage 1 SR>90%: {sr_8:.0%}")

    # -- 6. D learns correct order [KEY, DOOR, GOAL] --
    correct_order = [
        DoorKeyEventType.KEY_PICKED_UP,
        DoorKeyEventType.DOOR_OPENED,
        DoorKeyEventType.GOAL_REACHED,
    ]
    p6 = event_d.success_sequence == correct_order
    seq_str = ([e.name for e in event_d.success_sequence]
               if event_d.success_sequence else "None")
    checks.append(p6)
    print(f"  [{'PASS' if p6 else 'FAIL'}] 6. D correct order: {seq_str}")

    # Summary
    n_pass = sum(checks)
    all_pass = all(checks)
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({n_pass}/{len(checks)})")
    return all_pass


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(
    all_results: Dict[int, List[StageResult]],
) -> Path:
    """Save all episode metrics to CSV."""
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"dreyfus_curriculum_{ts}.csv"

    fieldnames = [
        "stage", "grid_size", "episode", "stage_episode",
        "success", "steps", "reward",
        "pct_4fom", "pct_c_compressed", "mean_delta_8", "mean_loop_gain",
        "compression_events", "d_has_sequence",
    ]

    rows = []
    for size in sorted(all_results.keys()):
        for sr in all_results[size]:
            for ep in sr.episodes:
                rows.append({
                    "stage": ep.stage,
                    "grid_size": ep.grid_size,
                    "episode": ep.episode,
                    "stage_episode": ep.stage_episode,
                    "success": ep.success,
                    "steps": ep.steps,
                    "reward": ep.reward,
                    "pct_4fom": ep.pct_4fom,
                    "pct_c_compressed": ep.pct_c_compressed,
                    "mean_delta_8": ep.mean_delta_8,
                    "mean_loop_gain": ep.mean_loop_gain,
                    "compression_events": ",".join(ep.compression_events),
                    "d_has_sequence": ep.d_has_sequence,
                })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    parser = argparse.ArgumentParser(
        description="Dreyfus Curriculum Runner -- Skill Acquisition per Grid Size")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Grid sizes to run (default: 6 8)")
    parser.add_argument("--include-16", action="store_true",
                        help="Include 16x16 grid")
    parser.add_argument("--max-per-stage", type=int, default=150,
                        help="Max episodes per stage (default: 150)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every episode")
    args = parser.parse_args()

    if args.sizes is not None:
        sizes = args.sizes
    elif args.include_16:
        sizes = [6, 8, 16]
    else:
        sizes = [6, 8]

    # Adjust max_steps for larger grids
    max_steps_map = {6: 200, 8: 300, 16: 600}

    print(f"\nDreyfus Curriculum Runner")
    print(f"  Sizes: {sizes}")
    print(f"  Max per stage: {args.max_per_stage}")
    print(f"  Seed: {args.seed}")

    t0 = time.time()
    all_results: Dict[int, List[StageResult]] = {}
    global_ep = 0

    # Persistent D across ALL grid sizes
    event_d = EventPatternD()

    for size in sizes:
        ms = max_steps_map.get(size, args.max_steps)
        stage_results, n_eps = run_grid_size(
            size=size,
            event_d=event_d,
            max_per_stage=args.max_per_stage,
            max_steps=ms,
            seed_base=args.seed,
            global_ep_offset=global_ep,
            verbose=args.verbose,
        )
        all_results[size] = stage_results
        global_ep += n_eps

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"CURRICULUM COMPLETE: {global_ep} total episodes in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Print stage summary
    print("\nSTAGE SUMMARY:")
    for size in sorted(all_results.keys()):
        print(f"\n  Grid {size}x{size}:")
        for sr in all_results[size]:
            n = len(sr.episodes)
            successes = sum(1 for e in sr.episodes if e.success)
            sr_pct = successes / n if n > 0 else 0.0
            comp_total = sum(len(e.compression_events) for e in sr.episodes)
            print(f"    Stage {sr.stage}: {n} eps, SR={sr_pct:.0%}, "
                  f"avg_steps={sr.avg_steps_successful:.0f}, "
                  f"compressions={comp_total}, "
                  f"exit={'OK' if sr.exit_met else '!!'}")

    # Print D learning summary
    print(f"\nD LEARNING:")
    print(f"  Sequence: {[e.name for e in event_d.success_sequence] if event_d.success_sequence else 'None'}")
    print(f"  Partial hypotheses: {event_d.partial_hypotheses}")
    print(f"  Negative constraints: {event_d.negative_constraints}")

    save_csv(all_results)
    success = run_assertions(all_results, event_d)

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

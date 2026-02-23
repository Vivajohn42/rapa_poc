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
from agents.online_distiller import OnlineDistiller, DistillerMode, TargetKind
from kernel.regime import RegimeController, DefRegime
from kernel.telemetry import Telemetry

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
    replan_burst_count: int = 0    # replan-burst activations this episode
    distiller_samples: int = 0     # total teacher samples flushed to replay
    distiller_replay: int = 0      # current replay buffer size (train+eval)
    distiller_accuracy: float = 0.0  # distiller eval accuracy
    distiller_acc_targeted: float = 0.0  # accuracy on targeted samples only
    distiller_acc_frontier: float = 0.0  # accuracy on frontier samples only
    distiller_mean_conf: float = 0.0  # mean prediction confidence
    distiller_conf_targeted: float = 0.0  # mean confidence on targeted samples
    distiller_conf_frontier: float = 0.0  # mean confidence on frontier samples
    distiller_mean_trust: float = 0.0    # mean trust (margin+entropy) on eval set
    distiller_trust_targeted: float = 0.0  # mean trust on targeted samples
    distiller_trust_frontier: float = 0.0  # mean trust on frontier samples
    distiller_enabled: bool = False  # whether distiller net was active
    distiller_mode: str = "OFF"      # overall mode: OFF / APPRENTICE / EXPERT
    distiller_mode_real: str = "OFF"    # per-kind mode for REAL
    distiller_mode_frontier: str = "OFF"  # per-kind mode for FRONTIER
    distiller_n_targeted: int = 0    # total targeted samples seen (at collection)
    distiller_n_targeted_eval: int = 0  # targeted samples in eval buffer
    distiller_n_frontier: int = 0    # total frontier samples seen
    distiller_n_frontier_eval: int = 0  # frontier samples in eval buffer
    distiller_train_count: int = 0   # number of train() calls so far
    no_target_ticks: int = 0         # ticks where no target was available
    regime: str = "ORIENT"            # active DEF regime (dominant this episode)
    # SAC Agent C metrics (Phase 5b)
    sac_c_mode: str = "OFF"           # learner mode: OFF/TRAINING/READY
    sac_c_accuracy: float = 0.0       # action agreement with deterministic C
    sac_c_episodes_trained: int = 0   # episodes trained so far
    # Meta-Controller D metrics (Phase 5c)
    meta_d_mode: str = "OFF"          # learner mode: OFF/TRAINING/READY
    meta_d_accuracy: float = 0.0      # combined phase+confidence agreement
    meta_d_episodes_trained: int = 0  # episodes trained so far


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
    sac_agent_c=None,
    meta_agent_d=None,
) -> EpisodeMetrics:
    """Run a single DoorKey episode, collecting governance metrics."""
    env = DoorKeyEnv(size=size, seed=seed)
    obs = env.reset()
    obj_mem = ObjectMemory(grid_size=size)

    # Per-episode agents (stateless — except B and SAC-C which are persistent)
    agent_a = DoorKeyAgentA()
    agent_b = kernel.agent_b  # Keep persistent B (DreamerAgentB accumulates replay)
    if sac_agent_c is not None:
        # Persistent SAC-C: reset per-episode state, sync ObjectMemory
        agent_c = sac_agent_c
        agent_c.set_object_memory(obj_mem)
    else:
        agent_c = AutonomousDoorKeyAgentC(goal_mode="seek")
        agent_c.set_object_memory(obj_mem)
    event_d.set_object_memory(obj_mem)
    event_d.reset_episode()

    # Swap agents on persistent kernel (B stays, SAC-C stays if provided)
    kernel.agent_a = agent_a
    kernel.agent_c = agent_c
    # kernel.agent_d stays: event_d is persistent

    kernel.reset_episode(goal_mode="seek", episode_id=f"dreyfus_s{stage}_e{global_ep}")

    # Metrics collectors
    regime_counter: Dict[str, int] = defaultdict(int)
    shadow_regime_counter: Dict[str, int] = defaultdict(int)
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
        if sac_agent_c is not None:
            # SACAgentC needs explicit attribute sync (forwarded to inner)
            agent_c.key_pos = obj_mem.key_pos
            agent_c.door_pos = obj_mem.door_pos
            agent_c.carrying_key = obj_mem.carrying_key
            agent_c.door_open = obj_mem.door_open
        else:
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

        # Track active regime per tick (skip CONSOLIDATE which is only episode-end)
        if result.regime is not None:
            shadow_regime_counter[result.regime] += 1

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

    # Online distiller episode lifecycle
    distiller = getattr(kernel, '_online_distiller', None)
    if distiller is not None:
        distiller.end_episode(success=success)

    # Update success rate EMA (for regime trigger signals)
    kernel.update_success_rate(success)

    # Shadow regime: dominant regime during episode (excl. CONSOLIDATE)
    _regime_ctrl = kernel.regime_controller
    _non_consol = {k: v for k, v in shadow_regime_counter.items()
                   if k != "CONSOLIDATE"}
    if _non_consol:
        _shadow_regime_name = max(_non_consol, key=_non_consol.get)
    elif _regime_ctrl is not None:
        _shadow_regime_name = _regime_ctrl.current.name
    else:
        _shadow_regime_name = "ORIENT"

    # Telemetry: emit episode_end event
    _tel = kernel.telemetry
    if _tel is not None:
        _tel.events.emit(
            "episode_end",
            success=success,
            steps=step_count,
            regime=_shadow_regime_name,
            stage=stage,
        )

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
        replan_burst_count=kernel.replan_burst_count,
        distiller_samples=distiller.total_samples if distiller is not None else 0,
        distiller_replay=distiller.replay_size if distiller is not None else 0,
        distiller_accuracy=round(distiller.accuracy, 3) if distiller is not None else 0.0,
        distiller_acc_targeted=round(distiller.accuracy_targeted, 3) if distiller is not None else 0.0,
        distiller_acc_frontier=round(distiller.accuracy_frontier, 3) if distiller is not None else 0.0,
        distiller_mean_conf=round(distiller.mean_confidence, 3) if distiller is not None else 0.0,
        distiller_conf_targeted=round(distiller.mean_conf_targeted, 3) if distiller is not None else 0.0,
        distiller_conf_frontier=round(distiller.mean_conf_frontier, 3) if distiller is not None else 0.0,
        distiller_mean_trust=round(distiller.mean_trust, 3) if distiller is not None else 0.0,
        distiller_trust_targeted=round(distiller.mean_trust_targeted, 3) if distiller is not None else 0.0,
        distiller_trust_frontier=round(distiller.mean_trust_frontier, 3) if distiller is not None else 0.0,
        distiller_enabled=distiller.is_enabled if distiller is not None else False,
        distiller_mode=distiller.mode.name if distiller is not None else "OFF",
        distiller_mode_real=(distiller.mode_for(TargetKind.REAL).name
                             if distiller is not None else "OFF"),
        distiller_mode_frontier=(distiller.mode_for(TargetKind.FRONTIER).name
                                 if distiller is not None else "OFF"),
        distiller_n_targeted=distiller.n_targeted_total if distiller is not None else 0,
        distiller_n_targeted_eval=distiller.n_targeted_eval if distiller is not None else 0,
        distiller_n_frontier=distiller.n_frontier_total if distiller is not None else 0,
        distiller_n_frontier_eval=(distiller.n_eval_for(TargetKind.FRONTIER)
                                   if distiller is not None else 0),
        distiller_train_count=distiller.train_count if distiller is not None else 0,
        no_target_ticks=kernel.no_target_ticks,
        regime=_shadow_regime_name,
        sac_c_mode=(sac_agent_c.learner.ready().mode.name
                     if sac_agent_c is not None else "OFF"),
        sac_c_accuracy=(sac_agent_c.learner.ready().accuracy
                         if sac_agent_c is not None else 0.0),
        sac_c_episodes_trained=(sac_agent_c.learner.ready().episodes_trained
                                 if sac_agent_c is not None else 0),
        meta_d_mode=(meta_agent_d.learner.ready().mode.name
                      if meta_agent_d is not None else "OFF"),
        meta_d_accuracy=(meta_agent_d.learner.ready().accuracy
                          if meta_agent_d is not None else 0.0),
        meta_d_episodes_trained=(meta_agent_d.learner.ready().episodes_trained
                                  if meta_agent_d is not None else 0),
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
    min_stage2_eps: int = 0,
    fast_fail: bool = False,
    sac_agent_c=None,
    meta_agent_d=None,
) -> StageResult:
    """Run one Dreyfus stage, checking exit criteria each episode.

    Args:
        min_stage2_eps: For Stage 2, minimum episodes before exit is allowed.
            This ensures the distiller has enough training time on large grids.
        fast_fail: If True, abort Stage 2 after 10 episodes if distiller
            is not making progress (FRONTIER still OFF, too few samples, etc.).
    """
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
            sac_agent_c=sac_agent_c,
            meta_agent_d=meta_agent_d,
        )
        result.episodes.append(metrics)

        # Print progress
        if verbose or ep < 5 or ep % 10 == 9:
            sr = _recent_sr(result.episodes, min(len(result.episodes), window))
            status = "OK" if metrics.success else "FAIL"
            comp = ",".join(metrics.compression_events) if metrics.compression_events else "-"
            dist_info = ""
            mode_str = metrics.distiller_mode
            mr = metrics.distiller_mode_real[0]   # O/A/E
            mf = metrics.distiller_mode_frontier[0]  # O/A/E
            if mode_str == "EXPERT":
                dist_info = (f"  EXP(R={mr},F={mf}"
                             f",at={metrics.distiller_acc_targeted:.0%}"
                             f",af={metrics.distiller_acc_frontier:.0%}"
                             f",tt={metrics.distiller_trust_targeted:.0%}"
                             f",tf={metrics.distiller_trust_frontier:.0%}"
                             f",tc={metrics.distiller_train_count})")
            elif mode_str == "APPRENTICE":
                dist_info = (f"  APR(R={mr},F={mf}"
                             f",at={metrics.distiller_acc_targeted:.0%}"
                             f",af={metrics.distiller_acc_frontier:.0%}"
                             f",tc={metrics.distiller_train_count})")
            elif metrics.distiller_samples > 0 or (
                    metrics.distiller_n_targeted + metrics.distiller_n_frontier) > 0:
                seen = metrics.distiller_n_targeted + metrics.distiller_n_frontier
                dist_info = (f"  OFF(seen={seen}"
                             f",replay={metrics.distiller_replay}"
                             f",nt={metrics.distiller_n_targeted}"
                             f",nf={metrics.distiller_n_frontier}"
                             f",tc={metrics.distiller_train_count})")
            nt_info = (f"  nt0={metrics.no_target_ticks}"
                       if metrics.no_target_ticks > 0 else "")
            burst_info = f"  burst={metrics.replan_burst_count}" if metrics.replan_burst_count > 0 else ""
            reg_info = (f"  reg={metrics.regime}"
                        if metrics.regime != "ORIENT" else "")
            sac_info = ""
            if metrics.sac_c_mode != "OFF":
                sac_info = (f"  SAC={metrics.sac_c_mode[:3]}"
                            f"(acc={metrics.sac_c_accuracy:.0%}"
                            f",ep={metrics.sac_c_episodes_trained})")
            meta_info = ""
            if metrics.meta_d_mode != "OFF":
                meta_info = (f"  META={metrics.meta_d_mode[:3]}"
                             f"(acc={metrics.meta_d_accuracy:.0%}"
                             f",ep={metrics.meta_d_episodes_trained})")
            print(f"  ep {ep:3d}: {status:4s}  steps={metrics.steps:3d}  "
                  f"SR={sr:.0%}  B={metrics.pct_c_compressed:.0%}  "
                  f"comp=[{comp}]  d8={metrics.mean_delta_8:.3f}"
                  f"{burst_info}{dist_info}{sac_info}{meta_info}"
                  f"{reg_info}{nt_info}")

        # Stagnation check for D reflection
        if ((ep + 1) % stagnation_window == 0
                and ep >= 2 * stagnation_window):
            recent_sr = _recent_sr(result.episodes, stagnation_window)
            prev_sr = _recent_sr(
                result.episodes[:-stagnation_window], stagnation_window)
            if recent_sr <= prev_sr + 0.05:
                event_d.reflect()

        # Fast-fail: abort Stage 2 early if distiller not progressing
        FAST_FAIL_EP = 10
        FAST_FAIL_MIN_SAMPLES = 50
        if (stage == 2 and ep == FAST_FAIL_EP - 1 and fast_fail):
            distiller = getattr(kernel, '_online_distiller', None)
            if distiller is not None:
                failures = []
                # 1. Enough samples collected?
                if distiller.total_samples < FAST_FAIL_MIN_SAMPLES:
                    failures.append(
                        f"samples={distiller.total_samples} < "
                        f"{FAST_FAIL_MIN_SAMPLES}")
                # 2. Warm-start fired?
                if distiller.train_count == 0:
                    failures.append("train_count=0 (warm-start never fired)")
                # 3. FRONTIER (dominant channel) at least APPRENTICE?
                if (distiller.mode_for(TargetKind.FRONTIER)
                        == DistillerMode.OFF):
                    failures.append(
                        "mode_frontier=OFF (no training signal)")
                # 4. Frontier eval exists?
                if (distiller.train_count > 0
                        and distiller.n_eval_for(TargetKind.FRONTIER) == 0):
                    failures.append(
                        "frontier n_eval=0 (no frontier samples in eval)")
                if failures:
                    print(f"\n  >> FAST-FAIL at S2 ep{ep + 1}:")
                    for f in failures:
                        print(f"    X {f}")
                    result.exit_reason = (
                        f"FAST-FAIL at ep {ep + 1}: "
                        + "; ".join(failures))
                    break

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
                # Stage 2 exit: B-takeover + SR + readiness guard
                # For large grids, ensure distiller has enough training time
                if ep < min_stage2_eps:
                    continue  # don't check exit yet

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
    trust_threshold: float = 0.35,
    fast_fail: bool = False,
    log_dir: Optional[Path] = None,
    dreamer_b: bool = False,
    sac_c: bool = False,
    meta_d: bool = False,
) -> Tuple[List[StageResult], int]:
    """Run all 3 Dreyfus stages for one grid size.

    Returns (stage_results, total_episodes_used).
    """
    print(f"\n{'#'*60}")
    print(f"  GRID SIZE: {size}x{size}")
    if dreamer_b:
        print(f"  B-Agent: DreamerAgentB (Phase 5a neural forward-model)")
    if sac_c:
        print(f"  C-Agent: SACAgentC (Phase 5b neural action selection)")
    if meta_d:
        print(f"  D-Agent: MetaControllerAgentD (Phase 5c neural meta-controller)")
    print(f"{'#'*60}")

    # Helper: create B agent (deterministic or dreamer wrapper)
    def _make_agent_b():
        inner = DoorKeyAgentB()
        if dreamer_b:
            from agents.dreamer_agent_b import DreamerAgentB
            return DreamerAgentB(inner=inner, use_neural=False)
        return inner

    # Helper: create persistent SAC-C agent (accumulates replay across stages)
    sac_agent_c_ref = None
    if sac_c:
        from agents.sac_agent_c import SACAgentC
        inner_c = AutonomousDoorKeyAgentC(goal_mode="seek")
        sac_agent_c_ref = SACAgentC(
            inner=inner_c,
            use_neural=False,  # Background-only: OFF mode, no governance impact
        )

    # Helper: create persistent Meta-D agent (accumulates replay across stages)
    meta_agent_d_ref = None
    if meta_d:
        from agents.meta_controller_agent_d import MetaControllerAgentD
        meta_agent_d_ref = MetaControllerAgentD(
            inner=event_d,
            use_neural=False,  # Background-only: OFF mode, no governance impact
        )

    stage_results: List[StageResult] = []
    ep_offset = global_ep_offset

    # ---- Stage 1: Novice (no UM, no compression) ----
    kernel_s1 = MvpKernel(
        agent_a=DoorKeyAgentA(),
        agent_b=_make_agent_b(),
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
        sac_agent_c=sac_agent_c_ref,
        meta_agent_d=meta_agent_d_ref,
    )
    stage_results.append(s1)
    ep_offset += len(s1.episodes)

    # ---- Stage 2: Proficient (UM + compression + online distiller) ----
    distiller = OnlineDistiller(
        replay_max=2000,
        train_interval=3,
        min_samples=200,
        min_targeted_for_enable=50,
        warm_start_epochs=5,
        train_interval_post_enable=2,
        min_accuracy_targeted=0.55,
        min_accuracy_frontier=0.45,
        min_mean_confidence=0.45,
        confidence_threshold=0.55,
        trust_threshold=trust_threshold,
        ema_alpha=0.5,
    )

    # DEF regime controller + telemetry (Phase 2: active overlay steering)
    regime_ctrl = RegimeController(hold_ticks=3)
    telemetry = Telemetry(log_dir=log_dir) if log_dir is not None else None

    kernel_s2 = MvpKernel(
        agent_a=DoorKeyAgentA(),
        agent_b=_make_agent_b(),
        agent_c=AutonomousDoorKeyAgentC(goal_mode="seek"),
        agent_d=event_d,
        goal_map=None,
        enable_governance=True,
        deconstruct_fn=deconstruct_doorkey_d_to_c,
        fallback_actions=FALLBACK_ACTIONS,
        use_unified_memory=True,
        active_compression=True,
        online_distiller=distiller,
        regime_controller=regime_ctrl,
        telemetry=telemetry,
        max_steps=max_steps,
    )

    # Size-adaptive replan-burst parameters
    if size >= 16:
        kernel_s2.set_replan_burst_params(
            stuck_window=5, burst_length=8, no_progress_threshold=12)
    elif size >= 8:
        kernel_s2.set_replan_burst_params(
            stuck_window=5, burst_length=5, no_progress_threshold=20)
    else:
        kernel_s2.set_replan_burst_params(
            stuck_window=5, burst_length=3, no_progress_threshold=30)

    # Stage 2 minimum duration: ensure distiller gets enough training time
    # on large grids before allowing exit to Stage 3.
    min_s2_eps = {16: 60, 8: 0, 6: 0}.get(size, 0)

    s2 = run_stage(
        stage=2, size=size, event_d=event_d, kernel=kernel_s2,
        max_episodes=max_per_stage, max_steps=max_steps,
        seed_base=seed_base, global_ep_offset=ep_offset,
        stage1_avg_steps=s1.avg_steps_successful,
        verbose=verbose,
        min_stage2_eps=min_s2_eps,
        fast_fail=fast_fail,
        sac_agent_c=sac_agent_c_ref,
        meta_agent_d=meta_agent_d_ref,
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
        sac_agent_c=sac_agent_c_ref,
        meta_agent_d=meta_agent_d_ref,
    )
    stage_results.append(s3)
    ep_offset += len(s3.episodes)

    # Close telemetry file handle
    if telemetry is not None:
        telemetry.close()

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
        "replan_burst_count", "distiller_samples", "distiller_replay",
        "distiller_accuracy", "distiller_acc_targeted", "distiller_acc_frontier",
        "distiller_mean_conf", "distiller_conf_targeted", "distiller_conf_frontier",
        "distiller_mean_trust", "distiller_trust_targeted", "distiller_trust_frontier",
        "distiller_enabled", "distiller_mode", "distiller_mode_real",
        "distiller_mode_frontier",
        "distiller_n_targeted", "distiller_n_targeted_eval",
        "distiller_n_frontier", "distiller_n_frontier_eval",
        "distiller_train_count", "no_target_ticks",
        "regime",
        "sac_c_mode", "sac_c_accuracy", "sac_c_episodes_trained",
        "meta_d_mode", "meta_d_accuracy", "meta_d_episodes_trained",
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
                    "replan_burst_count": ep.replan_burst_count,
                    "distiller_samples": ep.distiller_samples,
                    "distiller_replay": ep.distiller_replay,
                    "distiller_accuracy": ep.distiller_accuracy,
                    "distiller_acc_targeted": ep.distiller_acc_targeted,
                    "distiller_acc_frontier": ep.distiller_acc_frontier,
                    "distiller_mean_conf": ep.distiller_mean_conf,
                    "distiller_conf_targeted": ep.distiller_conf_targeted,
                    "distiller_conf_frontier": ep.distiller_conf_frontier,
                    "distiller_mean_trust": ep.distiller_mean_trust,
                    "distiller_trust_targeted": ep.distiller_trust_targeted,
                    "distiller_trust_frontier": ep.distiller_trust_frontier,
                    "distiller_enabled": ep.distiller_enabled,
                    "distiller_mode": ep.distiller_mode,
                    "distiller_mode_real": ep.distiller_mode_real,
                    "distiller_mode_frontier": ep.distiller_mode_frontier,
                    "distiller_n_targeted": ep.distiller_n_targeted,
                    "distiller_n_targeted_eval": ep.distiller_n_targeted_eval,
                    "distiller_n_frontier": ep.distiller_n_frontier,
                    "distiller_n_frontier_eval": ep.distiller_n_frontier_eval,
                    "distiller_train_count": ep.distiller_train_count,
                    "no_target_ticks": ep.no_target_ticks,
                    "regime": ep.regime,
                    "sac_c_mode": ep.sac_c_mode,
                    "sac_c_accuracy": ep.sac_c_accuracy,
                    "sac_c_episodes_trained": ep.sac_c_episodes_trained,
                    "meta_d_mode": ep.meta_d_mode,
                    "meta_d_accuracy": ep.meta_d_accuracy,
                    "meta_d_episodes_trained": ep.meta_d_episodes_trained,
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

def run_seed_sanity(
    n_seeds: int,
    size: int = 16,
    max_per_stage: int = 80,
    trust_threshold: float = 0.35,
    base_seed: int = 42,
) -> None:
    """Run the curriculum over multiple seeds, report Stage3 SR mean±std.

    This is a regression-protection check, not a proof.  Prints a compact
    summary table of Stage 3 success rate per seed, plus overall stats.

    Usage:
        python eval/run_dreyfus_curriculum.py --seed-sanity 5
    """
    import math

    max_steps = {6: 200, 8: 300, 16: 600}.get(size, 600)
    seeds = [base_seed + i * 1000 for i in range(n_seeds)]

    print(f"\n{'#'*60}")
    print(f"  SEED-SANITY: {size}x{size} over {n_seeds} seeds")
    print(f"  Max per stage: {max_per_stage}")
    print(f"  Trust threshold: tau={trust_threshold:.2f}")
    print(f"  Base seed: {seeds[0]}")
    print(f"{'#'*60}")

    stage3_srs: List[float] = []
    stage3_net_on: List[bool] = []
    per_seed_info: List[dict] = []

    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {i+1}/{n_seeds}: base_seed={seed}")
        print(f"{'='*60}")

        event_d = EventPatternD()
        stage_results, n_eps = run_grid_size(
            size=size,
            event_d=event_d,
            max_per_stage=max_per_stage,
            max_steps=max_steps,
            seed_base=seed,
            global_ep_offset=0,
            verbose=False,
            trust_threshold=trust_threshold,
        )

        # Extract Stage 2 metrics
        s2 = stage_results[1] if len(stage_results) >= 2 else None
        stage2_eps = len(s2.episodes) if s2 else 0
        # When did the mode first reach EXPERT in Stage 2?
        expert_ep_s2 = -1
        s2_end_acc_t = 0.0
        s2_end_conf_t = 0.0
        s2_end_trust_t = 0.0
        s2_end_mode = "OFF"
        s2_end_mode_r = "OFF"
        s2_end_mode_f = "OFF"
        if s2 and s2.episodes:
            for e in s2.episodes:
                if e.distiller_mode == "EXPERT" and expert_ep_s2 < 0:
                    expert_ep_s2 = e.stage_episode
            s2_end_acc_t = s2.episodes[-1].distiller_acc_targeted
            s2_end_conf_t = s2.episodes[-1].distiller_conf_targeted
            s2_end_trust_t = s2.episodes[-1].distiller_trust_targeted
            s2_end_mode = s2.episodes[-1].distiller_mode
            s2_end_mode_r = s2.episodes[-1].distiller_mode_real
            s2_end_mode_f = s2.episodes[-1].distiller_mode_frontier

        # Extract Stage 3 metrics
        s3 = stage_results[2] if len(stage_results) >= 3 else None
        if s3 and s3.episodes:
            successes = sum(1 for e in s3.episodes if e.success)
            sr = successes / len(s3.episodes)
            s3_end_mode = s3.episodes[-1].distiller_mode
            s3_end_mode_r = s3.episodes[-1].distiller_mode_real
            s3_end_mode_f = s3.episodes[-1].distiller_mode_frontier
            last_acc_t = s3.episodes[-1].distiller_acc_targeted
            last_acc_f = s3.episodes[-1].distiller_acc_frontier
            last_conf_t = s3.episodes[-1].distiller_conf_targeted
            last_trust_t = s3.episodes[-1].distiller_trust_targeted
            # Timeouts: episodes where agent hit max_steps
            timeouts_s3 = sum(1 for e in s3.episodes if e.steps >= max_steps)
            # Mean B-takeover in Stage 3
            b_takeover_s3 = (sum(e.pct_c_compressed for e in s3.episodes)
                             / len(s3.episodes))
            # Trust crossing: first episode where tt_targeted >= tau
            first_tt_cross_s3 = -1
            for e in s3.episodes:
                if e.distiller_trust_targeted >= trust_threshold and first_tt_cross_s3 < 0:
                    first_tt_cross_s3 = e.stage_episode
                    break
            # B% before and after crossing
            b_pct_before_cross = 0.0
            b_pct_after_cross = 0.0
            if first_tt_cross_s3 >= 0:
                pre = [e for e in s3.episodes if e.stage_episode < first_tt_cross_s3]
                post = [e for e in s3.episodes if e.stage_episode >= first_tt_cross_s3]
                if pre:
                    b_pct_before_cross = sum(e.pct_c_compressed for e in pre) / len(pre)
                if post:
                    b_pct_after_cross = sum(e.pct_c_compressed for e in post) / len(post)
        else:
            sr = 0.0
            s3_end_mode = "OFF"
            s3_end_mode_r = "OFF"
            s3_end_mode_f = "OFF"
            last_acc_t = 0.0
            last_acc_f = 0.0
            last_conf_t = 0.0
            last_trust_t = 0.0
            timeouts_s3 = 0
            b_takeover_s3 = 0.0
            first_tt_cross_s3 = -1
            b_pct_before_cross = 0.0
            b_pct_after_cross = 0.0

        stage3_srs.append(sr)
        stage3_net_on.append(s3_end_mode == "EXPERT")
        per_seed_info.append({
            "seed": seed,
            "sr": sr,
            "s3_mode": s3_end_mode,
            "s3_mode_r": s3_end_mode_r,
            "s3_mode_f": s3_end_mode_f,
            "n_eps_s3": len(s3.episodes) if s3 else 0,
            "stage2_eps": stage2_eps,
            "expert_ep_s2": expert_ep_s2,
            "s2_mode": s2_end_mode,
            "s2_mode_r": s2_end_mode_r,
            "s2_mode_f": s2_end_mode_f,
            "timeouts_s3": timeouts_s3,
            "b_takeover_s3": b_takeover_s3,
            "first_tt_cross_s3": first_tt_cross_s3,
            "b_pct_before_cross": b_pct_before_cross,
            "b_pct_after_cross": b_pct_after_cross,
            "s2_end_acc_t": s2_end_acc_t,
            "s2_end_conf_t": s2_end_conf_t,
            "s2_end_trust_t": s2_end_trust_t,
            "s3_end_acc_t": last_acc_t,
            "s3_end_conf_t": last_conf_t,
            "s3_end_trust_t": last_trust_t,
            "acc_f": last_acc_f,
        })

    elapsed = time.time() - t0

    # Compute stats
    mean_sr = sum(stage3_srs) / len(stage3_srs)
    variance = sum((x - mean_sr) ** 2 for x in stage3_srs) / len(stage3_srs)
    std_sr = math.sqrt(variance)
    min_sr = min(stage3_srs)
    max_sr = max(stage3_srs)

    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"  SEED-SANITY RESULTS: {size}x{size}  (tau={trust_threshold:.2f})")
    print(f"{'='*60}")

    # Header row
    print(f"\n  {'Seed':>6s}  {'SR':>5s}  {'mode':>3s}  {'R':>1s}{'F':>1s}  "
          f"{'s2ep':>4s}  {'exp@':>5s}  {'tout':>4s}  {'B%s3':>5s}  "
          f"{'cross':>5s}  {'B%<':>4s}  {'B%>':>4s}  "
          f"{'at_s2':>5s}  {'tt_s2':>5s}  "
          f"{'at_s3':>5s}  {'tt_s3':>5s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*3}  {'-'*2}  "
          f"{'-'*4}  {'-'*5}  {'-'*4}  {'-'*5}  "
          f"{'-'*5}  {'-'*4}  {'-'*4}  "
          f"{'-'*5}  {'-'*5}  "
          f"{'-'*5}  {'-'*5}")
    for info in per_seed_info:
        mode_str = info["s3_mode"][:3]  # OFF/APP/EXP
        mr = info["s3_mode_r"][0]  # O/A/E
        mf = info["s3_mode_f"][0]  # O/A/E
        exp_ep_str = (f"{info['expert_ep_s2']:>5d}"
                      if info["expert_ep_s2"] >= 0 else "  n/a")
        cross_str = (f"{info['first_tt_cross_s3']:>5d}"
                     if info["first_tt_cross_s3"] >= 0 else "    -")
        blt_str = (f"{info['b_pct_before_cross']:>3.0%}"
                   if info["first_tt_cross_s3"] >= 0 else "   -")
        bgt_str = (f"{info['b_pct_after_cross']:>3.0%}"
                   if info["first_tt_cross_s3"] >= 0 else "   -")
        print(f"  {info['seed']:>6d}  {info['sr']:>4.0%}  {mode_str:>3s}  "
              f"{mr}{mf}  "
              f"{info['stage2_eps']:>4d}  {exp_ep_str}  "
              f"{info['timeouts_s3']:>4d}  {info['b_takeover_s3']:>4.0%}  "
              f"{cross_str}  {blt_str}  {bgt_str}  "
              f"{info['s2_end_acc_t']:>4.0%}  "
              f"{info['s2_end_trust_t']:>4.0%}  "
              f"{info['s3_end_acc_t']:>4.0%}  "
              f"{info['s3_end_trust_t']:>4.0%}")

    print(f"\n  Stage3 SR: {mean_sr:.0%} +/- {std_sr:.0%}  "
          f"(min={min_sr:.0%}, max={max_sr:.0%})")
    print(f"  Trust threshold: tau={trust_threshold:.2f}")
    print(f"  EXPERT reached: {sum(stage3_net_on)}/{n_seeds} seeds")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/n_seeds:.0f}s/seed)")

    # Mode-progression analysis
    print(f"\n  Mode-progression per seed:")
    for info in per_seed_info:
        exp_ep = info["expert_ep_s2"]
        marker = ""
        if info["stage2_eps"] >= 60:
            marker += " [s2>=60]"
        if info["s3_mode"] == "OFF":
            marker += " [OFF: no training signal]"
        elif info["s3_mode"] == "APPRENTICE":
            marker += " [APR: readiness not met]"
        elif exp_ep >= 0 and exp_ep <= 20:
            marker += " [early EXPERT]"
        elif exp_ep >= 0:
            marker += f" [EXPERT @{exp_ep}]"
        print(f"    Seed {info['seed']}: s2={info['stage2_eps']}eps, "
              f"mode=R={info['s2_mode_r'][0]}/F={info['s2_mode_f'][0]}, "
              f"SR={info['sr']:.0%}{marker}")
    print()


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
    parser.add_argument("--seed-sanity", type=int, default=0, metavar="N",
                        help="Run 16x16 over N seeds, report Stage3 SR mean±std")
    parser.add_argument("--trust-threshold", type=float, default=0.35,
                        help="Trust gate threshold for B-takeover (default: 0.35)")
    parser.add_argument("--fast-fail", action="store_true",
                        help="Abort Stage 2 after 10 eps if distiller not progressing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every episode")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for JSONL telemetry log (default: None)")
    parser.add_argument("--dreamer-b", action="store_true",
                        help="Use DreamerAgentB (Phase 5a neural forward-model)")
    parser.add_argument("--sac-c", action="store_true",
                        help="Use SACAgentC (Phase 5b neural action selection)")
    parser.add_argument("--meta-d", action="store_true",
                        help="Use MetaControllerAgentD (Phase 5c neural meta-controller)")
    args = parser.parse_args()

    # Seed-sanity mode: quick multi-seed regression check
    if args.seed_sanity > 0:
        run_seed_sanity(
            n_seeds=args.seed_sanity,
            size=16,
            max_per_stage=args.max_per_stage,
            trust_threshold=args.trust_threshold,
            base_seed=args.seed,
        )
        return True

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
    print(f"  Trust threshold: tau={args.trust_threshold:.2f}")
    if args.sac_c:
        print(f"  SAC-C: enabled (Phase 5b, background learning)")
    if args.meta_d:
        print(f"  Meta-D: enabled (Phase 5c, background learning)")

    t0 = time.time()
    all_results: Dict[int, List[StageResult]] = {}
    global_ep = 0

    # Persistent D across ALL grid sizes
    event_d = EventPatternD()

    for size in sizes:
        ms = max_steps_map.get(size, args.max_steps)
        _log_dir = (Path(args.log_dir) / f"grid{size}"
                    if args.log_dir is not None else None)
        stage_results, n_eps = run_grid_size(
            size=size,
            event_d=event_d,
            max_per_stage=args.max_per_stage,
            max_steps=ms,
            seed_base=args.seed,
            global_ep_offset=global_ep,
            verbose=args.verbose,
            trust_threshold=args.trust_threshold,
            fast_fail=args.fast_fail,
            log_dir=_log_dir,
            dreamer_b=args.dreamer_b,
            sac_c=args.sac_c,
            meta_d=args.meta_d,
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

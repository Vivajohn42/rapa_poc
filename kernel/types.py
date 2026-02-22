"""Kernel governance types for rapa_mvp.

Ports the rapa_os message protocol (TickSignals, KernelDecision) to
Pydantic models for in-process use.  These types define the interface
between the MvpKernel orchestrator and the existing agents/router.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple


# Valid deconstruction trigger reasons (matching rapa_os)
VALID_REASONS = frozenset({
    "EPISODE_END", "MEM_BUDGET", "DRIFT",
    "MARGINAL_GAIN_LOW", "WAYPOINT_FOUND",
    "RESIDUUM_DIVERGENCE",
})

# All possible pairwise couplings
ALL_COUPLINGS = ["AB", "BC", "CD", "AC", "AD", "BD"]


@dataclass
class MvpTickSignals:
    """Stimulus signals computed from rapa_mvp state each tick.

    Maps to rapa_os TickSignals but computed from rapa_mvp observables
    (decision_delta, stuck detection, hint presence, etc.).
    """
    has_instr: bool = False
    plan_horizon: int = 1
    r_var: float = 0.0
    td_err: float = 0.0
    mem_cost: float = 0.0
    d_drift_score: float = 0.0
    marginal_gain_d: float = 0.5
    episode_end: bool = False
    need_explain: bool = False
    goal_target: Optional[Tuple[int, int]] = None
    goal_mode: str = "seek"
    need_narrative: bool = False

    @classmethod
    def from_state(
        cls,
        *,
        hint: Optional[str] = None,
        decision_delta: Optional[float] = None,
        reward_history: Optional[List[float]] = None,
        plan_horizon: int = 1,
        memory_size: int = 0,
        d_drift_score: float = 0.0,
        hint_just_learned: bool = False,
        done: bool = False,
        goal_mode: str = "seek",
        goal_target: Optional[Tuple[int, int]] = None,
        stuck: bool = False,
    ) -> MvpTickSignals:
        """Construct signals from current rapa_mvp state."""
        # has_instr: hint detected or goal target known
        has_instr = hint is not None or goal_target is not None

        # td_err: inverted decision_delta (low delta = high uncertainty)
        td_err = 0.0
        if decision_delta is not None:
            td_err = max(0.0, min(1.0, 1.0 - decision_delta))

        # r_var: rolling variance of recent rewards
        r_var = 0.0
        if reward_history and len(reward_history) >= 2:
            mean_r = sum(reward_history) / len(reward_history)
            r_var = min(1.0, (sum((r - mean_r) ** 2 for r in reward_history)
                              / len(reward_history)) ** 0.5)

        # mem_cost: normalized memory size
        mem_cost = min(1.0, memory_size / 10.0)

        # marginal_gain_d: high if hint just learned, decaying otherwise
        marginal_gain_d = 1.0 if hint_just_learned else 0.3

        # need_narrative: stuck and uncertain
        need_narrative = stuck and td_err >= 0.5

        return cls(
            has_instr=has_instr,
            plan_horizon=plan_horizon,
            r_var=r_var,
            td_err=td_err,
            mem_cost=mem_cost,
            d_drift_score=d_drift_score,
            marginal_gain_d=marginal_gain_d,
            episode_end=done,
            goal_mode=goal_mode,
            goal_target=goal_target,
            need_narrative=need_narrative,
        )


@dataclass
class MvpKernelDecision:
    """Gating & scheduling decision for one tick.

    Mirrors rapa_os KernelDecision: determines which streams are active,
    what couplings are scheduled, and whether deconstruction fires.
    """
    gC: int = 1           # 0=disabled, 1=enabled for Stream C
    gD: int = 0           # 0=disabled, 1=enabled for Stream D
    schedule: List[str] = field(default_factory=lambda: ["AB"])
    deconstruct: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class MvpLoopGain:
    """Per-tick loop stability metrics.

    Mirrors rapa_os LoopGain: G = g_BA * g_CB * g_DC * g_AD.
    """
    g_BA: float = 0.5
    g_CB: float = 0.5
    g_DC: float = 0.5
    g_AD: float = 1.0
    G: float = 0.0
    F: float = 0.0
    G_over_F: float = 1.0
    weakest_coupling: str = "AB"
    tick: int = 0


@dataclass
class ResidualSnapshot:
    """Per-tick closure residuum metrics.

    Δ₈ = √(Δ₄² + λ₁·c_term² + λ₂·d_term²)
    At fixpoint all three terms are zero.
    """
    delta_4: float = 0.0       # Prediction error (B model vs actual world)
    c_term: float = 0.0        # Valence-reference alignment
    d_term: float = 0.0        # Meaning-structure alignment
    delta_6: float = 0.0       # √(Δ₄² + λ₁·c²) — 6 FoM residuum
    delta_8: float = 0.0       # √(Δ₄² + λ₁·c² + λ₂·d²) — 8 FoM residuum
    d_delta_8_dt: float = 0.0  # Rate of change of Δ₈
    lambda_1: float = 1.0      # C-term weight
    lambda_2: float = 1.0      # D-term weight
    tick: int = 0
    # Dynamic threshold diagnostics
    sigma_delta_8: float = 0.0   # Rolling sigma of Δ₈
    F_delta_8: float = 0.0       # EMA baseline of Δ₈
    divergence_thr: float = 0.0  # Current divergence threshold (own sigma * I/E)
    activation_thr: float = 0.0  # Current D-activation threshold (F * S/N ratio)


@dataclass
class MeaningReport:
    """Structured D-stream output for kernel consumption.

    D reports WHAT it observes (confidence, suggested target, events).
    The kernel decides WHAT TO DO (regime, gating, target selection).

    This decouples D's implementation from kernel steering logic.
    New D implementations only need to produce a MeaningReport.
    """
    confidence: float = 0.0                                # D's self-assessment [0,1]
    suggested_target: Optional[Tuple[int, int]] = None     # D's navigation suggestion
    suggested_phase: Optional[str] = None                  # "find_key" / "open_door" / "reach_goal"
    events_detected: List[str] = field(default_factory=list)  # ["KEY_PICKED_UP", ...]
    hypothesis_strength: float = 0.0                       # Strength of current hypothesis [0,1]
    narrative_tags: List[str] = field(default_factory=list) # Raw tags for backward compat
    # Phase 3: Kernel-facing fields (no tag parsing needed)
    grounding_violations: int = 0                          # D's self-reported violations
    grounding_score: float = 1.0                           # 1.0=grounded, 0.0=hallucinating
    narrative_length: int = 0                              # For L3 summary storage


@dataclass
class MvpTickResult:
    """Result of a single MvpKernel.tick() call."""
    action: str
    scored: Optional[List] = None       # C's scored action list
    decision: Optional[MvpKernelDecision] = None
    gain: Optional[MvpLoopGain] = None
    residual: Optional[ResidualSnapshot] = None  # Closure residuum Δ₈
    d_activated: bool = False
    decon_fired: bool = False
    compression_stages: List[str] = field(default_factory=list)  # e.g. ["L3_L2", "L2_L1"]
    c_compressed: bool = False     # C ran in compressed mode (L2 compressed)
    d_suppressed: bool = False     # D skipped due to L3 compression
    replan_burst_active: bool = False  # This tick was a replan-burst tick (C forced)
    regime: Optional[str] = None  # Active DEF regime name (Phase 2: overlay steering)


# ======================================================================
# Phase 4: StreamLearner types
# ======================================================================

class LearnerMode(Enum):
    """Learning readiness of a stream.

    OFF:      Deterministic / no learning (default adapters)
    TRAINING: Learning active, not yet reliable
    READY:    Learning converged, stream is trustworthy
    """
    OFF = auto()
    TRAINING = auto()
    READY = auto()


@dataclass
class LearnerStatus:
    """Stream's self-reported learning status.

    Streams report WHAT they learned. The kernel decides WHAT TO DO
    (regime gating, compression eligibility).
    """
    mode: LearnerMode = LearnerMode.OFF
    accuracy: float = 0.0       # Self-reported accuracy [0,1]
    episodes_trained: int = 0   # Total training episodes
    label: str = ""             # Human-readable status (e.g. "BFS-deterministic")


@dataclass
class LearnerSignal:
    """Per-tick learning signal constructed by the kernel.

    Universal fields (every stream can use):
      tick, reward, done, episode_step

    Stream-relevant fields (each learner reads what it needs):
      delta_4         — B's prediction error (relevant for A, B)
      c_term          — C's valence-reference alignment (relevant for C)
      d_term          — D's meaning-structure alignment (relevant for D)
      action          — selected action this tick (relevant for B, C)
      scored          — C's scored action list (relevant for C)
      grounding_score — D's grounding (relevant for D)
    """
    tick: int = 0
    reward: float = 0.0
    done: bool = False
    episode_step: int = 0
    delta_4: float = 0.0
    c_term: float = 0.0
    d_term: float = 0.0
    action: str = ""
    scored: Optional[List] = None
    grounding_score: float = 1.0

"""Kernel governance types for rapa_mvp.

Ports the rapa_os message protocol (TickSignals, KernelDecision) to
Pydantic models for in-process use.  These types define the interface
between the MvpKernel orchestrator and the existing agents/router.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


# Valid deconstruction trigger reasons (matching rapa_os)
VALID_REASONS = frozenset({
    "EPISODE_END", "MEM_BUDGET", "DRIFT",
    "MARGINAL_GAIN_LOW", "WAYPOINT_FOUND",
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
class MvpTickResult:
    """Result of a single MvpKernel.tick() call."""
    action: str
    scored: Optional[List] = None       # C's scored action list
    decision: Optional[MvpKernelDecision] = None
    gain: Optional[MvpLoopGain] = None
    d_activated: bool = False
    decon_fired: bool = False

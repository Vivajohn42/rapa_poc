"""DEF-canonical regime controller for rapa_mvp.

Implements 7 canonical regimes driven by 4 abstract trigger signals.
Phase 2: RegimeController actively steers the kernel via overlay
semantics — RegimeGates constrain (never expand) the _route() output.
Gates control gC, gD, b_may_takeover, allow_decon, training_interval.

Regime hierarchy (macro → micro):
  Macro: EXPLORE, GOALSEEK, RECOVER
  Micro: ORIENT, HYPOTHESIZE, EVALUATE, CONSOLIDATE

Trigger signals are computed from existing kernel observables (no new
sensors). The compute_triggers() function takes only primitive values
for testability without a kernel instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# DefRegime Enum
# ---------------------------------------------------------------------------

class DefRegime(Enum):
    """7 canonical DEF regimes.

    Priority order (highest first):
      CONSOLIDATE > RECOVER > EVALUATE > GOALSEEK > HYPOTHESIZE > EXPLORE > ORIENT
    """
    ORIENT      = auto()  # New episode, no model yet
    EXPLORE     = auto()  # No target known, frontier exploration
    HYPOTHESIZE = auto()  # D has hypothesis, not yet confirmed
    GOALSEEK    = auto()  # Target known + reachable, B may navigate
    EVALUATE    = auto()  # Expectation violated, system re-checks
    RECOVER     = auto()  # Stuck / loop / budget crisis
    CONSOLIDATE = auto()  # Episode end, deconstruction


# ---------------------------------------------------------------------------
# TriggerSignals
# ---------------------------------------------------------------------------

@dataclass
class TriggerSignals:
    """4 abstract trigger signals derived from kernel observables.

    All values are in [0, 1]. Higher = more of the named quality.
    """
    surprise: float = 0.0    # td_err, delta_8, invalidation pulse
    progress: float = 0.0    # new_cells, target_known, success_rate
    readiness: float = 0.0   # distiller mode, l2_compressed, accuracy
    budget: float = 0.0      # mem_cost, stuck ratio, time ratio


# ---------------------------------------------------------------------------
# RegimeGates — per-regime kernel configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeGates:
    """Kernel behavior configuration per regime.

    Instead of if-else chains, each regime maps to a frozen gate config.
    """
    gC: int = 1
    gD: int = 0
    b_may_takeover: bool = False
    collect_distiller: bool = True
    allow_decon: bool = False
    training_interval_key: str = "default"  # "default" | "fast"
    flush_frontier: bool = True
    flush_real: bool = True


REGIME_GATE_MAP: Dict[DefRegime, RegimeGates] = {
    DefRegime.ORIENT: RegimeGates(
        gC=1, gD=1, b_may_takeover=False,
    ),
    DefRegime.EXPLORE: RegimeGates(
        gC=1, gD=0, b_may_takeover=True,
        flush_frontier=True, flush_real=False,
    ),
    DefRegime.HYPOTHESIZE: RegimeGates(
        gC=1, gD=1, b_may_takeover=False,
    ),
    DefRegime.GOALSEEK: RegimeGates(
        gC=1, gD=0, b_may_takeover=True,
        training_interval_key="fast",
    ),
    DefRegime.EVALUATE: RegimeGates(
        gC=1, gD=1, b_may_takeover=False,
    ),
    DefRegime.RECOVER: RegimeGates(
        gC=1, gD=1, b_may_takeover=False,
        allow_decon=True,
    ),
    DefRegime.CONSOLIDATE: RegimeGates(
        gC=0, gD=1, b_may_takeover=False,
        allow_decon=True,
    ),
}


# ---------------------------------------------------------------------------
# RegimeController
# ---------------------------------------------------------------------------

class RegimeController:
    """Priority-based regime selection with hysteresis.

    Evaluates TriggerSignals each tick and returns the active regime.
    Hysteresis: regime must be held for at least `hold_ticks` before
    switching, unless the new regime is urgent (CONSOLIDATE, RECOVER).

    Phase 2: Active steering via overlay. The kernel calls gates() to
    get RegimeGates for the current regime, then applies them as
    constraints over _route() output.
    """

    def __init__(self, hold_ticks: int = 3) -> None:
        self.current: DefRegime = DefRegime.ORIENT
        self._ticks_in_regime: int = 0
        self._hold_ticks = hold_ticks

    def update(
        self,
        triggers: TriggerSignals,
        episode_end: bool = False,
    ) -> DefRegime:
        """Evaluate candidates, apply hysteresis, return active regime."""
        candidate = self._evaluate(triggers, episode_end)
        if candidate != self.current:
            if (self._ticks_in_regime >= self._hold_ticks
                    or self._is_urgent(candidate)):
                self.current = candidate
                self._ticks_in_regime = 0
        self._ticks_in_regime += 1
        return self.current

    def gates(self) -> RegimeGates:
        """Return gate configuration for the current regime."""
        return REGIME_GATE_MAP[self.current]

    def reset_episode(self) -> None:
        """Reset to ORIENT at episode start."""
        self.current = DefRegime.ORIENT
        self._ticks_in_regime = 0

    # -- internals --

    def _evaluate(
        self,
        t: TriggerSignals,
        episode_end: bool,
    ) -> DefRegime:
        """Priority-based regime evaluation (highest priority first).

        Threshold values are initial calibration targets. Use JSONL
        telemetry to verify and adjust after smoke tests.
        """
        # P0: Episode end always triggers consolidation
        if episode_end:
            return DefRegime.CONSOLIDATE

        # P1: Budget crisis (stuck / timeout approaching)
        if t.budget > 0.85:
            return DefRegime.RECOVER

        # P2: High surprise + low progress = expectation violated
        # With damped surprise, this triggers on delta_8 spikes or
        # UM invalidations, not on routine decision uncertainty.
        if t.surprise > 0.6 and t.progress < 0.3:
            return DefRegime.EVALUATE

        # P3: High readiness + progress = goal-directed navigation
        if t.readiness > 0.7 and t.progress > 0.4:
            return DefRegime.GOALSEEK

        # P4: Moderate surprise = hypothesis forming
        if t.surprise > 0.25:
            return DefRegime.HYPOTHESIZE

        # P5: Low progress = exploration needed
        if t.progress < 0.3:
            return DefRegime.EXPLORE

        # P6: Default
        return DefRegime.ORIENT

    @staticmethod
    def _is_urgent(candidate: DefRegime) -> bool:
        """Urgent regimes bypass hysteresis hold_ticks."""
        return candidate in (DefRegime.CONSOLIDATE, DefRegime.RECOVER)


# ---------------------------------------------------------------------------
# compute_triggers — free function, testable without kernel
# ---------------------------------------------------------------------------

def compute_triggers(
    *,
    td_err: float = 0.0,
    delta_8: float = 0.0,
    invalidation_count: int = 0,
    target_known: bool = False,
    new_cells_ratio: float = 0.0,
    success_rate_ema: float = 0.0,
    distiller_mode: str = "OFF",
    l2_compressed: bool = False,
    distiller_accuracy: float = 0.0,
    mem_cost: float = 0.0,
    stuck_ticks: int = 0,
    max_steps: int = 200,
    current_step: int = 0,
) -> TriggerSignals:
    """Compute abstract trigger signals from kernel observables.

    All parameters are primitive values — no kernel/object references.
    This makes the function independently testable.

    Args:
        td_err: Temporal difference error [0,1] from MvpTickSignals
        delta_8: Closure residuum delta_8 from ClosureResiduum
        invalidation_count: UM invalidation count (0=calm, >0=surprise)
        target_known: Whether an explicit target exists in memory
        new_cells_ratio: Fraction of newly discovered cells [0,1]
        success_rate_ema: EMA of episode success rate [0,1]
        distiller_mode: "OFF" / "APPRENTICE" / "EXPERT"
        l2_compressed: Whether L2 is compressed in UnifiedMemory
        distiller_accuracy: Current distiller eval accuracy [0,1]
        mem_cost: Memory cost [0,1] from MvpTickSignals
        stuck_ticks: Number of consecutive stuck ticks
        max_steps: Maximum steps per episode
        current_step: Current step in episode
    """
    # ---- Surprise ----
    # td_err is decision uncertainty, not true surprise. Damp it.
    # True surprise comes from delta_8 (fixpoint deviation) and
    # inv_pulse (UM invalidation = new info entered the system).
    inv_pulse = min(1.0, invalidation_count * 0.3) if invalidation_count > 0 else 0.0
    td_surprise = td_err * 0.3  # damped: uncertainty, not surprise
    surprise = max(td_surprise, delta_8, inv_pulse)
    surprise = min(1.0, surprise)

    # ---- Progress ----
    # Weighted combination. Exploration progress (new_cells_ratio) is the
    # primary signal during FRONTIER ticks where target_known=False.
    target_score = 0.4 if target_known else 0.0
    cells_score = min(1.0, new_cells_ratio) * 0.35
    success_score = success_rate_ema * 0.25
    progress = min(1.0, target_score + cells_score + success_score)

    # ---- Readiness ----
    # Step function: OFF=0.0, APPRENTICE=0.3, EXPERT=0.7 (base)
    # l2_compressed adds 0.2, accuracy adds up to 0.1
    mode_map = {"OFF": 0.0, "APPRENTICE": 0.3, "EXPERT": 0.7}
    readiness = mode_map.get(distiller_mode, 0.0)
    if l2_compressed:
        readiness += 0.2
    readiness += min(0.1, distiller_accuracy * 0.1)
    readiness = min(1.0, readiness)

    # ---- Budget ----
    # stuck_ratio + time_ratio only. mem_cost is NOT a budget signal
    # (it measures dict key count, not actual memory pressure).
    # mem_cost still drives deconstruction via _route().
    stuck_ratio = min(1.0, stuck_ticks / max(1, max_steps * 0.05))
    time_ratio = current_step / max(1, max_steps)
    budget = max(stuck_ratio, time_ratio)
    budget = min(1.0, budget)

    return TriggerSignals(
        surprise=round(surprise, 4),
        progress=round(progress, 4),
        readiness=round(readiness, 4),
        budget=round(budget, 4),
    )

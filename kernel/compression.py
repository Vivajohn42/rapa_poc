"""Cascaded Compression Controller for RAPA v2 Unified Memory.

Extends the existing D→C deconstruction with C→B and B→A stages,
implemented as dimensional reduction within the UnifiedMemory store.

Compression stages:
  L3→L2: D's semantic knowledge → C's goal/valence state
         (reuses existing memory_manager.deconstruct pipeline)
  L2→L1: C's goal-conditioned navigation → B's direction prior
         (compresses target into directional preferences)
  L1→L0: B's prediction corrections → A's perception biases
         (pushes directional priors into attention hints)

Each stage fires when its per-layer delta drops BELOW the threshold
(layer is stable enough to compress) AND cooldown has elapsed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from kernel.unified_memory import UnifiedMemory


class CompressionController:
    """Stability-based cascaded compression within UnifiedMemory.

    Uses per-layer deltas from ClosureResiduum (delta_4, c_term, d_term)
    to decide when a layer is stable enough to compress into the layer below.
    Jung profile modulates thresholds, window, and cascade depth.
    """

    def __init__(
        self,
        um: UnifiedMemory,
        memory_manager: Any,  # MvpMemoryManager (avoid circular import)
        *,
        jung_profile: Any = None,
        l3_threshold: float = 0.20,
        l2_threshold: float = 0.15,
        l1_threshold: float = 0.25,
        cooldown: int = 3,
        max_cascade_depth: int = 2,
        max_invalidations_per_episode: int = 3,
    ) -> None:
        self._um = um
        self._mm = memory_manager

        # Apply Jung profile if provided
        if jung_profile is not None:
            self._thresholds = {
                "L3_L2": jung_profile.compression_threshold_l3,
                "L2_L1": jung_profile.compression_threshold_l2,
                "L1_L0": jung_profile.compression_threshold_l1,
            }
            self._cooldown = jung_profile.compression_window
            self._max_cascade = jung_profile.cascade_depth
        else:
            self._thresholds = {
                "L3_L2": l3_threshold,
                "L2_L1": l2_threshold,
                "L1_L0": l1_threshold,
            }
            self._cooldown = cooldown
            self._max_cascade = max_cascade_depth

        self._max_invalidations = max_invalidations_per_episode
        self._last_compress: Dict[str, int] = {
            "L3_L2": -999, "L2_L1": -999, "L1_L0": -999,
        }
        self._compress_log: List[Dict[str, Any]] = []
        self._invalidation_count: int = 0

    # ------------------------------------------------------------------
    # Evaluate & Execute Cascade
    # ------------------------------------------------------------------

    def evaluate_cascade(
        self,
        tick: int,
        residual: Any,  # ResidualSnapshot
        zC: Any,         # ZC
        zD: Optional[Any],  # ZD or None
        goal_map: Optional[Dict] = None,
    ) -> Tuple[List[str], Optional[Any]]:
        """Evaluate and execute compression stages.

        Returns:
            (fired_stages, updated_zC_or_None)
            fired_stages: list of stage names that fired (e.g. ["L3_L2"])
            updated_zC: new ZC if L3→L2 fired (deconstruct changes zC), else None
        """
        fired: List[str] = []
        updated_zC = None
        deltas = self._um.get_layer_deltas(residual)

        # L3→L2: D's semantic → C's target/phase
        # Fires when d_term is LOW (D has stabilized / converged)
        if (not self._um.is_compressed("L3")
                and deltas["L3"] < self._thresholds["L3_L2"]
                and zD is not None
                and tick - self._last_compress["L3_L2"] >= self._cooldown):
            updated_zC = self._mm.deconstruct(zC, zD, goal_map=goal_map)
            self._um.populate_from_zC(updated_zC, tick)
            self._um.compress("L3", "L2", compress_l3_to_l2, tick)
            self._last_compress["L3_L2"] = tick
            fired.append("L3_L2")
            zC = updated_zC  # Use updated zC for subsequent stages

        # Respect cascade depth limit
        if len(fired) >= self._max_cascade:
            self._log(tick, fired, deltas)
            return fired, updated_zC

        # L2→L1: C's target → B's direction prior
        # Fires when c_term is LOW (C's valence is stable)
        # Prerequisite: L3 must already be compressed (hierarchical constraint)
        if (self._um.is_compressed("L3")
                and not self._um.is_compressed("L2")
                and deltas["L2"] < self._thresholds["L2_L1"]
                and tick - self._last_compress["L2_L1"] >= self._cooldown):
            self._um.compress("L2", "L1", compress_l2_to_l1, tick)
            self._last_compress["L2_L1"] = tick
            fired.append("L2_L1")

        if len(fired) >= self._max_cascade:
            self._log(tick, fired, deltas)
            return fired, updated_zC

        # L1→L0: B's direction prior → A's attention bias
        # Fires when delta_4 is LOW (B's predictions are accurate)
        # Prerequisite: L2 must already be compressed (hierarchical constraint)
        if (self._um.is_compressed("L2")
                and not self._um.is_compressed("L1")
                and deltas["L1"] < self._thresholds["L1_L0"]
                and tick - self._last_compress["L1_L0"] >= self._cooldown):
            self._um.compress("L1", "L0", compress_l1_to_l0, tick)
            self._last_compress["L1_L0"] = tick
            fired.append("L1_L0")

        if fired:
            self._log(tick, fired, deltas)
        return fired, updated_zC

    # ------------------------------------------------------------------
    # Cache Invalidation (Surprise Trigger)
    # ------------------------------------------------------------------

    def check_invalidation(
        self,
        residual: Any,  # ResidualSnapshot
        divergence_threshold: float = 0.0,
    ) -> List[str]:
        """Check if compressed layers need reactivation (surprise trigger).

        Returns list of invalidated layers.
        """
        if self._invalidation_count >= self._max_invalidations:
            return []  # Stay deliberative for rest of episode

        invalidated: List[str] = []

        # High prediction error → reactivate L2 + L3
        if residual.delta_4 > 0.6 and self._um.is_compressed("L2"):
            self._um.invalidate("L2")  # Cascades to L3
            invalidated.extend(["L2", "L3"])
            self._invalidation_count += 1

        # High divergence rate → reactivate L3 only
        elif (divergence_threshold > 0
              and residual.d_delta_8_dt > divergence_threshold
              and self._um.is_compressed("L3")):
            self._um.invalidate("L3")
            invalidated.append("L3")
            self._invalidation_count += 1

        return invalidated

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Reset per-episode state (cooldowns, invalidation count)."""
        self._last_compress = {k: -999 for k in self._last_compress}
        self._invalidation_count = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def compression_log(self) -> List[Dict[str, Any]]:
        return self._compress_log

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic info for logging."""
        return {
            "thresholds": dict(self._thresholds),
            "cooldown": self._cooldown,
            "max_cascade": self._max_cascade,
            "last_compress": dict(self._last_compress),
            "total_compressions": len(self._compress_log),
            "invalidation_count": self._invalidation_count,
        }

    def _log(self, tick: int, fired: List[str], deltas: Dict[str, float]) -> None:
        self._compress_log.append({
            "tick": tick,
            "stages": list(fired),
            "deltas": dict(deltas),
        })


# ======================================================================
# Compression Transform Functions
# ======================================================================

def compress_l3_to_l2(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """D's semantic knowledge → C's goal/valence state.

    Extracts target and phase from D's meaning_tags.
    This mirrors what deconstruct_doorkey_d_to_c does, but expressed
    as dimensional reduction within the unified memory.

    Note: The actual D→C deconstruction is done by memory_manager.deconstruct()
    before this transform runs.  This transform marks the compression as complete
    by writing a compression marker.
    """
    result: Dict[str, Any] = {}

    # Mark that L3 content has been absorbed into L2
    tags = snapshot.get("meaning_tags", [])
    if tags:
        result["compressed_from_l3"] = True
        # Extract any target/phase that D may have written
        for tag in tags:
            tag_l = tag.strip().lower() if isinstance(tag, str) else ""
            if tag_l.startswith("phase:"):
                result["phase"] = tag_l.split(":", 1)[1]
            elif tag_l.startswith("target:"):
                result["target_type"] = tag_l.split(":", 1)[1]

    return result


def compress_l2_to_l1(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """C's goal-conditioned navigation → B's direction prior.

    Compresses 'go to target X' into 'prefer these directions'.
    """
    result: Dict[str, Any] = {}

    target = snapshot.get("target")
    agent_pos = snapshot.get("agent_pos")

    if target is not None and agent_pos is not None:
        dy = target[1] - agent_pos[1] if len(target) > 1 else 0
        dx = target[0] - agent_pos[0]

        direction_prior: List[str] = []
        if dx > 0:
            direction_prior.append("right")
        elif dx < 0:
            direction_prior.append("left")
        if dy > 0:
            direction_prior.append("down")
        elif dy < 0:
            direction_prior.append("up")

        result["direction_prior"] = direction_prior

    # Interaction priors from phase context
    memory = snapshot.get("memory", {})
    interaction_prior: Dict[str, bool] = {}
    phase = memory.get("phase", snapshot.get("phase", ""))
    carrying_key = memory.get("has_key", False)

    if phase == "open_door" and carrying_key:
        interaction_prior["at_door_toggle"] = True
    elif phase == "find_key" and not carrying_key:
        interaction_prior["at_key_pickup"] = True

    if interaction_prior:
        result["interaction_prior"] = interaction_prior

    result["compressed_from_l2"] = True
    return result


def compress_l1_to_l0(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """B's prediction corrections → A's perception biases.

    Pushes directional priors into attention hints for A.
    """
    result: Dict[str, Any] = {}

    direction_prior = snapshot.get("direction_prior", [])
    if direction_prior:
        result["attention_bias"] = direction_prior

    result["compressed_from_l1"] = True
    return result

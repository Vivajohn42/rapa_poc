"""Unified Memory — Single shared state store with layer projections.

All streams surf the same memory, but in different dimensionality.
Deconstruction is dimensional reduction within this store,
making higher dimensions redundant for the reading stream below.

Layer hierarchy:
  L0: Raw observations (A reads)     — agent_pos, agent_dir, obstacles
  L1: Sensorimotor state (B reads)   — belief_map, prediction_error
  L2: Goal/valence state (C reads)   — phase, target, carrying_key
  L3: Semantic/narrative (D reads)   — meaning_tags, narrative

Reading at layer N includes all layers 0..N.
Compression (e.g. L3→L2) makes the higher layer redundant.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set


LAYER_ORDER = ("L0", "L1", "L2", "L3")

# FoM regime for each highest-active layer
_REGIME_MAP = {
    frozenset():                  "2FoM",
    frozenset({"L1"}):            "4FoM",
    frozenset({"L1", "L2"}):      "6FoM",
    frozenset({"L1", "L2", "L3"}): "8FoM",
}


class UnifiedMemory:
    """Single shared state store for all RAPA streams.

    Each stream reads a dimensional projection (L0..Ln).
    Deconstruction = dimensional reduction within this store.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {
            layer: {} for layer in LAYER_ORDER
        }
        self._versions: Dict[str, int] = {
            layer: 0 for layer in LAYER_ORDER
        }
        self._last_written_tick: Dict[str, int] = {
            layer: -1 for layer in LAYER_ORDER
        }
        self._compressed: Set[str] = set()
        self._compression_log: List[Dict[str, Any]] = []
        self._invalidation_count: int = 0

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write(self, key: str, value: Any, layer: str, tick: int = -1) -> None:
        """Write a value at a specific dimensional layer."""
        assert layer in self._store, f"Unknown layer: {layer}"
        self._store[layer][key] = value
        self._versions[layer] += 1
        self._last_written_tick[layer] = tick

    def read(self, max_layer: str = "L3") -> Dict[str, Any]:
        """Read projection up to max_layer (inclusive of all below)."""
        result: Dict[str, Any] = {}
        for layer in LAYER_ORDER:
            result.update(self._store[layer])
            if layer == max_layer:
                break
        return result

    def read_layer(self, layer: str) -> Dict[str, Any]:
        """Read a single layer (no projection)."""
        return dict(self._store[layer])

    # ------------------------------------------------------------------
    # Compression (Dimensional Reduction)
    # ------------------------------------------------------------------

    def compress(
        self,
        from_layer: str,
        to_layer: str,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        tick: int = -1,
    ) -> None:
        """Dimensional reduction: compress higher layer into lower layer.

        transform_fn(full_snapshot_up_to_from_layer) -> dict of {key: value}
        to write into to_layer.  The higher layer keys are NOT deleted —
        they become redundant.
        """
        snapshot = self.read(from_layer)
        compressed_data = transform_fn(snapshot)
        for k, v in compressed_data.items():
            self.write(k, v, to_layer, tick)
        self._compressed.add(from_layer)
        self._compression_log.append({
            "tick": tick,
            "from": from_layer,
            "to": to_layer,
            "keys_written": list(compressed_data.keys()),
        })

    def is_compressed(self, layer: str) -> bool:
        """Check if a layer has been compressed into the layer below."""
        return layer in self._compressed

    def needs_layer(self, layer: str) -> bool:
        """Does this layer need active computation, or is it compressed?"""
        return layer not in self._compressed

    # ------------------------------------------------------------------
    # Cache Invalidation (Surprise Trigger)
    # ------------------------------------------------------------------

    def invalidate(self, layer: str) -> None:
        """Cache-flush: reactivate a compressed layer (surprise trigger).

        Cascades upward: invalidating L1 also invalidates L2 and L3.
        """
        idx = LAYER_ORDER.index(layer)
        for i in range(idx, len(LAYER_ORDER)):
            self._compressed.discard(LAYER_ORDER[i])
        self._invalidation_count += 1

    def invalidate_all(self) -> None:
        """Force full re-deliberation (e.g. after environment change)."""
        self._compressed.clear()
        self._invalidation_count += 1

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------

    def get_active_regime(self) -> str:
        """Return current FoM regime based on which layers are still needed."""
        needed = frozenset(
            layer for layer in ("L1", "L2", "L3") if self.needs_layer(layer)
        )
        return _REGIME_MAP.get(needed, "8FoM")

    # ------------------------------------------------------------------
    # Populate helpers (sync from legacy state objects)
    # ------------------------------------------------------------------

    def populate_from_zA(self, zA: Any, tick: int) -> None:
        """Sync L0 from Stream A output (ZA Pydantic model)."""
        self.write("agent_pos", zA.agent_pos, "L0", tick)
        self.write("agent_dir", getattr(zA, "direction", None), "L0", tick)
        self.write("obstacles", zA.obstacles, "L0", tick)
        self.write("hint", zA.hint, "L0", tick)
        self.write("width", zA.width, "L0", tick)
        self.write("height", zA.height, "L0", tick)
        self.write("goal_pos", zA.goal_pos, "L0", tick)

    def populate_from_zC(self, zC: Any, tick: int) -> None:
        """Sync L2 from kernel-managed ZC."""
        self.write("goal_mode", zC.goal_mode, "L2", tick)
        self.write("memory", dict(zC.memory), "L2", tick)
        # Unpack commonly-used keys from memory for direct L2 access
        mem = zC.memory
        if "target" in mem:
            self.write("target", mem["target"], "L2", tick)
        if "phase" in mem:
            self.write("phase", mem["phase"], "L2", tick)

    def populate_from_zD(self, zD: Any, tick: int) -> None:
        """Sync L3 from Stream D output (ZD Pydantic model)."""
        self.write("meaning_tags", list(zD.meaning_tags), "L3", tick)
        self.write("narrative", zD.narrative, "L3", tick)
        self.write("grounding_violations", zD.grounding_violations, "L3", tick)

    # ------------------------------------------------------------------
    # Delta mapping
    # ------------------------------------------------------------------

    def get_layer_deltas(self, residual: Any) -> Dict[str, float]:
        """Map residuum terms to layer stability metrics.

        Uses existing delta_4 (B prediction), c_term (C valence),
        d_term (D meaning) from ClosureResiduum.
        """
        return {
            "L1": residual.delta_4,
            "L2": residual.c_term,
            "L3": residual.d_term,
        }

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Reset for new episode.  L0/L1/L2 cleared, L3 + _compressed persist."""
        for layer in ("L0", "L1", "L2"):
            self._store[layer].clear()
            self._versions[layer] = 0
            self._last_written_tick[layer] = -1
        # L3 persists across episodes (learned compressions)
        # _compressed survives reset
        self._invalidation_count = 0

    def reset_full(self) -> None:
        """Full reset including L3 and compression state."""
        for layer in LAYER_ORDER:
            self._store[layer].clear()
            self._versions[layer] = 0
            self._last_written_tick[layer] = -1
        self._compressed.clear()
        self._compression_log.clear()
        self._invalidation_count = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot for logging/evaluation."""
        return {
            "store_sizes": {
                layer: len(self._store[layer]) for layer in LAYER_ORDER
            },
            "versions": dict(self._versions),
            "compressed": sorted(self._compressed),
            "active_regime": self.get_active_regime(),
            "compression_events": len(self._compression_log),
            "invalidation_count": self._invalidation_count,
        }

    @property
    def compression_log(self) -> List[Dict[str, Any]]:
        return self._compression_log

    @property
    def invalidation_count(self) -> int:
        return self._invalidation_count

    def version(self, layer: str) -> int:
        """Current version of a layer."""
        return self._versions[layer]

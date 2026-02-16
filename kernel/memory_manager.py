"""MvpMemoryManager: L3 persistent memory with D→C→B pipeline.

Port of rapa_os/kernel/memory_manager.py adapted for rapa_mvp's
Pydantic schemas (ZC, ZD) and symbolic agents.

The pipeline:
  1. D→C: deconstruct_d_to_c() parses D's tags into C's memory (existing)
  2. C→B: extracts tie_break_preference from C's memory into B priors
  3. L3:  persistent c_long + b_priors lists (capped at max_entries)

Cooldown is handled by MvpKernel._should_deconstruct(), not here.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from state.schema import ZC, ZD
from router.deconstruct import deconstruct_d_to_c


class MvpMemoryManager:
    """L3 persistent memory with D→C→B deconstruction pipeline."""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.L3: Dict[str, Any] = {
            "c_long": [],      # consolidated C summaries from D
            "b_priors": [],    # tie-break priors pushed to B
            "semantic_index": {},  # tag → coordinate index
        }

    def deconstruct(
        self,
        zC: ZC,
        zD: ZD,
        goal_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> ZC:
        """Full D→C→B pipeline.

        1. D→C via deconstruct_d_to_c (existing, tag-parsing)
        2. Store C summary in L3.c_long
        3. C→B: extract tie_break_preference and store in L3.b_priors
        4. Cap L3 lists

        Returns:
            Updated ZC with new memory entries.
        """
        # --- 1. D→C ---
        zC_new = deconstruct_d_to_c(zC, zD, goal_map=goal_map)

        # --- 2. Store in L3.c_long ---
        summary = {
            "tags": list(zD.meaning_tags),
            "narrative_len": zD.length_chars,
            "grounding_violations": zD.grounding_violations,
            "target": zC_new.memory.get("target"),
            "hint_goal": zC_new.memory.get("hint_goal"),
        }
        self.L3["c_long"].append(summary)

        # Update semantic index
        for tag in zD.meaning_tags:
            tag_lower = tag.strip().lower()
            if tag_lower.startswith("hint:") or tag_lower.startswith("target:"):
                val = tag_lower.split(":", 1)[1].upper()
                target = zC_new.memory.get("target")
                if target is not None:
                    self.L3["semantic_index"][val] = target

        # --- 3. C→B: extract tie_break_preference ---
        tie_break = self._extract_tie_break(zC_new)
        if tie_break:
            prior = {
                "tie_break_preference": tie_break,
                "source": "D_via_C",
                "target": zC_new.memory.get("target"),
            }
            self.L3["b_priors"].append(prior)

        # --- 4. Cap L3 ---
        self._cap_l3()

        return zC_new

    def get_b_priors(self) -> Dict[str, Any]:
        """Get the latest B priors from L3 (for B's predict_next priors)."""
        if self.L3["b_priors"]:
            return self.L3["b_priors"][-1]
        return {}

    def reset_episode(self) -> None:
        """Reset per-episode state. L3 persists across episodes."""
        # L3 persists intentionally — that's the point of L3.
        # Only per-episode buffers would be cleared here.
        pass

    def _extract_tie_break(self, zC: ZC) -> List[str]:
        """Extract tie_break_preference from C's memory.

        Uses the target position to compute preferred movement
        directions (seeking = towards target, avoiding = away from target).
        """
        target = zC.memory.get("target")
        if target is None:
            return []

        # The tie-break preference is already stored by deconstruct_d_to_c
        # via the hint/target system. Here we derive directional priors.
        pref = zC.memory.get("tie_break_preference")
        if isinstance(pref, list):
            return pref

        return []

    def _cap_l3(self) -> None:
        """Cap L3 lists to max_entries."""
        for key in ("c_long", "b_priors"):
            lst = self.L3[key]
            if len(lst) > self.max_entries:
                self.L3[key] = lst[-self.max_entries:]

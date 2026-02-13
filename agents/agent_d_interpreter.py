"""
Agent D with coded hint interpretation capability (Stufe 6).

Extends the deterministic AgentD with the ability to interpret
coded/directional hints into goal identifiers. This is the key
component that creates an information gap between D and no-D variants.

Without this interpreter, coded hints like "goal_at_bottom_right" are
opaque to deconstruct_d_to_c() (which only matches "hint:A", "hint:B").
With it, D translates coded hints into actionable tags.
"""

from typing import Optional, Dict, Tuple, List

from agents.agent_d import AgentD, Event
from env.coded_hints import HintEncoder, EliminationEncoder
from state.schema import ZD


class AgentDInterpreter(AgentD):
    """
    Extended D agent that interprets coded hints.

    In addition to standard narrative generation, this agent:
    1. Detects coded hints in the event buffer
    2. Translates them into goal identifiers using HintEncoder
    3. Adds proper hint:X tags that deconstruct_d_to_c can parse
    """

    def __init__(
        self,
        goal_map: Dict[str, Tuple[int, int]],
        grid_width: int,
        grid_height: int,
        difficulty: str = "medium",
    ):
        super().__init__()
        self.goal_map = goal_map
        self.difficulty = difficulty
        self._hint_encoder = HintEncoder(goal_map, grid_width, grid_height)
        self._elim_encoder = EliminationEncoder(goal_map, grid_width, grid_height)

    def _is_direct_hint(self, hint: str) -> bool:
        """Check if hint is a direct goal ID (not coded)."""
        return hint.upper() in self.goal_map or hint.startswith("not_")

    def interpret_coded_hint(self, coded_hint: str) -> Optional[str]:
        """
        Translate a coded hint into a goal identifier.

        For direct goal hints: decodes via HintEncoder
        For elimination hints: decodes via EliminationEncoder

        Returns:
            Goal ID string (e.g. "A") or None if uninterpretable
        """
        if self._is_direct_hint(coded_hint):
            # Direct hint — no interpretation needed
            return coded_hint.upper() if len(coded_hint) <= 2 else None

        # Try as direct goal hint
        decoded = self._hint_encoder.decode(coded_hint, self.difficulty)
        if decoded is not None:
            return decoded

        return None

    def interpret_coded_elimination(self, coded_hint: str) -> Optional[List[str]]:
        """
        Translate a coded elimination hint into a list of eliminated goal IDs.

        Returns:
            List of eliminated goal IDs or None if uninterpretable
        """
        if self._is_direct_hint(coded_hint) and coded_hint.startswith("not_"):
            # Direct elimination — parse normally
            parts = coded_hint[4:].split("_")
            return [p.upper() for p in parts if p.upper() in self.goal_map]

        # Try as coded elimination
        decoded = self._elim_encoder.decode_elimination(coded_hint, self.difficulty)
        return decoded

    def build_micro(self, goal_mode: str, goal_pos: tuple, last_n: int = 5) -> ZD:
        """
        Extended build_micro that interprets coded hints.

        Process:
        1. Generate standard ZD via parent class
        2. Scan recent events for coded hints
        3. Interpret coded hints and add proper tags
        """
        zd = super().build_micro(goal_mode, goal_pos, last_n)

        # Scan for coded hints in recent events
        slice_events = self.events[-last_n:]
        for e in reversed(slice_events):
            if e.hint is not None and not self._is_direct_hint(e.hint):
                # Coded hint found — try to interpret
                goal_id = self.interpret_coded_hint(e.hint)
                if goal_id is not None:
                    tag = f"hint:{goal_id.lower()}"
                    if tag not in zd.meaning_tags:
                        zd.meaning_tags.append(tag)
                    break

                # Try as elimination hint
                eliminated = self.interpret_coded_elimination(e.hint)
                if eliminated:
                    tag = "not_" + "_".join(g.lower() for g in eliminated)
                    if tag not in zd.meaning_tags:
                        zd.meaning_tags.append(tag)
                    break

        return zd

    def build(self, goal_mode: str, goal_pos: tuple) -> ZD:
        """
        Extended build that interprets coded hints across all events.
        """
        zd = super().build(goal_mode, goal_pos)

        # Scan all events for coded hints
        for e in reversed(self.events):
            if e.hint is not None and not self._is_direct_hint(e.hint):
                goal_id = self.interpret_coded_hint(e.hint)
                if goal_id is not None:
                    tag = f"hint:{goal_id.lower()}"
                    if tag not in zd.meaning_tags:
                        zd.meaning_tags.append(tag)
                    break

                eliminated = self.interpret_coded_elimination(e.hint)
                if eliminated:
                    tag = "not_" + "_".join(g.lower() for g in eliminated)
                    if tag not in zd.meaning_tags:
                        zd.meaning_tags.append(tag)
                    break

        return zd

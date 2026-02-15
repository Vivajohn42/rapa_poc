"""
Agent D-LLM with coded hint interpretation capability (Stufe 6-LLM).

Combines AgentDLLM (stochastic LLM-based narrative) with deterministic
HintEncoder/EliminationEncoder for coded hint interpretation.

Architecture insight: The narrative is LLM-generated (stochastic), but
hint interpretation is fully deterministic (HintEncoder.decode). This
tests whether LLM variability in narrative generation disrupts the
deterministic hint-decoding pipeline — a robustness result, not a
capability result.
"""

from typing import Optional, Dict, Tuple, List

from agents.agent_d_llm import AgentDLLM, Event
from env.coded_hints import HintEncoder, EliminationEncoder
from llm.provider import LLMProvider
from state.schema import ZA, ZD


class AgentDLLMInterpreter(AgentDLLM):
    """
    LLM-backed D with deterministic coded hint interpretation.

    Inherits LLM narrative generation from AgentDLLM, adds:
    1. Coded hint detection in event buffer
    2. Deterministic translation via HintEncoder/EliminationEncoder
    3. Proper hint:X / not_x_y tags for deconstruct_d_to_c
    """

    def __init__(
        self,
        llm: LLMProvider,
        goal_map: Dict[str, Tuple[int, int]],
        grid_width: int,
        grid_height: int,
        difficulty: str = "medium",
    ):
        super().__init__(llm)
        self.goal_map = goal_map
        self.difficulty = difficulty
        self._hint_encoder = HintEncoder(goal_map, grid_width, grid_height)
        self._elim_encoder = EliminationEncoder(goal_map, grid_width, grid_height)

    def _is_direct_hint(self, hint: str) -> bool:
        """Check if hint is a direct goal ID (not coded)."""
        return hint.upper() in self.goal_map or hint.startswith("not_")

    def interpret_coded_hint(self, coded_hint: str) -> Optional[str]:
        """
        Translate a coded hint into a goal identifier via HintEncoder.

        Returns:
            Goal ID string (e.g. "A") or None if uninterpretable
        """
        if self._is_direct_hint(coded_hint):
            return coded_hint.upper() if len(coded_hint) <= 2 else None

        decoded = self._hint_encoder.decode(coded_hint, self.difficulty)
        if decoded is not None:
            return decoded

        return None

    def interpret_coded_elimination(self, coded_hint: str) -> Optional[List[str]]:
        """
        Translate a coded elimination hint into eliminated goal IDs.

        Returns:
            List of eliminated goal IDs or None if uninterpretable
        """
        if self._is_direct_hint(coded_hint) and coded_hint.startswith("not_"):
            parts = coded_hint[4:].split("_")
            return [p.upper() for p in parts if p.upper() in self.goal_map]

        decoded = self._elim_encoder.decode_elimination(coded_hint, self.difficulty)
        return decoded

    def interpret_hint_only(self, zA_hint: ZA) -> Optional[ZD]:
        """
        Decode a coded hint deterministically WITHOUT making an LLM call.

        This is the fast path for hint processing: only HintEncoder/
        EliminationEncoder are used. No narrative is generated, no LLM
        inference happens. Returns a minimal ZD with the decoded tag,
        or None if the hint cannot be decoded.
        """
        if zA_hint.hint is None:
            return None

        hint = zA_hint.hint

        # Direct hints (e.g. "A", "B") are handled by parent's
        # deterministic tag injection in build_micro — skip here
        if self._is_direct_hint(hint):
            if hint.upper() in self.goal_map and len(hint) <= 2:
                return ZD(
                    narrative="hint_direct",
                    meaning_tags=[f"hint:{hint.upper()}"],
                    length_chars=11,
                    grounding_violations=0,
                )
            if hint.startswith("not_"):
                parts = hint[4:].split("_")
                elim = [p.upper() for p in parts if p.upper() in self.goal_map]
                if elim:
                    tag = "not_" + "_".join(g.lower() for g in elim)
                    return ZD(
                        narrative="elimination_direct",
                        meaning_tags=[tag],
                        length_chars=19,
                        grounding_violations=0,
                    )
            return None

        # Try coded goal hint
        goal_id = self.interpret_coded_hint(hint)
        if goal_id is not None:
            return ZD(
                narrative="hint_decoded",
                meaning_tags=[f"hint:{goal_id.lower()}"],
                length_chars=12,
                grounding_violations=0,
            )

        # Try coded elimination hint
        eliminated = self.interpret_coded_elimination(hint)
        if eliminated:
            tag = "not_" + "_".join(g.lower() for g in eliminated)
            return ZD(
                narrative="elimination_decoded",
                meaning_tags=[tag],
                length_chars=19,
                grounding_violations=0,
            )

        return None

    def build_micro(self, goal_mode: str, goal_pos=None, last_n: int = 6) -> ZD:
        """
        LLM narrative + deterministic coded hint interpretation.

        Process:
        1. Generate LLM-based ZD via parent class (stochastic narrative)
        2. Scan recent events for coded hints (deterministic)
        3. Decode coded hints via HintEncoder (deterministic)
        4. Add proper hint:X tags for deconstruct_d_to_c
        """
        zd = super().build_micro(goal_mode, goal_pos, last_n)

        slice_events = self.events[-last_n:]
        for e in reversed(slice_events):
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

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """
        LLM narrative + deterministic coded hint interpretation (full episode).
        """
        zd = super().build(goal_mode, goal_pos)

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

"""Environment-specific adapters for UniversalLlmD.

Each adapter translates between a specific environment's context and
the generic LLM prompt format used by UniversalLlmD.  The adapter
pattern separates WHAT the LLM sees (env context) from HOW D works
(observe, build, parse, ground).

Adapter responsibilities:
  - extract_event_context(): pull env-specific fields from ZA
  - build_system_prompt() / build_user_prompt(): format the LLM prompt
  - validate_grounding(): count hallucinations in LLM tags
  - force_deterministic_tags(): inject tags that must NOT rely on LLM
  - on_new_clue(): track collected clues/evidence for prompt building

Important: Adapter methods that receive event dicts must use .get()
for adapter-specific keys, because events from different environments
have different fields.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


class LlmDAdapter(ABC):
    """Abstract adapter protocol for environment-specific LLM-D context."""

    @abstractmethod
    def extract_event_context(self, zA) -> Dict[str, Any]:
        """Extract environment-specific fields from ZA for event recording.

        Returns a dict merged into the generic event dict.  Keys vary
        by environment (hint, room_id, clue, state_id, ...).
        """
        ...

    @abstractmethod
    def build_system_prompt(self, micro: bool) -> str:
        """Build the system prompt (role + rules + output format).

        The output format MUST instruct the LLM to produce exactly:
            NARRATIVE: <text>
            TAGS: <comma-separated tags>
        """
        ...

    @abstractmethod
    def build_user_prompt(
        self,
        events: List[Dict[str, Any]],
        goal_mode: str,
    ) -> str:
        """Build the user prompt from event history and goal mode."""
        ...

    @abstractmethod
    def validate_grounding(self, tags: List[str]) -> int:
        """Count grounding violations in LLM-generated tags.

        Returns the number of violations (0 = fully grounded).
        """
        ...

    @abstractmethod
    def force_deterministic_tags(
        self,
        tags: List[str],
        events: List[Dict[str, Any]],
        goal_mode: str,
    ) -> List[str]:
        """Inject deterministic tags that must NOT rely on LLM output.

        Critical reliability hook.  Examples:
        - GridWorld: forces hint:A/hint:B/not_x_y from event buffer
        - TextWorld/Riddle: ensures goal:{mode} tag is present

        Returns the (possibly modified) tags list.
        """
        ...

    @abstractmethod
    def on_new_clue(self, clue: str) -> None:
        """Called when a new clue/hint is observed.

        TextWorld/Riddle adapters accumulate clues for prompt building.
        GridWorld adapter is a no-op.
        """
        ...


# ── GridWorld Adapter ─────────────────────────────────────────────────


class GridWorldLlmAdapter(LlmDAdapter):
    """Adapter for GridWorld: positional FACTS + deterministic hint injection.

    Replicates the prompt format of AgentDLLM (agents/agent_d_llm.py).
    """

    def extract_event_context(self, zA) -> Dict[str, Any]:
        return {"hint": zA.hint}

    def build_system_prompt(self, micro: bool) -> str:
        n_sent = "1-2 short sentences" if micro else "2-4 short sentences"
        return (
            "You are a narrative/meaning module.\n"
            "RULES:\n"
            "- Use ONLY the FACTS provided.\n"
            "- Do NOT invent positions/actions/rewards/hints.\n"
            "- Output EXACTLY two lines:\n"
            f"NARRATIVE: <{n_sent}>\n"
            "TAGS: <comma-separated tags; include hint:A or hint:B "
            "if any hint appears>\n"
        )

    def build_user_prompt(
        self, events: List[Dict[str, Any]], goal_mode: str,
    ) -> str:
        facts = []
        for e in events:
            facts.append(
                f"t={e['t']} pos={e['agent_pos']} action={e['action']} "
                f"reward={e['reward']} done={e['done']} hint={e.get('hint')}"
            )
        return "FACTS:\n" + "\n".join(facts) + f"\nMODE={goal_mode}\n"

    def validate_grounding(self, tags: List[str]) -> int:
        # GridWorld has no entity hallucination risk.
        return 0

    def force_deterministic_tags(
        self,
        tags: List[str],
        events: List[Dict[str, Any]],
        goal_mode: str,
    ) -> List[str]:
        # 1. Scan events backward for simple hint values (A, B)
        hint_val = None
        for e in reversed(events):
            h = e.get("hint")
            if isinstance(h, str) and h.upper() in ("A", "B"):
                hint_val = h.upper()
                break

        if hint_val:
            forced = f"hint:{hint_val}"
            existing_lower = [t.lower() for t in tags]
            if forced.lower() not in existing_lower:
                tags.append(forced)

        # 2. Pass through elimination hints (not_x_y) directly as tags
        for e in reversed(events):
            h = e.get("hint")
            if isinstance(h, str) and h.startswith("not_"):
                if h not in tags and h.lower() not in [t.lower() for t in tags]:
                    tags.append(h)
                break  # only the most recent elimination hint

        return tags

    def on_new_clue(self, clue: str) -> None:
        pass  # GridWorld hints are event-driven, no accumulation needed


# ── TextWorld Adapter ─────────────────────────────────────────────────


class TextWorldLlmAdapter(LlmDAdapter):
    """Adapter for TextWorld: room properties + clue synthesis.

    Replicates the prompt format of TextAgentDLLM
    (agents/text_agent_d_llm.py).
    """

    def __init__(
        self,
        room_properties: Dict[str, Set[str]],
        all_rooms: List[str],
        room_index: Dict[str, int],
    ):
        self.room_properties = room_properties
        self.all_rooms = list(all_rooms)
        self.room_index = room_index
        self._index_to_room: Dict[int, str] = {
            v: k for k, v in room_index.items()
        }
        self._clues: List[str] = []

    def extract_event_context(self, zA) -> Dict[str, Any]:
        idx = zA.agent_pos[0]
        room_id = self._index_to_room.get(idx, "")
        return {"room_id": room_id, "clue": zA.hint}

    def build_system_prompt(self, micro: bool) -> str:
        return (
            "You are a detective analyzing clues to identify a target room.\n"
            "RULES:\n"
            "- Analyze ALL clues together to narrow down which room is the target.\n"
            "- Each clue eliminates some rooms or requires certain properties.\n"
            "- The target is the ONLY room that satisfies ALL clue constraints.\n"
            "- You MUST pick from the room list below. Do NOT invent rooms.\n"
            "- Output EXACTLY two lines:\n"
            "NARRATIVE: <1-2 sentences explaining your reasoning>\n"
            f"TAGS: target:<room_id>,clue_collected:{len(self._clues)},"
            "candidates:<N>\n"
            "  where <room_id> is your best guess (or UNKNOWN if unsure),\n"
            "  and <N> is how many rooms still match all constraints.\n"
        )

    def build_user_prompt(
        self, events: List[Dict[str, Any]], goal_mode: str,
    ) -> str:
        room_info = []
        for rid in self.all_rooms:
            props = self.room_properties.get(rid, set())
            room_info.append(f"  {rid}: properties={sorted(props)}")

        clue_info = []
        for i, clue in enumerate(self._clues, 1):
            clue_info.append(f'  Clue {i}: "{clue}"')

        parts = ["ROOMS:"] + room_info + ["", "CLUES COLLECTED:"]
        if clue_info:
            parts.extend(clue_info)
        else:
            parts.append("  (none yet)")
        parts.append("")
        parts.append("Which room is the target? Explain your reasoning.")
        return "\n".join(parts)

    def validate_grounding(self, tags: List[str]) -> int:
        violations = 0
        for tag in tags:
            if tag.startswith("target:"):
                claimed = tag.split(":", 1)[1].strip()
                if claimed != "UNKNOWN" and claimed not in self.all_rooms:
                    violations += 1
        return violations

    def force_deterministic_tags(
        self,
        tags: List[str],
        events: List[Dict[str, Any]],
        goal_mode: str,
    ) -> List[str]:
        if not any(t.startswith("goal:") for t in tags):
            tags.append(f"goal:{goal_mode}")
        return tags

    def on_new_clue(self, clue: str) -> None:
        if clue not in self._clues:
            self._clues.append(clue)


# ── Riddle Rooms Adapter ──────────────────────────────────────────────


class RiddleLlmAdapter(LlmDAdapter):
    """Adapter for Riddle Rooms: evidence/answer synthesis via LLM.

    NEW — no existing LLM-D for Riddle Rooms.  The LLM receives:
    - List of possible answers with their properties
    - Collected evidence (clue texts from test actions)
    - The puzzle description

    The LLM must identify which answer is correct based on elimination
    logic.  This is genuine reasoning, not pattern matching.
    """

    def __init__(
        self,
        answer_properties: Dict[str, Set[str]],
        answer_set: List[str],
        answer_index: Dict[str, int],
        clue_eliminates: Dict[str, List[str]],
        puzzle_description: str = "",
    ):
        self.answer_properties = answer_properties
        self.all_answers = list(answer_set)
        self.answer_index = answer_index
        self.clue_eliminates = clue_eliminates
        self.puzzle_description = puzzle_description
        self._clues: List[str] = []

    def extract_event_context(self, zA) -> Dict[str, Any]:
        return {
            "state_id": f"state_{zA.agent_pos[0]}",
            "clue": zA.hint,
        }

    def build_system_prompt(self, micro: bool) -> str:
        return (
            "You are a logic puzzle solver analyzing evidence to identify "
            "the correct answer.\n"
            "RULES:\n"
            "- Analyze ALL evidence clues together.\n"
            "- Each clue may eliminate one or more possible answers.\n"
            "- The correct answer is the ONLY one not eliminated by any clue.\n"
            "- You MUST pick from the answer list below. Do NOT invent answers.\n"
            "- Output EXACTLY two lines:\n"
            "NARRATIVE: <1-2 sentences explaining your reasoning>\n"
            "TAGS: answer:<answer_id>,candidates:<N>,evidence:<M>,"
            "eliminated:<id1>,eliminated:<id2>,...\n"
            "  where <answer_id> is your best guess (or UNKNOWN if unsure),\n"
            "  <N> is how many answers are still possible,\n"
            "  <M> is how many clues you have.\n"
            "  Add eliminated:<id> for each answer you can rule out.\n"
        )

    def build_user_prompt(
        self, events: List[Dict[str, Any]], goal_mode: str,
    ) -> str:
        parts: List[str] = []
        if self.puzzle_description:
            parts.append(f"PUZZLE: {self.puzzle_description}")
            parts.append("")

        parts.append("POSSIBLE ANSWERS:")
        for ans_id in self.all_answers:
            props = self.answer_properties.get(ans_id, set())
            parts.append(f"  {ans_id}: properties={sorted(props)}")
        parts.append("")

        parts.append("EVIDENCE COLLECTED:")
        if self._clues:
            for i, clue in enumerate(self._clues, 1):
                parts.append(f'  Evidence {i}: "{clue}"')
        else:
            parts.append("  (no evidence yet)")
        parts.append("")
        parts.append("Which answer is correct? Explain your reasoning.")
        return "\n".join(parts)

    def validate_grounding(self, tags: List[str]) -> int:
        violations = 0
        for tag in tags:
            if tag.startswith("answer:") or tag.startswith("target:"):
                claimed = tag.split(":", 1)[1].strip()
                if claimed != "UNKNOWN" and claimed not in self.all_answers:
                    violations += 1
            elif tag.startswith("eliminated:"):
                claimed = tag.split(":", 1)[1].strip()
                if claimed not in self.all_answers:
                    violations += 1
        return violations

    def force_deterministic_tags(
        self,
        tags: List[str],
        events: List[Dict[str, Any]],
        goal_mode: str,
    ) -> List[str]:
        if not any(t.startswith("goal:") for t in tags):
            tags.append(f"goal:{goal_mode}")
        if not any(t.startswith("evidence:") for t in tags):
            tags.append(f"evidence:{len(self._clues)}")
        return tags

    def on_new_clue(self, clue: str) -> None:
        if clue not in self._clues:
            self._clues.append(clue)

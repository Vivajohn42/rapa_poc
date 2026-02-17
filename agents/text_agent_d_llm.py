"""TextAgentDLLM: LLM-based clue synthesis for TextWorld.

Uses an LLM to synthesize scattered clue fragments and identify the target room.
This is D in its natural role: semantic processing of natural language clues.

g_AD becomes meaningful here:
  - LLM may hallucinate non-existent rooms → g_AD drops
  - LLM may contradict collected clues → g_AD drops
  - Correct synthesis → g_AD stays high

g_DC becomes a genuine bottleneck:
  - Wrong synthesis → C navigates to wrong room → agent fails
  - g_DC measures whether D's output is actionable
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from state.schema import ZA, ZD
from llm.provider import LLMProvider


@dataclass
class TextEvent:
    t: int
    room_id: str
    action: str
    reward: float
    done: bool
    clue: Optional[str]


from kernel.interfaces import StreamD


class TextAgentDLLM(StreamD):
    """LLM-backed clue synthesizer for TextWorld."""

    def __init__(
        self,
        llm: LLMProvider,
        room_properties: Dict[str, Set[str]],
        all_rooms: List[str],
        room_index: Dict[str, int],
    ):
        self.llm = llm
        self.room_properties = room_properties
        self.all_rooms = list(all_rooms)
        self.room_index = room_index
        self.events: List[TextEvent] = []
        self.seen_positions: Set[Tuple[int, int]] = set()
        self._clues: List[str] = []
        self._target: Optional[str] = None

    def observe_step(self, t: int, zA: ZA, action: str, reward: float, done: bool):
        """Record step. Capture clue if present in zA.hint."""
        idx = zA.agent_pos[0]
        room_id = ""
        for rid, ridx in self.room_index.items():
            if ridx == idx:
                room_id = rid
                break

        self.events.append(TextEvent(
            t=t, room_id=room_id, action=action,
            reward=reward, done=done, clue=zA.hint,
        ))
        self.seen_positions.add(zA.agent_pos)

        # Collect new clue
        if zA.hint is not None and zA.hint not in self._clues:
            self._clues.append(zA.hint)

    def _build_prompt(self, micro: bool = False, last_n: int = 5) -> Tuple[str, str]:
        """Build system + user prompt for LLM synthesis."""
        # Room info
        room_info = []
        for rid in self.all_rooms:
            props = self.room_properties.get(rid, set())
            room_info.append(f"  {rid}: properties={sorted(props)}")

        # Clue info
        clue_info = []
        for i, clue in enumerate(self._clues, 1):
            clue_info.append(f"  Clue {i}: \"{clue}\"")

        system = (
            "You are a detective analyzing clues to identify a target room.\n"
            "RULES:\n"
            "- Analyze ALL clues together to narrow down which room is the target.\n"
            "- Each clue eliminates some rooms or requires certain properties.\n"
            "- The target is the ONLY room that satisfies ALL clue constraints.\n"
            "- You MUST pick from the room list below. Do NOT invent rooms.\n"
            "- Output EXACTLY two lines:\n"
            "NARRATIVE: <1-2 sentences explaining your reasoning>\n"
            f"TAGS: target:<room_id>,clue_collected:{len(self._clues)},candidates:<N>\n"
            "  where <room_id> is your best guess (or UNKNOWN if unsure),\n"
            "  and <N> is how many rooms still match all constraints.\n"
        )

        user_parts = ["ROOMS:"]
        user_parts.extend(room_info)
        user_parts.append("")
        user_parts.append("CLUES COLLECTED:")
        if clue_info:
            user_parts.extend(clue_info)
        else:
            user_parts.append("  (none yet)")
        user_parts.append("")
        user_parts.append("Which room is the target? Explain your reasoning.")

        return system, "\n".join(user_parts)

    def _parse_llm_response(self, txt: str) -> Tuple[str, List[str]]:
        """Parse LLM response into narrative + tags."""
        narrative = ""
        tags: List[str] = []

        for line in txt.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith("NARRATIVE:"):
                narrative = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.upper().startswith("TAGS:"):
                raw = line_stripped.split(":", 1)[1].strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]

        # Fallback
        if not narrative:
            narrative = txt[:240]
        if not tags:
            tags = ["llm_format_fallback"]

        # Extract target from tags
        self._target = None
        for tag in tags:
            if tag.startswith("target:") and tag != "target:UNKNOWN":
                claimed_room = tag.split(":", 1)[1].strip()
                if claimed_room in self.all_rooms:
                    self._target = claimed_room
                # If LLM claims a non-existent room, _target stays None
                # (grounding violation detected by g_AD)

        return narrative, tags

    def _validate_grounding(self, tags: List[str]) -> int:
        """Count grounding violations in LLM output."""
        violations = 0

        for tag in tags:
            if tag.startswith("target:"):
                claimed = tag.split(":", 1)[1].strip()
                if claimed != "UNKNOWN" and claimed not in self.all_rooms:
                    violations += 1  # hallucinated room

        return violations

    def build_micro(self, goal_mode: str, goal_pos=None, last_n: int = 5) -> ZD:
        """Short LLM synthesis from collected clues."""
        system, user = self._build_prompt(micro=True, last_n=last_n)

        txt = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=150,
        ).strip()

        narrative, tags = self._parse_llm_response(txt)
        violations = self._validate_grounding(tags)

        # Ensure goal tag
        if not any(t.startswith("goal:") for t in tags):
            tags.append(f"goal:{goal_mode}")

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=violations,
        )

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Full episode synthesis via LLM."""
        system, user = self._build_prompt(micro=False)

        txt = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=200,
        ).strip()

        narrative, tags = self._parse_llm_response(txt)
        violations = self._validate_grounding(tags)

        if not any(t.startswith("goal:") for t in tags):
            tags.append(f"goal:{goal_mode}")

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=violations,
        )

"""TextAgentD: Deterministic clue synthesis for TextWorld.

Maintains a buffer of collected clue fragments and performs constraint propagation
to identify the target room. Unlike GridWorld D (which just passes through hints),
this D performs genuine multi-clue synthesis — no single clue identifies the target.

Tags emitted:
  - "target:{room_id}" when synthesis succeeds (unique room identified)
  - "candidates:{N}" number of remaining candidate rooms
  - "clue_collected:{N}" number of clues collected so far
  - "goal:seek" / "goal:avoid" current goal mode
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from state.schema import ZA, ZD
from kernel.interfaces import StreamD


@dataclass
class TextEvent:
    t: int
    room_id: str
    action: str
    reward: float
    done: bool
    clue: Optional[str]


class TextAgentD(StreamD):
    """Deterministic clue synthesizer via constraint propagation."""

    def __init__(
        self,
        room_properties: Dict[str, Set[str]],
        all_rooms: List[str],
        room_index: Dict[str, int],
    ):
        self.room_properties = room_properties
        self.all_rooms = list(all_rooms)
        self.room_index = room_index
        self.events: List[TextEvent] = []
        self.seen_positions = set()
        self._clues: List[str] = []
        self._candidates: Set[str] = set(all_rooms)
        self._target: Optional[str] = None

    def observe_step(self, t: int, zA: ZA, action: str, reward: float, done: bool):
        """Record step. Capture clue if present in zA.hint."""
        # Reverse-map pseudo-position to room_id
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
            self._synthesize()

    def _synthesize(self):
        """Constraint propagation: apply all clues to narrow candidates."""
        candidates = set(self.all_rooms)

        for clue in self._clues:
            clue_lower = clue.lower()

            # Property elimination: "not in a room with X"
            for prop in self._extract_negated_properties(clue_lower):
                candidates = {r for r in candidates
                              if prop not in self.room_properties.get(r, set())}

            # Property requirement: "where X meets Y", "near water", etc.
            for prop in self._extract_required_properties(clue_lower):
                candidates = {r for r in candidates
                              if prop in self.room_properties.get(r, set())}

        self._candidates = candidates
        if len(candidates) == 1:
            self._target = next(iter(candidates))
        else:
            self._target = None

    def _extract_negated_properties(self, clue: str) -> List[str]:
        """Extract properties that should NOT be present."""
        props = []
        negation_patterns = [
            ("not in a room with windows", "has_windows"),
            ("not in a room with", None),  # generic
            ("not exposed to the sky", "outdoor"),
            ("not outdoors", "outdoor"),
            ("not outdoor", "outdoor"),
            ("no electronic", "has_electronics"),
            ("neither bright", "bright"),
            ("nor large", "large"),
            ("not where people eat", "warm"),  # heuristic
            ("without machines", "has_equipment"),
        ]
        for pattern, prop in negation_patterns:
            if pattern in clue and prop:
                props.append(prop)

        # Generic: "not in a room with {property}"
        if "not in a room with" in clue:
            for prop_name in ["windows", "water", "books", "glass", "candles",
                              "tools", "electronics", "stone"]:
                if prop_name in clue:
                    mapped = f"has_{prop_name}"
                    if mapped not in props:
                        props.append(mapped)

        return props

    def _extract_required_properties(self, clue: str) -> List[str]:
        """Extract properties that MUST be present."""
        props = []
        requirement_patterns = [
            ("silence meets stone", "has_stone"),
            ("meets stone", "has_stone"),
            ("near water", "has_water"),
            ("wood meets water", "has_wood"),
            ("under a roof", "has_roof"),
            ("underground", "underground"),
            ("coldest room", "cold"),
            # "without machines" handled in _extract_negated_properties
            ("smaller than", "small"),
            ("where books sleep", "has_books"),
        ]
        for pattern, prop in requirement_patterns:
            if pattern in clue and prop:
                props.append(prop)

        return props

    def build_micro(self, goal_mode: str, goal_pos=None, last_n: int = 5) -> ZD:
        """Short synthesis result from collected clues."""
        tags = self._build_tags(goal_mode)
        narrative = self._build_narrative(micro=True, last_n=last_n)
        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0,
        )

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Full episode narrative with synthesis result."""
        tags = self._build_tags(goal_mode)
        narrative = self._build_narrative(micro=False)
        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0,
        )

    def _build_tags(self, goal_mode: str) -> List[str]:
        tags = []
        tags.append(f"goal:{goal_mode}")
        tags.append(f"clue_collected:{len(self._clues)}")
        tags.append(f"candidates:{len(self._candidates)}")

        if self._target:
            tags.append(f"target:{self._target}")

        if not self.events:
            tags.append("empty")

        return tags

    def _build_narrative(self, micro: bool = False, last_n: int = 5) -> str:
        if not self.events:
            return "No events recorded."

        evts = self.events[-last_n:] if micro else self.events
        rooms_visited = [e.room_id for e in evts if e.room_id]
        unique_rooms = list(dict.fromkeys(rooms_visited))

        parts = []
        if micro:
            parts.append(f"Micro-synthesis (last {len(evts)} steps):")
        else:
            parts.append("Episode synthesis:")

        parts.append(f"Visited rooms: {', '.join(unique_rooms)}.")
        parts.append(f"Clues collected: {len(self._clues)}.")

        if self._target:
            parts.append(f"Target identified: {self._target}.")
        else:
            parts.append(f"Target unknown. {len(self._candidates)} candidates remain.")

        return " ".join(parts)

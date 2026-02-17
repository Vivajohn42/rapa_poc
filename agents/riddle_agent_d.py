"""RiddleAgentD: Deterministic constraint-propagation D for Riddle Rooms.

Collects evidence fragments from test actions and applies constraint
propagation to eliminate impossible answers.  When only one candidate
remains, the answer is identified and tagged.

Tags emitted:
  - "answer:{id}" when synthesis succeeds (unique answer identified)
  - "eliminated:{id}" for each eliminated answer
  - "evidence:{N}" number of evidence fragments collected
  - "candidates:{N}" number of remaining candidate answers
  - "goal:seek" / "goal:avoid"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from state.schema import ZA, ZD
from kernel.interfaces import StreamD


@dataclass
class RiddleEvent:
    t: int
    state_id: str
    action: str
    reward: float
    done: bool
    clue: Optional[str]


class RiddleAgentD(StreamD):
    """Deterministic clue synthesizer via constraint propagation."""

    def __init__(
        self,
        answer_properties: Dict[str, Set[str]],
        answer_set: List[str],
        answer_index: Dict[str, int],
        clue_eliminates: Dict[str, List[str]],
    ):
        self.answer_properties = answer_properties
        self.all_answers = list(answer_set)
        self.answer_index = answer_index
        self.clue_eliminates = clue_eliminates  # clue_text -> list of answers to eliminate

        self.events: List[RiddleEvent] = []
        self.seen_positions: set = set()

        # Synthesis state
        self._clues: List[str] = []
        self._candidates: Set[str] = set(answer_set)
        self._target: Optional[str] = None

    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        state_id = f"state_{zA.agent_pos[0]}"
        clue = zA.hint
        self.events.append(RiddleEvent(
            t=t, state_id=state_id, action=action,
            reward=reward, done=done, clue=clue,
        ))
        self.seen_positions.add(zA.agent_pos)

        if clue is not None and clue not in self._clues:
            self._clues.append(clue)
            self._synthesize()

    def _synthesize(self) -> None:
        """Apply constraint propagation: eliminate answers based on clues."""
        for clue_text in self._clues:
            # Match clue text against known elimination rules
            eliminates = self.clue_eliminates.get(clue_text, [])
            for ans in eliminates:
                self._candidates.discard(ans)

        if len(self._candidates) == 1:
            self._target = next(iter(self._candidates))

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        return self._build_narrative(goal_mode, self.events)

    def build_micro(self, goal_mode: str, goal_pos=None, last_n: int = 5) -> ZD:
        return self._build_narrative(goal_mode, self.events[-last_n:])

    def _build_narrative(self, goal_mode: str, events: List[RiddleEvent]) -> ZD:
        if not events:
            return ZD(
                narrative="No events recorded.",
                meaning_tags=["empty"],
                length_chars=20,
                grounding_violations=0,
            )

        n_clues = len(self._clues)
        n_candidates = len(self._candidates)

        tags = [
            f"goal:{goal_mode}",
            f"evidence:{n_clues}",
            f"candidates:{n_candidates}",
        ]

        if self._target is not None:
            tags.append(f"answer:{self._target}")
            tags.append(f"target:{self._target}")

        # Add eliminated tags
        eliminated = set(self.all_answers) - self._candidates
        for ans in sorted(eliminated):
            tags.append(f"eliminated:{ans}")

        prefix = "Riddle synthesis:" if len(events) > 5 else "Riddle micro:"
        narrative = (
            f"{prefix} Collected {n_clues} clue(s). "
            f"Candidates remaining: {n_candidates} of {len(self.all_answers)}. "
        )
        if self._target:
            narrative += f"Answer identified: {self._target}."
        else:
            narrative += f"Still narrowing: {sorted(self._candidates)}."

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0,
        )

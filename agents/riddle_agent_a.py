"""RiddleAgentA: Perception stream for Riddle Rooms.

Maps evidence-gathering state to ZA using pseudo-positions:
  width  = len(answer_set)
  height = 1
  agent_pos = (evidence_hash % width, 0)
  hint = newly revealed clue (or None)
"""
from __future__ import annotations

from typing import Dict, Optional

from state.schema import ZA
from kernel.interfaces import StreamA


class RiddleAgentA(StreamA):
    """Parse riddle observation into ZA belief state."""

    def __init__(self, answer_index: Dict[str, int], n_answers: int):
        self.answer_index = answer_index
        self.n_answers = n_answers

    def infer_zA(self, obs) -> ZA:
        # Extract evidence hash from room_id
        room_id = obs.get("room_id", "state_0")
        try:
            state_idx = int(room_id.split("_", 1)[1]) % self.n_answers
        except (IndexError, ValueError):
            state_idx = 0

        return ZA(
            width=self.n_answers,
            height=1,
            agent_pos=(state_idx, 0),
            goal_pos=(-1, -1),
            obstacles=[],
            hint=obs.get("clue"),
        )

"""TextAgentA: Perception for TextWorld.

Parses text observations into ZA using pseudo-positions (room_index -> tuple).
"""
from __future__ import annotations

from typing import Dict, List

from state.schema import ZA


class TextAgentA:
    """Parse TextWorld observation dict into ZA."""

    def __init__(self, room_index: Dict[str, int], n_rooms: int):
        self.room_index = room_index
        self.n_rooms = n_rooms

    def infer_zA(self, obs: dict) -> ZA:
        room_id = obs["room_id"]
        idx = self.room_index.get(room_id, 0)
        return ZA(
            width=self.n_rooms,
            height=1,
            agent_pos=(idx, 0),
            goal_pos=(-1, -1),
            obstacles=[],
            hint=obs.get("clue"),
        )

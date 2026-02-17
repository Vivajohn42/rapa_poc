"""TextAgentB: Forward model for TextWorld.

Graph-based transition model: predict_next returns ZA with updated pseudo-position.
"""
from __future__ import annotations

from typing import Dict

from state.schema import ZA
from kernel.interfaces import StreamB


class TextAgentB(StreamB):
    """Deterministic room-graph forward model."""

    def __init__(
        self,
        room_graph: Dict[str, Dict[str, str]],
        room_index: Dict[str, int],
        index_to_room: Dict[int, str],
    ):
        self.room_graph = room_graph
        self.room_index = room_index
        self.index_to_room = index_to_room

    def predict_next(self, zA: ZA, action: str) -> ZA:
        """Predict next state after taking an exit.

        If action is a valid exit from the current room, returns ZA with
        the destination room's pseudo-position. Otherwise stays in place.
        """
        current_idx = zA.agent_pos[0]
        current_room = self.index_to_room.get(current_idx)

        if current_room and current_room in self.room_graph:
            exits = self.room_graph[current_room]
            if action in exits:
                dest_room = exits[action]
                dest_idx = self.room_index.get(dest_room, current_idx)
                return ZA(
                    width=zA.width,
                    height=zA.height,
                    agent_pos=(dest_idx, 0),
                    goal_pos=zA.goal_pos,
                    obstacles=zA.obstacles,
                    hint=None,
                )

        # Invalid exit or unknown room -> stay in place
        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=zA.agent_pos,
            goal_pos=zA.goal_pos,
            obstacles=zA.obstacles,
            hint=None,
        )

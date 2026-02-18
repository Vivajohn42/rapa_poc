"""DoorKeyAgentB: Deterministic forward model for DoorKey.

Predicts next state given rotation-based actions. Uses ZA.direction
to track agent orientation. Handles wall collisions and door blocking.

Note: pickup/toggle do not change position or direction — only the
environment handles their side effects (key acquisition, door state).
B's role is to predict agent_pos + direction after movement actions.
"""
from __future__ import annotations

from typing import Optional, Tuple

from kernel.interfaces import StreamB
from state.schema import ZA

ACTIONS = ("turn_left", "turn_right", "forward", "pickup", "toggle")

# Direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class DoorKeyAgentB(StreamB):
    """Rotation-aware forward model for DoorKey navigation."""

    def __init__(
        self,
        door_pos: Optional[Tuple[int, int]] = None,
        door_open: bool = False,
    ):
        self._door_pos = door_pos
        self._door_open = door_open

    def update_door_state(
        self,
        door_pos: Optional[Tuple[int, int]],
        door_open: bool,
    ) -> None:
        """Called by adapter to keep B's door model current."""
        self._door_pos = door_pos
        self._door_open = door_open

    def predict_next(self, zA: ZA, action: str) -> ZA:
        if action not in ACTIONS:
            return zA.model_copy()

        direction = zA.direction if zA.direction is not None else 0
        x, y = zA.agent_pos
        obstacle_set = set(zA.obstacles)

        new_dir = direction
        new_x, new_y = x, y

        if action == "turn_left":
            new_dir = (direction - 1) % 4
        elif action == "turn_right":
            new_dir = (direction + 1) % 4
        elif action == "forward":
            dx, dy = DIR_VEC[direction]
            nx, ny = x + dx, y + dy
            # Boundary check
            if 0 <= nx < zA.width and 0 <= ny < zA.height:
                # Wall check (obstacles includes walls + locked door)
                if (nx, ny) not in obstacle_set:
                    new_x, new_y = nx, ny
                # Door passable if open
                elif (nx, ny) == self._door_pos and self._door_open:
                    new_x, new_y = nx, ny
        # pickup/toggle: no position or direction change

        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=(new_x, new_y),
            goal_pos=zA.goal_pos,
            obstacles=zA.obstacles,
            hint=None,
            direction=new_dir,
        )

"""MiniGrid DoorKey wrapper for rapa_mvp.

Wraps gymnasium MiniGrid-DoorKey-{N}x{N}-v0 into a DoorKeyState
dataclass matching the rapa_mvp observation pattern.

Key adaptations:
  - Rotation-based actions mapped to/from string names
  - Privileged state access (env.unwrapped.*) for positions
  - Phase tracking: FIND_KEY -> OPEN_DOOR -> REACH_GOAL
  - Belief map accumulated from privileged grid access
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Must import minigrid before gym.make for env registration
import minigrid  # noqa: F401

# ── Action mapping ──────────────────────────────────────────────
ACTION_MAP: Dict[str, int] = {
    "turn_left":  0,
    "turn_right": 1,
    "forward":    2,
    "pickup":     3,
    "toggle":     5,
}
ACTION_NAMES: List[str] = list(ACTION_MAP.keys())

# ── MiniGrid constants ──────────────────────────────────────────
OBJ_WALL = 2
OBJ_DOOR = 4
OBJ_KEY  = 5
OBJ_GOAL = 8

DOOR_OPEN   = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2

# Direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


@dataclass
class DoorKeyState:
    """Observation returned by DoorKeyEnv, analogous to GridState."""
    width: int
    height: int
    agent_pos: Tuple[int, int]
    agent_dir: int                              # 0=right, 1=down, 2=left, 3=up
    goal_pos: Tuple[int, int]                   # (-1,-1) until discovered
    obstacles: List[Tuple[int, int]]            # walls + locked door
    carrying_key: bool
    key_pos: Optional[Tuple[int, int]]          # None if picked up or not yet seen
    door_pos: Optional[Tuple[int, int]]         # None if not yet seen
    door_state: Optional[int]                   # 0=open, 1=closed, 2=locked
    hint: Optional[str]                         # phase transition / discovery hint
    phase: str                                  # FIND_KEY, OPEN_DOOR, REACH_GOAL


class DoorKeyEnv:
    """Wrapper around MiniGrid DoorKey with privileged state access.

    Presents the same interface pattern as GridWorld:
      reset()  -> DoorKeyState
      step(action_str) -> (DoorKeyState, reward, done)
    """

    def __init__(
        self,
        size: int = 6,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        self._seed = seed
        self._max_steps = max_steps or (10 * size * size)
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        self._env = gym.make(env_id, max_steps=self._max_steps)

        # Discovered state (persists across steps within episode)
        self._known_walls: List[Tuple[int, int]] = []
        self._key_pos: Optional[Tuple[int, int]] = None
        self._door_pos: Optional[Tuple[int, int]] = None
        self._door_state: Optional[int] = None
        self._goal_pos: Tuple[int, int] = (-1, -1)
        self._phase: str = "FIND_KEY"
        self._prev_phase: str = "FIND_KEY"
        self._hint_pending: Optional[str] = None
        self.t: int = 0

    def reset(self) -> DoorKeyState:
        self._env.reset(seed=self._seed)
        self.t = 0
        self._known_walls.clear()
        self._key_pos = None
        self._door_pos = None
        self._door_state = None
        self._goal_pos = (-1, -1)
        self._phase = "FIND_KEY"
        self._prev_phase = "FIND_KEY"
        self._hint_pending = None
        self._scan_grid()
        self._update_phase()
        return self._observe()

    def step(self, action: str) -> Tuple[DoorKeyState, float, bool]:
        if action not in ACTION_MAP:
            return self._observe(), -0.01, False

        action_int = ACTION_MAP[action]
        _obs, reward, terminated, truncated, _info = self._env.step(action_int)
        self.t += 1
        done = terminated or truncated

        self._scan_grid()
        self._update_phase()

        # Generate hint on phase transition
        if self._phase != self._prev_phase:
            self._hint_pending = f"phase:{self._phase.lower()}"
            self._prev_phase = self._phase

        state = self._observe()
        return state, float(reward), done

    def _observe(self) -> DoorKeyState:
        uw = self._env.unwrapped
        carrying = uw.carrying is not None

        hint = self._hint_pending
        self._hint_pending = None

        # Obstacles = walls + locked/closed door
        obstacles = list(self._known_walls)
        if (self._door_pos is not None
                and self._door_state is not None
                and self._door_state != DOOR_OPEN):
            if self._door_pos not in obstacles:
                obstacles.append(self._door_pos)

        return DoorKeyState(
            width=self.size,
            height=self.size,
            agent_pos=(int(uw.agent_pos[0]), int(uw.agent_pos[1])),
            agent_dir=int(uw.agent_dir),
            goal_pos=self._goal_pos,
            obstacles=obstacles,
            carrying_key=carrying,
            key_pos=self._key_pos if not carrying else None,
            door_pos=self._door_pos,
            door_state=self._door_state,
            hint=hint,
            phase=self._phase,
        )

    def _scan_grid(self) -> None:
        """Scan full grid via privileged access for walls, key, door, goal."""
        uw = self._env.unwrapped
        grid = uw.grid

        self._known_walls.clear()
        for x in range(self.size):
            for y in range(self.size):
                cell = grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "wall":
                    self._known_walls.append((x, y))
                elif cell.type == "key":
                    self._key_pos = (x, y)
                elif cell.type == "door":
                    self._door_pos = (x, y)
                    # door.is_open, door.is_locked
                    if cell.is_open:
                        self._door_state = DOOR_OPEN
                    elif cell.is_locked:
                        self._door_state = DOOR_LOCKED
                    else:
                        self._door_state = DOOR_CLOSED
                elif cell.type == "goal":
                    self._goal_pos = (x, y)

        # Key picked up -> no longer on grid
        if uw.carrying is not None and uw.carrying.type == "key":
            self._key_pos = None

    def _update_phase(self) -> None:
        """Determine current task phase from privileged state."""
        uw = self._env.unwrapped
        carrying = uw.carrying is not None

        if self._door_state == DOOR_OPEN:
            self._phase = "REACH_GOAL"
        elif carrying:
            self._phase = "OPEN_DOOR"
        else:
            self._phase = "FIND_KEY"

    @property
    def available_actions(self) -> List[str]:
        return list(ACTION_NAMES)

    def close(self):
        self._env.close()

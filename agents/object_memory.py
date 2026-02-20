"""ObjectMemory: A-Level world model built from 7×7 ego-view.

Replaces the privileged _scan_grid() access in DoorKeyEnv with fair
ego-view parsing.  Only sees what the agent sees through MiniGrid's
7×7 partial observation window.

Proprioceptive state (agent_pos, agent_dir, carrying) is read from
env.unwrapped — these are the agent's own internal state, not the
full grid.  The grid itself (env.unwrapped.grid) is NEVER accessed.

Tracks:
  - known_walls / known_empty: confirmed cell types
  - visited: cells the agent has physically occupied
  - key_pos, door_pos, goal_pos: discovered object locations
  - carrying_key, door_state: inferred agent + world state
  - frontier: unvisited cells adjacent to known-empty (exploration target)
  - known_obstacles: walls + locked/closed door (for BFS)
"""
from __future__ import annotations

from typing import Optional, Set, Tuple

# MiniGrid object-type codes (from gen_obs image channel 0)
OBJ_UNSEEN = 0
OBJ_EMPTY  = 1
OBJ_WALL   = 2
OBJ_FLOOR  = 3
OBJ_DOOR   = 4
OBJ_KEY    = 5
OBJ_BALL   = 6
OBJ_BOX    = 7
OBJ_GOAL   = 8
OBJ_LAVA   = 9
OBJ_AGENT  = 10

# Door state codes (image channel 2 for door objects)
DOOR_OPEN   = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2

# Direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class ObjectMemory:
    """A-Level: builds world model from 7×7 ego-view observations.

    No privileged grid access.  All knowledge comes from what the agent
    can see in its forward-facing 7×7 partial observation window.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.known_walls: Set[Tuple[int, int]] = set()
        self.known_empty: Set[Tuple[int, int]] = set()
        self.visited: Set[Tuple[int, int]] = set()
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.door_state: Optional[int] = None   # 0=open, 1=closed, 2=locked
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self._key_picked_up: bool = False  # sticky: once True, stays True

    def reset(self) -> None:
        """Reset for a new episode."""
        self.known_walls.clear()
        self.known_empty.clear()
        self.visited.clear()
        self.key_pos = None
        self.door_pos = None
        self.door_state = None
        self.goal_pos = None
        self.carrying_key = False
        self._key_picked_up = False

    def update(self, env_unwrapped) -> None:
        """Scan the 7×7 ego-view and update the known world model.

        Uses env.unwrapped.gen_obs() for the ego-view image (7×7×3),
        plus agent_pos/agent_dir from unwrapped (proprioceptive state).
        Does NOT use env.unwrapped.grid (that would be privileged).
        """
        obs = env_unwrapped.gen_obs()
        image = obs["image"]  # shape (7, 7, 3)
        agent_dir = int(env_unwrapped.agent_dir)
        agent_pos = (int(env_unwrapped.agent_pos[0]),
                     int(env_unwrapped.agent_pos[1]))

        self.visited.add(agent_pos)
        self.known_empty.add(agent_pos)

        # Proprioceptive carrying detection
        self.carrying_key = env_unwrapped.carrying is not None
        if self.carrying_key:
            self._key_picked_up = True
            self.key_pos = None  # key no longer on grid

        view_size = 7
        for ei in range(view_size):
            for ej in range(view_size):
                obj_type = int(image[ei, ej, 0])
                obj_state = int(image[ei, ej, 2])

                if obj_type == OBJ_UNSEEN:
                    continue

                world_pos = self._ego_to_world(ei, ej, agent_pos, agent_dir)
                if world_pos is None:
                    continue

                wx, wy = world_pos
                if not (0 <= wx < self.grid_size
                        and 0 <= wy < self.grid_size):
                    continue

                if obj_type == OBJ_WALL:
                    self.known_walls.add(world_pos)
                    self.known_empty.discard(world_pos)
                elif obj_type in (OBJ_EMPTY, OBJ_FLOOR):
                    self.known_empty.add(world_pos)
                elif obj_type == OBJ_KEY:
                    if not self._key_picked_up:
                        self.key_pos = world_pos
                    self.known_empty.add(world_pos)
                elif obj_type == OBJ_DOOR:
                    self.door_pos = world_pos
                    self.door_state = obj_state
                    # Open door is passable, keep in known_empty
                    if obj_state == DOOR_OPEN:
                        self.known_empty.add(world_pos)
                    else:
                        self.known_empty.discard(world_pos)
                elif obj_type == OBJ_GOAL:
                    self.goal_pos = world_pos
                    self.known_empty.add(world_pos)
                elif obj_type == OBJ_AGENT:
                    # Agent's own cell — already handled
                    self.known_empty.add(world_pos)

    @staticmethod
    def _ego_to_world(
        ego_i: int,
        ego_j: int,
        agent_pos: Tuple[int, int],
        agent_dir: int,
    ) -> Optional[Tuple[int, int]]:
        """Convert ego-view (i, j) to world (x, y).

        MiniGrid's gen_obs_grid() does:
          1. Extract a 7×7 sub-grid aligned with agent's view
          2. Rotate left (agent_dir + 1) times
          3. Agent ends up at ego position (view_size//2, view_size-1)
             i.e. column 3, row 6 — facing "up" in ego space

        Inverse transform:
          rel_i = ego_i - 3      (horizontal offset from agent in ego)
          rel_j = 6 - ego_j      (forward distance in ego; row 6 = agent)
          Then rotate by agent_dir to get world offsets.
        """
        rel_i = ego_i - 3
        rel_j = 6 - ego_j

        if agent_dir == 0:    # facing right
            wx, wy = rel_j, rel_i
        elif agent_dir == 1:  # facing down
            wx, wy = -rel_i, rel_j
        elif agent_dir == 2:  # facing left
            wx, wy = -rel_j, -rel_i
        elif agent_dir == 3:  # facing up
            wx, wy = rel_i, -rel_j
        else:
            return None

        return (agent_pos[0] + wx, agent_pos[1] + wy)

    @property
    def frontier(self) -> Set[Tuple[int, int]]:
        """Unvisited cells adjacent to known-empty cells.

        These are cells the agent hasn't visited but are reachable
        (adjacent to at least one known passable cell, and not known
        to be walls).
        """
        result: Set[Tuple[int, int]] = set()
        for x, y in self.known_empty:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                pos = (nx, ny)
                if (0 <= nx < self.grid_size
                        and 0 <= ny < self.grid_size
                        and pos not in self.visited
                        and pos not in self.known_walls):
                    result.add(pos)
        return result

    @property
    def known_obstacles(self) -> Set[Tuple[int, int]]:
        """Walls + locked/closed door — the obstacle set for BFS.

        Unknown cells are NOT in this set (optimistic BFS: unknown
        cells are assumed passable, with cost penalty handled by
        AutonomousDoorKeyAgentC's weighted BFS).
        """
        obs = set(self.known_walls)
        if (self.door_pos is not None
                and self.door_state is not None
                and self.door_state != DOOR_OPEN):
            obs.add(self.door_pos)
        return obs

    @property
    def door_open(self) -> bool:
        """True if the door has been observed as open."""
        return self.door_state == DOOR_OPEN

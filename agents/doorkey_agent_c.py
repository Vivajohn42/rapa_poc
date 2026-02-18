"""DoorKeyAgentC: Multi-phase goal-directed action scoring for DoorKey.

Uses integrated BFS-distance + turn-cost scoring for rotation-based
navigation. BFS accounts for walls (unlike Manhattan), and turn cost
accounts for the rotation overhead.

Score = 1.0 / (bfs_distance + turns_needed + 1)

Three phases with different targets:
  FIND_KEY:   navigate toward key_pos, pickup when adjacent
  OPEN_DOOR:  navigate toward door_pos, toggle when adjacent + carrying key
  REACH_GOAL: navigate toward goal_pos

pickup/toggle score 3.0 when appropriate (>> any navigation score).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from kernel.interfaces import StreamC
from state.schema import ZA

ACTIONS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]
UNREACHABLE = 9999

# Direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class _DoorKeyGoalProxy:
    """Proxy satisfying GoalTarget protocol for kernel integration."""

    def __init__(self):
        self._target: Optional[Tuple[int, int]] = None

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        return self._target

    @target.setter
    def target(self, value: Tuple[int, int]) -> None:
        self._target = value


def _bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """BFS shortest-path distance avoiding obstacles."""
    if start == goal:
        return 0
    if start in obstacles or goal in obstacles:
        return UNREACHABLE

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if pos == goal:
                return dist + 1
            if (0 <= nx < width and 0 <= ny < height
                    and pos not in obstacles and pos not in visited):
                visited.add(pos)
                queue.append((pos, dist + 1))

    return UNREACHABLE


def _bfs_next_step(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Return the first cell on the BFS shortest path from start to goal.

    Returns None if no path exists or start == goal.
    """
    if start == goal:
        return None
    if start in obstacles or goal in obstacles:
        return None

    visited = {start}
    # Store (pos, first_step) where first_step is the first cell in path
    queue: deque[Tuple[Tuple[int, int], Tuple[int, int]]] = deque()

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = start[0] + dx, start[1] + dy
        pos = (nx, ny)
        if pos == goal:
            return pos
        if (0 <= nx < width and 0 <= ny < height
                and pos not in obstacles and pos not in visited):
            visited.add(pos)
            queue.append((pos, pos))  # first_step = this neighbor

    while queue:
        (x, y), first_step = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if pos == goal:
                return first_step
            if (0 <= nx < width and 0 <= ny < height
                    and pos not in obstacles and pos not in visited):
                visited.add(pos)
                queue.append((pos, first_step))

    return None


class DoorKeyAgentC(StreamC):
    """Multi-phase action scorer with BFS + turn-cost scoring."""

    def __init__(self, goal_mode: str = "seek"):
        self.goal_mode = goal_mode
        self._goal = _DoorKeyGoalProxy()

        # Phase-specific state (updated by adapter's inject_obs_metadata)
        self.phase: str = "FIND_KEY"
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self.door_open: bool = False

        # Visit counts for exploration when no target
        self._visit_counts: Dict[Tuple[int, int], int] = {}

    @property
    def goal(self) -> _DoorKeyGoalProxy:
        return self._goal

    @staticmethod
    def _dir_to_target(pos: Tuple[int, int],
                       next_cell: Tuple[int, int]) -> int:
        """Direction from pos to an adjacent cell."""
        dx = next_cell[0] - pos[0]
        dy = next_cell[1] - pos[1]
        # Map delta to direction: (1,0)=0, (0,1)=1, (-1,0)=2, (0,-1)=3
        for i, (ddx, ddy) in enumerate(DIR_VEC):
            if dx == ddx and dy == ddy:
                return i
        return 0  # fallback

    @staticmethod
    def _min_turns(current_dir: int, desired_dir: int) -> int:
        """Minimum turns between two directions (0, 1, or 2)."""
        diff = abs(desired_dir - current_dir) % 4
        return min(diff, 4 - diff)

    def _effective_distance(
        self,
        pos: Tuple[int, int],
        direction: int,
        target: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        width: int,
        height: int,
    ) -> float:
        """BFS distance + turns needed to face the BFS-optimal direction.

        Uses _bfs_next_step to determine which direction the agent should
        face, accounting for walls. This avoids the trap where Manhattan
        says "face right" but a wall blocks that path.
        """
        bfs = _bfs_distance(pos, target, obstacles, width, height)
        if bfs == 0:
            return 0.0
        if bfs >= UNREACHABLE:
            return float(UNREACHABLE)

        next_cell = _bfs_next_step(pos, target, obstacles, width, height)
        if next_cell is None:
            return float(UNREACHABLE)

        desired_dir = self._dir_to_target(pos, next_cell)
        turns = self._min_turns(direction, desired_dir)
        return float(bfs + turns)

    def _current_target(
        self,
        zA: ZA,
        memory: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Determine navigation target from D-provided memory or fallback.

        For OPEN_DOOR phase: the target is the door position itself.
        BFS scoring handles the obstacle issue by temporarily removing
        the door from obstacles when computing distance to it (since the
        agent needs to reach the cell adjacent to the door, not through it).
        """
        if memory and "target" in memory:
            return tuple(memory["target"])
        if self._goal.target is not None:
            return self._goal.target
        return None

    def _facing_pos(self, zA: ZA) -> Optional[Tuple[int, int]]:
        """Return the cell the agent is facing."""
        d = zA.direction if zA.direction is not None else 0
        dx, dy = DIR_VEC[d]
        fx, fy = zA.agent_pos[0] + dx, zA.agent_pos[1] + dy
        if 0 <= fx < zA.width and 0 <= fy < zA.height:
            return (fx, fy)
        return None

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        target = self._current_target(zA, memory)
        facing = self._facing_pos(zA)
        direction = zA.direction if zA.direction is not None else 0
        obstacle_set = set(zA.obstacles)
        scored: List[Tuple[str, float]] = []

        # Track visits for exploration
        self._visit_counts[zA.agent_pos] = (
            self._visit_counts.get(zA.agent_pos, 0) + 1
        )

        for action in ACTIONS:
            score = 0.0
            zA_next = predict_next_fn(zA, action)
            next_dir = (zA_next.direction
                        if zA_next.direction is not None else 0)

            if action in ("turn_left", "turn_right", "forward"):
                if target is not None:
                    # Remove target from obstacles for BFS if needed
                    # (e.g. door is an obstacle but is our navigation target)
                    nav_obstacles = obstacle_set
                    if target in obstacle_set:
                        nav_obstacles = obstacle_set - {target}
                    d_now = self._effective_distance(
                        zA.agent_pos, direction, target,
                        nav_obstacles, zA.width, zA.height)
                    d_next = self._effective_distance(
                        zA_next.agent_pos, next_dir, target,
                        nav_obstacles, zA.width, zA.height)
                    # Score: inverse of effective distance
                    score = 1.0 / (d_next + 1.0)
                    # Bonus for reducing effective distance
                    if d_next < d_now:
                        score += 0.5
                else:
                    # No target: exploration
                    if action == "forward":
                        if zA_next.agent_pos != zA.agent_pos:
                            visits = self._visit_counts.get(
                                zA_next.agent_pos, 0)
                            score = 1.0 / (visits + 1.0)
                        else:
                            score = -0.1
                    else:
                        score = 0.05

            elif action == "pickup":
                # pickup requires D-provided target to be set
                # (without D, agent has no knowledge of subgoal intent)
                if (target is not None
                        and self.phase == "FIND_KEY"
                        and not self.carrying_key
                        and self.key_pos is not None
                        and facing == self.key_pos):
                    score = 3.0
                else:
                    score = -1.0

            elif action == "toggle":
                # toggle requires D-provided target to be set
                if (target is not None
                        and self.phase == "OPEN_DOOR"
                        and self.carrying_key
                        and self.door_pos is not None
                        and facing == self.door_pos):
                    score = 3.0
                else:
                    score = -1.0

            scored.append((action, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0], scored

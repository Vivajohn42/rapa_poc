"""BFS expert oracle for DoorKey rotation-based navigation.

Provides label computation for training Neural DoorKey C.
For each (pos, dir, target) and navigation action, computes:

    label = effective_distance(now) - effective_distance(after_action)

Where effective_distance = bfs_dist + turns_to_face_bfs_optimal_direction.
Extracted from doorkey_agent_c.py as standalone functions (no ZA dependency).

Usage:
    from train.bfs_expert_doorkey import (
        bfs_distance_map, effective_distance, compute_next_state,
    )
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set, Tuple

UNREACHABLE = 9999

# Direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

NAV_ACTIONS = ("turn_left", "turn_right", "forward")


def bfs_distance(
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


def bfs_distance_map(
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Dict[Tuple[int, int], int]:
    """Compute BFS distance from every reachable cell to goal.

    Returns {pos: distance} for all reachable positions.
    Efficient: one BFS per goal per grid config.
    """
    if goal in obstacles:
        return {}

    dist_map = {goal: 0}
    queue = deque([(goal, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if (0 <= nx < width and 0 <= ny < height
                    and pos not in obstacles and pos not in dist_map):
                dist_map[pos] = dist + 1
                queue.append((pos, dist + 1))

    return dist_map


def bfs_next_step(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Return the first cell on the BFS shortest path from start to goal."""
    if start == goal:
        return None
    if start in obstacles or goal in obstacles:
        return None

    visited = {start}
    queue: deque[Tuple[Tuple[int, int], Tuple[int, int]]] = deque()

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = start[0] + dx, start[1] + dy
        pos = (nx, ny)
        if pos == goal:
            return pos
        if (0 <= nx < width and 0 <= ny < height
                and pos not in obstacles and pos not in visited):
            visited.add(pos)
            queue.append((pos, pos))

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


def dir_to_target(pos: Tuple[int, int], next_cell: Tuple[int, int]) -> int:
    """Direction from pos to an adjacent cell."""
    dx = next_cell[0] - pos[0]
    dy = next_cell[1] - pos[1]
    for i, (ddx, ddy) in enumerate(DIR_VEC):
        if dx == ddx and dy == ddy:
            return i
    return 0


def min_turns(current_dir: int, desired_dir: int) -> int:
    """Minimum turns between two directions (0, 1, or 2)."""
    diff = abs(desired_dir - current_dir) % 4
    return min(diff, 4 - diff)


def effective_distance(
    pos: Tuple[int, int],
    direction: int,
    target: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> float:
    """BFS distance + turns needed to face BFS-optimal direction.

    Matches doorkey_agent_c._effective_distance() exactly.
    """
    bfs = bfs_distance(pos, target, obstacles, width, height)
    if bfs == 0:
        return 0.0
    if bfs >= UNREACHABLE:
        return float(UNREACHABLE)

    next_cell = bfs_next_step(pos, target, obstacles, width, height)
    if next_cell is None:
        return float(UNREACHABLE)

    desired_dir = dir_to_target(pos, next_cell)
    turns = min_turns(direction, desired_dir)
    return float(bfs + turns)


def compute_next_state(
    pos: Tuple[int, int],
    direction: int,
    action: str,
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
    door_pos: Optional[Tuple[int, int]] = None,
    door_open: bool = False,
) -> Tuple[Tuple[int, int], int]:
    """Compute (new_pos, new_dir) after a navigation action.

    Mirrors DoorKeyAgentB.predict_next() logic but returns raw tuples.
    Only handles turn_left, turn_right, forward.
    """
    new_x, new_y = pos
    new_dir = direction

    if action == "turn_left":
        new_dir = (direction - 1) % 4
    elif action == "turn_right":
        new_dir = (direction + 1) % 4
    elif action == "forward":
        dx, dy = DIR_VEC[direction]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height:
            if (nx, ny) not in obstacles:
                new_x, new_y = nx, ny
            elif (nx, ny) == door_pos and door_open:
                new_x, new_y = nx, ny

    return (new_x, new_y), new_dir


def heuristic_distance(
    pos: Tuple[int, int],
    direction: int,
    target: Tuple[int, int],
) -> float:
    """Manhattan distance + turn cost approximation (no BFS).

    Used as a feature in the neural network (not as label).
    """
    manhattan = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
    if manhattan == 0:
        return 0.0
    # Approximate desired direction from displacement
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2
    else:
        desired_dir = 1 if dy > 0 else 3
    turns = min_turns(direction, desired_dir)
    return float(manhattan + turns)

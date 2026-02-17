"""BFS pathfinding oracle for training labels.

Computes the true shortest-path distance from any position to a goal,
accounting for obstacles.  On unweighted grids this is identical to A*.

The key training signal for Neural C:
    bfs_label = bfs_dist(current, goal) - bfs_dist(next, goal)

This differs from Manhattan when obstacles force detours.  The neural
network learns to replicate the BFS-optimal scoring where Manhattan fails.

Usage:
    from train.bfs_expert import bfs_distance, bfs_action_value
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple, Set, Optional

UNREACHABLE = 9999


def bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """BFS shortest-path distance from start to goal, avoiding obstacles.

    Returns UNREACHABLE (9999) if no path exists.
    """
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
                    and pos not in obstacles
                    and pos not in visited):
                visited.add(pos)
                queue.append((pos, dist + 1))

    return UNREACHABLE


def bfs_action_value(
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
    action: str,
) -> float:
    """Compute the BFS-based action value for a single action.

    Returns:
        bfs_dist(current, goal) - bfs_dist(next, goal)

    Positive = action moved closer to goal via shortest path.
    Zero     = no progress (or both unreachable).
    Negative = moved farther from goal.
    """
    # Compute next position after action
    x, y = agent_pos
    moves = {
        "up": (x, y - 1),
        "down": (x, y + 1),
        "left": (x - 1, y),
        "right": (x + 1, y),
    }

    nx, ny = moves.get(action, (x, y))

    # Boundary check
    if not (0 <= nx < width and 0 <= ny < height):
        nx, ny = x, y

    # Obstacle check
    if (nx, ny) in obstacles:
        nx, ny = x, y

    next_pos = (nx, ny)

    d_now = bfs_distance(agent_pos, goal_pos, obstacles, width, height)
    d_next = bfs_distance(next_pos, goal_pos, obstacles, width, height)

    # Both unreachable → 0
    if d_now >= UNREACHABLE and d_next >= UNREACHABLE:
        return 0.0

    return float(d_now - d_next)


def bfs_distance_map(
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> dict:
    """Compute BFS distance from every reachable cell to goal.

    Returns dict mapping (x, y) -> distance.  Unreachable cells omitted.
    Much more efficient than calling bfs_distance() per cell.
    """
    if goal in obstacles:
        return {}

    dist_map = {goal: 0}
    queue = deque([goal])

    while queue:
        x, y = queue.popleft()
        d = dist_map[(x, y)]

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)

            if (0 <= nx < width and 0 <= ny < height
                    and pos not in obstacles
                    and pos not in dist_map):
                dist_map[pos] = d + 1
                queue.append(pos)

    return dist_map

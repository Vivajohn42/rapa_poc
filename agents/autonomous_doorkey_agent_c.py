"""AutonomousDoorKeyAgentC: Frontier exploration + cost-weighted BFS on known grid.

Two operating modes:
  1. Target from D: Navigate using cost-weighted BFS on the known grid
     (only cells seen through ego-view, not privileged grid access)
  2. No target: Frontier-based exploration (curiosity-driven)

Recovery mode: When BFS reports target as UNREACHABLE (unseen walls block
the path), switches to frontier exploration for RECOVERY_STEPS ticks to
discover new cells that reconnect the map.

Key differences from DoorKeyAgentC:
  - BFS uses ObjectMemory.known_obstacles (only seen walls)
  - Cost-weighted BFS: known_empty=1, unknown=3 (reduces wall-bumping)
  - Frontier-based exploration instead of simple visit-count
  - Interaction heuristic: pickup/toggle purely positional (no D-gate)
  - Dead-end marking: cells with all neighbors known+visited get reduced score
  - Turn-cost malus: small penalty for turns to reduce jitter
  - Recovery mode: frontier fallback when BFS can't reach target
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from kernel.interfaces import StreamC
from state.schema import ZA
from agents.doorkey_agent_c import (
    _DoorKeyGoalProxy,
    DIR_VEC,
    UNREACHABLE,
)

ACTIONS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]

# Cost weights for BFS
COST_KNOWN_EMPTY = 1
COST_UNKNOWN = 3  # Penalizes traversing unseen cells (but still possible)

# Small malus for turns to reduce jitter
TURN_MALUS = 0.05

# Recovery mode: frontier exploration steps when BFS reports UNREACHABLE
RECOVERY_STEPS = 8


def _weighted_bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    known_empty: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """BFS with cost weighting: known_empty=1, unknown=COST_UNKNOWN.

    Uses Dijkstra-like expansion with a priority queue approximated by
    a simple BFS with cost buckets (since costs are small integers).
    Returns the weighted distance; UNREACHABLE if no path exists.
    """
    if start == goal:
        return 0
    if start in obstacles or goal in obstacles:
        return UNREACHABLE

    # Dijkstra with bounded integer costs
    import heapq
    dist = {start: 0}
    heap = [(0, start)]

    while heap:
        d, (x, y) = heapq.heappop(heap)
        if (x, y) == goal:
            return d
        if d > dist.get((x, y), UNREACHABLE):
            continue

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if pos in obstacles:
                continue

            # Cost depends on whether we've seen this cell
            step_cost = (COST_KNOWN_EMPTY
                         if pos in known_empty
                         else COST_UNKNOWN)
            new_dist = d + step_cost

            if new_dist < dist.get(pos, UNREACHABLE):
                dist[pos] = new_dist
                heapq.heappush(heap, (new_dist, pos))

    return UNREACHABLE


def _weighted_bfs_next_step(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    known_empty: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Return the first cell on the cost-weighted shortest path."""
    if start == goal:
        return None
    if start in obstacles or goal in obstacles:
        return None

    import heapq
    dist = {start: 0}
    parent = {start: None}
    heap = [(0, start)]

    while heap:
        d, (x, y) = heapq.heappop(heap)
        if (x, y) == goal:
            # Backtrack to find first step
            cell = goal
            while parent.get(cell) != start and parent.get(cell) is not None:
                cell = parent[cell]
            return cell
        if d > dist.get((x, y), UNREACHABLE):
            continue

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if pos in obstacles:
                continue

            step_cost = (COST_KNOWN_EMPTY
                         if pos in known_empty
                         else COST_UNKNOWN)
            new_dist = d + step_cost

            if new_dist < dist.get(pos, UNREACHABLE):
                dist[pos] = new_dist
                parent[pos] = (x, y)
                heapq.heappush(heap, (new_dist, pos))

    return None


class AutonomousDoorKeyAgentC(StreamC):
    """C-Level: cost-weighted BFS on known grid + frontier exploration.

    Interaction heuristic: pickup/toggle are purely positional — no
    D-essentiality gate (no `target is not None` check).
    """

    def __init__(self, goal_mode: str = "seek"):
        self.goal_mode = goal_mode
        self._goal = _DoorKeyGoalProxy()

        # Phase-specific state (set by runner each tick)
        self.phase: str = "FIND_KEY"
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self.door_open: bool = False

        # Visit counts for fallback exploration
        self._visit_counts: Dict[Tuple[int, int], int] = {}

        # Recovery mode: countdown when BFS can't reach target
        self._recovery_steps: int = 0

        # ObjectMemory link
        self._object_memory = None

    def set_object_memory(self, obj_mem) -> None:
        """Connect to ObjectMemory for known_obstacles and frontier."""
        self._object_memory = obj_mem

    @property
    def goal(self) -> _DoorKeyGoalProxy:
        return self._goal

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Score actions: BFS-guided if target set, frontier if not."""
        target = self._current_target(memory)
        facing = self._facing_pos(zA)
        direction = zA.direction if zA.direction is not None else 0

        om = self._object_memory
        if om is not None:
            obstacle_set = om.known_obstacles
            known_empty = om.known_empty
        else:
            obstacle_set = set(zA.obstacles)
            known_empty = set()  # no knowledge

        # Track visits
        self._visit_counts[zA.agent_pos] = (
            self._visit_counts.get(zA.agent_pos, 0) + 1
        )

        # Recovery mode: check if target is unreachable on known map
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            use_frontier = True
        elif target is not None:
            nav_obs = (obstacle_set - {target}
                       if target in obstacle_set else obstacle_set)
            d = _weighted_bfs_distance(
                zA.agent_pos, target, nav_obs, known_empty,
                zA.width, zA.height)
            if d >= UNREACHABLE:
                self._recovery_steps = RECOVERY_STEPS - 1
                use_frontier = True
            else:
                use_frontier = False
        else:
            use_frontier = True

        scored: List[Tuple[str, float]] = []

        for action in ACTIONS:
            score = 0.0
            zA_next = predict_next_fn(zA, action)
            next_dir = (zA_next.direction
                        if zA_next.direction is not None else 0)

            if action in ("turn_left", "turn_right", "forward"):
                if not use_frontier and target is not None:
                    score = self._score_navigation(
                        zA, zA_next, direction, next_dir,
                        target, obstacle_set, known_empty)
                else:
                    score = self._score_frontier_exploration(
                        zA, zA_next, action, direction, next_dir,
                        obstacle_set, known_empty)

                # Turn-cost malus to reduce jitter
                if action in ("turn_left", "turn_right"):
                    score -= TURN_MALUS

            elif action == "pickup":
                # Heuristic: pickup when facing key and not carrying
                if (not self.carrying_key
                        and self.key_pos is not None
                        and facing == self.key_pos):
                    score = 3.0
                else:
                    score = -1.0

            elif action == "toggle":
                # Heuristic: toggle when facing door, carrying key,
                # and door is NOT already open (prevents toggle loop)
                if (self.carrying_key
                        and not self.door_open
                        and self.door_pos is not None
                        and facing == self.door_pos):
                    score = 3.0
                else:
                    score = -1.0

            scored.append((action, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0], scored

    # ---- Navigation scoring (BFS-guided toward target) ----

    def _score_navigation(
        self,
        zA: ZA,
        zA_next: ZA,
        direction: int,
        next_dir: int,
        target: Tuple[int, int],
        obstacle_set: Set[Tuple[int, int]],
        known_empty: Set[Tuple[int, int]],
    ) -> float:
        """BFS-distance scoring toward target on known grid."""
        # Remove target from obstacles if needed (e.g. locked door is target)
        nav_obstacles = obstacle_set
        if target in obstacle_set:
            nav_obstacles = obstacle_set - {target}

        d_now = self._effective_distance(
            zA.agent_pos, direction, target,
            nav_obstacles, known_empty, zA.width, zA.height)
        d_next = self._effective_distance(
            zA_next.agent_pos, next_dir, target,
            nav_obstacles, known_empty, zA.width, zA.height)

        score = 1.0 / (d_next + 1.0)
        if d_next < d_now:
            score += 0.5
        return score

    def _effective_distance(
        self,
        pos: Tuple[int, int],
        direction: int,
        target: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        known_empty: Set[Tuple[int, int]],
        width: int,
        height: int,
    ) -> float:
        """Weighted BFS distance + turn cost to face optimal direction."""
        bfs = _weighted_bfs_distance(
            pos, target, obstacles, known_empty, width, height)
        if bfs == 0:
            return 0.0
        if bfs >= UNREACHABLE:
            return float(UNREACHABLE)

        next_cell = _weighted_bfs_next_step(
            pos, target, obstacles, known_empty, width, height)
        if next_cell is None:
            return float(UNREACHABLE)

        desired_dir = self._dir_to_target(pos, next_cell)
        turns = self._min_turns(direction, desired_dir)
        return float(bfs + turns)

    # ---- Frontier exploration scoring ----

    def _score_frontier_exploration(
        self,
        zA: ZA,
        zA_next: ZA,
        action: str,
        direction: int,
        next_dir: int,
        obstacle_set: Set[Tuple[int, int]],
        known_empty: Set[Tuple[int, int]],
    ) -> float:
        """Score based on proximity to frontier cells (curiosity)."""
        om = self._object_memory

        if om is None or not om.frontier:
            return self._simple_explore_score(zA, zA_next, action)

        frontier = om.frontier

        if action == "forward":
            if zA_next.agent_pos == zA.agent_pos:
                return -0.1  # wall hit

            # Distance from next position to nearest frontier cell
            min_dist = UNREACHABLE
            for f_cell in frontier:
                d = _weighted_bfs_distance(
                    zA_next.agent_pos, f_cell,
                    obstacle_set, known_empty,
                    zA.width, zA.height)
                if d < min_dist:
                    min_dist = d

            if min_dist >= UNREACHABLE:
                visits = self._visit_counts.get(zA_next.agent_pos, 0)
                return 1.0 / (visits + 1.0)

            score = 1.0 / (min_dist + 1.0)

            # Bonus for visiting a new cell
            if zA_next.agent_pos not in om.visited:
                score += 0.3

            # Dead-end penalty: if next cell has all neighbors known+visited
            if self._is_dead_end(zA_next.agent_pos, om, zA.width, zA.height):
                score -= 0.2

            return score

        else:
            # turn_left / turn_right: score based on facing toward frontier
            nearest = self._nearest_frontier(
                zA.agent_pos, obstacle_set, known_empty,
                zA.width, zA.height)
            if nearest is None:
                return 0.05

            next_cell = _weighted_bfs_next_step(
                zA.agent_pos, nearest,
                obstacle_set, known_empty,
                zA.width, zA.height)
            if next_cell is None:
                return 0.05

            desired_dir = self._dir_to_target(zA.agent_pos, next_cell)
            turns_remaining = self._min_turns(next_dir, desired_dir)

            if turns_remaining == 0:
                return 0.7  # facing toward frontier after this turn
            elif turns_remaining == 1:
                return 0.3
            return 0.05

    def _nearest_frontier(
        self,
        pos: Tuple[int, int],
        obstacle_set: Set[Tuple[int, int]],
        known_empty: Set[Tuple[int, int]],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        """Find the closest frontier cell by weighted BFS distance."""
        if self._object_memory is None:
            return None
        frontier = self._object_memory.frontier
        if not frontier:
            return None

        best = None
        best_dist = UNREACHABLE
        for f in frontier:
            d = _weighted_bfs_distance(
                pos, f, obstacle_set, known_empty, width, height)
            if d < best_dist:
                best_dist = d
                best = f
        return best

    @staticmethod
    def _is_dead_end(
        pos: Tuple[int, int],
        om,
        width: int,
        height: int,
    ) -> bool:
        """True if all neighbors are known and visited (boring cell)."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbor = (nx, ny)
            if neighbor in om.known_walls:
                continue
            if neighbor not in om.visited:
                return False  # unvisited passable neighbor exists
        return True

    def _simple_explore_score(
        self,
        zA: ZA,
        zA_next: ZA,
        action: str,
    ) -> float:
        """Fallback visit-count exploration (like DoorKeyAgentC)."""
        if action == "forward":
            if zA_next.agent_pos != zA.agent_pos:
                visits = self._visit_counts.get(zA_next.agent_pos, 0)
                return 1.0 / (visits + 1.0)
            return -0.1
        return 0.05

    # ---- Helpers ----

    def _current_target(
        self,
        memory: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Get target from memory (set by D's deconstruction)."""
        if memory and "target" in memory and memory["target"] is not None:
            return tuple(memory["target"])
        if self._goal.target is not None:
            return self._goal.target
        return None

    def _facing_pos(self, zA: ZA) -> Optional[Tuple[int, int]]:
        """Cell the agent is facing."""
        d = zA.direction if zA.direction is not None else 0
        dx, dy = DIR_VEC[d]
        fx, fy = zA.agent_pos[0] + dx, zA.agent_pos[1] + dy
        if 0 <= fx < zA.width and 0 <= fy < zA.height:
            return (fx, fy)
        return None

    @staticmethod
    def _dir_to_target(
        pos: Tuple[int, int],
        next_cell: Tuple[int, int],
    ) -> int:
        """Direction from pos to an adjacent cell."""
        dx = next_cell[0] - pos[0]
        dy = next_cell[1] - pos[1]
        for i, (ddx, ddy) in enumerate(DIR_VEC):
            if dx == ddx and dy == ddy:
                return i
        return 0

    @staticmethod
    def _min_turns(current_dir: int, desired_dir: int) -> int:
        """Minimum turns between two directions (0, 1, or 2)."""
        diff = abs(desired_dir - current_dir) % 4
        return min(diff, 4 - diff)

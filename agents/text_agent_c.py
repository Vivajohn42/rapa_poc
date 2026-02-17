"""TextAgentC: Valence/goal scoring for TextWorld.

Uses BFS graph distance instead of Manhattan distance.
Without a target from D, all exits score 0.0 -> decision_delta = 0 -> kernel activates D.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from state.schema import ZA
from kernel.interfaces import StreamC


class _TextGoalProxy:
    """Proxy object so kernel can write agent_c.goal.target = (idx, 0)."""

    def __init__(self, agent: "TextAgentC"):
        self._agent = agent
        self._target: Optional[Tuple[int, int]] = None

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        return self._target

    @target.setter
    def target(self, value: Tuple[int, int]):
        self._target = value
        # Map pseudo-position back to room_id for TextAgentC
        room = self._agent.index_to_room.get(value[0])
        self._agent.target_room = room


class TextAgentC(StreamC):
    """Graph-distance-based action scorer for TextWorld."""

    def __init__(
        self,
        room_graph: Dict[str, Dict[str, str]],
        room_index: Dict[str, int],
        index_to_room: Dict[int, str],
        goal_mode: str = "seek",
        anti_stay_penalty: float = 0.25,
    ):
        self.room_graph = room_graph
        self.room_index = room_index
        self.index_to_room = index_to_room
        self.goal_mode = goal_mode
        self.anti_stay_penalty = anti_stay_penalty
        self.target_room: Optional[str] = None  # set by deconstruction
        self._goal = _TextGoalProxy(self)       # kernel-compatible proxy

        # Precompute BFS distances between all room pairs
        self._distances = self._compute_all_distances()

    @property
    def goal(self) -> _TextGoalProxy:
        return self._goal

    def _compute_all_distances(self) -> Dict[str, Dict[str, int]]:
        """BFS all-pairs shortest path on the room graph."""
        distances = {}
        for start in self.room_graph:
            dist = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for dest in self.room_graph.get(current, {}).values():
                    if dest not in dist:
                        dist[dest] = dist[current] + 1
                        queue.append(dest)
            distances[start] = dist
        return distances

    def _bfs_distance(self, from_room: str, to_room: str) -> int:
        """Get BFS distance between two rooms. Returns large number if unreachable."""
        return self._distances.get(from_room, {}).get(to_room, 999)

    def _nearest_unvisited_dist(self, from_room: str, visited: set) -> int:
        """BFS distance from from_room to nearest unvisited room."""
        dists = self._distances.get(from_room, {})
        best = 999
        for room, dist in dists.items():
            if room not in visited and dist < best:
                best = dist
        return best

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn,
        memory: Optional[dict] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Score each available exit by graph distance to target room.

        If no target_room set (D hasn't synthesized yet), all exits score 0.0.
        This produces decision_delta = 0 -> kernel uncertainty trigger -> D activates.
        """
        current_idx = zA.agent_pos[0]
        current_room = self.index_to_room.get(current_idx, "")

        # Get available actions (exits from current room)
        exits = list(self.room_graph.get(current_room, {}).keys())
        if not exits:
            return "wait", [("wait", 0.0)]

        # Check if D has set a target via deconstruction
        target = None
        if memory and "target" in memory:
            target_pos = memory["target"]
            target = self.index_to_room.get(target_pos[0])
        if self.target_room:
            target = self.target_room

        # If at target room, claim immediately (highest priority)
        if target is not None and current_room == target:
            scored = [("claim", 2.0)]
            for action in exits:
                scored.append((action, -1.0))
            return "claim", scored

        # Track visited rooms for exploration heuristic
        visited = set()
        if memory and "visited_rooms" in memory:
            visited = memory["visited_rooms"]

        scored: List[Tuple[str, float]] = []
        for action in exits:
            if target is None:
                # No target -> explore toward nearest unvisited room
                zA_next = predict_next_fn(zA, action)
                next_idx = zA_next.agent_pos[0]
                next_room = self.index_to_room.get(next_idx, "")
                if next_room and next_room not in visited:
                    scored.append((action, 1.0))  # direct unvisited
                else:
                    # Score by BFS distance to nearest unvisited room
                    nearest_dist = self._nearest_unvisited_dist(next_room, visited)
                    score = 1.0 / (1.0 + nearest_dist)  # closer = higher
                    scored.append((action, score))
                continue

            # Score by distance reduction to target
            zA_next = predict_next_fn(zA, action)
            next_idx = zA_next.agent_pos[0]
            next_room = self.index_to_room.get(next_idx, "")

            dist_current = self._bfs_distance(current_room, target)
            dist_next = self._bfs_distance(next_room, target)

            if self.goal_mode == "seek":
                score = float(dist_current - dist_next)  # positive = closer
            else:
                score = float(dist_next - dist_current)  # positive = farther

            # Anti-stay penalty
            if zA_next.agent_pos == zA.agent_pos:
                score -= self.anti_stay_penalty

            scored.append((action, score))

        # Sort descending by score
        scored.sort(key=lambda x: -x[1])

        # Tie-breaking via memory preferences
        if (len(scored) >= 2
                and abs(scored[0][1] - scored[1][1]) < tie_break_delta
                and memory
                and "tie_break_preference" in memory):
            prefs = memory["tie_break_preference"]
            for pref in prefs:
                for i, (a, s) in enumerate(scored):
                    if a == pref:
                        scored.insert(0, scored.pop(i))
                        break

        best_action = scored[0][0]
        return best_action, scored

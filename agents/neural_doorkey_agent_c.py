"""Neural DoorKey Agent C: learned navigation + deterministic interaction.

Hybrid architecture:
  - Navigation (turn_left, turn_right, forward): alpha * neural + (1-alpha) * heuristic
  - Interaction (pickup, toggle): deterministic rules from DoorKeyAgentC

D-essentiality preserved: pickup/toggle require target is not None,
which is only set by D's deconstruction pipeline.

Interface-compatible: returns exactly (str, List[Tuple[str, float]])
so that Loop Gain, Residuum, and ClosureCore see no difference.

Usage:
    net = DoorKeyActionValueNet()
    net.load_state_dict(torch.load("train/checkpoints/doorkey_action_value_net.pt"))
    agent_c = NeuralDoorKeyAgentC(value_net=net)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from kernel.interfaces import StreamC
from state.schema import ZA
from models.doorkey_action_value_net import (
    DoorKeyActionValueNet,
    extract_doorkey_features,
)
from agents.doorkey_agent_c import (
    _DoorKeyGoalProxy,
    _bfs_distance,
    _bfs_next_step,
    DIR_VEC,
    UNREACHABLE,
)

ACTIONS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]


class NeuralDoorKeyAgentC(StreamC):
    """Hybrid: neural navigation scoring + deterministic interaction rules.

    Navigation actions scored via alpha * neural + (1-alpha) * BFS heuristic.
    Pickup/toggle scored deterministically (requires D-provided target).
    """

    def __init__(
        self,
        value_net: DoorKeyActionValueNet,
        goal_mode: str = "seek",
        alpha: float = 0.7,
        device: str = "cpu",
    ):
        self.value_net = value_net
        self.goal_mode = goal_mode
        self.alpha = alpha
        self.device = device
        self.value_net.to(device)
        self.value_net.eval()

        self._goal = _DoorKeyGoalProxy()

        # Phase-specific state (set by adapter's inject_obs_metadata)
        self.phase: str = "FIND_KEY"
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self.door_open: bool = False

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
        for i, (ddx, ddy) in enumerate(DIR_VEC):
            if dx == ddx and dy == ddy:
                return i
        return 0

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
        """BFS distance + turns needed to face BFS-optimal direction."""
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
        """Determine navigation target from D-provided memory."""
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

    def _score_navigation(
        self,
        zA: ZA,
        zA_next: ZA,
        target: Tuple[int, int],
    ) -> float:
        """Hybrid neural + heuristic scoring for a navigation action."""
        direction = zA.direction if zA.direction is not None else 0
        next_dir = zA_next.direction if zA_next.direction is not None else 0
        obstacle_set = set(zA.obstacles)

        # Remove target from obstacles if needed (e.g. door)
        nav_obstacles = obstacle_set
        if target in obstacle_set:
            nav_obstacles = obstacle_set - {target}

        # Heuristic component (BFS effective distance)
        d_now = self._effective_distance(
            zA.agent_pos, direction, target,
            nav_obstacles, zA.width, zA.height)
        d_next = self._effective_distance(
            zA_next.agent_pos, next_dir, target,
            nav_obstacles, zA.width, zA.height)

        heuristic_score = 1.0 / (d_next + 1.0)
        if d_next < d_now:
            heuristic_score += 0.5

        # Neural component
        features = extract_doorkey_features(
            agent_pos=zA.agent_pos,
            agent_dir=direction,
            next_pos=zA_next.agent_pos,
            next_dir=next_dir,
            target_pos=target,
            obstacles=zA.obstacles,
            width=zA.width,
            height=zA.height,
            phase=self.phase,
            carrying_key=self.carrying_key,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_neural = self.value_net(features).item()

        # Scale neural output to match heuristic range [0, ~1.5]
        neural_score = torch.sigmoid(torch.tensor(raw_neural)).item() * 1.5

        return self.alpha * neural_score + (1.0 - self.alpha) * heuristic_score

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        target = self._current_target(zA, memory)
        facing = self._facing_pos(zA)
        scored: List[Tuple[str, float]] = []

        # Track visits for exploration
        self._visit_counts[zA.agent_pos] = (
            self._visit_counts.get(zA.agent_pos, 0) + 1
        )

        for action in ACTIONS:
            score = 0.0
            zA_next = predict_next_fn(zA, action)

            if action in ("turn_left", "turn_right", "forward"):
                if target is not None:
                    score = self._score_navigation(zA, zA_next, target)
                else:
                    # No target: exploration fallback (same as det C)
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
                # DETERMINISTIC: requires D-provided target
                if (target is not None
                        and self.phase == "FIND_KEY"
                        and not self.carrying_key
                        and self.key_pos is not None
                        and facing == self.key_pos):
                    score = 3.0
                else:
                    score = -1.0

            elif action == "toggle":
                # DETERMINISTIC: requires D-provided target
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

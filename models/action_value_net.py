"""Goal-conditioned action value network for Neural Agent C.

Predicts a scalar score for a (state, action) pair that approximates
the BFS-optimal action value (true shortest-path improvement).

Input features (57 dims):
  - 7x7 local obstacle window around agent (49)
  - Normalised positions: agent_x, agent_y, goal_x, goal_y (4)
  - Next position delta: dx, dy (2)
  - Manhattan delta (1): manhattan(now,goal) - manhattan(next,goal)
  - Wall hit flag (1): 1 if next_pos == current_pos

Architecture: Linear(57, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
~5k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

LOCAL_WINDOW = 7  # 7x7 obstacle window around agent


class ActionValueNet(nn.Module):
    """MLP that scores a (state, action) pair for goal-seeking.

    Architecture: 57 -> 64 -> 64 -> 1  (~5k params)
    """

    def __init__(self, input_dim: int = 57, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 1)"""
        return self.net(x)


def extract_local_obstacles(
    agent_pos: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
    width: int,
    height: int,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Extract a local window of obstacles around the agent.

    Returns a flat tensor of window*window binary values.
    Cells outside the grid boundary are treated as obstacles (1).
    """
    ax, ay = agent_pos
    half = window // 2
    grid = torch.zeros(window * window, dtype=torch.float32)
    obstacle_set = set(obstacles)

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            wx, wy = ax + dx, ay + dy
            idx = (dy + half) * window + (dx + half)

            if not (0 <= wx < width and 0 <= wy < height):
                grid[idx] = 1.0  # boundary = obstacle
            elif (wx, wy) in obstacle_set:
                grid[idx] = 1.0

    return grid


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def extract_features(
    agent_pos: Tuple[int, int],
    next_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
    width: int,
    height: int,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Build the full 57-dim feature vector for ActionValueNet.

    Features:
      [0:49]  local obstacle window (7x7)
      [49:53] normalised positions (agent_x, agent_y, goal_x, goal_y)
      [53:55] next position delta (dx, dy)
      [55]    manhattan delta
      [56]    wall hit flag
    """
    norm = float(max(width, height, 1))

    # Local obstacle window
    local = extract_local_obstacles(agent_pos, obstacles, width, height, window)

    # Normalised positions
    positions = torch.tensor([
        agent_pos[0] / norm,
        agent_pos[1] / norm,
        goal_pos[0] / norm if goal_pos[0] >= 0 else 0.5,
        goal_pos[1] / norm if goal_pos[1] >= 0 else 0.5,
    ], dtype=torch.float32)

    # Next position delta
    delta = torch.tensor([
        (next_pos[0] - agent_pos[0]) / norm,
        (next_pos[1] - agent_pos[1]) / norm,
    ], dtype=torch.float32)

    # Manhattan delta
    d_now = manhattan(agent_pos, goal_pos)
    d_next = manhattan(next_pos, goal_pos)
    manh_delta = torch.tensor([(d_now - d_next) / norm], dtype=torch.float32)

    # Wall hit flag
    wall_hit = torch.tensor(
        [1.0 if next_pos == agent_pos else 0.0],
        dtype=torch.float32,
    )

    return torch.cat([local, positions, delta, manh_delta, wall_hit])

"""Direction prior network for L2→L1 compressed navigation (B-level).

Predicts C's navigation score using only L1-level features:
NO target position, NO phase, NO heuristic_delta (those are L2/L3 knowledge).

This net enables compressed-L2 operation where C is bypassed but the agent
can still navigate using learned directional preferences.

Input features (59 dims):
  - 7x7 local obstacle window around agent (49)
  - Normalised positions: agent_x, agent_y (2)
  - Next position delta: dx, dy (2)
  - Wall hit flag (1): 1 if next_pos == current_pos
  - Direction one-hot: [right, down, left, up] (4)
  - Carrying key flag (1)

Compared to DoorKeyActionValueNet (65 dims), we remove:
  - target_x, target_y (2): L2 knowledge (goal-conditioned)
  - heuristic_delta (1): requires target to compute
  - phase one-hot (3): L3 knowledge (semantic)

Architecture: Linear(59, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
~5.1k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Set, Tuple, Union

from models.action_value_net import extract_local_obstacles, LOCAL_WINDOW

DIRECTION_PRIOR_INPUT_DIM = 59


class DirectionPriorNet(nn.Module):
    """MLP that scores a (state, next_state) pair for compressed navigation.

    Architecture: 59 -> 64 -> 64 -> 1  (~5.1k params)
    """

    def __init__(self, input_dim: int = DIRECTION_PRIOR_INPUT_DIM, hidden: int = 64):
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


def extract_l1_features(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    next_pos: Tuple[int, int],
    next_dir: int,
    obstacles: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
    width: int,
    height: int,
    carrying_key: bool,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Build the 59-dim L1 feature vector for DirectionPriorNet.

    Features (L1-level only, no target/phase):
      [0:49]   local obstacle window (7x7)
      [49:51]  normalised agent position (x, y)
      [51:53]  next position delta (dx, dy)
      [53]     wall hit flag
      [54:58]  direction one-hot (4)
      [58]     carrying_key flag
    """
    norm = float(max(width, height, 1))

    # Convert to list if set
    obstacles_list = list(obstacles) if isinstance(obstacles, set) else obstacles

    # [0:49] Local obstacle window (reused from GridWorld nets)
    local = extract_local_obstacles(
        agent_pos, obstacles_list, width, height, window)

    # [49:51] Normalised agent position
    positions = torch.tensor([
        agent_pos[0] / norm,
        agent_pos[1] / norm,
    ], dtype=torch.float32)

    # [51:53] Next position delta
    delta = torch.tensor([
        (next_pos[0] - agent_pos[0]) / norm,
        (next_pos[1] - agent_pos[1]) / norm,
    ], dtype=torch.float32)

    # [53] Wall hit flag
    wall_hit = torch.tensor(
        [1.0 if next_pos == agent_pos else 0.0],
        dtype=torch.float32,
    )

    # [54:58] Direction one-hot (current direction after action)
    dir_onehot = torch.zeros(4, dtype=torch.float32)
    dir_onehot[agent_dir % 4] = 1.0

    # [58] Carrying key flag
    carry_flag = torch.tensor(
        [1.0 if carrying_key else 0.0],
        dtype=torch.float32,
    )

    return torch.cat([
        local, positions, delta, wall_hit,
        dir_onehot, carry_flag,
    ])

"""Goal-conditioned action value network for Neural DoorKey C.

Predicts a scalar score for a (state, next_state) pair that approximates
the BFS-optimal action value for rotation-based DoorKey navigation.

Input features (65 dims):
  - 7x7 local obstacle window around agent (49)
  - Normalised positions: agent_x, agent_y, target_x, target_y (4)
  - Next position delta: dx, dy (2)
  - Heuristic delta (1): manhattan+turns(now) - manhattan+turns(next)
  - Wall hit flag (1): 1 if next_pos == current_pos
  - Direction one-hot: [right, down, left, up] (4)
  - Phase one-hot: [find_key, open_door, reach_goal] (3)
  - Carrying key flag (1)

Architecture: Linear(65, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
~5.5k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple

from models.action_value_net import extract_local_obstacles, LOCAL_WINDOW

DOORKEY_INPUT_DIM = 65

PHASE_MAP = {
    "FIND_KEY": 0,
    "OPEN_DOOR": 1,
    "REACH_GOAL": 2,
}


class DoorKeyActionValueNet(nn.Module):
    """MLP that scores a (state, next_state) pair for DoorKey navigation.

    Architecture: 65 -> 64 -> 64 -> 1  (~5.5k params)
    """

    def __init__(self, input_dim: int = DOORKEY_INPUT_DIM, hidden: int = 64):
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


def extract_doorkey_features(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    next_pos: Tuple[int, int],
    next_dir: int,
    target_pos: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
    width: int,
    height: int,
    phase: str,
    carrying_key: bool,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Build the 65-dim feature vector for DoorKeyActionValueNet.

    Features:
      [0:49]   local obstacle window (7x7)
      [49:53]  normalised positions (agent, target)
      [53:55]  next position delta (dx, dy)
      [55]     heuristic delta (manhattan+turns)
      [56]     wall hit flag
      [57:61]  direction one-hot (4)
      [61:64]  phase one-hot (3)
      [64]     carrying_key flag
    """
    norm = float(max(width, height, 1))

    # [0:49] Local obstacle window (reused from GridWorld)
    local = extract_local_obstacles(
        agent_pos, obstacles, width, height, window)

    # [49:53] Normalised positions
    positions = torch.tensor([
        agent_pos[0] / norm,
        agent_pos[1] / norm,
        target_pos[0] / norm if target_pos[0] >= 0 else 0.5,
        target_pos[1] / norm if target_pos[1] >= 0 else 0.5,
    ], dtype=torch.float32)

    # [53:55] Next position delta
    delta = torch.tensor([
        (next_pos[0] - agent_pos[0]) / norm,
        (next_pos[1] - agent_pos[1]) / norm,
    ], dtype=torch.float32)

    # [55] Heuristic delta (manhattan + turns approximation)
    def _heuristic_dist(pos, d, tgt):
        manh = abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])
        if manh == 0:
            return 0.0
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
        if abs(dx) >= abs(dy):
            desired = 0 if dx > 0 else 2
        else:
            desired = 1 if dy > 0 else 3
        diff = abs(desired - d) % 4
        turns = min(diff, 4 - diff)
        return float(manh + turns)

    h_now = _heuristic_dist(agent_pos, agent_dir, target_pos)
    h_next = _heuristic_dist(next_pos, next_dir, target_pos)
    heur_delta = torch.tensor([(h_now - h_next) / norm], dtype=torch.float32)

    # [56] Wall hit flag
    wall_hit = torch.tensor(
        [1.0 if next_pos == agent_pos else 0.0],
        dtype=torch.float32,
    )

    # [57:61] Direction one-hot (current direction)
    dir_onehot = torch.zeros(4, dtype=torch.float32)
    dir_onehot[agent_dir % 4] = 1.0

    # [61:64] Phase one-hot
    phase_onehot = torch.zeros(3, dtype=torch.float32)
    phase_idx = PHASE_MAP.get(phase, 0)
    phase_onehot[phase_idx] = 1.0

    # [64] Carrying key flag
    carry_flag = torch.tensor(
        [1.0 if carrying_key else 0.0],
        dtype=torch.float32,
    )

    return torch.cat([
        local, positions, delta, heur_delta, wall_hit,
        dir_onehot, phase_onehot, carry_flag,
    ])

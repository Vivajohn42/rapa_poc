"""OnlineDirectionNet: 4-class direction classifier for C->B distillation.

Learns to predict C's navigation direction from L1-level features only.
Trained online during runtime (not offline).

Input features (60 dims):
  - Relative target delta: dx/norm, dy/norm (2)
  - 7x7 local obstacle window around agent (49)
  - Direction one-hot: [right, down, left, up] (4)
  - Phase one-hot: [find_key, open_door, reach_goal] (3)
  - carrying_key flag (1)
  - door_open flag (1)

Output: 4 classes (0=right/E, 1=down/S, 2=left/W, 3=up/N) with softmax
Architecture: 60 -> 64 -> 64 -> 4 (cross-entropy)
~5.1k parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Set, Tuple, Union

from models.action_value_net import extract_local_obstacles, LOCAL_WINDOW

ONLINE_DIR_INPUT_DIM = 60
DIRECTION_CLASSES = 4  # 0=right, 1=down, 2=left, 3=up (matches DIR_VEC)


class OnlineDirectionNet(nn.Module):
    """MLP that classifies navigation direction from L1-level features.

    Architecture: 60 -> 64 -> 64 -> 4  (~5.1k params)
    Output is raw logits (use cross-entropy loss, softmax for inference).
    """

    def __init__(self, input_dim: int = ONLINE_DIR_INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, DIRECTION_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 4) logits"""
        return self.net(x)

    def predict_direction(self, x: torch.Tensor) -> Tuple[int, float]:
        """Single-sample prediction: returns (direction_class, confidence).

        confidence = softmax probability of the winning class.
        """
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0))  # (1, 4)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (4,)
            direction = probs.argmax().item()
            confidence = probs[direction].item()
            return direction, confidence


def extract_online_features(
    agent_pos: Tuple[int, int],
    target: Tuple[int, int],
    agent_dir: int,
    obstacles: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
    width: int,
    height: int,
    phase: int,          # 0=find_key, 1=open_door, 2=reach_goal
    carrying_key: bool,
    door_open: bool,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Build the 60-dim feature vector for OnlineDirectionNet.

    Features:
      [0:2]    relative target delta (dx/norm, dy/norm)
      [2:51]   7x7 local obstacle window (49)
      [51:55]  direction one-hot (4)
      [55:58]  phase one-hot (3)
      [58]     carrying_key flag
      [59]     door_open flag
    """
    norm = float(max(width, height, 1))
    obstacles_list = list(obstacles) if isinstance(obstacles, set) else obstacles

    # [0:2] Relative target delta (normalized)
    rel_target = torch.tensor([
        (target[0] - agent_pos[0]) / norm,
        (target[1] - agent_pos[1]) / norm,
    ], dtype=torch.float32)

    # [2:51] Local obstacle window
    local = extract_local_obstacles(
        agent_pos, obstacles_list, width, height, window)

    # [51:55] Direction one-hot
    dir_onehot = torch.zeros(4, dtype=torch.float32)
    dir_onehot[agent_dir % 4] = 1.0

    # [55:58] Phase one-hot
    phase_onehot = torch.zeros(3, dtype=torch.float32)
    phase_onehot[min(phase, 2)] = 1.0

    # [58] carrying_key
    carry = torch.tensor([1.0 if carrying_key else 0.0], dtype=torch.float32)

    # [59] door_open
    door = torch.tensor([1.0 if door_open else 0.0], dtype=torch.float32)

    return torch.cat([rel_target, local, dir_onehot, phase_onehot, carry, door])

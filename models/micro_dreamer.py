"""MicroDreamer: small forward-model MLP for Stream B.

Learns (state, action) -> (delta_pos, next_dir, blocked) from experience.
Self-supervised: prediction error = training loss.

Input features (61 dims):
  - Normalised position: x/norm, y/norm (2)
  - Direction one-hot: [R, D, L, U] (4)
  - 7x7 local obstacle window around agent (49)
  - Action one-hot: [up, down, left, right] (4)
  - carrying_key flag (1)
  - door_open flag (1)

Output (7 dims):
  - delta_pos: dx, dy (2) -- regression, MSE loss
  - next_dir logits: 4 classes (4) -- classification, CE loss
  - blocked logit: 1 (1) -- binary, BCE loss

Architecture: 61 -> 64 -> 64 -> 7  (~3.3k params)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Union, Set

from models.action_value_net import extract_local_obstacles, LOCAL_WINDOW

DREAMER_INPUT_DIM = 61
DREAMER_OUTPUT_DIM = 7  # 2 (delta_pos) + 4 (dir_logits) + 1 (blocked_logit)

ACTION_TO_IDX = {"up": 0, "down": 1, "left": 2, "right": 3}
NUM_ACTIONS = 4
NUM_DIRS = 4


class MicroDreamerNet(nn.Module):
    """Forward-model MLP: (state_features, action) -> (delta_pos, next_dir, blocked).

    Architecture: 61 -> 64 -> 64 -> 7  (~3.3k params)
    Output layout: [dx, dy, dir_logit_0..3, blocked_logit]
    """

    def __init__(self, input_dim: int = DREAMER_INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, DREAMER_OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 7)"""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> Tuple[Tuple[float, float], int, bool]:
        """Single-sample deterministic prediction.

        Returns (delta_pos, next_dir, blocked).
        """
        with torch.no_grad():
            out = self.forward(x.unsqueeze(0)).squeeze(0)  # (7,)
            dx = out[0].item()
            dy = out[1].item()
            dir_logits = out[2:6]
            next_dir = dir_logits.argmax().item()
            blocked = out[6].item() > 0.0  # sigmoid threshold at 0.5 → logit > 0
            return (dx, dy), next_dir, blocked


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_dreamer_features(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    obstacles: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
    width: int,
    height: int,
    action: str,
    carrying_key: bool = False,
    door_open: bool = False,
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Build the 61-dim feature vector for MicroDreamerNet.

    Features:
      [0:2]    normalised position (x/norm, y/norm)
      [2:6]    direction one-hot (4)
      [6:55]   7x7 local obstacle window (49)
      [55:59]  action one-hot (4)
      [59]     carrying_key flag
      [60]     door_open flag
    """
    norm = float(max(width, height, 1))
    obstacles_list = list(obstacles) if isinstance(obstacles, set) else obstacles

    # [0:2] Normalised position
    pos = torch.tensor([
        agent_pos[0] / norm,
        agent_pos[1] / norm,
    ], dtype=torch.float32)

    # [2:6] Direction one-hot
    dir_onehot = torch.zeros(NUM_DIRS, dtype=torch.float32)
    dir_onehot[agent_dir % NUM_DIRS] = 1.0

    # [6:55] Local obstacle window (reuse from action_value_net)
    local = extract_local_obstacles(
        agent_pos, obstacles_list, width, height, window)

    # [55:59] Action one-hot
    act_onehot = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    act_idx = ACTION_TO_IDX.get(action, 0)
    act_onehot[act_idx] = 1.0

    # [59] carrying_key
    carry = torch.tensor([1.0 if carrying_key else 0.0], dtype=torch.float32)

    # [60] door_open
    door = torch.tensor([1.0 if door_open else 0.0], dtype=torch.float32)

    return torch.cat([pos, dir_onehot, local, act_onehot, carry, door])


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_dreamer_labels(
    zA_current: "ZA",
    zA_next: "ZA",
    action: str,
) -> Tuple[torch.Tensor, int, float]:
    """Extract training labels from a transition.

    Returns:
      pos_label:  (dx, dy) tensor -- regression target
      dir_label:  int 0-3 -- classification target
      blocked:    float 0.0 or 1.0 -- binary target
    """
    cx, cy = zA_current.agent_pos
    nx, ny = zA_next.agent_pos

    dx = float(nx - cx)
    dy = float(ny - cy)
    pos_label = torch.tensor([dx, dy], dtype=torch.float32)

    # Next direction (default to current if not available)
    next_dir = getattr(zA_next, "direction", None)
    if next_dir is None:
        next_dir = getattr(zA_current, "direction", None)
    dir_label = next_dir if next_dir is not None else 0

    # Blocked: agent tried to move but pos didn't change
    # (action is a movement action but delta_pos is zero)
    moved = (dx != 0.0 or dy != 0.0)
    blocked = 0.0 if moved else 1.0

    return pos_label, dir_label, blocked

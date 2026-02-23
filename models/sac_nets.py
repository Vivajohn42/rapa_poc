"""Discrete SAC networks for Stream C (Phase 5b).

Actor: state -> navigation action logits (3 classes: turn_left/turn_right/forward)
Critic: twin Q-networks, state -> Q-value per action (3 values)

Navigation-only masking: pickup/toggle remain deterministic (D-essentiality).
SAC only scores the 3 navigation actions that move/orient the agent.

Input features (60 dims): reuses extract_online_features from online_direction_net.
  [0:2]    relative target delta (dx/norm, dy/norm)
  [2:51]   7x7 local obstacle window (49)
  [51:55]  direction one-hot (4)
  [55:58]  phase one-hot (3)
  [58]     carrying_key flag
  [59]     door_open flag

Architecture: 60 -> 64 -> 64 -> 3 (matches codebase hidden_dim=64 convention)
~8.3k params per network, ~24.9k total trainable (actor + twin Q).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.online_direction_net import ONLINE_DIR_INPUT_DIM

# SAC constants
SAC_INPUT_DIM = ONLINE_DIR_INPUT_DIM  # 60
SAC_NAV_ACTIONS = 3  # turn_left=0, turn_right=1, forward=2

NAV_ACTION_NAMES = ["turn_left", "turn_right", "forward"]
NAV_ACTION_TO_IDX = {"turn_left": 0, "turn_right": 1, "forward": 2}


class SACActorNet(nn.Module):
    """Discrete SAC policy: state -> navigation action logits.

    Architecture: 60 -> 64 -> 64 -> 3  (~8.3k params)
    Output: raw logits for Categorical distribution over 3 nav actions.
    """

    def __init__(
        self,
        input_dim: int = SAC_INPUT_DIM,
        hidden: int = 64,
        n_actions: int = SAC_NAV_ACTIONS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 60) -> (batch, 3) logits"""
        return self.net(x)


class SACQNet(nn.Module):
    """Single Q-network: state -> Q-value per navigation action.

    Architecture: 60 -> 64 -> 64 -> 3  (~8.3k params)
    Output: Q(s, a) for each of the 3 navigation actions.
    """

    def __init__(
        self,
        input_dim: int = SAC_INPUT_DIM,
        hidden: int = 64,
        n_actions: int = SAC_NAV_ACTIONS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 60) -> (batch, 3) Q-values"""
        return self.net(x)


def copy_params(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters from source to target (hard update)."""
    target.load_state_dict(source.state_dict())


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1-tau) * target."""
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.mul_(1.0 - tau).add_(p_src.data * tau)

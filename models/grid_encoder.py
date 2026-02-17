"""Stream A neural model: GridWorld observation -> belief embedding.

Encodes the full GridWorld state (obstacle grid + normalised positions +
hint flag) into a compact belief vector in R^32.

Input encoding (padded to grid_max=15):
  - Binary obstacle grid:  grid_max * grid_max = 225
  - Normalised positions:  agent_x, agent_y, goal_x, goal_y = 4
  - Hint flag:             1
  - Width/height (norm):   1
  Total = 231

Architecture: Linear(231, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 32)
~19k parameters (matches rapa_os BeliefEncoder scale).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


GRID_MAX = 15  # maximum grid dimension for padding


class GridEncoder(nn.Module):
    """MLP that encodes a padded GridWorld observation into a belief vector.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
    Default: obs_dim=231 -> 64 -> 64 -> belief_dim=32  (~19k params)
    """

    def __init__(
        self,
        obs_dim: int = 231,
        belief_dim: int = 32,
        hidden: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, belief_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, obs_dim) -> (batch, belief_dim)"""
        return self.net(x)


def encode_grid_observation(
    width: int,
    height: int,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
    hint: Optional[str] = None,
    grid_max: int = GRID_MAX,
) -> torch.Tensor:
    """Convert a GridWorld observation into a flat tensor for GridEncoder.

    Returns a 1-D tensor of length grid_max*grid_max + 4 + 1 + 1 = 231.
    """
    # Binary obstacle grid (padded to grid_max x grid_max)
    grid = torch.zeros(grid_max * grid_max, dtype=torch.float32)
    for ox, oy in obstacles:
        if 0 <= ox < grid_max and 0 <= oy < grid_max:
            grid[oy * grid_max + ox] = 1.0

    # Normalised positions (0..1 range)
    norm = float(max(width, height, 1))
    positions = torch.tensor(
        [
            agent_pos[0] / norm,
            agent_pos[1] / norm,
            goal_pos[0] / norm if goal_pos[0] >= 0 else -1.0 / norm,
            goal_pos[1] / norm if goal_pos[1] >= 0 else -1.0 / norm,
        ],
        dtype=torch.float32,
    )

    # Hint flag and normalised grid size
    hint_flag = torch.tensor([1.0 if hint else 0.0], dtype=torch.float32)
    size_norm = torch.tensor([width * height / (grid_max * grid_max)], dtype=torch.float32)

    return torch.cat([grid, positions, hint_flag, size_norm])


def encode_za(zA, grid_max: int = GRID_MAX) -> torch.Tensor:
    """Convenience: encode a ZA Pydantic model into a flat tensor."""
    return encode_grid_observation(
        width=zA.width,
        height=zA.height,
        agent_pos=zA.agent_pos,
        goal_pos=zA.goal_pos,
        obstacles=zA.obstacles,
        hint=zA.hint,
        grid_max=grid_max,
    )

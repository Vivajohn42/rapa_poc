"""Meta-Controller networks for Stream D (Phase 5c).

Policy: abstract state -> phase logits + termination logits + confidence logit
Critic: abstract state -> V(s) state-value for REINFORCE baseline

D operates on abstract state (20 dims), not spatial features (60 dims).
Architecture: hidden=32 (proportional: 20-dim input vs B/C's 60-dim → 64 hidden).

Output structure (7 dims):
  [0:3]  phase logits (find_key=0, open_door=1, reach_goal=2)
  [3:6]  phase termination logits (per-phase beta, sigmoid → [0,1])
  [6]    confidence logit (sigmoid → [0,1])
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Meta-Controller constants
META_INPUT_DIM = 20
META_N_PHASES = 3       # find_key=0, open_door=1, reach_goal=2
META_OUTPUT_DIM = 7     # 3 phase + 3 termination + 1 confidence

PHASE_NAMES = ["find_key", "open_door", "reach_goal"]
PHASE_TO_IDX = {"find_key": 0, "open_door": 1, "reach_goal": 2}


class MetaControllerNet(nn.Module):
    """Meta-Controller policy: abstract state -> phase + termination + confidence.

    Architecture: 20 -> 32 -> 32 -> 7  (~3.5k params)
    Output: raw logits (phase selection via argmax, termination/confidence via sigmoid).
    """

    def __init__(
        self,
        input_dim: int = META_INPUT_DIM,
        hidden: int = 32,
        output_dim: int = META_OUTPUT_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 20) -> (batch, 7) raw logits"""
        return self.net(x)


class MetaCriticNet(nn.Module):
    """State-value critic for REINFORCE baseline.

    Architecture: 20 -> 32 -> 32 -> 1  (~3.4k params)
    Output: V(s) scalar.
    """

    def __init__(
        self,
        input_dim: int = META_INPUT_DIM,
        hidden: int = 32,
    ):
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
        """x: (batch, 20) -> (batch, 1) state-value"""
        return self.net(x)

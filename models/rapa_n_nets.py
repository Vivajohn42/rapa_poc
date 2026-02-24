"""Neural network definitions for standalone RAPA-N streams.

All networks for the redesigned RAPA-N architecture:
  SharedEncoder  — CNN backbone shared between A (perception) and C (policy)
  ForwardNet     — B's forward dynamics model (embedding + action → next embedding)
  SACActorNet5   — C's policy network (embedding → 5 action logits)
  SACQNet5       — C's Q-network (embedding → 5 Q-values)
  MetaGRUNet     — D's recurrent meta-controller (step sequence → phase + confidence)

Parameter budget (~85k total, fair vs PPO's ~77k):
  SharedEncoder: ~66k (CNN 57k + FC 65.6k, shared between A and C)
  ForwardNet:    ~17k (standalone)
  SACActorNet5:  ~8.8k (actor head only, backbone shared)
  SACQNet5:      ~8.8k × 2 = ~17.6k (twin Q, not counted in param comparison)
  MetaGRUNet:    ~28k (standalone)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── Constants ─────────────────────────────────────────────────────
EMBEDDING_DIM = 64
N_ACTIONS = 5  # turn_left, turn_right, forward, pickup, toggle
ACTION_NAMES = ["turn_left", "turn_right", "forward", "pickup", "toggle"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}

# D's GRU input: embedding + action_onehot + reward
META_INPUT_DIM = EMBEDDING_DIM + N_ACTIONS + 1  # 70
META_HIDDEN_DIM = 64
META_SEQ_LEN = 16  # last N steps for GRU

N_PHASES = 3  # find_key, open_door, reach_goal
PHASE_NAMES = ["find_key", "open_door", "reach_goal"]


# ── Shared Encoder (A's CNN backbone, shared with C) ─────────────

class SharedEncoder(nn.Module):
    """CNN encoder for 7×7×3 MiniGrid ego-view observations.

    Architecture identical to PPO's backbone for fair comparison:
      Conv2d(3, 16, k=2, s=1) → ReLU    # 7×7 → 6×6
      Conv2d(16, 32, k=2, s=1) → ReLU   # 6×6 → 5×5
      Conv2d(32, 64, k=2, s=1) → ReLU   # 5×5 → 4×4
      Flatten(1024)
      Linear(1024, 64) → ReLU

    Output: 64-dim embedding vector.
    ~66k params total.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        # Compute flattened size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 7, 7)
            cnn_out = self.cnn(dummy)
            self._cnn_flat = int(np.prod(cnn_out.shape[1:]))

        self.fc = nn.Sequential(
            nn.Linear(self._cnn_flat, EMBEDDING_DIM),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, 3, 7, 7) float32 [0, 1] → (batch, 64) embedding."""
        x = self.cnn(obs)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


# ── Forward Model (B's dynamics predictor) ────────────────────────

class ForwardNet(nn.Module):
    """Forward dynamics model: (embedding, action) → predicted next embedding.

    Architecture:
      Linear(69, 128) → ReLU → Linear(128, 64)  — embedding prediction
      Pos-Head: Linear(128, 6) — delta_pos(2) + dir_logits(4)

    ~17k params.
    """

    def __init__(self):
        super().__init__()
        input_dim = EMBEDDING_DIM + N_ACTIONS  # 69
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.emb_head = nn.Linear(128, EMBEDDING_DIM)  # → 64
        self.pos_head = nn.Linear(128, 6)  # → delta_pos(2) + dir_logits(4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self, emb: torch.Tensor, action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (predicted_embedding, delta_pos, dir_logits).

        emb:            (batch, 64)
        action_onehot:  (batch, 5)
        → predicted_emb: (batch, 64)
        → delta_pos:     (batch, 2)
        → dir_logits:    (batch, 4)
        """
        x = torch.cat([emb, action_onehot], dim=-1)
        h = self.shared(x)
        pred_emb = self.emb_head(h)
        pos_out = self.pos_head(h)
        delta_pos = pos_out[:, :2]
        dir_logits = pos_out[:, 2:]
        return pred_emb, delta_pos, dir_logits


# ── SAC Networks (C's policy and critics) ─────────────────────────

class SACActorNet5(nn.Module):
    """Discrete SAC policy: embedding → 5 action logits.

    Architecture: Linear(64, 128) → ReLU → Linear(128, 5)
    ~8.8k params.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor output head with smaller gain
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (batch, 64) → (batch, 5) logits."""
        return self.net(emb)


class SACQNet5(nn.Module):
    """Q-network: embedding → Q-value per action.

    Architecture: Linear(64, 128) → ReLU → Linear(128, 5)
    ~8.8k params.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (batch, 64) → (batch, 5) Q-values."""
        return self.net(emb)


# ── PPO Value Network (C's critic for on-policy learning) ────────

class PPOValueNet5(nn.Module):
    """Value function: embedding → scalar V(s).

    Architecture: Linear(64, 128) → ReLU → Linear(128, 1)
    ~8.3k params.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Value head with gain=1.0
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (batch, 64) → (batch, 1) value."""
        return self.net(emb)


# ── Meta-Controller GRU (D's recurrent network) ──────────────────

class MetaGRUNet(nn.Module):
    """Recurrent meta-controller: step sequence → phase + confidence.

    Input per step: (embedding[64], action_onehot[5], reward[1]) = 70
    GRU: hidden_size=64
    Head: Linear(64, 32) → ReLU → Linear(32, 4) → phase(3) + confidence(1)

    ~28k params.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=META_INPUT_DIM,
            hidden_size=META_HIDDEN_DIM,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(META_HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, N_PHASES + 1),  # phase(3) + confidence(1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process step sequence and return phase logits + confidence.

        seq:    (batch, seq_len, 70)
        hidden: (1, batch, 64) or None
        → phase_logits: (batch, 3)
        → confidence:   (batch, 1) raw logit (apply sigmoid externally)
        → hidden:       (1, batch, 64)
        """
        gru_out, hidden = self.gru(seq, hidden)
        # Use last timestep output
        last = gru_out[:, -1, :]  # (batch, 64)
        out = self.head(last)     # (batch, 4)
        phase_logits = out[:, :N_PHASES]
        confidence_logit = out[:, N_PHASES:]
        return phase_logits, confidence_logit, hidden


# ── Utility functions ─────────────────────────────────────────────

def action_to_onehot(action: str) -> torch.Tensor:
    """Convert action string to one-hot tensor (5,)."""
    idx = ACTION_TO_IDX.get(action, 2)  # default: forward
    oh = torch.zeros(N_ACTIONS)
    oh[idx] = 1.0
    return oh


def copy_params(source: nn.Module, target: nn.Module) -> None:
    """Hard parameter copy from source to target."""
    target.load_state_dict(source.state_dict())


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1-tau) * target."""
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.mul_(1.0 - tau).add_(p_src.data * tau)

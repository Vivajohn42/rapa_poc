"""Lightweight regime-based governance for PPO training.

Implements DEF regime switching (EXPLORE / GOALSEEK / RECOVER) with:
- Phase-conditioned reward shaping (+0.05 transition bonuses)
- Entropy scheduling (ent_coef 0.01→0.05 based on EXPLORE fraction)
- Action logit biasing (suppress useless actions per phase)
- Stuck detection → RECOVER (random exploration burst)

Zero learnable parameters. All state is per-environment.
"""
from __future__ import annotations

from collections import deque
from typing import List

import numpy as np


# Action indices matching ACTION_NAMES in rapa_n_nets.py
# 0=turn_left, 1=turn_right, 2=forward, 3=pickup, 4=toggle

# Phase-conditioned logit biases (applied during rollout sampling only)
PHASE_BIAS = {
    "FIND_KEY":   np.array([0.0, 0.0, 0.5, -1.0, -1.0], dtype=np.float32),
    "OPEN_DOOR":  np.array([0.0, 0.0, 0.0,  0.5,  0.5], dtype=np.float32),
    "REACH_GOAL": np.array([0.0, 0.0, 0.5, -1.0, -1.0], dtype=np.float32),
}
NO_BIAS = np.zeros(5, dtype=np.float32)


class GovernanceController:
    """DEF regime-based governance for PPO training.

    Tracks per-env state and provides governance signals:
    - observe(): shaped reward + state tracking
    - get_logit_bias(): phase-conditioned action priors
    - get_entropy_coef(): regime-based entropy coefficient
    """

    # Stuck detection
    STUCK_WINDOW = 8
    STUCK_UNIQUE_THRESHOLD = 2
    RECOVER_DURATION = 3
    RECOVER_NOISE_SCALE = 0.5

    # Entropy scheduling
    ENT_COEF_LOW = 0.01   # GOALSEEK (exploit)
    ENT_COEF_HIGH = 0.05  # EXPLORE (explore)

    def __init__(self, n_envs: int, max_steps: int) -> None:
        self.n_envs = n_envs
        self.max_steps = max_steps

        # Per-env state
        self.env_phases: List[str] = ["FIND_KEY"] * n_envs
        self.env_regimes: List[str] = ["EXPLORE"] * n_envs
        self.position_history: List[deque] = [
            deque(maxlen=self.STUCK_WINDOW) for _ in range(n_envs)
        ]
        self.recover_countdown: List[int] = [0] * n_envs

    def observe(self, env_idx: int, obs, action: str,
                reward: float, done: bool) -> float:
        """Track state, compute regime, return shaped reward."""
        base = reward - 0.001  # step penalty

        # Phase transition bonuses
        prev_phase = self.env_phases[env_idx]
        current_phase = obs.phase

        if prev_phase == "FIND_KEY" and current_phase == "OPEN_DOOR":
            base += 0.05  # picked up key
        elif prev_phase == "OPEN_DOOR" and current_phase == "REACH_GOAL":
            base += 0.05  # opened door

        # Update phase
        if done:
            self.on_episode_end(env_idx)
        else:
            self.env_phases[env_idx] = current_phase
            # Position tracking for stuck detection
            self.position_history[env_idx].append(obs.agent_pos)
            self._update_regime(env_idx, current_phase)

        return base

    def get_logit_bias(self, env_idx: int) -> np.ndarray:
        """Return logit bias vector (5,) for action sampling."""
        if self.env_regimes[env_idx] == "RECOVER":
            return np.random.randn(5).astype(np.float32) * self.RECOVER_NOISE_SCALE
        phase = self.env_phases[env_idx]
        return PHASE_BIAS.get(phase, NO_BIAS)

    def compute_entropy_coef(self, regimes: list) -> float:
        """Compute effective entropy coef from buffer regime distribution."""
        if not regimes:
            return self.ENT_COEF_LOW
        explore_count = sum(
            1 for r in regimes if r in ("EXPLORE", "RECOVER")
        )
        explore_frac = explore_count / len(regimes)
        return self.ENT_COEF_LOW + (
            (self.ENT_COEF_HIGH - self.ENT_COEF_LOW) * explore_frac
        )

    def on_episode_end(self, env_idx: int) -> None:
        """Reset per-env state at episode boundary."""
        self.env_phases[env_idx] = "FIND_KEY"
        self.env_regimes[env_idx] = "EXPLORE"
        self.position_history[env_idx].clear()
        self.recover_countdown[env_idx] = 0

    # -- internals --

    def _update_regime(self, env_idx: int, phase: str) -> None:
        """Determine regime for this env."""
        # RECOVER has priority
        if self.recover_countdown[env_idx] > 0:
            self.recover_countdown[env_idx] -= 1
            self.env_regimes[env_idx] = "RECOVER"
            return

        # Check stuck
        if self._is_stuck(env_idx):
            self.recover_countdown[env_idx] = self.RECOVER_DURATION
            self.env_regimes[env_idx] = "RECOVER"
            return

        # Phase-based regime
        if phase == "FIND_KEY":
            self.env_regimes[env_idx] = "EXPLORE"
        else:
            self.env_regimes[env_idx] = "GOALSEEK"

    def _is_stuck(self, env_idx: int) -> bool:
        """Detect position loops in the last STUCK_WINDOW steps."""
        hist = self.position_history[env_idx]
        if len(hist) < self.STUCK_WINDOW:
            return False
        unique = len(set(hist))
        return unique <= self.STUCK_UNIQUE_THRESHOLD

"""PPO baselines for MiniGrid DoorKey comparison.

Two variants:
  1. PPOActorCritic (CNN): raw 7x7x3 pixels, ~77k params — no features
  2. PPOFeaturesActorCritic (MLP): 60-dim features (same as RAPA), ~9k params

Both use the same PPO algorithm (GAE, clipping, entropy bonus).
The difference isolates ARCHITECTURE from FEATURES:

  Vanilla PPO:    raw pixels  + monolithic CNN  → tests raw learning ability
  PPO+Features:   60-dim feat + monolithic MLP  → same info as RAPA, no streams
  RAPA:           60-dim feat + 4 streams + gov  → tests decomposition value

If PPO+Features ≈ RAPA → features matter, not architecture.
If PPO+Features < RAPA → stream decomposition + governance add value.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.doorkey_gym_wrapper import DoorKeyGymWrapper, DoorKeyVecEnv

# ── PPO Hyperparameters (MiniGrid-standard) ────────────────────────
PPO_LR = 2.5e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_N_EPOCHS = 4
PPO_N_MINIBATCHES = 4
PPO_N_STEPS_PER_ENV = 128
PPO_N_ENVS = 8


class PPOActorCritic(nn.Module):
    """CNN Actor-Critic for 7x7x3 MiniGrid partial observations.

    Architecture:
      Conv2d(3, 16, 2, 1) -> ReLU      # 7x7 -> 6x6
      Conv2d(16, 32, 2, 1) -> ReLU     # 6x6 -> 5x5
      Conv2d(32, 64, 2, 1) -> ReLU     # 5x5 -> 4x4
      Flatten()                         # 64*4*4 = 1024
      Linear(1024, 64) -> ReLU         # shared backbone
      Actor: Linear(64, n_actions)     # action logits
      Critic: Linear(64, 1)            # state-value

    ~77k params total.
    """

    def __init__(self, obs_shape: Tuple[int, ...] = (3, 7, 7), n_actions: int = 7):
        super().__init__()
        c, h, w = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        # Compute flattened size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.cnn(dummy)
            self._cnn_flat = int(np.prod(cnn_out.shape[1:]))

        self.fc = nn.Sequential(
            nn.Linear(self._cnn_flat, 64),
            nn.ReLU(),
        )

        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

        # Orthogonal initialization (standard for PPO)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor and critic heads with smaller gain
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        """obs: (batch, C, H, W) -> (action_dist, value).

        Note: MiniGrid obs is (H, W, C) from env, must be transposed to (C, H, W).
        """
        features = self.cnn(obs)
        features = features.reshape(features.shape[0], -1)
        features = self.fc(features)
        logits = self.actor(features)
        value = self.critic(features)
        return Categorical(logits=logits), value.squeeze(-1)

    def get_action_value(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, entropy, value)."""
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class PPOBuffer:
    """Stores rollout data for GAE computation.

    Stores N_ENVS x N_STEPS transitions and computes returns + advantages
    using Generalized Advantage Estimation.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: Tuple[int, ...],
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Computed
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ):
        """Store one timestep of vectorized data."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns(self, last_value: np.ndarray):
        """Compute GAE advantages and discounted returns."""
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t]
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = (delta
                        + self.gamma * self.gae_lambda
                        * next_non_terminal * last_gae)
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_batches(
        self, n_minibatches: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Flatten and split into minibatches for PPO update."""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)
        batch_size = total // n_minibatches

        # Flatten all arrays: (n_steps, n_envs, ...) -> (total, ...)
        flat_obs = self.obs.reshape(total, *self.obs.shape[2:])
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)

        # Normalize advantages
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        batches = []
        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            batches.append({
                "obs": torch.from_numpy(flat_obs[idx]).permute(0, 3, 1, 2),
                "actions": torch.from_numpy(flat_actions[idx]),
                "log_probs": torch.from_numpy(flat_log_probs[idx]),
                "returns": torch.from_numpy(flat_returns[idx]),
                "advantages": torch.from_numpy(flat_advantages[idx]),
            })

        return batches


class PPOTrainer:
    """Standard PPO trainer with vectorized environments.

    Trains a monolithic PPO agent on raw MiniGrid DoorKey observations.
    Provides observability via frequent progress logging and checkpointing.
    """

    def __init__(
        self,
        size: int = 6,
        seed: int = 42,
        n_envs: int = PPO_N_ENVS,
        n_steps_per_env: int = PPO_N_STEPS_PER_ENV,
        lr: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        ent_coef: float = PPO_ENT_COEF,
        vf_coef: float = PPO_VF_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        n_epochs: int = PPO_N_EPOCHS,
        n_minibatches: int = PPO_N_MINIBATCHES,
    ):
        self.size = size
        self.seed = seed
        self.n_envs = n_envs
        self.n_steps_per_env = n_steps_per_env
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches

        # Environment
        self.vec_env = DoorKeyVecEnv(
            size=size, n_envs=n_envs, seed_base=seed)

        # Network
        self.net = PPOActorCritic(obs_shape=(3, 7, 7), n_actions=7)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # Buffer
        self.buffer = PPOBuffer(
            n_steps=n_steps_per_env,
            n_envs=n_envs,
            obs_shape=(7, 7, 3),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Metrics
        self.total_steps = 0
        self.episode_returns: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.episode_successes: deque = deque(maxlen=100)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def train(
        self,
        total_steps: int = 500_000,
        log_interval: int = 10_000,
        checkpoint_interval: int = 50_000,
    ) -> List[Dict]:
        """Train PPO agent. Returns list of checkpoint metrics.

        Prints progress every log_interval steps:
          PPO [6x6] seed=42:  50000/500000 steps | SR=12% | avg_len=180

        Saves checkpoints every checkpoint_interval steps.
        """
        obs = self.vec_env.reset()  # (n_envs, 7, 7, 3)
        checkpoints = []
        next_log = log_interval
        next_checkpoint = checkpoint_interval

        steps_per_rollout = self.n_steps_per_env * self.n_envs

        while self.total_steps < total_steps:
            # Collect rollout
            self.buffer.ptr = 0
            for step in range(self.n_steps_per_env):
                obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2)  # NHWC->NCHW

                with torch.no_grad():
                    dist, value = self.net(obs_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                actions_np = action.numpy()
                next_obs, rewards, dones = self.vec_env.step(actions_np)

                self.buffer.store(
                    obs=obs,
                    action=actions_np,
                    reward=rewards,
                    done=dones.astype(np.float32),
                    value=value.numpy(),
                    log_prob=log_prob.numpy(),
                )

                obs = next_obs

                # Collect completed episode metrics
                completed = self.vec_env.get_completed_episodes()
                for success, steps, reward in completed:
                    self.episode_successes.append(float(success))
                    self.episode_lengths.append(steps)
                    self.episode_returns.append(reward)

            # Compute returns
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2)
                _, last_value = self.net(obs_t)
                last_value = last_value.numpy()
            self.buffer.compute_returns(last_value)

            # PPO update
            self._update()

            self.total_steps += steps_per_rollout

            # Logging
            if self.total_steps >= next_log:
                self._print_progress(total_steps)
                next_log = self.total_steps + log_interval

            # Checkpoint
            if self.total_steps >= next_checkpoint:
                cp = self._make_checkpoint()
                checkpoints.append(cp)
                next_checkpoint = self.total_steps + checkpoint_interval

        # Final checkpoint
        cp = self._make_checkpoint()
        checkpoints.append(cp)
        return checkpoints

    def _update(self):
        """Perform PPO update from buffer."""
        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.n_minibatches)
            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                returns = batch["returns"]
                advantages = batch["advantages"]

                # Forward pass
                dist, values = self.net(obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def _print_progress(self, total_budget: int):
        """Print current training metrics."""
        sr = (sum(self.episode_successes) / len(self.episode_successes)
              if self.episode_successes else 0.0)
        avg_len = (sum(self.episode_lengths) / len(self.episode_lengths)
                   if self.episode_lengths else 0.0)
        avg_ret = (sum(self.episode_returns) / len(self.episode_returns)
                   if self.episode_returns else 0.0)
        n_eps = len(self.episode_successes)

        # Divergence warning
        diverge_warn = ""
        if self.total_steps >= 200_000 and sr < 0.01:
            diverge_warn = "  !! DIVERGED"

        print(f"  PPO [{self.size}x{self.size}] seed={self.seed}: "
              f"{self.total_steps:>7d}/{total_budget} steps | "
              f"SR={sr:>4.0%} | avg_len={avg_len:>5.0f} | "
              f"rew={avg_ret:.3f} | eps={n_eps}{diverge_warn}")

    def _make_checkpoint(self) -> Dict:
        """Create metrics checkpoint."""
        sr = (sum(self.episode_successes) / len(self.episode_successes)
              if self.episode_successes else 0.0)
        avg_len = (sum(self.episode_lengths) / len(self.episode_lengths)
                   if self.episode_lengths else 0.0)
        avg_ret = (sum(self.episode_returns) / len(self.episode_returns)
                   if self.episode_returns else 0.0)
        return {
            "steps": self.total_steps,
            "sr": sr,
            "avg_len": avg_len,
            "avg_return": avg_ret,
            "n_episodes": len(self.episode_successes),
        }

    def evaluate(
        self,
        n_episodes: int = 100,
        size: Optional[int] = None,
        seed_base: int = 10_000,
    ) -> Dict:
        """Evaluate trained agent (zero-shot if size != training size).

        Returns: {sr, avg_steps_ok, avg_return, n_episodes, size}
        """
        eval_size = size or self.size
        env = DoorKeyGymWrapper(
            size=eval_size,
            max_steps=10 * eval_size * eval_size,
        )

        successes = 0
        steps_ok = []
        returns = []

        self.net.eval()
        with torch.no_grad():
            for ep in range(n_episodes):
                obs = env.reset(seed=seed_base + ep)
                done = False
                ep_return = 0.0
                ep_steps = 0

                while not done:
                    obs_t = torch.from_numpy(obs).unsqueeze(0).permute(
                        0, 3, 1, 2)  # (1, C, H, W)
                    dist, _ = self.net(obs_t)
                    action = dist.sample().item()
                    obs, reward, done = env.step(action)
                    ep_return += reward
                    ep_steps += 1

                success = ep_return > 0
                if success:
                    successes += 1
                    steps_ok.append(ep_steps)
                returns.append(ep_return)

        self.net.train()
        env.close()

        sr = successes / n_episodes
        avg_steps = (sum(steps_ok) / len(steps_ok)) if steps_ok else 0.0
        avg_ret = sum(returns) / len(returns)

        return {
            "sr": sr,
            "avg_steps_ok": avg_steps,
            "avg_return": avg_ret,
            "n_episodes": n_episodes,
            "size": eval_size,
        }

    def close(self):
        self.vec_env.close()


# ═══════════════════════════════════════════════════════════════════
#  PPO + Features (MLP variant — same 60-dim features as RAPA)
# ═══════════════════════════════════════════════════════════════════

class PPOFeaturesActorCritic(nn.Module):
    """MLP Actor-Critic for 60-dim feature observations.

    Architecture:
      Linear(60, 64) -> ReLU
      Linear(64, 64) -> ReLU
      Actor: Linear(64, 7)    [7 MiniGrid actions]
      Critic: Linear(64, 1)   [state-value]

    ~9k params total (vs CNN's ~77k).
    Same features as RAPA's Stream C — isolates architecture effect.
    """

    def __init__(self, obs_dim: int = 60, n_actions: int = 7):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        """obs: (batch, 60) -> (action_dist, value)."""
        features = self.fc(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return Categorical(logits=logits), value.squeeze(-1)

    def get_action_value(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, entropy, value)."""
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class PPOFeaturesBuffer:
    """Stores rollout data for GAE computation (flat feature obs).

    Same as PPOBuffer but obs shape is (60,) instead of (7,7,3).
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int = 60,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def store(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns(self, last_value: np.ndarray):
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = (delta
                        + self.gamma * self.gae_lambda
                        * next_non_terminal * last_gae)
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_batches(
        self, n_minibatches: int
    ) -> List[Dict[str, torch.Tensor]]:
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)
        batch_size = total // n_minibatches

        flat_obs = self.obs.reshape(total, -1)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)

        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        batches = []
        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            batches.append({
                "obs": torch.from_numpy(flat_obs[idx]),  # (batch, 60)
                "actions": torch.from_numpy(flat_actions[idx]),
                "log_probs": torch.from_numpy(flat_log_probs[idx]),
                "returns": torch.from_numpy(flat_returns[idx]),
                "advantages": torch.from_numpy(flat_advantages[idx]),
            })
        return batches


class PPOFeaturesTrainer:
    """PPO trainer using 60-dim features (same as RAPA's Stream C).

    Identical training loop to PPOTrainer but uses:
      - DoorKeyFeaturesVecEnv (60-dim obs)
      - PPOFeaturesActorCritic (MLP, ~9k params)
    """

    def __init__(
        self,
        size: int = 6,
        seed: int = 42,
        n_envs: int = PPO_N_ENVS,
        n_steps_per_env: int = PPO_N_STEPS_PER_ENV,
        lr: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        ent_coef: float = PPO_ENT_COEF,
        vf_coef: float = PPO_VF_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        n_epochs: int = PPO_N_EPOCHS,
        n_minibatches: int = PPO_N_MINIBATCHES,
    ):
        self.size = size
        self.seed = seed
        self.n_envs = n_envs
        self.n_steps_per_env = n_steps_per_env
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches

        from env.doorkey_gym_wrapper import DoorKeyFeaturesVecEnv
        self.vec_env = DoorKeyFeaturesVecEnv(
            size=size, n_envs=n_envs, seed_base=seed)

        self.net = PPOFeaturesActorCritic(obs_dim=60, n_actions=7)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, eps=1e-5)

        self.buffer = PPOFeaturesBuffer(
            n_steps=n_steps_per_env,
            n_envs=n_envs,
            obs_dim=60,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.total_steps = 0
        self.episode_returns: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.episode_successes: deque = deque(maxlen=100)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def train(
        self,
        total_steps: int = 500_000,
        log_interval: int = 10_000,
        checkpoint_interval: int = 50_000,
    ) -> List[Dict]:
        """Train PPO+Features agent. Same interface as PPOTrainer.train()."""
        obs = self.vec_env.reset()  # (n_envs, 60)
        checkpoints = []
        next_log = log_interval
        next_checkpoint = checkpoint_interval
        steps_per_rollout = self.n_steps_per_env * self.n_envs

        while self.total_steps < total_steps:
            self.buffer.ptr = 0
            for step in range(self.n_steps_per_env):
                obs_t = torch.from_numpy(obs)  # (n_envs, 60)

                with torch.no_grad():
                    dist, value = self.net(obs_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                actions_np = action.numpy()
                next_obs, rewards, dones = self.vec_env.step(actions_np)

                self.buffer.store(
                    obs=obs,
                    action=actions_np,
                    reward=rewards,
                    done=dones.astype(np.float32),
                    value=value.numpy(),
                    log_prob=log_prob.numpy(),
                )
                obs = next_obs

                completed = self.vec_env.get_completed_episodes()
                for success, steps, reward in completed:
                    self.episode_successes.append(float(success))
                    self.episode_lengths.append(steps)
                    self.episode_returns.append(reward)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs)
                _, last_value = self.net(obs_t)
                last_value = last_value.numpy()
            self.buffer.compute_returns(last_value)

            self._update()
            self.total_steps += steps_per_rollout

            if self.total_steps >= next_log:
                self._print_progress(total_steps)
                next_log = self.total_steps + log_interval

            if self.total_steps >= next_checkpoint:
                cp = self._make_checkpoint()
                checkpoints.append(cp)
                next_checkpoint = self.total_steps + checkpoint_interval

        cp = self._make_checkpoint()
        checkpoints.append(cp)
        return checkpoints

    def _update(self):
        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.n_minibatches)
            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                returns = batch["returns"]
                advantages = batch["advantages"]

                dist, values = self.net(obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def _print_progress(self, total_budget: int):
        sr = (sum(self.episode_successes) / len(self.episode_successes)
              if self.episode_successes else 0.0)
        avg_len = (sum(self.episode_lengths) / len(self.episode_lengths)
                   if self.episode_lengths else 0.0)
        avg_ret = (sum(self.episode_returns) / len(self.episode_returns)
                   if self.episode_returns else 0.0)
        n_eps = len(self.episode_successes)
        diverge_warn = ""
        if self.total_steps >= 200_000 and sr < 0.01:
            diverge_warn = "  !! DIVERGED"

        print(f"  PPO+F [{self.size}x{self.size}] seed={self.seed}: "
              f"{self.total_steps:>7d}/{total_budget} steps | "
              f"SR={sr:>4.0%} | avg_len={avg_len:>5.0f} | "
              f"rew={avg_ret:.3f} | eps={n_eps}{diverge_warn}")

    def _make_checkpoint(self) -> Dict:
        sr = (sum(self.episode_successes) / len(self.episode_successes)
              if self.episode_successes else 0.0)
        avg_len = (sum(self.episode_lengths) / len(self.episode_lengths)
                   if self.episode_lengths else 0.0)
        avg_ret = (sum(self.episode_returns) / len(self.episode_returns)
                   if self.episode_returns else 0.0)
        return {
            "steps": self.total_steps,
            "sr": sr,
            "avg_len": avg_len,
            "avg_return": avg_ret,
            "n_episodes": len(self.episode_successes),
        }

    def evaluate(
        self,
        n_episodes: int = 100,
        size: Optional[int] = None,
        seed_base: int = 10_000,
    ) -> Dict:
        """Evaluate trained PPO+Features agent."""
        from env.doorkey_gym_wrapper import DoorKeyFeaturesWrapper
        eval_size = size or self.size
        env = DoorKeyFeaturesWrapper(
            size=eval_size,
            max_steps=10 * eval_size * eval_size,
        )

        successes = 0
        steps_ok = []
        returns = []

        self.net.eval()
        with torch.no_grad():
            for ep in range(n_episodes):
                obs = env.reset(seed=seed_base + ep)
                done = False
                ep_return = 0.0
                ep_steps = 0

                while not done:
                    obs_t = torch.from_numpy(obs).unsqueeze(0)  # (1, 60)
                    dist, _ = self.net(obs_t)
                    action = dist.sample().item()
                    obs, reward, done = env.step(action)
                    ep_return += reward
                    ep_steps += 1

                success = ep_return > 0
                if success:
                    successes += 1
                    steps_ok.append(ep_steps)
                returns.append(ep_return)

        self.net.train()
        env.close()

        sr = successes / n_episodes
        avg_steps = (sum(steps_ok) / len(steps_ok)) if steps_ok else 0.0
        avg_ret = sum(returns) / len(returns)

        return {
            "sr": sr,
            "avg_steps_ok": avg_steps,
            "avg_return": avg_ret,
            "n_episodes": n_episodes,
            "size": eval_size,
        }

    def close(self):
        self.vec_env.close()

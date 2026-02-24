"""NeuralPPO_C — On-policy PPO stream for RAPA-N.

Discrete PPO over all 5 DoorKey actions.
Shares CNN backbone with NeuralEncoderA (end-to-end training).
On-policy: collects rollouts per episode, trains with GAE-lambda + PPO clip.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from kernel.interfaces import StreamC, StreamLearner, GoalTarget
from kernel.types import LearnerSignal, LearnerStatus, LearnerMode
from state.schema import ZA
from models.rapa_n_nets import (
    SharedEncoder, SACActorNet5, PPOValueNet5,
    ACTION_NAMES, ACTION_TO_IDX, N_ACTIONS, EMBEDDING_DIM,
)


# ── Goal proxy (satisfies GoalTarget protocol) ───────────────────

class _GoalProxy:
    """Simple goal target store. Kernel writes via goal.target = (x, y)."""

    def __init__(self):
        self._target: Optional[Tuple[int, int]] = None

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        return self._target

    @target.setter
    def target(self, value: Tuple[int, int]) -> None:
        self._target = value


# ── Stream C implementation ───────────────────────────────────────

class NeuralPPO_C(StreamC):
    """PPO policy stream: embedding → action (all 5 DoorKey actions).

    Shares the CNN encoder with NeuralEncoderA. The actor network takes
    the 64-dim embedding as input and outputs logits over 5 actions.
    Uses on-policy PPO with GAE-lambda for advantage estimation.
    """

    def __init__(self, encoder: SharedEncoder):
        self._encoder = encoder
        self._goal_proxy = _GoalProxy()
        self._learner_inst = PPOLearnerC(encoder)
        self._current_obs_img: Optional[torch.Tensor] = None  # set before tick

    def set_current_obs(self, obs_img: torch.Tensor) -> None:
        """Set the raw image tensor for the current observation.

        Must be called before kernel.tick() so choose_action can cache it
        for end-to-end gradient flow in PPO updates.
        obs_img: (3, 7, 7) float32 [0, 1]
        """
        self._current_obs_img = obs_img

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Select action using PPO policy on embedding."""
        learner = self._learner_inst

        if zA.embedding is None:
            action = random.choice(ACTION_NAMES)
            scored = [(a, 0.0) for a in ACTION_NAMES]
            return action, scored

        emb = torch.tensor(zA.embedding, dtype=torch.float32)

        # Sample from policy
        emb_batch = emb.unsqueeze(0)
        with torch.no_grad():
            logits = learner.actor(emb_batch).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()
            value = learner.critic(emb_batch).squeeze().item()

        action = ACTION_NAMES[action_idx]

        # Scored list for kernel
        scored = [(ACTION_NAMES[i], probs[i].item()) for i in range(N_ACTIONS)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Cache for learner (rollout collection) — store raw obs for end-to-end training
        if self._current_obs_img is not None:
            learner.cache_step(self._current_obs_img, action_idx, log_prob, value)
        else:
            # Fallback: should not happen, but store a dummy
            learner.cache_step(torch.zeros(3, 7, 7), action_idx, log_prob, value)

        return action, scored

    @property
    def goal(self) -> GoalTarget:
        return self._goal_proxy

    @property
    def learner(self) -> StreamLearner:
        return self._learner_inst


# ── PPO Learner ──────────────────────────────────────────────────

class PPOLearnerC(StreamLearner):
    """On-policy PPO learner for Stream C.

    Training:
      - Collects (emb, action, log_prob, value, reward, done) per step
      - At episode end: compute GAE-lambda returns, then N PPO epochs
      - End-to-end: actor loss flows through shared encoder

    Key difference from SAC: on-policy, no replay buffer, GAE-lambda
    for temporal credit assignment through sparse rewards.
    """

    def __init__(self, encoder: SharedEncoder):
        self._encoder = encoder

        # Networks (reuse SACActorNet5 for actor, add value net)
        self.actor = SACActorNet5()
        self.critic = PPOValueNet5()

        # Single optimizer for actor + encoder + critic
        self._optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(encoder.parameters())
            + list(self.critic.parameters()),
            lr=2.5e-4,
            eps=1e-5,
        )

        # Episode rollout buffer (on-policy, cleared after each learn())
        # Store raw image tensors (3,7,7) instead of embeddings —
        # recompute embeddings WITH gradients during PPO update
        self._rollout_obs: List[torch.Tensor] = []
        self._rollout_actions: List[int] = []
        self._rollout_log_probs: List[float] = []
        self._rollout_values: List[float] = []
        self._rollout_rewards: List[float] = []
        self._rollout_dones: List[bool] = []

        # Per-step cache (set in cache_step, consumed in observe_transition)
        self._cached_obs: Optional[torch.Tensor] = None
        self._cached_action_idx: Optional[int] = None
        self._cached_log_prob: Optional[float] = None
        self._cached_value: Optional[float] = None

        # Hyperparameters (matching PPO baseline for fairness)
        self._gamma: float = 0.99
        self._gae_lambda: float = 0.95
        self._clip_eps: float = 0.2
        self._vf_coef: float = 0.5
        self._ent_coef: float = 0.01
        self._max_grad_norm: float = 0.5
        self._n_epochs: int = 4
        self._n_minibatches: int = 4
        # Accumulate rollouts over multiple episodes before updating
        # Simulates multi-env collection: gather ~1024 steps, then train
        self._min_rollout_steps: int = 1024

        # Tracking
        self.episodes_trained: int = 0
        self._recent_rewards: deque = deque(maxlen=20)  # env reward only
        self._episode_reward: float = 0.0  # env reward accumulator
        self._episode_env_reward: float = 0.0  # true env reward (no shaping)

    def cache_step(
        self,
        obs_img: torch.Tensor,
        action_idx: int,
        log_prob: float,
        value: float,
    ) -> None:
        """Cache step data from choose_action for rollout collection.

        obs_img: raw image tensor (3, 7, 7) float32 [0,1] — NOT the embedding.
        We store raw observations so we can recompute embeddings with gradients
        during the PPO update, enabling end-to-end encoder training.
        """
        self._cached_obs = obs_img.detach()
        self._cached_action_idx = action_idx
        self._cached_log_prob = log_prob
        self._cached_value = value

    def observe_transition(
        self,
        obs_img: torch.Tensor,
        action_idx: int,
        reward: float,
        done: bool,
        env_reward: float = 0.0,
    ) -> None:
        """Record a step into the rollout buffer.

        obs_img: raw image tensor (3, 7, 7) — stored for end-to-end training.
        reward: shaped reward (env + exploration bonus) for PPO training
        env_reward: true environment reward for readiness tracking
        """
        self._episode_reward += reward
        self._episode_env_reward += env_reward

        # Use cached values if available (from choose_action)
        if self._cached_obs is not None:
            self._rollout_obs.append(self._cached_obs)
            self._rollout_actions.append(self._cached_action_idx)
            self._rollout_log_probs.append(self._cached_log_prob)
            self._rollout_values.append(self._cached_value)
        else:
            # Fallback: compute from raw image
            self._rollout_obs.append(obs_img.detach())
            self._rollout_actions.append(action_idx)
            with torch.no_grad():
                emb = self._encoder(obs_img.unsqueeze(0)).squeeze(0)
                logits = self.actor(emb.unsqueeze(0)).squeeze(0)
                dist = Categorical(F.softmax(logits, dim=-1))
                lp = dist.log_prob(torch.tensor(action_idx)).item()
                v = self.critic(emb.unsqueeze(0)).squeeze().item()
            self._rollout_log_probs.append(lp)
            self._rollout_values.append(v)

        self._rollout_rewards.append(reward)
        self._rollout_dones.append(done)

        self._cached_obs = None
        self._cached_action_idx = None
        self._cached_log_prob = None
        self._cached_value = None

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Accumulate reward for tracking."""
        pass  # reward already tracked in observe_transition

    def learn(self) -> None:
        """PPO update — accumulates rollouts, trains when buffer is large enough.

        Simulates multi-env collection: gathers ~1024 steps across
        multiple episodes, then does a full PPO update. This gives
        more data diversity per update and matches PPO's batch size.
        """
        self._recent_rewards.append(self._episode_env_reward)
        self._episode_reward = 0.0
        self._episode_env_reward = 0.0
        self.episodes_trained += 1

        n_steps = len(self._rollout_obs)
        # Accumulate until we have enough steps for a proper PPO batch
        if n_steps < self._min_rollout_steps:
            return

        # Compute GAE-lambda returns
        advantages, returns = self._compute_gae(n_steps)

        # Convert to tensors — store raw obs, NOT embeddings
        obs_batch = torch.stack(self._rollout_obs)   # (n_steps, 3, 7, 7)
        action_batch = torch.tensor(self._rollout_actions, dtype=torch.long)
        old_log_probs = torch.tensor(self._rollout_log_probs, dtype=torch.float32)
        adv_batch = torch.tensor(advantages, dtype=torch.float32)
        ret_batch = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        if len(adv_batch) > 1:
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # PPO epochs
        for _ in range(self._n_epochs):
            # Shuffle and split into minibatches
            indices = torch.randperm(n_steps)
            mb_size = max(1, n_steps // self._n_minibatches)

            for start in range(0, n_steps, mb_size):
                end = min(start + mb_size, n_steps)
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = action_batch[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = adv_batch[mb_idx]
                mb_ret = ret_batch[mb_idx]

                # CRITICAL: Recompute embeddings WITH gradients
                # This enables end-to-end training of encoder + actor + critic
                mb_embs = self._encoder(mb_obs)  # (mb_size, 64) WITH grad_fn

                # Forward pass through actor
                logits = self.actor(mb_embs)
                dist = Categorical(F.softmax(logits, dim=-1))
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = (new_log_probs - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self._clip_eps,
                                    1.0 + self._clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.critic(mb_embs).squeeze(-1)
                value_loss = F.mse_loss(values, mb_ret)

                # Combined loss
                loss = (policy_loss
                        + self._vf_coef * value_loss
                        - self._ent_coef * entropy)

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self._encoder.parameters())
                    + list(self.critic.parameters()),
                    self._max_grad_norm,
                )
                self._optimizer.step()

        self._clear_rollout()

    def _compute_gae(self, n_steps: int) -> Tuple[List[float], List[float]]:
        """Compute GAE-lambda advantages and returns."""
        values = self._rollout_values
        rewards = self._rollout_rewards
        dones = self._rollout_dones

        # Bootstrap value for last step
        if dones[-1]:
            next_value = 0.0
        else:
            # Estimate value of next state (recompute embedding)
            with torch.no_grad():
                last_obs = self._rollout_obs[-1].unsqueeze(0)
                last_emb = self._encoder(last_obs)
                next_value = self.critic(last_emb).squeeze().item()

        advantages = [0.0] * n_steps
        returns = [0.0] * n_steps
        gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self._gamma * next_val * (1.0 - float(dones[t])) - values[t]
            gae = delta + self._gamma * self._gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def _clear_rollout(self) -> None:
        """Clear episode rollout buffer."""
        self._rollout_obs.clear()
        self._rollout_actions.clear()
        self._rollout_log_probs.clear()
        self._rollout_values.clear()
        self._rollout_rewards.clear()
        self._rollout_dones.clear()

    def set_b_ready(self, ready: bool) -> None:
        """Called by integration layer when B's readiness changes."""
        pass  # PPO doesn't depend on B's readiness for training

    def ready(self) -> LearnerStatus:
        """C is READY when average reward exceeds threshold."""
        if self.episodes_trained < 10:
            return LearnerStatus(
                mode=LearnerMode.TRAINING,
                accuracy=0.0,
                episodes_trained=self.episodes_trained,
                label="ppo-C-warmup",
            )

        avg_reward = (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards else 0.0
        )

        if avg_reward > 0.5 and self.episodes_trained > 40:
            mode = LearnerMode.READY
            label = f"ppo-C-ready(r={avg_reward:.2f})"
        else:
            mode = LearnerMode.TRAINING
            label = f"ppo-C-training(r={avg_reward:.2f})"

        return LearnerStatus(
            mode=mode,
            accuracy=min(1.0, max(0.0, avg_reward)),
            episodes_trained=self.episodes_trained,
            label=label,
        )

    def reset_episode(self) -> None:
        """Reset per-episode state (but keep accumulated rollout buffer)."""
        # Do NOT clear rollout buffer — we accumulate across episodes
        self._cached_obs = None
        self._cached_action_idx = None
        self._cached_log_prob = None
        self._cached_value = None
        self._episode_reward = 0.0
        self._episode_env_reward = 0.0

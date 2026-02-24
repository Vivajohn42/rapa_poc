"""NeuralSAC_C — Standalone SAC policy stream for RAPA-N.

Discrete Soft Actor-Critic over all 5 DoorKey actions.
No deterministic fallback, no BFS, no wrapper.
Shares CNN backbone with NeuralEncoderA (end-to-end training).
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import (
    Any, Callable, Dict, List, Optional, Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from kernel.interfaces import StreamC, StreamLearner, GoalTarget
from kernel.types import LearnerSignal, LearnerStatus, LearnerMode
from state.schema import ZA
from models.rapa_n_nets import (
    SharedEncoder, SACActorNet5, SACQNet5,
    copy_params, soft_update,
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

class NeuralSAC_C(StreamC):
    """SAC policy stream: embedding → action (all 5 DoorKey actions).

    Shares the CNN encoder with NeuralEncoderA. The actor network takes
    the 64-dim embedding as input and outputs logits over 5 actions.
    """

    def __init__(self, encoder: SharedEncoder):
        self._encoder = encoder  # Shared with A (for end-to-end gradient flow)
        self._goal_proxy = _GoalProxy()
        self._learner_inst = SACLearnerC(encoder)

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Select action using SAC policy on embedding."""
        learner = self._learner_inst

        if zA.embedding is None:
            # Fallback: random action (shouldn't happen after A runs)
            action = random.choice(ACTION_NAMES)
            scored = [(a, 0.0) for a in ACTION_NAMES]
            return action, scored

        emb = torch.tensor(zA.embedding, dtype=torch.float32)

        # During warmup: random exploration
        if learner.episodes_trained < learner.warmup_episodes:
            action_idx = random.randint(0, N_ACTIONS - 1)
            action = ACTION_NAMES[action_idx]
            scored = [(a, 0.0) for a in ACTION_NAMES]
            # Cache for learner
            learner.cache_action(emb, action_idx)
            return action, scored

        # Epsilon-greedy exploration that decays over training
        eps_start, eps_end, eps_decay = 0.3, 0.02, 500
        eps = eps_end + (eps_start - eps_end) * max(
            0.0, 1.0 - (learner.episodes_trained - learner.warmup_episodes) / eps_decay
        )

        if random.random() < eps:
            action_idx = random.randint(0, N_ACTIONS - 1)
            action = ACTION_NAMES[action_idx]
            scored = [(a, 0.0) for a in ACTION_NAMES]
            learner.cache_action(emb, action_idx)
            return action, scored

        # SAC policy: sample from softmax distribution
        emb_batch = emb.unsqueeze(0)
        with torch.no_grad():
            logits = learner.actor(emb_batch).squeeze(0)  # (5,)
            # Temperature-scaled softmax
            probs = F.softmax(logits / max(learner.alpha, 1e-4), dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample().item()

            # Q-values for scored list
            q1 = learner.q1(emb_batch).squeeze(0)
            q2 = learner.q2(emb_batch).squeeze(0)
            q_min = torch.min(q1, q2)

        action = ACTION_NAMES[action_idx]
        scored = [(ACTION_NAMES[i], q_min[i].item()) for i in range(N_ACTIONS)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Cache for learner
        learner.cache_action(emb, action_idx)

        return action, scored

    @property
    def goal(self) -> GoalTarget:
        return self._goal_proxy

    @property
    def learner(self) -> StreamLearner:
        return self._learner_inst


# ── SAC Learner ───────────────────────────────────────────────────

class SACLearnerC(StreamLearner):
    """Discrete SAC learner for Stream C.

    Training:
      - Warmup (episodes 0-19): random actions, no updates
      - SAC training (episode 20+): 10 updates per episode end from replay buffer
      - End-to-end: actor loss flows through shared encoder

    Readiness cascade: B must be READY before C starts SAC updates.
    """

    def __init__(self, encoder: SharedEncoder):
        self._encoder = encoder

        # Networks
        self.actor = SACActorNet5()
        self.q1 = SACQNet5()
        self.q2 = SACQNet5()
        self.q1_target = SACQNet5()
        self.q2_target = SACQNet5()
        copy_params(self.q1, self.q1_target)
        copy_params(self.q2, self.q2_target)

        # Auto-alpha (entropy temperature)
        self._target_entropy = -math.log(1.0 / N_ACTIONS) * 0.98
        self._log_alpha = torch.tensor(0.0, requires_grad=True)

        # Optimizers
        self._actor_optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(encoder.parameters()),
            lr=3e-4,
        )
        self._q_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=3e-4,
        )
        self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=3e-4)

        # Replay buffer
        self._replay: deque = deque(maxlen=50_000)
        self._episode_buffer: List[Tuple] = []

        # Per-tick state
        self._cached_emb: Optional[torch.Tensor] = None
        self._cached_action_idx: Optional[int] = None

        # Hyperparameters
        self.warmup_episodes: int = 50
        self._batch_size: int = 128
        self._train_updates: int = 20
        self._gamma: float = 0.99
        self._tau: float = 0.005

        # Tracking
        self.episodes_trained: int = 0
        self._recent_rewards: deque = deque(maxlen=20)
        self._episode_reward: float = 0.0
        self._b_ready: bool = False

    @property
    def alpha(self) -> float:
        return self._log_alpha.exp().item()

    def cache_action(self, emb: torch.Tensor, action_idx: int) -> None:
        """Cache the current step's embedding and action for transition building."""
        self._cached_emb = emb.detach()
        self._cached_action_idx = action_idx

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Accumulate (emb, action, reward, next_emb, done) transitions."""
        self._episode_reward += signal.reward

        # We need next_emb — it will be provided by observe_transition()
        # (called from the eval loop after the next A.infer_zA)
        pass

    def observe_transition(
        self,
        emb: torch.Tensor,
        action_idx: int,
        reward: float,
        next_emb: torch.Tensor,
        done: bool,
    ) -> None:
        """Record a complete transition for SAC training."""
        self._episode_buffer.append((
            emb.detach(),
            action_idx,
            reward,
            next_emb.detach(),
            float(done),
        ))

    def learn(self) -> None:
        """Train SAC at episode end."""
        # Move episode buffer to replay
        if self._episode_buffer:
            self._replay.extend(self._episode_buffer)
            self._episode_buffer.clear()

        self._recent_rewards.append(self._episode_reward)
        self._episode_reward = 0.0
        self.episodes_trained += 1

        # Don't train during warmup
        if self.episodes_trained <= self.warmup_episodes:
            return

        if len(self._replay) < self._batch_size:
            return

        for _ in range(self._train_updates):
            self._sac_update()

    def _sac_update(self) -> None:
        """Single SAC update step."""
        batch = random.sample(list(self._replay), self._batch_size)
        embs, actions, rewards, next_embs, dones = zip(*batch)

        emb_batch = torch.stack(embs)
        action_batch = torch.tensor(actions, dtype=torch.long)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        next_emb_batch = torch.stack(next_embs)
        done_batch = torch.tensor(dones, dtype=torch.float32)

        alpha = self._log_alpha.exp().detach()

        # ── Q update ──
        with torch.no_grad():
            next_logits = self.actor(next_emb_batch)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)

            q1_tgt = self.q1_target(next_emb_batch)
            q2_tgt = self.q2_target(next_emb_batch)
            q_tgt_min = torch.min(q1_tgt, q2_tgt)

            # V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            v_next = (next_probs * (q_tgt_min - alpha * next_log_probs)).sum(dim=-1)
            q_target = reward_batch + self._gamma * (1.0 - done_batch) * v_next

        q1_vals = self.q1(emb_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q2_vals = self.q2(emb_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_vals, q_target) + F.mse_loss(q2_vals, q_target)

        self._q_optim.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self._q_optim.step()

        # ── Actor update (with encoder gradient flow) ──
        logits = self.actor(emb_batch)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)

        with torch.no_grad():
            q1_curr = self.q1(emb_batch)
            q2_curr = self.q2(emb_batch)
            q_min = torch.min(q1_curr, q2_curr)

        # Actor loss: minimize E[alpha * log pi - Q]
        actor_loss = (probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self._encoder.parameters()), 1.0)
        self._actor_optim.step()

        # ── Alpha update ──
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = self._log_alpha * (entropy - self._target_entropy).detach()

        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        # ── Target network update ──
        soft_update(self.q1, self.q1_target, self._tau)
        soft_update(self.q2, self.q2_target, self._tau)

    def set_b_ready(self, ready: bool) -> None:
        """Called by integration layer when B's readiness changes."""
        self._b_ready = ready

    def ready(self) -> LearnerStatus:
        """C is READY when average reward exceeds threshold."""
        if self.episodes_trained <= self.warmup_episodes:
            return LearnerStatus(
                mode=LearnerMode.TRAINING,
                accuracy=0.0,
                episodes_trained=self.episodes_trained,
                label="sac-C-warmup",
            )

        avg_reward = (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards else 0.0
        )

        if avg_reward > 0.5 and self.episodes_trained > 40:
            mode = LearnerMode.READY
            label = f"sac-C-ready(r={avg_reward:.2f})"
        else:
            mode = LearnerMode.TRAINING
            label = f"sac-C-training(r={avg_reward:.2f})"

        return LearnerStatus(
            mode=mode,
            accuracy=min(1.0, max(0.0, avg_reward)),
            episodes_trained=self.episodes_trained,
            label=label,
        )

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._episode_buffer.clear()
        self._cached_emb = None
        self._cached_action_idx = None
        self._episode_reward = 0.0

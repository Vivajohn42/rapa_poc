"""SACAgentC: Discrete Soft Actor-Critic wrapper for Stream C (Phase 5b).

Wraps any deterministic StreamC implementation (e.g. AutonomousDoorKeyAgentC)
and attaches a SACLearnerC that learns navigation policy from experience.

Navigation-only masking: SAC learns turn_left/turn_right/forward (3 actions).
pickup/toggle remain deterministic (D-essentiality preserved).

Phase cascade: B must be READY/OFF before C starts TRAINING.
IL warmstart: cross-entropy from deterministic C teacher (~50 episodes).
RL finetuning: discrete SAC until action agreement > 80%.

While TRAINING: delegates choose_action() to inner deterministic C.
When READY: uses SAC policy for navigation, deterministic rules for interaction.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.interfaces import StreamC, StreamLearner
from kernel.types import LearnerMode, LearnerSignal, LearnerStatus
from models.sac_nets import (
    SACActorNet,
    SACQNet,
    NAV_ACTION_NAMES,
    NAV_ACTION_TO_IDX,
    SAC_NAV_ACTIONS,
    copy_params,
    soft_update,
)
from models.online_direction_net import extract_online_features
from state.schema import ZA

# Direction vectors for facing computation (matches DoorKey convention)
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

PHASE_TO_INT = {"FIND_KEY": 0, "find_key": 0,
                "OPEN_DOOR": 1, "open_door": 1,
                "REACH_GOAL": 2, "reach_goal": 2}

ALL_ACTIONS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]


class SACAgentC(StreamC):
    """Stream C with discrete SAC learning for navigation.

    Wraps a deterministic StreamC and attaches a SACLearnerC.
    choose_action() delegates to inner during TRAINING.
    When READY and use_neural=True, uses the SAC policy for navigation.
    Interaction actions (pickup/toggle) always stay deterministic.
    """

    def __init__(
        self,
        inner: Optional[StreamC] = None,
        *,
        use_neural: bool = True,
        b_learner_fn: Optional[Callable] = None,
        warmup_episodes: int = 50,
        ready_threshold: float = 0.80,
        ready_window: int = 20,
        replay_max: int = 5000,
        batch_size: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        train_epochs: int = 3,
    ):
        if inner is None:
            from agents.autonomous_doorkey_agent_c import AutonomousDoorKeyAgentC
            inner = AutonomousDoorKeyAgentC(goal_mode="seek")
        self._inner = inner
        self._use_neural = use_neural

        # Phase state (forwarded to inner in choose_action)
        self.phase: str = "FIND_KEY"
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self.door_open: bool = False

        # SAC learner
        self._learner_impl = SACLearnerC(
            agent=self,
            b_learner_fn=b_learner_fn,
            warmup_episodes=warmup_episodes,
            ready_threshold=ready_threshold,
            ready_window=ready_window,
            replay_max=replay_max,
            batch_size=batch_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            gamma=gamma,
            tau=tau,
            alpha_init=alpha_init,
            train_epochs=train_epochs,
        )

        # Cached state for transition tracking
        self._cached_features: Optional[torch.Tensor] = None
        self._cached_nav_action_idx: Optional[int] = None
        self._cached_target: Optional[Tuple[int, int]] = None

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Score actions: SAC for navigation when READY, deterministic otherwise.

        Also caches features for the learner's transition tracking.
        """
        # Forward phase state to inner
        self._sync_inner_state()

        # Always run deterministic C (for training labels + interaction rules)
        det_action, det_scored = self._inner.choose_action(
            zA, predict_next_fn, memory, tie_break_delta)

        # Resolve target for feature extraction
        target = self._resolve_target(memory)

        # Cache features for transition tracking (if we have a target)
        if target is not None:
            direction = zA.direction if zA.direction is not None else 0
            phase_int = PHASE_TO_INT.get(self.phase, 0)
            features = extract_online_features(
                agent_pos=zA.agent_pos,
                target=target,
                agent_dir=direction,
                obstacles=zA.obstacles,
                width=zA.width,
                height=zA.height,
                phase=phase_int,
                carrying_key=self.carrying_key,
                door_open=self.door_open,
            )
            self._cached_features = features
            self._cached_target = target

            # Cache the deterministic C's navigation action index (for IL labels)
            if det_action in NAV_ACTION_TO_IDX:
                self._cached_nav_action_idx = NAV_ACTION_TO_IDX[det_action]
            else:
                self._cached_nav_action_idx = None  # interaction action
        else:
            self._cached_features = None
            self._cached_target = None
            self._cached_nav_action_idx = None

        # If SAC not READY, return deterministic action
        if (not self._use_neural
                or self._learner_impl.mode != LearnerMode.READY):
            return det_action, det_scored

        # SAC READY: use neural policy for navigation
        if target is None or self._cached_features is None:
            # No target: fall back to deterministic exploration
            return det_action, det_scored

        # Run SAC actor (greedy at deployment)
        with torch.no_grad():
            logits = self._learner_impl._actor(
                self._cached_features.unsqueeze(0)).squeeze(0)
            action_idx = logits.argmax().item()

        nav_action = NAV_ACTION_NAMES[action_idx]

        # Check interaction overrides (deterministic pickup/toggle)
        facing = self._facing_pos(zA)
        if (not self.carrying_key
                and self.key_pos is not None
                and facing == self.key_pos):
            nav_action = "pickup"
        elif (self.carrying_key
              and not self.door_open
              and self.door_pos is not None
              and facing == self.door_pos):
            nav_action = "toggle"

        # Build scored list for kernel compatibility
        scored = self._build_scored_list(logits, nav_action, det_scored)
        return nav_action, scored

    @property
    def goal(self):
        return self._inner.goal

    @property
    def learner(self) -> StreamLearner:
        return self._learner_impl

    # AutonomousDoorKeyAgentC compatibility: forward ObjectMemory
    def set_object_memory(self, obj_mem) -> None:
        if hasattr(self._inner, "set_object_memory"):
            self._inner.set_object_memory(obj_mem)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_inner_state(self) -> None:
        """Forward phase state to inner agent."""
        self._inner.phase = self.phase
        if hasattr(self._inner, "key_pos"):
            self._inner.key_pos = self.key_pos
        if hasattr(self._inner, "door_pos"):
            self._inner.door_pos = self.door_pos
        if hasattr(self._inner, "carrying_key"):
            self._inner.carrying_key = self.carrying_key
        if hasattr(self._inner, "door_open"):
            self._inner.door_open = self.door_open

    def _resolve_target(
        self,
        memory: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Get navigation target from memory or goal property."""
        if memory and "target" in memory and memory["target"] is not None:
            return tuple(memory["target"])
        if self._inner.goal.target is not None:
            return self._inner.goal.target
        return None

    def _facing_pos(self, zA: ZA) -> Optional[Tuple[int, int]]:
        """Cell the agent is facing."""
        d = zA.direction if zA.direction is not None else 0
        dx, dy = DIR_VEC[d]
        fx, fy = zA.agent_pos[0] + dx, zA.agent_pos[1] + dy
        if 0 <= fx < zA.width and 0 <= fy < zA.height:
            return (fx, fy)
        return None

    def _build_scored_list(
        self,
        logits: torch.Tensor,
        chosen_action: str,
        det_scored: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Build a 5-action scored list from SAC logits + det interaction scores.

        Navigation actions get SAC softmax scores scaled to [0, 2].
        Interaction actions keep their deterministic scores.
        """
        probs = F.softmax(logits, dim=-1)
        det_scores = dict(det_scored)

        scored = []
        for i, name in enumerate(NAV_ACTION_NAMES):
            scored.append((name, probs[i].item() * 2.0))

        # Interaction actions from deterministic scores
        for act in ("pickup", "toggle"):
            scored.append((act, det_scores.get(act, -1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Ensure chosen action is first
        if scored[0][0] != chosen_action:
            for i, (a, s) in enumerate(scored):
                if a == chosen_action:
                    scored[0], scored[i] = scored[i], scored[0]
                    break

        return scored


class SACLearnerC(StreamLearner):
    """StreamLearner implementation for Discrete SAC on navigation.

    Lifecycle:
      - observe_signal(): records transitions for replay buffer
      - learn(): IL warmstart (< warmup_episodes) or SAC training
      - ready(): reports TRAINING/READY based on action agreement
      - reset_episode(): clears per-episode state

    Readiness cascade: B must be READY/OFF before C transitions OFF→TRAINING.
    """

    def __init__(
        self,
        agent: "SACAgentC",
        *,
        b_learner_fn: Optional[Callable] = None,
        warmup_episodes: int = 50,
        ready_threshold: float = 0.80,
        ready_window: int = 20,
        replay_max: int = 5000,
        batch_size: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        train_epochs: int = 3,
    ):
        self._agent = agent
        self._b_learner_fn = b_learner_fn

        # Networks
        self._actor = SACActorNet()
        self._q1 = SACQNet()
        self._q2 = SACQNet()
        self._q1_target = SACQNet()
        self._q2_target = SACQNet()
        copy_params(self._q1, self._q1_target)
        copy_params(self._q2, self._q2_target)

        # Optimizers
        self._actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=lr_actor)
        self._critic_optimizer = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()),
            lr=lr_critic)

        # Alpha (entropy coefficient) with auto-tuning
        self._log_alpha = torch.tensor(
            math.log(alpha_init), requires_grad=True)
        self._alpha_optimizer = torch.optim.Adam(
            [self._log_alpha], lr=lr_alpha)
        self._alpha = alpha_init
        self._target_entropy = -0.5 * math.log(SAC_NAV_ACTIONS)

        # Hyperparameters
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._train_epochs = train_epochs
        self._warmup_episodes = warmup_episodes
        self._ready_threshold = ready_threshold

        # Replay buffer: (features, action_idx, reward, next_features, done)
        self._replay: deque = deque(maxlen=replay_max)
        self._episode_buffer: List[Tuple[
            torch.Tensor, int, float, torch.Tensor, float]] = []

        # Accuracy tracking
        self._eval_window: deque = deque(maxlen=ready_window)
        self._episodes_trained: int = 0
        self._last_accuracy: float = 0.0

        # Mode: OFF if use_neural=False, else TRAINING once B is ready
        self._mode = (LearnerMode.TRAINING
                      if agent._use_neural else LearnerMode.OFF)
        self._b_check_done = False  # B readiness checked at least once

        # Per-tick transition tracking
        self._prev_features: Optional[torch.Tensor] = None
        self._prev_action_idx: Optional[int] = None
        self._prev_c_term: Optional[float] = None

        # Per-episode match tracking (for eval accuracy)
        self._episode_nav_ticks: int = 0
        self._episode_matches: int = 0

    @property
    def mode(self) -> LearnerMode:
        return self._mode

    # ------------------------------------------------------------------
    # StreamLearner interface
    # ------------------------------------------------------------------

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Record transition and track action agreement.

        Called every tick by the kernel. Builds transitions from
        consecutive (features, action, reward, next_features, done) tuples.
        """
        # Check B readiness on first signal (cascade check)
        if (self._agent._use_neural
                and not self._b_check_done
                and self._mode == LearnerMode.TRAINING):
            if not self._check_b_readiness():
                self._mode = LearnerMode.OFF
            self._b_check_done = True

        # Re-check B readiness periodically (every 50 episodes)
        if (self._mode == LearnerMode.OFF
                and self._agent._use_neural
                and self._episodes_trained > 0
                and self._episodes_trained % 50 == 0):
            if self._check_b_readiness():
                self._mode = LearnerMode.TRAINING

        # Get current cached features from agent
        curr_features = self._agent._cached_features
        curr_nav_idx = self._agent._cached_nav_action_idx

        # Track action agreement (SAC vs deterministic C)
        if curr_nav_idx is not None and curr_features is not None:
            self._episode_nav_ticks += 1
            # Check if SAC would pick the same action
            with torch.no_grad():
                logits = self._actor(curr_features.unsqueeze(0)).squeeze(0)
                sac_choice = logits.argmax().item()
            if sac_choice == curr_nav_idx:
                self._episode_matches += 1

        # Build transition from previous tick
        if (self._prev_features is not None
                and self._prev_action_idx is not None
                and curr_features is not None):
            reward = self._shape_reward(signal)
            done_flag = 1.0 if signal.done else 0.0
            self._episode_buffer.append((
                self._prev_features,
                self._prev_action_idx,
                reward,
                curr_features,
                done_flag,
            ))

        # Update previous state
        if curr_features is not None and curr_nav_idx is not None:
            self._prev_features = curr_features.clone()
            self._prev_action_idx = curr_nav_idx
        else:
            # Interaction tick (pickup/toggle) — don't track for SAC
            pass

        self._prev_c_term = signal.c_term

    def learn(self) -> None:
        """Train at episode end.

        IL warmstart for first warmup_episodes, then SAC training.
        """
        # Flush episode buffer to replay
        self._replay.extend(self._episode_buffer)

        # Track episode accuracy
        if self._episode_nav_ticks > 0:
            ep_acc = self._episode_matches / self._episode_nav_ticks
            self._eval_window.append(ep_acc)
            self._last_accuracy = ep_acc
        self._episode_buffer.clear()

        # Don't train if not enough data or mode is OFF
        if self._mode == LearnerMode.OFF:
            self._episodes_trained += 1
            return
        if len(self._replay) < self._batch_size:
            self._episodes_trained += 1
            return

        # IL warmstart or SAC training
        if self._episodes_trained < self._warmup_episodes:
            self._il_warmstart_train()
        else:
            for _ in range(self._train_epochs):
                self._sac_train_step()

        self._episodes_trained += 1
        self._update_mode()

    def ready(self) -> LearnerStatus:
        return LearnerStatus(
            mode=self._mode,
            accuracy=self._last_accuracy,
            episodes_trained=self._episodes_trained,
            label=f"SAC-C(acc={self._last_accuracy:.0%})",
        )

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._prev_features = None
        self._prev_action_idx = None
        self._prev_c_term = None
        self._episode_nav_ticks = 0
        self._episode_matches = 0
        self._episode_buffer.clear()

    # ------------------------------------------------------------------
    # B Readiness Cascade
    # ------------------------------------------------------------------

    def _check_b_readiness(self) -> bool:
        """Check if B is READY or OFF (prerequisite for C learning)."""
        if self._b_learner_fn is None:
            return True  # No B reference → assume ready
        b_status = self._b_learner_fn()
        return b_status.mode in (LearnerMode.OFF, LearnerMode.READY)

    # ------------------------------------------------------------------
    # Reward Shaping
    # ------------------------------------------------------------------

    def _shape_reward(self, signal: LearnerSignal) -> float:
        """Shape reward for SAC navigation learning.

        Components:
        - Step cost: -0.01 per tick (encourages efficiency)
        - Goal bonus: signal.reward (from env, +1 on success)
        - c_term improvement: bonus when getting closer to target
        """
        shaped = -0.01  # Step cost
        shaped += signal.reward  # Env reward

        # c_term improvement bonus
        if self._prev_c_term is not None and signal.c_term < 1.0:
            improvement = self._prev_c_term - signal.c_term
            if improvement > 0:
                shaped += 0.1 * improvement

        return shaped

    # ------------------------------------------------------------------
    # IL Warmstart
    # ------------------------------------------------------------------

    def _il_warmstart_train(self) -> None:
        """Cross-entropy from deterministic teacher's navigation decisions.

        Also pre-trains the critic with 1-step TD targets.
        """
        batch_size = min(len(self._replay), self._batch_size)
        batch = random.sample(list(self._replay), batch_size)

        features = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_features = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        # Actor: cross-entropy against teacher's action
        logits = self._actor(features)
        actor_loss = F.cross_entropy(logits, actions)

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # Critic: 1-step TD pre-training
        with torch.no_grad():
            next_q_min = torch.min(
                self._q1_target(next_features),
                self._q2_target(next_features))
            next_v = next_q_min.max(dim=-1).values
            target = rewards + self._gamma * (1.0 - dones) * next_v

        q1_pred = self._q1(features).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self._q2(features).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # Soft target update
        soft_update(self._q1, self._q1_target, self._tau)
        soft_update(self._q2, self._q2_target, self._tau)

    # ------------------------------------------------------------------
    # SAC Training
    # ------------------------------------------------------------------

    def _sac_train_step(self) -> None:
        """One step of discrete SAC training."""
        if len(self._replay) < self._batch_size:
            return

        batch = random.sample(list(self._replay), self._batch_size)

        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        # ---- Critic update ----
        with torch.no_grad():
            next_logits = self._actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits + 1e-8, dim=-1)
            q_target_min = torch.min(
                self._q1_target(next_states),
                self._q2_target(next_states))
            v_next = (next_probs * (
                q_target_min - self._alpha * next_log_probs)).sum(-1)
            target = rewards + self._gamma * (1.0 - dones) * v_next

        q1_pred = self._q1(states).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self._q2(states).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # ---- Actor update ----
        logits = self._actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits + 1e-8, dim=-1)

        with torch.no_grad():
            q_min = torch.min(self._q1(states), self._q2(states))

        actor_loss = (probs * (
            self._alpha * log_probs - q_min)).sum(-1).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # ---- Alpha auto-tuning ----
        entropy = -(probs * log_probs).sum(-1).detach()
        alpha_loss = -(self._log_alpha * (
            entropy - self._target_entropy)).mean()

        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        self._alpha = self._log_alpha.exp().item()

        # ---- Soft target update ----
        soft_update(self._q1, self._q1_target, self._tau)
        soft_update(self._q2, self._q2_target, self._tau)

    # ------------------------------------------------------------------
    # Mode Management
    # ------------------------------------------------------------------

    def _update_mode(self) -> None:
        """Transition TRAINING -> READY based on action agreement.

        Threshold: 80% agreement with deterministic C over 20 episodes.
        If use_neural=False, stays OFF (background learning).
        """
        if not self._agent._use_neural:
            self._mode = LearnerMode.OFF
            return

        # Check B readiness (cascade)
        if not self._check_b_readiness():
            self._mode = LearnerMode.OFF
            return

        if len(self._eval_window) >= self._eval_window.maxlen:
            avg = sum(self._eval_window) / len(self._eval_window)
            if avg >= self._ready_threshold:
                self._mode = LearnerMode.READY
            else:
                self._mode = LearnerMode.TRAINING
        else:
            self._mode = LearnerMode.TRAINING

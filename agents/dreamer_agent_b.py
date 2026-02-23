"""DreamerAgentB: Neural forward-model wrapper for Stream B (Phase 5a).

Wraps any deterministic StreamB implementation (AgentB or DoorKeyAgentB)
and attaches a MicroDreamerLearner that learns dynamics from experience.

While TRAINING: delegates predict_next() to the inner deterministic model.
When READY: can optionally use the neural model for predictions.

Transition tracking:
  predict_next() is called ~4x per tick (once per action by C).
  Only ONE transition per tick is real (the action actually executed).
  DreamerAgentB uses tick-based deduplication:
    - predict_next() caches zA on first call per tick
    - observe_signal() records the actual action (signal.action)
    - Next tick's predict_next() completes the transition with the new zA
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from kernel.interfaces import StreamB, StreamLearner
from kernel.types import LearnerMode, LearnerSignal, LearnerStatus
from models.micro_dreamer import (
    MicroDreamerNet,
    extract_dreamer_features,
    extract_dreamer_labels,
    ACTION_TO_IDX,
)
from state.schema import ZA


class DreamerAgentB(StreamB):
    """Stream B with neural forward-model learning.

    Wraps a deterministic StreamB and attaches a MicroDreamerLearner.
    predict_next() always delegates to the inner model during TRAINING.
    When READY and use_neural=True, uses the learned model instead.
    """

    def __init__(
        self,
        inner: Optional[StreamB] = None,
        *,
        use_neural: bool = True,
        ready_threshold: float = 0.95,
        ready_window: int = 20,
        replay_max: int = 5000,
        batch_size: int = 64,
        lr: float = 1e-3,
        train_epochs: int = 3,
    ):
        if inner is None:
            from agents.agent_b import AgentB
            inner = AgentB()
        self._inner = inner
        self._use_neural = use_neural

        # MicroDreamer learner
        net = MicroDreamerNet()
        self._learner_impl = MicroDreamerLearner(
            net=net,
            agent=self,
            ready_threshold=ready_threshold,
            ready_window=ready_window,
            replay_max=replay_max,
            batch_size=batch_size,
            lr=lr,
            train_epochs=train_epochs,
        )

        # Tick-deduplication state for transition tracking
        self._cached_tick: int = -1
        self._cached_zA: Optional[ZA] = None
        self._cached_action: Optional[str] = None

    def predict_next(self, zA: ZA, action: str) -> ZA:
        """Forward-model prediction.

        Also handles transition tracking via tick-deduplication:
        when a new tick's zA arrives, the previous tick's transition
        is flushed to the learner's replay buffer.
        """
        # Tick-deduplication: detect new tick by checking if the learner
        # has observed a new signal.tick since our last cache.
        # The actual flush happens in _maybe_flush_transition() below.
        # Here we just cache the zA on the first call per tick.
        # (C calls predict_next ~4x per tick, always with same zA)
        current_tick = self._learner_impl.last_observed_tick
        if current_tick != self._cached_tick and current_tick >= 0:
            # New tick detected — flush previous transition
            self._maybe_flush_transition(zA)
            self._cached_zA = zA
            self._cached_tick = current_tick

        # Delegate prediction
        if (self._use_neural
                and self._learner_impl.mode == LearnerMode.READY):
            return self._learner_impl.predict_za(zA, action)
        return self._inner.predict_next(zA, action)

    def _maybe_flush_transition(self, zA_next: ZA) -> None:
        """Record transition from previous tick if complete."""
        if (self._cached_zA is not None
                and self._cached_action is not None):
            self._learner_impl.record_transition(
                self._cached_zA, self._cached_action, zA_next)

    @property
    def learner(self) -> StreamLearner:
        return self._learner_impl

    # DoorKeyAgentB compatibility: forward door state updates
    def update_door_state(self, door_pos, door_open) -> None:
        if hasattr(self._inner, "update_door_state"):
            self._inner.update_door_state(door_pos, door_open)


class MicroDreamerLearner(StreamLearner):
    """StreamLearner implementation for MicroDreamer.

    Lifecycle:
      - observe_signal(): records tick + action for transition tracking
      - learn(): trains MLP on replay buffer at episode boundaries
      - ready(): reports TRAINING/READY based on rolling accuracy
      - reset_episode(): flushes episode staging buffer to replay
    """

    def __init__(
        self,
        net: MicroDreamerNet,
        agent: "DreamerAgentB",
        *,
        ready_threshold: float = 0.95,
        ready_window: int = 20,
        replay_max: int = 5000,
        batch_size: int = 64,
        lr: float = 1e-3,
        train_epochs: int = 3,
    ):
        self._net = net
        self._agent = agent  # back-reference for tick deduplication
        self._optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self._replay: deque = deque(maxlen=replay_max)
        self._episode_buffer: List[Tuple[torch.Tensor, torch.Tensor, int, float]] = []
        self._accuracy_window: deque = deque(maxlen=ready_window)
        self._episodes_trained: int = 0
        self._ready_threshold = ready_threshold
        # Start in TRAINING only when use_neural is True (intends to switch
        # to neural predictions). Otherwise stays OFF — collects data and
        # trains in background without affecting regime governance.
        self._mode = (LearnerMode.TRAINING
                      if agent._use_neural else LearnerMode.OFF)
        self._batch_size = batch_size
        self._train_epochs = train_epochs
        self._last_accuracy: float = 0.0

        # Tick tracking
        self._last_observed_tick: int = -1

    @property
    def last_observed_tick(self) -> int:
        return self._last_observed_tick

    @property
    def mode(self) -> LearnerMode:
        return self._mode

    # ------------------------------------------------------------------
    # StreamLearner interface
    # ------------------------------------------------------------------

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Record tick and actual action for transition tracking.

        The DreamerAgentB uses last_observed_tick for deduplication.
        The actual action is cached on the agent for the next tick's
        transition flush.
        """
        self._last_observed_tick = signal.tick
        self._agent._cached_action = signal.action

    def learn(self) -> None:
        """Train on replay buffer (called at episode end).

        Computes per-episode accuracy and updates mode.
        """
        # Flush episode buffer to replay
        self._replay.extend(self._episode_buffer)

        # Compute episode accuracy on episode_buffer before clearing
        if self._episode_buffer:
            ep_acc = self._evaluate_accuracy(self._episode_buffer)
            self._accuracy_window.append(ep_acc)
        self._episode_buffer.clear()

        # Train if enough data
        if len(self._replay) < self._batch_size:
            return

        self._train_on_replay()
        self._episodes_trained += 1
        self._update_mode()

    def ready(self) -> LearnerStatus:
        return LearnerStatus(
            mode=self._mode,
            accuracy=self._last_accuracy,
            episodes_trained=self._episodes_trained,
            label=f"MicroDreamer(acc={self._last_accuracy:.0%})",
        )

    def reset_episode(self) -> None:
        """Flush remaining transitions to episode buffer, reset caches."""
        # Flush the last transition of the episode (if agent was mid-tick)
        # The agent's cached state gets cleared
        self._agent._cached_tick = -1
        self._agent._cached_zA = None
        self._agent._cached_action = None
        self._last_observed_tick = -1

    # ------------------------------------------------------------------
    # Transition recording
    # ------------------------------------------------------------------

    def record_transition(
        self, zA_prev: ZA, action: str, zA_next: ZA,
    ) -> None:
        """Store a (features, labels) transition in episode buffer."""
        direction = zA_prev.direction if zA_prev.direction is not None else 0
        # Extract carrying_key and door_open from environment state if available
        carrying_key = False
        door_open = False
        if hasattr(self._agent._inner, "_door_open"):
            door_open = self._agent._inner._door_open

        features = extract_dreamer_features(
            agent_pos=zA_prev.agent_pos,
            agent_dir=direction,
            obstacles=zA_prev.obstacles,
            width=zA_prev.width,
            height=zA_prev.height,
            action=action,
            carrying_key=carrying_key,
            door_open=door_open,
        )
        pos_label, dir_label, blocked = extract_dreamer_labels(
            zA_prev, zA_next, action)

        self._episode_buffer.append((features, pos_label, dir_label, blocked))

    # ------------------------------------------------------------------
    # Neural prediction
    # ------------------------------------------------------------------

    def predict_za(self, zA: ZA, action: str) -> ZA:
        """Use the neural model to predict next ZA (deterministic)."""
        direction = zA.direction if zA.direction is not None else 0
        carrying_key = False
        door_open = False
        if hasattr(self._agent._inner, "_door_open"):
            door_open = self._agent._inner._door_open

        features = extract_dreamer_features(
            agent_pos=zA.agent_pos,
            agent_dir=direction,
            obstacles=zA.obstacles,
            width=zA.width,
            height=zA.height,
            action=action,
            carrying_key=carrying_key,
            door_open=door_open,
        )

        (dx, dy), next_dir, blocked = self._net.predict(features)

        if blocked:
            new_pos = zA.agent_pos
            new_dir = direction  # direction doesn't change when blocked
        else:
            # Round to nearest integer (grid is discrete)
            new_x = zA.agent_pos[0] + round(dx)
            new_y = zA.agent_pos[1] + round(dy)
            # Clamp to grid boundaries
            new_x = max(0, min(zA.width - 1, new_x))
            new_y = max(0, min(zA.height - 1, new_y))
            new_pos = (new_x, new_y)
            new_dir = next_dir

        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=new_pos,
            goal_pos=zA.goal_pos,
            obstacles=zA.obstacles,
            hint=None,
            direction=new_dir if zA.direction is not None else None,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_on_replay(self) -> None:
        """Sample mini-batches from replay and train for _train_epochs."""
        replay_list = list(self._replay)
        n = len(replay_list)

        for _ in range(self._train_epochs):
            # Sample mini-batch
            batch_size = min(self._batch_size, n)
            indices = random.sample(range(n), batch_size)

            features_batch = torch.stack([replay_list[i][0] for i in indices])
            pos_labels = torch.stack([replay_list[i][1] for i in indices])
            dir_labels = torch.tensor(
                [replay_list[i][2] for i in indices], dtype=torch.long)
            blocked_labels = torch.tensor(
                [replay_list[i][3] for i in indices], dtype=torch.float32)

            # Forward pass
            out = self._net(features_batch)  # (batch, 7)
            pred_pos = out[:, :2]           # (batch, 2)
            pred_dir = out[:, 2:6]          # (batch, 4)
            pred_blocked = out[:, 6]        # (batch,)

            # Multi-task loss
            loss_pos = nn.functional.mse_loss(pred_pos, pos_labels)
            loss_dir = nn.functional.cross_entropy(pred_dir, dir_labels)
            loss_blocked = nn.functional.binary_cross_entropy_with_logits(
                pred_blocked, blocked_labels)

            loss = loss_pos + loss_dir + 0.5 * loss_blocked

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _evaluate_accuracy(
        self, samples: List[Tuple[torch.Tensor, torch.Tensor, int, float]],
    ) -> float:
        """Compute weighted accuracy on a set of samples.

        Accuracy = 0.5*pos_acc + 0.2*dir_acc + 0.3*blocked_acc
        pos_acc: fraction where rounded prediction == actual delta
        dir_acc: fraction where argmax == actual direction
        blocked_acc: fraction where sigmoid(logit) > 0.5 matches actual
        """
        if not samples:
            return 0.0

        features = torch.stack([s[0] for s in samples])
        pos_labels = torch.stack([s[1] for s in samples])
        dir_labels = torch.tensor([s[2] for s in samples], dtype=torch.long)
        blocked_labels = torch.tensor(
            [s[3] for s in samples], dtype=torch.float32)

        with torch.no_grad():
            out = self._net(features)
            pred_pos = out[:, :2]
            pred_dir = out[:, 2:6]
            pred_blocked = out[:, 6]

            # Position accuracy: rounded delta matches actual
            pos_correct = (
                pred_pos.round() == pos_labels
            ).all(dim=1).float().mean().item()

            # Direction accuracy: argmax matches actual
            dir_correct = (
                pred_dir.argmax(dim=1) == dir_labels
            ).float().mean().item()

            # Blocked accuracy: sigmoid threshold matches actual
            blocked_pred = (pred_blocked > 0.0).float()
            blocked_correct = (
                blocked_pred == blocked_labels
            ).float().mean().item()

        accuracy = 0.5 * pos_correct + 0.2 * dir_correct + 0.3 * blocked_correct
        self._last_accuracy = accuracy
        return accuracy

    def _update_mode(self) -> None:
        """Transition TRAINING -> READY based on rolling accuracy window.

        95% threshold over 20 episodes. blocked_accuracy on 16x16 with
        7x7 ego-view requires ~30 episodes for sufficient wall coverage.

        If use_neural=False, stays OFF (background learning without
        affecting regime governance).
        """
        if not self._agent._use_neural:
            # Background-only mode: collect & train but don't affect regime
            self._mode = LearnerMode.OFF
            return

        if len(self._accuracy_window) >= self._accuracy_window.maxlen:
            avg = sum(self._accuracy_window) / len(self._accuracy_window)
            if avg >= self._ready_threshold:
                self._mode = LearnerMode.READY
            else:
                self._mode = LearnerMode.TRAINING
        else:
            self._mode = LearnerMode.TRAINING

    def _current_accuracy(self) -> float:
        return self._last_accuracy

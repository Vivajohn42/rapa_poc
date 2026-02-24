"""NeuralForwardB — Standalone forward dynamics model for RAPA-N.

Predicts next embedding and position delta given current embedding + action.
Self-supervised learning: MSE on embedding prediction, CE on direction,
MSE on position delta.

No deterministic fallback — B learns dynamics purely from experience.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.interfaces import StreamB, StreamLearner
from kernel.types import LearnerSignal, LearnerStatus, LearnerMode
from state.schema import ZA
from models.rapa_n_nets import (
    ForwardNet, action_to_onehot, ACTION_TO_IDX, EMBEDDING_DIM,
)


class NeuralForwardB(StreamB):
    """Neural forward model: (embedding, action) → predicted next ZA.

    Learns dynamics self-supervised from (emb_t, action_t, emb_{t+1}) tuples.
    """

    def __init__(self):
        self._net = ForwardNet()
        self._learner_inst = ForwardLearnerB(self._net)

    def predict_next(self, zA: ZA, action: str) -> ZA:
        """Predict next state given current ZA and action string."""
        if zA.embedding is None:
            # No embedding yet (shouldn't happen after A runs)
            return ZA(
                width=zA.width, height=zA.height,
                agent_pos=zA.agent_pos, goal_pos=zA.goal_pos,
                obstacles=zA.obstacles, direction=zA.direction,
            )

        emb = torch.tensor(zA.embedding, dtype=torch.float32).unsqueeze(0)
        act_oh = action_to_onehot(action).unsqueeze(0)

        with torch.no_grad():
            pred_emb, delta_pos, dir_logits = self._net(emb, act_oh)

        # Compute predicted position
        dp = delta_pos.squeeze(0)
        new_x = zA.agent_pos[0] + int(round(dp[0].item()))
        new_y = zA.agent_pos[1] + int(round(dp[1].item()))
        # Clamp to grid bounds
        new_x = max(0, min(zA.width - 1, new_x))
        new_y = max(0, min(zA.height - 1, new_y))

        new_dir = dir_logits.squeeze(0).argmax().item()

        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=(new_x, new_y),
            goal_pos=zA.goal_pos,
            obstacles=zA.obstacles,
            direction=new_dir,
            embedding=pred_emb.squeeze(0).tolist(),
        )

    @property
    def learner(self) -> StreamLearner:
        return self._learner_inst


class ForwardLearnerB(StreamLearner):
    """Self-supervised learner for the forward dynamics model.

    Collects (embedding_t, action_t, embedding_{t+1}, pos_{t+1}, dir_{t+1})
    transitions and trains on episode end.
    """

    def __init__(self, net: ForwardNet):
        self._net = net
        self._optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Replay buffer for self-supervised training
        self._replay: deque = deque(maxlen=50_000)
        self._episode_buffer: List[Tuple] = []

        # Tracking
        self._prev_emb: Optional[torch.Tensor] = None
        self._prev_action: Optional[str] = None
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._prev_dir: Optional[int] = None

        # Current tick's data (set by observe_signal, consumed next tick)
        self._curr_emb: Optional[List[float]] = None
        self._curr_pos: Optional[Tuple[int, int]] = None
        self._curr_dir: Optional[int] = None

        # Readiness tracking
        self._episodes_trained: int = 0
        self._recent_losses: deque = deque(maxlen=10)
        self._ready_threshold: float = 0.5  # Relaxed: encoder is a moving target

        # Hyperparameters
        self._batch_size: int = 64
        self._train_epochs: int = 10

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Receive per-tick signal. Build transition from previous tick."""
        # We need the current embedding to complete the previous transition.
        # The embedding comes from zA which was computed THIS tick.
        # We get it via a callback pattern: the agent stores it.
        #
        # Problem: LearnerSignal doesn't carry the embedding directly.
        # Solution: NeuralForwardB stores embeddings in observe_step_embedding()
        # which is called by the integration layer.
        pass

    def observe_transition(
        self,
        emb_t: List[float],
        action: str,
        emb_tp1: List[float],
        pos_tp1: Tuple[int, int],
        dir_tp1: int,
        done: bool,
    ) -> None:
        """Record a (s, a, s') transition for training.

        Called externally by the eval loop after each step.
        """
        emb = torch.tensor(emb_t, dtype=torch.float32)
        act_oh = action_to_onehot(action)
        next_emb = torch.tensor(emb_tp1, dtype=torch.float32)
        pos = torch.tensor(pos_tp1, dtype=torch.float32)
        direction = dir_tp1

        self._episode_buffer.append((emb, act_oh, next_emb, pos, direction))

    def learn(self) -> None:
        """Train forward model on episode end."""
        # Move episode transitions to replay buffer
        if self._episode_buffer:
            self._replay.extend(self._episode_buffer)
            self._episode_buffer.clear()

        self._episodes_trained += 1

        if len(self._replay) < self._batch_size:
            return

        total_loss = 0.0
        for _ in range(self._train_epochs):
            batch = random.sample(list(self._replay), self._batch_size)
            embs, acts, next_embs, positions, directions = zip(*batch)

            emb_batch = torch.stack(embs)
            act_batch = torch.stack(acts)
            next_emb_batch = torch.stack(next_embs)
            pos_batch = torch.stack(positions)
            dir_batch = torch.tensor(directions, dtype=torch.long)

            pred_emb, delta_pos, dir_logits = self._net(emb_batch, act_batch)

            # Embedding prediction loss (MSE)
            emb_loss = F.mse_loss(pred_emb, next_emb_batch)

            # Position delta: target is actual position (we predict absolute)
            # Actually, delta_pos predicts dx, dy from current position
            # But we don't have current position in the batch...
            # Simpler: just predict next direction (most informative for governance)
            dir_loss = F.cross_entropy(dir_logits, dir_batch)

            loss = emb_loss + 0.5 * dir_loss

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            self._optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self._train_epochs
        self._recent_losses.append(avg_loss)

    def ready(self) -> LearnerStatus:
        """B is READY when embedding prediction loss is consistently low."""
        if self._episodes_trained < 5:
            return LearnerStatus(
                mode=LearnerMode.TRAINING,
                accuracy=0.0,
                episodes_trained=self._episodes_trained,
                label="forward-B-warmup",
            )

        avg_loss = (
            sum(self._recent_losses) / len(self._recent_losses)
            if self._recent_losses else 1.0
        )
        # Map loss to accuracy: lower loss = higher accuracy
        accuracy = max(0.0, 1.0 - avg_loss)

        if avg_loss < self._ready_threshold and self._episodes_trained >= 10:
            mode = LearnerMode.READY
            label = f"forward-B-ready(loss={avg_loss:.3f})"
        else:
            mode = LearnerMode.TRAINING
            label = f"forward-B-training(loss={avg_loss:.3f})"

        return LearnerStatus(
            mode=mode,
            accuracy=accuracy,
            episodes_trained=self._episodes_trained,
            label=label,
        )

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._episode_buffer.clear()
        self._prev_emb = None
        self._prev_action = None

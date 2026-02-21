"""OnlineDistiller: C->B knowledge distillation during runtime.

Collects "clean teacher samples" when C is active and deciding well,
trains an OnlineDirectionNet to predict C's direction choice, and
provides the trained net for B's compressed-mode navigation.

Key design: Labels are derived from C's ACTUAL decision (the direction
C chose after BFS), NOT from a simple dx/dy toward target. This ensures
the net learns C's wall-avoidance behavior, not just "aim at target."

Two sample types:
  - targeted: C navigating toward a known key/door/goal (clean signal)
  - frontier: C exploring via frontier pseudo-target (noisier, multi-modal)

Training schedule (decoupled from success):
  - Replay buffer: only filled from successful episodes (clean data)
  - Training: every train_interval episodes (total, not just successful),
    once replay has >= min_samples. This ensures training is timely even
    when success rate is low (e.g. 16x16 Stage 1-2).

Enable logic (targeted-slice gating):
  - n_targeted_eval >= min_targeted_eval AND acc_targeted >= min_accuracy_targeted
  - At inference: only trust predictions with confidence >= confidence_threshold

Usage (from Dreyfus runner):
    distiller = OnlineDistiller()
    kernel = MvpKernel(..., online_distiller=distiller)
    # ... per episode:
    #   kernel handles collect_sample() automatically in tick()
    #   at episode end: distiller.end_episode(success)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from models.online_direction_net import (
    OnlineDirectionNet,
    extract_online_features,
    DIRECTION_CLASSES,
)

# Eval holdout fraction for enable-accuracy measurement
EVAL_HOLDOUT = 0.10


@dataclass
class TeacherSample:
    """One clean teacher sample from C's decision."""
    features: torch.Tensor       # (60,) feature vector
    direction_label: int         # 0-3: C's intended direction (from C's actual choice)
    is_frontier: bool = False    # True if target was a frontier pseudo-target


def _c_action_to_direction(c_action: str, agent_dir: int) -> Optional[int]:
    """Derive C's intended direction from its action + current facing.

    Returns direction class (0=right, 1=down, 2=left, 3=up) or None if
    the action doesn't imply a clear direction.

    - forward: C wants to go in the current facing direction
    - turn_left: C wants to face (and eventually go) the new direction
    - turn_right: C wants to face (and eventually go) the new direction
    """
    if c_action == "forward":
        # C chose to move forward -> intended direction = current facing
        return agent_dir % 4
    elif c_action == "turn_left":
        # C is turning toward this direction -> it intends to go there
        return (agent_dir - 1) % 4
    elif c_action == "turn_right":
        return (agent_dir + 1) % 4
    return None  # pickup/toggle don't imply a navigation direction


class OnlineDistiller:
    """C->B online distillation manager.

    Lifecycle:
    1. Each tick when C decides: kernel calls collect_sample() if clean
    2. At episode end: caller invokes end_episode(success)
    3. Training triggers: every train_interval total episodes once
       replay >= min_samples (decoupled from success rate)
    4. Enable: n_targeted_eval >= min_targeted_eval AND
       acc_targeted >= min_accuracy_targeted

    Parameters:
        replay_max: Maximum replay buffer size (circular).
        train_interval: Train every N total episodes (not just successful).
        min_samples: Minimum replay samples before training starts.
        min_accuracy_targeted: Min eval accuracy on targeted samples for enable.
        min_targeted_eval: Min number of targeted eval samples for enable.
        min_mean_confidence: Min mean confidence on eval set for enable (alt path).
        lr: Learning rate for Adam optimizer.
        confidence_threshold: Below this confidence, trigger replan-burst.
    """

    def __init__(
        self,
        *,
        replay_max: int = 2000,
        train_interval: int = 3,
        min_samples: int = 200,
        min_accuracy_targeted: float = 0.55,
        min_targeted_eval: int = 5,
        min_mean_confidence: float = 0.45,
        lr: float = 1e-3,
        confidence_threshold: float = 0.55,
        ema_alpha: float = 0.3,
    ):
        self.replay_max = replay_max
        self.train_interval = train_interval
        self.min_samples = min_samples
        self.min_accuracy_targeted = min_accuracy_targeted
        self.min_targeted_eval = min_targeted_eval
        self.min_mean_confidence = min_mean_confidence
        self.confidence_threshold = confidence_threshold
        self._ema_alpha = ema_alpha  # EMA smoothing for eval metrics

        # Net + optimizer
        self.net = OnlineDirectionNet()
        self._optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()

        # Replay buffer: 90% train, 10% eval (both circular)
        train_max = int(replay_max * (1.0 - EVAL_HOLDOUT))
        eval_max = replay_max - train_max
        self._train_buffer: Deque[TeacherSample] = deque(maxlen=train_max)
        self._eval_buffer: Deque[TeacherSample] = deque(maxlen=eval_max)

        # Per-episode staging buffer (flushed on success)
        self._episode_buffer: List[TeacherSample] = []

        # State
        self._episode_count: int = 0
        self._successful_episodes: int = 0
        self._enabled: bool = False
        self._has_trained_once: bool = False  # tracks warm-start
        self._train_accuracy: float = 0.0
        self._eval_accuracy: float = 0.0
        self._eval_accuracy_targeted: float = 0.0
        self._eval_accuracy_frontier: float = 0.0
        self._eval_mean_confidence: float = 0.0
        self._eval_mean_conf_targeted: float = 0.0
        self._eval_mean_conf_frontier: float = 0.0
        self._n_targeted_eval: int = 0
        self._n_frontier_eval: int = 0
        self._total_samples_collected: int = 0
        self._train_count: int = 0  # number of train() calls (for EMA init)

    # ------------------------------------------------------------------
    # Sample collection (called each tick from kernel)
    # ------------------------------------------------------------------

    def collect_sample(
        self,
        agent_pos: Tuple[int, int],
        target: Tuple[int, int],
        agent_dir: int,
        obstacles: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
        width: int,
        height: int,
        phase: int,              # 0/1/2
        carrying_key: bool,
        door_open: bool,
        c_action: str,           # C's chosen action
        c_scored: Optional[List[Tuple[str, float]]] = None,
        is_frontier: bool = False,  # True if target is a frontier pseudo-target
    ) -> None:
        """Record a clean teacher sample from C's decision.

        The label is C's INTENDED DIRECTION derived from its actual action:
        - forward -> label = current facing direction
        - turn_left -> label = new facing direction after turn
        - turn_right -> label = new facing direction after turn

        This is TRUE distillation: the net learns what C decided after
        BFS, not a rough dx/dy approximation.
        """
        # Derive direction label from C's actual action
        direction_label = _c_action_to_direction(c_action, agent_dir)
        if direction_label is None:
            return  # pickup/toggle: no navigation label

        features = extract_online_features(
            agent_pos=agent_pos,
            target=target,
            agent_dir=agent_dir,
            obstacles=obstacles,
            width=width,
            height=height,
            phase=phase,
            carrying_key=carrying_key,
            door_open=door_open,
        )

        sample = TeacherSample(
            features=features,
            direction_label=direction_label,
            is_frontier=is_frontier,
        )
        self._episode_buffer.append(sample)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def end_episode(self, success: bool) -> None:
        """Called at episode end.

        Flushes episode buffer to replay if episode was successful
        (clean data only). Training is decoupled from success: triggers
        every train_interval total episodes once replay >= min_samples.
        """
        self._episode_count += 1

        if success and self._episode_buffer:
            # Split into train (90%) and eval (10%)
            for sample in self._episode_buffer:
                if random.random() < EVAL_HOLDOUT:
                    self._eval_buffer.append(sample)
                else:
                    self._train_buffer.append(sample)
            self._total_samples_collected += len(self._episode_buffer)
            self._successful_episodes += 1

        self._episode_buffer.clear()

        # Training schedule: decoupled from success rate
        total_replay = len(self._train_buffer) + len(self._eval_buffer)
        if total_replay < self.min_samples:
            return  # not enough data yet

        # Warm-start: train immediately when crossing min_samples
        if not self._has_trained_once:
            self.train()
            self._has_trained_once = True
            return

        # Regular training: every train_interval total episodes
        if self._episode_count % self.train_interval == 0:
            self.train()

    def reset_episode(self) -> None:
        """Reset per-episode staging buffer."""
        self._episode_buffer.clear()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _ema(self, old: float, new: float) -> float:
        """Exponential moving average update.

        First call (train_count==1): use raw value (no history).
        Subsequent calls: blend with EMA alpha.
        """
        if self._train_count <= 1:
            return new
        alpha = self._ema_alpha
        return alpha * new + (1.0 - alpha) * old

    def train(self, epochs: int = 1) -> float:
        """Run mini-training on train buffer, evaluate on eval buffer.

        Measures (two-headed stats, EMA-smoothed):
        - Overall eval accuracy + mean confidence
        - Targeted-only: accuracy + mean confidence (clean navigation)
        - Frontier-only: accuracy + mean confidence (noisier exploration)

        EMA smoothing prevents thrashing when the eval buffer rotates.

        Returns overall eval accuracy.
        """
        if len(self._train_buffer) < 10:
            return 0.0

        self._train_count += 1

        # Train
        train_samples = list(self._train_buffer)
        random.shuffle(train_samples)
        train_features = torch.stack([s.features for s in train_samples])
        train_labels = torch.tensor(
            [s.direction_label for s in train_samples], dtype=torch.long)

        self.net.train()
        for _ in range(epochs):
            logits = self.net(train_features)
            loss = self._loss_fn(logits, train_labels)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # Measure train accuracy
        self.net.eval()
        with torch.no_grad():
            train_preds = self.net(train_features).argmax(dim=-1)
            self._train_accuracy = (
                (train_preds == train_labels).float().mean().item()
            )

        # Measure eval accuracy (on held-out 10%)
        if len(self._eval_buffer) >= 10:
            eval_samples = list(self._eval_buffer)
            eval_features = torch.stack([s.features for s in eval_samples])
            eval_labels = torch.tensor(
                [s.direction_label for s in eval_samples], dtype=torch.long)
            eval_frontier_mask = torch.tensor(
                [s.is_frontier for s in eval_samples], dtype=torch.bool)

            with torch.no_grad():
                eval_logits = self.net(eval_features)
                eval_preds = eval_logits.argmax(dim=-1)
                eval_probs = torch.softmax(eval_logits, dim=-1)
                eval_conf = eval_probs.max(dim=-1).values

                # Overall accuracy + mean confidence (EMA-smoothed)
                raw_acc = (eval_preds == eval_labels).float().mean().item()
                raw_conf = eval_conf.mean().item()
                self._eval_accuracy = self._ema(self._eval_accuracy, raw_acc)
                self._eval_mean_confidence = self._ema(
                    self._eval_mean_confidence, raw_conf)

                # Targeted-only (non-frontier samples)
                targeted_mask = ~eval_frontier_mask
                self._n_targeted_eval = int(targeted_mask.sum().item())
                if self._n_targeted_eval >= 5:
                    raw_acc_t = (
                        (eval_preds[targeted_mask] == eval_labels[targeted_mask])
                        .float().mean().item()
                    )
                    raw_conf_t = eval_conf[targeted_mask].mean().item()
                    self._eval_accuracy_targeted = self._ema(
                        self._eval_accuracy_targeted, raw_acc_t)
                    self._eval_mean_conf_targeted = self._ema(
                        self._eval_mean_conf_targeted, raw_conf_t)
                else:
                    self._eval_accuracy_targeted = self._eval_accuracy
                    self._eval_mean_conf_targeted = self._eval_mean_confidence

                # Frontier-only
                self._n_frontier_eval = int(eval_frontier_mask.sum().item())
                if self._n_frontier_eval >= 5:
                    raw_acc_f = (
                        (eval_preds[eval_frontier_mask] == eval_labels[eval_frontier_mask])
                        .float().mean().item()
                    )
                    raw_conf_f = eval_conf[eval_frontier_mask].mean().item()
                    self._eval_accuracy_frontier = self._ema(
                        self._eval_accuracy_frontier, raw_acc_f)
                    self._eval_mean_conf_frontier = self._ema(
                        self._eval_mean_conf_frontier, raw_conf_f)
                else:
                    self._eval_accuracy_frontier = self._eval_accuracy
                    self._eval_mean_conf_frontier = self._eval_mean_confidence
        else:
            # Fallback to train accuracy if not enough eval data
            self._eval_accuracy = self._train_accuracy
            self._eval_accuracy_targeted = self._train_accuracy
            self._eval_accuracy_frontier = self._train_accuracy
            self._eval_mean_confidence = 0.0
            self._eval_mean_conf_targeted = 0.0
            self._eval_mean_conf_frontier = 0.0
            self._n_targeted_eval = 0
            self._n_frontier_eval = 0

        # Enable if criteria met (targeted-slice gating)
        # Once enabled, stays ON ("latch ON") — confidence gating at
        # inference handles per-tick trust decisions.
        if not self._enabled:
            total_replay = len(self._train_buffer) + len(self._eval_buffer)
            if total_replay >= self.min_samples:
                # Primary: targeted accuracy with minimum sample count
                if (self._n_targeted_eval >= self.min_targeted_eval
                        and self._eval_accuracy_targeted >= self.min_accuracy_targeted):
                    self._enabled = True
                # Alternative: mean confidence over all eval samples
                elif self._eval_mean_confidence >= self.min_mean_confidence:
                    self._enabled = True

        return self._eval_accuracy

    # ------------------------------------------------------------------
    # Inference (called from kernel._compressed_l2_action)
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True if the net has enough data and accuracy to be used."""
        return self._enabled

    def predict(
        self,
        agent_pos: Tuple[int, int],
        target: Tuple[int, int],
        agent_dir: int,
        obstacles: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
        width: int,
        height: int,
        phase: int,
        carrying_key: bool,
        door_open: bool,
    ) -> Tuple[int, float]:
        """Predict desired direction + confidence.

        Returns (direction_class, confidence) where direction is 0-3
        and confidence is the softmax probability.
        """
        features = extract_online_features(
            agent_pos=agent_pos,
            target=target,
            agent_dir=agent_dir,
            obstacles=obstacles,
            width=width,
            height=height,
            phase=phase,
            carrying_key=carrying_key,
            door_open=door_open,
        )
        return self.net.predict_direction(features)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Total teacher samples collected (ever, across all episodes)."""
        return self._total_samples_collected

    @property
    def replay_size(self) -> int:
        """Current size of the replay buffer (train + eval)."""
        return len(self._train_buffer) + len(self._eval_buffer)

    @property
    def accuracy(self) -> float:
        """Last measured overall eval accuracy."""
        return self._eval_accuracy

    @property
    def accuracy_targeted(self) -> float:
        """Last measured eval accuracy on targeted (non-frontier) samples."""
        return self._eval_accuracy_targeted

    @property
    def accuracy_frontier(self) -> float:
        """Last measured eval accuracy on frontier samples."""
        return self._eval_accuracy_frontier

    @property
    def mean_confidence(self) -> float:
        """Last measured mean prediction confidence on eval set."""
        return self._eval_mean_confidence

    @property
    def mean_conf_targeted(self) -> float:
        """Mean confidence on targeted eval samples."""
        return self._eval_mean_conf_targeted

    @property
    def mean_conf_frontier(self) -> float:
        """Mean confidence on frontier eval samples."""
        return self._eval_mean_conf_frontier

    @property
    def episode_samples(self) -> int:
        """Number of samples collected in the current episode so far."""
        return len(self._episode_buffer)

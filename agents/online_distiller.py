"""OnlineDistiller: C->B knowledge distillation during runtime.

Collects "clean teacher samples" when C is active and deciding well,
trains an OnlineDirectionNet to predict C's direction choice, and
provides the trained net for B's compressed-mode navigation.

Key design: Labels are derived from C's ACTUAL decision (the direction
C chose after BFS), NOT from a simple dx/dy toward target. This ensures
the net learns C's wall-avoidance behavior, not just "aim at target."

Two target kinds:
  - REAL: C navigating toward a known key/door/goal (clean signal)
  - FRONTIER: C exploring via frontier pseudo-target (noisier, multi-modal)

Three operating modes (per target kind):
  - OFF: not enough data to train
  - APPRENTICE: training runs, readiness metrics moving
  - EXPERT: readiness fulfilled, B may navigate (trust-gated for REAL,
    unconditional for FRONTIER)

Training schedule (decoupled from success):
  - Replay buffer: only filled from successful episodes (clean data)
  - Training: every train_interval episodes (total, not just successful),
    once replay has >= min_samples. This ensures training is timely even
    when success rate is low (e.g. 16x16 Stage 1-2).

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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from models.online_direction_net import (
    OnlineDirectionNet,
    extract_online_features,
    DIRECTION_CLASSES,
)
from models.trust_primitive import TrustPrimitive

# Eval holdout fraction for enable-accuracy measurement
EVAL_HOLDOUT = 0.10


# ------------------------------------------------------------------
# OS-level enums
# ------------------------------------------------------------------

class DistillerMode(Enum):
    """Operating mode for the distiller (per target kind)."""
    OFF = auto()        # Not enough data to train
    APPRENTICE = auto() # Training runs, readiness metrics moving
    EXPERT = auto()     # Readiness fulfilled, B may navigate


class TargetKind(Enum):
    """Classification of the navigation target."""
    REAL = auto()      # key/door/goal visible — clean navigation signal
    FRONTIER = auto()  # exploration pseudo-target — noisier, multi-modal


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TeacherSample:
    """One clean teacher sample from C's decision."""
    features: torch.Tensor       # (60,) feature vector
    direction_label: int         # 0-3: C's intended direction (from C's actual choice)
    target_kind: TargetKind = TargetKind.REAL


@dataclass
class _KindState:
    """Per-target-kind tracking state."""
    mode: DistillerMode = DistillerMode.OFF
    n_total: int = 0           # samples collected (at collection time)
    n_eval: int = 0            # samples currently in eval buffer
    eval_accuracy: float = 0.0
    eval_mean_conf: float = 0.0
    eval_mean_trust: float = 0.0


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
    4. Per-kind mode transitions: OFF -> APPRENTICE -> EXPERT

    Parameters:
        replay_max: Maximum replay buffer size (circular).
        train_interval: Train every N total episodes (pre-enable schedule).
        min_samples: Minimum replay samples before training starts.
        min_accuracy_targeted: Min eval accuracy on REAL samples for EXPERT.
        min_accuracy_frontier: Min eval accuracy on FRONTIER samples for EXPERT.
        min_targeted_eval: Min number of eval samples per kind for EXPERT.
        min_mean_confidence: Min mean confidence (legacy, kept for compat).
        min_targeted_for_enable: Min REAL samples seen (at collection) for EXPERT.
        warm_start_epochs: Epochs for initial warm-start training.
        train_interval_post_enable: Train every N episodes after EXPERT mode.
        lr: Learning rate for Adam optimizer.
        confidence_threshold: Below this confidence, trigger replan-burst (legacy).
        trust_threshold: Below this trust, trigger C-fallback for REAL (hard gate).
        trust_threshold_frontier: Optional separate trust ref for FRONTIER scoring.
        ema_alpha: EMA smoothing factor for eval metrics.
    """

    def __init__(
        self,
        *,
        replay_max: int = 2000,
        train_interval: int = 3,
        min_samples: int = 200,
        min_accuracy_targeted: float = 0.55,
        min_accuracy_frontier: float = 0.45,
        min_targeted_eval: int = 5,
        min_mean_confidence: float = 0.45,
        min_targeted_for_enable: int = 50,
        warm_start_epochs: int = 5,
        train_interval_post_enable: int = 2,
        lr: float = 1e-3,
        confidence_threshold: float = 0.55,
        trust_threshold: float = 0.35,
        trust_threshold_frontier: Optional[float] = None,
        ema_alpha: float = 0.5,
    ):
        self.replay_max = replay_max
        self.train_interval = train_interval
        self.min_samples = min_samples
        self.min_accuracy_targeted = min_accuracy_targeted
        self.min_accuracy_frontier = min_accuracy_frontier
        self.min_targeted_eval = min_targeted_eval
        self.min_mean_confidence = min_mean_confidence
        self.min_targeted_for_enable = min_targeted_for_enable
        self.warm_start_epochs = warm_start_epochs
        self.train_interval_post_enable = train_interval_post_enable
        self.confidence_threshold = confidence_threshold
        self.trust_threshold = trust_threshold
        self._trust_threshold_frontier = trust_threshold_frontier
        self._ema_alpha = ema_alpha  # EMA smoothing for eval metrics

        # Net + optimizer
        self.net = OnlineDirectionNet()
        self._optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()

        # Trust primitive (margin + entropy based, stream-agnostic)
        self._trust_primitive = TrustPrimitive(n_classes=DIRECTION_CLASSES)

        # Replay buffer: 90% train, 10% eval (both circular)
        train_max = int(replay_max * (1.0 - EVAL_HOLDOUT))
        eval_max = replay_max - train_max
        self._train_buffer: Deque[TeacherSample] = deque(maxlen=train_max)
        self._eval_buffer: Deque[TeacherSample] = deque(maxlen=eval_max)

        # Per-episode staging buffer (flushed on success)
        self._episode_buffer: List[TeacherSample] = []

        # Per-kind readiness state
        self._kind: Dict[TargetKind, _KindState] = {
            TargetKind.REAL: _KindState(),
            TargetKind.FRONTIER: _KindState(),
        }

        # Scalar state
        self._episode_count: int = 0
        self._successful_episodes: int = 0
        self._has_trained_once: bool = False  # tracks warm-start
        self._train_accuracy: float = 0.0
        self._total_samples_collected: int = 0
        self._train_count: int = 0  # number of train() calls (for EMA init)

        # Overall metrics (legacy, for display/CSV backward compat)
        self._eval_accuracy: float = 0.0
        self._eval_mean_confidence: float = 0.0
        self._eval_mean_trust: float = 0.0

    # ------------------------------------------------------------------
    # Mode API (always derived, never stored)
    # ------------------------------------------------------------------

    @property
    def mode(self) -> DistillerMode:
        """Overall mode = highest mode across all kinds. Always derived."""
        if any(ks.mode == DistillerMode.EXPERT
               for ks in self._kind.values()):
            return DistillerMode.EXPERT
        if any(ks.mode == DistillerMode.APPRENTICE
               for ks in self._kind.values()):
            return DistillerMode.APPRENTICE
        return DistillerMode.OFF

    @property
    def is_enabled(self) -> bool:
        """Backward-compatible: True once any kind reaches EXPERT."""
        return self.mode == DistillerMode.EXPERT

    def mode_for(self, kind: TargetKind) -> DistillerMode:
        """Mode for a specific target kind."""
        return self._kind[kind].mode

    def n_eval_for(self, kind: TargetKind) -> int:
        """Number of eval samples for a specific kind."""
        return self._kind[kind].n_eval

    def n_total_for(self, kind: TargetKind) -> int:
        """Total samples collected for a specific kind."""
        return self._kind[kind].n_total

    def trust_for(self, kind: TargetKind) -> float:
        """EMA-smoothed trust for a specific kind."""
        return self._kind[kind].eval_mean_trust

    def trust_threshold_for(self, kind: TargetKind) -> float:
        """Trust reference value per kind (REAL: hard gate, FRONTIER: score weight)."""
        if (kind == TargetKind.FRONTIER
                and self._trust_threshold_frontier is not None):
            return self._trust_threshold_frontier
        return self.trust_threshold

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
        target_kind: TargetKind = TargetKind.REAL,
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
            target_kind=target_kind,
        )
        self._episode_buffer.append(sample)
        self._kind[target_kind].n_total += 1

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def end_episode(self, success: bool) -> None:
        """Called at episode end.

        Flush policy (per target kind):
          - FRONTIER: always flush (exploration is the dominant learning
            channel; fail-episodes contain valuable exploration steps)
          - REAL: only flush on success (cleaner navigation signal)

        Training is decoupled from success: triggers every train_interval
        total episodes once replay >= min_samples.
        """
        self._episode_count += 1

        if self._episode_buffer:
            flushed = 0
            for sample in self._episode_buffer:
                # FRONTIER: always flush.  REAL: only on success.
                if sample.target_kind == TargetKind.FRONTIER or success:
                    if random.random() < EVAL_HOLDOUT:
                        self._eval_buffer.append(sample)
                    else:
                        self._train_buffer.append(sample)
                    flushed += 1
            self._total_samples_collected += flushed
            if success:
                self._successful_episodes += 1

        self._episode_buffer.clear()

        # Training schedule: decoupled from success rate
        total_replay = len(self._train_buffer) + len(self._eval_buffer)
        if total_replay < self.min_samples:
            return  # not enough data yet

        # Warm-start: more gradient steps on first training
        if not self._has_trained_once:
            self.train(epochs=self.warm_start_epochs)
            self._has_trained_once = True
            return

        # Regular: faster interval once any kind is EXPERT
        interval = (self.train_interval_post_enable
                    if self.mode == DistillerMode.EXPERT
                    else self.train_interval)
        if self._episode_count % interval == 0:
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

        Per-kind eval metrics (EMA-smoothed):
        - eval_accuracy, eval_mean_conf, eval_mean_trust

        Mode transitions (per kind):
        - OFF -> APPRENTICE: train_count >= 1 AND kind has samples
        - APPRENTICE -> EXPERT: per-kind readiness criteria met

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
            # Build kind tensor for masking
            eval_kind = [s.target_kind for s in eval_samples]

            with torch.no_grad():
                eval_logits = self.net(eval_features)
                eval_preds = eval_logits.argmax(dim=-1)
                eval_probs = torch.softmax(eval_logits, dim=-1)
                eval_conf = eval_probs.max(dim=-1).values

                # Trust signal (margin + entropy, better calibrated than softmax)
                eval_trust = self._trust_primitive.compute_trust_batch(
                    eval_logits)

                # Overall metrics (legacy, for display/CSV)
                raw_acc = (eval_preds == eval_labels).float().mean().item()
                raw_conf = eval_conf.mean().item()
                raw_trust = eval_trust.mean().item()
                self._eval_accuracy = self._ema(
                    self._eval_accuracy, raw_acc)
                self._eval_mean_confidence = self._ema(
                    self._eval_mean_confidence, raw_conf)
                self._eval_mean_trust = self._ema(
                    self._eval_mean_trust, raw_trust)

                # Per-kind eval metrics
                for kind in TargetKind:
                    mask = torch.tensor(
                        [k == kind for k in eval_kind], dtype=torch.bool)
                    ks = self._kind[kind]
                    ks.n_eval = int(mask.sum().item())
                    if ks.n_eval >= 5:
                        raw_acc_k = (
                            (eval_preds[mask] == eval_labels[mask])
                            .float().mean().item()
                        )
                        raw_conf_k = eval_conf[mask].mean().item()
                        raw_trust_k = eval_trust[mask].mean().item()
                        ks.eval_accuracy = self._ema(
                            ks.eval_accuracy, raw_acc_k)
                        ks.eval_mean_conf = self._ema(
                            ks.eval_mean_conf, raw_conf_k)
                        ks.eval_mean_trust = self._ema(
                            ks.eval_mean_trust, raw_trust_k)
                    # else: n_eval < 5 → leave ks.eval_* unchanged
                    # Promotion checks n_eval >= min_targeted_eval anyway.
        else:
            # Fallback to train accuracy if not enough eval data
            self._eval_accuracy = self._train_accuracy
            self._eval_mean_confidence = 0.0
            self._eval_mean_trust = 0.0
            for ks in self._kind.values():
                ks.n_eval = 0

        # Per-kind mode transitions
        total_replay = len(self._train_buffer) + len(self._eval_buffer)
        for kind in TargetKind:
            ks = self._kind[kind]

            # OFF -> APPRENTICE: trained at least once AND this kind has data
            if (ks.mode == DistillerMode.OFF
                    and self._train_count >= 1
                    and ks.n_total > 0):
                ks.mode = DistillerMode.APPRENTICE

            # APPRENTICE -> EXPERT: per-kind readiness
            if ks.mode == DistillerMode.APPRENTICE:
                if total_replay >= self.min_samples:
                    if kind == TargetKind.REAL:
                        if (ks.n_total >= self.min_targeted_for_enable
                                and ks.n_eval >= self.min_targeted_eval
                                and ks.eval_accuracy
                                >= self.min_accuracy_targeted):
                            ks.mode = DistillerMode.EXPERT
                    elif kind == TargetKind.FRONTIER:
                        if (ks.n_eval >= self.min_targeted_eval
                                and ks.eval_accuracy
                                >= self.min_accuracy_frontier):
                            ks.mode = DistillerMode.EXPERT

        return self._eval_accuracy

    # ------------------------------------------------------------------
    # Inference (called from kernel._compressed_l2_action)
    # ------------------------------------------------------------------

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
    ) -> Tuple[int, float, float]:
        """Predict desired direction + confidence + trust.

        Returns (direction_class, confidence, trust) where direction is 0-3,
        confidence is the softmax probability (legacy), and trust is the
        margin+entropy based signal from TrustPrimitive.
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
        return self.net.predict_direction_with_trust(
            features, self._trust_primitive)

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
        """Eval accuracy on REAL (targeted) samples."""
        return self._kind[TargetKind.REAL].eval_accuracy

    @property
    def accuracy_frontier(self) -> float:
        """Eval accuracy on FRONTIER samples."""
        return self._kind[TargetKind.FRONTIER].eval_accuracy

    @property
    def mean_confidence(self) -> float:
        """Last measured mean prediction confidence on eval set."""
        return self._eval_mean_confidence

    @property
    def mean_conf_targeted(self) -> float:
        """Mean confidence on REAL eval samples."""
        return self._kind[TargetKind.REAL].eval_mean_conf

    @property
    def mean_conf_frontier(self) -> float:
        """Mean confidence on FRONTIER eval samples."""
        return self._kind[TargetKind.FRONTIER].eval_mean_conf

    @property
    def mean_trust(self) -> float:
        """Last measured mean trust (margin+entropy) on eval set."""
        return self._eval_mean_trust

    @property
    def mean_trust_targeted(self) -> float:
        """Mean trust on REAL eval samples."""
        return self._kind[TargetKind.REAL].eval_mean_trust

    @property
    def mean_trust_frontier(self) -> float:
        """Mean trust on FRONTIER eval samples."""
        return self._kind[TargetKind.FRONTIER].eval_mean_trust

    @property
    def n_targeted_total(self) -> int:
        """Total REAL (targeted) samples seen (counted at collection)."""
        return self._kind[TargetKind.REAL].n_total

    @property
    def n_targeted_eval(self) -> int:
        """REAL samples currently in eval buffer."""
        return self._kind[TargetKind.REAL].n_eval

    @property
    def n_frontier_total(self) -> int:
        """Total FRONTIER samples seen."""
        return self._kind[TargetKind.FRONTIER].n_total

    @property
    def train_count(self) -> int:
        """Number of train() calls so far."""
        return self._train_count

    @property
    def episode_samples(self) -> int:
        """Number of samples collected in the current episode so far."""
        return len(self._episode_buffer)

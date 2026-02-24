"""MetaControllerAgentD: Neural meta-controller wrapper for Stream D (Phase 5c).

Wraps any deterministic StreamD implementation (e.g. EventPatternD) and attaches
a MetaControllerLearnerD that learns D's two core decisions:
  1. Phase selection (find_key / open_door / reach_goal)
  2. Confidence calibration ([0, 1])

While TRAINING: delegates report_meaning() to inner deterministic D.
When READY: neural override for phase + confidence in MeaningReport.

Event detection stays deterministic (inner's observe_step).
Tag generation stays deterministic (inner's build/build_micro).
Only the abstract decisions (phase + confidence) become neural.

d_term from ClosureResiduum is D's natural feedback signal:
  d_term = 0.0  -> D identified target correctly, no hallucinations
  d_term = 0.8  -> D produced nothing useful
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.interfaces import StreamD, StreamLearner
from kernel.types import LearnerMode, LearnerSignal, LearnerStatus
from models.meta_controller_nets import (
    MetaControllerNet,
    MetaCriticNet,
    META_INPUT_DIM,
    META_N_PHASES,
    PHASE_NAMES,
    PHASE_TO_IDX,
)
from state.schema import ZA, ZD


class MetaControllerAgentD(StreamD):
    """Stream D with neural meta-controller for phase selection + confidence.

    Wraps a deterministic StreamD and attaches a MetaControllerLearnerD.
    report_meaning() delegates to inner during TRAINING.
    When READY (or forced after warmup), overrides phase + confidence in MeaningReport.
    Event detection and tag generation always stay deterministic.
    """

    def __init__(
        self,
        inner: Optional[StreamD] = None,
        *,
        use_neural: bool = True,
        force_neural_after_warmup: bool = False,
        warmup_episodes: int = 30,
        ready_threshold: float = 0.80,
        ready_window: int = 20,
        replay_max: int = 3000,
        batch_size: int = 32,
        lr: float = 3e-4,
        train_epochs: int = 2,
    ):
        if inner is None:
            from agents.event_pattern_d import EventPatternD
            inner = EventPatternD()
        self._inner = inner
        self._use_neural = use_neural
        self._force_neural_after_warmup = force_neural_after_warmup

        # StreamD requires events and seen_positions attributes
        self.events = self._inner.events
        self.seen_positions = self._inner.seen_positions

        # Meta-Controller learner
        self._learner_impl = MetaControllerLearnerD(
            agent=self,
            warmup_episodes=warmup_episodes,
            ready_threshold=ready_threshold,
            ready_window=ready_window,
            replay_max=replay_max,
            batch_size=batch_size,
            lr=lr,
            train_epochs=train_epochs,
        )

        # Cached neural output (for learner transition tracking)
        self._cached_features: Optional[torch.Tensor] = None
        self._cached_phase_idx: Optional[int] = None
        self._cached_confidence: Optional[float] = None

    # ------------------------------------------------------------------
    # StreamD interface — delegation to inner
    # ------------------------------------------------------------------

    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        """Delegate event detection to inner. Cache features for learner."""
        self._inner.observe_step(t, zA, action, reward, done)
        # Sync events/seen_positions references (inner may have rebuilt lists)
        self.events = self._inner.events
        self.seen_positions = self._inner.seen_positions

        # Cache features for transition tracking
        self._cached_features = self._extract_features()
        if self._cached_features is not None:
            with torch.no_grad():
                out = self._learner_impl._net(
                    self._cached_features.unsqueeze(0)).squeeze(0)
                self._cached_phase_idx = out[:3].argmax().item()
                self._cached_confidence = torch.sigmoid(out[6]).item()

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Delegate tag generation to inner (stays deterministic)."""
        return self._inner.build(goal_mode, goal_pos)

    def build_micro(
        self,
        goal_mode: str,
        goal_pos=None,
        last_n: int = 5,
    ) -> ZD:
        """Delegate micro narrative to inner (stays deterministic)."""
        return self._inner.build_micro(goal_mode, goal_pos, last_n)

    def report_meaning(self) -> "MeaningReport":
        """Neural override for phase + confidence when READY.

        Inner's MeaningReport is the base: events, tags, grounding stay
        deterministic. Only suggested_phase, suggested_target, and
        confidence are overridden by the neural meta-controller.
        """
        from kernel.types import MeaningReport

        base = self._inner.report_meaning()

        # Gate: use neural when READY, or when forced after warmup
        can_use_neural = (
            self._use_neural
            and (self._learner_impl.mode == LearnerMode.READY
                 or (self._force_neural_after_warmup
                     and self._learner_impl._episodes_trained
                         >= self._learner_impl._warmup_episodes))
        )
        if not can_use_neural:
            return base

        # Neural override
        features = self._extract_features()
        if features is None:
            return base

        with torch.no_grad():
            out = self._learner_impl._net(features.unsqueeze(0)).squeeze(0)
            phase_logits = out[:3]
            confidence_logit = out[6]

        phase_idx = phase_logits.argmax().item()
        confidence = torch.sigmoid(confidence_logit).item()
        suggested_phase = PHASE_NAMES[phase_idx]
        suggested_target = self._target_for_phase(phase_idx)

        return MeaningReport(
            confidence=confidence,
            suggested_target=suggested_target,
            suggested_phase=suggested_phase,
            events_detected=base.events_detected,
            hypothesis_strength=base.hypothesis_strength,
            narrative_tags=base.narrative_tags,
            grounding_violations=base.grounding_violations,
            grounding_score=base.grounding_score,
            narrative_length=base.narrative_length,
        )

    @property
    def learner(self) -> StreamLearner:
        return self._learner_impl

    # ------------------------------------------------------------------
    # EventPatternD compatibility — forward to inner
    # ------------------------------------------------------------------

    def set_object_memory(self, obj_mem) -> None:
        if hasattr(self._inner, "set_object_memory"):
            self._inner.set_object_memory(obj_mem)

    def reset_episode(self) -> None:
        self._inner.reset_episode()
        self.events = self._inner.events
        self.seen_positions = self._inner.seen_positions
        self._cached_features = None
        self._cached_phase_idx = None
        self._cached_confidence = None

    def end_episode(self, success: bool, steps: int) -> None:
        if hasattr(self._inner, "end_episode"):
            self._inner.end_episode(success, steps)

    def suggest_target(self) -> Optional[Tuple[int, int]]:
        """Suggest target: neural phase when READY, else deterministic."""
        if (self._use_neural
                and self._learner_impl.mode == LearnerMode.READY
                and self._cached_phase_idx is not None):
            return self._target_for_phase(self._cached_phase_idx)
        if hasattr(self._inner, "suggest_target"):
            return self._inner.suggest_target()
        return None

    def reflect(self) -> None:
        if hasattr(self._inner, "reflect"):
            self._inner.reflect()

    @property
    def success_sequence(self):
        return getattr(self._inner, "success_sequence", None)

    @property
    def partial_hypotheses(self):
        return getattr(self._inner, "partial_hypotheses", [])

    @property
    def negative_constraints(self):
        return getattr(self._inner, "negative_constraints", [])

    @property
    def has_hypothesis(self) -> bool:
        return getattr(self._inner, "has_hypothesis", False)

    def current_sequence_step(self) -> int:
        if hasattr(self._inner, "current_sequence_step"):
            return self._inner.current_sequence_step()
        return 0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self) -> Optional[torch.Tensor]:
        """Extract 20-dim abstract state for the meta-controller.

        Features:
          [0:5]   Object knowledge flags (key_known, door_known, goal_known,
                   carrying_key, door_open)
          [5:8]   Current deterministic phase one-hot
          [8:11]  Current neural phase one-hot (or det if not yet active)
          [11:13] Episode progress (normalized step, success rate)
          [13:16] Recent residuum terms (d_term, c_term, grounding_score)
          [16:20] Event history this episode (key, door, blocked, goal)
        """
        om = getattr(self._inner, "_object_memory", None)
        if om is None:
            return None

        features = torch.zeros(META_INPUT_DIM)

        # [0:5] Object knowledge flags
        features[0] = float(om.key_pos is not None)
        features[1] = float(om.door_pos is not None)
        features[2] = float(om.goal_pos is not None)
        features[3] = float(om.carrying_key)
        features[4] = float(om.door_open)

        # [5:8] Current deterministic phase one-hot
        det_step = self.current_sequence_step()
        if 0 <= det_step < 3:
            features[5 + det_step] = 1.0

        # [8:11] Current neural phase one-hot
        if self._cached_phase_idx is not None and 0 <= self._cached_phase_idx < 3:
            features[8 + self._cached_phase_idx] = 1.0
        elif 0 <= det_step < 3:
            features[8 + det_step] = 1.0

        # [11:13] Episode progress
        inner_events = getattr(self._inner, "events", [])
        features[11] = min(float(len(inner_events)) / 200.0, 1.0)
        ep_buffer = getattr(self._inner, "episode_buffer", [])
        successes = sum(1 for ep in ep_buffer if ep.success)
        total = max(len(ep_buffer), 1)
        features[12] = successes / total

        # [13:16] Recent residuum terms
        features[13] = self._learner_impl._prev_d_term
        features[14] = self._learner_impl._prev_c_term
        features[15] = self._learner_impl._prev_grounding

        # [16:20] Event history this episode
        from agents.event_pattern_d import DoorKeyEventType
        ep_events = getattr(self._inner, "_episode_events", [])
        features[16] = float(DoorKeyEventType.KEY_PICKED_UP in ep_events)
        features[17] = float(DoorKeyEventType.DOOR_OPENED in ep_events)
        blocked = sum(
            1 for e in ep_events
            if e == DoorKeyEventType.BLOCKED_AT_DOOR)
        features[18] = min(blocked / 3.0, 1.0)
        features[19] = float(DoorKeyEventType.GOAL_REACHED in ep_events)

        return features

    def _target_for_phase(
        self, phase_idx: int
    ) -> Optional[Tuple[int, int]]:
        """Map neural phase index to navigation target from ObjectMemory."""
        om = getattr(self._inner, "_object_memory", None)
        if om is None:
            return None
        if phase_idx == 0:  # find_key
            return om.key_pos
        elif phase_idx == 1:  # open_door
            return om.door_pos
        elif phase_idx == 2:  # reach_goal
            return om.goal_pos
        return None


class MetaControllerLearnerD(StreamLearner):
    """StreamLearner implementation for Meta-Controller on Stream D.

    Lifecycle:
      - observe_signal(): records d_term, tracks phase/confidence agreement
      - learn(): IL warmstart (< warmup_episodes) or REINFORCE (after)
      - ready(): reports TRAINING/READY based on combined accuracy
      - reset_episode(): clears per-episode state

    No phase cascade: D has no dependency on B or C for learning.
    """

    def __init__(
        self,
        agent: "MetaControllerAgentD",
        *,
        warmup_episodes: int = 30,
        ready_threshold: float = 0.80,
        ready_window: int = 20,
        replay_max: int = 3000,
        batch_size: int = 32,
        lr: float = 3e-4,
        train_epochs: int = 2,
    ):
        self._agent = agent

        # Networks
        self._net = MetaControllerNet()
        self._critic = MetaCriticNet()

        # Single optimizer for all params
        self._optimizer = torch.optim.Adam(
            list(self._net.parameters()) + list(self._critic.parameters()),
            lr=lr,
        )

        # Hyperparameters
        self._batch_size = batch_size
        self._train_epochs = train_epochs
        self._warmup_episodes = warmup_episodes
        self._ready_threshold = ready_threshold

        # Replay buffer: (features, phase_idx, det_phase_idx,
        #                  confidence, det_confidence, d_term, done)
        self._replay: deque = deque(maxlen=replay_max)
        self._episode_buffer: List[Tuple] = []

        # Accuracy tracking
        self._eval_window: deque = deque(maxlen=ready_window)
        self._episodes_trained: int = 0
        self._last_accuracy: float = 0.0

        # Mode
        self._mode = (LearnerMode.TRAINING
                      if agent._use_neural else LearnerMode.OFF)

        # Per-tick residuum tracking (from observe_signal)
        self._prev_d_term: float = 0.5
        self._prev_c_term: float = 0.5
        self._prev_grounding: float = 1.0

        # Per-episode agreement tracking
        self._episode_phase_ticks: int = 0
        self._episode_phase_matches: int = 0
        self._episode_conf_ticks: int = 0
        self._episode_conf_matches: int = 0

    @property
    def mode(self) -> LearnerMode:
        return self._mode

    # ------------------------------------------------------------------
    # StreamLearner interface
    # ------------------------------------------------------------------

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Record transition and track phase/confidence agreement.

        Called every tick by the kernel. Records d_term for reward,
        deterministic phase for IL labels, and tracks agreement metrics.
        """
        # Update residuum tracking
        self._prev_d_term = signal.d_term if signal.d_term is not None else 0.5
        self._prev_c_term = signal.c_term if signal.c_term is not None else 0.5
        self._prev_grounding = (signal.grounding_score
                                if signal.grounding_score is not None else 1.0)

        # Get cached features and neural output from agent
        features = self._agent._cached_features
        neural_phase = self._agent._cached_phase_idx
        neural_conf = self._agent._cached_confidence

        if features is None:
            return

        # Deterministic phase (from inner)
        det_step = self._agent.current_sequence_step()

        # Deterministic confidence (from inner's report_meaning pattern)
        inner = self._agent._inner
        if getattr(inner, "success_sequence", None) is not None:
            det_conf = 1.0
        elif getattr(inner, "partial_hypotheses", []):
            det_conf = 0.5
        else:
            det_conf = 0.0

        # Track agreement
        if neural_phase is not None:
            self._episode_phase_ticks += 1
            if neural_phase == det_step:
                self._episode_phase_matches += 1

        if neural_conf is not None:
            self._episode_conf_ticks += 1
            if abs(neural_conf - det_conf) < 0.2:
                self._episode_conf_matches += 1

        # Build transition
        d_term = signal.d_term if signal.d_term is not None else 0.5
        done_flag = 1.0 if signal.done else 0.0
        self._episode_buffer.append((
            features.clone(),
            neural_phase if neural_phase is not None else det_step,
            det_step,
            neural_conf if neural_conf is not None else det_conf,
            det_conf,
            d_term,
            done_flag,
        ))

    def learn(self) -> None:
        """Train at episode end. IL warmstart then REINFORCE."""
        # Flush episode buffer to replay
        self._replay.extend(self._episode_buffer)
        self._episode_buffer.clear()

        # Track episode accuracy
        if self._episode_phase_ticks > 0:
            phase_acc = (self._episode_phase_matches
                         / self._episode_phase_ticks)
            conf_acc = (self._episode_conf_matches
                        / max(self._episode_conf_ticks, 1))
            combined_acc = 0.7 * phase_acc + 0.3 * conf_acc
            self._eval_window.append(combined_acc)
            self._last_accuracy = combined_acc

        # Don't train if OFF or not enough data
        # (skip OFF-check when force_neural is active — keep training)
        if (self._mode == LearnerMode.OFF
                and not self._agent._force_neural_after_warmup):
            self._episodes_trained += 1
            return
        if len(self._replay) < self._batch_size:
            self._episodes_trained += 1
            return

        # IL warmstart or REINFORCE
        for _ in range(self._train_epochs):
            if self._episodes_trained < self._warmup_episodes:
                self._il_warmstart_train()
            else:
                self._reinforce_train()

        self._episodes_trained += 1
        self._update_mode()

    def ready(self) -> LearnerStatus:
        return LearnerStatus(
            mode=self._mode,
            accuracy=self._last_accuracy,
            episodes_trained=self._episodes_trained,
            label=f"Meta-D(acc={self._last_accuracy:.0%})",
        )

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._episode_phase_ticks = 0
        self._episode_phase_matches = 0
        self._episode_conf_ticks = 0
        self._episode_conf_matches = 0
        self._episode_buffer.clear()

    # ------------------------------------------------------------------
    # IL Warmstart
    # ------------------------------------------------------------------

    def _il_warmstart_train(self) -> None:
        """Cross-entropy for phase + MSE for confidence from deterministic D.

        Teacher labels come from EventPatternD's current_sequence_step()
        and hardcoded confidence logic.
        """
        batch_size = min(len(self._replay), self._batch_size)
        batch = random.sample(list(self._replay), batch_size)

        features = torch.stack([t[0] for t in batch])
        det_phases = torch.tensor(
            [t[2] for t in batch], dtype=torch.long)
        det_confs = torch.tensor(
            [t[4] for t in batch], dtype=torch.float32)

        out = self._net(features)
        phase_logits = out[:, :3]
        conf_logits = out[:, 6]

        # Phase: cross-entropy against teacher
        phase_loss = F.cross_entropy(phase_logits, det_phases)

        # Confidence: MSE against deterministic
        conf_pred = torch.sigmoid(conf_logits)
        conf_loss = F.mse_loss(conf_pred, det_confs)

        loss = phase_loss + 0.5 * conf_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    # ------------------------------------------------------------------
    # REINFORCE Training
    # ------------------------------------------------------------------

    def _reinforce_train(self) -> None:
        """REINFORCE with baseline. Reward = -d_term (from ClosureResiduum).

        Policy gradient for phase selection, MSE for confidence calibration,
        MSE for state-value critic baseline.
        """
        batch_size = min(len(self._replay), self._batch_size)
        batch = random.sample(list(self._replay), batch_size)

        features = torch.stack([t[0] for t in batch])
        phase_actions = torch.tensor(
            [t[1] for t in batch], dtype=torch.long)
        d_terms = torch.tensor(
            [t[5] for t in batch], dtype=torch.float32)

        # Reward: lower d_term is better
        rewards = -d_terms

        # Baseline
        with torch.no_grad():
            baseline = self._critic(features).squeeze(-1)

        advantage = rewards - baseline

        # Policy loss (REINFORCE)
        out = self._net(features)
        phase_logits = out[:, :3]
        log_probs = F.log_softmax(phase_logits, dim=-1)
        selected_log_probs = log_probs.gather(
            1, phase_actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(selected_log_probs * advantage.detach()).mean()

        # Confidence loss: MSE toward ideal confidence
        # Ideal: confidence = 1.0 - d_term
        conf_logits = out[:, 6]
        conf_pred = torch.sigmoid(conf_logits)
        ideal_conf = (1.0 - d_terms).clamp(0.0, 1.0)
        conf_loss = F.mse_loss(conf_pred, ideal_conf)

        # Critic loss (value baseline)
        v_pred = self._critic(features).squeeze(-1)
        critic_loss = F.mse_loss(v_pred, rewards)

        # Combined
        total_loss = policy_loss + 0.3 * conf_loss + 0.5 * critic_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

    # ------------------------------------------------------------------
    # Mode Management
    # ------------------------------------------------------------------

    def _update_mode(self) -> None:
        """Transition TRAINING -> READY based on combined accuracy.

        Combined = 0.7 * phase_agreement + 0.3 * confidence_agreement
        Threshold: 80% over 20 episodes.
        If use_neural=False, stays OFF.
        """
        if not self._agent._use_neural:
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

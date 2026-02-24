"""NeuralMetaD — Standalone recurrent meta-controller for RAPA-N.

GRU-based D stream that processes step history (embedding + action + reward)
to predict task phase and confidence. Learns via REINFORCE with d_term
from the closure residuum as reward signal.

No EventPatternD, no ObjectMemory, no deterministic fallback.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.interfaces import StreamD, StreamLearner
from kernel.types import (
    LearnerSignal, LearnerStatus, LearnerMode, MeaningReport,
)
from state.schema import ZA, ZD
from models.rapa_n_nets import (
    MetaGRUNet, action_to_onehot,
    ACTION_NAMES, N_ACTIONS, EMBEDDING_DIM, META_SEQ_LEN,
    N_PHASES, PHASE_NAMES,
)


class NeuralMetaD(StreamD):
    """GRU meta-controller: step history → phase + confidence.

    Maintains a rolling history of (embedding, action, reward) per step.
    report_meaning() runs the GRU to produce phase prediction and confidence.
    """

    def __init__(self):
        self._net = MetaGRUNet()
        self._learner_inst = MetaLearnerD(self._net)

        # Step history (rolling window)
        self._history: List[Tuple[List[float], int, float]] = []

        # Required by kernel (hasattr checks)
        self.events: List = []
        self.seen_positions: set = set()

        # GRU hidden state (persistent within episode)
        self._gru_hidden: Optional[torch.Tensor] = None

    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        """Record step for GRU processing."""
        emb = zA.embedding if zA.embedding is not None else [0.0] * EMBEDDING_DIM
        act_idx = ACTION_NAMES.index(action) if action in ACTION_NAMES else 2
        self._history.append((emb, act_idx, reward))

        # Also record for learner's replay-forward
        self._learner_inst.add_step(emb, act_idx, reward)

        # Track seen positions for kernel
        self.seen_positions.add(zA.agent_pos)

        if done:
            self._gru_hidden = None

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Build full narrative — minimal for neural D."""
        return ZD(
            narrative="neural-meta",
            meaning_tags=[],
            length_chars=len(self._history),
            grounding_violations=0,
        )

    def build_micro(
        self,
        goal_mode: str,
        goal_pos=None,
        last_n: int = 5,
    ) -> ZD:
        """Build micro-narrative — minimal for neural D."""
        return ZD(
            narrative="neural-meta-micro",
            meaning_tags=[],
            length_chars=min(last_n, len(self._history)),
            grounding_violations=0,
        )

    def report_meaning(self) -> MeaningReport:
        """Run GRU over recent history to produce phase + confidence."""
        if len(self._history) < 2:
            return MeaningReport(
                confidence=0.0,
                grounding_score=0.0,
            )

        # Prepare GRU input: last META_SEQ_LEN steps
        recent = self._history[-META_SEQ_LEN:]
        seq = []
        for emb, act_idx, reward in recent:
            emb_t = torch.tensor(emb, dtype=torch.float32)
            act_oh = torch.zeros(N_ACTIONS)
            act_oh[act_idx] = 1.0
            rew_t = torch.tensor([reward], dtype=torch.float32)
            step_vec = torch.cat([emb_t, act_oh, rew_t])  # (70,)
            seq.append(step_vec)

        seq_tensor = torch.stack(seq).unsqueeze(0)  # (1, seq_len, 70)

        with torch.no_grad():
            phase_logits, conf_logit, self._gru_hidden = self._net(
                seq_tensor, self._gru_hidden
            )

        phase_idx = phase_logits.squeeze(0).argmax().item()
        confidence = torch.sigmoid(conf_logit.squeeze()).item()

        # Cache for learner (REINFORCE)
        self._learner_inst.cache_prediction(phase_logits.squeeze(0), conf_logit.squeeze())

        return MeaningReport(
            confidence=confidence,
            suggested_phase=PHASE_NAMES[phase_idx],
            suggested_target=None,
            events_detected=[],
            grounding_score=confidence,
            narrative_length=len(self._history),
        )

    @property
    def learner(self) -> StreamLearner:
        return self._learner_inst

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._history.clear()
        self.events.clear()
        self.seen_positions.clear()
        self._gru_hidden = None


class MetaLearnerD(StreamLearner):
    """REINFORCE learner for the GRU meta-controller.

    Reward signal: -d_term from closure residuum (lower = better).
    Uses a replay-forward approach: stores episode step data, then
    re-runs the GRU with gradients enabled during learn().
    """

    def __init__(self, net: MetaGRUNet):
        self._net = net
        self._optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        # Episode step data for replay-forward in learn()
        self._episode_steps: List[Tuple[List[float], int, float]] = []
        self._episode_d_terms: List[float] = []
        # Phase choices (indices) made during report_meaning()
        self._episode_phase_choices: List[int] = []

        # Tracking
        self._episodes_trained: int = 0
        self._recent_confidence: deque = deque(maxlen=20)
        self._baseline: float = 0.0  # Running baseline for variance reduction

    def cache_prediction(
        self,
        phase_logits: torch.Tensor,
        conf_logit: torch.Tensor,
    ) -> None:
        """Cache prediction metadata (phase choice, confidence)."""
        phase_idx = phase_logits.argmax().item()
        self._episode_phase_choices.append(phase_idx)

        confidence = torch.sigmoid(conf_logit).item()
        self._recent_confidence.append(confidence)

    def add_step(self, emb: List[float], act_idx: int, reward: float) -> None:
        """Record a step for replay-forward during learn()."""
        self._episode_steps.append((emb, act_idx, reward))

    def observe_signal(self, signal: LearnerSignal) -> None:
        """Accumulate d_term rewards for REINFORCE."""
        self._episode_d_terms.append(signal.d_term)

    def learn(self) -> None:
        """REINFORCE update via replay-forward at episode end."""
        self._episodes_trained += 1

        if (len(self._episode_steps) < 2
                or not self._episode_d_terms
                or not self._episode_phase_choices):
            self._episode_steps.clear()
            self._episode_d_terms.clear()
            self._episode_phase_choices.clear()
            return

        # Reward: -d_term (lower residuum = better)
        rewards = [-d for d in self._episode_d_terms]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Update baseline (EMA)
        self._baseline = 0.9 * self._baseline + 0.1 * avg_reward
        advantage = avg_reward - self._baseline

        # Replay-forward: re-run GRU with gradients enabled
        # Build input sequence from stored steps
        seq = []
        for emb, act_idx, rew in self._episode_steps:
            emb_t = torch.tensor(emb, dtype=torch.float32)
            act_oh = torch.zeros(N_ACTIONS)
            act_oh[act_idx] = 1.0
            rew_t = torch.tensor([rew], dtype=torch.float32)
            step_vec = torch.cat([emb_t, act_oh, rew_t])
            seq.append(step_vec)

        seq_tensor = torch.stack(seq).unsqueeze(0)  # (1, seq_len, 70)

        # Forward with gradients
        phase_logits, conf_logit, _ = self._net(seq_tensor, None)
        phase_probs = F.softmax(phase_logits.squeeze(0), dim=-1)

        # Use the last phase choice as the action taken
        if self._episode_phase_choices:
            phase_idx = self._episode_phase_choices[-1]
        else:
            phase_idx = phase_probs.argmax().item()

        log_prob = torch.log(phase_probs[phase_idx] + 1e-8)
        loss = -(log_prob * advantage)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._optimizer.step()

        self._episode_steps.clear()
        self._episode_d_terms.clear()
        self._episode_phase_choices.clear()

    def ready(self) -> LearnerStatus:
        """D is READY when confidence is consistently high."""
        avg_conf = (
            sum(self._recent_confidence) / len(self._recent_confidence)
            if self._recent_confidence else 0.0
        )

        if avg_conf > 0.6 and self._episodes_trained > 30:
            mode = LearnerMode.READY
            label = f"meta-D-ready(conf={avg_conf:.2f})"
        else:
            mode = LearnerMode.TRAINING
            label = f"meta-D-training(conf={avg_conf:.2f})"

        return LearnerStatus(
            mode=mode,
            accuracy=avg_conf,
            episodes_trained=self._episodes_trained,
            label=label,
        )

    def reset_episode(self) -> None:
        """Reset per-episode learning state."""
        self._episode_steps.clear()
        self._episode_d_terms.clear()
        self._episode_phase_choices.clear()

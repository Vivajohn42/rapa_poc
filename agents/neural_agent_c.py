"""Neural Agent C: learned action scoring under kernel governance.

Replaces Manhattan-distance scoring with a learned value network that
approximates BFS-optimal action values.  Uses a hybrid blend:

    combined_score = alpha * neural_score + (1 - alpha) * manhattan_score

This preserves robustness (Manhattan is never wrong by much) while adding
the ability to see around obstacles (where Manhattan fails).

Interface-compatible: returns exactly (str, List[Tuple[str, float]])
so that Loop Gain, Residuum, and ClosureCore see no difference.

Usage:
    net = ActionValueNet()
    net.load_state_dict(torch.load("train/checkpoints/action_value_net.pt"))
    agent_c = NeuralAgentC(goal=GoalSpec(mode="seek", target=(14,14)), value_net=net)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import torch

from state.schema import ZA
from kernel.interfaces import StreamC
from agents.agent_c import GoalSpec
from models.action_value_net import ActionValueNet, extract_features

ACTIONS = ("up", "down", "left", "right")


class NeuralAgentC(StreamC):
    """Goal/Valence agent with learned obstacle-aware scoring.

    Blends a neural score (BFS-approximated) with the Manhattan baseline.
    The neural component learns to see around obstacles via the 7x7 local
    window; the Manhattan component provides a stable gradient everywhere.
    """

    def __init__(
        self,
        goal: GoalSpec,
        value_net: ActionValueNet,
        alpha: float = 0.7,
        anti_stay_penalty: float = 0.25,
        device: str = "cpu",
    ):
        if goal.mode not in ("seek", "avoid"):
            raise ValueError("goal.mode must be 'seek' or 'avoid'")
        self._goal = goal
        self.value_net = value_net
        self.alpha = alpha
        self.anti_stay_penalty = float(anti_stay_penalty)
        self.device = device
        self.value_net.to(device)
        self.value_net.eval()

    @property
    def goal(self) -> GoalSpec:
        return self._goal

    @goal.setter
    def goal(self, value: GoalSpec) -> None:
        self._goal = value

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _score_action(self, zA: ZA, zA_next: ZA) -> float:
        """Combined neural + manhattan scoring."""
        # Manhattan component
        d_now = self._manhattan(zA.agent_pos, self.goal.target)
        d_next = self._manhattan(zA_next.agent_pos, self.goal.target)

        if self.goal.mode == "seek":
            manhattan_score = float(d_now - d_next)
        else:
            manhattan_score = float(d_next - d_now)

        # Neural component
        features = extract_features(
            agent_pos=zA.agent_pos,
            next_pos=zA_next.agent_pos,
            goal_pos=self.goal.target,
            obstacles=zA.obstacles,
            width=zA.width,
            height=zA.height,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            neural_score = self.value_net(features).item()

        # Flip neural score for avoid mode (trained on seek)
        if self.goal.mode == "avoid":
            neural_score = -neural_score

        # Blend
        combined = self.alpha * neural_score + (1.0 - self.alpha) * manhattan_score

        # Anti-stay penalty
        if zA_next.agent_pos == zA.agent_pos:
            combined -= self.anti_stay_penalty

        return combined

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn,
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Score all actions and return (best_action, scored_list).

        Identical interface to deterministic AgentC.
        """
        scored: List[Tuple[str, float]] = []
        for a in ACTIONS:
            zA_next = predict_next_fn(zA, a)
            s = self._score_action(zA, zA_next)
            scored.append((a, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Uncertainty margin
        if len(scored) >= 2:
            delta = scored[0][1] - scored[1][1]
        else:
            delta = 999.0

        # Tie-break using memory preference (identical to det C)
        if memory and delta < tie_break_delta:
            pref = memory.get("tie_break_preference")
            if isinstance(pref, list) and pref:
                best = scored[0][1]
                near_best = [a for a, s in scored if (best - s) <= tie_break_delta]

                for p in pref:
                    if p in near_best:
                        return p, scored

        return scored[0][0], scored

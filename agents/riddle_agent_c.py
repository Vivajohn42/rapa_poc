"""RiddleAgentC: Valence/goal scoring for Riddle Rooms.

Scoring logic:
  - test actions: score positively (information gathering) when no target known
  - submit_{answer}: score 2.0 if target in memory and action matches,
    else score negatively (prevent premature guessing)
  - When D has identified the answer (target in memory), submit is dominant

Without D: all submits score 0.0, tests score slightly positive -> random guessing.
With D: submit_{correct_answer} scores 2.0 -> deterministic correct submission.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any

from state.schema import ZA
from kernel.interfaces import StreamC


class _RiddleGoalProxy:
    """Proxy so kernel can write agent_c.goal.target = (answer_idx, 0)."""

    def __init__(self, agent: "RiddleAgentC"):
        self._agent = agent
        self._target: Optional[Tuple[int, int]] = None

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        return self._target

    @target.setter
    def target(self, value: Tuple[int, int]) -> None:
        self._target = value
        # Map pseudo-position back to answer_id
        answer = self._agent.index_to_answer.get(value[0])
        self._agent.target_answer = answer


class RiddleAgentC(StreamC):
    """Action scorer for logic puzzles."""

    def __init__(
        self,
        answer_set: List[str],
        answer_index: Dict[str, int],
        test_names: List[str],
    ):
        self.answer_set = answer_set
        self.answer_index = answer_index
        self.index_to_answer = {i: a for a, i in answer_index.items()}
        self.test_names = test_names
        self.target_answer: Optional[str] = None
        self._goal = _RiddleGoalProxy(self)

    @property
    def goal(self) -> _RiddleGoalProxy:
        return self._goal

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        scored: List[Tuple[str, float]] = []
        memory = memory or {}

        target_answer = self.target_answer
        revealed = memory.get("revealed_tests", set())

        # Score test actions
        for test_name in self.test_names:
            if test_name in revealed:
                # Already revealed — low score
                scored.append((test_name, -0.5))
            elif target_answer is not None:
                # We already know the answer — no need to test
                scored.append((test_name, -0.3))
            else:
                # Unrevealed test — positive (information gathering)
                scored.append((test_name, 0.5))

        # Score submit actions
        for ans in self.answer_set:
            action_name = f"submit_{ans}"
            if target_answer is not None and ans == target_answer:
                # D identified this as the answer — high confidence submit
                scored.append((action_name, 2.0))
            elif target_answer is not None:
                # Wrong answer after D has spoken
                scored.append((action_name, -1.0))
            else:
                # No target — don't guess
                scored.append((action_name, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Tie-break using memory preference
        if memory and len(scored) >= 2:
            delta = scored[0][1] - scored[1][1]
            if delta < tie_break_delta:
                pref = memory.get("tie_break_preference")
                if isinstance(pref, list) and pref:
                    best = scored[0][1]
                    near_best = [a for a, s in scored if (best - s) <= tie_break_delta]
                    for p in pref:
                        if p in near_best:
                            return p, scored

        return scored[0][0], scored

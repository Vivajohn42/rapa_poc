"""RiddleAgentB: Forward model for Riddle Rooms.

Predicts what the state looks like after taking an action.
For test actions: agent_pos changes (new evidence changes the hash).
For submit actions: agent_pos stays (binary outcome, no state change).
"""
from __future__ import annotations

from typing import Dict, List, Set

from state.schema import ZA
from kernel.interfaces import StreamB


class RiddleAgentB(StreamB):
    """Deterministic forward model for riddle puzzle actions."""

    def __init__(
        self,
        test_names: List[str],
        n_answers: int,
    ):
        self.test_names = test_names
        self.n_answers = n_answers

    def predict_next(self, zA: ZA, action: str) -> ZA:
        # Test actions change the evidence state (pseudo-position shifts)
        if action in self.test_names:
            # Predict a different state hash (simple: increment mod n_answers)
            new_x = (zA.agent_pos[0] + 1) % self.n_answers
            return ZA(
                width=zA.width,
                height=zA.height,
                agent_pos=(new_x, 0),
                goal_pos=zA.goal_pos,
                obstacles=[],
                hint=None,  # clue only appears in actual observation
            )

        # Submit actions don't change the evidence state
        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=zA.agent_pos,
            goal_pos=zA.goal_pos,
            obstacles=[],
            hint=None,
        )

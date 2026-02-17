"""Riddle Rooms environment adapter for rapa_mvp.

Wraps the RiddleRooms environment and provides standardized agent
construction, deconstruction wiring, and per-tick metadata injection.
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable

from kernel.interfaces import EnvironmentAdapter, StreamA, StreamB, StreamC, StreamD
from env.riddle_rooms import RiddleRooms
from agents.riddle_agent_a import RiddleAgentA
from agents.riddle_agent_b import RiddleAgentB
from agents.riddle_agent_c import RiddleAgentC
from agents.riddle_agent_d import RiddleAgentD
from router.deconstruct_riddle import deconstruct_riddle_d_to_c


class RiddleRoomsAdapter(EnvironmentAdapter):
    """EnvironmentAdapter for the Riddle Rooms domain."""

    def __init__(
        self,
        seed: int = 42,
        puzzle_id: Optional[str] = None,
    ):
        self.env = RiddleRooms(seed=seed, puzzle_id=puzzle_id)
        self._answer_index = self.env.answer_index

    def reset(self) -> dict:
        return self.env.reset()

    def step(self, action: str) -> Tuple[dict, float, bool]:
        return self.env.step(action)

    def available_actions(self, obs: dict) -> List[str]:
        return obs.get("exits", [])

    def make_agents(
        self,
        variant: str = "with_d",
    ) -> Tuple[StreamA, StreamB, StreamC, Optional[StreamD]]:
        puzzle = self.env.puzzle
        ai = self._answer_index
        n = self.env.n_answers

        A = RiddleAgentA(ai, n)
        B = RiddleAgentB(self.env.all_test_names, n)
        C = RiddleAgentC(
            answer_set=self.env.answer_set,
            answer_index=ai,
            test_names=self.env.all_test_names,
        )

        D: Optional[StreamD] = None
        if variant != "no_d":
            # Build clue_eliminates mapping: clue_text -> answer IDs to eliminate
            clue_eliminates = {}
            for test_name, clue in puzzle.tests.items():
                clue_eliminates[clue.text] = list(clue.eliminates)
            # Also map initial_clue if present (not elimination, just info)
            if puzzle.initial_clue:
                clue_eliminates[puzzle.initial_clue] = []

            D = RiddleAgentD(
                answer_properties=puzzle.answer_properties,
                answer_set=self.env.answer_set,
                answer_index=ai,
                clue_eliminates=clue_eliminates,
            )

        return A, B, C, D

    def get_goal_map(self) -> Optional[Dict[str, Tuple[int, int]]]:
        return self._answer_index

    def get_deconstruct_fn(self) -> Callable:
        ai = self._answer_index

        def fn(zC, zD, goal_map=None):
            return deconstruct_riddle_d_to_c(
                zC, zD, goal_map=goal_map, answer_index=ai,
            )

        return fn

    def inject_obs_metadata(self, kernel, obs: dict) -> None:
        """Track which tests have been revealed."""
        kernel.zC.memory["revealed_tests"] = obs.get("visited", set())

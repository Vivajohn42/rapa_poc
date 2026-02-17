"""TextWorld environment adapter for rapa_mvp.

Wraps the TextWorld environment and provides standardized agent
construction, deconstruction wiring, and per-tick metadata injection.
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable

from kernel.interfaces import EnvironmentAdapter, StreamA, StreamB, StreamC, StreamD
from env.textworld import TextWorld
from agents.text_agent_a import TextAgentA
from agents.text_agent_b import TextAgentB
from agents.text_agent_c import TextAgentC
from agents.text_agent_d import TextAgentD
from router.deconstruct_text import deconstruct_text_d_to_c


class TextWorldAdapter(EnvironmentAdapter):
    """EnvironmentAdapter for the TextWorld (Clue Rooms) domain."""

    def __init__(self, seed: int = 42, scenario_id: Optional[int] = None):
        self.env = TextWorld(seed=seed, scenario_id=scenario_id)
        self._room_index: Optional[Dict[str, int]] = None
        self._index_to_room: Optional[Dict[int, str]] = None

    def reset(self) -> dict:
        obs = self.env.reset()
        room_ids = self.env.room_ids
        self._room_index = {rid: i for i, rid in enumerate(room_ids)}
        self._index_to_room = {i: rid for rid, i in self._room_index.items()}
        return obs

    def step(self, action: str) -> Tuple[dict, float, bool]:
        return self.env.step(action)

    def available_actions(self, obs: dict) -> List[str]:
        return obs.get("exits", []) + ["claim"]

    def make_agents(
        self,
        variant: str = "with_d",
    ) -> Tuple[StreamA, StreamB, StreamC, Optional[StreamD]]:
        assert self._room_index is not None, "Call reset() before make_agents()"
        ri = self._room_index
        itr = self._index_to_room
        rg = self.env.room_graph
        n = len(self.env.room_ids)

        A = TextAgentA(ri, n)
        B = TextAgentB(rg, ri, itr)
        C = TextAgentC(rg, ri, itr, goal_mode="seek")
        D: Optional[StreamD] = None
        if variant != "no_d":
            D = TextAgentD(
                self.env.room_properties,
                self.env.room_ids,
                ri,
            )
        return A, B, C, D

    def get_goal_map(self) -> Optional[Dict[str, Tuple[int, int]]]:
        return self._room_index

    def get_deconstruct_fn(self) -> Callable:
        ri = self._room_index

        def fn(zC, zD, goal_map=None):
            return deconstruct_text_d_to_c(zC, zD, goal_map=goal_map, room_index=ri)

        return fn

    def inject_obs_metadata(self, kernel, obs: dict) -> None:
        """Inject visited rooms set into kernel C-state for exploration."""
        kernel.zC.memory["visited_rooms"] = obs.get("visited", set())

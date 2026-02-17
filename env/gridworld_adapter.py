"""GridWorld environment adapter for rapa_mvp.

Wraps the GridWorld environment and provides standardized agent
construction, deconstruction wiring, and per-tick metadata injection.
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable

from kernel.interfaces import EnvironmentAdapter, StreamA, StreamB, StreamC, StreamD
from env.gridworld import GridWorld
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d import AgentD
from router.deconstruct import deconstruct_d_to_c


class GridWorldAdapter(EnvironmentAdapter):
    """EnvironmentAdapter for the GridWorld domain."""

    def __init__(self, seed: Optional[int] = None, **grid_kwargs):
        self.env = GridWorld(seed=seed, **grid_kwargs)
        self._obs = None  # cached last observation for make_agents

    def reset(self) -> dict:
        self._obs = self.env.reset()
        return self._obs

    def step(self, action: str) -> Tuple[dict, float, bool]:
        obs, reward, done = self.env.step(action)
        self._obs = obs
        return obs, reward, done

    def available_actions(self, obs: dict) -> List[str]:
        return ["up", "down", "left", "right"]

    def make_agents(
        self,
        variant: str = "with_d",
    ) -> Tuple[StreamA, StreamB, StreamC, Optional[StreamD]]:
        A = AgentA()
        B = AgentB()
        zA0 = A.infer_zA(self.env.observe())
        default_target = (zA0.width - 1, zA0.height - 1)
        C = AgentC(
            goal=GoalSpec(mode="seek", target=default_target),
            anti_stay_penalty=1.1,
        )
        D: Optional[StreamD] = AgentD() if variant != "no_d" else None
        return A, B, C, D

    def get_goal_map(self) -> Optional[Dict[str, Tuple[int, int]]]:
        return getattr(self.env, "_goal_map", None)

    def get_deconstruct_fn(self) -> Callable:
        return deconstruct_d_to_c

    def inject_obs_metadata(self, kernel, obs: dict) -> None:
        """Redirect C toward the hint cell when no target is known yet."""
        if "target" not in kernel.zC.memory:
            if hasattr(self.env, "hint_cell") and self.env.hint_cell:
                kernel.agent_c.goal.target = self.env.hint_cell

"""DoorKey environment adapter for rapa_mvp.

Bridges MiniGrid DoorKey and the MvpKernel EnvironmentAdapter interface.
Handles agent construction, deconstruction wiring, and per-tick metadata.

Critical D-essentiality design:
  inject_obs_metadata sets C's phase-specific fields (key_pos, door_pos, etc.)
  for scoring context, but does NOT write kernel.zC.memory["target"].
  Only D's deconstruction pipeline writes target — making D essential.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from agents.doorkey_agent_a import DoorKeyAgentA
from agents.doorkey_agent_b import DoorKeyAgentB
from agents.doorkey_agent_c import DoorKeyAgentC
from agents.doorkey_agent_d import DoorKeyAgentD
from env.doorkey import DOOR_OPEN, DoorKeyEnv
from kernel.interfaces import (
    EnvironmentAdapter, StreamA, StreamB, StreamC, StreamD,
)
from router.deconstruct_doorkey import deconstruct_doorkey_d_to_c


class DoorKeyAdapter(EnvironmentAdapter):
    """EnvironmentAdapter for MiniGrid DoorKey."""

    def __init__(
        self,
        size: int = 6,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.env = DoorKeyEnv(size=size, seed=seed, max_steps=max_steps)
        self._obs = None
        self._agent_b: Optional[DoorKeyAgentB] = None

    def reset(self) -> object:
        self._obs = self.env.reset()
        return self._obs

    def step(self, action: str) -> Tuple[object, float, bool]:
        obs, reward, done = self.env.step(action)
        self._obs = obs
        return obs, reward, done

    def available_actions(self, obs=None) -> List[str]:
        return self.env.available_actions

    def make_agents(
        self,
        variant: str = "with_d",
    ) -> Tuple[StreamA, StreamB, StreamC, Optional[StreamD]]:
        A = DoorKeyAgentA()
        B = DoorKeyAgentB(
            door_pos=self._obs.door_pos if self._obs else None,
            door_open=False,
        )
        self._agent_b = B
        C = DoorKeyAgentC(goal_mode="seek")
        D: Optional[StreamD] = (
            DoorKeyAgentD() if variant != "no_d" else None
        )
        return A, B, C, D

    def get_goal_map(self) -> Optional[Dict[str, Tuple[int, int]]]:
        # DoorKey goal is always at (size-2, size-2)
        g = self.env.size - 2
        return {"goal": (g, g)}

    def get_deconstruct_fn(self) -> Callable:
        return deconstruct_doorkey_d_to_c

    def inject_obs_metadata(self, kernel, obs) -> None:
        """Sync C and B with environment state each tick.

        Sets C's phase-specific fields for scoring context.
        Does NOT write kernel.zC.memory["target"] — only D's
        deconstruction pipeline does that (D-essentiality design).
        """
        if not hasattr(obs, "phase"):
            return

        c = kernel.agent_c
        if isinstance(c, DoorKeyAgentC):
            c.phase = obs.phase
            c.key_pos = obs.key_pos
            c.door_pos = obs.door_pos
            c.carrying_key = obs.carrying_key
            c.door_open = (
                obs.door_state == DOOR_OPEN
                if obs.door_state is not None
                else False
            )

        # Keep B's door model current
        if self._agent_b is not None:
            is_open = (
                obs.door_state == DOOR_OPEN
                if obs.door_state is not None
                else False
            )
            self._agent_b.update_door_state(obs.door_pos, is_open)

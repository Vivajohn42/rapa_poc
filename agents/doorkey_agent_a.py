"""DoorKeyAgentA: Perception stream for MiniGrid DoorKey.

Maps DoorKeyState to ZA. Uses privileged state from the env wrapper.
The direction field is populated for B's rotation-aware forward model.
"""
from __future__ import annotations

from kernel.interfaces import StreamA
from state.schema import ZA


class DoorKeyAgentA(StreamA):
    """Parse DoorKeyState into ZA belief state."""

    def infer_zA(self, obs) -> ZA:
        # obs is a DoorKeyState dataclass
        hint = obs.hint

        # Generate discovery hints when no phase-transition hint present
        if hint is None and obs.key_pos is not None and obs.phase == "FIND_KEY":
            hint = f"key_at:{obs.key_pos[0]}_{obs.key_pos[1]}"
        if hint is None and obs.door_pos is not None and obs.phase == "OPEN_DOOR":
            hint = f"door_at:{obs.door_pos[0]}_{obs.door_pos[1]}"
        if hint is None and obs.goal_pos != (-1, -1) and obs.phase == "REACH_GOAL":
            hint = f"goal_at:{obs.goal_pos[0]}_{obs.goal_pos[1]}"

        return ZA(
            width=obs.width,
            height=obs.height,
            agent_pos=obs.agent_pos,
            goal_pos=obs.goal_pos,
            obstacles=obs.obstacles,
            hint=hint,
            direction=obs.agent_dir,
        )

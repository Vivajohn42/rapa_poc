"""DoorKeyAgentD: Deterministic narrative/meaning stream for DoorKey.

Tracks phase transitions, key/door/goal discovery, and emits meaning
tags that drive deconstruction (D -> C target updates).

Tags emitted:
  - "phase:{find_key|open_door|reach_goal}"
  - "target:{key|door|goal}" — current subgoal type
  - "key_at:{x}_{y}" — key location
  - "door_at:{x}_{y}" — door location
  - "goal_at:{x}_{y}" — goal location
  - "carrying_key" — agent has the key
  - "door_open" — door has been opened
  - "goal:seek" / "goal:avoid"
  - "progress:{0-2}" — completed subgoal count
  - "success" / "timeout"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from kernel.interfaces import StreamD
from state.schema import ZA, ZD


@dataclass
class DoorKeyEvent:
    t: int
    agent_pos: Tuple[int, int]
    agent_dir: int
    action: str
    reward: float
    done: bool
    hint: Optional[str]


class DoorKeyAgentD(StreamD):
    """Deterministic phase-aware narrative for DoorKey."""

    def __init__(self):
        self.events: List[DoorKeyEvent] = []
        self.seen_positions: set = set()
        # Accumulated state from hints/observations
        self._key_pos: Optional[Tuple[int, int]] = None
        self._door_pos: Optional[Tuple[int, int]] = None
        self._goal_pos: Optional[Tuple[int, int]] = None
        self._carrying_key: bool = False
        self._door_opened: bool = False
        self._phase: str = "FIND_KEY"

    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        self.events.append(DoorKeyEvent(
            t=t,
            agent_pos=zA.agent_pos,
            agent_dir=zA.direction or 0,
            action=action,
            reward=reward,
            done=done,
            hint=zA.hint,
        ))
        self.seen_positions.add(zA.agent_pos)

        # Parse hints to accumulate knowledge
        hint = zA.hint
        if hint is None:
            return
        if hint.startswith("key_at:"):
            parts = hint.split(":", 1)[1].split("_")
            self._key_pos = (int(parts[0]), int(parts[1]))
        elif hint.startswith("door_at:"):
            parts = hint.split(":", 1)[1].split("_")
            self._door_pos = (int(parts[0]), int(parts[1]))
        elif hint.startswith("goal_at:"):
            parts = hint.split(":", 1)[1].split("_")
            self._goal_pos = (int(parts[0]), int(parts[1]))
        elif hint.startswith("phase:"):
            new_phase = hint.split(":", 1)[1].upper()
            if new_phase == "OPEN_DOOR":
                self._carrying_key = True
            elif new_phase == "REACH_GOAL":
                self._door_opened = True
            self._phase = new_phase

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        return self._build_narrative(goal_mode, self.events)

    def build_micro(
        self, goal_mode: str, goal_pos=None, last_n: int = 5
    ) -> ZD:
        return self._build_narrative(goal_mode, self.events[-last_n:])

    def _build_narrative(
        self, goal_mode: str, events: List[DoorKeyEvent]
    ) -> ZD:
        if not events:
            return ZD(
                narrative="No events recorded.",
                meaning_tags=["empty"],
                length_chars=20,
                grounding_violations=0,
            )

        tags: List[str] = [f"goal:{goal_mode}"]

        # Phase tag
        tags.append(f"phase:{self._phase.lower()}")

        # Subgoal target + position tags
        if self._phase == "FIND_KEY":
            tags.append("target:key")
            if self._key_pos is not None:
                tags.append(f"key_at:{self._key_pos[0]}_{self._key_pos[1]}")
        elif self._phase == "OPEN_DOOR":
            tags.append("target:door")
            tags.append("carrying_key")
            if self._door_pos is not None:
                tags.append(
                    f"door_at:{self._door_pos[0]}_{self._door_pos[1]}")
        elif self._phase == "REACH_GOAL":
            tags.append("target:goal")
            tags.append("door_open")
            if self._goal_pos is not None:
                tags.append(
                    f"goal_at:{self._goal_pos[0]}_{self._goal_pos[1]}")

        # Progress
        progress = 0
        if self._carrying_key:
            progress += 1
        if self._door_opened:
            progress += 1
        tags.append(f"progress:{progress}")

        # Success / failure
        last = events[-1]
        if last.done:
            tags.append("success" if last.reward > 0 else "timeout")

        # Narrative text
        steps = len(events)
        pos = last.agent_pos
        prefix = "DoorKey synthesis:" if steps > 5 else "DoorKey micro:"
        narrative = (
            f"{prefix} Phase={self._phase}, pos={pos}, steps={steps}. "
            f"Key={'held' if self._carrying_key else 'seeking'}. "
            f"Door={'open' if self._door_opened else 'locked'}. "
            f"Progress={progress}/2 subgoals."
        )

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0,
        )

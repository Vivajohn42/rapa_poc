"""EventPatternD: Experience-based D-stream for autonomous DoorKey learning.

Replaces the deterministic DoorKeyAgentD (which relies on hints from the
privileged environment).  Instead, learns the task structure purely from
observed state changes across episodes:

  - Detects events: KEY_PICKED_UP, DOOR_OPENED, GOAL_REACHED, BLOCKED_AT_DOOR
  - Learns success_sequence from recurring patterns in successful episodes
  - Learns constraints from partial successes and failures
  - Suggests navigation targets based on learned sequence + known positions
  - reflect() for episodic reprocessing when learning stagnates

No BFS-expert, no labels, no LLM.  Cross-episode buffer persists across
kernel.reset_episode() calls (only per-episode state is cleared).

Tag output is compatible with deconstruct_doorkey_d_to_c (no code changes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from kernel.interfaces import StreamD
from state.schema import ZA, ZD


class DoorKeyEventType(Enum):
    """Observable events in DoorKey episodes."""
    KEY_PICKED_UP = auto()
    DOOR_OPENED = auto()
    GOAL_REACHED = auto()
    BLOCKED_AT_DOOR = auto()


@dataclass
class EpisodeRecord:
    """One completed episode's event trace + outcome."""
    episode_id: int
    events: List[DoorKeyEventType]
    success: bool
    steps: int


# Direction vectors: 0=right, 1=down, 2=left, 3=up
_DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class EventPatternD(StreamD):
    """D-Level: learns DoorKey success sequence from experience.

    Cross-episode state (episode_buffer, success_sequence, constraints)
    persists across episodes.  Per-episode state is reset each episode.
    """

    def __init__(self):
        # ---- Cross-episode (persistent) ----
        self.episode_buffer: List[EpisodeRecord] = []
        self.success_sequence: Optional[List[DoorKeyEventType]] = None
        self.partial_hypotheses: List[str] = []
        self.negative_constraints: List[str] = []
        self._episode_counter: int = 0

        # ---- Per-episode (reset each episode) ----
        self.events: List[Dict[str, Any]] = []
        self.seen_positions: Set[Tuple[int, int]] = set()
        self._episode_events: List[DoorKeyEventType] = []
        self._prev_zA: Optional[ZA] = None
        self._prev_carrying: bool = False
        self._prev_door_open: bool = False
        self._blocked_before_key: bool = False

        # ---- ObjectMemory link ----
        self._object_memory = None

    def set_object_memory(self, obj_mem) -> None:
        """Connect to ObjectMemory for event detection and target suggestions."""
        self._object_memory = obj_mem

    # ---- StreamD interface ----

    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        """Record step and detect events from state changes."""
        self.events.append({
            "t": t, "agent_pos": zA.agent_pos,
            "agent_dir": zA.direction or 0,
            "action": action, "reward": reward, "done": done,
        })
        self.seen_positions.add(zA.agent_pos)

        # ---- Event detection via ObjectMemory ----
        if self._object_memory is not None:
            om = self._object_memory

            # KEY_PICKED_UP: carrying changed False → True
            if om.carrying_key and not self._prev_carrying:
                self._episode_events.append(DoorKeyEventType.KEY_PICKED_UP)

            # DOOR_OPENED: door state changed to open
            if om.door_open and not self._prev_door_open:
                self._episode_events.append(DoorKeyEventType.DOOR_OPENED)

            # BLOCKED_AT_DOOR: forward failed + facing door + no key
            if (action == "forward"
                    and self._prev_zA is not None
                    and zA.agent_pos == self._prev_zA.agent_pos
                    and om.door_pos is not None
                    and not om.carrying_key):
                d = zA.direction if zA.direction is not None else 0
                dx, dy = _DIR_VEC[d]
                facing = (zA.agent_pos[0] + dx, zA.agent_pos[1] + dy)
                if facing == om.door_pos:
                    self._episode_events.append(
                        DoorKeyEventType.BLOCKED_AT_DOOR)
                    # Track: blocked before having key
                    if not om.carrying_key:
                        self._blocked_before_key = True

            self._prev_carrying = om.carrying_key
            self._prev_door_open = om.door_open

        # GOAL_REACHED: done with positive reward
        if done and reward > 0:
            self._episode_events.append(DoorKeyEventType.GOAL_REACHED)

        self._prev_zA = zA

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Full episode narrative with tags for deconstruction."""
        return self._build_zd(goal_mode)

    def build_micro(
        self,
        goal_mode: str,
        goal_pos=None,
        last_n: int = 5,
    ) -> ZD:
        """Micro narrative (same as build for EventPatternD)."""
        return self._build_zd(goal_mode)

    # ---- Episode lifecycle ----

    def reset_episode(self) -> None:
        """Reset per-episode state.  Cross-episode buffer persists."""
        self.events.clear()
        self.seen_positions.clear()
        self._episode_events.clear()
        self._prev_zA = None
        self._prev_carrying = False
        self._prev_door_open = False
        self._blocked_before_key = False

    def end_episode(self, success: bool, steps: int) -> None:
        """Called at episode end: record and learn."""
        # Ensure GOAL_REACHED is in the event list for successful episodes
        # (may not have been caught by observe_step if the done tick
        #  doesn't trigger a full observe cycle)
        if success and DoorKeyEventType.GOAL_REACHED not in self._episode_events:
            self._episode_events.append(DoorKeyEventType.GOAL_REACHED)

        record = EpisodeRecord(
            episode_id=self._episode_counter,
            events=list(self._episode_events),
            success=success,
            steps=steps,
        )
        self.episode_buffer.append(record)
        self._episode_counter += 1
        self._learn_from_episodes()

    # ---- Target suggestion ----

    def suggest_target(self) -> Optional[Tuple[int, int]]:
        """Suggest navigation target based on learned sequence + positions.

        Returns None if no hypothesis or missing object position.
        """
        om = self._object_memory
        if om is None:
            return None

        step = self.current_sequence_step()

        # If we have a full success_sequence, follow it
        if self.success_sequence is not None:
            if step == 0:
                return om.key_pos  # None if key not yet seen
            elif step == 1:
                return om.door_pos
            elif step == 2:
                return om.goal_pos
            return None

        # Partial hypotheses: even without full sequence, act on constraints
        if "key_before_door" in self.partial_hypotheses:
            if step == 0:
                return om.key_pos
            elif step == 1:
                return om.door_pos
            elif step == 2:
                return om.goal_pos

        # No hypothesis at all → explore (return None)
        return None

    def current_sequence_step(self) -> int:
        """Which step of the success_sequence are we on?

        0 = no key yet
        1 = have key, door not open
        2 = door open, need to reach goal
        """
        om = self._object_memory
        if om is None:
            return 0
        if om.door_open:
            return 2
        elif om.carrying_key:
            return 1
        return 0

    @property
    def has_hypothesis(self) -> bool:
        """True if D has any hypothesis (full sequence or partial)."""
        return (self.success_sequence is not None
                or len(self.partial_hypotheses) > 0
                or len(self.negative_constraints) > 0)

    # ---- Learning ----

    def _learn_from_episodes(self) -> None:
        """Extract success_sequence, partial hypotheses, and constraints."""
        # --- Learn from full successes ---
        successes = [ep for ep in self.episode_buffer if ep.success]
        if len(successes) >= 2:
            sequences = [ep.events for ep in successes]
            candidate = self._find_common_subsequence(sequences)
            if len(candidate) >= 2:
                self.success_sequence = candidate

        # --- Learn from partial successes and failures ---
        # BLOCKED_AT_DOOR before KEY_PICKED_UP → need key before door
        for ep in self.episode_buffer:
            events = ep.events
            blocked_idx = None
            key_idx = None
            for i, ev in enumerate(events):
                if ev == DoorKeyEventType.BLOCKED_AT_DOOR and blocked_idx is None:
                    blocked_idx = i
                if ev == DoorKeyEventType.KEY_PICKED_UP and key_idx is None:
                    key_idx = i
            if blocked_idx is not None:
                constraint = "need_key_before_door"
                if constraint not in self.negative_constraints:
                    self.negative_constraints.append(constraint)

        # KEY_PICKED_UP observed → partial hypothesis: key is part of sequence
        any_key_picked = any(
            DoorKeyEventType.KEY_PICKED_UP in ep.events
            for ep in self.episode_buffer
        )
        if any_key_picked:
            hyp = "key_before_door"
            if hyp not in self.partial_hypotheses:
                self.partial_hypotheses.append(hyp)

        # DOOR_OPENED observed → door is part of sequence after key
        any_door_opened = any(
            DoorKeyEventType.DOOR_OPENED in ep.events
            for ep in self.episode_buffer
        )
        if any_door_opened and any_key_picked:
            hyp = "door_after_key"
            if hyp not in self.partial_hypotheses:
                self.partial_hypotheses.append(hyp)

            # If we've seen both key pickup and door opened in any episode,
            # we can form the full sequence even without a success
            if self.success_sequence is None:
                self.success_sequence = [
                    DoorKeyEventType.KEY_PICKED_UP,
                    DoorKeyEventType.DOOR_OPENED,
                    DoorKeyEventType.GOAL_REACHED,
                ]

    def reflect(self) -> None:
        """Episodic reprocessing on stagnation.

        Re-analyze recent episodes with stricter criteria:
          - Recompute from last 20 episodes
          - Check failure patterns
          - Strengthen constraints
        """
        recent = (self.episode_buffer[-20:]
                  if len(self.episode_buffer) > 20
                  else self.episode_buffer)

        # Recompute from recent successes
        successes = [ep for ep in recent if ep.success]
        if len(successes) >= 2:
            sequences = [ep.events for ep in successes]
            candidate = self._find_common_subsequence(sequences)
            if len(candidate) >= 2:
                self.success_sequence = candidate

        # Analyze failure patterns in recent episodes
        failures = [ep for ep in recent if not ep.success]
        blocked_count = sum(
            1 for ep in failures
            if DoorKeyEventType.BLOCKED_AT_DOOR in ep.events
        )
        if (blocked_count >= 2
                and "need_key_before_door" not in self.negative_constraints):
            self.negative_constraints.append("need_key_before_door")

    # ---- LCS algorithm ----

    def _find_common_subsequence(
        self, sequences: List[List[DoorKeyEventType]]
    ) -> List[DoorKeyEventType]:
        """Find longest common subsequence across all event sequences.

        Uses pairwise LCS reduction.  With 3-4 event types this is trivial.
        """
        if not sequences:
            return []
        result = list(sequences[0])
        for seq in sequences[1:]:
            result = self._lcs(result, seq)
            if not result:
                break
        return result

    @staticmethod
    def _lcs(a: List, b: List) -> List:
        """Standard LCS (dynamic programming)."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # Backtrack
        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                result.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return result[::-1]

    # ---- ZD construction ----

    def _build_zd(self, goal_mode: str) -> ZD:
        """Build ZD with tags compatible with deconstruct_doorkey_d_to_c."""
        tags: List[str] = [f"goal:{goal_mode}"]
        om = self._object_memory

        step = self.current_sequence_step()

        # Phase tags
        if step == 0:
            tags.append("phase:find_key")
            tags.append("target:key")
            if om and om.key_pos:
                tags.append(f"key_at:{om.key_pos[0]}_{om.key_pos[1]}")
        elif step == 1:
            tags.append("phase:open_door")
            tags.append("target:door")
            tags.append("carrying_key")
            if om and om.door_pos:
                tags.append(f"door_at:{om.door_pos[0]}_{om.door_pos[1]}")
        elif step == 2:
            tags.append("phase:reach_goal")
            tags.append("target:goal")
            tags.append("door_open")
            if om and om.goal_pos:
                tags.append(f"goal_at:{om.goal_pos[0]}_{om.goal_pos[1]}")

        # Progress
        tags.append(f"progress:{step}")

        # Episode outcome
        if self.events and self.events[-1].get("done"):
            reward = self.events[-1].get("reward", 0)
            tags.append("success" if reward > 0 else "timeout")

        # Narrative
        if self.success_sequence is not None:
            seq_str = " -> ".join(e.name for e in self.success_sequence)
            narrative = (
                f"Autonomous DoorKey: step={step}, "
                f"learned_seq=[{seq_str}], "
                f"episodes_seen={self._episode_counter}, "
                f"constraints={self.negative_constraints}"
            )
        elif self.partial_hypotheses:
            narrative = (
                f"Autonomous DoorKey: step={step}, "
                f"partial_hypotheses={self.partial_hypotheses}, "
                f"constraints={self.negative_constraints}, "
                f"episodes_seen={self._episode_counter}"
            )
        else:
            narrative = (
                f"Autonomous DoorKey: exploring (no hypothesis yet), "
                f"episodes_seen={self._episode_counter}"
            )

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0,
        )

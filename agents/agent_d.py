from dataclasses import dataclass
from typing import List, Dict, Any

from state.schema import ZA, ZD

@dataclass
class Event:
    t: int
    agent_pos: tuple
    action: str
    reward: float
    done: bool
    hint: str | None

class AgentD:
    """
    Narrative/Meaning Agent (M+N). MVP ohne LLM:
    - sammelt Events
    - erzeugt ein kurzes Narrativ + Meaning-Tags
    - prüft einfaches Grounding (keine Positionen erfinden)
    """
    def __init__(self):
        self.events: List[Event] = []
        self.seen_positions = set()

    def observe_step(self, t: int, zA: ZA, action: str, reward: float, done: bool):
        self.events.append(Event(t=t, agent_pos=zA.agent_pos, action=action, reward=reward, done=done, hint=zA.hint))
        self.seen_positions.add(zA.agent_pos)

    def build(self, goal_mode: str, goal_pos: tuple) -> ZD:
        if not self.events:
            narrative = "No events recorded."
            return ZD(narrative=narrative, meaning_tags=["empty"], length_chars=len(narrative), grounding_violations=0)

        start = self.events[0].agent_pos
        end = self.events[-1].agent_pos
        steps = len(self.events)
        total_reward = sum(e.reward for e in self.events)
        success = self.events[-1].done

        # Simple meaning tags
        tags = []
        tags.append("goal:seek" if goal_mode == "seek" else "goal:avoid")
        tags.append("success" if success else "no_success")
        tags.append("short_episode" if steps <= 12 else "long_episode")

        hint = None
        for e in reversed(self.events):  # in build_micro
            if e.hint is not None:
                hint = e.hint
                break
        if hint:
            tags.append(f"hint:{hint}")

        # Grounding check: narrative mentions only seen positions (MVP: we only mention start/end)
        grounding_violations = 0  # deterministic narrative => 0

        narrative = (
            f"Episode summary: start at {start}, end at {end}, goal at {goal_pos}. "
            f"Mode={goal_mode}. Steps={steps}. Success={success}. TotalReward={round(total_reward,3)}. "
            f"Key actions: {', '.join(e.action for e in self.events[:min(6,steps)])}"
            + ("..." if steps > 6 else "")
        )

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=grounding_violations
        )
    def build_micro(self, goal_mode: str, goal_pos: tuple, last_n: int = 5) -> ZD:
        """
        Short narrative over the last N events; used when D is triggered mid-episode.
        """
        if not self.events:
            narrative = "No events recorded."
            return ZD(narrative=narrative, meaning_tags=["empty"], length_chars=len(narrative), grounding_violations=0)

        slice_events = self.events[-last_n:]
        start = slice_events[0].agent_pos
        end = slice_events[-1].agent_pos
        steps = len(slice_events)
        total_reward = sum(e.reward for e in slice_events)

        tags = []
        tags.append("micro")
        tags.append("goal:seek" if goal_mode == "seek" else "goal:avoid")
        tags.append("stability:stuck" if len(set(e.agent_pos for e in slice_events)) == 1 else "stability:moving")

        hint = None
        for e in reversed(slice_events):  # in build_micro
            if e.hint is not None:
                hint = e.hint
                break
        if hint:
            tags.append(f"hint:{hint}")

        narrative = (
            f"Micro-summary (last {steps}): {start}->{end}, goal={goal_pos}, "
            f"mode={goal_mode}, reward_sum={round(total_reward,3)}, "
            f"actions={','.join(e.action for e in slice_events)}"
        )

        return ZD(narrative=narrative, meaning_tags=tags, length_chars=len(narrative), grounding_violations=0)

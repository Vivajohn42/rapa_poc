from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from state.schema import ZA, ZD
from llm.provider import LLMProvider


@dataclass
class Event:
    t: int
    agent_pos: tuple
    action: str
    reward: float
    done: bool
    hint: Optional[str]


class AgentDLLM:
    """
    LLM-backed D:
    - consumes grounded events (facts buffer)
    - outputs: narrative + meaning tags
    - returns ZD (drop-in)
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.events: List[Event] = []

    def observe_step(self, t: int, zA: ZA, action: str, reward: float, done: bool):
        self.events.append(Event(
            t=t,
            agent_pos=zA.agent_pos,
            action=action,
            reward=reward,
            done=done,
            hint=zA.hint
        ))

    def build_micro(self, goal_mode: str, goal_pos=None, last_n: int = 6) -> ZD:
        # goal_pos is accepted for drop-in compatibility with AgentD (can be None / ignored)
        slice_events = self.events[-last_n:] if self.events else []
        facts = [
            f"t={e.t} pos={e.agent_pos} action={e.action} reward={e.reward} done={e.done} hint={e.hint}"
            for e in slice_events
        ]

        system = (
            "You are a narrative/meaning module.\n"
            "RULES:\n"
            "- Use ONLY the FACTS provided.\n"
            "- Do NOT invent positions/actions/rewards/hints.\n"
            "- Output EXACTLY two lines:\n"
            "NARRATIVE: <1-2 short sentences>\n"
            "TAGS: <comma-separated tags; include hint:A or hint:B if any hint appears>\n"
        )

        user = "FACTS:\n" + "\n".join(facts) + f"\nMODE={goal_mode}\n"


        txt = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=160,
        ).strip()

        narrative = ""
        tags: List[str] = []

        for line in txt.splitlines():
            if line.startswith("NARRATIVE:"):
                narrative = line.split(":", 1)[1].strip()
            elif line.startswith("TAGS:"):
                raw = line.split(":", 1)[1].strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]

        # Deterministic hint tag injection (do NOT rely on LLM for this)
        # If any event contains hint "A" or "B", enforce tag "hint:A"/"hint:B".
        hint_val = None
        for e in reversed(slice_events):
            if e.hint in ("A", "B"):
                hint_val = e.hint
                break
        if hint_val:
            forced = f"hint:{hint_val}"
            # normalize duplicates
            if forced not in tags and forced.lower() not in [t.lower() for t in tags]:
                tags.append(forced)


        # Fallback if model didn't follow format
        if not narrative:
            narrative = txt[:240]
        if not tags:
            tags = ["llm_format_fallback"]

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0
        )

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """
        Final narrative over the whole episode (or last N if you prefer).
        Signature matches AgentD for drop-in compatibility.
        """
        # Use the whole event list, but cap to last 30 to keep prompts small
        slice_events = self.events[-30:] if self.events else []
        facts = [
            f"t={e.t} pos={e.agent_pos} action={e.action} reward={e.reward} done={e.done} hint={e.hint}"
            for e in slice_events
        ]

        system = (
            "You are a narrative/meaning module.\n"
            "RULES:\n"
            "- Use ONLY the FACTS provided.\n"
            "- Do NOT invent positions/actions/rewards/hints.\n"
            "- Output EXACTLY two lines:\n"
            "NARRATIVE: <2-4 short sentences summarizing the episode>\n"
            "TAGS: <comma-separated tags; include hint:A or hint:B if any hint appears>\n"
        )

        user = "FACTS:\n" + "\n".join(facts) + f"\nMODE={goal_mode}\n"

        txt = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=220,
        ).strip()

        narrative = ""
        tags: List[str] = []

        for line in txt.splitlines():
            if line.startswith("NARRATIVE:"):
                narrative = line.split(":", 1)[1].strip()
            elif line.startswith("TAGS:"):
                raw = line.split(":", 1)[1].strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]

        # Deterministic hint tag injection (do NOT rely on LLM for this)
        # If any event contains hint "A" or "B", enforce tag "hint:A"/"hint:B".
        hint_val = None
        for e in reversed(slice_events):
            if e.hint in ("A", "B"):
                hint_val = e.hint
                break
        if hint_val:
            forced = f"hint:{hint_val}"
            # normalize duplicates
            if forced not in tags and forced.lower() not in [t.lower() for t in tags]:
                tags.append(forced)


        if not narrative:
            narrative = txt[:400]
        if not tags:
            tags = ["llm_format_fallback"]

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=0
        )

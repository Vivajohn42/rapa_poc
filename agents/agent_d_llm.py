from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from state.schema import ZA, ZD
from llm.provider import LLMProvider
from llm.output_parser import parse_narrative_tags, parse_full_output


@dataclass
class Event:
    t: int
    agent_pos: tuple
    action: str
    reward: float
    done: bool
    hint: Optional[str]


from kernel.interfaces import StreamD


class AgentDLLM(StreamD):
    """
    LLM-backed D:
    - consumes grounded events (facts buffer)
    - outputs: narrative + meaning tags
    - returns ZD (drop-in)
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.events: List[Event] = []
        self.seen_positions = set()

    def observe_step(self, t: int, zA: ZA, action: str, reward: float, done: bool):
        self.events.append(Event(
            t=t,
            agent_pos=zA.agent_pos,
            action=action,
            reward=reward,
            done=done,
            hint=zA.hint
        ))
        self.seen_positions.add(zA.agent_pos)

    def build_micro(
        self, goal_mode: str, goal_pos=None, last_n: int = 6,
        action: str | None = None, zA_next: ZA | None = None,
    ) -> ZD:
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

        # F.4: Action context for prediction
        if action is not None:
            user += f"ACTION_TAKEN={action}\n"
        if zA_next is not None:
            user += f"NEXT_POS={zA_next.agent_pos}\n"

        # F.4: Self-correction — inject previous prediction
        prev_prediction = ""
        if hasattr(self.llm, "canvas_manager") and self.llm.canvas_manager is not None:
            prev = getattr(self.llm.canvas_manager, "get_prediction", lambda: None)()
            if prev:
                prev_prediction = prev
                user += f"PREV_PREDICTION={prev}\n"

        # Single-pass generation: model produces NARRATIVE + PREDICTION + TAGS
        # in one block (SFT-aligned models), parsed by parse_full_output.
        try:
            result = self.llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=120,
            )
        except TypeError:
            result = ""

        txt = result.strip() if isinstance(result, str) else str(result)
        narrative, prediction, tags, fmt_quality = parse_full_output(txt)

        # Deterministic hint tag injection (do NOT rely on LLM for this)
        hint_val = None
        for e in reversed(slice_events):
            if e.hint in ("A", "B"):
                hint_val = e.hint
                break
        if hint_val:
            forced = f"hint:{hint_val}"
            if forced not in tags and forced.lower() not in [t.lower() for t in tags]:
                tags.append(forced)

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=fmt_quality,
            prediction=prediction,
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
            max_tokens=120,
        ).strip()

        narrative, tags, fmt_quality = parse_narrative_tags(txt)

        # Deterministic hint tag injection (do NOT rely on LLM for this)
        hint_val = None
        for e in reversed(slice_events):
            if e.hint in ("A", "B"):
                hint_val = e.hint
                break
        if hint_val:
            forced = f"hint:{hint_val}"
            if forced not in tags and forced.lower() not in [t.lower() for t in tags]:
                tags.append(forced)

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=fmt_quality,
        )

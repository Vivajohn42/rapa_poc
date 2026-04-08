from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from state.schema import ZA, ZD
from llm.provider import LLMProvider
from llm.output_parser import parse_narrative_tags, parse_full_output, parse_reasoning_output


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
            "- Output EXACTLY two lines, for example:\n"
            "NARRATIVE: The agent moved right and discovered a hint.\n"
            "TAGS: movement, hint:A, goal:seek\n"
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

    # ─── Reasoning Mode (Phase G) ─────────────────────────────

    def build_reasoning(self, canvas_manager, question: str | None = None) -> ZD:
        """Canvas-based reasoning: read [MEMORY] → think → REASONING + ANSWER.

        Always uses [MEMORY] format, never falls back to FACTS format.
        The reasoning model speaks only this language.
        """
        # Canvas als [MEMORY] block — immer, auch wenn leer
        memory_block = canvas_manager.to_prefix()
        if not memory_block:
            memory_block = (
                "[MEMORY]\n"
                f"agent_pos: {self.events[-1].agent_pos if self.events else '(0,0)'}\n"
                "status: No facts discovered yet.\n"
                "[/MEMORY]"
            )

        # Frage automatisch ableiten wenn nicht gegeben
        if question is None:
            question = self._infer_question(canvas_manager)

        prompt = memory_block + "\n" + f"QUESTION: {question}\n"

        try:
            txt = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
        except Exception:
            txt = ""

        if isinstance(txt, tuple):
            txt = txt[0]
        txt = txt.strip() if isinstance(txt, str) else str(txt)

        reasoning, answer, tags, fmt_quality = parse_reasoning_output(txt)

        # Deterministic hint injection (same as build_micro)
        if self.events:
            for e in reversed(self.events[-6:]):
                if e.hint in ("A", "B"):
                    forced = f"hint:{e.hint}"
                    if forced not in tags and forced.lower() not in [t.lower() for t in tags]:
                        tags.append(forced)
                    break

        return ZD(
            narrative=reasoning,     # REASONING = "laut denken"
            meaning_tags=tags,
            length_chars=len(reasoning),
            grounding_violations=fmt_quality,
            prediction=answer,       # ANSWER = Handlungsempfehlung
        )

    def _infer_question(self, canvas_manager) -> str:
        """Derive the right question from Canvas state + recent events."""
        slots = canvas_manager.slots if hasattr(canvas_manager, 'slots') else {}

        # Check if target is known
        has_target = "target" in slots
        has_hint = any(k.startswith("hint_") for k in slots)

        # Check if agent is stuck (same position for last N events)
        is_stuck = False
        if len(self.events) >= 4:
            recent_pos = [e.agent_pos for e in self.events[-4:]]
            is_stuck = len(set(str(p) for p in recent_pos)) == 1

        if is_stuck:
            return "The agent is stuck and not making progress. What should it do?"
        elif has_target:
            return "Given the known target, what is the best strategy to reach it?"
        elif has_hint:
            return "A hint has been discovered. What does it tell us about the target?"
        elif len(slots) <= 1:
            return "What should the agent do in an unknown environment?"
        else:
            return "Based on the available information, what should the agent prioritize?"

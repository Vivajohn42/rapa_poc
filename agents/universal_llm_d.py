"""Universal LLM-backed D stream that works across all environments.

Uses the adapter pattern to separate environment-specific context
(prompts, grounding, forced tags) from the core D logic (event
recording, LLM calling, NARRATIVE:/TAGS: parsing).

The same UniversalLlmD class works for GridWorld, TextWorld, and
Riddle Rooms — only the adapter differs.

Usage:
    from llm.provider import OllamaProvider
    from agents.llm_d_adapters import TextWorldLlmAdapter
    from agents.universal_llm_d import UniversalLlmD

    llm = OllamaProvider(model="mistral:latest")
    adapter = TextWorldLlmAdapter(room_properties, all_rooms, room_index)
    D = UniversalLlmD(llm, adapter)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from state.schema import ZA, ZD
from kernel.interfaces import StreamD
from agents.llm_d_adapters import LlmDAdapter


class UniversalLlmD(StreamD):
    """Universal LLM-backed D stream with adapter-based context injection.

    Architecture:
      1. observe_step() → records generic event + adapter context
      2. build/build_micro() → adapter builds prompt → LLM generates
      3. Parse NARRATIVE:/TAGS: format (case-insensitive, fallback on failure)
      4. Adapter validates grounding (hallucinated entities → violations)
      5. Adapter forces deterministic tags (hints, goal mode)

    The LLM only ever sees the adapter's formatted prompt.
    """

    def __init__(self, llm, adapter: LlmDAdapter):
        self.llm = llm
        self.adapter = adapter
        self.events: List[Dict[str, Any]] = []
        self.seen_positions: set = set()

    # ── StreamD interface ─────────────────────────────────────────────

    def observe_step(
        self, t: int, zA: ZA, action: str, reward: float, done: bool,
    ) -> None:
        """Record a step with both generic and adapter-specific context."""
        event: Dict[str, Any] = {
            "t": t,
            "agent_pos": zA.agent_pos,
            "action": action,
            "reward": reward,
            "done": done,
        }
        ctx = self.adapter.extract_event_context(zA)
        event.update(ctx)

        self.events.append(event)
        self.seen_positions.add(zA.agent_pos)

        # Notify adapter of new clues/hints
        clue = ctx.get("clue") or ctx.get("hint")
        if clue is not None:
            self.adapter.on_new_clue(clue)

    def build_micro(
        self, goal_mode: str, goal_pos=None, last_n: int = 6,
    ) -> ZD:
        """Short LLM synthesis from recent events."""
        slice_events = self.events[-last_n:] if self.events else []
        return self._build_from_events(slice_events, goal_mode, micro=True)

    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Full episode synthesis via LLM."""
        slice_events = self.events[-30:] if self.events else []
        return self._build_from_events(slice_events, goal_mode, micro=False)

    # ── Core build pipeline ───────────────────────────────────────────

    def _build_from_events(
        self,
        events: List[Dict[str, Any]],
        goal_mode: str,
        micro: bool,
    ) -> ZD:
        """Prompt → LLM → parse → validate → force tags."""
        if not events:
            return ZD(
                narrative="No events recorded.",
                meaning_tags=["empty"],
                length_chars=20,
                grounding_violations=0,
            )

        # 1. Adapter builds prompts
        system = self.adapter.build_system_prompt(micro=micro)
        user = self.adapter.build_user_prompt(events, goal_mode)

        # 2. LLM call with error handling
        max_tokens = 100 if micro else 200
        try:
            txt = self.llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=max_tokens,
            ).strip()
        except Exception:
            return ZD(
                narrative="LLM call failed.",
                meaning_tags=["llm_error"],
                length_chars=15,
                grounding_violations=1,
            )

        # 3. Parse NARRATIVE:/TAGS: format
        narrative, tags = self._parse_response(txt)

        # 4. Adapter validates grounding
        violations = self.adapter.validate_grounding(tags)

        # 5. Adapter forces deterministic tags
        tags = self.adapter.force_deterministic_tags(tags, events, goal_mode)

        return ZD(
            narrative=narrative,
            meaning_tags=tags,
            length_chars=len(narrative),
            grounding_violations=violations,
        )

    # ── Response parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_response(txt: str) -> Tuple[str, List[str]]:
        """Parse NARRATIVE:/TAGS: format from LLM output.

        Case-insensitive prefix matching with robust fallback.
        Handles missing lines, extra whitespace, varied casing.
        """
        narrative = ""
        tags: List[str] = []

        for line in txt.splitlines():
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("NARRATIVE:"):
                narrative = stripped.split(":", 1)[1].strip()
            elif upper.startswith("TAGS:"):
                raw = stripped.split(":", 1)[1].strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]

        # Fallback if model didn't follow format
        if not narrative:
            narrative = txt[:240]
        if not tags:
            tags = ["llm_format_fallback"]

        return narrative, tags

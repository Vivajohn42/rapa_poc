"""Robust NARRATIVE/TAGS parser for LLM output.

Multi-strategy parser that handles format variations from non-instruction-tuned
models. Falls back gracefully through 4 strategies:
  0 = strict prefix match
  1 = case-insensitive prefix match
  2 = regex extraction (handles multi-line narrative, TAGS: anywhere)
  3 = heuristic fallback (entire text = narrative)

Used by AgentDLLM and UniversalLlmD to parse DEF model output.
"""
from __future__ import annotations

import re


def parse_narrative_tags(txt: str) -> tuple[str, list[str], int]:
    """Parse LLM output into (narrative, tags, format_quality).

    Args:
        txt: Raw LLM output text.

    Returns:
        narrative: Extracted narrative string (may be empty on total failure).
        tags: List of tag strings (may contain "llm_format_fallback").
        format_quality: 0=perfect, 1=case-insensitive, 2=regex, 3=fallback.
    """
    if not txt or not txt.strip():
        return "", ["llm_format_fallback"], 3

    # Strategy 0: strict prefix match (original AgentDLLM logic)
    narrative, tags = _parse_strict(txt)
    if narrative and tags:
        return narrative, tags, 0

    # Strategy 1: case-insensitive prefix match
    narrative, tags = _parse_case_insensitive(txt)
    if narrative and tags:
        return narrative, tags, 1

    # Strategy 2: regex extraction
    narrative, tags = _parse_regex(txt)
    if narrative and tags:
        return narrative, tags, 2

    # Strategy 3: heuristic fallback
    narrative = txt.strip()[:400]
    return narrative, ["llm_format_fallback"], 3


def _parse_strict(txt: str) -> tuple[str, list[str]]:
    """Strategy 0: exact 'NARRATIVE:' and 'TAGS:' prefix match."""
    narrative = ""
    tags: list[str] = []
    for line in txt.splitlines():
        stripped = line.strip()
        if stripped.startswith("NARRATIVE:"):
            narrative = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("TAGS:"):
            raw = stripped.split(":", 1)[1].strip()
            tags = _split_tags(raw)
    return narrative, tags


def _parse_case_insensitive(txt: str) -> tuple[str, list[str]]:
    """Strategy 1: case-insensitive 'narrative:' and 'tags:' prefix."""
    narrative = ""
    tags: list[str] = []
    for line in txt.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("NARRATIVE:"):
            narrative = stripped[len("NARRATIVE:"):].strip()
        elif upper.startswith("TAGS:"):
            raw = stripped[len("TAGS:"):].strip()
            tags = _split_tags(raw)
    return narrative, tags


def _parse_regex(txt: str) -> tuple[str, list[str]]:
    """Strategy 2: regex extraction (handles multi-line, TAGS: anywhere)."""
    narrative = ""
    tags: list[str] = []

    # Narrative: everything between NARRATIVE: and TAGS: (or end)
    m = re.search(
        r'NARRATIVE:\s*(.+?)(?=\n\s*TAGS:|\Z)',
        txt, re.IGNORECASE | re.DOTALL,
    )
    if m:
        narrative = m.group(1).strip()

    # Tags: everything after TAGS:
    t = re.search(r'TAGS:\s*(.+)', txt, re.IGNORECASE)
    if t:
        raw = t.group(1).strip()
        tags = _split_tags(raw)

    return narrative, tags


def _split_tags(raw: str) -> list[str]:
    """Split tag string on commas or semicolons, strip whitespace."""
    return [t.strip() for t in re.split(r'[,;]', raw) if t.strip()]

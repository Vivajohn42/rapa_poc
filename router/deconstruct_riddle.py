"""Deconstruction pipeline: Riddle D -> C.

Parses D's meaning_tags for answer identification and updates C's memory.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from state.schema import ZC, ZD


def deconstruct_riddle_d_to_c(
    zC: ZC,
    zD: ZD,
    goal_map: Optional[Dict] = None,
    answer_index: Optional[Dict[str, int]] = None,
) -> ZC:
    """Parse D's answer/eliminated tags into C's memory.

    answer_index maps answer_id -> pseudo-position index.
    """
    answer_index = answer_index or {}
    tags = [t.lower().strip() for t in zD.meaning_tags]

    for tag in tags:
        # answer:{id} or target:{id} -> set target in memory
        if tag.startswith("answer:") or tag.startswith("target:"):
            answer_id = tag.split(":", 1)[1]
            if answer_id in answer_index:
                zC.memory["target"] = (answer_index[answer_id], 0)
                zC.memory["target_answer"] = answer_id

        # eliminated:{id} -> track eliminated answers
        elif tag.startswith("eliminated:"):
            elim_id = tag.split(":", 1)[1]
            eliminated = zC.memory.get("eliminated_answers", [])
            if elim_id not in eliminated:
                eliminated.append(elim_id)
                zC.memory["eliminated_answers"] = eliminated

        # evidence:{N} -> track evidence count
        elif tag.startswith("evidence:"):
            try:
                n = int(tag.split(":", 1)[1])
                zC.memory["evidence_count"] = n
            except ValueError:
                pass

        # candidates:{N} -> track remaining candidates
        elif tag.startswith("candidates:"):
            try:
                n = int(tag.split(":", 1)[1])
                zC.memory["candidates_remaining"] = n
            except ValueError:
                pass

    # Store narrative for introspection
    zC.memory["last_narrative"] = zD.narrative
    zC.memory["last_tags"] = list(zD.meaning_tags)

    return zC

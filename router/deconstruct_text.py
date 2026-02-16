"""TextWorld deconstruction: D tags -> C memory.

Maps TextAgentD's synthesized tags into C's memory with pseudo-positions,
so TextAgentC can navigate to the identified target room.

Tag patterns:
  "target:{room_id}" -> memory["target"] = (room_index[room_id], 0)
  "candidates:{N}"   -> memory["candidates_remaining"] = N
  "clue_collected:{N}" -> memory["clues_collected"] = N
"""
from __future__ import annotations

from typing import Dict, Optional

from state.schema import ZC, ZD


def deconstruct_text_d_to_c(
    zC: ZC,
    zD: ZD,
    goal_map: Optional[Dict] = None,
    room_index: Optional[Dict[str, int]] = None,
) -> ZC:
    """Parse D's tags into C's memory for TextWorld.

    Args:
        zC: Current C state.
        zD: D's output with meaning_tags.
        goal_map: Unused (kept for interface compat with GridWorld deconstruct).
        room_index: Maps room_id -> integer index for pseudo-position.
    """
    if room_index is None:
        room_index = {}

    tags = [t.strip().lower() for t in zD.meaning_tags]

    for tag in tags:
        # "target:vault" -> memory["target"] = (room_index["vault"], 0)
        if tag.startswith("target:"):
            room_id = tag.split(":", 1)[1].strip()
            if room_id in room_index:
                zC.memory["target"] = (room_index[room_id], 0)
                zC.memory["target_room"] = room_id

        # "candidates:3" -> memory["candidates_remaining"] = 3
        elif tag.startswith("candidates:"):
            try:
                n = int(tag.split(":", 1)[1].strip())
                zC.memory["candidates_remaining"] = n
            except ValueError:
                pass

        # "clue_collected:2" -> memory["clues_collected"] = 2
        elif tag.startswith("clue_collected:"):
            try:
                n = int(tag.split(":", 1)[1].strip())
                zC.memory["clues_collected"] = n
            except ValueError:
                pass

    return zC

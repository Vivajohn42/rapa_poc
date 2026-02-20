"""Deconstruction pipeline: DoorKey D -> C.

Parses D's phase/target tags into C's memory with concrete positions.
The key mechanism: D tells C what the current subgoal is and where it is.

Tag patterns:
  "target:key"       -> subgoal = "key"
  "target:door"      -> subgoal = "door"
  "target:goal"      -> subgoal = "goal"
  "key_at:{x}_{y}"   -> memory["key_pos"]  = (x, y)
  "door_at:{x}_{y}"  -> memory["door_pos"] = (x, y)
  "goal_at:{x}_{y}"  -> memory["goal_pos"] = (x, y)
  "phase:{name}"     -> memory["phase"]    = name
  "carrying_key"     -> memory["has_key"]  = True
  "door_open"        -> memory["door_open"] = True

The target is set based on phase + discovered positions.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from state.schema import ZC, ZD


def deconstruct_doorkey_d_to_c(
    zC: ZC,
    zD: ZD,
    goal_map: Optional[Dict[str, Tuple[int, int]]] = None,
) -> ZC:
    """Parse D's tags into C's memory for DoorKey navigation."""
    tags = [t.strip().lower() for t in zD.meaning_tags]

    for tag in tags:
        if tag.startswith("phase:"):
            zC.memory["phase"] = tag.split(":", 1)[1]

        elif tag.startswith("key_at:"):
            parts = tag.split(":", 1)[1].split("_")
            try:
                zC.memory["key_pos"] = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        elif tag.startswith("door_at:"):
            parts = tag.split(":", 1)[1].split("_")
            try:
                zC.memory["door_pos"] = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        elif tag.startswith("goal_at:"):
            parts = tag.split(":", 1)[1].split("_")
            try:
                zC.memory["goal_pos"] = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        elif tag.startswith("target:"):
            zC.memory["subgoal"] = tag.split(":", 1)[1]

        elif tag == "carrying_key":
            zC.memory["has_key"] = True

        elif tag == "door_open":
            zC.memory["door_open"] = True

        elif tag.startswith("progress:"):
            try:
                zC.memory["progress"] = int(tag.split(":", 1)[1])
            except ValueError:
                pass

    # Set navigation target based on phase + discovered positions.
    # If the required position isn't known yet, clear target so C
    # falls back to frontier exploration instead of chasing a stale target.
    phase = zC.memory.get("phase", "find_key")

    if phase == "find_key" and "key_pos" in zC.memory:
        zC.memory["target"] = zC.memory["key_pos"]
    elif phase == "open_door":
        if "door_pos" in zC.memory:
            zC.memory["target"] = zC.memory["door_pos"]
        else:
            zC.memory["target"] = None  # explore until door found
    elif phase == "reach_goal":
        if "goal_pos" in zC.memory:
            zC.memory["target"] = zC.memory["goal_pos"]
        elif goal_map and "goal" in goal_map:
            zC.memory["target"] = goal_map["goal"]
        else:
            zC.memory["target"] = None  # explore until goal found

    # Store narrative for introspection
    zC.memory["last_narrative"] = zD.narrative
    zC.memory["last_tags"] = list(zD.meaning_tags)

    return zC

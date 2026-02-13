from typing import Dict, Tuple, Optional
from state.schema import ZC, ZD


def normalize_tags(tags):
    """
    Normalize meaning tags to lowercase, strip whitespace.
    """
    return set(t.strip().lower() for t in tags if isinstance(t, str))


# Default goal map for backward compatibility (5x5 grid)
_DEFAULT_GOAL_MAP = {
    "A": (4, 4),
    "B": (4, 0),
}


def deconstruct_d_to_c(
    zC: ZC,
    zD: ZD,
    goal_map: Optional[Dict[str, Tuple[int, int]]] = None,
) -> ZC:
    """
    Deconstruction:
    Translates D-stream narrative + meaning tags into persistent C-memory updates.

    This function is intentionally:
    - deterministic
    - bounded
    - schema-safe

    Args:
        zC: Current C-stream state
        zD: D-stream output with narrative and meaning tags
        goal_map: Optional mapping from goal_id -> (x, y) coordinates.
                  If None, uses default 5x5 grid coordinates.
    """
    if goal_map is None:
        goal_map = _DEFAULT_GOAL_MAP

    mem = dict(zC.memory) if zC.memory else {}

    tags = normalize_tags(zD.meaning_tags)

    # --- Hidden-goal extraction ---
    # Accept several tag spellings for robustness with LLMs
    # Check all goal_ids in the goal_map
    for goal_id in goal_map:
        gid_lower = goal_id.lower()
        if (f"hint:{gid_lower}" in tags
                or f"goal:{gid_lower}" in tags
                or f"target:{gid_lower}" in tags):
            mem["hint_goal"] = goal_id
            mem["target"] = goal_map[goal_id]

    # --- Multi-goal elimination hints ---
    # Parse "not_X_Y" style hints that eliminate candidates
    # These come from HintCellDef.hint_text in multi-goal scenarios
    eliminated = mem.get("eliminated_goals", [])
    for tag in tags:
        if tag.startswith("not_"):
            # Parse "not_c_d" -> eliminate C, D
            parts = tag[4:].split("_")
            for p in parts:
                gid = p.upper()
                if gid in goal_map and gid not in eliminated:
                    eliminated.append(gid)
    if eliminated:
        mem["eliminated_goals"] = eliminated
        # If only one goal remains, set it as target
        remaining = [gid for gid in goal_map if gid not in eliminated]
        if len(remaining) == 1:
            mem["hint_goal"] = remaining[0]
            mem["target"] = goal_map[remaining[0]]

    # --- Optional: store last narrative for introspection/debug ---
    mem["last_narrative"] = zD.narrative

    # --- Optional: store raw tags (for analysis) ---
    mem["last_tags"] = list(tags)

    # Return updated ZC
    return ZC(goal_mode=zC.goal_mode, memory=mem)

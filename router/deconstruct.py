from state.schema import ZC, ZD


def normalize_tags(tags):
    """
    Normalize meaning tags to lowercase, strip whitespace.
    """
    return set(t.strip().lower() for t in tags if isinstance(t, str))


def deconstruct_d_to_c(zC: ZC, zD: ZD) -> ZC:
    """
    Deconstruction:
    Translates D-stream narrative + meaning tags into persistent C-memory updates.

    This function is intentionally:
    - deterministic
    - bounded
    - schema-safe
    """

    mem = dict(zC.memory) if zC.memory else {}

    tags = normalize_tags(zD.meaning_tags)

    # --- Hidden-goal extraction ---
    # Accept several tag spellings for robustness with LLMs
    if "hint:a" in tags or "goal:a" in tags or "target:a" in tags:
        mem["hint_goal"] = "A"
        mem["target"] = None  # will be filled below

    if "hint:b" in tags or "goal:b" in tags or "target:b" in tags:
        mem["hint_goal"] = "B"
        mem["target"] = None

    # --- Convert hint_goal into concrete target if known grid size ---
    # (Grid is 5x5 in MVP; adapt later if generalized)
    if "hint_goal" in mem:
        if mem["hint_goal"] == "A":
            mem["target"] = (4, 4)   # bottom-right
        elif mem["hint_goal"] == "B":
            mem["target"] = (4, 0)   # top-right

    # --- Optional: store last narrative for introspection/debug ---
    mem["last_narrative"] = zD.narrative

    # --- Optional: store raw tags (for analysis) ---
    mem["last_tags"] = list(tags)

    # Return updated ZC
    return ZC(goal_mode=zC.goal_mode, memory=mem)

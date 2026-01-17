from state.schema import ZD

def narrative_metrics(zD: ZD) -> dict:
    """
    Minimal metrics:
    - length_chars
    - grounding_violations
    - tag_count
    """
    return {
        "narrative_length_chars": zD.length_chars,
        "grounding_violations": zD.grounding_violations,
        "meaning_tag_count": len(zD.meaning_tags),
    }

"""
Drift metrics for Agent D's narrative output over time.

Measures how D's output changes between consecutive invocations:
- Tag flip rate: how often tags appear/disappear between calls
- Narrative similarity: word-level Jaccard overlap between consecutive narratives
- Grounding violation trend: does grounding quality degrade over time?
- Tag stability: how consistent is the tag set across a window?
"""

from typing import List, Dict, Tuple
from state.schema import ZD


def tag_flip_rate(zd_prev: ZD, zd_curr: ZD) -> float:
    """
    Fraction of tags that changed between two consecutive D outputs.
    Returns value in [0, 1]. 0 = perfectly stable, 1 = completely different.
    """
    prev_tags = set(t.lower().strip() for t in zd_prev.meaning_tags)
    curr_tags = set(t.lower().strip() for t in zd_curr.meaning_tags)

    if not prev_tags and not curr_tags:
        return 0.0

    union = prev_tags | curr_tags
    intersection = prev_tags & curr_tags

    if not union:
        return 0.0

    # Symmetric difference / union = Jaccard distance
    return 1.0 - (len(intersection) / len(union))


def narrative_similarity(zd_prev: ZD, zd_curr: ZD) -> float:
    """
    Word-level Jaccard similarity between two narratives.
    Returns value in [0, 1]. 1 = identical, 0 = no overlap.
    """
    prev_words = set(zd_prev.narrative.lower().split())
    curr_words = set(zd_curr.narrative.lower().split())

    if not prev_words and not curr_words:
        return 1.0

    union = prev_words | curr_words
    if not union:
        return 1.0

    intersection = prev_words & curr_words
    return len(intersection) / len(union)


def compute_drift_series(zd_series: List[ZD]) -> Dict[str, List[float]]:
    """
    Compute drift metrics over a series of consecutive D outputs.

    Args:
        zd_series: List of ZD outputs in chronological order.

    Returns:
        Dict with keys:
        - tag_flip_rates: per-step tag flip rate (len = n-1)
        - narrative_similarities: per-step word Jaccard (len = n-1)
        - grounding_violations: per-step violation count (len = n)
        - narrative_lengths: per-step narrative length (len = n)
        - tag_counts: per-step tag count (len = n)
    """
    n = len(zd_series)
    result = {
        "tag_flip_rates": [],
        "narrative_similarities": [],
        "grounding_violations": [zd.grounding_violations for zd in zd_series],
        "narrative_lengths": [zd.length_chars for zd in zd_series],
        "tag_counts": [len(zd.meaning_tags) for zd in zd_series],
    }

    for i in range(1, n):
        result["tag_flip_rates"].append(tag_flip_rate(zd_series[i - 1], zd_series[i]))
        result["narrative_similarities"].append(
            narrative_similarity(zd_series[i - 1], zd_series[i])
        )

    return result


def windowed_tag_stability(zd_series: List[ZD], window: int = 5) -> List[float]:
    """
    Compute tag stability over sliding windows.
    For each window of `window` consecutive D outputs, compute the fraction
    of tags that appear in ALL outputs within the window.

    Returns a list of stability scores (len = n - window + 1).
    Higher = more stable.
    """
    n = len(zd_series)
    if n < window:
        return []

    stabilities = []
    for i in range(n - window + 1):
        w = zd_series[i:i + window]
        tag_sets = [set(t.lower().strip() for t in zd.meaning_tags) for zd in w]

        # Tags appearing in all windows
        if not tag_sets:
            stabilities.append(1.0)
            continue

        all_tags = set()
        for ts in tag_sets:
            all_tags |= ts

        if not all_tags:
            stabilities.append(1.0)
            continue

        # Intersection = tags present in every output
        common = tag_sets[0]
        for ts in tag_sets[1:]:
            common = common & ts

        stabilities.append(len(common) / len(all_tags))

    return stabilities


def drift_summary(zd_series: List[ZD]) -> Dict[str, float]:
    """
    Compute summary drift statistics over an entire series.
    """
    if len(zd_series) < 2:
        return {
            "mean_tag_flip_rate": 0.0,
            "mean_narrative_similarity": 1.0,
            "total_grounding_violations": 0,
            "mean_narrative_length": 0.0,
            "tag_stability_w5": 1.0,
            "drift_trend": 0.0,
        }

    series = compute_drift_series(zd_series)

    flip_rates = series["tag_flip_rates"]
    similarities = series["narrative_similarities"]
    violations = series["grounding_violations"]

    mean_flip = sum(flip_rates) / len(flip_rates) if flip_rates else 0.0
    mean_sim = sum(similarities) / len(similarities) if similarities else 1.0
    total_violations = sum(violations)
    mean_len = sum(series["narrative_lengths"]) / len(series["narrative_lengths"])

    # Tag stability over windows
    stabilities = windowed_tag_stability(zd_series, window=5)
    mean_stability = sum(stabilities) / len(stabilities) if stabilities else 1.0

    # Drift trend: is drift getting worse over time?
    # Compare first-half flip rate to second-half flip rate
    if len(flip_rates) >= 4:
        mid = len(flip_rates) // 2
        first_half = sum(flip_rates[:mid]) / mid
        second_half = sum(flip_rates[mid:]) / (len(flip_rates) - mid)
        drift_trend = second_half - first_half  # positive = drift increasing
    else:
        drift_trend = 0.0

    return {
        "mean_tag_flip_rate": mean_flip,
        "mean_narrative_similarity": mean_sim,
        "total_grounding_violations": total_violations,
        "mean_narrative_length": mean_len,
        "tag_stability_w5": mean_stability,
        "drift_trend": drift_trend,
    }

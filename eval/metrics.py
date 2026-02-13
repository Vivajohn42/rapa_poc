from typing import Dict, List, Tuple, Any
from state.schema import ZA

ACTIONS = ("up", "down", "left", "right")


def compute_symmetry_metrics(
    zA: ZA,
    predict_next_fn,
    score_action_seek_fn,
    score_action_avoid_fn,
) -> Dict[str, Any]:
    """
    Compute seek/avoid symmetry metrics for a single state.

    Args:
        zA: Current observation state
        predict_next_fn: AgentB.predict_next (callable)
        score_action_seek_fn: AgentC.score_action with mode=seek (callable(zA, zA_next) -> float)
        score_action_avoid_fn: AgentC.score_action with mode=avoid (callable(zA, zA_next) -> float)

    Returns dict with:
        - score_negation_error: avg |seek_score + avoid_score| (expect ~0)
        - ranking_inversion_score: how perfectly avoid reverses seek ranking (expect 1.0)
        - top_action_flipped: bool, whether top seek != top avoid
        - ab_prediction_valid: bool, whether all B predictions are within grid
    """
    scored_seek = []
    scored_avoid = []
    ab_valid = True

    for a in ACTIONS:
        zA_next = predict_next_fn(zA, a)

        x, y = zA_next.agent_pos
        if not (0 <= x < zA.width and 0 <= y < zA.height):
            ab_valid = False

        s_seek = score_action_seek_fn(zA, zA_next)
        s_avoid = score_action_avoid_fn(zA, zA_next)
        scored_seek.append((a, s_seek))
        scored_avoid.append((a, s_avoid))

    neg_err = score_negation_error(scored_seek, scored_avoid)

    rank_seek = ranking_from_scored(sorted(scored_seek, key=lambda x: x[1], reverse=True))
    rank_avoid = ranking_from_scored(sorted(scored_avoid, key=lambda x: x[1], reverse=True))
    inv_score = ranking_inversion_score(rank_seek, rank_avoid)

    top_seek = sorted(scored_seek, key=lambda x: x[1], reverse=True)[0][0]
    top_avoid = sorted(scored_avoid, key=lambda x: x[1], reverse=True)[0][0]

    return {
        "score_negation_error": neg_err,
        "ranking_inversion_score": inv_score,
        "top_action_flipped": top_seek != top_avoid,
        "ab_prediction_valid": ab_valid,
    }


def ab_identity_check(zA: ZA, predict_next_fn) -> Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Returns dict action -> (predicted_next_pos, predicted_next_pos)
    (The function itself doesn't know the goal; this is a helper for comparison.)
    We'll use it by calling it for SEEK and AVOID and comparing results.
    """
    out = {}
    for a in ACTIONS:
        zA_next = predict_next_fn(zA, a)
        out[a] = zA_next.agent_pos
    return out


def ranking_from_scored(scored: List[Tuple[str, float]]) -> List[str]:
    """Extract ranking order from scored list [(action, score), ...] already sorted descending."""
    return [a for a, _ in scored]


def ranking_inversion_score(rank_seek: List[str], rank_avoid: List[str]) -> float:
    """
    Measures how much AVOID ranking is the reverse of SEEK ranking.
    Returns score in [0,1], where 1 is perfect inversion.
    """
    # convert rank lists to position maps
    pos_seek = {a: i for i, a in enumerate(rank_seek)}
    pos_avoid = {a: i for i, a in enumerate(rank_avoid)}

    n = len(rank_seek)
    # perfect inversion means pos_avoid[a] == (n-1 - pos_seek[a])
    matches = 0
    for a in rank_seek:
        if pos_avoid[a] == (n - 1 - pos_seek[a]):
            matches += 1
    return matches / n

def spearman_corr(x: List[float], y: List[float]) -> float:
    """
    Minimal Spearman rank correlation without external deps.
    Ties handled by average ranks.
    Returns correlation in [-1,1].
    """
    def rankdata(a: List[float]) -> List[float]:
        # assign average ranks for ties
        sorted_idx = sorted(range(len(a)), key=lambda i: a[i])
        ranks = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[sorted_idx[j + 1]] == a[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)

    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    denx = sum((rx[i] - mx) ** 2 for i in range(len(rx)))
    deny = sum((ry[i] - my) ** 2 for i in range(len(ry)))
    den = (denx * deny) ** 0.5
    return 0.0 if den == 0 else num / den


def score_negation_error(scored_seek: List[Tuple[str, float]], scored_avoid: List[Tuple[str, float]]) -> float:
    """
    Average absolute error of (score_avoid + score_seek) across actions.
    For perfect negation, this is 0.
    """
    m_seek = {a: s for a, s in scored_seek}
    m_avoid = {a: s for a, s in scored_avoid}
    actions = sorted(m_seek.keys())
    return sum(abs(m_avoid[a] + m_seek[a]) for a in actions) / len(actions)

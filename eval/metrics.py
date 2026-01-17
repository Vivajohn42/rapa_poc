from typing import Dict, List, Tuple
from state.schema import ZA

ACTIONS = ("up", "down", "left", "right")


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

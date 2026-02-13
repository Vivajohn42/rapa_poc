"""
Statistical utilities for RAPA ablation evaluation.

Provides:
- 95% confidence intervals (t-distribution based)
- Mann-Whitney U test for comparing two variants
- Cohen's d effect size
- Aggregation with statistics for EpisodeResult lists
"""

import math
from typing import List, Tuple, Dict, Any, Optional


# ── t-distribution critical values (two-tailed 95%) ──────────────────────
# Precomputed for common df values; falls back to z=1.96 for df>120.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021,
    50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}


def _t_crit(df: int) -> float:
    """Approximate two-tailed t critical value at 95% confidence."""
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # find nearest key <= df
    keys = sorted(_T_CRIT_95.keys())
    for k in reversed(keys):
        if k <= df:
            return _T_CRIT_95[k]
    return 1.96  # fallback (large sample)


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def std(xs: List[float], ddof: int = 1) -> float:
    """Sample standard deviation."""
    if len(xs) <= ddof:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def confidence_interval_95(xs: List[float]) -> Tuple[float, float, float]:
    """
    Returns (mean, ci_low, ci_high) for a 95% confidence interval
    based on the t-distribution.
    """
    n = len(xs)
    if n < 2:
        m = mean(xs)
        return m, m, m

    m = mean(xs)
    se = std(xs) / math.sqrt(n)
    t = _t_crit(n - 1)
    return m, m - t * se, m + t * se


def confidence_interval_proportion(successes: int, n: int) -> Tuple[float, float, float]:
    """
    Wilson score interval for a proportion (e.g., success rate).
    Returns (proportion, ci_low, ci_high).
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    z = 1.96  # 95%
    p_hat = successes / n

    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)

    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return p_hat, max(0.0, lo), min(1.0, hi)


# ── Mann-Whitney U test ──────────────────────────────────────────────────

def mann_whitney_u(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """
    Two-sided Mann-Whitney U test (normal approximation).
    Returns (U_statistic, p_value).
    Suitable for n >= 20 per group.
    """
    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    # Rank all values together
    combined = [(v, 'x') for v in xs] + [(v, 'y') for v in ys]
    combined.sort(key=lambda t: t[0])

    # Assign average ranks for ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # Sum of ranks for x group
    r_x = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 'x')

    u_x = r_x - nx * (nx + 1) / 2
    u_y = nx * ny - u_x

    u = min(u_x, u_y)
    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)

    if sigma == 0:
        return u, 1.0

    z = (u - mu) / sigma
    # Two-tailed p-value approximation using the error function
    p = 2 * _normal_cdf(-abs(z))

    return u, p


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── Cohen's d ────────────────────────────────────────────────────────────

def cohens_d(xs: List[float], ys: List[float]) -> float:
    """
    Cohen's d effect size (pooled standard deviation).
    Positive means xs > ys.
    """
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 0.0

    mx, my = mean(xs), mean(ys)
    sx, sy = std(xs), std(ys)

    pooled_var = ((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-9

    return (mx - my) / pooled_sd


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ── Comparison report ────────────────────────────────────────────────────

def compare_variants(
    name_a: str, values_a: List[float],
    name_b: str, values_b: List[float],
    metric_name: str = "metric",
    is_proportion: bool = False,
) -> Dict[str, Any]:
    """
    Full statistical comparison of two variants on a single metric.
    Returns a dict with all statistics.
    """
    if is_proportion:
        # values are 0/1 booleans
        sa = int(sum(values_a))
        sb = int(sum(values_b))
        p_a, ci_a_lo, ci_a_hi = confidence_interval_proportion(sa, len(values_a))
        p_b, ci_b_lo, ci_b_hi = confidence_interval_proportion(sb, len(values_b))
        m_a, m_b = p_a, p_b
        ci_a = (ci_a_lo, ci_a_hi)
        ci_b = (ci_b_lo, ci_b_hi)
    else:
        m_a, ci_a_lo, ci_a_hi = confidence_interval_95(values_a)
        m_b, ci_b_lo, ci_b_hi = confidence_interval_95(values_b)
        ci_a = (ci_a_lo, ci_a_hi)
        ci_b = (ci_b_lo, ci_b_hi)

    u, p_value = mann_whitney_u(values_a, values_b)
    d = cohens_d(values_a, values_b)

    return {
        "metric": metric_name,
        "variant_a": name_a,
        "variant_b": name_b,
        "mean_a": m_a,
        "ci_a": ci_a,
        "mean_b": m_b,
        "ci_b": ci_b,
        "mann_whitney_u": u,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": d,
        "effect_size": effect_size_label(d),
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


def format_comparison(report: Dict[str, Any]) -> str:
    """Pretty-print a comparison report."""
    r = report
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
    return (
        f"  {r['metric']:25s}  "
        f"{r['variant_a']}: {r['mean_a']:.3f} [{r['ci_a'][0]:.3f}, {r['ci_a'][1]:.3f}]  vs  "
        f"{r['variant_b']}: {r['mean_b']:.3f} [{r['ci_b'][0]:.3f}, {r['ci_b'][1]:.3f}]  "
        f"p={r['p_value']:.4f}{sig}  d={r['cohens_d']:.2f}({r['effect_size']})"
    )

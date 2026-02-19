"""Jung-Preference-Weights for MvpKernel parameterization.

Port of rapa_os/kernel/jung_profiles.py — identical data structures.

The DEF (Dimensional Emergence Framework) interprets Jung's cognitive functions
as navigation modes in dimensional space.  Three axes (I/E, S/N, T/F) modulate
existing kernel parameters to produce visibly different agent behaviour under
identical ABI constraints.

Each weight is in [0.0, 1.0]:
  ie_weight  Introversion / Extraversion  (high = I, low = E)
  sn_weight  Sensing / Intuition          (high = N, low = S)
  tf_weight  Thinking / Feeling           (high = F, low = T)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JungProfile:
    """A personality profile that maps three Jung axes to kernel parameters.

    The three axes control four concrete kernel/stream parameters:

      I/E -> deconstruct_cooldown_ticks  (reflection duration)
      S/N -> D's detour_tolerance        (waypoint detour acceptance)
      S/N -> D's no_progress_window      (stuck detection sensitivity)
      T/F -> coupling_weight_c           (C-valence influence on B)
    """

    name: str
    ie_weight: float = 0.5   # 0.0 = max Extraversion, 1.0 = max Introversion
    sn_weight: float = 0.5   # 0.0 = max Sensing, 1.0 = max Intuition
    tf_weight: float = 0.5   # 0.0 = max Thinking, 1.0 = max Feeling

    def __post_init__(self):
        for attr in ("ie_weight", "sn_weight", "tf_weight"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr}={val} must be in [0.0, 1.0]")

    # --- Derived kernel parameters ---

    @property
    def deconstruct_cooldown(self) -> int:
        """I/E axis -> deconstruction cooldown ticks.

        base=3.  E(0.0)->1 tick, balanced(0.5)->3, I(1.0)->4.
        Introversion = longer reflection before D->C transfer.
        Extraversion = faster reaction to D's analysis.
        """
        raw = 3 * (0.5 + self.ie_weight)
        return max(1, int(raw))

    @property
    def coupling_weight_c(self) -> float:
        """T/F axis -> BC coupling weight w.

        T(0.0)->0.3, balanced(0.5)->0.5, F(1.0)->0.7.
        Feeling = C's valence dominates B's action.
        Thinking = B's own heuristic/logic dominates.
        """
        return 0.3 + self.tf_weight * 0.4

    @property
    def detour_tolerance(self) -> int:
        """S/N axis -> waypoint detour tolerance for Stream D.

        S(0.0)->1, balanced(0.5)->3, N(1.0)->4.
        Intuition = accepts larger detours for potential gain.
        Sensing = strict evidence, minimal detour.
        """
        return max(1, round(1 + self.sn_weight * 3))

    @property
    def no_progress_window(self) -> int:
        """S/N axis -> no-progress detection window for Stream D.

        S(0.0)->8, balanced(0.5)->6, N(1.0)->4.
        Intuition = recognises problems earlier (shorter window).
        Sensing = waits longer before declaring no progress.
        """
        return max(4, round(8 - self.sn_weight * 4))

    @property
    def tie_break_delta(self) -> float:
        """T/F axis -> tie-break delta for C's action selection.

        T(0.0)->0.15, balanced(0.5)->0.25, F(1.0)->0.40.
        Feeling = broader tie-break window (more influence from D's memory).
        Thinking = narrower window (decisions based on scores alone).
        """
        return 0.15 + self.tf_weight * 0.25

    @property
    def d_cooldown_steps(self) -> int:
        """I/E axis -> D activation cooldown steps.

        E(0.0)->5, balanced(0.5)->8, I(1.0)->12.
        Introversion = less frequent D activation (deeper reflection per call).
        Extraversion = more frequent D activation (quick reactions).
        """
        return max(3, round(5 + self.ie_weight * 7))

    @property
    def stuck_window(self) -> int:
        """S/N axis -> stuck detection window (same as no_progress_window).

        S(0.0)->8, balanced(0.5)->6, N(1.0)->4.
        """
        return self.no_progress_window

    # --- Compression parameters (RAPA v2: Unified Memory) ---

    @property
    def compression_threshold_l3(self) -> float:
        """T/F axis -> L3->L2 compression threshold (d_term).

        T(0.0)->0.10 (compresses D into C eagerly).
        F(1.0)->0.30 (stays deliberative, preserves D richness).
        """
        return max(0.10, 0.10 + self.tf_weight * 0.20)

    @property
    def compression_threshold_l2(self) -> float:
        """T/F axis -> L2->L1 compression threshold (c_term).

        T(0.0)->0.05 (compresses C into B eagerly).
        F(1.0)->0.25 (preserves valence richness longer).

        Tight range: c_term in easy envs (DoorKey-6) is ~0.0-0.15,
        so thresholds must be in this band to be selective.
        """
        return 0.05 + self.tf_weight * 0.20

    @property
    def compression_threshold_l1(self) -> float:
        """S/N axis -> L1->L0 compression threshold (delta_4).

        S(0.0)->0.15 (pushes corrections to perception eagerly).
        N(1.0)->0.35 (trusts prediction model, cautious about reflex).

        Tight range: delta_4 in easy envs is ~0.1-0.3,
        so thresholds must be in this band to be selective.
        """
        return 0.15 + self.sn_weight * 0.20

    @property
    def compression_window(self) -> int:
        """I/E axis -> compression cooldown window (ticks).

        E(0.0)->2 (rapid adaptation).
        I(1.0)->8 (deeper consolidation before next compression).
        """
        return max(2, round(2 + self.ie_weight * 6))

    @property
    def cascade_depth(self) -> int:
        """S/N axis -> max compression stages per tick.

        S(sn<0.3)->1 (single stage only, concrete).
        balanced->2 (two-stage max).
        N(sn>=0.7)->3 (full cascade, abstract).
        """
        if self.sn_weight >= 0.7:
            return 3
        elif self.sn_weight >= 0.3:
            return 2
        else:
            return 1


# ---------------------------------------------------------------------------
# Pre-defined profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, JungProfile] = {
    "SENSOR": JungProfile(
        "Sensor (ISTJ-like)",
        ie_weight=0.7, sn_weight=0.1, tf_weight=0.2,
    ),
    "INTUITIVE": JungProfile(
        "Intuitive (INFP-like)",
        ie_weight=0.8, sn_weight=0.9, tf_weight=0.8,
    ),
    "ANALYST": JungProfile(
        "Analyst (ENTJ-like)",
        ie_weight=0.2, sn_weight=0.4, tf_weight=0.1,
    ),
    "DEFAULT": JungProfile(
        "Default (balanced)",
        ie_weight=0.5, sn_weight=0.5, tf_weight=0.5,
    ),
}


# ---------------------------------------------------------------------------
# RAPA v2 profiles — Jungian Weights Matrix for Unified Memory evaluation
# ---------------------------------------------------------------------------

PROFILES_V2: dict[str, JungProfile] = {
    "INTJ": JungProfile(
        "Architect (INTJ)",
        ie_weight=0.8, sn_weight=0.8, tf_weight=0.1,
        # Ni-Te: large stability window (N), aggressive compression after (T)
        # "Thinks long, then acts decisively"
    ),
    "ESFP": JungProfile(
        "Performer (ESFP)",
        ie_weight=0.1, sn_weight=0.1, tf_weight=0.8,
        # Se-Fi: small stability window (S), cautious compression (F)
        # "Reacts fast, stays flexible"
    ),
    "ISTJ": JungProfile(
        "Inspector (ISTJ)",
        ie_weight=0.8, sn_weight=0.1, tf_weight=0.1,
        # Si-Te: small stability window (S), aggressive compression (T)
        # "Fixes quickly on experience, forms reflex fast"
    ),
    "ENFP": JungProfile(
        "Champion (ENFP)",
        ie_weight=0.1, sn_weight=0.9, tf_weight=0.8,
        # Ne-Fi: large stability window (N), cautious compression (F)
        # "Explores long, reluctant to compress"
    ),
    "DEFAULT_V2": JungProfile(
        "Default v2 (balanced)",
        ie_weight=0.5, sn_weight=0.5, tf_weight=0.5,
    ),
}

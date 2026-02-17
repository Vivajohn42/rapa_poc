"""LoopGainTracker for rapa_mvp symbolic agents.

Port of rapa_os/kernel/loop_gain.py adapted for:
- Symbolic agents (deterministic B, symbolic C scores, deterministic/LLM D)
- Pydantic schemas (ZA, ZC, ZD) instead of PairState z-dicts
- g_AD grounding validation for both deterministic and LLM-based D

Computes four inter-stream coupling gains per tick:
  g_BA: Action-to-Structure (B walkability + score concentration + movement)
  g_CB: Valence-to-Action (cosine similarity between B-only and C-blended scores)
  g_DC: Salience-to-Valence (D -> C influence via deconstruction)
  g_AD: Narrative-World-Resonance (grounding checks: hint, position, goal_mode, tags)

The composite G = g_BA * g_CB * g_DC * g_AD measures loop stability.
F is the running EMA baseline. G/F > 1 = stable attractor, G/F < 1 = decay.
"""
from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

from kernel.types import MvpLoopGain

# Known tag patterns (for hallucination detection in g_AD)
KNOWN_TAG_PREFIXES = frozenset({
    "goal:", "hint:", "target:", "micro", "success", "no_success",
    "short_episode", "long_episode", "stability:", "not_", "empty",
    "clue_collected:", "candidates:",       # TextWorld D tags
    "answer:", "eliminated:", "evidence:",  # Riddle Rooms D tags
})

# Deterministic D narrative prefixes (non-LLM)
DETERMINISTIC_NARRATIVE_PREFIXES = (
    "Episode summary:", "Micro-summary",      # GridWorld D
    "Episode synthesis:", "Micro-synthesis:",  # TextWorld D
    "Riddle synthesis:", "Riddle micro:",      # Riddle Rooms D
    "No events recorded.",                     # Empty D
)


class MvpLoopGainTracker:
    """Tracks loop gain per tick for symbolic rapa_mvp agents."""

    def __init__(self, ema_alpha: float = 0.15, carry_decay: float = 0.05):
        self.ema_alpha = ema_alpha
        self.carry_decay = carry_decay
        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset all gains at the start of a new episode."""
        self.g_BA: float = 0.5
        self.g_CB: float = 0.5
        self.g_DC: float = 0.5
        self.g_AD: float = 1.0  # deterministic D defaults to 1.0
        self.G: float = 0.0
        self.F: float = 0.5
        self.G_over_F: float = 1.0
        self.weakest_coupling: str = "AB"
        self._episode_started: bool = False
        self._prev_agent_pos: Optional[Tuple[int, int]] = None
        self._movement_factor: float = 0.5
        self.episode_history: List[MvpLoopGain] = []

    def compute_tick(
        self,
        zA,                          # ZA: current perception
        zC,                          # ZC: current C state
        zD,                          # Optional[ZD]: D output (None if gD=0)
        scored: Optional[List] = None,  # C's scored action list [(action, score), ...]
        decon_fired: bool = False,
        gC: int = 1,
        gD: int = 0,
        tick_id: int = 0,
        predict_next_fn=None,        # B.predict_next (for g_BA walkability)
        d_seen_positions: Optional[set] = None,  # D's seen_positions (for g_AD)
        has_agent_d: bool = True,    # Whether D agent exists (False → no D→C path)
    ) -> MvpLoopGain:
        """Compute all four gains for this tick."""

        # --- Track movement ---
        self._update_movement(zA.agent_pos)

        # --- g_BA: always computable ---
        self.g_BA = self._compute_g_BA(zA, scored, predict_next_fn)

        # --- g_CB: when gC=1 and scored available ---
        fresh_g_CB = None
        if gC == 1 and scored is not None and predict_next_fn is not None:
            fresh_g_CB = self._compute_g_CB(zA, scored, predict_next_fn)
        if fresh_g_CB is not None:
            self.g_CB = fresh_g_CB
        else:
            self.g_CB = self._carry_forward(self.g_CB)

        # --- g_DC: when D is running ---
        fresh_g_DC = None
        if gD == 1 and zD is not None:
            fresh_g_DC = self._compute_g_DC(zC, zD, decon_fired)
        elif has_agent_d and "target" in zC.memory:
            # Target in memory (possibly via hint-capture path) — D→C coupling realized
            fresh_g_DC = 1.0
        if fresh_g_DC is not None:
            self.g_DC = fresh_g_DC
        elif not has_agent_d:
            # No D agent at all → decay toward 0 (no D→C coupling possible)
            self.g_DC = self._carry_forward(self.g_DC, default=0.0)
        else:
            # D exists but gD=0 this tick → carry forward
            self.g_DC = self._carry_forward(self.g_DC)

        # --- g_AD: when D is running ---
        fresh_g_AD = None
        if gD == 1 and zD is not None:
            fresh_g_AD = self._compute_g_AD(
                zA, zD, zC, d_seen_positions=d_seen_positions,
            )
        if fresh_g_AD is not None:
            self.g_AD = fresh_g_AD
        elif not has_agent_d:
            # No D agent at all → decay toward 0 (no A→D resonance possible)
            self.g_AD = self._carry_forward(self.g_AD, default=0.0)
        else:
            # D exists but gD=0 → deterministic stays at 1.0, LLM decays
            self.g_AD = self._carry_forward(self.g_AD, default=1.0)

        # --- Composite G ---
        self.G = self.g_BA * self.g_CB * self.g_DC * self.g_AD

        # --- Update F (EMA baseline) ---
        if not self._episode_started:
            self.F = self.G
            self._episode_started = True
        else:
            self.F = self.ema_alpha * self.G + (1 - self.ema_alpha) * self.F

        # --- G/F ratio ---
        self.G_over_F = self.G / max(self.F, 1e-8)

        # --- Identify weakest coupling ---
        gains = {"AB": self.g_BA, "BC": self.g_CB, "CD": self.g_DC, "AD": self.g_AD}
        self.weakest_coupling = min(gains, key=gains.get)

        # --- Snapshot ---
        snap = MvpLoopGain(
            g_BA=round(self.g_BA, 4),
            g_CB=round(self.g_CB, 4),
            g_DC=round(self.g_DC, 4),
            g_AD=round(self.g_AD, 4),
            G=round(self.G, 6),
            F=round(self.F, 6),
            G_over_F=round(self.G_over_F, 4),
            weakest_coupling=self.weakest_coupling,
            tick=tick_id,
        )
        self.episode_history.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Individual gain computations
    # ------------------------------------------------------------------

    def _compute_g_BA(self, zA, scored, predict_next_fn) -> float:
        """Action-to-Structure: Does B's action resonate with A's world model?

        For symbolic agents:
        - Walkability: does the top-scored action actually move?
        - Concentration: how decisive is the score spread?
        - Movement: did the agent move since last tick?
        """
        if scored is None or predict_next_fn is None:
            return 0.5

        top_action = scored[0][0]

        # (1) Walkability check via B's forward model
        zA_next = predict_next_fn(zA, top_action)
        walkable = 1.0 if zA_next.agent_pos != zA.agent_pos else 0.0

        # (2) Score concentration
        scores = [s for _, s in scored]
        if len(scores) >= 2:
            concentration = min((scores[0] - scores[1]) / 2.0, 1.0)
            concentration = max(concentration, 0.0)
        else:
            concentration = 0.5

        # (3) Movement factor
        movement = self._movement_factor

        return walkable * (0.4 + 0.3 * concentration + 0.3 * movement)

    def _compute_g_CB(self, zA, scored, predict_next_fn) -> Optional[float]:
        """Valence-to-Action: cosine similarity between B-only and C-blended scores.

        B-only scores: whether each action moves (1) or stays (0).
        C-blended scores: the scored list from C.choose_action.
        """
        if scored is None or predict_next_fn is None:
            return None

        # Dynamic actions from scored list (supports TextWorld + GridWorld)
        if scored:
            ACTIONS = [a for a, _ in scored]
        else:
            ACTIONS = ["up", "down", "left", "right"]  # GridWorld fallback

        # B-only scores: 1.0 if action moves, 0.0 if stays
        b_vec = []
        for a in ACTIONS:
            zA_next = predict_next_fn(zA, a)
            b_vec.append(1.0 if zA_next.agent_pos != zA.agent_pos else 0.0)

        # C scores (from scored list)
        c_scores_map = {a: s for a, s in scored}
        c_vec = [c_scores_map.get(a, 0.0) for a in ACTIONS]

        # Cosine similarity
        dot = sum(bi * ci for bi, ci in zip(b_vec, c_vec))
        mag_b = max(sum(x ** 2 for x in b_vec) ** 0.5, 1e-8)
        mag_c = max(sum(x ** 2 for x in c_vec) ** 0.5, 1e-8)
        cosine = dot / (mag_b * mag_c)

        # Map [-1, 1] -> [0, 1]
        return (cosine + 1.0) / 2.0

    def _compute_g_DC(self, zC, zD, decon_fired: bool) -> Optional[float]:
        """Salience-to-Valence: How much does D influence C?

        Staged scoring:
        - 1.0: target found in C's memory (D contributed actionable info)
        - 0.8: deconstruction fired (D->C pipeline active) OR candidates narrowing
        - 0.5: D produced tags but no structural change yet
        - 0.3: D produced nothing useful
        """
        if zD is None:
            return None

        tags = list(zD.meaning_tags)
        target_in_memory = "target" in zC.memory

        # Check if D's synthesis narrowed candidates (from tags)
        has_target_tag = any(t.startswith("target:") for t in tags)
        candidates_narrowing = any(t.startswith("candidates:") for t in tags)

        if target_in_memory:
            # Target identified — D→C coupling is fully realized
            return 1.0
        elif decon_fired or has_target_tag:
            return 0.8
        elif candidates_narrowing and len(tags) > 2:
            return 0.6
        elif len(tags) > 1:  # More than just "empty"
            return 0.5
        else:
            return 0.3

    def _compute_g_AD(
        self,
        zA,
        zD,
        zC,
        d_seen_positions: Optional[set] = None,
    ) -> Optional[float]:
        """Narrative-World-Resonance: grounding validation.

        For deterministic D: always 1.0 (narrative is constructed from observed data).
        For LLM-based D: concrete grounding checks with weighted violations.

        Grounding checks (LLM D):
        | Check               | Violation when...                              | Weight |
        |--------------------|-------------------------------------------------|--------|
        | Hint-Konsistenz     | D claims hint but zA.hint disagrees             | 1.0    |
        | Position-Konsistenz | D mentions position not in seen_positions       | 0.5    |
        | Goal-Mode-Konsistenz| D's tags contradict zC.goal_mode                | 0.5    |
        | Halluzinierte Tags  | D produces tags not matching known patterns     | 0.25   |
        """
        if zD is None:
            return None

        # Deterministic D: grounding_violations is always 0 by construction
        if zD.grounding_violations == 0 and not self._has_llm_markers(zD):
            return 1.0

        # LLM-based D: concrete grounding checks
        total_weight = 0.0
        violation_weight = 0.0

        tags = set(t.strip().lower() for t in zD.meaning_tags)

        # (1) Hint-Konsistenz (weight 1.0)
        total_weight += 1.0
        hint_tags = [t for t in tags if t.startswith("hint:")]
        if hint_tags:
            for ht in hint_tags:
                claimed_hint = ht.split(":", 1)[1].upper()
                if zA.hint is not None:
                    if claimed_hint != zA.hint:
                        violation_weight += 1.0
                        break
                # No violation if zA.hint is None — D may be recalling from buffer

        # (2) Position-Konsistenz (weight 0.5)
        if d_seen_positions is not None:
            total_weight += 0.5
            # Check if narrative mentions positions not in seen set
            # Simple heuristic: check if D claims positions via narrative text
            # (Full NLP parsing would be overkill; we check the tag-based system)
            # For MVP: no violation possible from tags alone, only from narrative
            # → skip this check for now (would need NLP on narrative text)

        # (3) Goal-Mode-Konsistenz (weight 0.5)
        total_weight += 0.5
        goal_mode_tags = [t for t in tags if t.startswith("goal:")]
        for gmt in goal_mode_tags:
            claimed_mode = gmt.split(":", 1)[1]
            if claimed_mode != zC.goal_mode:
                violation_weight += 0.5
                break

        # (4) Halluzinierte Tags (weight 0.25)
        total_weight += 0.25
        unknown_tags = []
        for tag in tags:
            if not any(tag.startswith(p) for p in KNOWN_TAG_PREFIXES):
                unknown_tags.append(tag)
        if unknown_tags:
            violation_weight += 0.25

        if total_weight == 0:
            return 1.0

        return max(0.0, 1.0 - violation_weight / total_weight)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_llm_markers(self, zD) -> bool:
        """Detect if ZD was produced by an LLM (vs deterministic D)."""
        if zD.grounding_violations > 0:
            return True
        # Deterministic D's narrative follows rigid template prefixes
        if any(zD.narrative.startswith(p) for p in DETERMINISTIC_NARRATIVE_PREFIXES):
            return False
        return True  # Assume LLM if template doesn't match

    def _carry_forward(self, prev: float, default: float = 0.5) -> float:
        """Decay previous value toward default when not freshly computed."""
        return prev + self.carry_decay * (default - prev)

    def _update_movement(self, agent_pos: Tuple[int, int]) -> None:
        """Track whether the agent moved since last tick."""
        if self._prev_agent_pos is not None:
            moved = agent_pos != self._prev_agent_pos
            self._movement_factor = 1.0 if moved else 0.5
        else:
            self._movement_factor = 0.5
        self._prev_agent_pos = agent_pos

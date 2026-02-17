"""Closure Residuum tracker for rapa_mvp.

Computes the multi-dimensional distance from equilibrium fixpoint
each tick.  Sits alongside LoopGainTracker (which measures coupling
strength) — together they form the complete stability matrix.

Three terms:
  delta_4:  Prediction error (B's model vs actual world)
  c_term:   Valence-reference alignment (C scoring vs optimal)
  d_term:   Meaning-structure alignment (D narrative vs world)

Composed:
  delta_6 = sqrt(delta_4^2 + lambda_1 * c_term^2)       — 6 FoM
  delta_8 = sqrt(delta_4^2 + lambda_1 * c_term^2
                            + lambda_2 * d_term^2)       — 8 FoM

At fixpoint all three terms are zero.

Dynamic thresholds (no magic constants):
  ema_alpha:  derived from schedule topology (1 / information cycle length)
  divergence_threshold:  own sigma * I/E modulation
  activation_threshold:  own EMA baseline * S/N modulation
"""
from __future__ import annotations

import math
from typing import Optional, List, Tuple

from kernel.types import ResidualSnapshot


class ClosureResiduum:
    """Passive delta_8 computation per tick."""

    def __init__(
        self,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        carry_decay: float = 0.05,
        *,
        n_schedule_templates: int = 2,
        ie_weight: float = 0.5,
        sn_weight: float = 0.5,
    ):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.carry_decay = carry_decay

        # --- EMA alpha derived from schedule topology ---
        # Full information cycle = n_templates * n_full_rotations + 1 (A-D lag)
        # 4FoM (1 template): 1 rotation sufficient -> alpha = 1/(1*1+1) = 0.50
        # 6FoM (2 templates): 3 rotations = 6 ticks -> alpha = 1/(2*3+1) = 0.143
        # 8FoM (3 templates): 2 rotations = 6 ticks -> alpha = 1/(3*2+1) = 0.143
        if n_schedule_templates <= 1:
            n_full_rotations = 1
        elif n_schedule_templates == 2:
            n_full_rotations = 3
        else:
            n_full_rotations = 2
        self.ema_alpha = 1.0 / (n_schedule_templates * n_full_rotations + 1)

        # --- Jung profile weights for dynamic thresholds ---
        self._ie_weight = ie_weight  # I/E: modulates divergence threshold
        self._sn_weight = sn_weight  # S/N: modulates activation threshold

        # Weakest-coupling frequency counters (for adaptive lambda)
        self._bc_ratio: float = 0.5
        self._cd_ratio: float = 0.5

        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self.episode_history: List[ResidualSnapshot] = []
        self._prev_delta_8: Optional[float] = None
        self._prev_predicted_pos: Optional[Tuple[int, int]] = None
        self._prev_c_term: float = 0.5
        self._prev_d_term: float = 0.5

        # Rolling window for sigma computation
        self._delta_8_window: List[float] = []
        # EMA baseline of delta_8 (analog to F in loop gain)
        self._F_delta_8: float = 0.5

    def compute_tick(
        self,
        zA,                          # ZA: current perception
        zC,                          # ZC: current C state
        zD,                          # Optional[ZD]: D output (None if gD=0)
        scored: Optional[List] = None,
        predict_next_fn=None,
        gC: int = 1,
        gD: int = 0,
        tick_id: int = 0,
        has_agent_d: bool = True,
        weakest_coupling: str = "AB",
    ) -> ResidualSnapshot:
        """Compute closure residuum for this tick.

        All three terms are in [0, 1]:
          0.0 = perfect alignment (fixpoint)
          1.0 = maximum deviation
        """
        # --- delta_4: prediction error (retroactive) ---
        delta_4 = self._compute_delta_4(zA, scored, predict_next_fn)

        # --- c_term: valence-reference alignment ---
        c_term = self._compute_c_term(zA, zC, scored, gC)

        # --- d_term: meaning-structure alignment ---
        d_term = self._compute_d_term(zA, zC, zD, gD, has_agent_d)

        # --- Compose delta_6 and delta_8 ---
        delta_6 = math.sqrt(
            delta_4 ** 2 + self.lambda_1 * c_term ** 2
        )
        delta_8 = math.sqrt(
            delta_4 ** 2
            + self.lambda_1 * c_term ** 2
            + self.lambda_2 * d_term ** 2
        )

        # --- d(delta_8)/dt ---
        if self._prev_delta_8 is not None:
            d_delta_8_dt = delta_8 - self._prev_delta_8
        else:
            d_delta_8_dt = 0.0
        self._prev_delta_8 = delta_8

        # --- Update rolling window for sigma ---
        self._delta_8_window.append(delta_8)
        max_window = max(5, int(1.0 / self.ema_alpha))
        if len(self._delta_8_window) > max_window:
            self._delta_8_window = self._delta_8_window[-max_window:]

        # --- Update F_delta_8 (EMA baseline) ---
        self._F_delta_8 += self.ema_alpha * (delta_8 - self._F_delta_8)

        # --- Adaptive lambda calibration ---
        self.adapt_lambdas(weakest_coupling)

        # --- Snapshot ---
        snap = ResidualSnapshot(
            delta_4=round(delta_4, 4),
            c_term=round(c_term, 4),
            d_term=round(d_term, 4),
            delta_6=round(delta_6, 4),
            delta_8=round(delta_8, 4),
            d_delta_8_dt=round(d_delta_8_dt, 4),
            lambda_1=round(self.lambda_1, 4),
            lambda_2=round(self.lambda_2, 4),
            tick=tick_id,
            sigma_delta_8=round(self.sigma_delta_8, 4),
            F_delta_8=round(self._F_delta_8, 4),
            divergence_thr=round(self.divergence_threshold, 4),
            activation_thr=round(self.activation_threshold, 4),
        )
        self.episode_history.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Dynamic thresholds (derived, not tuned)
    # ------------------------------------------------------------------

    @property
    def sigma_delta_8(self) -> float:
        """Rolling standard deviation of delta_8.

        Window size = effective EMA window (1/alpha).  Returns a safe
        fallback (based on ema_alpha) until enough data is collected.
        """
        if len(self._delta_8_window) < 3:
            # Fallback: approximate sigma from alpha
            # (at alpha=0.143, fallback=0.143 ≈ 0.15 — matches old value)
            return self.ema_alpha
        mean = sum(self._delta_8_window) / len(self._delta_8_window)
        var = sum((x - mean) ** 2 for x in self._delta_8_window) / len(self._delta_8_window)
        return max(0.01, var ** 0.5)

    @property
    def divergence_threshold(self) -> float:
        """Dynamic dDelta_8/dt threshold for deconstruction trigger.

        threshold = max(0.2, 1.0 - ie_weight) * sigma(delta_8)

        Extravert (ie=0.0):  threshold = 1.0 * sigma  (reacts at 1 std dev)
        Balanced  (ie=0.5):  threshold = 0.5 * sigma
        Introvert (ie=1.0):  threshold = 0.2 * sigma  (floor, rarely reacts)
        """
        ie_factor = max(0.2, 1.0 - self._ie_weight)
        return ie_factor * self.sigma_delta_8

    @property
    def activation_threshold(self) -> float:
        """Dynamic delta_8 threshold for D activation in router.

        activate_d = (delta_8 > F_delta_8 * activation_ratio)

        activation_ratio = 1.0 + (1.0 - sn_weight) * 0.5

        Intuitive (sn=1.0):  ratio = 1.0  (any exceedance triggers D)
        Balanced  (sn=0.5):  ratio = 1.25
        Sensor    (sn=0.0):  ratio = 1.5  (delta_8 must be 50% above baseline)
        """
        activation_ratio = 1.0 + (1.0 - self._sn_weight) * 0.5
        return self._F_delta_8 * activation_ratio

    # ------------------------------------------------------------------
    # Individual term computations
    # ------------------------------------------------------------------

    def _compute_delta_4(
        self, zA, scored, predict_next_fn,
    ) -> float:
        """Prediction Error: B's model vs actual world (retroactive).

        Compares B's prediction from tick t-1 with the actual observation
        at tick t.  This measures the Existence-Happening discrepancy
        (model error), NOT decision quality (which is g_BA).

        - Agent walks into known wall: delta_4 = 0 (B predicted correctly)
        - B predicts free path but obstacle blocks: delta_4 > 0 (model error)
        """
        # --- Retroactive check: previous prediction vs current observation ---
        if self._prev_predicted_pos is not None and zA is not None:
            pred = self._prev_predicted_pos
            actual = zA.agent_pos
            if pred == actual:
                delta_4 = 0.0  # Model correct
            else:
                dx = abs(pred[0] - actual[0])
                dy = abs(pred[1] - actual[1])
                max_dist = max(zA.width, zA.height, 1)
                delta_4 = min(1.0, (dx + dy) / max_dist)
        else:
            delta_4 = 0.5  # Unknown (first tick)

        # --- Store prediction for next tick ---
        if scored and predict_next_fn and zA is not None:
            top_action = scored[0][0]
            zA_pred = predict_next_fn(zA, top_action)
            self._prev_predicted_pos = zA_pred.agent_pos
        else:
            self._prev_predicted_pos = None

        return delta_4

    def _compute_c_term(
        self, zA, zC, scored, gC: int,
    ) -> float:
        """Valence-reference alignment: C scoring vs optimal.

        When target is known: measures whether top action reduces distance
        to target optimally.  When no target: c_term = 1.0 (max uncertainty).

        At fixpoint (agent moves directly toward target): c_term -> 0.
        """
        if gC == 0 or scored is None:
            # C not active — carry forward
            self._prev_c_term = self._carry_forward(self._prev_c_term)
            return self._prev_c_term

        target = zC.memory.get("target")
        if target is None:
            # No target known — maximum valence uncertainty
            self._prev_c_term = 1.0
            return 1.0

        target = tuple(target)
        agent_pos = zA.agent_pos

        # Current distance to target
        cur_dist = abs(agent_pos[0] - target[0]) + abs(agent_pos[1] - target[1])

        if cur_dist == 0:
            # Already at target — perfect alignment
            self._prev_c_term = 0.0
            return 0.0

        # Top action's resulting distance (from scored list)
        top_action = scored[0][0]
        # We need the predicted next position for the top action
        # Use scored scores as proxy: higher score = better alignment
        top_score = scored[0][1]
        if len(scored) >= 2:
            worst_score = scored[-1][1]
            score_range = top_score - worst_score
            if score_range > 0:
                # Normalized: 0 = top action is much better, 1 = all equal
                c_term = 1.0 - min(1.0, score_range / max(cur_dist, 1))
            else:
                # All actions score equally — no valence signal
                c_term = 1.0
        else:
            c_term = 0.5  # Single action — partial information

        c_term = max(0.0, min(1.0, c_term))
        self._prev_c_term = c_term
        return c_term

    def _compute_d_term(
        self, zA, zC, zD, gD: int, has_agent_d: bool,
    ) -> float:
        """Meaning-structure alignment: D narrative vs world.

        When D is active and has identified target correctly: d_term -> 0.
        When D produces grounding violations: d_term increases.
        When D is absent: carry forward with decay.
        """
        if gD == 1 and zD is not None:
            # D is active this tick — compute fresh
            target_in_memory = "target" in zC.memory
            tags = list(zD.meaning_tags)
            has_target_tag = any(t.startswith("target:") for t in tags)
            violations = zD.grounding_violations

            if target_in_memory and has_target_tag and violations == 0:
                # D identified target correctly, no hallucinations
                d_term = 0.0
            elif target_in_memory:
                # Target in memory but some issues
                d_term = 0.1 + 0.1 * violations
            elif has_target_tag:
                # D produced target tag but not yet in memory
                d_term = 0.3
            elif len(tags) > 1:
                # D produced some tags (progress)
                d_term = 0.5
            else:
                # D produced nothing useful
                d_term = 0.8

            d_term = min(1.0, d_term)
            self._prev_d_term = d_term
            return d_term

        elif has_agent_d and "target" in zC.memory:
            # D exists, not active this tick, but target already in memory
            self._prev_d_term = 0.0
            return 0.0

        elif not has_agent_d:
            # No D agent at all — decay toward 1.0 (max uncertainty)
            self._prev_d_term = self._carry_forward(
                self._prev_d_term, default=1.0
            )
            return self._prev_d_term

        else:
            # D exists but gD=0 — carry forward
            self._prev_d_term = self._carry_forward(self._prev_d_term)
            return self._prev_d_term

    # ------------------------------------------------------------------
    # Adaptive lambda calibration (Phase 2)
    # ------------------------------------------------------------------

    def adapt_lambdas(self, weakest_coupling: str) -> None:
        """Frequency-based lambda adaptation.

        Lambda weights converge to the ratio of weakest-coupling
        frequencies, normalized so that lambda_1 + lambda_2 = 2.0.
        No external hyperparameter beyond EMA alpha.
        """
        # Update frequency estimators (EMA)
        is_c_weak = 1.0 if weakest_coupling in ("BC", "AC") else 0.0
        is_d_weak = 1.0 if weakest_coupling in ("CD", "AD") else 0.0

        self._bc_ratio += self.ema_alpha * (is_c_weak - self._bc_ratio)
        self._cd_ratio += self.ema_alpha * (is_d_weak - self._cd_ratio)

        # Normalize to sum = 2.0
        epsilon = 0.01
        total = self._bc_ratio + self._cd_ratio + epsilon
        lambda_1_target = 2.0 * self._bc_ratio / total
        lambda_2_target = 2.0 * self._cd_ratio / total

        # Smooth convergence
        self.lambda_1 += self.ema_alpha * (lambda_1_target - self.lambda_1)
        self.lambda_2 += self.ema_alpha * (lambda_2_target - self.lambda_2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _carry_forward(self, prev: float, default: float = 0.5) -> float:
        """Decay previous value toward default when not freshly computed."""
        return prev + self.carry_decay * (default - prev)

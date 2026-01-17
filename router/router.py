from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RouterConfig:
    d_every_k_steps: int = 10
    stuck_window: int = 4
    enable_stuck_trigger: bool = True

    enable_uncertainty_trigger: bool = True
    uncertainty_threshold: float = 0.25  # NEW: smaller => stricter

    d_cooldown_steps: int = 8


class Router:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self._cooldown_until_t = -1

    def should_activate_d(
        self,
        t: int,
        last_positions: Tuple[Tuple[int, int], ...],
        decision_delta: Optional[float] = None
    ) -> tuple[bool, str]:
        # Cooldown gate
        if t < self._cooldown_until_t:
            return False, "cooldown"

        # Uncertainty trigger (highest priority)
        if self.cfg.enable_uncertainty_trigger and decision_delta is not None:
            if decision_delta < self.cfg.uncertainty_threshold:
                self._cooldown_until_t = t + self.cfg.d_cooldown_steps
                return True, f"uncertainty_delta<{self.cfg.uncertainty_threshold:.2f}"

        # Periodic trigger
        if self.cfg.d_every_k_steps > 0 and (t > 0) and (t % self.cfg.d_every_k_steps == 0):
            self._cooldown_until_t = t + self.cfg.d_cooldown_steps
            return True, f"periodic_every_{self.cfg.d_every_k_steps}"

        # Stuck trigger
        if self.cfg.enable_stuck_trigger and len(last_positions) >= self.cfg.stuck_window:
            window = last_positions[-self.cfg.stuck_window:]
            if len(set(window)) == 1:
                self._cooldown_until_t = t + self.cfg.d_cooldown_steps
                return True, f"stuck_{self.cfg.stuck_window}"

        return False, "none"

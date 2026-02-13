from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict


@dataclass
class RouterConfig:
    d_every_k_steps: int = 10
    stuck_window: int = 4
    enable_stuck_trigger: bool = True

    enable_uncertainty_trigger: bool = True
    uncertainty_threshold: float = 0.25  # smaller => stricter

    d_cooldown_steps: int = 8


@dataclass
class RegimeStep:
    """Record of the active regime at a single timestep."""
    t: int
    regime: str       # "3D" or "4D" (2D not used in current arch since C is always on)
    d_activated: bool
    trigger: str      # reason D was activated, or "none"
    decision_delta: Optional[float] = None


class Router:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self._cooldown_until_t = -1
        self.regime_log: List[RegimeStep] = []

    def should_activate_d(
        self,
        t: int,
        last_positions: Tuple[Tuple[int, int], ...],
        decision_delta: Optional[float] = None
    ) -> tuple[bool, str]:
        # Cooldown gate
        if t < self._cooldown_until_t:
            self.regime_log.append(RegimeStep(
                t=t, regime="3D", d_activated=False,
                trigger="cooldown", decision_delta=decision_delta,
            ))
            return False, "cooldown"

        # Uncertainty trigger (highest priority)
        if self.cfg.enable_uncertainty_trigger and decision_delta is not None:
            if decision_delta < self.cfg.uncertainty_threshold:
                self._cooldown_until_t = t + self.cfg.d_cooldown_steps
                trigger = f"uncertainty_delta<{self.cfg.uncertainty_threshold:.2f}"
                self.regime_log.append(RegimeStep(
                    t=t, regime="4D", d_activated=True,
                    trigger=trigger, decision_delta=decision_delta,
                ))
                return True, trigger

        # Periodic trigger
        if self.cfg.d_every_k_steps > 0 and (t > 0) and (t % self.cfg.d_every_k_steps == 0):
            self._cooldown_until_t = t + self.cfg.d_cooldown_steps
            trigger = f"periodic_every_{self.cfg.d_every_k_steps}"
            self.regime_log.append(RegimeStep(
                t=t, regime="4D", d_activated=True,
                trigger=trigger, decision_delta=decision_delta,
            ))
            return True, trigger

        # Stuck trigger
        if self.cfg.enable_stuck_trigger and len(last_positions) >= self.cfg.stuck_window:
            window = last_positions[-self.cfg.stuck_window:]
            if len(set(window)) == 1:
                self._cooldown_until_t = t + self.cfg.d_cooldown_steps
                trigger = f"stuck_{self.cfg.stuck_window}"
                self.regime_log.append(RegimeStep(
                    t=t, regime="4D", d_activated=True,
                    trigger=trigger, decision_delta=decision_delta,
                ))
                return True, trigger

        # Default: 3D regime (C active, D not)
        self.regime_log.append(RegimeStep(
            t=t, regime="3D", d_activated=False,
            trigger="none", decision_delta=decision_delta,
        ))
        return False, "none"

    def regime_summary(self) -> Dict[str, int]:
        """Count steps in each regime."""
        counts = {"3D": 0, "4D": 0}
        for step in self.regime_log:
            counts[step.regime] = counts.get(step.regime, 0) + 1
        return counts

    def regime_switches(self) -> int:
        """Count how many times the regime changed."""
        if len(self.regime_log) < 2:
            return 0
        switches = 0
        for i in range(1, len(self.regime_log)):
            if self.regime_log[i].regime != self.regime_log[i - 1].regime:
                switches += 1
        return switches

    def reset_log(self):
        """Clear the regime log for a new episode."""
        self.regime_log.clear()
        self._cooldown_until_t = -1

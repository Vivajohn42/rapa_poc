"""MvpKernel: in-process governance orchestrator for rapa_mvp.

Implements the same tick lifecycle as rapa_os RapaKernel, but without
ZMQ — agents are called directly via Python method calls.

Usage:
    from kernel.kernel import MvpKernel
    from agents.agent_a import AgentA
    from agents.agent_b import AgentB
    from agents.agent_c import AgentC, GoalSpec
    from agents.agent_d import AgentD

    kernel = MvpKernel(
        agent_a=AgentA(),
        agent_b=AgentB(),
        agent_c=AgentC(goal=GoalSpec(mode="seek", target=(4,4))),
        agent_d=AgentD(),
    )

    obs = env.reset()
    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)
        obs, reward, done = env.step(result.action)
        kernel.observe_reward(reward)
        if done:
            kernel.tick(t+1, obs, done=True)  # episode-end tick
            break
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Dict, Any, List, Tuple

from kernel.types import (
    MvpTickSignals, MvpKernelDecision, MvpTickResult,
)
from kernel.abi import enforce_constraints
from kernel.scheduler import schedule_for
from kernel.closure_core import ClosureCore
from kernel.memory_manager import MvpMemoryManager
from kernel.loop_gain import MvpLoopGainTracker

from state.schema import ZC, ZD


class MvpKernel:
    """In-process governance orchestrator.

    Mirrors RapaKernel's tick lifecycle:
    1. A always runs (perception)
    2. Compute signals from current state
    3. Route → (gC, gD, deconstruct, reasons)
    4. Schedule → coupling template
    5. ABI enforce_constraints
    6. ClosureCore validate_decision
    7. Execute agents in schedule order
    8. Deconstruction if triggered
    9. Return (action, diagnostics)
    """

    def __init__(
        self,
        agent_a,
        agent_b,
        agent_c,
        agent_d=None,
        *,
        goal_map: Optional[Dict[str, Tuple[int, int]]] = None,
        enable_governance: bool = True,
        jung_profile=None,
        d_cooldown_steps: int = 8,
        d_uncertainty_threshold: float = 0.25,
        stuck_window: int = 4,
        tie_break_delta: float = 0.25,
        deconstruct_cooldown_ticks: int = 3,
        deconstruct_fn=None,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.agent_d = agent_d
        self.goal_map = goal_map
        self.enable_governance = enable_governance
        self.jung_profile = jung_profile

        # Apply Jung profile overrides (if provided)
        if jung_profile is not None:
            d_cooldown_steps = jung_profile.d_cooldown_steps
            stuck_window = jung_profile.stuck_window
            tie_break_delta = jung_profile.tie_break_delta
            deconstruct_cooldown_ticks = jung_profile.deconstruct_cooldown

        # Router config
        self.d_cooldown_steps = d_cooldown_steps
        self.d_uncertainty_threshold = d_uncertainty_threshold
        self.stuck_window = stuck_window
        self.tie_break_delta = tie_break_delta
        self.deconstruct_cooldown_ticks = deconstruct_cooldown_ticks

        # Closure core
        self._closure = ClosureCore()

        # Memory manager (L3 persistent, survives episode resets)
        self._memory = MvpMemoryManager(deconstruct_fn=deconstruct_fn)

        # Loop gain tracker (M4)
        self._loop_gain = MvpLoopGainTracker()

        # Per-episode state
        self._last_decision_delta: Optional[float] = None
        self._last_positions: deque = deque(maxlen=20)
        self._reward_history: deque = deque(maxlen=10)
        self._d_cooldown_until: int = -1
        self._last_deconstruct_tick: int = -999
        self._hint_just_learned: bool = False
        self._d_last_tags: List[str] = []

        # C state (managed by kernel)
        self._zC: Optional[ZC] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_episode(self, goal_mode: str = "seek", episode_id: str = "") -> ZC:
        """Reset kernel state for a new episode."""
        self._last_decision_delta = None
        self._last_positions.clear()
        self._reward_history.clear()
        self._d_cooldown_until = -1
        self._last_deconstruct_tick = -999
        self._hint_just_learned = False
        self._d_last_tags = []

        # Reset loop gain tracker per episode
        self._loop_gain.reset_episode()

        self._zC = ZC(goal_mode=goal_mode, memory={})
        self._zC.memory["episode_id"] = episode_id

        # Reset D's event buffer if present
        if self.agent_d is not None and hasattr(self.agent_d, "events"):
            self.agent_d.events.clear()
            if hasattr(self.agent_d, "seen_positions"):
                self.agent_d.seen_positions.clear()

        return self._zC

    def observe_reward(self, reward: float) -> None:
        """Record a reward for signal computation."""
        self._reward_history.append(reward)

    def tick(
        self,
        t: int,
        obs: dict,
        done: bool = False,
    ) -> MvpTickResult:
        """Execute one governance-controlled tick.

        Args:
            t: current step number within episode
            obs: raw environment observation
            done: True if episode has ended

        Returns:
            MvpTickResult with action, scored actions, decision, etc.
        """
        zC = self._zC
        assert zC is not None, "Call reset_episode() before tick()"

        # ---- 1. A always runs ----
        zA = self.agent_a.infer_zA(obs)
        self._last_positions.append(zA.agent_pos)

        # ---- 2. Immediate hint capture (before routing) ----
        # If a hint is visible, capture it via D immediately
        if zA.hint is not None and self.agent_d is not None:
            self.agent_d.observe_step(
                t=t, zA=zA, action="hint", reward=0.0, done=False
            )
            zD_hint = self.agent_d.build_micro(
                goal_mode=zC.goal_mode,
                goal_pos=(-1, -1),
                last_n=1,
            )
            old_target = zC.memory.get("target")
            zC = self._memory._deconstruct_fn(zC, zD_hint, goal_map=self.goal_map)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            new_target = zC.memory.get("target")
            self._hint_just_learned = (new_target != old_target and new_target is not None)
            self._zC = zC

        # ---- 3. Compute signals ----
        stuck = self._is_stuck()
        signals = MvpTickSignals.from_state(
            hint=zA.hint,
            decision_delta=self._last_decision_delta,
            reward_history=list(self._reward_history),
            memory_size=len(zC.memory),
            d_drift_score=0.0,  # Phase 4 will compute this
            hint_just_learned=self._hint_just_learned,
            done=done,
            goal_mode=zC.goal_mode,
            goal_target=zC.memory.get("target"),
            stuck=stuck,
        )
        self._hint_just_learned = False  # consumed

        # ---- 4. Route ----
        gC, gD, route_decon, reasons = self._route(t, signals)

        # ---- 5. Schedule ----
        # 6FoM+D overlay: D runs out-of-band, not in schedule
        sched_gD = 0  # D never in schedule (same as rapa_os M3+)
        sched = schedule_for(t, gC, sched_gD)
        sched = enforce_constraints(gC, sched_gD, sched)

        # ---- 6. Build decision ----
        decision = MvpKernelDecision(
            gC=gC, gD=gD, schedule=sched,
            deconstruct=route_decon, reasons=reasons,
        )

        # ---- 7. Validate ----
        if self.enable_governance:
            self._closure.validate_decision(decision)

        # ---- 8. Execute agents in schedule order ----
        action = None
        scored = None

        # Update C's target from memory
        if "target" in zC.memory:
            self.agent_c.goal.target = tuple(zC.memory["target"])

        if gC == 1:
            # C active: use C's goal-directed action selection
            action, scored = self.agent_c.choose_action(
                zA,
                self.agent_b.predict_next,
                memory=zC.memory,
                tie_break_delta=self.tie_break_delta,
            )
            if len(scored) >= 2:
                self._last_decision_delta = scored[0][1] - scored[1][1]
            else:
                self._last_decision_delta = 0.0
        else:
            # AB only: use B's forward model with simple heuristic
            action = self._ab_only_action(zA)
            self._last_decision_delta = None

        # ---- 9. D out-of-band (if gD=1) ----
        d_activated = False
        zD = None
        if gD == 1 and self.agent_d is not None:
            d_activated = True
            self.agent_d.observe_step(
                t=t, zA=zA, action=action, reward=0.0, done=done,
            )
            zD = self.agent_d.build_micro(
                goal_mode=zC.goal_mode,
                goal_pos=(-1, -1),
                last_n=5,
            )
            self._d_last_tags = list(zD.meaning_tags)
        elif self.agent_d is not None:
            # D observes but doesn't build (cheap recording)
            self.agent_d.observe_step(
                t=t, zA=zA, action=action, reward=0.0, done=done,
            )

        # ---- 10. Deconstruction (via MemoryManager for L3) ----
        decon_fired = False
        do_decon = self._should_deconstruct(t, gD, signals, route_decon, reasons)
        if do_decon and zD is not None:
            zC = self._memory.deconstruct(zC, zD, goal_map=self.goal_map)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            self._zC = zC
            self._last_deconstruct_tick = t
            decon_fired = True

        # Episode-end deconstruction (full narrative via MemoryManager)
        if done and self.agent_d is not None:
            zD_final = self.agent_d.build(
                goal_mode=zC.goal_mode, goal_pos=(-1, -1),
            )
            zC = self._memory.deconstruct(zC, zD_final, goal_map=self.goal_map)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            self._zC = zC

        # ---- 11. Loop Gain (M4) ----
        d_seen = None
        if self.agent_d is not None and hasattr(self.agent_d, "seen_positions"):
            d_seen = self.agent_d.seen_positions

        gain = self._loop_gain.compute_tick(
            zA=zA,
            zC=zC,
            zD=zD,
            scored=scored,
            decon_fired=decon_fired,
            gC=gC,
            gD=gD,
            tick_id=t,
            predict_next_fn=self.agent_b.predict_next,
            d_seen_positions=d_seen,
            has_agent_d=(self.agent_d is not None),
        )

        return MvpTickResult(
            action=action,
            scored=scored,
            decision=decision,
            gain=gain,
            d_activated=d_activated,
            decon_fired=decon_fired,
        )

    @property
    def zC(self) -> ZC:
        """Current C state."""
        assert self._zC is not None
        return self._zC

    @property
    def memory_manager(self) -> MvpMemoryManager:
        """L3 memory manager (persists across episodes)."""
        return self._memory

    @property
    def loop_gain(self) -> MvpLoopGainTracker:
        """Loop gain tracker (per-episode history)."""
        return self._loop_gain

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route(
        self, t: int, signals: MvpTickSignals
    ) -> tuple[int, int, bool, list[str]]:
        """Compute gating decision from signals.

        Port of rapa_os/kernel/router.py:route().
        """
        # gC activation
        gC = 0
        if (signals.has_instr
                or signals.plan_horizon >= 3
                or signals.td_err >= 0.15
                or signals.r_var >= 0.15):
            gC = 1

        # In rapa_mvp, C is effectively always needed for goal-directed
        # navigation (unlike rapa_os where A+B suffice for basic movement).
        # Override: gC=1 when goal_target is known.
        if signals.goal_target is not None:
            gC = 1

        # For the MVP, default to gC=1 to match existing behavior
        # (C is always active in all Stufe 0-9 variants except ab_only)
        if gC == 0:
            gC = 1  # conservative default for MVP

        # gD activation
        gD = 0
        if signals.need_narrative:
            gD = 1
        elif self.agent_d is not None:
            # Uncertainty trigger (matching rapa_mvp Router logic)
            if (signals.td_err >= (1.0 - self.d_uncertainty_threshold)
                    and t >= self._d_cooldown_until):
                gD = 1
                self._d_cooldown_until = t + self.d_cooldown_steps
            # Stuck trigger
            elif self._is_stuck() and t >= self._d_cooldown_until:
                gD = 1
                self._d_cooldown_until = t + self.d_cooldown_steps

        # Deconstruction triggers
        decon = False
        reasons: list[str] = []
        if signals.episode_end:
            decon = True
            reasons.append("EPISODE_END")
        if signals.mem_cost >= 0.85:
            decon = True
            reasons.append("MEM_BUDGET")
        if signals.d_drift_score >= 0.25:
            decon = True
            reasons.append("DRIFT")
        if signals.marginal_gain_d <= 0.05:
            decon = True
            reasons.append("MARGINAL_GAIN_LOW")

        return gC, gD, decon, reasons

    def _is_stuck(self) -> bool:
        """Check if agent is stuck (same position for stuck_window ticks)."""
        if len(self._last_positions) < self.stuck_window:
            return False
        window = list(self._last_positions)[-self.stuck_window:]
        return len(set(window)) == 1

    def _should_deconstruct(
        self, t: int, gD: int, signals: MvpTickSignals,
        route_decon: bool, reasons: list[str],
    ) -> bool:
        """Check if deconstruction should fire, respecting cooldown."""
        if not route_decon:
            return False
        if gD == 0:
            # Filter D-specific reasons when D is off
            filtered = [r for r in reasons if r in ("EPISODE_END", "MEM_BUDGET")]
            if not filtered:
                return False
        # Cooldown check (episode_end always overrides)
        if "EPISODE_END" not in reasons:
            if (t - self._last_deconstruct_tick) < self.deconstruct_cooldown_ticks:
                return False
        return True

    def _ab_only_action(self, zA) -> str:
        """Simple AB-only action selection (no C).

        Uses B's forward model to pick the action that moves to a new cell,
        avoiding walls. Fallback when gC=0.
        """
        import random
        actions = ["up", "down", "left", "right"]
        # Prefer actions that actually move
        moving = []
        for a in actions:
            zA_next = self.agent_b.predict_next(zA, a)
            if zA_next.agent_pos != zA.agent_pos:
                moving.append(a)
        if moving:
            return random.choice(moving)
        return random.choice(actions)

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
from kernel.closure_residuum import ClosureResiduum
from kernel.interfaces import StreamA, StreamB, StreamC, StreamD
from kernel.unified_memory import UnifiedMemory
from kernel.compression import CompressionController

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
        agent_a: StreamA,
        agent_b: StreamB,
        agent_c: StreamC,
        agent_d: Optional[StreamD] = None,
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
        fallback_actions: Optional[List[str]] = None,
        use_unified_memory: bool = False,
        active_compression: bool = False,
        online_distiller=None,
        regime_controller=None,
        telemetry=None,
        max_steps: int = 200,
        canvas_manager=None,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.agent_d = agent_d
        self._canvas_manager = canvas_manager  # F.4: for writing predictions
        self.goal_map = goal_map
        self.enable_governance = enable_governance
        self.jung_profile = jung_profile
        self._fallback_actions = fallback_actions
        self._active_compression = active_compression
        self._direction_prior_net = None  # Set via set_direction_prior_net()
        self._online_distiller = online_distiller  # OnlineDistiller or None
        self._regime_controller = regime_controller  # RegimeController or None
        self._active_regime_steering: bool = (regime_controller is not None)
        self._telemetry = telemetry  # Telemetry or None
        self.max_steps = max_steps  # for budget trigger signal

        # Replan-burst parameters (configurable via set_replan_burst_params)
        self._replan_stuck_window: int = 5
        self._replan_burst_length: int = 5

        # Replan-burst per-episode state
        self._replan_states: deque = deque(maxlen=20)  # (pos, dir) tuples
        self._replan_burst_remaining: int = 0
        self._replan_burst_count: int = 0

        # Exploration-progress stuck detection: track new cells discovered
        self._replan_known_cells_count: int = 0  # last known cell count
        self._replan_no_progress_ticks: int = 0  # consecutive ticks with no new cells
        self._replan_no_progress_threshold: int = 12  # ticks without new cells -> stuck

        # active_compression implies use_unified_memory
        if active_compression:
            use_unified_memory = True

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

        # Closure residuum tracker
        # D is always out-of-band in MVP → effective schedule is 6FoM (2 templates)
        ie_weight = jung_profile.ie_weight if jung_profile is not None else 0.5
        sn_weight = jung_profile.sn_weight if jung_profile is not None else 0.5
        self._residuum = ClosureResiduum(
            n_schedule_templates=2,  # len(TEMPL_6FOM)
            ie_weight=ie_weight,
            sn_weight=sn_weight,
        )

        # Unified Memory + Cascaded Compression (RAPA v2)
        self._unified_memory: Optional[UnifiedMemory] = None
        self._compression: Optional[CompressionController] = None
        if use_unified_memory:
            self._unified_memory = UnifiedMemory()
            self._compression = CompressionController(
                um=self._unified_memory,
                memory_manager=self._memory,
                jung_profile=jung_profile,
            )

        # Per-episode state
        self._episode_step: int = 0          # current step for budget trigger
        self._episode_stuck_ticks: int = 0   # stuck ticks for budget trigger
        self._success_rate_ema: float = 0.0  # EMA of episode success rate
        self._no_target_ticks: int = 0  # ticks where no target available (diagnostic)
        self._last_decision_delta: Optional[float] = None
        self._last_positions: deque = deque(maxlen=20)
        self._reward_history: deque = deque(maxlen=10)
        self._d_cooldown_until: int = -1
        self._last_deconstruct_tick: int = -999
        self._hint_just_learned: bool = False
        # C state (managed by kernel)
        self._zC: Optional[ZC] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_episode(self, goal_mode: str = "seek", episode_id: str = "") -> ZC:
        """Reset kernel state for a new episode."""
        self._no_target_ticks = 0
        self._last_decision_delta = None
        self._last_positions.clear()
        self._reward_history.clear()
        self._d_cooldown_until = -1
        self._last_deconstruct_tick = -999
        self._hint_just_learned = False
        # Reset replan-burst state
        self._replan_states.clear()
        self._replan_burst_remaining = 0
        self._replan_burst_count = 0
        self._replan_known_cells_count = 0
        self._replan_no_progress_ticks = 0

        # Reset online distiller episode buffer
        if self._online_distiller is not None:
            self._online_distiller.reset_episode()

        # Reset loop gain tracker per episode
        self._loop_gain.reset_episode()

        # Reset closure residuum tracker per episode
        self._residuum.reset_episode()

        self._zC = ZC(goal_mode=goal_mode, memory={})
        self._zC.memory["episode_id"] = episode_id

        # Reset Unified Memory + Compression per episode
        if self._unified_memory is not None:
            self._unified_memory.reset_episode()
        if self._compression is not None:
            self._compression.reset_episode()

        # Reset D's event buffer if present
        if self.agent_d is not None and hasattr(self.agent_d, "events"):
            self.agent_d.events.clear()
            if hasattr(self.agent_d, "seen_positions"):
                self.agent_d.seen_positions.clear()

        # Reset regime controller (active overlay steering)
        if self._regime_controller is not None:
            self._regime_controller.reset_episode()

        # Reset episode-scoped telemetry metrics
        if self._telemetry is not None:
            self._telemetry.metrics.reset_episode()

        # Phase 4: Reset StreamLearner episode state
        for _agent in (self.agent_a, self.agent_b,
                       self.agent_c, self.agent_d):
            if _agent is not None:
                _agent.learner.reset_episode()

        # Per-episode success rate tracking (for trigger signals)
        self._episode_step: int = 0
        self._episode_stuck_ticks: int = 0

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

        # UM shadow: sync L0 from A's perception
        if self._unified_memory is not None:
            self._unified_memory.populate_from_zA(zA, t)

        # ---- 2. Immediate hint capture (before routing) ----
        # If a hint is visible, capture it via D immediately
        if zA.hint is not None and self.agent_d is not None:
            self.agent_d.observe_step(
                t=t, zA=zA, action="hint", reward=0.0, done=False
            )
            # build_micro stays: D needs it for internal state update
            self.agent_d.build_micro(
                goal_mode=zC.goal_mode,
                goal_pos=(-1, -1),
                last_n=1,
            )
            # Phase 3: read MeaningReport instead of parsing ZD tags
            old_target = zC.memory.get("target")
            _hint_report = self.agent_d.report_meaning()
            # Always write target (even None) to clear stale targets
            zC.memory["target"] = _hint_report.suggested_target
            if _hint_report.suggested_phase is not None:
                zC.memory["phase"] = _hint_report.suggested_phase
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            new_target = zC.memory.get("target")
            self._hint_just_learned = (new_target != old_target and new_target is not None)
            self._zC = zC
            # UM shadow: sync L2 after hint capture
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zC(zC, t)

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

        # ---- 4a. Regime overlay (constrains _route output) ----
        gC, gD, route_decon, reasons = self._apply_regime_overlay(
            gC, gD, route_decon, reasons)

        # ---- 4b. Regime computation (for next tick's overlay) ----
        self._episode_step = t
        if stuck:
            self._episode_stuck_ticks += 1
        shadow_regime = None
        if self._regime_controller is not None:
            from kernel.regime import compute_triggers
            # Gather observables for trigger computation
            _latest_residual = (
                self._residuum.episode_history[-1]
                if self._residuum.episode_history else None
            )
            _delta_8 = _latest_residual.delta_8 if _latest_residual else 0.0
            _inv_count = (
                self._unified_memory.invalidation_count
                if self._unified_memory is not None
                and hasattr(self._unified_memory, "invalidation_count")
                else 0
            )
            _distiller_mode_str = "OFF"
            _distiller_acc = 0.0
            _l2_comp = False
            if self._online_distiller is not None:
                _distiller_mode_str = self._online_distiller.mode.name
                _distiller_acc = getattr(
                    self._online_distiller, "_eval_accuracy", 0.0)
            if (self._unified_memory is not None
                    and self._unified_memory.is_compressed("L2")):
                _l2_comp = True
            _target_known = (
                "target" in zC.memory
                and zC.memory.get("target") is not None
            )
            # New cells ratio: from ObjectMemory if available
            _om = getattr(self.agent_c, "_object_memory", None)
            _new_cells_ratio = 0.0
            if _om is not None:
                _total = max(1, zA.width * zA.height)
                _known = len(getattr(_om, "known_empty", set())) + len(
                    getattr(_om, "known_walls", set()))
                _new_cells_ratio = min(1.0, _known / _total)

            # Phase 4: learner readiness (weakest-link across all streams)
            from kernel.types import LearnerMode
            _learner_readiness = 1.0
            for _agent in (self.agent_a, self.agent_b,
                           self.agent_c, self.agent_d):
                if _agent is not None:
                    _lstatus = _agent.learner.ready()
                    if _lstatus.mode == LearnerMode.TRAINING:
                        _learner_readiness = min(
                            _learner_readiness, 0.3)

            triggers = compute_triggers(
                td_err=signals.td_err,
                delta_8=_delta_8,
                invalidation_count=_inv_count,
                target_known=_target_known,
                new_cells_ratio=_new_cells_ratio,
                success_rate_ema=self._success_rate_ema,
                distiller_mode=_distiller_mode_str,
                l2_compressed=_l2_comp,
                distiller_accuracy=_distiller_acc,
                mem_cost=signals.mem_cost,
                stuck_ticks=self._episode_stuck_ticks,
                max_steps=self.max_steps,
                current_step=t,
                learner_readiness=_learner_readiness,
            )
            shadow_regime = self._regime_controller.update(
                triggers, episode_end=done)

            # Training interval modulation (GOALSEEK → fast training)
            if (self._active_regime_steering
                    and self._online_distiller is not None):
                _gates = self._regime_controller.gates()
                if _gates.training_interval_key == "fast":
                    self._online_distiller.train_interval = 1
                    self._online_distiller.train_interval_post_enable = 1
                else:
                    self._online_distiller.train_interval = 3
                    self._online_distiller.train_interval_post_enable = 2

            if self._telemetry is not None:
                shadow_gates = self._regime_controller.gates()
                self._telemetry.metrics.gauge(
                    "ep.regime", float(shadow_regime.value))
                self._telemetry.events.emit(
                    "regime_tick",
                    regime=shadow_regime.name,
                    triggers={
                        "s": triggers.surprise, "p": triggers.progress,
                        "r": triggers.readiness, "b": triggers.budget,
                    },
                    gates={
                        "gC": shadow_gates.gC, "gD": shadow_gates.gD,
                        "b_may": shadow_gates.b_may_takeover,
                    },
                    step=t,
                )

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
        c_compressed = False

        # Update C's target from memory (None = no target → frontier exploration)
        if "target" in zC.memory and zC.memory["target"] is not None:
            self.agent_c.goal.target = tuple(zC.memory["target"])
        elif "target" in zC.memory and zC.memory["target"] is None:
            self.agent_c.goal.target = None

        # Active Mode: check if L2 is compressed -> C runs in compressed mode
        # Two paths to L2-compressed:
        #   (a) UM says L2 compressed AND zC.memory has explicit target (D's deconstruction)
        #   (b) Distiller has a resolved target AND is EXPERT for that kind
        # Path (b) enables B-takeover during FRONTIER exploration, where
        # zC.memory["target"] is never set but _resolve_target_active finds one.
        _um_l2 = (
            self._active_compression
            and self._unified_memory is not None
            and self._unified_memory.is_compressed("L2")
        )
        _has_explicit_target = (
            "target" in zC.memory
            and zC.memory.get("target") is not None
        )
        _distiller_ready = False
        if self._online_distiller is not None and _um_l2:
            from agents.online_distiller import DistillerMode
            _res_target, _res_kind = self._resolve_target_active(zA, zC)
            if _res_target is not None:
                _distiller_ready = (
                    self._online_distiller.mode_for(_res_kind)
                    == DistillerMode.EXPERT
                )
        l2_compressed = _um_l2 and (_has_explicit_target or _distiller_ready)

        burst_active = False  # True if this tick is a replan-burst tick

        if l2_compressed and self._replan_burst_remaining > 0:
            # Replan-burst: force C (full BFS) instead of compressed B
            action, scored = self.agent_c.choose_action(
                zA,
                self.agent_b.predict_next,
                memory=zC.memory,
                tie_break_delta=self.tie_break_delta,
            )
            c_compressed = False  # C is deciding, not B
            burst_active = True
            self._replan_burst_remaining -= 1
            if len(scored) >= 2:
                self._last_decision_delta = scored[0][1] - scored[1][1]
            else:
                self._last_decision_delta = 0.0

        elif l2_compressed:
            # Track (pos, dir) for stuck detection
            direction = zA.direction if zA.direction is not None else 0
            self._replan_states.append((zA.agent_pos, direction))

            # Track exploration progress (new cells discovered)
            self._update_exploration_progress()

            # Regime gate: b_may_takeover (hard block)
            _regime_b_blocked = False
            if (self._active_regime_steering
                    and self._regime_controller is not None):
                _regime_b_blocked = (
                    not self._regime_controller.gates().b_may_takeover)

            # Per-kind mode + trust gating for C-fallback.
            # REAL: EXPERT + hard trust gate (conservative)
            # FRONTIER: EXPERT → B takes over (trust as score weight only)
            use_c_fallback = False
            if _regime_b_blocked:
                use_c_fallback = True  # Regime says: C decides
            elif self._online_distiller is not None:
                from agents.online_distiller import DistillerMode, TargetKind
                target_tk, current_kind = self._resolve_target_active(
                    zA, zC)
                if target_tk is not None:
                    mode_k = self._online_distiller.mode_for(current_kind)
                    if mode_k == DistillerMode.EXPERT:
                        if current_kind == TargetKind.REAL:
                            # REAL: hard trust gate
                            c_agent = self.agent_c
                            phase_int = {"FIND_KEY": 0, "OPEN_DOOR": 1,
                                         "REACH_GOAL": 2}.get(
                                getattr(c_agent, "phase", "FIND_KEY"), 0)
                            _, _conf, trust = self._online_distiller.predict(
                                agent_pos=zA.agent_pos,
                                target=tuple(target_tk),
                                agent_dir=direction,
                                obstacles=set(zA.obstacles),
                                width=zA.width,
                                height=zA.height,
                                phase=phase_int,
                                carrying_key=getattr(
                                    c_agent, "carrying_key", False),
                                door_open=getattr(
                                    c_agent, "door_open", False),
                            )
                            if trust < self._online_distiller.trust_threshold:
                                use_c_fallback = True
                        # FRONTIER EXPERT: no hard trust gate, B takes over
                    else:
                        use_c_fallback = True  # not EXPERT for this kind
                else:
                    use_c_fallback = True  # no target available

            if use_c_fallback:
                # Net says "I'm not sure" → let C decide (full BFS)
                action, scored = self.agent_c.choose_action(
                    zA,
                    self.agent_b.predict_next,
                    memory=zC.memory,
                    tie_break_delta=self.tie_break_delta,
                )
                c_compressed = False  # C is deciding
            else:
                # B navigates via compressed prior (net or analytical)
                action, scored = self._compressed_l2_action(zA)
                c_compressed = True

            # Check if B is stuck (position/direction loop OR no exploration progress)
            trigger_burst = self._is_b_stuck()

            if trigger_burst:
                self._replan_burst_remaining = self._replan_burst_length
                self._replan_burst_count += 1
                self._replan_no_progress_ticks = 0  # reset after burst trigger

            if len(scored) >= 2:
                self._last_decision_delta = scored[0][1] - scored[1][1]
            else:
                self._last_decision_delta = 0.0

        elif gC == 1:
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

        # Online distiller: collect teacher sample when C decides (not compressed)
        if (self._online_distiller is not None
                and not c_compressed
                and gC == 1
                and action is not None
                and scored is not None):
            target_cs, target_kind_cs = self._resolve_target_active(zA, zC)
            if target_cs is not None:
                c_agent = self.agent_c
                phase_int = {"FIND_KEY": 0, "OPEN_DOOR": 1,
                             "REACH_GOAL": 2}.get(
                    getattr(c_agent, "phase", "FIND_KEY"), 0)
                self._online_distiller.collect_sample(
                    agent_pos=zA.agent_pos,
                    target=tuple(target_cs),
                    agent_dir=zA.direction if zA.direction is not None else 0,
                    obstacles=set(zA.obstacles),
                    width=zA.width,
                    height=zA.height,
                    phase=phase_int,
                    carrying_key=getattr(c_agent, "carrying_key", False),
                    door_open=getattr(c_agent, "door_open", False),
                    c_action=action,
                    c_scored=scored,
                    target_kind=target_kind_cs,
                )
            else:
                self._no_target_ticks += 1

        # ---- 9. D out-of-band (if gD=1) ----
        d_activated = False
        d_suppressed = False
        zD = None

        # Active Mode: check if L3 is compressed → skip D's build_micro
        # BUT: D is still needed if no target exists (D-essentiality)
        # or if invalidation just happened (surprise trigger)
        l3_compressed = (
            self._active_compression
            and self._unified_memory is not None
            and self._unified_memory.is_compressed("L3")
        )
        # D must remain active if target is unknown (D-essentiality)
        target_known = "target" in zC.memory and zC.memory["target"] is not None
        d_really_suppressed = l3_compressed and target_known

        if d_really_suppressed and self.agent_d is not None:
            # D suppressed: still observe for LoopGain but don't build
            d_suppressed = True
            self.agent_d.observe_step(
                t=t, zA=zA, action=action, reward=0.0, done=done,
            )
        elif gD == 1 and self.agent_d is not None:
            d_activated = True
            self.agent_d.observe_step(
                t=t, zA=zA, action=action, reward=0.0, done=done,
            )
            # F.4: B's prediction for D's action context
            zA_next = None
            if action and self.agent_b is not None:
                try:
                    zA_next = self.agent_b.predict_next(zA, action)
                except Exception:
                    pass
            zD = self.agent_d.build_micro(
                goal_mode=zC.goal_mode,
                goal_pos=(-1, -1),
                last_n=5,
                action=action,
                zA_next=zA_next,
            )
            # F.4: Write prediction to canvas (kernel is the conductor)
            if (
                hasattr(zD, "prediction") and zD.prediction
                and self._canvas_manager is not None
            ):
                self._canvas_manager.write_prediction(zD.prediction)
            # Phase 3: UM sync via MeaningReport (no tag parsing)
            if self._unified_memory is not None:
                _d_report = self.agent_d.report_meaning()
                self._unified_memory.populate_from_meaning_report(
                    _d_report, t)
        elif self.agent_d is not None:
            # D observes but doesn't build (cheap recording)
            self.agent_d.observe_step(
                t=t, zA=zA, action=action, reward=0.0, done=done,
            )

        # ---- 10. Deconstruction (via MemoryManager for L3) ----
        decon_fired = False
        do_decon = self._should_deconstruct(t, gD, signals, route_decon, reasons)
        if do_decon and zD is not None:
            _decon_report = self.agent_d.report_meaning() if self.agent_d else None
            zC = self._memory.deconstruct(
                zC, zD, goal_map=self.goal_map,
                meaning_report=_decon_report)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            self._zC = zC
            self._last_deconstruct_tick = t
            decon_fired = True
            # UM shadow: sync L2 after deconstruction
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zC(zC, t)

        # Episode-end deconstruction (full narrative via MemoryManager)
        if done and self.agent_d is not None:
            zD_final = self.agent_d.build(
                goal_mode=zC.goal_mode, goal_pos=(-1, -1),
            )
            _ep_report = self.agent_d.report_meaning()
            zC = self._memory.deconstruct(
                zC, zD_final, goal_map=self.goal_map,
                meaning_report=_ep_report)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            self._zC = zC
            # UM shadow: sync L2 after episode-end deconstruction
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zC(zC, t)

        # ---- 11. Loop Gain (M4) ----
        d_seen = None
        if self.agent_d is not None and hasattr(self.agent_d, "seen_positions"):
            d_seen = self.agent_d.seen_positions

        # Phase 3: MeaningReport for tag-free g_DC/g_AD/d_term
        _tick_report = (self.agent_d.report_meaning()
                        if self.agent_d is not None else None)

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
            meaning_report=_tick_report,
        )

        # ---- 12. Closure Residuum ----
        residual = self._residuum.compute_tick(
            zA=zA,
            zC=zC,
            zD=zD,
            scored=scored,
            predict_next_fn=self.agent_b.predict_next,
            gC=gC,
            gD=gD,
            tick_id=t,
            has_agent_d=(self.agent_d is not None),
            weakest_coupling=gain.weakest_coupling,
            meaning_report=_tick_report,
        )

        # ---- 13. Cascaded Compression (RAPA v2 Unified Memory) ----
        compression_stages: List[str] = []
        if self._unified_memory is not None and self._compression is not None and residual is not None:
            # Cache invalidation check (surprise → reactivate layers)
            invalidated = self._compression.check_invalidation(
                residual, divergence_threshold=self._residuum.divergence_threshold,
            )

            # Evaluate cascade compression
            fired, zC_updated = self._compression.evaluate_cascade(
                tick=t, residual=residual,
                zC=zC, zD=zD, goal_map=self.goal_map,
            )
            compression_stages = fired
            if zC_updated is not None:
                self._zC = zC_updated
                zC = zC_updated
                decon_fired = True  # Legacy-compatible

        # ---- 14. StreamLearner signals (Phase 4) ----
        if residual is not None or gain is not None:
            from kernel.types import LearnerSignal
            _signal = LearnerSignal(
                tick=t,
                reward=(self._reward_history[-1]
                        if self._reward_history else 0.0),
                done=done,
                episode_step=self._episode_step,
                delta_4=residual.delta_4 if residual else 0.0,
                c_term=residual.c_term if residual else 0.0,
                d_term=residual.d_term if residual else 0.0,
                action=action,
                scored=scored,
                grounding_score=(
                    _tick_report.grounding_score
                    if _tick_report is not None else 1.0),
            )
            for _agent in (self.agent_a, self.agent_b,
                           self.agent_c, self.agent_d):
                if _agent is not None:
                    _agent.learner.observe_signal(_signal)

        # ---- 14a. B dynamics → L1 persistence (Phase 5a) ----
        if self.agent_b is not None and self._unified_memory is not None:
            _b_status = self.agent_b.learner.ready()
            if _b_status.mode != LearnerMode.OFF:
                self._unified_memory.populate_from_b_dynamics(
                    {"prediction_accuracy": _b_status.accuracy}, tick=t)

        # ---- 14b. Episode-end learning ----
        if done:
            for _agent in (self.agent_a, self.agent_b,
                           self.agent_c, self.agent_d):
                if _agent is not None:
                    _agent.learner.learn()

        return MvpTickResult(
            action=action,
            scored=scored,
            decision=decision,
            gain=gain,
            residual=residual,
            d_activated=d_activated,
            decon_fired=decon_fired,
            compression_stages=compression_stages,
            c_compressed=c_compressed,
            d_suppressed=d_suppressed,
            replan_burst_active=burst_active,
            regime=(
                shadow_regime.name if shadow_regime is not None else None),
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

    @property
    def residuum(self) -> ClosureResiduum:
        """Closure residuum tracker (per-episode history)."""
        return self._residuum

    @property
    def unified_memory(self) -> Optional[UnifiedMemory]:
        """Unified memory store (None if use_unified_memory=False)."""
        return self._unified_memory

    @property
    def compression(self) -> Optional[CompressionController]:
        """Compression controller (None if use_unified_memory=False)."""
        return self._compression

    @property
    def regime_controller(self):
        """Regime controller (active overlay steering in Phase 2)."""
        return self._regime_controller

    @property
    def telemetry(self):
        """Telemetry subsystem (None if not configured)."""
        return self._telemetry

    def update_success_rate(self, success: bool, alpha: float = 0.1) -> None:
        """Update EMA of episode success rate (for regime trigger signals)."""
        val = 1.0 if success else 0.0
        self._success_rate_ema = (
            alpha * val + (1.0 - alpha) * self._success_rate_ema
        )

    # ------------------------------------------------------------------
    # Active Compression API
    # ------------------------------------------------------------------

    def set_direction_prior_net(self, net) -> None:
        """Attach a trained DirectionPriorNet for neural compressed-L2 scoring.

        When set, _compressed_l2_action uses the neural net instead of
        the analytical direction_prior from UnifiedMemory L1 layer.
        """
        self._direction_prior_net = net

    def set_replan_burst_params(
        self,
        stuck_window: int = 5,
        burst_length: int = 5,
        no_progress_threshold: int = 12,
    ) -> None:
        """Configure replan-burst parameters.

        Args:
            stuck_window: Number of (pos, dir) states to check for looping.
            burst_length: Number of ticks to force C when stuck.
            no_progress_threshold: Ticks without discovering new cells -> stuck.
        """
        self._replan_stuck_window = stuck_window
        self._replan_burst_length = burst_length
        self._replan_no_progress_threshold = no_progress_threshold

    @property
    def replan_burst_count(self) -> int:
        """Number of replan-burst activations this episode."""
        return self._replan_burst_count

    @property
    def no_target_ticks(self) -> int:
        """Ticks where no target was available (diagnostic counter)."""
        return self._no_target_ticks

    def _compressed_l2_action(
        self, zA,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Action selection when L2 is compressed (C bypassed).

        Navigation: scores based on direction_prior from UM L1 layer,
        or via DirectionPriorNet if available.
        Interaction: deterministic D-essentiality rules preserved —
        reads phase/key_pos/door_pos from agent_c (set by adapter).
        """
        from agents.doorkey_agent_c import ACTIONS, DIR_VEC

        scored: List[Tuple[str, float]] = []
        direction = zA.direction if zA.direction is not None else 0
        obstacle_set = set(zA.obstacles)

        # Compute live direction_prior from current position + resolved target
        # (Static UM L1 prior is stale after agent moves — must recompute)
        zC = self._zC
        target, _current_kind = self._resolve_target_active(zA, zC)
        direction_prior: List[str] = []
        if target is not None:
            dx = target[0] - zA.agent_pos[0]
            dy = target[1] - zA.agent_pos[1]
            if dx > 0:
                direction_prior.append("right")
            elif dx < 0:
                direction_prior.append("left")
            if dy > 0:
                direction_prior.append("down")
            elif dy < 0:
                direction_prior.append("up")

        # Read interaction priors from UM L1 (these are state-based, not position-based)
        l1_data = self._unified_memory.read_layer("L1") if self._unified_memory else {}
        interaction_prior = l1_data.get("interaction_prior", {})

        # Read phase-specific state from agent_c (set by adapter.inject_obs_metadata)
        c = self.agent_c
        phase = getattr(c, "phase", "FIND_KEY")
        key_pos = getattr(c, "key_pos", None)
        door_pos = getattr(c, "door_pos", None)
        carrying_key = getattr(c, "carrying_key", False)

        # Facing position
        dx, dy = DIR_VEC[direction]
        fx, fy = zA.agent_pos[0] + dx, zA.agent_pos[1] + dy
        facing = None
        if 0 <= fx < zA.width and 0 <= fy < zA.height:
            facing = (fx, fy)

        for action in ACTIONS:
            score = 0.0
            zA_next = self.agent_b.predict_next(zA, action)
            next_dir = zA_next.direction if zA_next.direction is not None else 0

            if action in ("turn_left", "turn_right", "forward"):
                from agents.online_distiller import DistillerMode
                if (self._online_distiller is not None
                        and target is not None
                        and self._online_distiller.mode_for(_current_kind)
                        == DistillerMode.EXPERT):
                    # Priority 1: Online direction net (learned from C)
                    score = self._online_net_l2_score(
                        zA, zA_next, action, direction, next_dir,
                        tuple(target))
                elif direction_prior:
                    # Priority 2: Analytical scoring (BFS-based)
                    score = self._analytical_l2_score(
                        zA, zA_next, action, direction, next_dir,
                        direction_prior, obstacle_set)
                elif self._direction_prior_net is not None:
                    # Priority 3: Legacy neural fallback (no target)
                    score = self._neural_l2_score(
                        zA, zA_next, action, next_dir)
                else:
                    # Priority 4: Basic move preference
                    if action == "forward":
                        if zA_next.agent_pos != zA.agent_pos:
                            score = 0.3
                        else:
                            score = -0.1
                    else:
                        score = 0.05

            elif action == "pickup":
                # D-essentiality: deterministic rules
                if (phase == "FIND_KEY"
                        and not carrying_key
                        and key_pos is not None
                        and facing == key_pos):
                    score = 3.0
                else:
                    score = -1.0

            elif action == "toggle":
                # D-essentiality: deterministic rules
                if (phase == "OPEN_DOOR"
                        and carrying_key
                        and door_pos is not None
                        and facing == door_pos):
                    score = 3.0
                else:
                    score = -1.0

            scored.append((action, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0], scored

    def _analytical_l2_score(
        self, zA, zA_next, action: str, direction: int, next_dir: int,
        direction_prior: List[str], obstacle_set: set,
    ) -> float:
        """Score a navigation action using BFS-based distance to compressed target.

        Uses L1-level knowledge: position + obstacles + compressed target.
        No phase/goal semantics — just "move toward target avoiding walls".
        This is the analytical compressed-C: BFS scoring without full C.
        """
        from agents.doorkey_agent_c import DIR_VEC, _bfs_distance, UNREACHABLE

        # Get compressed target from zC memory
        zC = self._zC
        target = zC.memory.get("target") if zC else None
        if target is None:
            # No target: basic exploration
            if action == "forward":
                return 0.3 if zA_next.agent_pos != zA.agent_pos else -0.1
            return 0.05

        target = tuple(target)
        # Remove target from obstacles if needed (door is obstacle but target)
        nav_obstacles = obstacle_set
        if target in obstacle_set:
            nav_obstacles = obstacle_set - {target}

        if action == "forward":
            # Wall hit check: forward didn't change position
            if zA_next.agent_pos == zA.agent_pos:
                return -0.1

            # BFS distance after moving forward
            d_now = _bfs_distance(zA.agent_pos, target, nav_obstacles, zA.width, zA.height)
            d_next = _bfs_distance(zA_next.agent_pos, target, nav_obstacles, zA.width, zA.height)

            if d_next >= UNREACHABLE:
                return -0.1

            score = 1.0 / (d_next + 1.0)
            if d_next < d_now:
                score += 0.5
            return score

        else:
            # turn_left / turn_right: compute BFS distance from current pos
            # toward the direction we would face after turning
            from agents.doorkey_agent_c import _bfs_next_step
            next_cell = _bfs_next_step(
                zA.agent_pos, target, nav_obstacles, zA.width, zA.height)
            if next_cell is None:
                return 0.05

            # Direction we want to face (toward BFS next step)
            desired_dx = next_cell[0] - zA.agent_pos[0]
            desired_dy = next_cell[1] - zA.agent_pos[1]
            desired_dir = 0
            for i, (ddx, ddy) in enumerate(DIR_VEC):
                if desired_dx == ddx and desired_dy == ddy:
                    desired_dir = i
                    break

            # How many turns from next_dir to desired_dir
            diff = abs(next_dir - desired_dir) % 4
            turns_remaining = min(diff, 4 - diff)
            if turns_remaining == 0:
                return 0.9  # Facing the right direction after this turn
            elif turns_remaining == 1:
                return 0.5  # One more turn needed
            return 0.1  # Turning away

    def _neural_l2_score(
        self, zA, zA_next, action: str, next_dir: int,
    ) -> float:
        """Score a navigation action using DirectionPriorNet.

        Uses the raw regression output (trained to approximate C's score),
        with wall-hit penalty applied explicitly.
        """
        import torch
        from models.direction_prior_net import extract_l1_features

        # Explicit wall-hit check (neural net may not penalize enough)
        if action == "forward" and zA_next.agent_pos == zA.agent_pos:
            return -0.1

        features = extract_l1_features(
            agent_pos=zA.agent_pos,
            agent_dir=zA.direction if zA.direction is not None else 0,
            next_pos=zA_next.agent_pos,
            next_dir=next_dir,
            obstacles=set(zA.obstacles),
            width=zA.width,
            height=zA.height,
            carrying_key=getattr(self.agent_c, "carrying_key", False),
        )
        with torch.no_grad():
            raw = self._direction_prior_net(features.unsqueeze(0))
            return raw.item()  # Direct regression output (MSE on C's score)

    def _online_net_l2_score(
        self, zA, zA_next, action: str, direction: int, next_dir: int,
        target: Tuple[int, int],
    ) -> float:
        """Score a navigation action using the online direction net.

        Uses the net's predicted direction + confidence to score actions.
        Actions aligned with the predicted direction score higher.
        Low-confidence predictions return low scores (replan-burst handles it).
        """
        from agents.doorkey_agent_c import DIR_VEC

        c_agent = self.agent_c
        phase_int = {"FIND_KEY": 0, "OPEN_DOOR": 1, "REACH_GOAL": 2}.get(
            getattr(c_agent, "phase", "FIND_KEY"), 0)

        predicted_dir, _confidence, trust = self._online_distiller.predict(
            agent_pos=zA.agent_pos,
            target=target,
            agent_dir=direction,
            obstacles=set(zA.obstacles),
            width=zA.width,
            height=zA.height,
            phase=phase_int,
            carrying_key=getattr(c_agent, "carrying_key", False),
            door_open=getattr(c_agent, "door_open", False),
        )

        # Low trust -> return modest score; replan-burst will handle
        if trust < self._online_distiller.trust_threshold:
            if action == "forward":
                return 0.1 if zA_next.agent_pos != zA.agent_pos else -0.1
            return 0.05

        if action == "forward":
            # Wall hit check
            if zA_next.agent_pos == zA.agent_pos:
                return -0.1
            # Is the agent facing the predicted direction?
            if direction == predicted_dir:
                return 1.0 * trust
            return 0.2  # Moving but not in predicted direction

        else:
            # turn_left / turn_right: how close to predicted direction?
            diff = abs(next_dir - predicted_dir) % 4
            turns_remaining = min(diff, 4 - diff)
            if turns_remaining == 0:
                return 0.9 * trust  # Aligned after this turn
            elif turns_remaining == 1:
                return 0.5 * trust
            return 0.1

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
            # Phase 4: Delta_8-based D activation (high fixpoint deviation)
            elif (self._residuum.episode_history
                  and t >= self._d_cooldown_until):
                latest_delta_8 = self._residuum.episode_history[-1].delta_8
                if latest_delta_8 > self._residuum.activation_threshold:
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
        # Phase 3: dDelta_8/dt divergence trigger
        if self._residuum.episode_history:
            latest = self._residuum.episode_history[-1]
            if latest.d_delta_8_dt > self._residuum.divergence_threshold:
                decon = True
                reasons.append("RESIDUUM_DIVERGENCE")

        return gC, gD, decon, reasons

    def _apply_regime_overlay(
        self, gC: int, gD: int, route_decon: bool, reasons: list[str],
    ) -> tuple[int, int, bool, list[str]]:
        """Apply regime gates as constraints over _route() output.

        Overlay semantics: gates only RESTRICT, never expand.
        If the regime controller is absent or regime steering is off,
        returns inputs unchanged (Stage 1 compatibility).
        """
        if not self._active_regime_steering or self._regime_controller is None:
            return gC, gD, route_decon, reasons

        gates = self._regime_controller.gates()

        # gC: regime can suppress C (CONSOLIDATE: gC=0)
        if gates.gC == 0:
            gC = 0

        # gD: regime can suppress D (EXPLORE, GOALSEEK: gD=0)
        if gates.gD == 0:
            gD = 0

        # Decon: regime can suppress out-of-band decon
        # EPISODE_END always fires regardless of regime
        if not gates.allow_decon:
            if route_decon and "EPISODE_END" not in reasons:
                route_decon = False
                reasons = [r for r in reasons if r == "EPISODE_END"]

        return gC, gD, route_decon, reasons

    def _is_stuck(self) -> bool:
        """Check if agent is stuck (same position for stuck_window ticks)."""
        if len(self._last_positions) < self.stuck_window:
            return False
        window = list(self._last_positions)[-self.stuck_window:]
        return len(set(window)) == 1

    def _update_exploration_progress(self) -> None:
        """Track whether B is discovering new cells via ObjectMemory.

        Increments a no-progress counter if the known cell count hasn't
        increased since last tick. Resets counter when new cells are found.
        """
        om = getattr(self.agent_c, "_object_memory", None)
        if om is None:
            return
        current_known = len(om.known_empty) + len(om.known_walls)
        if current_known > self._replan_known_cells_count:
            # Progress: new cells discovered
            self._replan_known_cells_count = current_known
            self._replan_no_progress_ticks = 0
        else:
            self._replan_no_progress_ticks += 1

    def _is_b_stuck(self) -> bool:
        """Check if B (compressed mode) is stuck.

        Three conditions (any triggers burst):
        1. Position/direction loop: last K (pos,dir) states have ≤2 unique
        2. All same position (turns only, no movement)
        3. Exploration stagnation: no new cells discovered for N ticks
        """
        w = self._replan_stuck_window
        if len(self._replan_states) < w:
            return False

        # Condition 1: (pos, dir) loop
        window = list(self._replan_states)[-w:]
        if len(set(window)) <= 2:
            return True

        # Condition 2: all positions same (turns only)
        positions = [s[0] for s in window]
        if len(set(positions)) == 1:
            return True

        # Condition 3: exploration stagnation (no new cells for N ticks)
        if self._replan_no_progress_ticks >= self._replan_no_progress_threshold:
            return True

        return False

    def _resolve_target_via_meaning(
        self, zA, zC,
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """Target resolution via MeaningReport.

        D reports meaning, kernel decides. No phase-specific if-chains.
        Returns (target, source) where source is "meaning"/"memory"/"frontier".

        Phase 2.5: active target resolution path (replaces tag-parsing).
        """
        # 1. D's MeaningReport
        if (self.agent_d is not None
                and hasattr(self.agent_d, "report_meaning")):
            from kernel.types import MeaningReport
            report = self.agent_d.report_meaning()
            if (report.confidence >= 0.5
                    and report.suggested_target is not None):
                return (tuple(report.suggested_target), "meaning")

        # 2. Memory-based (zC.memory["target"])
        target = zC.memory.get("target") if zC else None
        if target is not None:
            return (tuple(target), "memory")

        # 3. Frontier fallback
        t = self._nearest_frontier_target(zA)
        return (t, "frontier")  # t can be None

    def _resolve_target_active(
        self, zA, zC,
    ) -> Tuple[Optional[Tuple[int, int]], "TargetKind"]:
        """Target resolution via MeaningReport (Phase 3: always active).

        D's MeaningReport provides the target, source maps to TargetKind
        for distiller compatibility:
          "meaning" / "memory" → REAL (concrete object target)
          "frontier"           → FRONTIER (exploration pseudo-target)
        """
        from agents.online_distiller import TargetKind

        target, source = self._resolve_target_via_meaning(zA, zC)
        kind = (TargetKind.REAL if source in ("meaning", "memory")
                else TargetKind.FRONTIER)
        return (target, kind)

    def _nearest_frontier_target(self, zA) -> Optional[Tuple[int, int]]:
        """Find nearest frontier cell from ObjectMemory as pseudo-target.

        Used for distiller sample collection during exploration phases
        when no explicit target (key/door/goal) is known. The frontier
        represents C's exploration objective, so it's a valid proxy for
        "where C is trying to go."

        Returns None if no ObjectMemory or no frontier cells exist.
        """
        om = getattr(self.agent_c, "_object_memory", None)
        if om is None:
            return None
        frontier = om.frontier
        if not frontier:
            return None
        # Pick nearest frontier cell by Manhattan distance
        best = None
        best_dist = float("inf")
        ax, ay = zA.agent_pos
        for fx, fy in frontier:
            d = abs(fx - ax) + abs(fy - ay)
            if d < best_dist:
                best_dist = d
                best = (fx, fy)
        return best

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
        actions = self._fallback_actions or ["up", "down", "left", "right"]
        # Prefer actions that actually move
        moving = []
        for a in actions:
            zA_next = self.agent_b.predict_next(zA, a)
            if zA_next.agent_pos != zA.agent_pos:
                moving.append(a)
        if moving:
            return random.choice(moving)
        return random.choice(actions)

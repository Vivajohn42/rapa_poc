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
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.agent_d = agent_d
        self.goal_map = goal_map
        self.enable_governance = enable_governance
        self.jung_profile = jung_profile
        self._fallback_actions = fallback_actions
        self._active_compression = active_compression
        self._direction_prior_net = None  # Set via set_direction_prior_net()

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
            # UM shadow: sync L2 after hint deconstruction
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

        # Update C's target from memory
        if "target" in zC.memory:
            self.agent_c.goal.target = tuple(zC.memory["target"])

        # Active Mode: check if L2 is compressed → C runs in compressed mode
        # C only runs compressed if a target is known (direction_prior needs it)
        l2_compressed = (
            self._active_compression
            and self._unified_memory is not None
            and self._unified_memory.is_compressed("L2")
            and "target" in zC.memory
            and zC.memory.get("target") is not None
        )

        if l2_compressed:
            # C compressed: use direction prior from L1 instead of full C
            action, scored = self._compressed_l2_action(zA)
            c_compressed = True
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
            zD = self.agent_d.build_micro(
                goal_mode=zC.goal_mode,
                goal_pos=(-1, -1),
                last_n=5,
            )
            self._d_last_tags = list(zD.meaning_tags)
            # UM shadow: sync L3 from D's analysis
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zD(zD, t)
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
            # UM shadow: sync L2 after deconstruction
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zC(zC, t)

        # Episode-end deconstruction (full narrative via MemoryManager)
        if done and self.agent_d is not None:
            zD_final = self.agent_d.build(
                goal_mode=zC.goal_mode, goal_pos=(-1, -1),
            )
            zC = self._memory.deconstruct(zC, zD_final, goal_map=self.goal_map)
            zC.memory["episode_id"] = self._zC.memory.get("episode_id", "")
            self._zC = zC
            # UM shadow: sync L2 after episode-end deconstruction
            if self._unified_memory is not None:
                self._unified_memory.populate_from_zC(zC, t)

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

    # ------------------------------------------------------------------
    # Active Compression API
    # ------------------------------------------------------------------

    def set_direction_prior_net(self, net) -> None:
        """Attach a trained DirectionPriorNet for neural compressed-L2 scoring.

        When set, _compressed_l2_action uses the neural net instead of
        the analytical direction_prior from UnifiedMemory L1 layer.
        """
        self._direction_prior_net = net

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

        # Compute live direction_prior from current position + compressed target
        # (Static UM L1 prior is stale after agent moves — must recompute)
        zC = self._zC
        target = zC.memory.get("target") if zC else None
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
                if direction_prior:
                    # Analytical scoring (BFS-based, uses compressed target)
                    score = self._analytical_l2_score(
                        zA, zA_next, action, direction, next_dir,
                        direction_prior, obstacle_set)
                elif self._direction_prior_net is not None:
                    # Neural fallback (no target available)
                    score = self._neural_l2_score(
                        zA, zA_next, action, next_dir)
                else:
                    # No prior available: basic move preference
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

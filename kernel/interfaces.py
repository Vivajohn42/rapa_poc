"""Abstract stream interfaces and environment adapter for rapa_mvp.

These ABCs formalize the contract between MvpKernel and domain-specific
stream implementations.  Any new environment must provide concrete
implementations of StreamA, StreamB, StreamC, StreamD, and (optionally)
EnvironmentAdapter.

The signatures are derived directly from MvpKernel.tick() call sites
(kernel.py lines 187-340).

Layers:
  Kernel  (governance, loop gain, Delta_8) -- universal, interfaces only
  Streams (A/B/C/D)                        -- 4 abstract classes, domain-free
  Adapter (1 per domain)                   -- eval-script convenience
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Protocol, Optional, Dict, Any, List, Tuple, Callable,
    runtime_checkable,
)

from state.schema import ZA, ZD


# ---------------------------------------------------------------------------
# GoalTarget protocol (structural subtyping -- no inheritance required)
# ---------------------------------------------------------------------------

@runtime_checkable
class GoalTarget(Protocol):
    """Protocol for C's goal object.

    Must support ``goal.target`` read and ``goal.target = (x, y)`` write.
    Satisfied by GoalSpec (dataclass field) and _TextGoalProxy (property).
    """

    @property
    def target(self) -> Optional[Tuple[int, int]]: ...

    @target.setter
    def target(self, value: Tuple[int, int]) -> None: ...


# ---------------------------------------------------------------------------
# Stream abstract base classes
# ---------------------------------------------------------------------------

class StreamA(ABC):
    """Perception stream: raw observation -> typed belief state (ZA)."""

    @abstractmethod
    def infer_zA(self, obs) -> ZA:
        """Convert a raw environment observation into a ZA belief state.

        The observation format is domain-specific (GridState, dict, etc.).
        The returned ZA uses width/height as state-space dimensions and
        agent_pos as the current position in that space.
        """
        ...

    @property
    def learner(self) -> "StreamLearner":
        """Learning interface (Phase 4). Default: NullLearner."""
        if not hasattr(self, "_learner"):
            self._learner = NullLearner(label="deterministic-A")
        return self._learner


class StreamB(ABC):
    """Dynamics / forward model stream: predict next state given action."""

    @abstractmethod
    def predict_next(self, zA: ZA, action: str) -> ZA:
        """Return the predicted ZA after taking *action* in state *zA*.

        Must be deterministic for the same (zA, action) pair so that
        loop gain and closure residuum computations are stable.
        """
        ...

    @property
    def learner(self) -> "StreamLearner":
        """Learning interface (Phase 4). Default: NullLearner."""
        if not hasattr(self, "_learner"):
            self._learner = NullLearner(label="deterministic-B")
        return self._learner


class StreamC(ABC):
    """Valence / goal stream: score actions and select the best one."""

    @abstractmethod
    def choose_action(
        self,
        zA: ZA,
        predict_next_fn: Callable[[ZA, str], ZA],
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Score all available actions and return (best_action, scored_list).

        *predict_next_fn* is StreamB.predict_next (injected by kernel).
        *memory* is zC.memory (kernel-managed per-episode state).
        *scored_list* is [(action, score), ...] sorted descending.
        """
        ...

    @property
    @abstractmethod
    def goal(self) -> GoalTarget:
        """The goal object whose .target the kernel writes to.

        Implementations must return an object satisfying the GoalTarget
        protocol (i.e. has a ``target`` property with getter and setter).
        """
        ...

    @property
    def learner(self) -> "StreamLearner":
        """Learning interface (Phase 4). Default: NullLearner."""
        if not hasattr(self, "_learner"):
            self._learner = NullLearner(label="deterministic-C")
        return self._learner


class StreamD(ABC):
    """Narrative / meaning stream: event observation + synthesis.

    Implementations must also provide two instance attributes:
      events      -- list of recorded events (supports .clear())
      seen_positions -- set of visited state positions (supports .clear())
    The kernel accesses these via hasattr() checks.
    """

    @abstractmethod
    def observe_step(
        self,
        t: int,
        zA: ZA,
        action: str,
        reward: float,
        done: bool,
    ) -> None:
        """Record a single step for later synthesis."""
        ...

    @abstractmethod
    def build(self, goal_mode: str, goal_pos=None) -> ZD:
        """Build a full-episode narrative from all recorded events."""
        ...

    @abstractmethod
    def build_micro(
        self,
        goal_mode: str,
        goal_pos=None,
        last_n: int = 5,
    ) -> ZD:
        """Build a micro-narrative from the last *last_n* events."""
        ...

    @property
    def learner(self) -> "StreamLearner":
        """Learning interface (Phase 4). Default: NullLearner."""
        if not hasattr(self, "_learner"):
            self._learner = NullLearner(label="deterministic-D")
        return self._learner

    def report_meaning(self) -> "MeaningReport":
        """Structured MeaningReport for kernel consumption (Phase 3).

        Default adapter: derives MeaningReport from build_micro() ZD.
        Tag-parsing lives here (inside D abstraction), NOT in the kernel.
        D agents with native report_meaning() override this for richer
        structured data.  The default ensures all existing D agents work
        without modification.
        """
        from kernel.types import MeaningReport
        try:
            zd = self.build_micro(
                goal_mode="seek", goal_pos=(-1, -1), last_n=5)
        except Exception:
            return MeaningReport()

        # Auto-extract suggested_target and phase from tags
        suggested_target = None
        suggested_phase = None
        for tag in zd.meaning_tags:
            tl = tag.strip().lower()
            if tl.startswith(("key_at:", "door_at:", "goal_at:")):
                parts = tl.split(":", 1)[1].split("_")
                try:
                    suggested_target = (int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    pass
            elif tl.startswith("phase:"):
                suggested_phase = tl.split(":", 1)[1]

        return MeaningReport(
            confidence=0.5 if len(zd.meaning_tags) > 1 else 0.0,
            suggested_target=suggested_target,
            suggested_phase=suggested_phase,
            events_detected=[],
            narrative_tags=list(zd.meaning_tags),
            grounding_violations=zd.grounding_violations,
            grounding_score=max(
                0.0, 1.0 - zd.grounding_violations * 0.1),
            narrative_length=zd.length_chars,
        )


# ---------------------------------------------------------------------------
# StreamLearner ABC (Phase 4: learning abstraction for all streams)
# ---------------------------------------------------------------------------

class StreamLearner(ABC):
    """Abstract learning interface for any RAPA stream.

    Streams report WHAT they learned (status). The kernel decides
    WHAT TO DO (regime gating, compression eligibility).

    Lifecycle per episode:
      1. observe_signal() called every tick (data accumulation)
      2. learn() called at episode end (training step)
      3. ready() queried for regime/gating decisions

    Phase 4: Interface only. Existing agents use NullLearner (default).
    Phase 5: Plug in real implementations (VAE, Dreamer, SAC, etc.)
    """

    @abstractmethod
    def observe_signal(self, signal: "LearnerSignal") -> None:
        """Receive per-tick learning signal from kernel.

        Called after loop gain and closure residuum are computed,
        so all kernel observables are available in the signal.
        """
        ...

    @abstractmethod
    def learn(self) -> None:
        """Perform one learning step.

        Called at episode boundaries (done=True). The learner decides
        internally whether to actually train (enough data, cooldown, etc.).
        """
        ...

    @abstractmethod
    def ready(self) -> "LearnerStatus":
        """Report current learning readiness.

        The kernel uses this for regime gating: streams in TRAINING
        mode prevent GOALSEEK activation (weakest-link principle).
        """
        ...

    def reset_episode(self) -> None:
        """Reset per-episode learning state. Default: no-op."""
        pass


class NullLearner(StreamLearner):
    """Default adapter: deterministic agent, always ready.

    All existing agents use this via the .learner property on their
    Stream ABC. Phase 5 replaces NullLearner with real implementations.
    """

    def __init__(self, label: str = "deterministic"):
        self._label = label

    def observe_signal(self, signal: "LearnerSignal") -> None:
        pass  # Deterministic agents don't accumulate data

    def learn(self) -> None:
        pass  # Nothing to learn

    def ready(self) -> "LearnerStatus":
        from kernel.types import LearnerStatus, LearnerMode
        return LearnerStatus(
            mode=LearnerMode.OFF,
            accuracy=1.0,
            label=self._label,
        )

    def reset_episode(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Environment adapter (eval-script infrastructure, NOT used by kernel)
# ---------------------------------------------------------------------------

class EnvironmentAdapter(ABC):
    """Domain adapter that standardizes eval-script setup across environments.

    The MvpKernel does NOT depend on this class.  It exists to eliminate
    boilerplate duplication in eval scripts: environment creation, agent
    construction, deconstruction function wiring, and per-tick metadata.
    """

    @abstractmethod
    def reset(self) -> dict:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[dict, float, bool]:
        """Execute *action* and return (obs, reward, done)."""
        ...

    @abstractmethod
    def available_actions(self, obs: dict) -> List[str]:
        """Return the list of valid actions for the given observation."""
        ...

    @abstractmethod
    def make_agents(
        self,
        variant: str = "with_d",
    ) -> Tuple[StreamA, StreamB, StreamC, Optional[StreamD]]:
        """Construct domain-specific agents for the given variant.

        *variant* is typically "with_d" or "no_d".
        Returns (A, B, C, D_or_None).
        """
        ...

    @abstractmethod
    def get_goal_map(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """Return the goal_map for deconstruction, or None."""
        ...

    @abstractmethod
    def get_deconstruct_fn(self) -> Callable:
        """Return the domain-specific D->C deconstruction function."""
        ...

    @abstractmethod
    def inject_obs_metadata(self, kernel, obs: dict) -> None:
        """Inject domain-specific metadata into kernel state per tick.

        Examples: TextWorld injects visited_rooms; GridWorld redirects
        C's target toward the hint cell when no target is known yet.
        """
        ...

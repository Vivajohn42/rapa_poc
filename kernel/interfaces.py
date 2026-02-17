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


class StreamB(ABC):
    """Dynamics / forward model stream: predict next state given action."""

    @abstractmethod
    def predict_next(self, zA: ZA, action: str) -> ZA:
        """Return the predicted ZA after taking *action* in state *zA*.

        Must be deterministic for the same (zA, action) pair so that
        loop gain and closure residuum computations are stable.
        """
        ...


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

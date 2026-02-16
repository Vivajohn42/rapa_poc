"""
TaskChangeGridWorld: Two-phase environment for testing deconstruct's
stabilizing effect during task changes.

Phase 1: Agent seeks Goal A. Hint cell 1 reveals "target is A".
Phase 2: After switch, true goal changes to B. Hint cell 2 reveals "target is B".

The switch is triggered either by step count or by Phase 1 goal arrival.
"""

from typing import Tuple, Optional, List, Dict

from env.gridworld import GridWorld, GoalDef, HintCellDef, GridState


class TaskChangeGridWorld:
    """
    Wrapper around GridWorld that introduces a mid-episode goal switch.

    - Phase 1 uses hint cell 0 (reveals phase1_goal_id)
    - Phase 2 uses hint cell 1 (reveals phase2_goal_id, enabled at switch)
    - Agent position, obstacles, and grid state persist across the switch
    - Only the true goal and hint availability change
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: Optional[int] = None,
        phase1_goal_id: str = "A",
        phase2_goal_id: str = "B",
        switch_after_steps: Optional[int] = None,
        n_random_obstacles: int = 5,
    ):
        self.width = width
        self.height = height
        self._phase1_goal_id = phase1_goal_id
        self._phase2_goal_id = phase2_goal_id
        self._switch_after_steps = switch_after_steps
        self._switch_on_phase1_complete = (switch_after_steps is None)

        # Goal definitions
        goal_a = GoalDef("A", (width - 1, height - 1))
        goal_b = GoalDef("B", (width - 1, 0))
        self._goals = [goal_a, goal_b]
        self._goal_map: Dict[str, Tuple[int, int]] = {
            g.goal_id: g.pos for g in self._goals
        }

        # Hint cell positions
        self.hint1_pos = (0, height - 1)                   # Phase 1: corner
        self.hint2_pos = (width // 2, height // 2)         # Phase 2: center

        # Build hint cells — both use legacy/direct mode
        # Phase 1 hint: reveals phase1_goal_id directly
        # Phase 2 hint: starts disabled, text set at switch time
        hint_cells = [
            HintCellDef(pos=self.hint1_pos, hint_text=phase1_goal_id),
            HintCellDef(pos=self.hint2_pos, hint_text=""),  # empty until switch
        ]

        # Build underlying GridWorld — force phase1 goal as the chosen goal
        self.env = GridWorld(
            width=width,
            height=height,
            seed=seed,
            goals=self._goals,
            hint_cells=hint_cells,
            obstacles=[(2, 2)] if width >= 5 else [],
            n_random_obstacles=n_random_obstacles,
        )

        # Phase state
        self._phase = 0  # set properly by reset()
        self._phase_switched = False
        self._switch_step: Optional[int] = None

    @property
    def goal_map(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._goal_map)

    @property
    def current_phase(self) -> int:
        return self._phase

    @property
    def phase_switched(self) -> bool:
        return self._phase_switched

    @property
    def switch_step(self) -> Optional[int]:
        return self._switch_step

    @property
    def agent_pos(self) -> Tuple[int, int]:
        return self.env.agent_pos

    def reset(self) -> GridState:
        obs = self.env.reset()

        # Force Phase 1 goal (GridWorld.reset() chooses randomly)
        self.env.goal_id = self._phase1_goal_id
        self.env.true_goal_pos = self._goal_map[self._phase1_goal_id]

        # Phase 1 hint: enabled, reveals phase1_goal_id
        self.env._hints_available[0] = True
        self.env._hint_texts[0] = self._phase1_goal_id

        # Phase 2 hint: disabled until switch
        self.env._hints_available[1] = False
        self.env._hint_texts[1] = ""

        # Reset phase state
        self._phase = 1
        self._phase_switched = False
        self._switch_step = None

        return obs

    def step(self, action: str) -> Tuple[GridState, float, bool]:
        # Check switch condition BEFORE step
        should_switch = False
        if self._phase == 1 and not self._phase_switched:
            if self._switch_on_phase1_complete:
                # Switch when agent reaches Phase 1 goal
                if self.env.agent_pos == self._goal_map[self._phase1_goal_id]:
                    should_switch = True
            elif self._switch_after_steps is not None:
                if self.env.t >= self._switch_after_steps:
                    should_switch = True

        if should_switch:
            self._do_switch()

        # Delegate to GridWorld
        obs, reward, _ = self.env.step(action)

        # Override done: check against current phase's goal
        done = (self.env.agent_pos == self.env.true_goal_pos)
        # In Phase 1 (before switch), reaching Goal A does NOT end the episode
        # — the switch condition above triggers the goal change instead.
        # After switch, reaching Goal B ends the episode.
        if self._phase == 1 and not self._phase_switched:
            done = False

        reward = 1.0 if done else -0.01

        return obs, reward, done

    def _do_switch(self):
        """Switch from Phase 1 to Phase 2."""
        self._phase = 2
        self._phase_switched = True
        self._switch_step = self.env.t

        # Update true goal to Phase 2 goal
        self.env.goal_id = self._phase2_goal_id
        self.env.true_goal_pos = self._goal_map[self._phase2_goal_id]

        # Enable Phase 2 hint cell
        self.env._hints_available[1] = True
        self.env._hint_texts[1] = self._phase2_goal_id

    def observe(self) -> GridState:
        return self.env.observe()

from dataclasses import dataclass, field
import random
from typing import List, Tuple, Optional, Dict


@dataclass
class GridState:
    width: int
    height: int
    agent_pos: tuple
    goal_pos: tuple          # always hidden => (-1,-1)
    obstacles: list
    hint: str | None         # appears ONLY when stepping on hint cell (one-time)


@dataclass
class GoalDef:
    """Definition of a candidate goal."""
    goal_id: str             # e.g. "A", "B", "C", "D"
    pos: Tuple[int, int]     # grid coordinates


@dataclass
class HintCellDef:
    """
    Definition of a hint cell that partitions goal candidates into two groups.

    When the agent visits this cell, it learns which group the true goal is NOT in.
    The partition is defined by `group_a` and `group_b` (lists of goal_ids).
    At reset, the env computes which group to eliminate based on the true goal.

    For backward-compatible 2-goal mode: leave groups empty and set
    eliminates/hint_text manually, or use the default hint cell.
    """
    pos: Tuple[int, int]
    group_a: List[str] = None  # e.g. ["A", "B"] — one partition
    group_b: List[str] = None  # e.g. ["C", "D"] — other partition
    # Legacy fields (used for backward compatibility or direct override)
    eliminates: List[str] = None
    hint_text: str = ""

    def __post_init__(self):
        if self.group_a is None:
            self.group_a = []
        if self.group_b is None:
            self.group_b = []
        if self.eliminates is None:
            self.eliminates = []


class GridWorld:
    """
    Parametrizable gridworld environment.

    Backward-compatible: GridWorld() produces the original 5x5 with 2 goals,
    1 hint cell, 1 obstacle.

    New features for Stufe 3:
    - Variable grid size (width, height)
    - Multiple candidate goals (2-N)
    - Multiple hint cells, each eliminating some candidates
    - Configurable obstacles (static list or random count)
    - Dynamic obstacles (move periodically if enabled)
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        seed: int | None = None,
        goals: Optional[List[GoalDef]] = None,
        hint_cells: Optional[List[HintCellDef]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        n_random_obstacles: int = 0,
        dynamic_obstacles: bool = False,
        dynamic_obstacle_period: int = 5,
    ):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        # Goal configuration
        if goals is not None:
            self._goal_defs = goals
        else:
            # Default: 2 goals (backward-compatible)
            self._goal_defs = [
                GoalDef("A", (width - 1, height - 1)),
                GoalDef("B", (width - 1, 0)),
            ]

        # Hint cell configuration
        if hint_cells is not None:
            self._hint_cell_defs = hint_cells
        else:
            # Default: 1 hint cell at bottom-left (backward-compatible)
            self._hint_cell_defs = [
                HintCellDef(pos=(0, height - 1), eliminates=[], hint_text=""),
            ]

        # Obstacle configuration
        self._fixed_obstacles = obstacles if obstacles is not None else [(2, 2)]
        self._n_random_obstacles = n_random_obstacles
        self._dynamic_obstacles_enabled = dynamic_obstacles
        self._dynamic_period = dynamic_obstacle_period

        # Build goal lookup
        self._goal_map: Dict[str, Tuple[int, int]] = {
            g.goal_id: g.pos for g in self._goal_defs
        }

        # For backward compatibility: hint_cell property returns first hint cell pos
        self.hint_cell = self._hint_cell_defs[0].pos if self._hint_cell_defs else None

        self.reset()

    def reset(self):
        self.t = 0
        self.agent_pos = (0, 0)

        # Choose true goal randomly from candidates
        chosen = self.rng.choice(self._goal_defs)
        self.goal_id = chosen.goal_id
        self.true_goal_pos = chosen.pos

        # Backward compatibility: goal_A, goal_B
        if len(self._goal_defs) >= 2:
            self.goal_A = self._goal_defs[0].pos
            self.goal_B = self._goal_defs[1].pos

        # Static obstacles
        self.obstacles = list(self._fixed_obstacles)

        # Random obstacles
        if self._n_random_obstacles > 0:
            reserved = {(0, 0), self.true_goal_pos}
            reserved.update(g.pos for g in self._goal_defs)
            reserved.update(h.pos for h in self._hint_cell_defs)
            reserved.update(tuple(o) for o in self.obstacles)

            candidates = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if (x, y) not in reserved
            ]
            n_add = min(self._n_random_obstacles, len(candidates))
            self.obstacles.extend(self.rng.sample(candidates, n_add))

        # Dynamic obstacles state
        self._dynamic_obstacles: List[Tuple[int, int]] = []
        if self._dynamic_obstacles_enabled:
            self._init_dynamic_obstacles()

        # Hint cell state: track which hint cells have been visited
        self._hints_available: Dict[int, bool] = {
            i: True for i in range(len(self._hint_cell_defs))
        }
        self._hint_pending: Optional[str] = None

        # Compute per-hint-cell elimination based on the chosen true goal
        self._hint_eliminates: Dict[int, List[str]] = {}
        self._hint_texts: Dict[int, str] = {}
        for i, hdef in enumerate(self._hint_cell_defs):
            if hdef.group_a and hdef.group_b:
                # Partition mode: eliminate the group the true goal is NOT in
                if self.goal_id in hdef.group_a:
                    # True goal in group_a → eliminate group_b
                    self._hint_eliminates[i] = list(hdef.group_b)
                    self._hint_texts[i] = "not_" + "_".join(
                        g.lower() for g in hdef.group_b
                    )
                else:
                    # True goal in group_b → eliminate group_a
                    self._hint_eliminates[i] = list(hdef.group_a)
                    self._hint_texts[i] = "not_" + "_".join(
                        g.lower() for g in hdef.group_a
                    )
            else:
                # Legacy/direct mode
                self._hint_eliminates[i] = list(hdef.eliminates)
                self._hint_texts[i] = hdef.hint_text

        # For multi-goal: track which goals have been eliminated by hints
        self.eliminated_goals: List[str] = []

        return self.observe()

    def _init_dynamic_obstacles(self):
        """Place dynamic obstacles that move periodically."""
        reserved = {(0, 0), self.true_goal_pos}
        reserved.update(g.pos for g in self._goal_defs)
        reserved.update(h.pos for h in self._hint_cell_defs)
        reserved.update(tuple(o) for o in self.obstacles)

        candidates = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in reserved
        ]
        n_dyn = min(max(1, self.width * self.height // 20), len(candidates))
        self._dynamic_obstacles = self.rng.sample(candidates, n_dyn)

    def _move_dynamic_obstacles(self):
        """Move each dynamic obstacle to a random adjacent cell."""
        if not self._dynamic_obstacles:
            return

        static_set = set(self.obstacles)
        goal_set = {g.pos for g in self._goal_defs}
        hint_set = {h.pos for h in self._hint_cell_defs}
        protected = static_set | goal_set | hint_set | {(0, 0), self.agent_pos}

        new_positions = []
        occupied = set(self.obstacles)
        for ox, oy in self._dynamic_obstacles:
            neighbors = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ox + dx, oy + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and (nx, ny) not in protected
                        and (nx, ny) not in occupied
                        and (nx, ny) not in new_positions):
                    neighbors.append((nx, ny))
            if neighbors:
                new_pos = self.rng.choice(neighbors)
            else:
                new_pos = (ox, oy)  # stay if no valid move
            new_positions.append(new_pos)
            occupied.add(new_pos)

        self._dynamic_obstacles = new_positions

    @property
    def all_obstacles(self) -> List[Tuple[int, int]]:
        """All obstacles (static + dynamic)."""
        return self.obstacles + self._dynamic_obstacles

    def observe(self):
        # goal always hidden
        visible_goal = (-1, -1)

        # hint only returned once when pending
        hint = self._hint_pending
        self._hint_pending = None

        return GridState(
            width=self.width,
            height=self.height,
            agent_pos=self.agent_pos,
            goal_pos=visible_goal,
            obstacles=self.all_obstacles,
            hint=hint
        )

    def step(self, action: str):
        self.t += 1

        # Move dynamic obstacles periodically
        if self._dynamic_obstacles_enabled and self.t % self._dynamic_period == 0:
            self._move_dynamic_obstacles()

        x, y = self.agent_pos
        moves = {
            "up": (x, y - 1),
            "down": (x, y + 1),
            "left": (x - 1, y),
            "right": (x + 1, y)
        }

        if action in moves:
            nx, ny = moves[action]
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in set(self.all_obstacles):
                    self.agent_pos = (nx, ny)

        # Check all hint cells
        for i, hdef in enumerate(self._hint_cell_defs):
            if self._hints_available[i] and self.agent_pos == hdef.pos:
                self._hints_available[i] = False
                hint_text = self._hint_texts.get(i, "")
                eliminates = self._hint_eliminates.get(i, [])
                if hint_text:
                    # Multi-goal mode: dynamically computed hint
                    self._hint_pending = hint_text
                    self.eliminated_goals.extend(eliminates)
                else:
                    # Backward-compatible mode: emit goal_id directly
                    self._hint_pending = self.goal_id

        done = self.agent_pos == self.true_goal_pos
        reward = 1 if done else -0.01

        return self.observe(), reward, done

    # ── Multi-goal helpers ────────────────────────────────────────────

    @property
    def remaining_goals(self) -> List[GoalDef]:
        """Goals not yet eliminated by hints."""
        return [g for g in self._goal_defs if g.goal_id not in self.eliminated_goals]

    @property
    def n_goals(self) -> int:
        return len(self._goal_defs)

    @property
    def goal_positions(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._goal_map)

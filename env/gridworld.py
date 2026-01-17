from dataclasses import dataclass
import random

@dataclass
class GridState:
    width: int
    height: int
    agent_pos: tuple
    goal_pos: tuple          # always hidden => (-1,-1)
    obstacles: list
    hint: str | None         # appears ONLY when stepping on hint cell (one-time)


class GridWorld:
    def __init__(self, width=5, height=5, seed: int | None = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.agent_pos = (0, 0)

        # Two possible true goals
        self.goal_A = (self.width - 1, self.height - 1)  # (4,4)
        self.goal_B = (self.width - 1, 0)                # (4,0)
        self.goal_id = self.rng.choice(["A", "B"])
        self.true_goal_pos = self.goal_A if self.goal_id == "A" else self.goal_B

        self.obstacles = [(2, 2)]

        # Hint cell: agent must reach it to see hint (A2)
        self.hint_cell = (0, self.height - 1)  # (0,4)
        self._hint_available = True
        self._hint_pending = None  # set when stepping on hint cell, returned once

        return self.observe()

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
            obstacles=self.obstacles,
            hint=hint
        )

    def step(self, action: str):
        self.t += 1

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
                if (nx, ny) not in self.obstacles:
                    self.agent_pos = (nx, ny)

        # If agent reaches hint cell, emit hint once
        if self._hint_available and self.agent_pos == self.hint_cell:
            self._hint_available = False
            self._hint_pending = self.goal_id  # "A" or "B"

        done = self.agent_pos == self.true_goal_pos
        reward = 1 if done else -0.01

        return self.observe(), reward, done

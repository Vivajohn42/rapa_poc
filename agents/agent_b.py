from state.schema import ZA
from kernel.interfaces import StreamB

ACTIONS = ("up", "down", "left", "right")


class AgentB(StreamB):
    """
    Deterministisches Dynamikmodell für die Gridworld.
    Vorhersage: zA_next = f(zA, action)
    """
    def predict_next(self, zA: ZA, action: str) -> ZA:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        x, y = zA.agent_pos
        if action == "up":
            nx, ny = x, y - 1
        elif action == "down":
            nx, ny = x, y + 1
        elif action == "left":
            nx, ny = x - 1, y
        else:  # right
            nx, ny = x + 1, y

        # boundary check
        if not (0 <= nx < zA.width and 0 <= ny < zA.height):
            nx, ny = x, y

        # obstacle check
        if (nx, ny) in set(zA.obstacles):
            nx, ny = x, y

        return ZA(
            width=zA.width,
            height=zA.height,
            agent_pos=(nx, ny),
            goal_pos=zA.goal_pos,
            obstacles=zA.obstacles,
        )

from typing import Tuple, List
from state.schema import ZA

ACTIONS = ("up", "down", "left", "right")


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def baseline_monolithic_policy(zA: ZA, mode: str = "seek") -> str:
    """
    Monolithic baseline: single policy that chooses action directly from zA.
    No AgentB rollout, no Router, no D.
    Uses simple Manhattan heuristics + anti-stay rule.
    """
    assert mode in ("seek", "avoid")

    x, y = zA.agent_pos
    gx, gy = zA.goal_pos

    # candidate moves (naive)
    candidates: List[str] = list(ACTIONS)

    # Score each action by the *intended* next position, without consulting B.
    # This is intentionally "monolithic" and less grounded.
    best_a = None
    best_s = -1e9

    for a in candidates:
        nx, ny = x, y
        if a == "up":
            ny -= 1
        elif a == "down":
            ny += 1
        elif a == "left":
            nx -= 1
        elif a == "right":
            nx += 1

        # clamp boundaries (still doesn't model obstacles)
        if not (0 <= nx < zA.width and 0 <= ny < zA.height):
            nx, ny = x, y

        d_now = manhattan((x, y), (gx, gy))
        d_next = manhattan((nx, ny), (gx, gy))

        if mode == "seek":
            s = d_now - d_next
        else:
            s = d_next - d_now

        # anti-stay to avoid degenerate freeze
        if (nx, ny) == (x, y):
            s -= 1.1

        if s > best_s:
            best_s = s
            best_a = a

    return best_a or "down"

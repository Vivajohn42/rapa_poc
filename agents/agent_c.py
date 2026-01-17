from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

from state.schema import ZA

ACTIONS = ("up", "down", "left", "right")


@dataclass
class GoalSpec:
    mode: str  # "seek" or "avoid"
    target: Tuple[int, int]  # usually zA.goal_pos


class AgentC:
    """
    Valence/Goal Agent:
    - bewertet Aktionen relativ zu einem Ziel (seek/avoid)
    - nutzt B (predict_next) nur als "Rollout"
    - kann optional ein Memory für tie-breaker nutzen
    """

    def __init__(self, goal: GoalSpec, anti_stay_penalty: float = 0.25):
        if goal.mode not in ("seek", "avoid"):
            raise ValueError("goal.mode must be 'seek' or 'avoid'")
        self.goal = goal
        self.anti_stay_penalty = float(anti_stay_penalty)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def score_action(self, zA: ZA, zA_next: ZA) -> float:
        """
        Base score:
          seek  -> positive if we got closer (d_now - d_next)
          avoid -> positive if we got farther (d_next - d_now)

        Anti-stay penalty:
          if action results in no movement (hit wall/obstacle), subtract a small penalty
          to prevent degenerate "freeze" policies.
        """
        d_now = self._manhattan(zA.agent_pos, self.goal.target)
        d_next = self._manhattan(zA_next.agent_pos, self.goal.target)

        if self.goal.mode == "seek":
            base = float(d_now - d_next)
        else:
            base = float(d_next - d_now)

        # discourage staying in place (wall/obstacle bumps)
        if zA_next.agent_pos == zA.agent_pos:
            base -= self.anti_stay_penalty

        return base

    def choose_action(
        self,
        zA: ZA,
        predict_next_fn,
        memory: Optional[Dict[str, Any]] = None,
        tie_break_delta: float = 0.25
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        predict_next_fn: function(zA, action) -> zA_next (from Agent B)

        Returns:
          best_action, scored_list

        Tie-break:
          If decision uncertainty is high (top-1 and top-2 within tie_break_delta),
          and memory contains "tie_break_preference" list, pick the first preferred
          action among near-best actions.
        """
        scored: List[Tuple[str, float]] = []
        for a in ACTIONS:
            zA_next = predict_next_fn(zA, a)
            s = self.score_action(zA, zA_next)
            scored.append((a, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # uncertainty margin
        if len(scored) >= 2:
            delta = scored[0][1] - scored[1][1]
        else:
            delta = 999.0

        # tie-break using memory preference if uncertain
        if memory and delta < tie_break_delta:
            pref = memory.get("tie_break_preference")
            if isinstance(pref, list) and pref:
                best = scored[0][1]
                near_best = [a for a, s in scored if (best - s) <= tie_break_delta]

                for p in pref:
                    if p in near_best:
                        return p, scored

        return scored[0][0], scored

"""
Shadow-D: Forward-planning agent using B's deterministic model (Stufe 8).

Unlike narrative-D which processes past events, Shadow-D plans future
trajectories. It uses Agent B's predict_next as a forward model to
evaluate action sequences via beam search.

Key advantage over C's 1-step lookahead: Shadow-D looks N steps ahead,
which lets it detect and avoid dead-ends and obstacles that C cannot see
from its single-step horizon.

Output flows to C via deconstruct_plan_to_c -> tie_break_preference.
"""

from typing import Tuple, List, Optional, Callable, Dict, Any
from dataclasses import dataclass

from state.schema import ZA, ZPlan

ACTIONS = ("up", "down", "left", "right")


@dataclass
class _RolloutResult:
    """Internal result of a single action-sequence rollout."""
    action_sequence: List[str]
    final_pos: Tuple[int, int]
    manhattan_improvement: float
    wall_hits: int
    unique_positions: int
    reward_estimate: float


class AgentShadowD:
    """
    Forward-planning agent that evaluates multi-step action sequences.

    Uses B's predict_next for deterministic rollouts. Scores sequences by:
    - Manhattan distance improvement toward target
    - Wall/obstacle avoidance (penalty for blocked moves)
    - Position diversity (penalty for revisiting same positions)
    """

    def __init__(
        self,
        predict_next_fn: Callable,
        rollout_depth: int = 5,
        beam_width: int = 8,
    ):
        self.predict_next = predict_next_fn
        self.rollout_depth = rollout_depth
        self.beam_width = beam_width

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(
        self,
        zA: ZA,
        target: Tuple[int, int],
        goal_mode: str,
    ) -> ZPlan:
        """
        Generate a plan by evaluating action sequences via beam search.

        Uses B's forward model to simulate trajectories and scores them
        by Manhattan improvement, obstacle avoidance, and diversity.

        Args:
            zA: Current observation state
            target: Goal position to navigate toward
            goal_mode: "seek" or "avoid"

        Returns:
            ZPlan with recommended actions and confidence
        """
        if target == (-1, -1):
            # No target known — can't plan
            return ZPlan(confidence=0.0)

        # Phase 1: Expand first actions
        initial_candidates = []
        for action in ACTIONS:
            zA_next = self.predict_next(zA, action)
            initial_candidates.append((action, zA_next))

        # Phase 2: Beam search over action sequences
        # Each beam element: (action_sequence, current_zA)
        beams = [([ a], zA_next) for a, zA_next in initial_candidates]

        for depth in range(1, self.rollout_depth):
            next_beams = []
            for seq, current_zA in beams:
                for action in ACTIONS:
                    zA_next = self.predict_next(current_zA, action)
                    next_beams.append((seq + [action], zA_next))

            # Score and prune to beam_width
            scored = []
            for seq, final_zA in next_beams:
                score = self._score_sequence(zA, final_zA, seq, target, goal_mode)
                scored.append((score, seq, final_zA))

            scored.sort(key=lambda x: x[0], reverse=True)
            beams = [(seq, final_zA) for _, seq, final_zA in scored[:self.beam_width]]

        # Phase 3: Score final beams
        rollout_results = []
        for seq, final_zA in beams:
            result = self._evaluate_rollout(zA, seq, target, goal_mode)
            rollout_results.append(result)

        # Phase 4: Rank and select best
        rollout_results.sort(key=lambda r: r.reward_estimate, reverse=True)

        if not rollout_results:
            return ZPlan(confidence=0.0)

        best = rollout_results[0]

        # Compute confidence: how much better is the best vs average?
        scores = [r.reward_estimate for r in rollout_results]
        avg_score = sum(scores) / len(scores)
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0.0

        if score_range > 0:
            confidence = min(1.0, (best.reward_estimate - avg_score) / score_range)
        elif best.reward_estimate > 0:
            confidence = 0.8  # All paths equally good
        else:
            confidence = 0.1  # No good path found

        # Risk: fraction of wall hits in best plan
        risk = best.wall_hits / self.rollout_depth if self.rollout_depth > 0 else 0.0

        return ZPlan(
            recommended_actions=best.action_sequence,
            plan_horizon=len(best.action_sequence),
            plan_score=best.reward_estimate,
            risk_score=risk,
            alternative_plans=len(rollout_results) - 1,
            confidence=max(0.0, confidence),
        )

    def _score_sequence(
        self,
        zA_start: ZA,
        zA_end: ZA,
        action_seq: List[str],
        target: Tuple[int, int],
        goal_mode: str,
    ) -> float:
        """Quick score for beam pruning."""
        d_start = self._manhattan(zA_start.agent_pos, target)
        d_end = self._manhattan(zA_end.agent_pos, target)

        if goal_mode == "seek":
            return float(d_start - d_end)
        else:
            return float(d_end - d_start)

    def _evaluate_rollout(
        self,
        zA_start: ZA,
        action_seq: List[str],
        target: Tuple[int, int],
        goal_mode: str,
    ) -> _RolloutResult:
        """Full evaluation of an action sequence."""
        current = zA_start
        positions = [current.agent_pos]
        wall_hits = 0
        manhattan_improvement = 0.0

        d_start = self._manhattan(zA_start.agent_pos, target)

        for action in action_seq:
            next_zA = self.predict_next(current, action)

            # Detect wall/obstacle hit (position didn't change)
            if next_zA.agent_pos == current.agent_pos:
                wall_hits += 1

            positions.append(next_zA.agent_pos)
            current = next_zA

        d_end = self._manhattan(current.agent_pos, target)

        if goal_mode == "seek":
            manhattan_improvement = float(d_start - d_end)
        else:
            manhattan_improvement = float(d_end - d_start)

        unique_positions = len(set(positions))

        # Reward estimate: manhattan improvement + diversity bonus - wall penalty
        reward = manhattan_improvement
        reward += 0.1 * unique_positions  # encourage exploration
        reward -= 0.3 * wall_hits          # penalize bumping into walls

        return _RolloutResult(
            action_sequence=list(action_seq),
            final_pos=current.agent_pos,
            manhattan_improvement=manhattan_improvement,
            wall_hits=wall_hits,
            unique_positions=unique_positions,
            reward_estimate=reward,
        )

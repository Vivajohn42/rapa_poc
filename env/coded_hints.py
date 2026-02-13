"""
Coded hint system for Stufe 6: Semantic Ambiguity.

Instead of direct hints ("A", "B", "not_c_d"), the environment emits
coded/directional clues that require interpretation. Only Agent D
(with interpretation capability) can translate these into actionable
goal identifiers.

Three difficulty levels:
- easy:   Absolute directional clues ("goal_at_bottom_right")
- medium: Relative/comparative clues ("goal_further_from_start")
- hard:   Abstract property clues ("first_coord_large")
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from env.gridworld import GridWorld, GridState
from state.schema import ZA


@dataclass
class CodedHintResult:
    """Result of encoding a hint."""
    code: str           # The coded clue string
    true_goal_id: str   # Which goal it actually identifies
    difficulty: str     # "easy", "medium", "hard"


class HintEncoder:
    """
    Encodes goal identification into indirect/directional clues.

    The encoder is deterministic: given a goal_map and grid dimensions,
    it produces consistent coded hints for each goal_id.
    """

    def __init__(
        self,
        goal_map: Dict[str, Tuple[int, int]],
        grid_width: int,
        grid_height: int,
    ):
        self.goal_map = goal_map
        self.width = grid_width
        self.height = grid_height
        self._mid_x = grid_width / 2.0
        self._mid_y = grid_height / 2.0

    def encode(self, true_goal_id: str, difficulty: str = "medium") -> str:
        """
        Generate a coded hint that indirectly identifies the true goal.

        The coded string is NOT parsable by deconstruct_d_to_c (which only
        matches "hint:A", "hint:B", etc.). Only AgentDInterpreter can
        translate it back to a goal_id.

        Args:
            true_goal_id: The actual goal ID to encode
            difficulty: "easy", "medium", or "hard"

        Returns:
            Coded hint string
        """
        pos = self.goal_map[true_goal_id]
        x, y = pos

        if difficulty == "easy":
            return self._encode_easy(true_goal_id, x, y)
        elif difficulty == "medium":
            return self._encode_medium(true_goal_id, x, y)
        elif difficulty == "hard":
            return self._encode_hard(true_goal_id, x, y)
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

    def _encode_easy(self, goal_id: str, x: int, y: int) -> str:
        """
        Absolute directional clue based on position in grid.
        Uses quadrant + distance qualifier for uniqueness.
        Example: "goal_at_bottom_right_far" for (9,9) in a 10x10 grid.
        """
        h_dir = "right" if x >= self._mid_x else "left"
        v_dir = "bottom" if y >= self._mid_y else "top"
        # Add distance qualifier for disambiguation when multiple goals
        # share the same quadrant
        dist = abs(x) + abs(y)
        max_dist = self.width + self.height - 2
        if dist > max_dist * 0.6:
            qualifier = "far"
        elif dist < max_dist * 0.4:
            qualifier = "near"
        else:
            qualifier = "mid"
        return f"goal_at_{v_dir}_{h_dir}_{qualifier}"

    def _encode_medium(self, goal_id: str, x: int, y: int) -> str:
        """
        Relative/comparative clue that requires reasoning about goal positions.
        Examples:
        - "goal_further_from_start" (further from (0,0))
        - "goal_closer_to_center"
        - "goal_highest_x_coord"
        """
        # Compute distances from origin for all goals
        distances = {}
        for gid, (gx, gy) in self.goal_map.items():
            distances[gid] = abs(gx) + abs(gy)

        # Find the property that uniquely identifies this goal
        # Strategy: use ranking-based descriptions
        sorted_by_dist = sorted(self.goal_map.keys(), key=lambda g: distances[g])

        if sorted_by_dist[-1] == goal_id:
            return "goal_furthest_from_origin"
        elif sorted_by_dist[0] == goal_id:
            return "goal_closest_to_origin"

        # Fall back to coordinate-based
        x_coords = {gid: gx for gid, (gx, gy) in self.goal_map.items()}
        y_coords = {gid: gy for gid, (gx, gy) in self.goal_map.items()}

        sorted_by_x = sorted(self.goal_map.keys(), key=lambda g: x_coords[g])
        if sorted_by_x[-1] == goal_id:
            return "goal_rightmost"
        if sorted_by_x[0] == goal_id:
            return "goal_leftmost"

        sorted_by_y = sorted(self.goal_map.keys(), key=lambda g: y_coords[g])
        if sorted_by_y[-1] == goal_id:
            return "goal_lowest"  # highest y = bottom of grid
        if sorted_by_y[0] == goal_id:
            return "goal_highest"  # lowest y = top of grid

        # Fallback: use distance from center
        center_dists = {}
        for gid, (gx, gy) in self.goal_map.items():
            center_dists[gid] = abs(gx - self._mid_x) + abs(gy - self._mid_y)

        sorted_by_center = sorted(self.goal_map.keys(), key=lambda g: center_dists[g])
        if sorted_by_center[0] == goal_id:
            return "goal_nearest_center"
        return "goal_far_from_center"

    def _encode_hard(self, goal_id: str, x: int, y: int) -> str:
        """
        Abstract property clue that requires multi-step reasoning.
        Uses coordinate properties without explicit directions.
        Examples:
        - "first_coord_large" (x > midpoint)
        - "coords_sum_high" (x + y > threshold)
        - "coord_difference_positive" (x > y)
        """
        total = x + y
        max_total = self.width + self.height - 2

        # Use coordinate sum as primary discriminator
        if total > max_total * 0.6:
            return "coords_sum_high"
        elif total < max_total * 0.4:
            return "coords_sum_low"

        # Use coordinate difference
        if x > y:
            return "first_coord_dominates"
        elif y > x:
            return "second_coord_dominates"

        return "coords_balanced"

    def decode(self, coded_hint: str, difficulty: str = "medium") -> Optional[str]:
        """
        Decode a coded hint back to a goal_id.

        This is the inverse of encode() — used by AgentDInterpreter.
        Returns None if the hint cannot be decoded.
        """
        # Try encoding each goal_id and see which one matches
        for goal_id in self.goal_map:
            if self.encode(goal_id, difficulty) == coded_hint:
                return goal_id
        return None

    def can_uniquely_identify(self, difficulty: str) -> bool:
        """
        Check if the encoding at this difficulty uniquely identifies each goal.
        Returns True if no two goals produce the same coded hint.
        """
        codes = set()
        for goal_id in self.goal_map:
            code = self.encode(goal_id, difficulty)
            if code in codes:
                return False
            codes.add(code)
        return True


# ── Multi-goal elimination encoding ──────────────────────────────────

class EliminationEncoder:
    """
    Encodes partition-based elimination hints as coded clues.

    For multi-goal scenarios, instead of "not_c_d", emits coded clues like
    "eliminate_bottom_goals" or "keep_right_goals".
    """

    def __init__(
        self,
        goal_map: Dict[str, Tuple[int, int]],
        grid_width: int,
        grid_height: int,
    ):
        self.goal_map = goal_map
        self.width = grid_width
        self.height = grid_height
        self._hint_encoder = HintEncoder(goal_map, grid_width, grid_height)

    def encode_elimination(
        self,
        eliminated_ids: List[str],
        difficulty: str = "medium",
    ) -> str:
        """
        Encode an elimination hint (which goals to remove) as a coded clue.

        Args:
            eliminated_ids: List of goal IDs being eliminated
            difficulty: Coding difficulty

        Returns:
            Coded elimination string
        """
        if difficulty == "easy":
            return self._encode_elim_easy(eliminated_ids)
        elif difficulty == "medium":
            return self._encode_elim_medium(eliminated_ids)
        elif difficulty == "hard":
            return self._encode_elim_hard(eliminated_ids)
        return f"eliminate_{'_'.join(sorted(eliminated_ids))}"

    def _encode_elim_easy(self, eliminated_ids: List[str]) -> str:
        """Directional: 'eliminate_2_upper_right_goals' based on positions."""
        positions = [self.goal_map[gid] for gid in eliminated_ids]
        avg_y = sum(y for _, y in positions) / len(positions)
        mid_y = self.height / 2.0
        avg_x = sum(x for x, _ in positions) / len(positions)
        mid_x = self.width / 2.0

        v = "lower" if avg_y >= mid_y else "upper"
        h = "right" if avg_x >= mid_x else "left"
        count = len(eliminated_ids)
        return f"eliminate_{count}_{v}_{h}_goals"

    def _encode_elim_medium(self, eliminated_ids: List[str]) -> str:
        """Comparative: 'eliminate_2_further_goals' etc."""
        positions = [self.goal_map[gid] for gid in eliminated_ids]
        avg_dist = sum(abs(x) + abs(y) for x, y in positions) / len(positions)

        remaining = [gid for gid in self.goal_map if gid not in eliminated_ids]
        remaining_pos = [self.goal_map[gid] for gid in remaining]
        avg_rem_dist = sum(abs(x) + abs(y) for x, y in remaining_pos) / len(remaining_pos) if remaining_pos else 0

        count = len(eliminated_ids)
        if avg_dist > avg_rem_dist:
            return f"eliminate_{count}_further_goals"
        else:
            return f"eliminate_{count}_closer_goals"

    def _encode_elim_hard(self, eliminated_ids: List[str]) -> str:
        """Abstract: 'eliminate_2_high_coord_sum' etc."""
        positions = [self.goal_map[gid] for gid in eliminated_ids]
        avg_sum = sum(x + y for x, y in positions) / len(positions)
        threshold = (self.width + self.height) / 2.0

        count = len(eliminated_ids)
        if avg_sum > threshold:
            return f"eliminate_{count}_high_coord_sum"
        else:
            return f"eliminate_{count}_low_coord_sum"

    def decode_elimination(
        self,
        coded_hint: str,
        difficulty: str = "medium",
    ) -> Optional[List[str]]:
        """
        Decode a coded elimination hint back to eliminated goal IDs.

        Tries all possible elimination subsets and returns the one whose
        encoding matches the coded hint.
        """
        # For 2 goals, there are only 2 possible eliminations (each single goal)
        # For 4 goals with 2-partition, there are limited combos
        from itertools import combinations

        all_ids = list(self.goal_map.keys())
        # Try eliminating 1 to N-1 goals
        for k in range(1, len(all_ids)):
            for combo in combinations(all_ids, k):
                elim = list(combo)
                if self.encode_elimination(elim, difficulty) == coded_hint:
                    return elim
        return None


# ── CodedGridWorld Wrapper ────────────────────────────────────────────

class CodedGridWorld:
    """
    Wrapper around GridWorld that replaces direct hints with coded clues.

    All GridWorld attributes and methods are delegated, except step()
    which intercepts hint emissions and encodes them.
    """

    def __init__(
        self,
        env: GridWorld,
        difficulty: str = "medium",
    ):
        self.env = env
        self.difficulty = difficulty

        goal_map = env.goal_positions
        self._hint_encoder = HintEncoder(goal_map, env.width, env.height)
        self._elim_encoder = EliminationEncoder(goal_map, env.width, env.height)

        # Track original hint for verification
        self.last_original_hint: Optional[str] = None
        self.last_coded_hint: Optional[str] = None

    def __getattr__(self, name):
        """Delegate all attribute access to the wrapped env."""
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        # Re-create encoders in case goal_map changed after reset
        goal_map = self.env.goal_positions
        self._hint_encoder = HintEncoder(goal_map, self.env.width, self.env.height)
        self._elim_encoder = EliminationEncoder(goal_map, self.env.width, self.env.height)
        self.last_original_hint = None
        self.last_coded_hint = None
        return self._encode_obs(obs)

    def observe(self):
        obs = self.env.observe()
        return self._encode_obs(obs)

    def step(self, action: str):
        obs, reward, done = self.env.step(action)
        coded_obs = self._encode_obs(obs)
        return coded_obs, reward, done

    def _encode_obs(self, obs: GridState) -> GridState:
        """Replace direct hint with coded version if present."""
        if obs.hint is None:
            return obs

        self.last_original_hint = obs.hint
        coded = self._encode_hint(obs.hint)
        self.last_coded_hint = coded

        return GridState(
            width=obs.width,
            height=obs.height,
            agent_pos=obs.agent_pos,
            goal_pos=obs.goal_pos,
            obstacles=obs.obstacles,
            hint=coded,
        )

    def _encode_hint(self, raw_hint: str) -> str:
        """
        Encode a raw hint string into a coded clue.

        Handles two cases:
        1. Direct goal ID ("A", "B"): encode as directional/property clue
        2. Elimination hint ("not_c_d"): encode as coded elimination
        """
        if raw_hint.startswith("not_"):
            # Elimination hint: parse eliminated IDs
            parts = raw_hint[4:].split("_")
            eliminated = [p.upper() for p in parts]
            return self._elim_encoder.encode_elimination(
                eliminated, self.difficulty
            )
        elif len(raw_hint) <= 2 and raw_hint.upper() in self.env.goal_positions:
            # Direct goal ID hint
            return self._hint_encoder.encode(
                raw_hint.upper(), self.difficulty
            )
        else:
            # Unknown format — pass through
            return raw_hint

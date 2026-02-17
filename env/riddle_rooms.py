"""Riddle Rooms: a non-spatial, propositional logic puzzle environment.

Proves that the DEF/RAPA architecture works beyond navigation.
Agent D is essential: individual clues are ambiguous, only multi-clue
synthesis via constraint propagation can identify the correct answer.

State:   set of known propositions (evidence gathered so far)
Actions: test_X (reveals new evidence), submit_Y (proposes answer)
Success: submit correct answer after gathering sufficient evidence

No room graph, no movement — purely logical state transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random as _random


@dataclass
class RiddleClue:
    """A single clue revealed by a test action."""
    text: str
    eliminates: List[str]   # answer IDs this clue rules out
    requires: List[str]     # properties the answer must have (informational)


@dataclass
class RiddlePuzzle:
    """A logic puzzle with evidence-gathering actions and a hidden answer."""
    puzzle_id: str
    description: str                           # initial setup text
    answer_set: List[str]                      # all possible answers
    answer: str                                # correct answer
    tests: Dict[str, RiddleClue]               # test_name -> clue revealed
    answer_properties: Dict[str, Set[str]]     # answer_id -> set of properties
    initial_clue: Optional[str] = None         # hint shown at start


# ---------------------------------------------------------------------------
# 5 hand-crafted puzzles
# ---------------------------------------------------------------------------

def _make_puzzles() -> List[RiddlePuzzle]:
    """Return 5 puzzles of increasing complexity."""

    puzzles = []

    # 1. Liar Boxes (3 answers, 2 tests)
    puzzles.append(RiddlePuzzle(
        puzzle_id="liar_boxes",
        description="Three boxes: A, B, C. Each has a label, but ALL labels lie. One box contains gold.",
        answer_set=["box_a", "box_b", "box_c"],
        answer="box_b",
        tests={
            "test_label_a": RiddleClue(
                text="Box A's label says: 'The gold is in Box A.' Since all labels lie, the gold is NOT in Box A.",
                eliminates=["box_a"],
                requires=[],
            ),
            "test_label_c": RiddleClue(
                text="Box C's label says: 'The gold is in Box C.' Since all labels lie, the gold is NOT in Box C.",
                eliminates=["box_c"],
                requires=[],
            ),
        },
        answer_properties={
            "box_a": {"labeled", "claims_self"},
            "box_b": {"unlabeled", "middle"},
            "box_c": {"labeled", "claims_self"},
        },
        initial_clue="All box labels are guaranteed to be false.",
    ))

    # 2. Alibi Check (4 answers, 3 tests)
    puzzles.append(RiddlePuzzle(
        puzzle_id="alibi_check",
        description="Four suspects: Alice, Bob, Carol, Dave. One committed the crime. Check alibis to narrow it down.",
        answer_set=["alice", "bob", "carol", "dave"],
        answer="carol",
        tests={
            "test_alibi_alice": RiddleClue(
                text="Alice was seen at the library during the crime. Alice has a confirmed alibi.",
                eliminates=["alice"],
                requires=["no_alibi"],
            ),
            "test_alibi_bob": RiddleClue(
                text="Bob was on camera at the gym during the crime. Bob has a confirmed alibi.",
                eliminates=["bob"],
                requires=["no_alibi"],
            ),
            "test_alibi_dave": RiddleClue(
                text="Dave was verified to be abroad during the crime. Dave has a confirmed alibi.",
                eliminates=["dave"],
                requires=["no_alibi"],
            ),
        },
        answer_properties={
            "alice": {"has_alibi", "library"},
            "bob": {"has_alibi", "gym"},
            "carol": {"no_alibi", "no_witness"},
            "dave": {"has_alibi", "abroad"},
        },
    ))

    # 3. Sequence Rule (4 answers, 3 tests)
    puzzles.append(RiddlePuzzle(
        puzzle_id="sequence_rule",
        description="A number sequence follows a hidden rule. Determine the next number: 2, 4, 8, ?",
        answer_set=["10", "12", "16", "32"],
        answer="16",
        tests={
            "test_example_1": RiddleClue(
                text="The sequence goes 2, 4, 8. Each number is double the previous. Rule: multiply by 2.",
                eliminates=["10", "12", "32"],
                requires=["doubling"],
            ),
            "test_example_2": RiddleClue(
                text="Checking: 2*2=4, 4*2=8. The pattern is consistent doubling, not addition.",
                eliminates=["10", "12"],
                requires=["doubling"],
            ),
            "test_example_3": RiddleClue(
                text="If the rule were 'add increasing amounts': 2+2=4, 4+4=8, 8+8=16. Both rules give 16.",
                eliminates=["10", "32"],
                requires=[],
            ),
        },
        answer_properties={
            "10": {"addition_2"},
            "12": {"addition_4"},
            "16": {"doubling", "addition_increasing"},
            "32": {"doubling_twice"},
        },
    ))

    # 4. Schedule Puzzle (3 answers = 3 assignments, 3 tests)
    puzzles.append(RiddlePuzzle(
        puzzle_id="schedule_puzzle",
        description="Assign 3 people (X, Y, Z) to 3 time slots (Morning, Afternoon, Evening). Who is in the Morning slot?",
        answer_set=["person_x", "person_y", "person_z"],
        answer="person_z",
        tests={
            "test_constraint_1": RiddleClue(
                text="Person X cannot work in the Morning — X has a doctor's appointment.",
                eliminates=["person_x"],
                requires=[],
            ),
            "test_constraint_2": RiddleClue(
                text="Person Y must work in the Afternoon — Y has a standing commitment.",
                eliminates=["person_y"],
                requires=[],
            ),
            "test_constraint_3": RiddleClue(
                text="Confirming: X is not Morning, Y is Afternoon. By elimination, Z must be Morning.",
                eliminates=["person_x", "person_y"],
                requires=["elimination"],
            ),
        },
        answer_properties={
            "person_x": {"no_morning", "has_appointment"},
            "person_y": {"afternoon_only", "has_commitment"},
            "person_z": {"flexible", "available_morning"},
        },
    ))

    # 5. Inference Chain (4 answers, 4 tests)
    puzzles.append(RiddlePuzzle(
        puzzle_id="inference_chain",
        description="Given a set of if-then rules, determine which conclusion follows. Options: alpha, beta, gamma, delta.",
        answer_set=["alpha", "beta", "gamma", "delta"],
        answer="gamma",
        tests={
            "test_rule_1": RiddleClue(
                text="Rule: If the input is positive, then alpha is excluded. The input IS positive.",
                eliminates=["alpha"],
                requires=["positive_input"],
            ),
            "test_rule_2": RiddleClue(
                text="Rule: If alpha is excluded and the input is even, then beta is excluded. The input is even.",
                eliminates=["beta"],
                requires=["even_input"],
            ),
            "test_rule_3": RiddleClue(
                text="Rule: If beta is excluded, then the answer is either gamma or delta.",
                eliminates=["alpha", "beta"],
                requires=[],
            ),
            "test_rule_4": RiddleClue(
                text="Rule: If the input is less than 100, then delta is excluded. The input is 42.",
                eliminates=["delta"],
                requires=["small_input"],
            ),
        },
        answer_properties={
            "alpha": {"positive_excluded"},
            "beta": {"even_excluded"},
            "gamma": {"survives_all"},
            "delta": {"large_input_only"},
        },
    ))

    return puzzles


ALL_PUZZLES = _make_puzzles()


class RiddleRooms:
    """Non-spatial logic puzzle environment.

    The agent gathers evidence via ``test_*`` actions and submits
    an answer via ``submit_*``.  D is essential because individual
    clues only eliminate 1-2 options; multi-clue synthesis is needed
    to identify the unique answer.
    """

    def __init__(
        self,
        seed: int = 42,
        puzzle_id: Optional[str] = None,
    ):
        self._rng = _random.Random(seed)
        self._seed = seed
        if puzzle_id is not None:
            matches = [p for p in ALL_PUZZLES if p.puzzle_id == puzzle_id]
            if not matches:
                raise ValueError(f"Unknown puzzle_id: {puzzle_id}")
            self._puzzle = matches[0]
        else:
            self._puzzle = self._rng.choice(ALL_PUZZLES)

        # Public attributes (set on reset)
        self.puzzle: RiddlePuzzle = self._puzzle
        self.answer_set: List[str] = list(self._puzzle.answer_set)
        self.answer_index: Dict[str, int] = {
            a: i for i, a in enumerate(self.answer_set)
        }
        self.n_answers: int = len(self.answer_set)

        # Episode state
        self._evidence: List[str] = []       # collected clue texts
        self._revealed_tests: Set[str] = set()
        self._step_count: int = 0
        self._done: bool = False

    def reset(self) -> dict:
        """Reset and return the initial observation."""
        self._evidence = []
        self._revealed_tests = set()
        self._step_count = 0
        self._done = False

        # Build initial observation
        initial_clue = self._puzzle.initial_clue
        return self._make_obs(new_clue=initial_clue)

    def step(self, action: str) -> Tuple[dict, float, bool]:
        """Execute action. Returns (obs, reward, done)."""
        assert not self._done, "Episode already finished"
        self._step_count += 1

        # Submit action
        if action.startswith("submit_"):
            answer_id = action[len("submit_"):]
            if answer_id == self._puzzle.answer:
                return self._make_obs(), 1.0, True
            else:
                self._done = True
                return self._make_obs(), -0.3, True

        # Test action
        if action in self._puzzle.tests:
            if action not in self._revealed_tests:
                self._revealed_tests.add(action)
                clue = self._puzzle.tests[action]
                self._evidence.append(clue.text)
                return self._make_obs(new_clue=clue.text), 0.1, False
            else:
                # Already tested — no new info
                return self._make_obs(), -0.01, False

        # Unknown action
        return self._make_obs(), -0.01, False

    def _make_obs(self, new_clue: Optional[str] = None) -> dict:
        """Build observation dict (TextWorld-compatible structure)."""
        # Evidence hash for pseudo-position
        evidence_hash = hash(frozenset(self._revealed_tests)) % self.n_answers

        # Available actions
        exits = []
        for test_name in sorted(self._puzzle.tests.keys()):
            if test_name not in self._revealed_tests:
                exits.append(test_name)
        for ans in self.answer_set:
            exits.append(f"submit_{ans}")

        description = (
            f"{self._puzzle.description}\n"
            f"Evidence gathered: {len(self._evidence)} clue(s).\n"
            f"Possible answers: {', '.join(self.answer_set)}"
        )

        return {
            "room_id": f"state_{evidence_hash}",
            "description": description,
            "exits": exits,
            "clue": new_clue,
            "visited": set(self._revealed_tests),
        }

    @property
    def all_test_names(self) -> List[str]:
        """Return all available test action names."""
        return sorted(self._puzzle.tests.keys())

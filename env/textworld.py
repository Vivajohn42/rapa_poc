"""TextWorld: Clue Rooms environment where D is essential.

A network of rooms connected by named exits. Clues are scattered across rooms.
No single clue identifies the target room — only D's multi-clue synthesis can.
Without D, Agent C has no target and all exits score equally.

Usage:
    env = TextWorld(seed=42)
    obs = env.reset()
    obs, reward, done = env.step("north")
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Room:
    room_id: str
    description: str
    exits: Dict[str, str]          # exit_name -> destination room_id
    clue: Optional[str] = None     # shown once on first visit
    properties: Set[str] = field(default_factory=set)


@dataclass
class TextWorldScenario:
    rooms: Dict[str, Room]
    start_room: str
    target_room: str
    required_clues: int            # how many clues needed for synthesis
    clue_rules: List[Dict]         # [{property: ..., negate: bool}, ...] per clue


def _build_scenarios() -> List[TextWorldScenario]:
    """Hand-crafted scenario templates. Each requires 2-3 clues for synthesis."""
    scenarios = []

    # --- Scenario 0: Treasure Hunt (5 rooms, 2 clues) ---
    rooms_0 = {
        "hall": Room(
            "hall", "A grand entrance hall with marble floors and tall windows.",
            {"north": "library", "east": "kitchen"},
            properties={"has_windows", "large"},
        ),
        "library": Room(
            "library", "Dusty shelves line every wall. A torn page is pinned to a board.",
            {"south": "hall", "east": "study"},
            clue="The treasure is not in a room with windows.",
            properties={"has_books", "no_windows"},
        ),
        "kitchen": Room(
            "kitchen", "A warm kitchen with a window overlooking the garden.",
            {"west": "hall", "north": "study"},
            properties={"has_windows", "warm"},
        ),
        "study": Room(
            "study", "A small, windowless room with a single desk and candle.",
            {"west": "library", "south": "kitchen", "north": "vault"},
            clue="The treasure lies in the coldest room.",
            properties={"no_windows", "small"},
        ),
        "vault": Room(
            "vault", "A cold stone chamber with no windows. The air is still.",
            {"south": "study"},
            properties={"no_windows", "cold"},
        ),
    }
    scenarios.append(TextWorldScenario(
        rooms=rooms_0, start_room="hall", target_room="vault",
        required_clues=2,
        clue_rules=[
            {"eliminates_property": "has_windows"},  # clue 1
            {"requires_property": "has_stone"},       # clue 2
        ],
    ))

    # --- Scenario 1: Lost Key (6 rooms, 2 clues) ---
    rooms_1 = {
        "garden": Room(
            "garden", "A bright garden with flowers and a fountain.",
            {"north": "corridor", "east": "greenhouse"},
            properties={"outdoor", "has_water", "bright"},
        ),
        "corridor": Room(
            "corridor", "A long dim corridor with paintings on both sides.",
            {"south": "garden", "east": "bedroom", "north": "attic"},
            clue="The key was last seen near water, but not outdoors.",
            properties={"indoor", "dim"},
        ),
        "greenhouse": Room(
            "greenhouse", "Glass walls let in sunlight. Plants everywhere.",
            {"west": "garden"},
            properties={"has_glass", "bright", "outdoor"},
        ),
        "bedroom": Room(
            "bedroom", "A cozy room with a large bed and curtains drawn.",
            {"west": "corridor", "north": "bathroom"},
            properties={"indoor", "dark", "cozy"},
        ),
        "bathroom": Room(
            "bathroom", "White tiles and a dripping faucet. No windows.",
            {"south": "bedroom"},
            clue="The key prefers a room smaller than the garden.",
            properties={"indoor", "has_water", "small", "no_windows"},
        ),
        "attic": Room(
            "attic", "A cramped attic with old boxes. A skylight lets in faint light.",
            {"south": "corridor"},
            properties={"indoor", "small", "has_glass"},
        ),
    }
    scenarios.append(TextWorldScenario(
        rooms=rooms_1, start_room="garden", target_room="bathroom",
        required_clues=2,
        clue_rules=[
            {"requires_property": "has_water", "eliminates_property": "outdoor"},
            {"requires_property": "small"},
        ],
    ))

    # --- Scenario 2: Hidden Message (6 rooms, 3 clues) ---
    rooms_2 = {
        "foyer": Room(
            "foyer", "An elegant foyer with chandeliers and a red carpet.",
            {"north": "gallery", "east": "dining"},
            properties={"elegant", "bright", "large"},
        ),
        "gallery": Room(
            "gallery", "Paintings cover every wall. The room smells of oil paint.",
            {"south": "foyer", "east": "workshop"},
            clue="The message is hidden in a room that is neither bright nor large.",
            properties={"has_art", "bright", "medium"},
        ),
        "dining": Room(
            "dining", "A long table set for a feast. Candles flicker.",
            {"west": "foyer", "north": "workshop"},
            properties={"warm", "medium", "has_candles"},
        ),
        "workshop": Room(
            "workshop", "Tools and wood shavings cover every surface.",
            {"west": "gallery", "south": "dining", "north": "cellar"},
            clue="It is not where people eat or sleep.",
            properties={"messy", "medium", "has_tools"},
        ),
        "cellar": Room(
            "cellar", "A damp underground room. Stone walls seep moisture.",
            {"south": "workshop"},
            properties={"dark", "small", "damp", "underground"},
        ),
        "bedroom_2": Room(
            "bedroom_2", "A small bedroom tucked in the corner.",
            {"west": "workshop"},
            clue="The message is underground.",
            properties={"dark", "small", "cozy"},
        ),
    }
    # Fix exits: workshop needs east exit to bedroom_2
    rooms_2["workshop"].exits["east"] = "bedroom_2"
    rooms_2["bedroom_2"].exits = {"west": "workshop"}
    scenarios.append(TextWorldScenario(
        rooms=rooms_2, start_room="foyer", target_room="cellar",
        required_clues=3,
        clue_rules=[
            {"eliminates_property": "bright", "eliminates_property_2": "large"},
            {"eliminates_has_function": "eat_or_sleep"},
            {"requires_property": "underground"},
        ],
    ))

    # --- Scenario 3: Secret Lab (7 rooms, 2 clues) ---
    rooms_3 = {
        "lobby": Room(
            "lobby", "A modern lobby with glass doors and a reception desk.",
            {"north": "office", "east": "lab_a"},
            properties={"modern", "bright", "has_glass"},
        ),
        "office": Room(
            "office", "Desks with computers. A whiteboard covered in equations.",
            {"south": "lobby", "east": "server_room"},
            clue="The secret lab has no electronic equipment.",
            properties={"modern", "has_electronics", "medium"},
        ),
        "lab_a": Room(
            "lab_a", "Beakers and test tubes on metal counters.",
            {"west": "lobby", "north": "server_room", "east": "storage"},
            properties={"has_equipment", "cold", "sterile"},
        ),
        "server_room": Room(
            "server_room", "Rows of blinking servers. The hum is deafening.",
            {"south": "lab_a", "west": "office"},
            properties={"has_electronics", "loud", "cold"},
        ),
        "storage": Room(
            "storage", "Shelves of old chemicals and dusty manuals.",
            {"west": "lab_a", "north": "basement"},
            clue="The secret lab is the coldest room without machines.",
            properties={"dusty", "medium", "quiet"},
        ),
        "basement": Room(
            "basement", "A cold, silent room carved from bedrock. Perfectly clean.",
            {"south": "storage"},
            properties={"cold", "quiet", "sterile", "no_electronics"},
        ),
        "break_room": Room(
            "break_room", "A cozy room with a coffee machine and couches.",
            {"west": "office"},
            properties={"warm", "cozy", "has_electronics"},
        ),
    }
    # Add exit from office to break_room
    rooms_3["office"].exits["north"] = "server_room"
    rooms_3["office"].exits["east"] = "break_room"
    # Fix: office east should go to break_room, not server_room
    rooms_3["office"].exits = {"south": "lobby", "east": "break_room", "north": "server_room"}
    scenarios.append(TextWorldScenario(
        rooms=rooms_3, start_room="lobby", target_room="basement",
        required_clues=2,
        clue_rules=[
            {"eliminates_property": "has_electronics"},
            {"requires_property": "cold", "eliminates_property": "has_equipment"},
        ],
    ))

    # --- Scenario 4: Pirate Cove (5 rooms, 2 clues) ---
    rooms_4 = {
        "beach": Room(
            "beach", "Golden sand stretches along the shore. Waves crash gently.",
            {"north": "jungle", "east": "dock"},
            properties={"outdoor", "wet", "bright", "sandy"},
        ),
        "jungle": Room(
            "jungle", "Dense vegetation blocks out the sun. Vines hang everywhere.",
            {"south": "beach", "east": "clearing"},
            clue="The treasure rests where wood meets water, but under a roof.",
            properties={"outdoor", "dark", "dense"},
        ),
        "dock": Room(
            "dock", "Wooden planks extend over the water. An old boat is tied up.",
            {"west": "beach", "north": "clearing"},
            properties={"has_wood", "wet", "outdoor"},
        ),
        "clearing": Room(
            "clearing", "A small open area in the jungle. Sunlight filters through.",
            {"south": "dock", "west": "jungle", "north": "shack"},
            clue="The treasure is not exposed to the sky.",
            properties={"outdoor", "bright", "open"},
        ),
        "shack": Room(
            "shack", "A weathered wooden shack with a leaky roof. Puddles on the floor.",
            {"south": "clearing"},
            properties={"has_wood", "wet", "indoor", "has_roof", "small"},
        ),
    }
    scenarios.append(TextWorldScenario(
        rooms=rooms_4, start_room="beach", target_room="shack",
        required_clues=2,
        clue_rules=[
            {"requires_property": "has_wood", "requires_property_2": "wet",
             "requires_property_3": "has_roof"},
            {"eliminates_property": "outdoor"},
        ],
    ))

    return scenarios


SCENARIOS = _build_scenarios()


class TextWorld:
    """Clue Rooms environment. Requires D for clue synthesis."""

    def __init__(self, seed: int = 42, scenario_id: Optional[int] = None):
        self._rng = random.Random(seed)
        if scenario_id is not None:
            self._scenario_idx = scenario_id % len(SCENARIOS)
        else:
            self._scenario_idx = self._rng.randint(0, len(SCENARIOS) - 1)
        self._scenario = SCENARIOS[self._scenario_idx]
        self._current_room: str = ""
        self._visited: Set[str] = set()
        self._clues_shown: Set[str] = set()
        self.t: int = 0

    @property
    def scenario(self) -> TextWorldScenario:
        return self._scenario

    @property
    def room_graph(self) -> Dict[str, Dict[str, str]]:
        return {rid: dict(r.exits) for rid, r in self._scenario.rooms.items()}

    @property
    def room_ids(self) -> List[str]:
        return list(self._scenario.rooms.keys())

    @property
    def room_properties(self) -> Dict[str, Set[str]]:
        return {rid: set(r.properties) for rid, r in self._scenario.rooms.items()}

    @property
    def current_room(self) -> str:
        return self._current_room

    @property
    def agent_pos(self) -> str:
        return self._current_room

    def reset(self) -> dict:
        self._current_room = self._scenario.start_room
        self._visited = {self._current_room}
        self._clues_shown = set()
        self.t = 0
        return self._observe()

    def step(self, action: str) -> Tuple[dict, float, bool]:
        self.t += 1
        room = self._scenario.rooms[self._current_room]
        reward = -0.01

        # "claim" action: agent declares current room is the target
        if action == "claim":
            done = (self._current_room == self._scenario.target_room)
            if done:
                reward += 1.0
            else:
                reward -= 0.2  # wrong claim penalty
            obs = self._observe()
            return obs, reward, done

        # Move if valid exit
        if action in room.exits:
            dest = room.exits[action]
            self._current_room = dest
            if dest not in self._visited:
                self._visited.add(dest)

        # Check for clue in new room
        obs = self._observe()

        # Small reward for discovering a new clue
        if obs.get("clue") is not None:
            reward += 0.05

        done = False  # must explicitly claim
        return obs, reward, done

    def _observe(self) -> dict:
        room = self._scenario.rooms[self._current_room]

        # Clue shown only on first visit
        clue = None
        if room.clue and self._current_room not in self._clues_shown:
            clue = room.clue
            self._clues_shown.add(self._current_room)

        return {
            "room_id": room.room_id,
            "description": room.description,
            "exits": list(room.exits.keys()),
            "clue": clue,
            "visited": set(self._visited),
        }

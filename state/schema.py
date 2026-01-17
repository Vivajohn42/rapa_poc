from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional

class ZA(BaseModel):
    width: int
    height: int
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]                  # visible goal (hidden => (-1,-1))
    obstacles: List[Tuple[int, int]]
    hint: Optional[str] = None                 # NEW

class ZC(BaseModel):
    goal_mode: str
    memory: Dict[str, Any] = Field(default_factory=dict)

class ZD(BaseModel):
    narrative: str
    meaning_tags: List[str] = Field(default_factory=list)
    length_chars: int = 0
    grounding_violations: int = 0

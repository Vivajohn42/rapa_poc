from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional

class ZA(BaseModel):
    width: int
    height: int
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]                  # visible goal (hidden => (-1,-1))
    obstacles: List[Tuple[int, int]]
    hint: Optional[str] = None                 # NEW
    embedding: Optional[List[float]] = None    # Neural belief vector (N0)

class ZC(BaseModel):
    goal_mode: str
    memory: Dict[str, Any] = Field(default_factory=dict)

class ZD(BaseModel):
    narrative: str
    meaning_tags: List[str] = Field(default_factory=list)
    length_chars: int = 0
    grounding_violations: int = 0


class ZPlan(BaseModel):
    """Planning output from Shadow-D (Stufe 8)."""
    recommended_actions: List[str] = Field(default_factory=list)
    plan_horizon: int = 0
    plan_score: float = 0.0
    risk_score: float = 0.0
    alternative_plans: int = 0
    confidence: float = 0.0

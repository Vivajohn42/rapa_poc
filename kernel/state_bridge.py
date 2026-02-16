"""Bridge between rapa_mvp Pydantic models and rapa_os z-dict format.

Pure functions that convert ZA/ZC/ZD/ZPlan to PairState-compatible
z-dicts and back.  This allows the MvpKernel to track state in a
format compatible with rapa_os governance (policy validation, loop gain).
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple


def zA_to_z(zA) -> Dict[str, Any]:
    """Convert ZA (perception) to stream-A z-dict."""
    return {
        "belief": {
            "agent_pos": list(zA.agent_pos),
            "goal_pos": list(zA.goal_pos),
            "width": zA.width,
            "height": zA.height,
            "obstacles": [list(o) for o in zA.obstacles],
            "hint": zA.hint,
        },
    }


def zC_to_z(zC, scored: Optional[List] = None) -> Dict[str, Any]:
    """Convert ZC (valence) to stream-C z-dict."""
    z: Dict[str, Any] = {
        "goal": zC.goal_mode,
    }
    if scored is not None:
        z["valence"] = {
            "action_scores": {a: s for a, s in scored},
        }
    return z


def zD_to_z(zD) -> Dict[str, Any]:
    """Convert ZD (narrative) to stream-D z-dict."""
    return {
        "meaning": {
            "tags": list(zD.meaning_tags),
            "grounding_violations": zD.grounding_violations,
            "length_chars": zD.length_chars,
        },
        "narrative": zD.narrative,
    }


def zPlan_to_b_priors(zPlan) -> Dict[str, Any]:
    """Convert ZPlan to stream-B priors (from C via planning)."""
    return {
        "priors_from_C": {
            "recommended_actions": list(zPlan.recommended_actions),
            "plan_confidence": zPlan.confidence,
            "source": "planner_BC",
        },
    }

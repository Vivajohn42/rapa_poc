"""
Plan-to-C knowledge transfer for Shadow-D (Stufe 8).

Translates Shadow-D's ZPlan output into C-memory updates, specifically
populating tie_break_preference which C already reads (agent_c.py:90-98)
but was never populated until now.
"""

from state.schema import ZC, ZPlan


def deconstruct_plan_to_c(
    zC: ZC,
    zPlan: ZPlan,
    confidence_threshold: float = 0.3,
) -> ZC:
    """
    Translate Shadow-D's plan into C-memory updates.

    Sets:
    - tie_break_preference: first actions from plan's recommended sequence
    - plan_active: True when a plan is in effect
    - plan_confidence: from ZPlan
    - plan_score: from ZPlan

    When confidence is below threshold, clears the plan to avoid
    misguiding C with low-quality plans.

    Args:
        zC: Current C-stream state
        zPlan: Shadow-D's planning output
        confidence_threshold: Minimum confidence to accept plan
    """
    mem = dict(zC.memory) if zC.memory else {}

    if (zPlan.recommended_actions
            and zPlan.confidence >= confidence_threshold):
        # Set tie-break preference to first 3 actions from plan
        mem["tie_break_preference"] = list(zPlan.recommended_actions[:3])
        mem["plan_active"] = True
        mem["plan_confidence"] = zPlan.confidence
        mem["plan_score"] = zPlan.plan_score
    else:
        # Low confidence -> clear plan
        mem.pop("tie_break_preference", None)
        mem["plan_active"] = False
        mem["plan_confidence"] = zPlan.confidence
        mem["plan_score"] = 0.0

    return ZC(goal_mode=zC.goal_mode, memory=mem)

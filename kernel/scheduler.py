"""Deterministic coupling schedule templates.

Port of rapa_os/kernel/scheduler.py.  Provides the same template
rotation per tick, adapted for in-process use (no ZMQ).
"""
from __future__ import annotations

# Schedule templates per regime (FoM = Freiheitsgrade / degrees of freedom)
TEMPL_4FOM = [["AB"]]
TEMPL_6FOM = [["AB", "BC"], ["AB", "AC"]]
TEMPL_8FOM = [["AB", "CD"], ["AB", "BC"], ["AB", "AD"]]


def schedule_for(
    tick_id: int,
    gC: int,
    gD: int,
    priority_coupling: str | None = None,
) -> list[str]:
    """Select coupling schedule for this tick.

    M4: If priority_coupling is set and valid under current gates,
    it replaces the rotation's extra coupling.  This respects the
    'AB + max 1 extra' ABI invariant.
    """
    # Base schedule from deterministic rotation
    if gC == 0 and gD == 0:
        base = TEMPL_4FOM[tick_id % len(TEMPL_4FOM)].copy()
    elif gC == 1 and gD == 0:
        base = TEMPL_6FOM[tick_id % len(TEMPL_6FOM)].copy()
    else:
        base = TEMPL_8FOM[tick_id % len(TEMPL_8FOM)].copy()

    if priority_coupling is None:
        return base

    # Validate: priority must be allowed by current gates
    if "C" in priority_coupling and gC == 0:
        return base
    if "D" in priority_coupling and gD == 0:
        return base
    if priority_coupling == "AB":
        return base  # AB is always there, nothing to override

    # Already scheduled? No change needed
    if priority_coupling in base:
        return base

    # Replace the extra coupling with the priority
    return ["AB", priority_coupling]

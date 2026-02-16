"""ABI constraints for coupling fabric.

Port of rapa_os/common/abi.py.  Four hard rules enforce the DEF
coupling invariants regardless of transport (ZMQ or in-process).
"""
from __future__ import annotations

ALL_COUPLINGS = ["AB", "BC", "CD", "AC", "AD", "BD"]


def enforce_constraints(gC: int, gD: int, schedule: list[str]) -> list[str]:
    """Apply ABI constraints to a proposed coupling schedule.

    Rules:
        1. AB is always active.
        2. gC=0 → remove all couplings containing C (BC, AC).
        3. gD=0 → remove all couplings containing D (CD, AD).
        4. Max 1 additional coupling beyond AB per tick.
    """
    # 1) AB always
    if "AB" not in schedule:
        schedule = ["AB"] + schedule

    # 2) gate C
    if gC == 0:
        schedule = [c for c in schedule if "C" not in c]

    # 3) gate D
    if gD == 0:
        schedule = [c for c in schedule if "D" not in c]

    # 4) AB + max one additional coupling per tick
    extras = [c for c in schedule if c != "AB"]
    if len(extras) > 1:
        schedule = ["AB", extras[0]]

    return schedule

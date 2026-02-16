"""Closure core invariant validation.

Port of rapa_os/kernel/closure_core.py.  Validates that every
KernelDecision satisfies the DEF coupling invariants.
"""
from __future__ import annotations


class ClosureCore:
    """Validates governance invariants on every tick."""

    def validate_decision(self, decision) -> None:
        """Assert the four ABI invariants hold.

        Raises AssertionError if any invariant is violated.
        """
        # 1. AB is always active
        assert "AB" in decision.schedule, (
            f"AB missing from schedule: {decision.schedule}"
        )

        # 2. gC=0 => no C couplings in schedule
        assert not (
            decision.gC == 0
            and any("C" in c for c in decision.schedule)
        ), f"gC=0 but C-coupling in schedule: {decision.schedule}"

        # 3. gD=0 => no D couplings in schedule
        assert not (
            decision.gD == 0
            and any("D" in c for c in decision.schedule)
        ), f"gD=0 but D-coupling in schedule: {decision.schedule}"

        # 4. Max 1 extra coupling beyond AB
        extras = [c for c in decision.schedule if c != "AB"]
        assert len(extras) <= 1, (
            f"More than 1 extra coupling: {extras}"
        )

    def validate_stream_write(
        self, stream: str, payload: dict, target_tier: str
    ) -> None:
        """Security rule: D cannot write directly to L3."""
        if target_tier == "L3" and stream in ("D",):
            raise PermissionError("D cannot write to L3 directly")

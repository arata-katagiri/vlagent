"""Plan validation, execution, and postcondition verification."""

from __future__ import annotations

from typing import Callable

from ..sim.backends import ArmBackend
from .schema import ActionCall, ActionResult, Safety
from .safety import classify  # re-exported: callers import classify from here


def validate_plan(plan: list[ActionCall], state: dict) -> list[str]:
    """Symbolically check the whole plan in order. Empty list means valid.

    Simulates the state forward across steps so that "pick A then place A on tray"
    validates even though A is not held at the time step 1 is checked.
    """
    raise NotImplementedError("Phase 2")


def execute(
    call: ActionCall, backend: ArmBackend, get_state: Callable[[], dict]
) -> ActionResult:
    """Run one step, then verify its postcondition against ground truth."""
    raise NotImplementedError("Phase 2")


__all__ = ["validate_plan", "execute", "classify", "ActionCall", "ActionResult", "Safety"]

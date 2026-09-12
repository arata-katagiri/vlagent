"""Typed action definitions. Units live in parameter names; bounds are explicit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Safety(StrEnum):
    """Safety class of a single action call.

    Never a bare string: the rich badge, the confirmation gate and the run log all
    key off this, so a typo must not be able to silently become "safe".
    """

    SAFE = "safe"
    CAUTION = "caution"
    IRREVERSIBLE = "irreversible"


# Directions accepted by push().
DIRECTIONS = ("left", "right", "forward", "back")

# Bounds, in metres.
MAX_PUSH_DISTANCE_M = 0.3
PLACEMENT_TOLERANCE_M = 0.02


@dataclass
class ActionCall:
    """One step of a plan, as proposed by the LLM."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class ActionResult:
    """Outcome of executing one step, verified against ground truth."""

    ok: bool
    reason: str
    state_after: dict

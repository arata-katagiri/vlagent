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


# Directions accepted by push(), as (dx, dy) unit vectors in world metres.
# The arm sits at the origin looking down +x, so "forward" is away from the base
# and "left" is +y from the user's point of view.
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}
DIRECTIONS = tuple(DIRECTION_VECTORS)

# Bounds, in metres.
MAX_PUSH_DISTANCE_M = 0.3
PLACEMENT_TOLERANCE_M = 0.02
LIFT_HEIGHT_M = 0.15
APPROACH_HEIGHT_M = 0.08

# Arm workspace, measured from the base at the world origin.
MIN_REACH_M = 0.25
MAX_REACH_M = 0.80


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


# name -> (required arg names, one-line description for the LLM).
ACTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "pick": (("object",), "Grasp an object that is free on top and within reach."),
    "place_on": (("target",), "Place the held object on top of a target object."),
    "place_at": (("x_m", "y_m"), "Place the held object at a point on the table."),
    "push": (
        ("object", "direction", "distance_m"),
        "Push an object along the table in one of four directions.",
    ),
    "home": ((), "Return the arm to its rest pose."),
    "ask_user": (("question",), "Ask the user a question when a reference is ambiguous."),
}
ACTION_NAMES = tuple(ACTIONS)

# Exact argument keys and units, as the planner must emit them. Generated into
# the system prompt so the model is never guessing: with `args` typed as a bare
# object and no catalogue in the prompt, models omit the keys entirely.
SIGNATURES: dict[str, str] = {
    "pick": 'object: str  (e.g. {"object": "red_block"})',
    "place_on": 'target: str  (e.g. {"target": "tray"})',
    "place_at": 'x_m: float, y_m: float  — metres in world coordinates',
    "push": (
        'object: str, direction: one of '
        + "|".join(DIRECTIONS)
        + f", distance_m: float in (0, {MAX_PUSH_DISTANCE_M}]"
    ),
    "home": "no arguments: {}",
    "ask_user": 'question: str  (e.g. {"question": "which cup?"})',
}


def describe_actions() -> str:
    """The action catalogue for the system prompt, from a single source."""
    lines = []
    for name, (required, description) in ACTIONS.items():
        lines.append(f"  {name}({SIGNATURES[name]})")
        lines.append(f"      {description}")
        if required:
            lines.append(f"      required keys in args: {', '.join(required)}")
    return "\n".join(lines)


def plan_tool_schema() -> list[dict]:
    """The tool definitions handed to the LLM.

    `name` is pinned to an enum of the six real actions. Without it, models invent
    plausible-sounding steps -- gpt-4o proposed a `locate_red_block` on the first
    try during Phase 0. The enum plus the symbolic validator keeps a hallucinated
    plan out of the executor.
    """
    step = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(ACTION_NAMES)},
            "args": {
                "type": "object",
                "description": (
                    "Arguments for this action, using the exact keys listed for it "
                    "in the system prompt. Must not be empty unless the action is home."
                ),
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence on why this step is needed.",
            },
        },
        "required": ["name", "args", "rationale"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "propose_plan",
                "description": "Propose the complete plan of physical actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "array", "items": step},
                        "summary": {
                            "type": "string",
                            "description": "One sentence describing the plan to the user.",
                        },
                    },
                    "required": ["steps", "summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user to resolve a genuinely ambiguous reference.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        },
    ]


__all__ = [
    "Safety", "ActionCall", "ActionResult", "ACTIONS", "ACTION_NAMES",
    "DIRECTIONS", "DIRECTION_VECTORS", "MAX_PUSH_DISTANCE_M",
    "PLACEMENT_TOLERANCE_M", "LIFT_HEIGHT_M", "APPROACH_HEIGHT_M",
    "MIN_REACH_M", "MAX_REACH_M", "plan_tool_schema", "SIGNATURES", "describe_actions",
]

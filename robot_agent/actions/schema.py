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

    Two things here are load-bearing.

    `name` is pinned to an enum of the six real actions. Without it, models
    invent plausible-sounding steps -- gpt-4o proposed a `locate_red_block` on
    the first try.

    The arguments are **flat, named properties on the step**, not a nested
    free-form `args` object. With a nested object, models routinely emitted
    `args: {}` and the plan was rejected for missing arguments even with the
    action catalogue in the system prompt. Named properties are validated by the
    provider and are far more reliably filled in.
    """
    step = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": list(ACTION_NAMES) + _learned_names(),
                "description": "Which action or learned skill to run.",
            },
            "object": {
                "type": "string",
                "description": "Object to act on. Required by pick and push.",
            },
            "target": {
                "type": "string",
                "description": "Object to place onto. Required by place_on.",
            },
            "x_m": {"type": "number", "description": "Required by place_at, in metres."},
            "y_m": {"type": "number", "description": "Required by place_at, in metres."},
            "direction": {
                "type": "string",
                "enum": list(DIRECTIONS),
                "description": "Required by push.",
            },
            "distance_m": {
                "type": "number",
                "description": f"Required by push, in metres, 0 to {MAX_PUSH_DISTANCE_M}.",
            },
            "question": {"type": "string", "description": "Required by ask_user."},
            "angle_deg": {"type": "number", "description": "Degrees, for learned skills that turn."},
            "height_m": {"type": "number", "description": "Metres, for learned skills that lift or tap."},
            "count": {"type": "integer", "description": "Repetitions, for learned skills."},
            "speed": {"type": "number", "description": "0.1 (slow) to 2.0 (fast), for learned skills."},
            "rationale": {
                "type": "string",
                "description": "One short sentence on why this step is needed.",
            },
        },
        "required": ["name", "rationale"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "propose_plan",
                "description": (
                    "Propose the complete plan of physical actions. Use this only "
                    "when the user wants something moved."
                ),
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
                "name": "answer",
                "description": (
                    "Answer the user directly, without moving anything. Use this for "
                    "questions about the scene ('what is on the table?', 'is anything "
                    "fragile?') and to explain what you cannot do."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The answer, in plain English."}
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": (
                    "Ask the user to resolve a genuinely ambiguous reference. Only use "
                    "this when their answer would change which action you take."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        },
        define_skill_tool(),
    ]


def _learned_names() -> list[str]:
    from ..skills.registry import REGISTRY  # noqa: PLC0415

    return REGISTRY.names()


def define_skill_tool() -> dict:
    """The tool that lets the planner write a new skill (see skills/prompt.py)."""
    from ..skills.registry import EFFECT_KINDS, SKILL_ARG_KEYS  # noqa: PLC0415

    return {
        "type": "function",
        "function": {
            "name": "define_skill",
            "description": (
                "Write a new skill when no action or learned skill can do what was "
                "asked. It is rehearsed in the simulator and measured before it is "
                "kept; afterwards you will be asked to plan again using it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case verb, e.g. rotate, knock_over, sweep"},
                    "doc": {"type": "string", "description": "One line: what the skill does, in general."},
                    "effect": {"type": "string", "description": "The outcome, one sentence, for the user."},
                    "effect_kind": {
                        "type": "string",
                        "enum": list(EFFECT_KINDS),
                        "description": (
                            "What the simulator should measure to confirm the skill worked. "
                            "Use 'other' only if none fits; then only your check() can judge it."
                        ),
                    },
                    "effect_of": {
                        "type": "string",
                        "enum": list(SKILL_ARG_KEYS),
                        "description": "Which argument names the object the effect applies to (usually 'object').",
                    },
                    "effect_value": {
                        "type": "string",
                        "description": (
                            "For moves_at_least / rotates_by: a number, or the name of the argument "
                            "that holds it (e.g. 'angle_deg'). Empty otherwise."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SKILL_ARG_KEYS)},
                        "description": "Argument names run() accepts.",
                    },
                    "example_args_json": {
                        "type": "string",
                        "description": 'JSON object of argument values for the current request, e.g. {"object": "red_block", "angle_deg": 90}',
                    },
                    "code": {
                        "type": "string",
                        "description": "Python defining run(arm, **args) and optionally check(before, after, **args).",
                    },
                    "summary": {"type": "string", "description": "One sentence for the user."},
                    "derived_from": {
                        "type": "string",
                        "description": "Your opinion: the existing skill this one is built from or adapted from, if any.",
                    },
                    "similar_to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Your opinion: existing skills this one resembles. The measured motion is stored separately.",
                    },
                },
                "required": ["name", "doc", "effect", "effect_kind", "effect_of", "args",
                             "example_args_json", "code", "summary"],
            },
        },
    }


# Step fields that carry action arguments, as opposed to bookkeeping.
ARG_KEYS = ("object", "target", "x_m", "y_m", "direction", "distance_m", "question",
            "angle_deg", "height_m", "count", "speed")


__all__ = [
    "Safety", "ActionCall", "ActionResult", "ACTIONS", "ACTION_NAMES",
    "DIRECTIONS", "DIRECTION_VECTORS", "MAX_PUSH_DISTANCE_M",
    "PLACEMENT_TOLERANCE_M", "LIFT_HEIGHT_M", "APPROACH_HEIGHT_M",
    "MIN_REACH_M", "MAX_REACH_M", "plan_tool_schema", "SIGNATURES", "describe_actions",
    "ARG_KEYS",
]

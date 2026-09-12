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

# Arm workspace, as a radius from an arm's own base. In the single-arm scene the
# base is the world origin; in the two-arm scene each arm has its own base and
# safety.reachable() is given that base.
MIN_REACH_M = 0.25
MAX_REACH_M = 0.80

# Which actions are performed *by an arm*, and therefore need an `arm` argument
# once the scene has more than one. ask_user is not one of them.
ARM_ACTIONS = ("pick", "place_on", "place_at", "push", "home")

# Names of the arms in the loaded scene, set once at startup by app.main.
# Empty means a single-arm scene, and nothing below changes shape.
_ARM_NAMES: tuple[str, ...] = ()


def set_arms(names) -> None:
    """Declare the arms in the loaded scene. Called once, before the first plan."""
    global _ARM_NAMES
    _ARM_NAMES = tuple(names)


def arms() -> tuple[str, ...]:
    return _ARM_NAMES


def multi_arm() -> bool:
    return len(_ARM_NAMES) > 1


def required_args(name: str) -> tuple[str, ...]:
    """Required argument keys for an action, including `arm` in a multi-arm scene."""
    base = ACTIONS[name][0] if name in ACTIONS else ()
    if multi_arm() and name in ARM_ACTIONS:
        return base + ("arm",)
    return base


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
    arm_note = f', arm: one of {"|".join(_ARM_NAMES)}' if multi_arm() else ""
    for name, (_, description) in ACTIONS.items():
        sig = SIGNATURES[name]
        if multi_arm() and name in ARM_ACTIONS:
            sig = "arm: str" if sig == "no arguments: {}" else sig + arm_note
        lines.append(f"  {name}({sig})")
        lines.append(f"      {description}")
        required = required_args(name)
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
    if multi_arm():
        # Pinned to an enum for the same reason `name` is: an invented arm name
        # would reach the executor and fail there instead of at the schema.
        #
        # And listed in `required`: with `arm` merely described, gpt-4o proposed
        # a correct five-step hand-off three times in a row and omitted the arm
        # on every step, even after the validator handed back "missing required
        # argument: arm" as feedback. Provider-side required is what actually
        # makes the field appear. ask_user steps get a spurious arm, which the
        # executor ignores.
        step["required"] = list(step["required"]) + ["arm"]
        step["properties"]["arm"] = {
            "type": "string",
            "enum": list(_ARM_NAMES),
            "description": (
                "Which arm performs this step. Required by "
                + ", ".join(ARM_ACTIONS)
                + ". Each arm can only reach part of the table."
            ),
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
        define_rule_tool(),
    ]


def _learned_names() -> list[str]:
    from ..skills.registry import REGISTRY  # noqa: PLC0415

    return REGISTRY.names()


def define_rule_tool() -> dict:
    """The tool that lets the planner propose a safety rule (see skills/rules.py)."""
    return {
        "type": "function",
        "function": {
            "name": "define_rule",
            "description": (
                "Propose a new safety rule when the user states a policy about what is "
                "dangerous or must never happen. It becomes deterministic code that runs on "
                "every plan and every learned skill; it can only add refusals or raise levels. "
                "The user sees what it would block right now and decides whether to keep it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case, e.g. corrosive_separation"},
                    "doc": {"type": "string", "description": "One line: the policy in plain English."},
                    "tags_json": {
                        "type": "string",
                        "description": 'JSON object of tags to attach to objects, e.g. {"cup": ["corrosive"], "glass": ["water"]}. Empty if none.',
                    },
                    "code": {
                        "type": "string",
                        "description": "Python defining check(before, after, action, args) and optionally level(before, after, action, args).",
                    },
                    "summary": {"type": "string", "description": "One sentence for the user."},
                },
                "required": ["name", "doc", "tags_json", "code", "summary"],
            },
        },
    }


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
            "angle_deg", "height_m", "count", "speed", "arm")


__all__ = [
    "Safety", "ActionCall", "ActionResult", "ACTIONS", "ACTION_NAMES",
    "DIRECTIONS", "DIRECTION_VECTORS", "MAX_PUSH_DISTANCE_M",
    "PLACEMENT_TOLERANCE_M", "LIFT_HEIGHT_M", "APPROACH_HEIGHT_M",
    "MIN_REACH_M", "MAX_REACH_M", "plan_tool_schema", "SIGNATURES", "describe_actions",
    "ARM_ACTIONS", "set_arms", "arms", "multi_arm", "required_args",
    "ARG_KEYS",
]

"""OpenAI-compatible LLM client plus an offline mock.

Provider comes from the environment (OPENAI_BASE_URL / OPENAI_API_KEY /
OPENAI_MODEL) so we can switch between OpenRouter, a local endpoint, or any
OpenAI-compatible service without touching code. Never hardcode or print a key.

Both clients answer with the same two-case result: a Plan or a Question. The
planner does not care which client produced it, which is what lets the whole
loop run offline with --mock-llm.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..actions.schema import ActionCall, describe_actions, plan_tool_schema

TIMEOUT_S = 30.0
MAX_RETRIES = 1

_RULES = """\
You control a real robot arm above a table. Act only through the provided tools.

Actions, with the exact argument keys you must use:
{actions}

Rules:
- Never invent objects or action names. Use exactly the names in the scene state.
- Always fill in every required argument key. An empty args object is never valid
  except for home.
- Resolve every reference ("the cup", "the red one") against the scene state.
- To move an object onto something, pick it first, then place_on. The gripper
  holds at most one object at a time.
- There is no bin, no floor target, and no way to make an object disappear. The
  only way to get something off the table is place_at at a point beyond the table
  bounds, and the safety checker will class that as irreversible and put it to the
  user. Propose it if that is what was asked, and let the user decide.
- Never propose a plan whose net effect is nothing, such as picking an object up
  and putting it back where it already is. If the request cannot be achieved with
  these actions, call ask_user and say what you cannot do.
- Safety is enforced outside of you by deterministic code. You cannot approve,
  downgrade, or argue past it. Propose the action you believe is right; if it is
  unsafe, the system will stop it and tell you why.
"""

SYSTEM_PROMPT = _RULES.format(actions=describe_actions())


@dataclass
class Plan:
    """A proposed sequence of physical actions."""

    steps: list[ActionCall]
    summary: str = ""


@dataclass
class Question:
    """The model needs the user to disambiguate before it can plan."""

    text: str


@dataclass
class Usage:
    """Token accounting, so a run log can show what the demo cost."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage) -> None:
        self.calls += 1
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def describe_state(state: dict) -> str:
    """Serialize the scene compactly for the prompt.

    Deliberately terse: positions to the centimetre, and only the facts that
    change what the agent may do.
    """
    t = state["table_bounds_m"]
    lines = [
        f"table: x {t['x_min']:.2f}..{t['x_max']:.2f} m, "
        f"y {t['y_min']:.2f}..{t['y_max']:.2f} m, top at z {t['top_z']:.2f} m",
        f"gripper: at {_xyz(state['gripper']['position_m'])}, "
        f"holding {state['gripper']['holding'] or 'nothing'}",
        "objects:",
    ]
    for name, obj in state["objects"].items():
        tags = []
        if obj["fragile"]:
            tags.append("FRAGILE")
        if not obj["graspable"]:
            tags.append("fixed, cannot be picked up")
        if obj["on_top_of"]:
            tags.append(f"on top of {obj['on_top_of']}")
        if obj["supporting"]:
            tags.append("supporting " + ", ".join(obj["supporting"]))
        if obj["near_table_edge"]:
            tags.append("near the table edge")
        if obj["held"]:
            tags.append("currently held")
        if not obj["on_table"] and not obj["held"]:
            tags.append("off the table")
        suffix = f" [{'; '.join(tags)}]" if tags else ""
        lines.append(f"  {name} ({obj['color']}) at {_xyz(obj['position_m'])}{suffix}")
    return "\n".join(lines)


def _xyz(p) -> str:
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"


def _steps_from_args(raw: dict) -> list[ActionCall]:
    return [
        ActionCall(
            name=str(s.get("name", "")),
            args=dict(s.get("args") or {}),
            rationale=str(s.get("rationale", "")),
        )
        for s in raw.get("steps", [])
    ]


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat completions API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # noqa: PLC0415

        self.model = model or os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini")
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            timeout=TIMEOUT_S,
            max_retries=MAX_RETRIES,
        )
        self.usage = Usage()

    def propose(
        self, command: str, state: dict, feedback: list[str] | None = None
    ) -> Plan | Question:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Scene:\n{describe_state(state)}\n\nCommand: {command}",
            },
        ]
        if feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That plan was rejected by the safety checker:\n"
                        + "\n".join(f"- {p}" for p in feedback)
                        + "\nPropose a corrected plan that avoids these problems."
                    ),
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=plan_tool_schema(),
            tool_choice="required",
            max_tokens=800,
        )
        self.usage.add(getattr(response, "usage", None))

        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            return Question(text=message.content or "I could not form a plan.")

        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return Question(text="I produced an unreadable plan. Could you rephrase?")

        if call.function.name == "ask_user":
            return Question(text=str(args.get("question", "Could you clarify?")))
        return Plan(steps=_steps_from_args(args), summary=str(args.get("summary", "")))


@dataclass
class MockLLM:
    """Maps the scripted demo commands to fixed plans, so --mock-llm is offline.

    Keyword matching, not language understanding: this exists so the loop, the
    safety gate and the UI can be exercised with no network. It answers replan
    requests with a genuinely different plan, which is what makes the refusal
    branch of the demo work.
    """

    usage: Usage = field(default_factory=Usage)

    def propose(
        self, command: str, state: dict, feedback: list[str] | None = None
    ) -> Plan | Question:
        self.usage.calls += 1
        text = command.lower()

        def step(name, rationale, **args):
            return ActionCall(name=name, args=args, rationale=rationale)

        # After a refusal or a validation failure, offer the safe alternative.
        if feedback:
            if "glass" in text and "push" in text:
                return Plan(
                    steps=[
                        step("pick", "lifting is reversible, unlike pushing it off the table", object="glass"),
                        step("place_on", "the tray is the designated safe place", target="tray"),
                    ],
                    summary="Move the glass to the tray instead of pushing it.",
                )
            if "glass" in text:
                return Plan(
                    steps=[
                        step("pick", "the blue block still needs somewhere to go", object="blue_block"),
                        step("place_on", "the tray is flat and stable, unlike the glass", target="tray"),
                    ],
                    summary="Put the blue block on the tray instead of the glass.",
                )
            # A placement that missed: put the named object back where it belongs.
            joined = " ".join(feedback).lower()
            for name in state["objects"]:
                if name in joined and state["objects"][name]["graspable"]:
                    return Plan(
                        steps=[
                            step("pick", f"{name} did not land where it should", object=name),
                            step("place_on", "placing it properly this time", target="tray"),
                        ],
                        summary=f"Re-place {name} on the tray after the failed attempt.",
                    )
            return Question(text="I could not find a safe way to do that. What would you like instead?")

        if "red" in text and "green" in text:
            return Plan(
                steps=[
                    step("pick", "the red block is clear and within reach", object="red_block"),
                    step("place_on", "the tray is the designated safe place", target="tray"),
                    step("pick", "the green block is next", object="green_block"),
                    step("place_on", "keeping both blocks together on the tray", target="tray"),
                ],
                summary="Clear the red and green blocks onto the tray.",
            )

        if "blue" in text and "glass" in text:
            return Plan(
                steps=[
                    step("pick", "the blue block is clear and within reach", object="blue_block"),
                    step("place_on", "the user asked for it to go on the glass", target="glass"),
                ],
                summary="Put the blue block on the glass.",
            )

        if "push" in text and "glass" in text:
            direction = "left" if "left" in text else "right"
            return Plan(
                steps=[
                    step(
                        "push",
                        "the user asked for the glass to be moved out of the way",
                        object="glass", direction=direction, distance_m=0.3,
                    )
                ],
                summary=f"Push the glass {direction} out of the way.",
            )

        if "cup" in text and ("safe" in text or "away" in text):
            return Plan(
                steps=[
                    step("pick", "the cup is clear and within reach", object="cup"),
                    step("place_on", "the tray is the designated safe place", target="tray"),
                ],
                summary="Put the cup on the tray.",
            )

        if "home" in text or "rest" in text:
            return Plan(steps=[step("home", "returning the arm to its rest pose")],
                        summary="Send the arm home.")

        return Question(
            text="I only know the scripted demo commands in mock mode. "
                 "Try: clear the red and green blocks onto the tray."
        )


def build_llm(mock: bool) -> LLMClient | MockLLM:
    """Pick a client. Falls back to the mock if credentials are missing."""
    if mock:
        return MockLLM()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; copy .env.example to .env or pass --mock-llm")
    return LLMClient()


__all__ = [
    "LLMClient", "MockLLM", "Plan", "Question", "Usage",
    "describe_state", "build_llm", "SYSTEM_PROMPT",
]

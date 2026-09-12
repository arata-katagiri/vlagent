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

from ..actions.safety import STACKABLE_TARGETS, policy_enabled
from ..actions.schema import ARG_KEYS, ActionCall, describe_actions, plan_tool_schema

TIMEOUT_S = 30.0
MAX_RETRIES = 1
# Conversation memory handed to the model: the most recent turns, each clipped.
HISTORY_TURNS = 30
HISTORY_CHARS = 400

_RULES = """\
You control a real robot arm above a table. Act only through the provided tools.

Actions, with the exact argument keys you must use:
{actions}

Choosing a tool:
- A question about the scene ("what is on the table?", "is anything fragile?",
  "is there something yellow?") is answered with the answer tool. Answer it from
  the scene state you were given. Do not turn a question back into a question.
- Something you cannot do is explained with the answer tool. Objects may only be
  set down on the listed place_on targets; that list is about destinations, not
  about which objects can be carried or pushed. Note the direction: an object
  that is not a valid place_on target can still be picked up, pushed, or put
  somewhere else. Any object on the table can be pushed.
- When a step failed and you are replanning, change something: a different
  spot (place_at with coordinates spread well apart), a different order, or a
  different surface. Do not repeat the step that just failed unchanged.
- Do not refuse on your own guess about whether something will fit or balance.
  Propose the plan; the safety checker measures it and will tell you if it fails.
  Only refuse outright for what the scene state plainly rules out.
- The conversation so far (earlier commands and what happened to them) comes
  before the current command. Use it to resolve follow-ups such as "now the
  green one", "do that again", "put it back", or "why not?". A question about
  what you did or why something failed is answered with the answer tool,
  quoting the recorded reason.
- ask_user is only for a genuine ambiguity where their reply changes which action
  you would take.
- propose_plan is only for actually moving something.

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
  user. Propose it if that is what was asked, and let the user decide. The far
  edge of the table is beyond the arm's reach, so use a point 5-8 cm past the
  nearer side edge (y just beyond y_min or y_max) at the object's current x.
- Never propose a plan whose net effect is nothing, such as picking an object up
  and putting it back where it already is. If the request cannot be achieved with
  these actions, call ask_user and say what you cannot do.
- Safety is enforced outside of you by deterministic code. You cannot approve,
  downgrade, or argue past it. Propose the action you believe is right; if it is
  unsafe, the system will stop it and tell you why.
"""

_UNRESTRICTED = """\
The safety policy is disabled for this session. Nothing you propose will be
refused for being unwise, unstable or irreversible, and you will not be asked to
confirm anything. Preconditions that describe what is physically impossible still
apply: an object must exist, the gripper holds one thing at a time, and a target
must be within reach.

Take the user at their word. If they ask for something precarious, plan it and
let the physics decide. Do not lecture, do not hedge, and do not refuse on your
own judgement -- if it can be expressed with these actions, propose it.
"""

SYSTEM_PROMPT = _RULES.format(actions=describe_actions())


# Extra rule paragraphs a scenario adds to the system prompt (see sim/lab_scene.py).
SCENARIO_RULES: list[str] = []

# Added only when the loaded scene has more than one arm.
_ARM_RULES = """

There are two robot arms at this table, `a` and `b`, facing each other. Every
physical step must say which arm performs it.

- Each arm reaches about 0.25-0.80 m from its own base, so neither can reach the
  whole table. The scene lists, for each object, which arms can reach it.
- Choose the arm that can reach the object. If only one arm can, that is the arm.
- To move something from one arm's side to the other, hand it over: the first arm
  places it on the tray in the middle, which both arms can reach, then the second
  arm picks it up from there. Plan both halves as ordinary steps.
- Each arm has its own gripper and holds at most one object. One arm holding
  something does not stop the other from working.
- The arms share one workspace and do not avoid each other. Send an arm `home`
  before the other reaches into the middle.
"""


def system_prompt() -> str:
    """Rules + live catalogue (built-ins and learned skills) + how to write a skill,
    for the current policy setting."""
    from ..skills.prompt import AUTHORING_GUIDE, RULE_GUIDE  # noqa: PLC0415
    from ..skills.registry import REGISTRY  # noqa: PLC0415
    from ..skills.rules import RULES  # noqa: PLC0415

    actions = describe_actions()
    learned = REGISTRY.describe()
    if learned:
        actions += "\n\n" + learned
    from ..actions.schema import multi_arm  # noqa: PLC0415

    base = _RULES.format(actions=actions)
    if multi_arm():
        base += _ARM_RULES
    policy = RULES.describe()
    base += "".join(SCENARIO_RULES) + ("\n" + policy + "\n" if policy else "") + AUTHORING_GUIDE + RULE_GUIDE
    if policy_enabled():
        return base
    return base + "\n" + _UNRESTRICTED


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
class Answer:
    """A direct reply, with nothing to execute.

    Questions about the scene are a normal thing to ask a home assistant. Without
    this, every such request came back as a clarification, so "tell me what is on
    the table" was answered with "do you want a full list?".
    """

    text: str


@dataclass
class SkillProposal:
    """The model wants to write a new skill rather than plan with what exists."""

    name: str
    doc: str
    effect: str
    args: list[str]
    code: str
    example_args: dict
    summary: str = ""
    effect_kind: str = "other"
    effect_of: str = "object"
    effect_value: str = ""
    derived_from: str = ""
    similar_to: list[str] = field(default_factory=list)


@dataclass
class RuleProposal:
    """The model wants to add a safety rule rather than plan."""

    name: str
    doc: str
    code: str
    tags: dict
    summary: str = ""


def _rule_from_args(raw: dict) -> RuleProposal:
    try:
        tags = json.loads(raw.get("tags_json") or "{}")
        if not isinstance(tags, dict):
            tags = {}
    except json.JSONDecodeError:
        tags = {}
    return RuleProposal(
        name=str(raw.get("name", "")).strip().lower().replace(" ", "_"),
        doc=str(raw.get("doc", "")),
        code=str(raw.get("code", "")),
        tags={str(k): [str(t) for t in (v if isinstance(v, list) else [v])] for k, v in tags.items()},
        summary=str(raw.get("summary", "")),
    )


def _skill_from_args(raw: dict) -> SkillProposal:
    try:
        example = json.loads(raw.get("example_args_json") or "{}")
        if not isinstance(example, dict):
            example = {}
    except json.JSONDecodeError:
        example = {}
    return SkillProposal(
        name=str(raw.get("name", "")).strip().lower().replace(" ", "_"),
        doc=str(raw.get("doc", "")),
        effect=str(raw.get("effect", "")),
        args=[str(a) for a in (raw.get("args") or [])],
        code=str(raw.get("code", "")),
        example_args=example,
        summary=str(raw.get("summary", "")),
        effect_kind=str(raw.get("effect_kind") or "other"),
        effect_of=str(raw.get("effect_of") or "object"),
        effect_value=str(raw.get("effect_value") or ""),
        derived_from=str(raw.get("derived_from") or ""),
        similar_to=[str(x) for x in (raw.get("similar_to") or []) if x],
    )


def _feedback_text(feedback: list[str]) -> str:
    body = "\n".join(f"- {p}" for p in feedback)
    if any(p.startswith("rehearsal") for p in feedback):
        return (
            "The skill you defined was rehearsed in the simulator and did not pass:\n"
            + body
            + "\nRevise the code and call define_skill again with the same name, "
            "fixing these problems. Do not give up on the skill unless it is impossible."
        )
    if any(p.startswith("the skill '") for p in feedback):
        return body + "\nNow propose the plan for the request using it."
    return (
        "That plan was rejected by the safety checker:\n"
        + body
        + "\nPropose a corrected plan that avoids these problems."
    )


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
    from ..actions.safety import base_of, reachable  # noqa: PLC0415

    t = state["table_bounds_m"]
    arms = state.get("arms")
    lines = [
        f"table: x {t['x_min']:.2f}..{t['x_max']:.2f} m, "
        f"y {t['y_min']:.2f}..{t['y_max']:.2f} m, top at z {t['top_z']:.2f} m",
    ]
    if arms:
        for key, arm in arms.items():
            bx, by = arm["base_m"][:2]
            lines.append(
                f"arm {key}: base at ({bx:.2f}, {by:.2f}) m, gripper at "
                f"{_xyz(arm['position_m'])}, holding {arm['holding'] or 'nothing'}"
            )
    else:
        lines.append(
            f"gripper: at {_xyz(state['gripper']['position_m'])}, "
            f"holding {state['gripper']['holding'] or 'nothing'}"
        )
    lines.append("objects:")
    for name, obj in state["objects"].items():
        tags = []
        if obj["fragile"]:
            tags.append("FRAGILE")
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
        if arms:
            x, y = obj["position_m"][0], obj["position_m"][1]
            who = [k for k in arms if reachable(x, y, base_of(state, k))]
            tags.append(
                "reachable by " + (" and ".join(f"arm {k}" for k in who) if who
                                   else "neither arm")
            )
        suffix = f" [{'; '.join(tags)}]" if tags else ""
        lines.append(f"  {name} ({obj['color']}) at {_xyz(obj['position_m'])}{suffix}")

    # A standalone list of destinations, not a tag on each object. As a per-object
    # tag ("not a valid place_on target") the model read it as a property of the
    # object itself and refused to put the cup *onto* anything.
    # Capabilities are listed as names, not tagged onto each object's line. As a
    # per-object tag the model read them as properties of the object in whatever
    # role the sentence put it in: "tray [fixed, cannot be picked up]" became a
    # reason it could not place something *onto* the tray.
    lines.append(
        "can be picked up: "
        + ", ".join(n for n, o in state["objects"].items() if o["graspable"])
    )
    if policy_enabled():
        lines.append(
            "can be placed onto (the only valid place_on targets): "
            + ", ".join(n for n in state["objects"] if n in STACKABLE_TARGETS)
        )
    else:
        lines.append(
            "can be placed onto: any object here (the usual restriction to flat, "
            "stable surfaces is disabled for this session)"
        )
    lines.append(
        "these two lists are independent: an object may be liftable but not a "
        "target, or a target but not liftable."
    )
    return "\n".join(lines)


def _xyz(p) -> str:
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"


def history_messages(transcript: list[dict] | None) -> list[dict]:
    """Prior turns as chat messages, oldest first.

    The current command is always the last user entry when propose() runs
    (main logs it before planning), so a trailing user entry is dropped here
    and delivered once, with the scene, in the final user message. Consecutive
    entries from the same role are merged into one message.
    """
    if not transcript:
        return []
    turns = list(transcript[-HISTORY_TURNS:])
    if turns and turns[-1]["role"] == "user":
        turns.pop()
    merged: list[dict] = []
    for t in turns:
        text = str(t["content"])[:HISTORY_CHARS]
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += "\n" + text
        else:
            merged.append({"role": t["role"], "content": text})
    return merged


def _step_args(step: dict) -> dict:
    """Collect a step's arguments from flat fields, or a nested args object.

    The schema asks for flat named properties because models fill those in far
    more reliably, but a model may still nest them under `args`; accept both.
    """
    args = {k: step[k] for k in ARG_KEYS if step.get(k) is not None}
    nested = step.get("args")
    if isinstance(nested, dict):
        for key, value in nested.items():
            args.setdefault(key, value)
    return args


def _steps_from_args(raw: dict) -> list[ActionCall]:
    return [
        ActionCall(
            name=str(s.get("name", "")),
            args=_step_args(s),
            rationale=str(s.get("rationale", "")),
        )
        for s in raw.get("steps", [])
        if isinstance(s, dict)
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
        # Conversation memory; main.py points this at the run log's transcript.
        self.transcript: list[dict] | None = None

    def propose(
        self, command: str, state: dict, feedback: list[str] | None = None
    ) -> Plan | Question | Answer:
        messages = [
            {"role": "system", "content": system_prompt()},
            *history_messages(self.transcript),
            {
                "role": "user",
                "content": f"Scene:\n{describe_state(state)}\n\nCommand: {command}",
            },
        ]
        if feedback:
            messages.append(
                {
                    "role": "user",
                    "content": _feedback_text(feedback),
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=plan_tool_schema(),
            tool_choice="required",
            max_tokens=2000,
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
        if call.function.name == "answer":
            return Answer(text=str(args.get("text", "")))
        if call.function.name == "define_skill":
            return _skill_from_args(args)
        if call.function.name == "define_rule":
            return _rule_from_args(args)
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
    transcript: list[dict] | None = None  # accepted for parity; the mock ignores it

    def propose(
        self, command: str, state: dict, feedback: list[str] | None = None
    ) -> Plan | Question | Answer:
        self.usage.calls += 1
        text = command.lower()

        if not feedback and any(
            q in text for q in ("what is on", "what's on", "list", "is there", "anything")
        ):
            names = ", ".join(
                n for n, o in state["objects"].items() if o["on_table"] and o["graspable"]
            )
            return Answer(text=f"On the table: {names}, plus the tray they can go on.")

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
    "LLMClient", "MockLLM", "Plan", "Question", "Answer", "Usage",
    "describe_state", "build_llm", "SYSTEM_PROMPT", "system_prompt", "history_messages",
]

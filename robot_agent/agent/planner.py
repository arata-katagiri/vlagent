"""Command -> plan, clarification, and replanning.

The LLM may only answer by calling propose_plan(steps, summary) or
ask_user(question). Validation failures are fed back for at most MAX_REPLANS
retries; execution failures trigger at most MAX_REPLAN_ATTEMPTS re-observations.
"""

from __future__ import annotations

MAX_REPLANS = 2
MAX_REPLAN_ATTEMPTS = 2

SYSTEM_PROMPT = """\
You control a real robot arm. Act only through the provided tools.
Never invent objects: resolve every reference using the scene state you are given.
Ask instead of guessing when an ambiguity actually changes what you would do.
Safety is enforced outside of you; you cannot approve or downgrade an unsafe action.
"""


def plan(command: str, state: dict, llm) -> list:
    """Turn a natural-language command into a validated list of ActionCall."""
    raise NotImplementedError("Phase 3")

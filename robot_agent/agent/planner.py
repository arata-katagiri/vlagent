"""Command -> plan, clarification, and replanning.

The LLM may only answer by calling propose_plan(steps, summary) or
ask_user(question). A proposal is checked symbolically before the user ever
sees it; validation failures are handed back to the model as feedback, at most
MAX_REPLANS times. The model never gets to overrule the checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..actions.executor import validate_plan
from ..actions.schema import ActionCall
from .llm import Plan, Question

MAX_REPLANS = 2
MAX_REPLAN_ATTEMPTS = 2


@dataclass
class Outcome:
    """What the planner produced for one user command."""

    steps: list[ActionCall] = field(default_factory=list)
    summary: str = ""
    question: str | None = None
    problems: list[str] = field(default_factory=list)
    attempts: int = 0
    rejected: list[tuple[list[ActionCall], list[str]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and not self.problems and self.question is None


def plan(
    command: str,
    state: dict,
    llm,
    max_replans: int = MAX_REPLANS,
    feedback: list[str] | None = None,
) -> Outcome:
    """Turn a natural-language command into a validated list of ActionCall.

    `feedback` seeds the first proposal with what already went wrong, so a
    recovery or a refusal reaches the model through the same channel a
    validation failure does, rather than being smuggled into the command text.
    """
    outcome = Outcome()

    for attempt in range(1, max_replans + 2):
        outcome.attempts = attempt
        proposal = llm.propose(command, state, feedback)

        if isinstance(proposal, Question):
            outcome.question = proposal.text
            return outcome

        if not isinstance(proposal, Plan) or not proposal.steps:
            outcome.problems = ["the planner returned an empty plan"]
            return outcome

        problems = validate_plan(proposal.steps, state)
        if not problems:
            outcome.steps = proposal.steps
            outcome.summary = proposal.summary
            outcome.problems = []
            return outcome

        outcome.rejected.append((proposal.steps, problems))
        outcome.steps = proposal.steps
        outcome.summary = proposal.summary
        outcome.problems = problems
        feedback = problems

    return outcome


def replan_after_failure(
    command: str, state: dict, llm, failure: str, max_replans: int = MAX_REPLANS
) -> Outcome:
    """Re-plan the remaining goal after an execution step failed verification.

    The failure reason is appended to the command so the model knows what went
    wrong, and the state passed in is the freshly observed one.
    """
    context = (
        f"{command}\n\n"
        "The scene state above is the current one. Plan only what still needs doing."
    )
    return plan(context, state, llm, max_replans=max_replans, feedback=[failure])


__all__ = ["plan", "replan_after_failure", "Outcome", "MAX_REPLANS", "MAX_REPLAN_ATTEMPTS"]

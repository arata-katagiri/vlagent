"""Phase 3: the planner validates before the user sees anything, and replans.

No network: a scripted fake stands in for the LLM so the control flow is
deterministic. MockLLM is exercised separately.
"""

import pytest

from robot_agent.actions.schema import ActionCall
from robot_agent.agent import planner
from robot_agent.agent.llm import MockLLM, Plan, Question, describe_state
from robot_agent.sim.backends import initial_state


def step(name, **args):
    return ActionCall(name=name, args=args, rationale="because")


class ScriptedLLM:
    """Returns queued proposals and records the feedback it was given."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.seen_feedback = []
        self.commands = []

    def propose(self, command, state, feedback=None):
        self.commands.append(command)
        self.seen_feedback.append(feedback)
        return self.proposals.pop(0) if self.proposals else Question(text="out of scripted replies")


@pytest.fixture
def state():
    return initial_state()


def test_a_valid_plan_is_returned_on_the_first_attempt(state):
    llm = ScriptedLLM(Plan(steps=[step("pick", object="red_block")], summary="pick it up"))
    outcome = planner.plan("pick up the red block", state, llm)
    assert outcome.ok and outcome.attempts == 1
    assert outcome.summary == "pick it up"
    assert llm.seen_feedback == [None]


def test_an_invalid_plan_is_replanned_with_the_problems_as_feedback(state):
    bad = Plan(steps=[step("pick", object="red_block"), step("place_on", target="glass")])
    good = Plan(steps=[step("pick", object="red_block"), step("place_on", target="tray")])
    llm = ScriptedLLM(bad, good)

    outcome = planner.plan("put the red block on the glass", state, llm)

    assert outcome.ok and outcome.attempts == 2
    assert llm.seen_feedback[0] is None
    assert any("fragile" in p for p in llm.seen_feedback[1])
    assert outcome.rejected and "fragile" in outcome.rejected[0][1][0]


def test_a_plan_that_never_validates_gives_up_with_problems(state):
    bad = Plan(steps=[step("pick", object="banana")])
    llm = ScriptedLLM(bad, bad, bad, bad)
    outcome = planner.plan("pick the banana", state, llm)
    assert not outcome.ok
    assert outcome.attempts == planner.MAX_REPLANS + 1
    assert any("banana" in p for p in outcome.problems)


def test_a_question_is_passed_through_without_validation(state):
    llm = ScriptedLLM(Question(text="which block do you mean?"))
    outcome = planner.plan("move the block", state, llm)
    assert outcome.question == "which block do you mean?"
    assert not outcome.ok and outcome.steps == []


def test_an_empty_plan_is_reported_not_executed(state):
    llm = ScriptedLLM(Plan(steps=[], summary="nothing to do"))
    outcome = planner.plan("do nothing", state, llm)
    assert not outcome.ok and outcome.problems


def test_replan_after_failure_passes_the_reason_as_feedback(state):
    llm = ScriptedLLM(Plan(steps=[step("pick", object="red_block")]))
    planner.replan_after_failure("clear the table", state, llm, "red_block slipped")
    assert llm.seen_feedback[0] == ["red_block slipped"]
    assert "clear the table" in llm.commands[0]


# --- MockLLM ------------------------------------------------------------

def test_mock_llm_handles_each_demo_command(state):
    llm = MockLLM()
    assert [s.name for s in llm.propose("clear the red and green blocks onto the tray", state).steps] == [
        "pick", "place_on", "pick", "place_on"
    ]
    assert [s.name for s in llm.propose("put the blue block on the glass", state).steps] == [
        "pick", "place_on"
    ]
    push = llm.propose("push the glass out of the way to the left", state)
    assert push.steps[0].args == {"object": "glass", "direction": "left", "distance_m": 0.3}


def test_mock_llm_offers_a_reversible_alternative_after_a_refusal(state):
    llm = MockLLM()
    alt = llm.propose(
        "push the glass out of the way to the left", state,
        feedback=["the user refused the irreversible step"],
    )
    assert [s.name for s in alt.steps] == ["pick", "place_on"]
    assert alt.steps[1].args["target"] == "tray"


def test_mock_llm_recovers_a_named_object_after_a_failed_placement(state):
    llm = MockLLM()
    out = llm.propose(
        "clear the red and green blocks onto the tray", state,
        feedback=["red_block is 0.080 m from where it should have landed on tray"],
    )
    assert isinstance(out, Plan)
    assert out.steps[0].args["object"] == "red_block"


def test_mock_llm_asks_rather_than_guessing_on_an_unknown_command(state):
    assert isinstance(MockLLM().propose("make me a sandwich", state), Question)


# --- prompt serialization -----------------------------------------------

def test_describe_state_marks_the_facts_the_agent_must_respect(state):
    state["objects"]["red_block"]["supporting"] = ["green_block"]
    state["objects"]["green_block"]["on_top_of"] = "red_block"
    text = describe_state(state)
    assert "FRAGILE" in text                      # the glass
    assert "tray" not in text.split("can be picked up:")[1].split("\n")[0]  # the tray
    assert "supporting green_block" in text
    assert "on top of red_block" in text
    assert "holding nothing" in text


# --- the model must be told the argument keys ---------------------------

def test_the_action_catalogue_names_every_action_and_its_keys():
    from robot_agent.actions.schema import ACTION_NAMES, describe_actions

    catalogue = describe_actions()
    for name in ACTION_NAMES:
        assert name in catalogue, f"{name} missing from the catalogue"
    assert '"object": "red_block"' in catalogue
    assert "distance_m: float in (0, 0.3]" in catalogue


def test_the_system_prompt_carries_the_catalogue_and_the_no_bin_rule():
    from robot_agent.agent.llm import SYSTEM_PROMPT

    # Without these, models emit steps with empty args and propose no-op plans.
    assert "required keys in args: object" in SYSTEM_PROMPT
    assert "An empty args object is never valid" in SYSTEM_PROMPT
    assert "no bin" in SYSTEM_PROMPT
    assert "net effect is nothing" in SYSTEM_PROMPT


# --- answering, rather than bouncing every question back ----------------

def test_an_answer_is_surfaced_and_is_not_an_executable_plan(state):
    from robot_agent.agent.llm import Answer

    llm = ScriptedLLM(Answer(text="On the table: red_block, cup, glass."))
    outcome = planner.plan("what is on the table?", state, llm)
    assert outcome.answer == "On the table: red_block, cup, glass."
    assert outcome.steps == [] and not outcome.ok and outcome.question is None


def test_mock_llm_answers_a_scene_question_instead_of_asking_one(state):
    from robot_agent.agent.llm import Answer, MockLLM

    reply = MockLLM().propose("tell me what's on the table", state)
    assert isinstance(reply, Answer)
    assert "red_block" in reply.text


def test_the_tool_schema_uses_flat_named_arguments(state):
    """A nested free-form args object made models emit `args: {}`."""
    from robot_agent.actions.schema import plan_tool_schema

    tools = plan_tool_schema()
    assert {t["function"]["name"] for t in tools} == {"propose_plan", "answer", "ask_user", "define_skill", "define_rule"}
    props = tools[0]["function"]["parameters"]["properties"]["steps"]["items"]["properties"]
    for key in ("object", "target", "x_m", "y_m", "direction", "distance_m"):
        assert key in props, f"{key} must be a named property, not nested under args"


def test_flat_and_nested_arguments_are_both_accepted():
    from robot_agent.agent.llm import _steps_from_args

    steps = _steps_from_args({"steps": [
        {"name": "pick", "object": "red_block", "rationale": "flat"},
        {"name": "place_on", "args": {"target": "tray"}, "rationale": "nested"},
    ]})
    assert steps[0].args == {"object": "red_block"}
    assert steps[1].args == {"target": "tray"}


def test_capabilities_are_listed_as_names_not_tagged_on_each_object(state):
    """As per-object tags the model read these directionally: "tray [fixed,
    cannot be picked up]" became a reason it could not place something *onto*
    the tray, and "cup [not a valid target]" blocked carrying the cup at all."""
    text = describe_state(state)
    cup_line = next(l for l in text.splitlines() if l.strip().startswith("cup "))
    tray_line = next(l for l in text.splitlines() if l.strip().startswith("tray "))
    assert "target" not in cup_line and "picked up" not in tray_line

    picked = next(l for l in text.splitlines() if l.startswith("can be picked up:"))
    targets = next(l for l in text.splitlines() if l.startswith("can be placed onto"))
    assert "cup" in picked and "tray" not in picked
    assert "tray" in targets and "cup" not in targets

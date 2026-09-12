"""Phase 2: plan validation and postcondition verification, on MockBackend.

Headless and instant: no simulator, no LLM.
"""

import pytest

from robot_agent.actions.executor import execute, validate_plan
from robot_agent.actions.schema import ActionCall
from robot_agent.sim.backends import MockBackend, initial_state


def call(name, **args):
    return ActionCall(name=name, args=args, rationale="test")


@pytest.fixture
def state():
    return initial_state()


@pytest.fixture
def backend(state):
    return MockBackend(state)


# --- validation simulates forward ---------------------------------------

def test_a_multi_step_plan_validates_because_state_is_simulated_forward(state):
    # place_on is only legal once pick has happened. A validator that checked
    # every step against the *initial* state would wrongly reject step 2.
    plan = [call("pick", object="red_block"), call("place_on", target="tray")]
    assert validate_plan(plan, state) == []


def test_two_picks_without_a_place_is_rejected(state):
    plan = [call("pick", object="red_block"), call("pick", object="green_block")]
    problems = validate_plan(plan, state)
    assert any("step 2" in p and "already holding red_block" in p for p in problems)


def test_clearing_two_blocks_onto_the_tray_validates(state):
    plan = [
        call("pick", object="red_block"),
        call("place_on", target="tray"),
        call("pick", object="green_block"),
        call("place_on", target="tray"),
    ]
    assert validate_plan(plan, state) == []


def test_validation_reports_the_failing_step_number(state):
    plan = [call("pick", object="red_block"), call("place_on", target="glass")]
    problems = validate_plan(plan, state)
    assert problems and problems[0].startswith("step 2 (place_on)")


# --- execution and verification -----------------------------------------

def test_pick_then_place_on_tray_executes_and_verifies(state, backend):
    get = lambda: state  # noqa: E731

    r1 = execute(call("pick", object="red_block"), backend, get)
    assert r1.ok, r1.reason
    assert state["gripper"]["holding"] == "red_block"

    r2 = execute(call("place_on", target="tray"), backend, get)
    assert r2.ok, r2.reason
    assert state["gripper"]["holding"] is None
    assert state["objects"]["red_block"]["on_top_of"] == "tray"
    assert "red_block" in state["objects"]["tray"]["supporting"]


def test_execute_refuses_a_call_whose_preconditions_fail(state, backend):
    state["gripper"]["holding"] = "cup"
    state["objects"]["cup"]["held"] = True
    r = execute(call("pick", object="red_block"), backend, lambda: state)
    assert not r.ok and "already holding cup" in r.reason


def test_postcondition_failure_is_caught_when_the_object_drifts(state):
    # drift_m nudges the object on release: the placement lands outside the 2 cm
    # tolerance, and verification must catch it even though every motion
    # "succeeded". This is what --inject-failure exercises.
    backend = MockBackend(state, drift_m=0.08)
    get = lambda: state  # noqa: E731
    assert execute(call("pick", object="red_block"), backend, get).ok

    r = execute(call("place_on", target="tray"), backend, get)
    assert not r.ok, "an 8 cm drift must fail the 2 cm placement tolerance"
    assert "red_block" in r.reason


def test_unreachable_target_reports_a_motion_failure(state):
    backend = MockBackend(state, fail_moves_beyond_m=0.30)
    r = execute(call("pick", object="cup"), backend, lambda: state)
    assert not r.ok and "could not reach" in r.reason


def test_place_at_verifies_the_requested_point(state, backend):
    get = lambda: state  # noqa: E731
    assert execute(call("pick", object="green_block"), backend, get).ok
    r = execute(call("place_at", x_m=0.55, y_m=-0.10), backend, get)
    assert r.ok, r.reason
    assert state["objects"]["green_block"]["position_m"][:2] == pytest.approx([0.55, -0.10])


def test_push_moves_the_object_and_verifies_travel(state, backend):
    r = execute(call("push", object="red_block", direction="forward", distance_m=0.10), backend, lambda: state)
    assert r.ok, r.reason
    assert "moved" in r.reason


def test_a_backend_exception_becomes_a_failed_result_not_a_crash(state):
    class Exploding(MockBackend):
        def attach(self, obj_name):
            raise RuntimeError("gripper jammed")

    backend = Exploding(state)
    r = execute(call("pick", object="red_block"), backend, lambda: state)
    assert not r.ok and "gripper jammed" in r.reason


def test_stacking_then_picking_the_bottom_block_is_rejected(state, backend):
    get = lambda: state  # noqa: E731
    assert execute(call("pick", object="green_block"), backend, get).ok
    assert execute(call("place_on", target="red_block"), backend, get).ok

    problems = validate_plan([call("pick", object="red_block")], state)
    assert any("green_block on top" in p for p in problems)


def test_two_objects_land_side_by_side_on_the_tray(state, backend):
    """The demo clears two blocks onto the tray; the second must not land on the first."""
    get = lambda: state  # noqa: E731
    for name in ("red_block", "green_block"):
        assert execute(call("pick", object=name), backend, get).ok
        r = execute(call("place_on", target="tray"), backend, get)
        assert r.ok, r.reason

    tray = state["objects"]["tray"]
    assert sorted(tray["supporting"]) == ["green_block", "red_block"]
    red = state["objects"]["red_block"]["position_m"]
    green = state["objects"]["green_block"]["position_m"]
    assert red[:2] != green[:2], "both blocks were dropped at the same spot"
    assert state["objects"]["green_block"]["on_top_of"] == "tray"


def test_place_on_spot_is_stable_for_an_empty_surface(state):
    from robot_agent.actions.executor import place_on_spot

    state["gripper"]["holding"] = "red_block"
    spot = place_on_spot(state, "tray", "red_block")
    tray = state["objects"]["tray"]
    assert spot[:2] == pytest.approx(tray["position_m"][:2])


# --- plans that achieve nothing -----------------------------------------

def test_picking_an_object_and_putting_it_straight_back_is_rejected(state):
    """The bug behind "remove the red cube from the table": every step was
    individually legal, so the agent reported success for a no-op."""
    tray = state["objects"]["tray"]
    red = state["objects"]["red_block"]
    red["position_m"] = [
        tray["position_m"][0],
        tray["position_m"][1],
        round(tray["position_m"][2] + tray["half_extent_m"][2] + red["half_extent_m"][2], 4),
    ]
    red["on_top_of"] = "tray"
    tray["supporting"] = ["red_block"]

    plan = [call("pick", object="red_block"), call("place_on", target="tray")]
    problems = validate_plan(plan, state)
    assert problems and "leave the scene exactly as it is" in problems[0]


def test_a_plan_that_actually_moves_something_is_not_a_no_op(state):
    from robot_agent.actions.executor import is_no_op

    assert not is_no_op(
        [call("pick", object="red_block"), call("place_on", target="tray")], state
    )


def test_home_and_ask_user_are_not_treated_as_no_ops(state):
    from robot_agent.actions.executor import is_no_op

    assert not is_no_op([call("home")], state)
    assert not is_no_op([call("ask_user", question="which one?")], state)
    assert validate_plan([call("home")], state) == []

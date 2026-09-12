"""Phase 2: safety is deterministic code, so it is tested as pure functions.

No simulator, no LLM. Every case here is a hand-built state dict.
"""

import pytest

from robot_agent.actions.safety import (
    check_preconditions,
    classify,
    overhang_m,
    predict_push_end,
    reachable,
)
from robot_agent.actions.schema import MAX_PUSH_DISTANCE_M, ActionCall, Safety
from robot_agent.sim.backends import initial_state


@pytest.fixture
def state():
    return initial_state()


def call(name, **args):
    return ActionCall(name=name, args=args, rationale="test")


# --- the four cases the build plan names explicitly ----------------------

def test_pushing_the_glass_off_the_table_is_irreversible(state):
    # The glass sits at y=0.22 and the table ends at y=0.38, so a 0.3 m push
    # to the left puts it 0.14 m past the edge.
    c = call("push", object="glass", direction="left", distance_m=0.3)
    assert check_preconditions(c, state) == []
    level, reason = classify(c, state)
    assert level is Safety.IRREVERSIBLE
    assert "past the table edge" in reason and "glass would fall" in reason


def test_placing_on_the_glass_is_rejected(state):
    state["gripper"]["holding"] = "blue_block"
    state["objects"]["blue_block"]["held"] = True
    problems = check_preconditions(call("place_on", target="glass"), state)
    assert problems, "stacking on a fragile object must be refused"
    assert "fragile" in problems[0]


def test_picking_an_object_with_something_on_top_fails(state):
    state["objects"]["red_block"]["supporting"] = ["green_block"]
    state["objects"]["green_block"]["on_top_of"] = "red_block"
    problems = check_preconditions(call("pick", object="red_block"), state)
    assert any("green_block on top" in p for p in problems)


def test_a_short_push_that_stays_on_the_table_is_not_irreversible(state):
    c = call("push", object="red_block", direction="forward", distance_m=0.05)
    level, _ = classify(c, state)
    assert level is not Safety.IRREVERSIBLE


# --- preconditions -------------------------------------------------------

def test_unknown_object_is_refused(state):
    problems = check_preconditions(call("pick", object="banana"), state)
    assert problems == ["there is no object called 'banana'"]


def test_unknown_action_is_refused(state):
    assert check_preconditions(call("teleport", object="cup"), state) == [
        "unknown action 'teleport'"
    ]


def test_missing_argument_is_refused(state):
    problems = check_preconditions(ActionCall(name="pick", args={}), state)
    assert "missing required argument" in problems[0]


def test_cannot_pick_with_a_full_gripper(state):
    state["gripper"]["holding"] = "cup"
    state["objects"]["cup"]["held"] = True
    problems = check_preconditions(call("pick", object="red_block"), state)
    assert any("already holding cup" in p for p in problems)


def test_the_tray_cannot_be_picked_up(state):
    problems = check_preconditions(call("pick", object="tray"), state)
    assert any("fixed in place" in p for p in problems)


def test_place_on_without_holding_anything_is_refused(state):
    problems = check_preconditions(call("place_on", target="tray"), state)
    assert any("not holding anything" in p for p in problems)


def test_push_while_holding_is_refused(state):
    state["gripper"]["holding"] = "cup"
    problems = check_preconditions(call("push", object="red_block", direction="left", distance_m=0.1), state)
    assert any("put it down before pushing" in p for p in problems)


@pytest.mark.parametrize("distance", [0.0, -0.1, MAX_PUSH_DISTANCE_M + 0.01])
def test_push_distance_bounds(state, distance):
    problems = check_preconditions(
        call("push", object="red_block", direction="left", distance_m=distance), state
    )
    assert any("distance_m" in p for p in problems)


def test_bad_push_direction_is_refused(state):
    problems = check_preconditions(
        call("push", object="red_block", direction="sideways", distance_m=0.1), state
    )
    assert any("direction must be one of" in p for p in problems)


def test_place_at_out_of_reach_is_refused(state):
    state["gripper"]["holding"] = "cup"
    problems = check_preconditions(call("place_at", x_m=2.0, y_m=0.0), state)
    assert any("outside the arm's reach" in p for p in problems)


# --- classification ------------------------------------------------------

def test_place_at_off_the_table_is_irreversible(state):
    state["gripper"]["holding"] = "cup"
    level, reason = classify(call("place_at", x_m=0.45, y_m=0.55), state)
    assert level is Safety.IRREVERSIBLE
    assert "cup would fall" in reason


def test_place_at_near_the_edge_is_caution(state):
    state["gripper"]["holding"] = "cup"
    level, reason = classify(call("place_at", x_m=0.45, y_m=0.35), state)
    assert level is Safety.CAUTION
    assert "from the table edge" in reason


def test_picking_a_fragile_object_is_caution(state):
    level, reason = classify(call("pick", object="glass"), state)
    assert level is Safety.CAUTION and "fragile" in reason


def test_picking_a_sturdy_object_is_safe(state):
    level, _ = classify(call("pick", object="red_block"), state)
    assert level is Safety.SAFE


def test_home_and_ask_user_are_safe(state):
    assert classify(call("home"), state)[0] is Safety.SAFE
    assert classify(call("ask_user", question="which cup?"), state)[0] is Safety.SAFE


# --- geometry helpers ----------------------------------------------------

def test_predict_push_end_follows_the_direction_convention(state):
    x, y = predict_push_end(state, "red_block", "left", 0.1)
    assert y == pytest.approx(state["objects"]["red_block"]["position_m"][1] + 0.1)
    x2, _ = predict_push_end(state, "red_block", "forward", 0.1)
    assert x2 == pytest.approx(state["objects"]["red_block"]["position_m"][0] + 0.1)


def test_overhang_is_zero_on_the_table(state):
    assert overhang_m(state, 0.5, 0.0) == 0.0
    assert overhang_m(state, 0.5, 0.48) == pytest.approx(0.10)


def test_reachable_annulus():
    assert not reachable(0.05, 0.0)
    assert reachable(0.5, 0.0)
    assert not reachable(1.2, 0.0)

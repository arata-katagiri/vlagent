"""The lab bench is a second scenario: activating it swaps the objects and
installs its policy through the safety hooks; restoring puts everything back.
Pure functions and hand-built state, no simulator."""

import pytest

from robot_agent.actions import safety
from robot_agent.actions.safety import check_preconditions, classify
from robot_agent.actions.schema import ActionCall, Safety
from robot_agent.agent import llm
from robot_agent.sim import lab_scene, scene
from robot_agent.sim.backends import initial_state


@pytest.fixture
def lab():
    restore = lab_scene.activate()
    try:
        yield initial_state()
    finally:
        restore()


def call(name, **args):
    return ActionCall(name=name, args=args, rationale="test")


def _hold(state, name):
    state["objects"][name].update(held=True, on_table=False, on_top_of=None)
    state["gripper"]["holding"] = name


def test_activation_swaps_objects_and_restore_puts_them_back():
    before = set(initial_state()["objects"])
    restore = lab_scene.activate()
    try:
        during = set(initial_state()["objects"])
        assert {"acid_bottle", "water_flask", "red_sample", "tray"} <= during
        assert "cup" not in during and "red_block" not in during
        assert "red_sample" in safety.STACKABLE_TARGETS
        assert lab_scene.precondition_hook in safety.PRECONDITION_HOOKS
        assert any("Lab policy" in r for r in llm.SCENARIO_RULES)
    finally:
        restore()
    assert set(initial_state()["objects"]) == before
    assert "red_block" in safety.STACKABLE_TARGETS
    assert lab_scene.precondition_hook not in safety.PRECONDITION_HOOKS
    assert not any("Lab policy" in r for r in llm.SCENARIO_RULES)
    assert "cup" in scene.ITEMS and "acid_bottle" not in scene.ITEMS


def test_place_at_next_to_the_flask_is_refused(lab):
    _hold(lab, "acid_bottle")
    fx, fy = lab["objects"]["water_flask"]["position_m"][:2]
    problems = check_preconditions(call("place_at", x_m=fx, y_m=fy - 0.05), lab)
    assert any("must stay at least" in p for p in problems), problems


def test_place_at_far_from_the_flask_is_fine(lab):
    _hold(lab, "acid_bottle")
    fx, fy = lab["objects"]["water_flask"]["position_m"][:2]
    assert check_preconditions(call("place_at", x_m=fx, y_m=fy - 0.30), lab) == []


def test_pushing_the_acid_into_the_flask_is_refused(lab):
    ax, ay, az = lab["objects"]["acid_bottle"]["position_m"]
    lab["objects"]["water_flask"]["position_m"] = [ax, ay + 0.15, az]
    problems = check_preconditions(call("push", object="acid_bottle", direction="left", distance_m=0.1), lab)
    assert any("must stay at least" in p for p in problems), problems


def test_picking_the_acid_is_caution(lab):
    level, reason = classify(call("pick", object="acid_bottle"), lab)
    assert level is Safety.CAUTION and "corrosive" in reason


def test_samples_still_stack_and_the_flask_is_fragile(lab):
    _hold(lab, "blue_sample")
    assert check_preconditions(call("place_on", target="red_sample"), lab) == []
    assert any("fragile" in p for p in check_preconditions(call("place_on", target="water_flask"), lab))


def test_policy_off_disables_the_lab_rule(lab):
    _hold(lab, "acid_bottle")
    fx, fy = lab["objects"]["water_flask"]["position_m"][:2]
    safety.set_policy_enabled(False)
    try:
        assert check_preconditions(call("place_at", x_m=fx, y_m=fy - 0.05), lab) == []
    finally:
        safety.set_policy_enabled(True)


def test_default_scene_has_no_lab_rule():
    state = initial_state()
    _hold(state, "cup")
    gx, gy = state["objects"]["glass"]["position_m"][:2]
    assert check_preconditions(call("place_at", x_m=gx, y_m=gy - 0.05), state) == []

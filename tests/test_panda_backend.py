"""Phase 4: the real IK backend, headless.

Slower than the rest of the suite (it steps physics), but this is the only test
that exercises mink, the weld grasp and the executor together. It never opens
the viewer.
"""

import numpy as np
import pytest

from robot_agent.actions.executor import execute
from robot_agent.actions.schema import ActionCall
from robot_agent.sim.backends import PandaIKBackend
from robot_agent.sim.scene import GRASP_SITE, build_scene, settle
from robot_agent.sim.world_state import get_world_state


@pytest.fixture(scope="module")
def arm():
    model, data = build_scene()
    settle(model, data, 0.3)
    return model, data, PandaIKBackend(model, data)


def test_ik_reaches_a_tabletop_target(arm):
    model, data, backend = arm
    target = [0.42, -0.16, 0.52]
    assert backend.move_to(target)
    assert np.linalg.norm(data.site(GRASP_SITE).xpos - np.array(target)) <= 0.01


def test_pick_and_place_a_block_end_to_end():
    model, data = build_scene()
    settle(model, data, 0.3)
    backend = PandaIKBackend(model, data)
    get = lambda: get_world_state(model, data)  # noqa: E731

    picked = execute(ActionCall("pick", {"object": "red_block"}, "test"), backend, get)
    assert picked.ok, picked.reason
    assert get()["gripper"]["holding"] == "red_block"

    placed = execute(ActionCall("place_on", {"target": "tray"}, "test"), backend, get)
    assert placed.ok, placed.reason

    state = get()
    assert state["objects"]["red_block"]["on_top_of"] == "tray"
    assert state["gripper"]["holding"] is None


def test_the_weld_holds_the_object_against_gravity():
    model, data = build_scene()
    settle(model, data, 0.3)
    backend = PandaIKBackend(model, data)
    get = lambda: get_world_state(model, data)  # noqa: E731

    assert execute(ActionCall("pick", {"object": "cup"}, "test"), backend, get).ok
    held_z = get()["objects"]["cup"]["position_m"][2]

    # Carry it sideways; an unwelded object would be left behind on the table.
    assert backend.move_to([0.45, 0.10, 0.58], tolerance_m=0.025)
    carried = get()["objects"]["cup"]["position_m"]
    assert carried[2] > held_z - 0.02, "the cup was dropped mid-carry"
    assert abs(carried[1] - 0.10) < 0.05, "the cup did not follow the gripper"


def test_injected_drift_is_visible_in_the_observed_state():
    model, data = build_scene()
    settle(model, data, 0.3)
    backend = PandaIKBackend(model, data, drift_m=0.08)
    get = lambda: get_world_state(model, data)  # noqa: E731

    assert execute(ActionCall("pick", {"object": "red_block"}, "test"), backend, get).ok
    result = execute(ActionCall("place_on", {"target": "tray"}, "test"), backend, get)
    assert not result.ok, "an 8 cm nudge must fail verification"
    assert "red_block" in result.reason

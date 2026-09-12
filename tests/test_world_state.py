"""Phase 1: the derived fields of get_world_state must reflect real scene changes.

Headless. Builds the scene but never opens the viewer.
"""

import mujoco
import numpy as np
import pytest

from robot_agent.sim.scene import (
    ITEMS,
    TABLE_BOUNDS_M,
    build_scene,
    half_height,
    object_qposadr,
    settle,
)
from robot_agent.sim.world_state import get_world_state


@pytest.fixture(scope="module")
def scene():
    model, data = build_scene()
    settle(model, data, 0.5)
    return model, data


def _place(model, data, name, x, y, z):
    adr = object_qposadr(model, name)
    data.qpos[adr : adr + 3] = [x, y, z]
    data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    data.qvel[model.joint(int(model.body(name).jntadr[0])).dofadr[0] :][:6] = 0
    mujoco.mj_forward(model, data)


def test_baseline_all_objects_on_table(scene):
    state = get_world_state(*scene)
    assert set(state["objects"]) == set(ITEMS)
    for name, obj in state["objects"].items():
        assert obj["on_table"], f"{name} is not on the table"
        assert obj["on_top_of"] is None
        assert obj["held"] is False
    assert state["gripper"]["holding"] is None


def test_stacking_is_detected(scene):
    model, data = build_scene()
    settle(model, data, 0.3)
    tray = ITEMS["tray"]
    tray_top = tray["pos"][2] + half_height(tray)
    _place(
        model, data, "blue_block",
        tray["pos"][0], tray["pos"][1], tray_top + half_height(ITEMS["blue_block"]),
    )
    state = get_world_state(model, data)
    assert state["objects"]["blue_block"]["on_top_of"] == "tray"
    assert state["objects"]["tray"]["supporting"] == ["blue_block"]
    # A block that is not stacked must not claim a parent.
    assert state["objects"]["red_block"]["on_top_of"] is None


def test_near_table_edge_is_detected(scene):
    model, data = build_scene()
    settle(model, data, 0.3)
    state = get_world_state(model, data)
    assert state["objects"]["cup"]["near_table_edge"] is False

    _place(model, data, "cup", 0.45, TABLE_BOUNDS_M["y_max"] - 0.02,
           TABLE_BOUNDS_M["top_z"] + half_height(ITEMS["cup"]))
    state = get_world_state(model, data)
    assert state["objects"]["cup"]["near_table_edge"] is True


def test_off_table_is_not_on_table(scene):
    model, data = build_scene()
    _place(model, data, "cup", 1.5, 0.0, 0.05)
    state = get_world_state(model, data)
    assert state["objects"]["cup"]["on_table"] is False
    assert state["objects"]["cup"]["near_table_edge"] is False


def test_active_weld_reports_held(scene):
    model, data = build_scene()
    settle(model, data, 0.3)
    eq_id = next(
        i for i in range(model.neq) if model.equality(i).name == "weld_red_block"
    )
    data.eq_active[eq_id] = True
    mujoco.mj_forward(model, data)

    state = get_world_state(model, data)
    assert state["objects"]["red_block"]["held"] is True
    assert state["gripper"]["holding"] == "red_block"
    assert state["objects"]["green_block"]["held"] is False


def test_gripper_position_tracks_the_grasp_site(scene):
    model, data = scene
    state = get_world_state(model, data)
    assert np.allclose(state["gripper"]["position_m"], data.site("grasp_site").xpos, atol=1e-3)

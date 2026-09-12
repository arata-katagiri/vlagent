"""Two Pandas facing each other across one table.

Standalone: nothing in the single-arm path imports this module. The scene is
assembled the other way round from scene.py -- an empty world spec into which
the Menagerie Panda is *attached* twice with name prefixes ("a_", "b_"), so
every body, joint, actuator, site and keyframe exists once per arm.

Frame: arm A's base is the world origin facing +x; arm B's base is 1.05 m along
+x, rotated 180 deg about z, so it faces A. The table sits between them and the
shared reach zone covers the whole table top.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import OBJECTS, TCP_OFFSET_M, half_height, panda_xml_path

CAMERA_NAME = "demo"
GRASP_SITE = "grasp_site"  # per arm: f"{prefix}{GRASP_SITE}"
HAND_BODY = "hand"          # per arm: f"{prefix}{HAND_BODY}"

# Menagerie `home` keyframe for the bare 9-DOF arm (7 joints + 2 fingers).
# Franka "ready" pose rather than Menagerie `home`: with two arms 1.05 m apart the
# Menagerie pose puts the two hands 6 cm from each other at reset. Ready keeps the
# wrist retracted (TCP ~0.31 m from the base) and the gripper still points down.
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
GRIPPER_OPEN_CTRL = 255.0

ARMS: dict[str, dict] = {
    "a": dict(prefix="a_", base=(0.0, 0.0, 0.0), yaw=0.0),
    "b": dict(prefix="b_", base=(1.05, 0.0, 0.0), yaw=math.pi),
}

# Table between the bases. Both arms reach 0.25-0.80 m radially from their own
# base, so x in [0.30, 0.75] is inside both envelopes along the centre line.
TABLE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.30, "x_max": 0.75, "y_min": -0.38, "y_max": 0.38, "top_z": 0.40,
}
_T = TABLE_BOUNDS_M
_CX = 0.5 * (_T["x_min"] + _T["x_max"])
_CY = 0.5 * (_T["y_min"] + _T["y_max"])
_HX = 0.5 * (_T["x_max"] - _T["x_min"])
_HY = 0.5 * (_T["y_max"] - _T["y_min"])

# Same objects as the single-arm scene, re-laid out: blocks on A's side, cups on
# B's side, the tray in the middle as the hand-off zone.
LAYOUT: dict[str, tuple[float, float]] = {
    "red_block": (0.40, -0.16),
    "green_block": (0.42, 0.02),
    "blue_block": (0.40, 0.20),
    "cup": (0.66, -0.20),
    "glass": (0.64, 0.22),
}
OBJECTS2: dict[str, dict] = {
    name: dict(OBJECTS[name], pos=(x, y, 0.0)) for name, (x, y) in LAYOUT.items()
}
GRASPABLE2 = tuple(n for n, o in OBJECTS2.items() if o["graspable"])
TRAY2 = dict(
    half=(0.09, 0.09, 0.006), rgba=(0.30, 0.55, 0.65, 1.0),
    pos=(_CX, _CY, _T["top_z"] + 0.006),
)


def rest_z(obj: dict) -> float:
    return _T["top_z"] + half_height(obj)


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def build_two_arm_spec() -> mujoco.MjSpec:
    world = mujoco.MjSpec()
    world.modelname = "two_pandas"
    # panda.xml asks for implicitfast; the parent's option wins on attach, so
    # set it here or the arms integrate differently from the single-arm scene.
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.visual.global_.offwidth = 1280
    world.visual.global_.offheight = 720

    wb = world.worldbody

    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.30, 0.32, 0.36, 1.0]

    light = wb.add_light()
    light.pos = [_CX, 0.0, 2.0]
    light.dir = [0.0, 0.0, -1.0]

    top_half = 0.02
    table = wb.add_body()
    table.name = "table"
    table.pos = [_CX, _CY, _T["top_z"] - top_half]
    tg = table.add_geom()
    tg.name = "table_top"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [_HX, _HY, top_half]
    tg.rgba = [0.76, 0.64, 0.48, 1.0]

    leg_half = 0.022
    leg_h = 0.5 * (_T["top_z"] - 2 * top_half)
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = wb.add_body()
            leg.name = f"table_leg_{'p' if sx > 0 else 'm'}{'p' if sy > 0 else 'm'}"
            leg.pos = [_CX + sx * (_HX - leg_half - 0.01), _CY + sy * (_HY - leg_half - 0.01), leg_h]
            lg = leg.add_geom()
            lg.name = leg.name + "_geom"
            lg.type = mujoco.mjtGeom.mjGEOM_BOX
            lg.size = [leg_half, leg_half, leg_h]
            lg.rgba = [0.55, 0.45, 0.34, 1.0]

    tray = wb.add_body()
    tray.name = "tray"
    tray.pos = list(TRAY2["pos"])
    yg = tray.add_geom()
    yg.name = "tray_geom"
    yg.type = mujoco.mjtGeom.mjGEOM_BOX
    yg.size = list(TRAY2["half"])
    yg.rgba = list(TRAY2["rgba"])

    for name, obj in OBJECTS2.items():
        body = wb.add_body()
        body.name = name
        body.pos = [obj["pos"][0], obj["pos"][1], rest_z(obj)]
        body.add_freejoint()
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.rgba = list(obj["rgba"])
        if obj["kind"] == "box":
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = list(obj["half"])
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [obj["half"][0], obj["half"][1], 0.0]

    # The arms. Add the grasp site to the child *before* attaching so it gets
    # the prefix like everything else.
    for arm in ARMS.values():
        child = mujoco.MjSpec.from_file(panda_xml_path().as_posix())
        site = child.body(HAND_BODY).add_site()
        site.name = GRASP_SITE
        site.pos = list(TCP_OFFSET_M)
        site.size = [0.006, 0.006, 0.006]
        site.rgba = [1.0, 0.2, 0.8, 0.6]
        frame = wb.add_frame(pos=list(arm["base"]), quat=_yaw_quat(arm["yaw"]))
        world.attach(child, prefix=arm["prefix"], frame=frame)

        # One inactive weld per (arm, graspable object).
        for name in GRASPABLE2:
            eq = world.add_equality()
            eq.name = f"{arm['prefix']}weld_{name}"
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = f"{arm['prefix']}{HAND_BODY}"
            eq.name2 = name
            eq.active = False

    cam = wb.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = [_CX + 0.9, -2.0, 1.45]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "table"
    return world


def object_qposadr(model: mujoco.MjModel, name: str) -> int:
    return int(model.joint(int(model.body(name).jntadr[0])).qposadr[0])


def arm_joint_qposadr(model: mujoco.MjModel, prefix: str) -> list[int]:
    names = [f"{prefix}joint{i}" for i in range(1, 8)] + [f"{prefix}finger_joint1", f"{prefix}finger_joint2"]
    return [int(model.joint(n).qposadr[0]) for n in names]


def arm_actuator_ids(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [int(model.actuator(f"{prefix}actuator{i}").id) for i in range(1, 9)]


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Both arms at Menagerie `home`, objects at their rest poses. No keyframe
    trick: the attached keyframes are per-arm and do not know about objects."""
    mujoco.mj_resetData(model, data)
    for arm in ARMS.values():
        for adr, q in zip(arm_joint_qposadr(model, arm["prefix"]), HOME_QPOS):
            data.qpos[adr] = q
        acts = arm_actuator_ids(model, arm["prefix"])
        data.ctrl[acts[:7]] = HOME_QPOS[:7]
        data.ctrl[acts[7]] = GRIPPER_OPEN_CTRL
    for name, obj in OBJECTS2.items():
        adr = object_qposadr(model, name)
        data.qpos[adr: adr + 3] = [obj["pos"][0], obj["pos"][1], rest_z(obj)]
        data.qpos[adr + 3: adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def build_two_arm_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_two_arm_spec().compile()
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def settle(model, data, seconds: float = 0.5) -> None:
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


__all__ = [
    "ARMS", "CAMERA_NAME", "GRASP_SITE", "HAND_BODY", "HOME_QPOS", "TABLE_BOUNDS_M",
    "OBJECTS2", "GRASPABLE2", "TRAY2", "LAYOUT", "rest_z",
    "build_two_arm_spec", "build_two_arm_scene", "reset", "settle",
    "object_qposadr", "arm_joint_qposadr", "arm_actuator_ids",
]

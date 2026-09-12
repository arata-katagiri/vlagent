"""Builds the scene: Panda + table + objects + inactive grasp welds.

The Panda comes from the robot_descriptions package (panda_mj_description),
which reads the MuJoCo Menagerie clone cached under ~/.cache. Falls back to a
local checkout via MENAGERIE_PATH, so the scene still loads with the venue
Wi-Fi down.

We assemble with mujoco.MjSpec rather than an MJCF <include>: the Menagerie
panda.xml declares meshdir="assets" relative to itself, which breaks when it is
included from a file in another directory. MjSpec.from_file resolves the assets
against the right directory, and lets us add a grasp site inside the existing
`hand` body, which plain MJCF cannot do.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

# Named camera shared by the viewer, the offscreen renderer and the recorder, so
# every image of this project is framed identically.
CAMERA_NAME = "demo"

# Frame the IK drives. A site we add to the `hand` body, because Menagerie's
# panda.xml has no sites at all (verified in Phase 0).
GRASP_SITE = "grasp_site"

# Where the grasp site sits inside the `hand` body, measured at `home`.
#
# The finger-body midpoint is 0.0584 m down the hand's local z, but the
# collidable fingertip pads reach a further 0.053 m below that. Putting the site
# at the midpoint means commanding the TCP to an object's centre drives the pads
# 3 cm into the table, and the arm stalls against the contact. The site is
# therefore placed at the pad centre, so "TCP at the object's centre" is a grasp
# the fingers can actually close on.
TCP_OFFSET_M = (0.0, 0.0, 0.1029)
FINGER_MIDPOINT_OFFSET_M = 0.0584

HAND_BODY = "hand"

# Single source of truth for the table extents, in metres.
TABLE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.28,
    "x_max": 0.82,
    "y_min": -0.38,
    "y_max": 0.38,
    "top_z": 0.40,
}

_T = TABLE_BOUNDS_M
_CX = 0.5 * (_T["x_min"] + _T["x_max"])
_CY = 0.5 * (_T["y_min"] + _T["y_max"])
_HX = 0.5 * (_T["x_max"] - _T["x_min"])
_HY = 0.5 * (_T["y_max"] - _T["y_min"])

# name -> properties. `half` is the geom half-extent used for both the visual
# size and the placement height. Aliases are what the planner may resolve.
OBJECTS: dict[str, dict] = {
    "red_block": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.85, 0.15, 0.15, 1.0),
        pos=(0.42, -0.16, 0.0), color="red", fragile=False, graspable=True,
        aliases=("red block", "the red one", "red cube"),
    ),
    "green_block": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.15, 0.70, 0.20, 1.0),
        pos=(0.42, 0.0, 0.0), color="green", fragile=False, graspable=True,
        aliases=("green block", "the green one", "green cube"),
    ),
    "blue_block": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.15, 0.30, 0.85, 1.0),
        pos=(0.36, 0.20, 0.0), color="blue", fragile=False, graspable=True,
        aliases=("blue block", "the blue one", "blue cube"),
    ),
    "cup": dict(
        kind="cylinder", half=(0.035, 0.035), rgba=(0.90, 0.75, 0.30, 1.0),
        pos=(0.52, -0.24, 0.0), color="yellow", fragile=False, graspable=True,
        aliases=("the cup", "mug"),
    ),
    "glass": dict(
        kind="cylinder", half=(0.026, 0.060), rgba=(0.75, 0.85, 0.95, 0.45),
        pos=(0.50, 0.24, 0.0), color="clear", fragile=True, graspable=True,
        aliases=("the glass", "wine glass", "tumbler"),
    ),
}

# Fixed, not a free body: the "safe place".
# Kept inside the arm's envelope: placing the 12 cm glass on a tray at radius
# 0.68 m demands ~0.90 m of reach, past what the Panda has.
TRAY = dict(
    half=(0.105, 0.085, 0.006), rgba=(0.30, 0.55, 0.65, 1.0),
    pos=(0.57, -0.02, _T["top_z"] + 0.006),
)

GRASPABLE = tuple(n for n, o in OBJECTS.items() if o["graspable"])

# Everything the agent can refer to, including the fixed tray. The tray is a
# placement target, never a grasp target.
ITEMS: dict[str, dict] = dict(
    OBJECTS,
    tray=dict(
        kind="box", half=TRAY["half"], rgba=TRAY["rgba"], pos=TRAY["pos"],
        color="teal", fragile=False, graspable=False,
        aliases=("the tray", "somewhere safe", "safe place"),
    ),
)


def half_height(item: dict) -> float:
    """Half the vertical extent of an item's geom."""
    return item["half"][2] if item["kind"] == "box" else item["half"][1]


def footprint(item: dict) -> tuple[float, float]:
    """Half-extents of an item in x and y."""
    if item["kind"] == "box":
        return item["half"][0], item["half"][1]
    return item["half"][0], item["half"][0]


def panda_xml_path() -> Path:
    """Locate the Menagerie Panda MJCF, preferring a local checkout."""
    override = os.environ.get("MENAGERIE_PATH")
    if override:
        candidate = Path(override) / "franka_emika_panda" / "panda.xml"
        if candidate.exists():
            return candidate
    from robot_descriptions import panda_mj_description  # noqa: PLC0415

    return Path(panda_mj_description.MJCF_PATH)


def _rest_z(obj: dict) -> float:
    """Height of the body origin when the object rests on the table top."""
    return _T["top_z"] + half_height(obj)


# Scenario decoration: callables that receive the finished spec before it is
# compiled (backdrops, fixed props). Scenarios append to this in activate().
DECORATORS: list = []


def build_spec() -> mujoco.MjSpec:
    """Assemble the full scene spec (Panda + table + objects + welds)."""
    spec = mujoco.MjSpec.from_file(panda_xml_path().as_posix())
    world = spec.worldbody

    # Menagerie ships a 640x480 offscreen framebuffer; our stills and the demo
    # recording need more than that.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # Ground plane. panda.xml carries its own light but no floor.
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.30, 0.32, 0.36, 1.0]

    # Table: a thin top at TABLE_BOUNDS_M["top_z"] on four legs. A full-height
    # box would read as a crate on video and hide the arm's base.
    top_half = 0.02
    table = world.add_body()
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
            leg = world.add_body()
            leg.name = f"table_leg_{'p' if sx > 0 else 'm'}{'p' if sy > 0 else 'm'}"
            leg.pos = [
                _CX + sx * (_HX - leg_half - 0.01),
                _CY + sy * (_HY - leg_half - 0.01),
                leg_h,
            ]
            lg = leg.add_geom()
            lg.name = leg.name + "_geom"
            lg.type = mujoco.mjtGeom.mjGEOM_BOX
            lg.size = [leg_half, leg_half, leg_h]
            lg.rgba = [0.55, 0.45, 0.34, 1.0]

    # Tray: fixed, the designated safe place.
    tray = world.add_body()
    tray.name = "tray"
    tray.pos = list(TRAY["pos"])
    yg = tray.add_geom()
    yg.name = "tray_geom"
    yg.type = mujoco.mjtGeom.mjGEOM_BOX
    yg.size = list(TRAY["half"])
    yg.rgba = list(TRAY["rgba"])

    # Manipulable objects: free bodies resting on the table.
    for name, obj in OBJECTS.items():
        body = world.add_body()
        body.name = name
        body.pos = [obj["pos"][0], obj["pos"][1], _rest_z(obj)]
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

    # Grasp site inside the existing hand body, at the measured TCP offset.
    hand = spec.body(HAND_BODY)
    site = hand.add_site()
    site.name = GRASP_SITE
    site.pos = list(TCP_OFFSET_M)
    site.size = [0.006, 0.006, 0.006]
    site.rgba = [1.0, 0.2, 0.8, 0.6]

    # One inactive weld per graspable object. attach() flips these on after
    # writing the live relative pose into eq_data; see backends.py.
    for name in GRASPABLE:
        eq = spec.add_equality()
        eq.name = f"weld_{name}"
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = HAND_BODY
        eq.name2 = name
        eq.active = False

    # Fixed camera that frames the arm and the whole table.
    cam = world.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = [1.78, -1.30, 1.52]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "table"

    for decorate in DECORATORS:
        decorate(spec)

    return spec


def object_qposadr(model: mujoco.MjModel, name: str) -> int:
    """Index into qpos of an object's free joint."""
    return int(model.joint(int(model.body(name).jntadr[0])).qposadr[0])


def _extend_home_keyframe(model: mujoco.MjModel) -> None:
    """Write the object rest poses into the `home` keyframe.

    Menagerie's `home` keyframe was authored for the bare 9-DOF arm. Adding five
    free bodies grows nq from 9 to 44, and MuJoCo zero-fills the new entries, so
    a plain mj_resetDataKeyframe drops every object at the world origin. We patch
    the keyframe once here, so every later reset (demo mode, --inject-failure)
    restores the real tabletop layout.
    """
    qpos = model.key("home").qpos.copy()
    for name, obj in OBJECTS.items():
        adr = object_qposadr(model, name)
        qpos[adr : adr + 3] = [obj["pos"][0], obj["pos"][1], _rest_z(obj)]
        qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    model.key("home").qpos = qpos


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Return a compiled (model, data) with the arm and objects at `home`."""
    model = build_spec().compile()
    _extend_home_keyframe(model)
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Restore the patched `home` keyframe and refresh derived quantities."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)


def settle(model, data, seconds: float = 1.0) -> None:
    """Step physics so free bodies come to rest before the first observation."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


__all__ = [
    "CAMERA_NAME", "GRASP_SITE", "HAND_BODY", "TCP_OFFSET_M",
    "TABLE_BOUNDS_M", "OBJECTS", "TRAY", "ITEMS", "GRASPABLE",
    "half_height", "footprint",
    "build_spec", "build_scene", "reset", "settle", "panda_xml_path",
    "object_qposadr",
]

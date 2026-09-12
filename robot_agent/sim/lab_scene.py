"""Second scenario: a lab bench. Selected with `--scene lab`; the default
scene is untouched unless activate() is called.

Same arm, same table, same geometry classes as the kitchen scene (so the grasp
and placement tuning carries over), different objects and one extra policy:

- acid_bottle is corrosive: picking or pushing it is classed CAUTION.
- acid_bottle and water_flask must never end up within MIN_SEPARATION_M of each
  other. Checked on every landing spot the safety layer can predict: a place_on
  slot, a place_at point, and a push end point.

Everything is registered through extension points rather than edits to the
built-in action branches: safety.PRECONDITION_HOOKS / CLASSIFY_HOOKS and
llm.SCENARIO_RULES. activate() returns a restore() callable so tests can put the
default scene back.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from ..actions import safety
from ..actions.schema import DIRECTIONS, ActionCall, Safety
from ..agent import llm as _llm
from . import scene

# ----------------------------------------------------------------- objects

LAB_OBJECTS: dict[str, dict] = {
    "red_sample": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.85, 0.15, 0.15, 1.0),
        pos=(0.42, -0.16, 0.0), color="red", fragile=False, graspable=True,
        aliases=("red sample", "the red one", "red sample box"),
    ),
    "green_sample": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.15, 0.70, 0.20, 1.0),
        pos=(0.42, 0.0, 0.0), color="green", fragile=False, graspable=True,
        aliases=("green sample", "the green one", "green sample box"),
    ),
    "blue_sample": dict(
        kind="box", half=(0.02, 0.02, 0.02), rgba=(0.15, 0.30, 0.85, 1.0),
        pos=(0.36, 0.20, 0.0), color="blue", fragile=False, graspable=True,
        aliases=("blue sample", "the blue one", "blue sample box"),
    ),
    "acid_bottle": dict(
        kind="cylinder", half=(0.035, 0.035), rgba=(0.95, 0.80, 0.20, 1.0),
        pos=(0.52, -0.24, 0.0), color="yellow", fragile=False, graspable=True,
        aliases=("the acid", "acid", "the bottle", "the corrosive one"),
    ),
    "water_flask": dict(
        kind="cylinder", half=(0.026, 0.060), rgba=(0.70, 0.85, 1.0, 0.45),
        pos=(0.50, 0.24, 0.0), color="clear", fragile=True, graspable=True,
        aliases=("the flask", "water", "the water flask"),
    ),
}

TRAY_ALIASES = ("the tray", "containment tray", "somewhere safe", "safe place")

# ------------------------------------------------------------------ policy

HAZARDOUS = ("acid_bottle",)
INCOMPATIBLE = (("acid_bottle", "water_flask"),)
MIN_SEPARATION_M = 0.10

LAB_RULES = f"""\
- Lab policy: acid_bottle and water_flask must stay at least {MIN_SEPARATION_M:.2f} m
  apart on the bench. The checker measures every predicted landing spot and
  refuses plans that bring them closer, so plan around it: use the far side of
  the tray, or move one of them away first. acid_bottle is corrosive and is
  handled with caution.
"""


def incompatible_with(name: str) -> list[str]:
    return [b if a == name else a for a, b in INCOMPATIBLE if name in (a, b)]


def separation_problem(state: dict, moving: str, x_m: float, y_m: float) -> str | None:
    """Why `moving` may not end up at (x, y): too close to an incompatible item."""
    for other in incompatible_with(moving):
        o = state["objects"].get(other)
        if o is None or o["held"] or not o["on_table"]:
            continue
        d = math.dist((x_m, y_m), tuple(o["position_m"][:2]))
        if d < MIN_SEPARATION_M:
            return (
                f"{moving} would end up {d:.2f} m from {other}; they must stay at least "
                f"{MIN_SEPARATION_M:.2f} m apart"
            )
    return None


def precondition_hook(call: ActionCall, state: dict) -> list[str]:
    """Separation check on the predicted landing spot of a built-in action."""
    name, args = call.name, call.args
    holding = state["gripper"]["holding"]
    try:
        if name == "place_on" and holding:
            from ..actions.executor import place_on_spot, resolve_place_target  # noqa: PLC0415

            target = resolve_place_target(state, str(args.get("target")), holding)
            if target not in state["objects"]:
                return []
            sx, sy, _ = place_on_spot(state, target, holding)
            sep = separation_problem(state, holding, sx, sy)
        elif name == "place_at" and holding:
            sep = separation_problem(state, holding, float(args["x_m"]), float(args["y_m"]))
        elif name == "push" and args.get("direction") in DIRECTIONS:
            target = str(args.get("object"))
            if target not in state["objects"]:
                return []
            ex, ey = safety.predict_push_end(state, target, args["direction"], float(args["distance_m"]))
            sep = separation_problem(state, target, ex, ey)
        else:
            sep = None
    except (KeyError, TypeError, ValueError):
        return []  # malformed args are reported by the built-in checks
    return [sep] if sep else []


def classify_hook(call: ActionCall, state: dict):
    """Corrosive items are handled as caution; everything else is left to the
    built-in classifier (return None)."""
    name, args = call.name, call.args
    if name == "pick" and args.get("object") in HAZARDOUS:
        return Safety.CAUTION, f"{args['object']} is corrosive; it will be handled gently"
    if name == "push" and args.get("object") in HAZARDOUS:
        return Safety.CAUTION, f"{args['object']} is corrosive and pushing it is hard to control"
    return None


# -------------------------------------------------------------- activation

# -------------------------------------------------------------- decoration
# The chemistry lab from mujoco-scene-editor (assets/scene_editor, MIT): a
# human-scale room of primitives. We use its furniture as a backdrop behind the
# arm and a few of its props as fixed decoration along the far edge of our
# table. Its bench, stool, floor and lights are dropped; ours stay.
CHEM_XML = Path(__file__).resolve().parents[2] / "assets" / "scene_editor" / "scene_chemistry_lab.xml"
BACKDROP_DROP = ("bench", "lab_stool", "light_fixture_1", "light_fixture_2", "fire_extinguisher",
                 "microscope", "balance_scale", "test_tube_rack", "beaker", "erlenmeyer_flask",
                 "graduated_cylinder", "bunsen_burner", "waste_bin", "rolling_cart")
BACKDROP_YAW_DEG = 225.0                 # their back counter ends up behind the arm, back-left of the camera
BACKDROP_POS = (1.4, -1.4, 0.0)
PROPS = {                                # fixed props on our table's far strip (x >= 0.70 is beyond the tray)
    "microscope": (0.70, 0.31),
    "erlenmeyer_flask": (0.76, -0.10),
    "beaker": (0.76, -0.26),
    "bunsen_burner": (0.76, 0.12),
}


def _chem_spec(keep: set[str] | None = None, drop: tuple[str, ...] = ()):
    """A fresh copy of the chemistry spec with bodies removed and no floor/lights."""
    import mujoco  # noqa: PLC0415

    spec = mujoco.MjSpec.from_file(CHEM_XML.as_posix())
    for body in list(spec.bodies):
        if not body.name or body.name == "world":
            continue
        if keep is not None and body.name not in keep:
            spec.delete(body)
        elif body.name in drop:
            spec.delete(body)
    for geom in list(spec.worldbody.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    return spec


def decorate(spec) -> None:
    """Attach the backdrop and the props into the live scene spec."""
    import math  # noqa: PLC0415

    if not CHEM_XML.exists():
        return
    world = spec.worldbody
    half = math.radians(BACKDROP_YAW_DEG) / 2
    frame = world.add_frame(pos=list(BACKDROP_POS), quat=[math.cos(half), 0, 0, math.sin(half)])
    spec.attach(_chem_spec(drop=BACKDROP_DROP), prefix="lab_", frame=frame)
    for name, (x, y) in PROPS.items():
        child = _chem_spec(keep={name})
        for body in child.bodies:
            if body.name == name:
                for joint in list(body.joints):
                    child.delete(joint)          # props are fixed decoration
        f = world.add_frame(pos=[x, y, scene.TABLE_BOUNDS_M["top_z"]])
        spec.attach(child, prefix=f"prop_{name}_", frame=f)


def activate():
    """Swap the lab objects into the live scene registries and install the
    policy. Returns restore(), which puts the default scene back."""
    saved = dict(
        objects=copy.deepcopy(scene.OBJECTS),
        items=copy.deepcopy(scene.ITEMS),
        graspable=scene.GRASPABLE,
        stackable=safety.STACKABLE_TARGETS,
    )

    # Registries are mutated in place: world_state, backends and build_spec all
    # hold references to these same dicts.
    scene.OBJECTS.clear()
    scene.OBJECTS.update(copy.deepcopy(LAB_OBJECTS))
    scene.GRASPABLE = tuple(n for n, o in scene.OBJECTS.items() if o["graspable"])
    tray = dict(saved["items"]["tray"], aliases=TRAY_ALIASES)
    scene.ITEMS.clear()
    scene.ITEMS.update(scene.OBJECTS)
    scene.ITEMS["tray"] = tray

    safety.STACKABLE_TARGETS = ("tray", "red_sample", "green_sample", "blue_sample")
    if decorate not in scene.DECORATORS:
        scene.DECORATORS.append(decorate)
    safety.PRECONDITION_HOOKS.append(precondition_hook)
    safety.CLASSIFY_HOOKS.append(classify_hook)
    _llm.SCENARIO_RULES.append(LAB_RULES)

    def restore():
        scene.OBJECTS.clear()
        scene.OBJECTS.update(saved["objects"])
        scene.ITEMS.clear()
        scene.ITEMS.update(saved["items"])
        scene.GRASPABLE = saved["graspable"]
        safety.STACKABLE_TARGETS = saved["stackable"]
        if decorate in scene.DECORATORS:
            scene.DECORATORS.remove(decorate)
        if precondition_hook in safety.PRECONDITION_HOOKS:
            safety.PRECONDITION_HOOKS.remove(precondition_hook)
        if classify_hook in safety.CLASSIFY_HOOKS:
            safety.CLASSIFY_HOOKS.remove(classify_hook)
        if LAB_RULES in _llm.SCENARIO_RULES:
            _llm.SCENARIO_RULES.remove(LAB_RULES)

    return restore


__all__ = [
    "LAB_OBJECTS", "HAZARDOUS", "INCOMPATIBLE", "MIN_SEPARATION_M", "LAB_RULES",
    "separation_problem", "incompatible_with", "precondition_hook", "classify_hook",
    "activate",
]

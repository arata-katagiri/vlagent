"""Plan validation, execution, and postcondition verification."""

from __future__ import annotations

import copy
import math
from typing import Callable

from ..sim.backends import ArmBackend
from .safety import check_preconditions, classify
from .schema import (
    APPROACH_HEIGHT_M,
    DIRECTION_VECTORS,
    LIFT_HEIGHT_M,
    PLACEMENT_TOLERANCE_M,
    ActionCall,
    ActionResult,
    Safety,
)

# How far an object must rise off the table for a pick to count as verified.
MIN_LIFT_M = 0.03
# A push must achieve at least this fraction of its predicted travel.
MIN_PUSH_FRACTION = 0.5


# Clearance between two items sharing a surface, in metres.
SIDE_BY_SIDE_CLEARANCE_M = 0.045


def place_on_spot(state: dict, target: str, held: str) -> list[float]:
    """Where to set `held` down on `target`.

    Dead centre when the surface is empty, otherwise the first free slot in a
    small grid across the target's footprint. Without this, clearing two blocks
    onto the tray drops the second one on top of the first, because both
    placements aim at the same point.

    Both the executor and the verifier call this with the *pre-action* state, so
    they agree on where the object was supposed to land.
    """
    tobj = state["objects"][target]
    tx, ty, tz = tobj["position_m"]
    thx, thy, thz = tobj["half_extent_m"]
    top_z = tz + thz
    drop_z = top_z + state["objects"][held]["half_extent_m"][2]

    occupied = [
        state["objects"][n]["position_m"][:2]
        for n in tobj["supporting"]
        if n in state["objects"]
    ]
    if not occupied:
        return [tx, ty, drop_z]

    hx, hy, _ = state["objects"][held]["half_extent_m"]
    span_x = max(0.0, thx - hx)
    span_y = max(0.0, thy - hy)
    candidates = [(0.0, 0.0)]
    for sx in (-1.0, 1.0):
        candidates.append((sx * span_x, 0.0))
    for sy in (-1.0, 1.0):
        candidates.append((0.0, sy * span_y))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            candidates.append((sx * span_x, sy * span_y))

    for ox, oy in candidates:
        x, y = tx + ox, ty + oy
        if all(math.dist([x, y], o) >= SIDE_BY_SIDE_CLEARANCE_M for o in occupied):
            return [x, y, drop_z]
    return [tx, ty, drop_z]


def _apply_effects(call: ActionCall, state: dict) -> dict:
    """Symbolically advance the state as if the call succeeded.

    Used only by validate_plan, so that "pick A, then place A on the tray"
    validates even though A is not held when step 1 is checked. This is a
    deliberately coarse model: it tracks what the preconditions care about
    (what is held, what supports what, where things are) and nothing else.
    """
    s = copy.deepcopy(state)
    objects, gripper = s["objects"], s["gripper"]
    name, args = call.name, call.args

    if name == "pick":
        target = args["object"]
        if target in objects:
            obj = objects[target]
            parent = obj["on_top_of"]
            if parent and parent in objects and target in objects[parent]["supporting"]:
                objects[parent]["supporting"].remove(target)
            obj.update(held=True, on_top_of=None, on_table=False)
            obj["position_m"] = [
                obj["position_m"][0],
                obj["position_m"][1],
                obj["position_m"][2] + LIFT_HEIGHT_M,
            ]
            gripper["holding"] = target

    elif name == "place_on":
        held, target = gripper["holding"], args["target"]
        if held and target in objects:
            spot = place_on_spot(s, target, held)
            objects[held].update(
                held=False, on_top_of=target, on_table=True, position_m=spot,
            )
            objects[target]["supporting"].append(held)
            gripper["holding"] = None

    elif name == "place_at":
        held = gripper["holding"]
        if held:
            x, y = float(args["x_m"]), float(args["y_m"])
            t = s["table_bounds_m"]
            on = t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]
            objects[held].update(
                held=False, on_top_of=None, on_table=on,
                position_m=[x, y, t["top_z"] + 0.02 if on else 0.0],
            )
            gripper["holding"] = None

    elif name == "push":
        target = args["object"]
        if target in objects:
            dx, dy = DIRECTION_VECTORS.get(args["direction"], (0.0, 0.0))
            d = float(args["distance_m"])
            pos = objects[target]["position_m"]
            nx, ny = pos[0] + dx * d, pos[1] + dy * d
            t = s["table_bounds_m"]
            on = t["x_min"] <= nx <= t["x_max"] and t["y_min"] <= ny <= t["y_max"]
            objects[target].update(position_m=[nx, ny, pos[2] if on else 0.0], on_table=on)

    return s


def validate_plan(plan: list[ActionCall], state: dict) -> list[str]:
    """Symbolically check the whole plan in order. Empty list means valid."""
    problems: list[str] = []
    current = state
    for i, call in enumerate(plan, start=1):
        for reason in check_preconditions(call, current):
            problems.append(f"step {i} ({call.name}): {reason}")
        current = _apply_effects(call, current)
    return problems


def _verify(call: ActionCall, before: dict, after: dict) -> tuple[bool, str]:
    """Check the postcondition of a call against ground truth."""
    name, args = call.name, call.args
    objects = after["objects"]

    if name == "pick":
        target = args["object"]
        obj = objects.get(target)
        if obj is None:
            return False, f"{target} vanished from the scene"
        if after["gripper"]["holding"] != target:
            return False, f"the gripper is not holding {target}"
        rise = obj["position_m"][2] - before["objects"][target]["position_m"][2]
        if rise < MIN_LIFT_M:
            return False, f"{target} only rose {rise:.3f} m; it was not lifted clear"
        return True, f"holding {target}, lifted {rise:.3f} m"

    if name == "place_on":
        target = args["target"]
        held = before["gripper"]["holding"]
        if after["gripper"]["holding"] is not None:
            return False, f"the gripper is still holding {after['gripper']['holding']}"
        obj = objects.get(held)
        if obj is None:
            return False, f"{held} vanished from the scene"
        want = place_on_spot(before, target, held)
        err = math.dist(obj["position_m"][:2], want[:2])
        if obj["on_top_of"] != target:
            landed = obj["on_top_of"] or "the table"
            return False, f"{held} came to rest on {landed}, not on {target}"
        if err > PLACEMENT_TOLERANCE_M:
            return False, f"{held} is {err:.3f} m from where it should have landed on {target}"
        return True, f"{held} rests on {target}, {err:.3f} m from the intended spot"

    if name == "place_at":
        held = before["gripper"]["holding"]
        obj = objects.get(held)
        if obj is None:
            return False, f"{held} vanished from the scene"
        if after["gripper"]["holding"] is not None:
            return False, f"the gripper is still holding {held}"
        err = math.dist(obj["position_m"][:2], [float(args["x_m"]), float(args["y_m"])])
        if err > PLACEMENT_TOLERANCE_M:
            return False, f"{held} came to rest {err:.3f} m from the requested point"
        return True, f"{held} placed within {err:.3f} m of the requested point"

    if name == "push":
        target = args["object"]
        dx, dy = DIRECTION_VECTORS[args["direction"]]
        want = float(args["distance_m"])
        b = before["objects"][target]["position_m"]
        a = objects[target]["position_m"]
        travelled = (a[0] - b[0]) * dx + (a[1] - b[1]) * dy
        if travelled < MIN_PUSH_FRACTION * want:
            return False, f"{target} only moved {travelled:.3f} m of the {want:.3f} m requested"
        return True, f"{target} moved {travelled:.3f} m"

    if name == "home":
        return True, "arm returned home"

    return True, "no postcondition to verify"


def execute(
    call: ActionCall, backend: ArmBackend, get_state: Callable[[], dict]
) -> ActionResult:
    """Run one step, then verify its postcondition against ground truth."""
    # Snapshot, do not alias: a backend may hand back the same dict it mutates
    # (MockBackend does), and then every before/after delta would measure zero.
    before = copy.deepcopy(get_state())

    problems = check_preconditions(call, before)
    if problems:
        return ActionResult(False, "; ".join(problems), before)

    try:
        ok, reason = _run(call, backend, before)
    except Exception as exc:  # a backend failure must not kill the loop
        return ActionResult(False, f"{call.name} raised {type(exc).__name__}: {exc}", get_state())

    after = get_state()
    if not ok:
        return ActionResult(False, reason, after)

    verified, detail = _verify(call, before, after)
    return ActionResult(verified, detail, after)


def _run(call: ActionCall, backend: ArmBackend, state: dict) -> tuple[bool, str]:
    """Drive the backend through the motion for one call."""
    name, args = call.name, call.args

    if name == "pick":
        target = args["object"]
        pos = list(state["objects"][target]["position_m"])
        if not backend.move_to([pos[0], pos[1], pos[2] + APPROACH_HEIGHT_M]):
            return False, f"could not reach the approach pose above {target}"
        backend.open_gripper()
        if not backend.move_to(pos):
            return False, f"could not descend onto {target}"
        backend.close_gripper()
        backend.attach(target)
        if not backend.move_to([pos[0], pos[1], pos[2] + LIFT_HEIGHT_M]):
            return False, f"could not lift {target}"
        return True, ""

    if name == "place_on":
        target = args["target"]
        held = state["gripper"]["holding"]
        drop = place_on_spot(state, target, held)
        if not backend.move_to([drop[0], drop[1], drop[2] + APPROACH_HEIGHT_M]):
            return False, f"could not reach the approach pose above {target}"
        if not backend.move_to(drop):
            return False, f"could not descend onto {target}"
        backend.detach()
        backend.open_gripper()
        backend.move_to([drop[0], drop[1], drop[2] + APPROACH_HEIGHT_M])
        return True, f"released {held}"

    if name == "place_at":
        x, y = float(args["x_m"]), float(args["y_m"])
        z = state["table_bounds_m"]["top_z"] + 0.02
        if not backend.move_to([x, y, z + APPROACH_HEIGHT_M]):
            return False, "could not reach the approach pose above the target point"
        if not backend.move_to([x, y, z]):
            return False, "could not descend to the target point"
        backend.detach()
        backend.open_gripper()
        backend.move_to([x, y, z + APPROACH_HEIGHT_M])
        return True, ""

    if name == "push":
        target = args["object"]
        dx, dy = DIRECTION_VECTORS[args["direction"]]
        d = float(args["distance_m"])
        pos = list(state["objects"][target]["position_m"])
        start = [pos[0] - dx * 0.06, pos[1] - dy * 0.06, pos[2]]
        backend.close_gripper()
        if not backend.move_to([start[0], start[1], start[2] + APPROACH_HEIGHT_M]):
            return False, f"could not reach the approach pose behind {target}"
        if not backend.move_to(start):
            return False, f"could not get behind {target}"
        if not backend.move_to([pos[0] + dx * d, pos[1] + dy * d, pos[2]]):
            return False, f"the push of {target} did not complete"
        return True, ""

    if name == "home":
        return (True, "") if backend.home() else (False, "the arm did not reach its home pose")

    if name == "ask_user":
        return True, ""

    return False, f"unknown action '{name}'"


__all__ = [
    "validate_plan", "execute", "classify", "check_preconditions", "place_on_spot",
    "ActionCall", "ActionResult", "Safety",
]

"""Deterministic preconditions and irreversibility prediction.

Nothing here consults the LLM. The planner proposes; this module decides. A model
cannot talk its way past a precondition or downgrade a classification, because it
never sees this code run -- it only ever receives the verdict.

Everything operates on the plain state dict from world_state.get_world_state, so
this module has no simulator dependency and is tested with hand-written dicts.
"""

from __future__ import annotations

import math

from .schema import (
    ACTIONS,
    DIRECTION_VECTORS,
    DIRECTIONS,
    MAX_PUSH_DISTANCE_M,
    MAX_REACH_M,
    MIN_REACH_M,
    ActionCall,
    Safety,
)

# --- policy switch -------------------------------------------------------
#
# Two different kinds of check live in this module:
#
#   correctness - the action is impossible or malformed (no such object, gripper
#                 already full, argument missing, target out of reach). Disabling
#                 these would just crash the executor, so they always run.
#   policy      - the action is possible but we would rather it did not happen
#                 (stacking on something fragile, an object that will not balance,
#                 anything irreversible). These are what --no-safety turns off.
#
# With policy off, classify() reports SAFE for everything, so nothing is gated
# behind a confirmation and physics decides what happens.

_POLICY_ENABLED = True


def set_policy_enabled(enabled: bool) -> None:
    """Enable or disable the policy checks. Called once at startup."""
    global _POLICY_ENABLED
    _POLICY_ENABLED = enabled


def policy_enabled() -> bool:
    return _POLICY_ENABLED


# A surface must be flat and stable to stack on. Anything else is refused.
STACKABLE_TARGETS = ("tray", "red_block", "green_block", "blue_block")


def _obj(state: dict, name: str) -> dict | None:
    return state["objects"].get(name)


def reachable(x_m: float, y_m: float) -> bool:
    """Whether a tabletop point is inside the arm's annular workspace."""
    r = math.hypot(x_m, y_m)
    return MIN_REACH_M <= r <= MAX_REACH_M


def off_table(state: dict, x_m: float, y_m: float) -> bool:
    t = state["table_bounds_m"]
    return not (t["x_min"] <= x_m <= t["x_max"] and t["y_min"] <= y_m <= t["y_max"])


def predict_push_end(state: dict, name: str, direction: str, distance_m: float) -> tuple[float, float]:
    """Where an object ends up if the push runs to completion."""
    dx, dy = DIRECTION_VECTORS[direction]
    pos = state["objects"][name]["position_m"]
    return pos[0] + dx * distance_m, pos[1] + dy * distance_m


def overhang_m(state: dict, x_m: float, y_m: float) -> float:
    """How far a point lies past the nearest table edge. Zero if on the table."""
    t = state["table_bounds_m"]
    return max(
        0.0,
        t["x_min"] - x_m,
        x_m - t["x_max"],
        t["y_min"] - y_m,
        y_m - t["y_max"],
    )


# How far a stacked object may overhang its support on each side, in metres.
STACK_OVERHANG_M = 0.01


def stacking_overhang(state: dict, held: str, target: str) -> tuple[float, float] | None:
    """Return (held width, target width) if `held` would not balance on `target`.

    None means it fits. Without this, "put the cup on the red block" passes every
    other precondition and the cup topples off a support less than half its width.
    """
    held_obj = state["objects"].get(held)
    target_obj = state["objects"].get(target)
    if not held_obj or not target_obj:
        return None
    hx, hy, _ = held_obj.get("half_extent_m", (0.0, 0.0, 0.0))
    tx, ty, _ = target_obj.get("half_extent_m", (0.0, 0.0, 0.0))
    if not (tx or ty):
        return None
    if hx <= tx + STACK_OVERHANG_M and hy <= ty + STACK_OVERHANG_M:
        return None
    return 2 * max(hx, hy), 2 * max(tx, ty)


def check_preconditions(call: ActionCall, state: dict) -> list[str]:
    """Return human-readable reasons the call cannot run now. Empty means OK."""
    name, args = call.name, call.args
    if name not in ACTIONS:
        return [f"unknown action '{name}'"]

    required, _ = ACTIONS[name]
    missing = [a for a in required if a not in args]
    if missing:
        return [f"{name} is missing required argument(s): {', '.join(missing)}"]

    holding = state["gripper"]["holding"]
    problems: list[str] = []

    if name == "pick":
        target = args["object"]
        obj = _obj(state, target)
        if obj is None:
            problems.append(f"there is no object called '{target}'")
        else:
            if not obj.get("graspable", True):
                problems.append(f"{target} is fixed in place and cannot be picked up")
            if obj["supporting"] and _POLICY_ENABLED:
                on_top = ", ".join(obj["supporting"])
                problems.append(f"{target} has {on_top} on top of it")
            if not obj["on_table"]:
                problems.append(f"{target} is not on the table")
        if holding is not None:
            problems.append(f"the gripper is already holding {holding}")

    elif name == "place_on":
        target = args["target"]
        obj = _obj(state, target)
        if holding is None:
            problems.append("the gripper is not holding anything")
        if obj is None:
            problems.append(f"there is no object called '{target}'")
        else:
            if target == holding:
                problems.append(f"cannot place {target} on itself")
            elif _POLICY_ENABLED:
                if obj["fragile"]:
                    problems.append(
                        f"{target} is fragile and is not a stable surface to stack on"
                    )
                elif target not in STACKABLE_TARGETS:
                    problems.append(f"{target} is not a flat surface to stack on")
                elif holding is not None:
                    fit = stacking_overhang(state, holding, target)
                    if fit is not None:
                        held_w, target_w = fit
                        problems.append(
                            f"{holding} is {held_w:.2f} m across and {target} is only "
                            f"{target_w:.2f} m across; it would not balance"
                        )

    elif name == "place_at":
        if holding is None:
            problems.append("the gripper is not holding anything")
        x, y = float(args["x_m"]), float(args["y_m"])
        if not reachable(x, y):
            problems.append(
                f"({x:.2f}, {y:.2f}) m is outside the arm's reach "
                f"({MIN_REACH_M}-{MAX_REACH_M} m from the base)"
            )

    elif name == "push":
        target = args["object"]
        obj = _obj(state, target)
        direction = args["direction"]
        try:
            distance = float(args["distance_m"])
        except (TypeError, ValueError):
            distance = float("nan")
        if obj is None:
            problems.append(f"there is no object called '{target}'")
        elif not obj["on_table"]:
            problems.append(f"{target} is not on the table")
        if holding is not None:
            problems.append(f"the gripper is holding {holding}; put it down before pushing")
        if direction not in DIRECTIONS:
            problems.append(f"direction must be one of {', '.join(DIRECTIONS)}, got '{direction}'")
        if not (0.0 < distance <= MAX_PUSH_DISTANCE_M):
            problems.append(f"distance_m must be greater than 0 and at most {MAX_PUSH_DISTANCE_M}")

    return problems


def classify(call: ActionCall, state: dict) -> tuple[Safety, str]:
    """Classify a call and give the concrete reason shown to the user.

    The reason is written to be readable straight out of a confirmation prompt,
    so it names the object and the measured consequence.
    """
    name, args = call.name, call.args

    if not _POLICY_ENABLED:
        return Safety.SAFE, "safety policy disabled (--no-safety)"

    if name == "pick":
        obj = _obj(state, args.get("object", ""))
        if obj and obj["fragile"]:
            return Safety.CAUTION, f"{args['object']} is fragile; it will be handled gently"
        return Safety.SAFE, "picking up an object is reversible"

    if name == "place_on":
        return Safety.SAFE, f"placing on {args.get('target')} is reversible"

    if name == "place_at":
        x, y = float(args["x_m"]), float(args["y_m"])
        over = overhang_m(state, x, y)
        held = state["gripper"]["holding"]
        what = held or "the held object"
        if over > 0.0:
            return (
                Safety.IRREVERSIBLE,
                f"({x:.2f}, {y:.2f}) m is {over:.2f} m past the table edge; {what} would fall",
            )
        t = state["table_bounds_m"]
        margin = min(x - t["x_min"], t["x_max"] - x, y - t["y_min"], t["y_max"] - y)
        if margin <= 0.06:
            return Safety.CAUTION, f"that spot is only {margin:.2f} m from the table edge"
        return Safety.SAFE, "the target spot is well inside the table"

    if name == "push":
        target = args["object"]
        obj = _obj(state, target)
        end_x, end_y = predict_push_end(state, target, args["direction"], float(args["distance_m"]))
        over = overhang_m(state, end_x, end_y)
        if over > 0.0:
            return (
                Safety.IRREVERSIBLE,
                f"predicted final position is {over:.2f} m past the table edge; "
                f"{target} would fall",
            )
        if obj and obj["fragile"]:
            return Safety.CAUTION, f"{target} is fragile and pushing it is hard to control"
        return Safety.CAUTION, f"{target} would end up at ({end_x:.2f}, {end_y:.2f}) m"

    return Safety.SAFE, "no physical consequence"


__all__ = [
    "check_preconditions", "classify", "reachable", "off_table", "stacking_overhang",
    "set_policy_enabled", "policy_enabled",
    "overhang_m", "predict_push_end", "STACKABLE_TARGETS",
]

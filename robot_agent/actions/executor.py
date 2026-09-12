"""Plan validation, execution, and postcondition verification."""

from __future__ import annotations

import copy
import math
from typing import Callable

from ..sim.backends import ArmBackend
from .safety import check_preconditions, classify, holding_of, policy_enabled
from .schema import (
    ACTIONS,
    required_args,
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
# How far below an object's top the gripper closes, in metres.
GRASP_DEPTH_M = 0.025
# Approach, lift and retreat only need to clear obstacles, not land precisely.
CLEARANCE_TOLERANCE_M = 0.025
# Below this, a plan's net effect counts as no change at all.
NO_OP_TOLERANCE_M = 0.01


# Clearance between two items sharing a surface, centre to centre, in metres.
# Wide enough for the fingers around a held block to clear a neighbour.
SIDE_BY_SIDE_CLEARANCE_M = 0.065
# A descent that stalls this close to its target (something in the way, a
# neighbour brushing the hand) still counts: release or grasp from there and let
# physics settle, then let the verifier judge the outcome.
DESCENT_SLACK_M = 0.04
# Height above the table top the empty gripper travels at between actions.
SAFE_TRAVEL_M = 0.15
# Push contact: this far above the object's bottom, so tall or round things
# slide instead of tipping; and this far behind the object's face at the start.
PUSH_CONTACT_Z_M = 0.02
PUSH_STANDOFF_M = 0.03


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

    # First slot that meets the clearance wins. When none does, take the slot
    # farthest from everything already there, so a crowded surface degrades to
    # "as spread out as possible" instead of dropping the newcomer on whatever
    # sits at the centre.
    best, best_gap = None, -1.0
    for ox, oy in candidates:
        x, y = tx + ox, ty + oy
        gap = min(math.dist([x, y], o) for o in occupied)
        if gap >= SIDE_BY_SIDE_CLEARANCE_M:
            return [x, y, drop_z]
        if gap > best_gap:
            best, best_gap = [x, y, drop_z], gap
    return best or [tx, ty, drop_z]


def resolve_place_target(state: dict, target: str, held: str | None) -> str:
    """Placing onto something that already carries an object means the top of
    that stack, unless the surface has room beside what is there (the tray)."""
    seen: set[str] = set()
    while target in state["objects"] and target not in seen:
        seen.add(target)
        tobj = state["objects"][target]
        stacked = [n for n in tobj["supporting"] if n in state["objects"] and n != held]
        if not stacked:
            return target
        hx, hy = 0.0, 0.0
        if held in state["objects"]:
            hx, hy, _ = state["objects"][held]["half_extent_m"]
        thx, thy, _ = tobj["half_extent_m"]
        if thx - hx >= SIDE_BY_SIDE_CLEARANCE_M / 2 or thy - hy >= SIDE_BY_SIDE_CLEARANCE_M / 2:
            return target
        target = stacked[-1]
    return target


def _arm_record(s: dict, arm: str | None) -> dict:
    """The mutable per-arm record to update, or the single gripper record."""
    arms = s.get("arms")
    if not arms:
        return s["gripper"]
    if arm in arms:
        return arms[arm]
    return arms[next(iter(arms))]


def _resync_gripper(s: dict) -> None:
    """Refresh the compatibility "gripper" view after mutating an arm."""
    arms = s.get("arms")
    if not arms:
        return
    busy = next((k for k in arms if arms[k]["holding"]), next(iter(arms)))
    s["gripper"] = dict(arms[busy])


def _apply_effects(call: ActionCall, state: dict) -> dict:
    """Symbolically advance the state as if the call succeeded.

    Used only by validate_plan, so that "pick A, then place A on the tray"
    validates even though A is not held when step 1 is checked. This is a
    deliberately coarse model: it tracks what the preconditions care about
    (what is held, what supports what, where things are) and nothing else.
    """
    s = copy.deepcopy(state)
    name, args = call.name, call.args
    objects = s["objects"]
    gripper = _arm_record(s, args.get("arm"))

    # Defensive throughout: a model can emit any shape, and a malformed step
    # must fail validation with a readable message, never crash the validator.
    if name == "pick":
        target = args.get("object")
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
            _resync_gripper(s)

    elif name == "place_on":
        held, target = gripper["holding"], args.get("target")
        if held and target in objects:
            target = resolve_place_target(s, target, held)
            spot = place_on_spot(s, target, held)
            objects[held].update(
                held=False, on_top_of=target, on_table=True, position_m=spot,
            )
            objects[target]["supporting"].append(held)
            gripper["holding"] = None
            _resync_gripper(s)

    elif name == "place_at":
        held = gripper["holding"]
        if held and "x_m" in args and "y_m" in args:
            try:
                x, y = float(args["x_m"]), float(args["y_m"])
            except (TypeError, ValueError):
                return s
            t = s["table_bounds_m"]
            on = t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]
            objects[held].update(
                held=False, on_top_of=None, on_table=on,
                position_m=[x, y, t["top_z"] + 0.02 if on else 0.0],
            )
            gripper["holding"] = None
            _resync_gripper(s)

    elif name == "push":
        target = args.get("object")
        if target in objects:
            dx, dy = DIRECTION_VECTORS.get(args.get("direction", ""), (0.0, 0.0))
            try:
                d = float(args.get("distance_m", 0.0))
            except (TypeError, ValueError):
                return s
            pos = objects[target]["position_m"]
            nx, ny = pos[0] + dx * d, pos[1] + dy * d
            t = s["table_bounds_m"]
            on = t["x_min"] <= nx <= t["x_max"] and t["y_min"] <= ny <= t["y_max"]
            objects[target].update(position_m=[nx, ny, pos[2] if on else 0.0], on_table=on)

    return s


def classify_plan(plan: list[ActionCall], state: dict) -> list[tuple[Safety, str]]:
    """Classify every step against the state as it will be when that step runs.

    Classifying all steps against the initial state produces vague reasons -- a
    place_at in step 2 does not yet know which object the gripper will be
    holding, so the prompt says "the held object" instead of naming the cup.
    """
    levels: list[tuple[Safety, str]] = []
    current = state
    for call in plan:
        levels.append(classify(call, current))
        current = _apply_effects(call, current)
    return levels


def _is_skill(name: str) -> bool:
    from ..skills.registry import REGISTRY  # noqa: PLC0415

    return name in REGISTRY


_LAST_SKILL_NOTE = ""


def _run_skill(call: ActionCall, backend, get_state) -> tuple[bool, str]:
    """Run a learned skill for real, with the same measurement as rehearsal."""
    global _LAST_SKILL_NOTE
    from ..skills.body import Arm  # noqa: PLC0415
    from ..skills.registry import REGISTRY  # noqa: PLC0415
    from ..skills.rehearse import classify as classify_metrics, measure, observe, resolve, verify_effect  # noqa: PLC0415
    from ..skills.sandbox import compile_skill  # noqa: PLC0415

    skill = REGISTRY.get(call.name)
    fns = compile_skill(skill.code)
    handle, skill_args = resolve(backend, call.args)
    before = observe(backend, get_state)
    arm = Arm(handle, get_state, registry=REGISTRY)
    fns["run"](arm, **skill_args)
    after = observe(backend, get_state)
    after["touched"] = sorted(arm.touched)
    skill.uses += 1
    mentioned = {v for v in skill_args.values() if isinstance(v, str) and v in before["objects"]}
    metrics = measure(before, after, mentioned, touched=arm.touched)
    if skill.trust == "human":
        strays = [n for n in metrics["off_table"] if n not in mentioned]
        if strays:
            return False, f"{', '.join(strays)} left the table and was not part of the request"
        ok, why, verified = True, "human-verified skill; the world did not check the effect", False
        fns["check"] = None
    else:
        ok, why, verified = verify_effect(skill, skill_args, before, after, metrics, mentioned)
    if not ok:
        return False, f"effect not achieved: {why}"
    if policy_enabled():
        from ..skills.rules import RULES  # noqa: PLC0415

        broken = RULES.check(before, after, call.name, call.args)
        if broken:
            return False, "; ".join(broken)
    note = why if verified else ""
    if fns["check"] is not None:
        result = fns["check"](before, after, **skill_args)
        c_ok, c_why = (result if isinstance(result, tuple) else (bool(result), ""))
        if not c_ok:
            return False, f"the skill's own check failed: {c_why or skill.effect}"
        note = note or c_why or skill.effect
    tag = " [world-verified]" if verified else (" [human-verified]" if skill.trust == "human" else " [self-reported]")
    _LAST_SKILL_NOTE = (note or classify_metrics(metrics)[1]) + tag
    return True, ""


def is_no_op(plan: list[ActionCall], state: dict) -> bool:
    if any(_is_skill(s.name) for s in plan):
        return False
    """Whether the plan would leave the world materially unchanged.

    Picking an object up and setting it back down where it already was passes
    every precondition, so without this check the agent reports "Done, every
    step was verified" for a plan that achieved nothing. Plans made only of
    home/ask_user are exempt: those are legitimately not about moving anything.
    """
    if not plan or all(c.name in ("home", "ask_user") for c in plan):
        return False

    final = state
    for call in plan:
        final = _apply_effects(call, final)

    if state.get("arms"):
        for key, arm in state["arms"].items():
            if final["arms"][key]["holding"] != arm["holding"]:
                return False
    elif final["gripper"]["holding"] != state["gripper"]["holding"]:
        return False
    for name, obj in state["objects"].items():
        after = final["objects"].get(name)
        if after is None:
            return False
        if math.dist(obj["position_m"], after["position_m"]) > NO_OP_TOLERANCE_M:
            return False
    return True


def validate_plan(plan: list[ActionCall], state: dict) -> list[str]:
    """Symbolically check the whole plan in order. Empty list means valid."""
    problems: list[str] = []
    current = state
    for i, call in enumerate(plan, start=1):
        for reason in check_preconditions(call, current):
            problems.append(f"step {i} ({call.name}): {reason}")
        current = _apply_effects(call, current)

    if not problems and policy_enabled() and is_no_op(plan, state):
        problems.append(
            "this plan would leave the scene exactly as it is; it does not achieve "
            "anything. Either propose a plan that actually changes something, or "
            "use ask_user to say what you cannot do."
        )
    return problems


def _verify(call: ActionCall, before: dict, after: dict) -> tuple[bool, str]:
    if _is_skill(call.name):
        return True, _LAST_SKILL_NOTE or "learned skill ran; effect measured during the run"
    """Check the postcondition of a call against ground truth."""
    name, args = call.name, call.args
    objects = after["objects"]
    arm = args.get("arm")
    who = f"arm {arm}" if arm else "the gripper"
    missing = [a for a in required_args(name) if a not in args]
    if missing:
        return False, f"{name} is missing argument(s): {', '.join(missing)}"

    if name == "pick":
        target = args["object"]
        obj = objects.get(target)
        if obj is None:
            return False, f"{target} vanished from the scene"
        if holding_of(after, arm) != target:
            return False, f"{who} is not holding {target}"
        rise = obj["position_m"][2] - before["objects"][target]["position_m"][2]
        if rise < MIN_LIFT_M:
            return False, f"{target} only rose {rise:.3f} m; it was not lifted clear"
        return True, f"holding {target}, lifted {rise:.3f} m"

    if name == "place_on":
        held = holding_of(before, arm)
        target = resolve_place_target(before, args["target"], held)
        still = holding_of(after, arm)
        if still is not None:
            return False, f"{who} is still holding {still}"
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
        held = holding_of(before, arm)
        obj = objects.get(held)
        if obj is None:
            return False, f"{held} vanished from the scene"
        if holding_of(after, arm) is not None:
            return False, f"{who} is still holding {held}"
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
        return True, f"{who} returned home"

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

    # Route the step to the arm that was planned for it. A backend without
    # for_arm is single-arm and takes every step itself.
    router = getattr(backend, "for_arm", None)
    arm_backend = router(call.args.get("arm")) if router else backend

    try:
        ok, reason = _run(call, arm_backend, before, get_state)
    except Exception as exc:  # a backend failure must not kill the loop
        return ActionResult(False, f"{call.name} raised {type(exc).__name__}: {exc}", get_state())

    after = get_state()
    if not ok:
        return ActionResult(False, reason, after)

    verified, detail = _verify(call, before, after)
    return ActionResult(verified, detail, after)


def _grasp_offset_z(state: dict, held: str | None, arm: str | None = None) -> float:
    """How far above the held object's centre the gripper is holding it.

    Measured live rather than assumed, so it works for any backend and any grasp
    height. Placing an object means putting the *gripper* at the object's
    intended centre plus this offset.
    """
    if not held or held not in state["objects"]:
        return 0.0
    arms = state.get("arms")
    src = arms[arm] if arms and arm in arms else state["gripper"]
    return src["position_m"][2] - state["objects"][held]["position_m"][2]



def _descend(backend, target, get_state, what: str) -> tuple[bool, str]:
    """Move down onto `target`; tolerate a stall within DESCENT_SLACK_M.

    A neighbour brushing the hand, or a surface a few millimetres higher than
    modelled, stops the servo short of its tolerance. Failing the whole step for
    that is brittle: releasing or grasping from a couple of centimetres up works
    in practice, and the postcondition check still decides whether it did.
    """
    if backend.move_to(target):
        return True, ""
    if get_state is None:
        return False, f"could not descend {what}"
    g = get_state()["gripper"]["position_m"]
    short = g[2] - target[2]
    lateral = math.dist(g[:2], target[:2])
    if -0.005 <= short <= DESCENT_SLACK_M and lateral <= PLACEMENT_TOLERANCE_M:
        return True, f"stopped {short:.3f} m short {what}; continuing"
    return False, f"could not descend {what} (stopped {short:.3f} m short, {lateral:.3f} m off)"


def _lift_clear(backend, get_state, state: dict, arm: str | None = None) -> None:
    """Start every action by going straight up if the gripper is low, so the
    traverse to the next approach pose does not sweep through the scene."""
    if get_state is None:
        return
    now = get_state()
    arms = now.get("arms")
    g = (arms[arm] if arms and arm in arms else now["gripper"])["position_m"]
    held = holding_of(state, arm)
    hang = 0.0
    if held in state["objects"]:
        hang = _grasp_offset_z(state, held, arm) + state["objects"][held]["half_extent_m"][2]
    safe_z = state["table_bounds_m"]["top_z"] + SAFE_TRAVEL_M + hang
    if g[2] < safe_z - 0.01:
        backend.move_to([g[0], g[1], safe_z], tolerance_m=CLEARANCE_TOLERANCE_M)


def _run(call: ActionCall, backend: ArmBackend, state: dict, get_state=None) -> tuple[bool, str]:
    if _is_skill(call.name):
        return _run_skill(call, backend, get_state or (lambda: state))
    """Drive the backend through the motion for one call."""
    name, args = call.name, call.args
    arm = args.get("arm")
    if name in ("pick", "place_on", "place_at", "push"):
        _lift_clear(backend, get_state, state, arm)

    if name == "pick":
        target = args["object"]
        obj = state["objects"][target]
        x, y, cz = obj["position_m"]
        top = cz + obj["half_extent_m"][2]
        # Grasp near the top of a tall object. Aiming at the centre of the glass
        # drives the hand into its rim, because the object is taller than the
        # gripper's reach below the grasp point.
        grasp_z = max(cz, top - GRASP_DEPTH_M)

        if not backend.move_to([x, y, top + APPROACH_HEIGHT_M], tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, f"could not reach the approach pose above {target}"
        backend.open_gripper()
        ok, note = _descend(backend, [x, y, grasp_z], get_state, f"onto {target}")
        if not ok:
            return False, note
        backend.close_gripper()
        backend.attach(target)
        if not backend.move_to([x, y, grasp_z + LIFT_HEIGHT_M], tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, f"could not lift {target}"
        return True, ""

    if name == "place_on":
        held = holding_of(state, arm)
        target = resolve_place_target(state, args["target"], held)
        drop = place_on_spot(state, target, held)
        lift = _grasp_offset_z(state, held, arm)
        if not backend.move_to([drop[0], drop[1], drop[2] + lift + APPROACH_HEIGHT_M],
                               tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, f"could not reach the approach pose above {target}"
        ok, note = _descend(backend, [drop[0], drop[1], drop[2] + lift], get_state, f"onto {target}")
        if not ok:
            return False, note
        backend.detach()
        backend.open_gripper()
        backend.move_to([drop[0], drop[1], drop[2] + lift + APPROACH_HEIGHT_M],
                        tolerance_m=CLEARANCE_TOLERANCE_M)
        return True, f"released {held}"

    if name == "place_at":
        x, y = float(args["x_m"]), float(args["y_m"])
        held = state["gripper"]["holding"]
        z = state["table_bounds_m"]["top_z"] + state["objects"][held]["half_extent_m"][2]
        lift = _grasp_offset_z(state, held, arm)
        if not backend.move_to([x, y, z + lift + APPROACH_HEIGHT_M], tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, "could not reach the approach pose above the target point"
        ok, note = _descend(backend, [x, y, z + lift], get_state, "to the target point")
        if not ok:
            return False, note
        backend.detach()
        backend.open_gripper()
        backend.move_to([x, y, z + lift + APPROACH_HEIGHT_M], tolerance_m=CLEARANCE_TOLERANCE_M)
        return True, ""

    if name == "push":
        target = args["object"]
        dx, dy = DIRECTION_VECTORS[args["direction"]]
        d = float(args["distance_m"])
        obj = state["objects"][target]
        pos = list(obj["position_m"])
        hx, hy, hz = obj["half_extent_m"]
        # Contact low on the object so tall or round things slide, not tip; for
        # a low object the centre is already low enough.
        contact_z = min(pos[2], pos[2] - hz + PUSH_CONTACT_Z_M)
        contact_z = max(contact_z, state["table_bounds_m"]["top_z"] + PUSH_CONTACT_Z_M)
        # Start behind the object's face, with room for the closed fingers.
        back = hx * abs(dx) + hy * abs(dy) + PUSH_STANDOFF_M
        start = [pos[0] - dx * back, pos[1] - dy * back, contact_z]
        end = [pos[0] + dx * d, pos[1] + dy * d, contact_z]
        backend.close_gripper()
        if not backend.move_to([start[0], start[1], start[2] + APPROACH_HEIGHT_M],
                               tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, f"could not reach the approach pose behind {target}"
        if not backend.move_to(start, tolerance_m=CLEARANCE_TOLERANCE_M):
            return False, f"could not get behind {target}"
        # The sweep may stall against friction or a neighbour. The verifier
        # measures how far the object actually travelled, so do not fail here.
        backend.move_to(end, tolerance_m=CLEARANCE_TOLERANCE_M)
        backend.move_to([end[0], end[1], end[2] + APPROACH_HEIGHT_M],
                        tolerance_m=CLEARANCE_TOLERANCE_M)
        return True, ""

    if name == "home":
        return (True, "") if backend.home() else (False, "the arm did not reach its home pose")

    if name == "ask_user":
        return True, ""

    return False, f"unknown action '{name}'"


__all__ = [
    "validate_plan", "classify_plan", "is_no_op", "execute", "classify",
    "check_preconditions",
    "place_on_spot", "resolve_place_target",
    "ActionCall", "ActionResult", "Safety",
]

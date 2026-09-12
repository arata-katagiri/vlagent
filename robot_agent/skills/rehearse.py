"""Run a skill in a snapshot of the world, measure what happened, restore.

The measurement is the safety input for learned skills: it says what left the
table, what fragile thing moved, and whether the skill's own check() agrees
that the declared effect happened. Nothing here asks the model anything.
"""

from __future__ import annotations

import copy
import math
import traceback
from dataclasses import dataclass, field

from .body import Arm, BudgetExceeded
from .registry import EFFECT_KINDS, Skill
from .sandbox import SkillCodeError, compile_skill

MOVED_M = 0.01
TIPPED_DEG = 45.0
ROTATE_TOL_DEG = 20.0
EDGE_REPORT_M = 0.08


def _yaw_tilt(quat) -> tuple[float, float]:
    """Yaw about world z, and tilt = angle between the body's z axis and world z."""
    w, x, y, z = quat
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    # third column of the rotation matrix, z component
    zz = 1 - 2 * (x * x + y * y)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))
    return round(yaw, 1), round(tilt, 1)


@dataclass
class Verdict:
    ok: bool
    reason: str
    level: str = "caution"          # safe | caution | irreversible
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    verified: bool = False          # True when the world, not the model, confirmed the effect

    def as_record(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "level": self.level,
                "metrics": self.metrics, "error": self.error, "verified": self.verified}


# -- the declared effect, checked by the world ------------------------------
def edge_distance(state: dict, name: str) -> float | None:
    """How far inside the nearest table edge the object's centre is (negative = beyond)."""
    obj = state["objects"].get(name)
    if obj is None:
        return None
    t = state["table_bounds_m"]
    x, y = obj["position_m"][0], obj["position_m"][1]
    return round(min(x - t["x_min"], t["x_max"] - x, y - t["y_min"], t["y_max"] - y), 3)


def _resolve_value(skill: Skill, args: dict) -> float | None:
    v = (skill.effect_value or "").strip()
    if not v:
        return None
    if v in args:
        try:
            return float(args[v])
        except (TypeError, ValueError):
            return None
    try:
        return float(v)
    except ValueError:
        return None


def verify_effect(skill: Skill, args: dict, before: dict, after: dict, metrics: dict,
                  mentioned: set[str]) -> tuple[bool, str, bool]:
    """(ok, reason, world_verified). The model may claim an effect; only the
    world confirms it. Unintended objects leaving the table always fail."""
    strays = [n for n in metrics.get("off_table", []) if n not in mentioned]
    if strays:
        return False, f"{', '.join(strays)} left the table and was not part of the request", True
    stray_fragile = [n for n in metrics.get("fragile_moved", []) if n not in mentioned]
    if stray_fragile:
        return False, f"fragile {', '.join(stray_fragile)} was moved and was not part of the request", True

    kind = skill.effect_kind if skill.effect_kind in EFFECT_KINDS else "other"
    if kind == "other":
        return True, "effect not measurable by the world; relying on the skill's own check", False

    name = args.get(skill.effect_of)
    if not isinstance(name, str) or name not in before["objects"]:
        return False, f"the declared effect refers to argument '{skill.effect_of}', which names no object", True
    a, b = after["objects"][name], before["objects"][name]
    value = _resolve_value(skill, args)

    if kind == "leaves_table":
        if name in metrics["off_table"]:
            return True, f"{name} left the table", True
        d = edge_distance(after, name)
        where = f"{d:.3f} m inside the nearest edge" if d is not None else "on the table"
        return False, f"{name} is still on the table, {where}", True
    if kind == "moves":
        if name in metrics["moved"]:
            return True, f"{name} moved {metrics['moved'][name]:.3f} m", True
        return False, f"{name} did not move", True
    if kind == "moves_at_least":
        got = metrics["moved"].get(name, 0.0)
        want = value if value is not None else MOVED_M
        if got >= want:
            return True, f"{name} moved {got:.3f} m (needed {want:.3f})", True
        return False, f"{name} moved {got:.3f} m, less than the {want:.3f} m declared", True
    if kind == "rotates_by":
        got = metrics["rotated"].get(name, 0.0)
        if value is None:
            return (abs(got) > 5, f"{name} turned {got:+.0f} deg", True)
        if abs(abs(got) - abs(value)) <= ROTATE_TOL_DEG:
            return True, f"{name} turned {got:+.0f} deg (asked {value:.0f})", True
        return False, f"{name} turned {got:+.0f} deg, not the {value:.0f} declared", True
    if kind == "tips_over":
        if name in metrics["tipped"]:
            return True, f"{name} tipped over (tilt {a.get('tilt_deg', 0):.0f} deg)", True
        return False, f"{name} is still upright (tilt {a.get('tilt_deg', 0):.0f} deg)", True
    if kind == "touches":
        if name in metrics["touched"]:
            return True, f"the gripper touched {name}", True
        return False, f"the gripper never touched {name}", True
    if kind == "rests_on":
        target = args.get("target")
        if a.get("on_top_of") == target:
            return True, f"{name} rests on {target}", True
        return False, f"{name} rests on {a.get('on_top_of') or 'the table'}, not on {target}", True
    if kind == "stays":
        if name not in metrics["moved"]:
            return True, f"{name} stayed put", True
        return False, f"{name} moved {metrics['moved'][name]:.3f} m but was declared to stay", True
    return True, "", False


# -- arms and world snapshot -------------------------------------------------
def handles(backend) -> list:
    """Every arm over the world: [a, b] for the two-arm backend, else [backend]."""
    if hasattr(backend, "for_arm"):
        return [h for h in (getattr(backend, "a", None), getattr(backend, "b", None)) if h is not None]
    return [backend]


def resolve(backend, args: dict):
    """(handle, args without 'arm'). A learned skill runs on one arm; the step's
    `arm` field picks it and must not reach run(arm, ...) as a keyword."""
    args = dict(args or {})
    which = args.pop("arm", None)
    if hasattr(backend, "for_arm"):
        return backend.for_arm(which), args
    return backend, args


def snapshot(backend) -> dict:
    if hasattr(backend, "data"):
        d, m = backend.data, backend.model
        return {
            "physics": True,
            "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
            "ctrl": d.ctrl.copy(), "time": float(d.time),
            "eq_active": d.eq_active.copy(), "eq_data": m.eq_data.copy(),
            "held": [getattr(h, "held", None) for h in handles(backend)],
            "drift": [getattr(h, "drift_m", 0.0) for h in handles(backend)],
        }
    return {"physics": False, "state": copy.deepcopy(backend.state), "held": getattr(backend, "held", None)}


def restore(backend, snap: dict) -> None:
    if snap["physics"]:
        import mujoco  # noqa: PLC0415

        d, m = backend.data, backend.model
        d.qpos[:] = snap["qpos"]; d.qvel[:] = snap["qvel"]; d.act[:] = snap["act"]
        d.ctrl[:] = snap["ctrl"]; d.time = snap["time"]
        d.eq_active[:] = snap["eq_active"]; m.eq_data[:] = snap["eq_data"]
        mujoco.mj_forward(m, d)
        for h, held, drift in zip(handles(backend), snap["held"], snap["drift"]):
            h.held = held
            if hasattr(h, "drift_m"):
                h.drift_m = drift
            cfg = getattr(h, "configuration", None)
            if cfg is not None:
                cfg.update(d.qpos)
        sync = getattr(backend, "sync", None) or getattr(handles(backend)[0], "sync", None)
        if sync:
            sync()
    else:
        backend.state.clear()
        backend.state.update(copy.deepcopy(snap["state"]))
        if hasattr(backend, "held"):
            backend.held = snap["held"]


# -- measurement ------------------------------------------------------------
def observe(backend, get_state) -> dict:
    """The world state plus per-object yaw, which the plain state lacks."""
    s = copy.deepcopy(get_state())
    if hasattr(backend, "data"):
        for name, obj in s["objects"].items():
            try:
                obj["yaw_deg"], obj["tilt_deg"] = _yaw_tilt(backend.data.body(name).xquat)
            except Exception:  # noqa: BLE001
                obj["yaw_deg"], obj["tilt_deg"] = 0.0, 0.0
            obj["upright"] = obj["tilt_deg"] < TIPPED_DEG
    return s


def measure(before: dict, after: dict, mentioned: set[str], touched=()) -> dict:
    moved, off_table, fragile_moved, rotated, tipped = {}, [], [], {}, []
    for name, b in before["objects"].items():
        a = after["objects"].get(name)
        if a is None:
            continue
        d = math.dist(b["position_m"], a["position_m"])
        if d > MOVED_M:
            moved[name] = round(d, 3)
            if b.get("fragile"):
                fragile_moved.append(name)
        if b.get("on_table") and not a.get("on_table") and not a.get("held"):
            off_table.append(name)
        dy = (a.get("yaw_deg", 0.0) - b.get("yaw_deg", 0.0) + 180) % 360 - 180
        if abs(dy) > 5:
            rotated[name] = round(dy, 1)
        if b.get("upright", True) and not a.get("upright", True):
            tipped.append(name)
    return {
        "touched": sorted(touched),
        "moved": moved,
        "rotated": rotated,
        "tipped": tipped,
        "off_table": off_table,
        "fragile_moved": fragile_moved,
        "side_effects": sorted(set(moved) - mentioned),
        "held_after": after["gripper"]["holding"],
    }


def classify(metrics: dict) -> tuple[str, str]:
    if metrics["off_table"]:
        return "irreversible", f"{', '.join(metrics['off_table'])} left the table"
    if metrics["fragile_moved"]:
        return "caution", f"fragile {', '.join(metrics['fragile_moved'])} was moved"
    if metrics["side_effects"]:
        return "caution", f"also moved {', '.join(metrics['side_effects'])}"
    return "safe", "only the intended objects moved and everything stayed on the table"


# -- the rehearsal ----------------------------------------------------------
def rehearse(skill: Skill, args: dict, backend, get_state, registry=None,
             budget_s: float = 15.0) -> Verdict:
    try:
        fns = compile_skill(skill.code)
    except SkillCodeError as exc:
        return Verdict(False, f"code rejected: {exc}", error=str(exc))

    handle, args = resolve(backend, args)
    snap = snapshot(backend)
    before = observe(backend, get_state)
    arm = Arm(handle, get_state, budget_s=budget_s, registry=registry)
    error = None
    stopped = False
    try:
        fns["run"](arm, **args)
    except KeyboardInterrupt:
        stopped = True
        error = "stopped by the user"
    except BudgetExceeded as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<skill>"), None)
        where = f" (skill line {line})" if line else ""
        error = f"{type(exc).__name__}: {exc}{where}"
    after = observe(backend, get_state)
    after["touched"] = sorted(arm.touched)

    mentioned = {v for v in args.values() if isinstance(v, str) and v in before["objects"]}
    metrics = measure(before, after, mentioned, touched=arm.touched)
    metrics["sim_seconds"] = round(arm.used_s, 2)
    metrics["edge_m"] = {n: d for n in mentioned
                         if (d := edge_distance(after, n)) is not None and d < EDGE_REPORT_M}
    # graph material: composition edges and the measured motion signature
    from .graph import motion_signature  # noqa: PLC0415

    focus = args.get(skill.effect_of)
    anchor = (before["objects"][focus]["position_m"]
              if isinstance(focus, str) and focus in before["objects"] else None)
    metrics["calls"] = list(dict.fromkeys(arm.calls))
    metrics["signature"] = motion_signature(arm.path, arm.grip, anchor, metrics)
    try:
        restore(backend, snap)
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, f"could not restore the world after rehearsal: {exc}", metrics=metrics, error=error)

    if stopped:
        return Verdict(False, "stopped by the user; the world was restored", metrics=metrics, error="stopped")
    if error:
        return Verdict(False, f"the code failed: {error}", metrics=metrics, error=error)

    # The world judges the declared effect first; the model's check() is an
    # extra condition, never a substitute.
    ok, why, verified = verify_effect(skill, args, before, after, metrics, mentioned)
    if not ok:
        return Verdict(False, f"effect not achieved: {why}", metrics=metrics, verified=verified)
    world_note = why if verified else ""

    # Learned safety rules judge the measured after-state (policy on only).
    from ..actions.safety import policy_enabled  # noqa: PLC0415
    from .rules import LEVEL_RANK, RULES  # noqa: PLC0415

    raised = None
    if policy_enabled() and RULES.rules:
        broken = RULES.check(before, after, skill.name, args)
        if broken:
            metrics["rules_broken"] = broken
            return Verdict(False, "; ".join(broken), metrics=metrics, verified=True)
        raised = RULES.level(before, after, skill.name, args)

    if fns["check"] is not None:
        try:
            result = fns["check"](before, after, **args)
            c_ok, c_why = (result if isinstance(result, tuple) else (bool(result), ""))
        except Exception as exc:  # noqa: BLE001
            return Verdict(False, f"check() raised {type(exc).__name__}: {exc}", metrics=metrics, verified=verified)
        if not c_ok:
            return Verdict(False, f"the skill's own check failed: {c_why or skill.effect}", metrics=metrics, verified=verified)
    elif not verified and not metrics["moved"] and not metrics["rotated"] and not metrics["tipped"] and not metrics["touched"]:
        missed = [n for n in mentioned if n not in arm.touched]
        why = (f"the gripper never touched {', '.join(missed)} and nothing moved"
               if missed else "nothing moved and nothing was touched, so the skill had no measurable effect")
        return Verdict(False, why, metrics=metrics, verified=False)

    if metrics["held_after"]:
        return Verdict(False, f"the skill ended still holding {metrics['held_after']}; release before returning", metrics=metrics, verified=verified)

    level, safety_why = classify(metrics)
    if raised is not None and LEVEL_RANK[raised[0]] > LEVEL_RANK[level]:
        level, safety_why = raised
    reason = f"{world_note}; {safety_why}" if world_note else safety_why
    return Verdict(True, reason, level=level, metrics=metrics, verified=verified)


__all__ = ["rehearse", "Verdict", "snapshot", "restore", "observe", "measure", "classify",
           "verify_effect", "edge_distance", "resolve", "handles"]

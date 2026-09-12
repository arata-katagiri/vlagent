"""Layer 1: the motion layer, and the `Arm` object skills are written against.

Everything task-level is above this file. What lives here is what the body
physically offers: go to a pose, follow a timed trajectory, open, close, weld
and release, and read the world. Orientation is yaw about world z and tilt
about the yawed y axis, both relative to the top-down home orientation.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

SPEED_MS = 0.35            # default straight-line speed, matches the backend
MIN_MOVE_S = 0.4           # a pure rotation still takes this long
SETTLE_STEPS = 40
CORRECTION_ROUNDS = 2
TOLERANCE_M = 0.012
GRASP_RADIUS_M = 0.06      # grasp() refuses if the gripper is not at the object


class BudgetExceeded(RuntimeError):
    """The skill used more simulated time than allowed."""


class Arm:
    def __init__(self, backend, get_state: Callable[[], dict], budget_s: float = 15.0,
                 registry=None):
        self._b = backend
        self._get_state = get_state
        self.budget_s = budget_s
        self.used_s = 0.0
        self._registry = registry
        self._yaw = 0.0
        self._tilt = 0.0
        self._physics = hasattr(backend, "_servo") and hasattr(backend, "data")
        # Which objects the hand or fingers made contact with, from MuJoCo's
        # contact list, sampled after every physics step this Arm drives. This
        # is what makes contact-only skills (tap, press, nudge) measurable.
        self.touched: set[str] = set()
        # Graph material, recorded as the skill runs: which skills it called,
        # where the tool centre point went, and whether the fingers were closed.
        self.calls: list[str] = []
        self.path: list[list[float]] = []
        self.grip: list[int] = []
        self._closed = False
        self._tick = 0
        self._hand_ids: set[int] = set()
        self._obj_by_body: dict[int, str] = {}
        if self._physics:
            import mujoco  # noqa: PLC0415

            self._mj = mujoco
            self._dt = float(backend.model.opt.timestep)
            self._mink = backend._mink
            m = backend.model
            prefix = getattr(backend, "prefix", "")      # "a_" / "b_" in the two-arm scene
            self._hand_ids = {i for i in range(m.nbody)
                              if m.body(i).name == f"{prefix}hand"
                              or (m.body(i).name.startswith(prefix) and "finger" in m.body(i).name)}
            for name in self._get_state()["objects"]:
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    self._obj_by_body[bid] = name

    # ------------------------------------------------------------ perception
    def state(self) -> dict:
        """The full scene dict: objects, gripper, table bounds."""
        return self._get_state()

    def objects(self) -> list[str]:
        """Names of the movable objects."""
        return [n for n, o in self.state()["objects"].items() if o.get("graspable", True)]

    def pos(self, name: str) -> list[float]:
        """World xyz of an object's centre, metres."""
        return list(self._obj(name)["position_m"])

    def half(self, name: str) -> list[float]:
        """Half extents [hx, hy, hz] of an object."""
        return list(self._obj(name).get("half_extent_m", [0.02, 0.02, 0.02]))

    def top(self, name: str) -> float:
        """Height of an object's top surface."""
        return self.pos(name)[2] + self.half(name)[2]

    def yaw(self, name: str) -> float:
        """Object yaw in degrees, from the simulator when available."""
        if not self._physics:
            return 0.0
        q = self._b.data.body(name).xquat
        w, x, y, z = q
        return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def tilt(self, name: str) -> float:
        """Degrees between the object's own z axis and vertical; 0 is upright, 90 is on its side."""
        if not self._physics:
            return 0.0
        w, x, y, z = self._b.data.body(name).xquat
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))

    def gripper_pos(self) -> list[float]:
        """World xyz of the tool centre point, between the fingertips."""
        return list(self.state()["gripper"]["position_m"])

    def holding(self) -> str | None:
        """Name of the object welded to the hand, or None."""
        return self.state()["gripper"]["holding"]

    def table(self) -> dict:
        """Table bounds x_min/x_max/y_min/y_max and top_z."""
        return dict(self.state()["table_bounds_m"])

    # --------------------------------------------------------------- motion
    def move_to(self, xyz, yaw_deg: float = 0.0, tilt_deg: float = 0.0,
                seconds: float | None = None) -> bool:
        """Straight line to xyz with yaw about vertical and tilt from top-down; seconds sets speed."""
        goal = np.asarray(xyz, dtype=float)
        if not self._physics:
            self._charge(seconds or 0.5)
            return bool(self._b.move_to(list(goal)))
        start = self._site_pos()
        dist = float(np.linalg.norm(goal - start))
        if seconds is None:
            seconds = max(MIN_MOVE_S, dist / SPEED_MS)
        self._charge(seconds)
        steps = max(1, int(seconds / self._dt))
        y0, t0 = self._yaw, self._tilt
        for i in range(1, steps + 1):
            a = i / steps
            self._servo(start + (goal - start) * a, y0 + (yaw_deg - y0) * a, t0 + (tilt_deg - t0) * a)
        self._yaw, self._tilt = float(yaw_deg), float(tilt_deg)
        # close the loop on the observed tool centre point (gravity droop)
        offset = np.zeros(3)
        for _ in range(CORRECTION_ROUNDS + 1):
            for _ in range(SETTLE_STEPS):
                self._servo(goal + offset, self._yaw, self._tilt)
            err = goal - self._site_pos()
            if float(np.linalg.norm(err)) <= TOLERANCE_M:
                return True
            offset += err
        return float(np.linalg.norm(goal - self._site_pos())) <= TOLERANCE_M

    def follow(self, keyframes: list[dict]) -> bool:
        """Follow timed keyframes. Spacing of `t` sets the speed.

        keyframe: {"t": s, "xyz": [x,y,z], "yaw_deg": 0, "tilt_deg": 0,
                   "gripper": None | "open" | "close" | "release" | "grasp:<name>"}
        """
        if not keyframes:
            return True
        frames = sorted(keyframes, key=lambda k: float(k.get("t", 0.0)))
        if not self._physics:
            for k in frames:
                self._charge(0.3)
                self._b.move_to(list(k["xyz"]))
                self._gripper_event(k.get("gripper"))
            return True
        t_prev = 0.0
        p_prev = self._site_pos()
        y_prev, tl_prev = self._yaw, self._tilt
        for k in frames:
            t = float(k.get("t", t_prev))
            p = np.asarray(k["xyz"], dtype=float)
            y = float(k.get("yaw_deg", y_prev))
            tl = float(k.get("tilt_deg", tl_prev))
            span = max(t - t_prev, self._dt)
            self._charge(span)
            steps = max(1, int(span / self._dt))
            for i in range(1, steps + 1):
                a = i / steps
                self._servo(p_prev + (p - p_prev) * a, y_prev + (y - y_prev) * a, tl_prev + (tl - tl_prev) * a)
            self._yaw, self._tilt = y, tl
            self._gripper_event(k.get("gripper"))
            t_prev, p_prev, y_prev, tl_prev = t, p, y, tl
        return True

    def open(self) -> None:
        """Open the fingers."""
        self._charge(0.12)
        self._closed = False
        self._b.open_gripper()
        self._scan_contacts()
        self._sample(force=True)

    def close(self) -> None:
        """Close the fingers (no weld)."""
        self._charge(0.12)
        self._closed = True
        self._b.close_gripper()
        self._scan_contacts()
        self._sample(force=True)

    def grasp(self, name: str) -> str:
        """Close on `name` and weld it to the hand. The gripper must be at it."""
        self._obj(name)
        gap = math.dist(self.gripper_pos(), self.pos(name))
        if gap > GRASP_RADIUS_M:
            raise RuntimeError(f"gripper is {gap:.3f} m from {name}; move onto it before grasping")
        self.close()
        self._b.attach(name)
        return name

    def release(self) -> None:
        """Drop the weld and open; the object keeps its velocity."""
        self._charge(0.2)
        self._closed = False
        self._b.detach()
        self._b.open_gripper()
        self._sample(force=True)

    def home(self) -> bool:
        """Return to the rest pose."""
        self._charge(1.0)
        self._yaw, self._tilt = 0.0, 0.0
        return bool(self._b.home())

    def wait(self, seconds: float) -> None:
        """Let physics run for a while without moving."""
        self._charge(seconds)
        if self._physics:
            self._b._step_physics(max(1, int(seconds / self._dt)))
            self._scan_contacts()

    # ---------------------------------------------------------- composition
    def skill(self, name: str, **args):
        """Call another learned skill (recorded as a composition edge)."""
        if self._registry is None or name not in self._registry:
            raise RuntimeError(f"no skill called '{name}'")
        self.calls.append(name)
        from .sandbox import compile_skill  # noqa: PLC0415

        fns = compile_skill(self._registry.get(name).code)
        return fns["run"](self, **args)

    # ------------------------------------------------------------ internals
    def _obj(self, name: str) -> dict:
        objs = self.state()["objects"]
        if name not in objs:
            raise KeyError(f"no object called '{name}'; objects are {', '.join(objs)}")
        return objs[name]

    def _charge(self, seconds: float) -> None:
        self.used_s += float(seconds)
        if self.used_s > self.budget_s:
            raise BudgetExceeded(f"skill exceeded its {self.budget_s:.0f} s budget of simulated time")

    def _site_pos(self) -> np.ndarray:
        key = getattr(self._b, "site_id", None)
        if key is None:
            key = self._b.site_name          # two-arm handles name their grasp site
        return self._b.data.site(key).xpos.copy()

    def _rotation(self, yaw_deg: float, tilt_deg: float):
        SO3 = self._mink.SO3
        return (
            SO3.from_z_radians(math.radians(yaw_deg))
            @ SO3.from_y_radians(math.radians(tilt_deg))
            @ self._b.home_rotation
        )

    def _servo(self, pos, yaw_deg: float, tilt_deg: float) -> None:
        self._b._servo(pos, self._rotation(yaw_deg, tilt_deg))
        self._scan_contacts()
        self._sample()

    def _sample(self, force: bool = False, every: int = 10) -> None:
        self._tick += 1
        if not force and self._tick % every:
            return
        if self._physics:
            p = self._site_pos()
            self.path.append([float(p[0]), float(p[1]), float(p[2])])
        else:
            self.path.append(list(self.gripper_pos()))
        self.grip.append(int(self._closed))

    def _scan_contacts(self) -> None:
        if not self._physics:
            return
        d, m = self._b.data, self._b.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 in self._hand_ids and b2 in self._obj_by_body:
                self.touched.add(self._obj_by_body[b2])
            elif b2 in self._hand_ids and b1 in self._obj_by_body:
                self.touched.add(self._obj_by_body[b1])

    def _gripper_event(self, event) -> None:
        if not event:
            return
        if event == "open":
            self.open()
        elif event == "close":
            self.close()
        elif event == "release":
            self.release()
        elif isinstance(event, str) and event.startswith("grasp:"):
            self.grasp(event.split(":", 1)[1].strip())
        else:
            raise ValueError(f"unknown gripper event {event!r}")


__all__ = ["Arm", "BudgetExceeded"]

"""Arm backends. One Protocol, three implementations, built in this order.

MockBackend    - no MuJoCo. Lets the agent loop be built and tested first.
PandaIKBackend - mink differential IK driving the Panda position actuators.

The planned FloatingGripperBackend escape hatch was not needed: the IK backend
picks and places every object in the scene reliably, so carrying a second
simulated arm would be dead weight.
"""

from __future__ import annotations

import math
from typing import Protocol

import mujoco
import numpy as np

from .scene import GRASP_SITE, HAND_BODY, ITEMS, footprint, half_height, object_qposadr

# How close the empty gripper must pass to an object to shove it, in metres.
PUSH_CONTACT_M = 0.07


class ArmBackend(Protocol):
    """Everything the executor is allowed to ask of a body."""

    def move_to(
        self,
        pos_m: list[float],
        yaw_rad: float = 0.0,
        timeout_s: float = 5.0,
        tolerance_m: float | None = None,
    ) -> bool:
        """Move the gripper to a world position.

        False if the tool centre point does not arrive within `tolerance_m`
        (default: the backend's precision tolerance) before the timeout.
        """
        ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...

    def attach(self, obj_name: str) -> None:
        """Activate the weld equality holding obj_name to the hand."""
        ...

    def detach(self) -> None:
        """Deactivate the active weld; the object keeps its velocity and settles."""
        ...

    def home(self) -> bool: ...


class MockBackend:
    """A symbolic arm over a plain state dict. No MuJoCo, no physics.

    Phase 2. It is deliberately more than a stub: placement drops the held object
    onto whatever is beneath it, so the executor's postcondition checks genuinely
    pass or fail rather than being rubber-stamped. That is what lets the whole
    agent loop be built and tested before the real arm exists.

    `fail_moves_beyond_m` makes move_to refuse targets past a radius, for testing
    the "could not reach" path. `drift_m` nudges an object after release, which
    is how --inject-failure exercises verification and replanning.
    """

    HOME_POS = [0.55, 0.0, 0.56]

    def __init__(self, state: dict, fail_moves_beyond_m: float | None = None,
                 drift_m: float = 0.0) -> None:
        self.state = state
        self.fail_moves_beyond_m = fail_moves_beyond_m
        self.drift_m = drift_m
        self.gripper_open = True
        self.moves: list[list[float]] = []
        self.state["gripper"]["position_m"] = list(self.HOME_POS)

    # -- ArmBackend ------------------------------------------------------
    def move_to(self, pos_m, yaw_rad: float = 0.0, timeout_s: float = 5.0,
                tolerance_m: float | None = None) -> bool:
        if self.fail_moves_beyond_m is not None:
            if math.hypot(pos_m[0], pos_m[1]) > self.fail_moves_beyond_m:
                return False
        start = list(self.state["gripper"]["position_m"])
        self.moves.append(list(pos_m))
        self.state["gripper"]["position_m"] = list(pos_m)

        held = self.state["gripper"]["holding"]
        if held is not None:
            self.state["objects"][held]["position_m"] = list(pos_m)
        else:
            self._sweep(start, list(pos_m))
        return True

    def _sweep(self, start: list[float], end: list[float]) -> None:
        """Shove anything the empty gripper drags through. The mock's only contact.

        A horizontal move at object height carries everything close to the swept
        segment along to the end point, which is what makes push() verifiable
        without physics.
        """
        if abs(end[2] - start[2]) > 0.02:  # a vertical move does not shove
            return
        for name, obj in self.state["objects"].items():
            if obj["held"] or not ITEMS[name]["graspable"]:
                continue
            px, py, pz = obj["position_m"]
            if abs(pz - end[2]) > PUSH_CONTACT_M:
                continue
            if self._point_to_segment(px, py, start, end) > PUSH_CONTACT_M:
                continue
            obj["position_m"] = [round(end[0], 4), round(end[1], 4), round(pz, 4)]
            obj["on_table"] = self._on_table(end[0], end[1])

    @staticmethod
    def _point_to_segment(px: float, py: float, a: list[float], b: list[float]) -> float:
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def open_gripper(self) -> None:
        self.gripper_open = True

    def close_gripper(self) -> None:
        self.gripper_open = False

    def attach(self, obj_name: str) -> None:
        obj = self.state["objects"][obj_name]
        parent = obj["on_top_of"]
        if parent and obj_name in self.state["objects"][parent]["supporting"]:
            self.state["objects"][parent]["supporting"].remove(obj_name)
        obj.update(held=True, on_top_of=None, on_table=False)
        obj["position_m"] = list(self.state["gripper"]["position_m"])
        self.state["gripper"]["holding"] = obj_name

    def detach(self) -> None:
        held = self.state["gripper"]["holding"]
        if held is None:
            return
        obj = self.state["objects"][held]
        x, y, _ = obj["position_m"]
        # One-shot: the injected fault happens once so that the agent's recovery
        # can succeed. A permanent drift would just loop until it gave up.
        x += self.drift_m
        self.drift_m = 0.0
        support, top_z = self._support_under(held, x, y)
        obj.update(
            held=False,
            on_top_of=support,
            on_table=self._on_table(x, y),
            position_m=[round(x, 4), round(y, 4), round(top_z + self._half_h(held), 4)],
        )
        if support is not None:
            self.state["objects"][support]["supporting"].append(held)
        self.state["gripper"]["holding"] = None

    def home(self) -> bool:
        return self.move_to(list(self.HOME_POS))

    # -- internals -------------------------------------------------------
    def _half_h(self, name: str) -> float:
        return half_height(ITEMS[name])

    def _on_table(self, x: float, y: float) -> bool:
        t = self.state["table_bounds_m"]
        return t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]

    def _support_under(self, held: str, x: float, y: float) -> tuple[str | None, float]:
        """Highest item whose footprint covers (x, y), else the table or the floor."""
        best: tuple[float, str | None] = (
            self.state["table_bounds_m"]["top_z"] if self._on_table(x, y) else 0.0,
            None,
        )
        for name, obj in self.state["objects"].items():
            if name == held or obj["held"]:
                continue
            hx, hy = footprint(ITEMS[name])
            px, py, pz = obj["position_m"]
            if abs(x - px) <= hx and abs(y - py) <= hy:
                top = pz + half_height(ITEMS[name])
                if top > best[0]:
                    best = (top, name)
        return best[1], best[0]


def initial_state() -> dict:
    """The scene's starting state as a plain dict, without touching MuJoCo.

    Same shape as world_state.get_world_state, built from the static registry.
    Lets MockBackend and the whole agent loop run with no simulator at all.
    """
    from .scene import TABLE_BOUNDS_M, ITEMS as _ITEMS

    objects = {}
    for name, item in _ITEMS.items():
        x, y = item["pos"][0], item["pos"][1]
        z = (
            item["pos"][2]
            if name == "tray"
            else TABLE_BOUNDS_M["top_z"] + half_height(item)
        )
        hx, hy = footprint(item)
        objects[name] = {
            "position_m": [round(x, 4), round(y, 4), round(z, 4)],
            "half_extent_m": [round(hx, 4), round(hy, 4), round(half_height(item), 4)],
            "color": item["color"],
            "fragile": bool(item["fragile"]),
            "graspable": bool(item["graspable"]),
            "on_table": True,
            "on_top_of": None,
            "supporting": [],
            "near_table_edge": False,
            "held": False,
        }
    return {
        "objects": objects,
        "gripper": {"position_m": list(MockBackend.HOME_POS), "holding": None},
        "table_bounds_m": dict(TABLE_BOUNDS_M),
    }


# IK and motion tuning.
IK_SOLVER = "daqp"
IK_POS_TOL_M = 1e-4
IK_ORI_TOL_RAD = 1e-4
IK_MAX_ITERS = 20
CARTESIAN_SPEED_MS = 0.35
# Precision tolerance, for grasping and setting down. Clearance waypoints pass a
# looser value: carrying a load at reach leaves ~1.2 cm of droop, which is
# irrelevant when the move only has to lift the object clear of the table.
MOVE_TOLERANCE_M = 0.01
CORRECTION_ROUNDS = 3
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


class PandaIKBackend:
    """mink differential IK driving the Panda position actuators.

    Shape follows mink's own arm_panda.py: iterate solve_ik and integrate_inplace
    to convergence, then write configuration.q into ctrl and step. Cartesian moves
    are interpolated at a fixed speed so the motion reads as real time on video.

    Grasping is faked with the weld equalities built into the scene: attach()
    writes the live relative pose into eq_data and activates the weld, so the
    object keeps real velocity on release instead of popping.
    """

    def __init__(self, model, data, sync=None, settle_steps: int = 40,
                 drift_m: float = 0.0):
        import mink  # noqa: PLC0415

        self._mink = mink
        self.model = model
        self.data = data
        self.sync = sync or (lambda: None)
        self.settle_steps = settle_steps

        self.configuration = mink.Configuration(model)
        self.configuration.update(data.qpos)

        self.eef_task = mink.FrameTask(
            frame_name=GRASP_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=model, cost=1e-2)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.tasks = [self.eef_task, self.posture_task]

        # Top-down orientation, taken from the home pose rather than derived from
        # a Euler convention: the hand's local z already points down at `home`.
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.home_rotation = mink.SO3.from_matrix(
            data.site(GRASP_SITE).xmat.reshape(3, 3).copy()
        )
        self.home_pos = data.site(GRASP_SITE).xpos.copy()

        self.welds = {
            name: i
            for i in range(model.neq)
            for name in [model.equality(i).name.removeprefix("weld_")]
            if model.equality(i).name.startswith("weld_")
        }
        self.held: str | None = None
        self.drift_m = drift_m
        self.data.ctrl[7] = GRIPPER_OPEN

    # -- ArmBackend ------------------------------------------------------
    def move_to(self, pos_m, yaw_rad: float = 0.0, timeout_s: float = 5.0,
                tolerance_m: float | None = None) -> bool:
        tolerance = MOVE_TOLERANCE_M if tolerance_m is None else tolerance_m
        target_R = self._mink.SO3.from_z_radians(yaw_rad) @ self.home_rotation
        start = self.data.site(GRASP_SITE).xpos.copy()
        goal = np.asarray(pos_m, dtype=float)

        distance = float(np.linalg.norm(goal - start))
        steps = max(1, int(distance / (CARTESIAN_SPEED_MS * self.model.opt.timestep)))
        steps = min(steps, int(timeout_s / self.model.opt.timestep))

        for i in range(1, steps + 1):
            waypoint = start + (goal - start) * (i / steps)
            self._servo(waypoint, target_R)

        # Close the loop on the measured TCP. The IK reference is accurate to a
        # fraction of a millimetre, but the position actuators hold a ~1 cm
        # steady-state error against gravity, so commanding the nominal goal is
        # not enough. Offset the reference by the observed error and re-settle.
        offset = np.zeros(3)
        for _ in range(CORRECTION_ROUNDS + 1):
            for _ in range(self.settle_steps):
                self._servo(goal + offset, target_R)
            error_vec = goal - self.data.site(GRASP_SITE).xpos
            if float(np.linalg.norm(error_vec)) <= tolerance:
                return True
            offset = offset + error_vec

        return float(np.linalg.norm(self.data.site(GRASP_SITE).xpos - goal)) <= tolerance

    def open_gripper(self) -> None:
        self._drive_gripper(GRIPPER_OPEN)

    def close_gripper(self) -> None:
        self._drive_gripper(GRIPPER_CLOSED)

    def attach(self, obj_name: str) -> None:
        """Activate the weld holding obj_name, at its current relative pose."""
        eq = self.welds[obj_name]
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        pos, quat = self._relative_pose(self.hand_id, body)
        self.model.eq_data[eq, :3] = 0.0
        self.model.eq_data[eq, 3:6] = pos
        self.model.eq_data[eq, 6:10] = quat
        self.model.eq_data[eq, 10] = 1.0
        self.data.eq_active[eq] = True
        self.held = obj_name
        self._step_physics(5)

    def detach(self) -> None:
        if self.held is None:
            return
        released = self.held
        self.data.eq_active[self.welds[released]] = False
        self.held = None
        if self.drift_m:
            # --inject-failure: nudge the object as it is released, once, so the
            # agent's verification catches a placement that looks like it worked.
            adr = object_qposadr(self.model, released)
            self.data.qpos[adr] += self.drift_m
            self.drift_m = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._step_physics(self.settle_steps)

    def home(self) -> bool:
        return self.move_to(self.home_pos)

    # -- internals -------------------------------------------------------
    def _servo(self, position, rotation) -> None:
        target = self._mink.SE3.from_rotation_and_translation(
            rotation, np.asarray(position, dtype=float)
        )
        self.eef_task.set_target(target)
        for _ in range(IK_MAX_ITERS):
            velocity = self._mink.solve_ik(
                self.configuration, self.tasks, self.model.opt.timestep,
                IK_SOLVER, damping=1e-3,
            )
            self.configuration.integrate_inplace(velocity, self.model.opt.timestep)
            err = self.eef_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err[:3]) <= IK_POS_TOL_M
                and np.linalg.norm(err[3:]) <= IK_ORI_TOL_RAD
            ):
                break
        self.data.ctrl[:7] = self.configuration.q[:7]
        self._step_physics(1)

    def _drive_gripper(self, value: float, steps: int = 60) -> None:
        self.data.ctrl[7] = value
        self._step_physics(steps)

    def _step_physics(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.sync()

    def _relative_pose(self, body1: int, body2: int):
        """Pose of body2 expressed in body1's frame, as (pos, quat)."""
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, self.data.xquat[body1])
        delta = self.data.xpos[body2] - self.data.xpos[body1]
        pos = np.zeros(3)
        mujoco.mju_rotVecQuat(pos, delta, q1_inv)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q1_inv, self.data.xquat[body2])
        return pos, quat



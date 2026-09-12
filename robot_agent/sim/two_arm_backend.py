"""One ArmHandle per Panda in the two-arm scene, sharing a single MjModel/MjData.

Mirrors PandaIKBackend method for method, with every name and index resolved
through the arm's prefix instead of hardcoded. Moves are sequential by design
(the app is single-threaded): a move on arm A steps physics for both arms, and
arm B simply holds its last ctrl. There is no collision avoidance between the
arms, so callers should return one arm home before moving the other into the
shared zone.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .two_arm_scene import (
    ARMS, GRASP_SITE, HAND_BODY, HOME_QPOS, GRASPABLE2,
    arm_actuator_ids, arm_joint_qposadr, build_two_arm_scene, object_qposadr,
)

IK_SOLVER = "daqp"
IK_POS_TOL_M = 1e-4
IK_ORI_TOL_RAD = 1e-4
IK_MAX_ITERS = 20
CARTESIAN_SPEED_MS = 0.35
MOVE_TOLERANCE_M = 0.01
CORRECTION_ROUNDS = 3
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


class ArmHandle:
    def __init__(self, model, data, prefix: str, sync=None, settle_steps: int = 40):
        import mink  # noqa: PLC0415

        self._mink = mink
        self.model, self.data = model, data
        self.prefix = prefix
        self.sync = sync or (lambda: None)
        self.settle_steps = settle_steps

        self.site_name = f"{prefix}{GRASP_SITE}"
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{HAND_BODY}")
        self.qpos_ids = arm_joint_qposadr(model, prefix)   # 7 joints + 2 fingers
        self.act_ids = arm_actuator_ids(model, prefix)     # 7 joints + gripper
        self.base_xy = np.array(ARMS[prefix.rstrip("_")]["base"][:2])

        self.configuration = mink.Configuration(model)
        self.configuration.update(data.qpos)
        self.eef_task = mink.FrameTask(
            frame_name=self.site_name, frame_type="site",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=model, cost=1e-2)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.tasks = [self.eef_task, self.posture_task]

        self.home_rotation = mink.SO3.from_matrix(data.site(self.site_name).xmat.reshape(3, 3).copy())
        self.home_pos = data.site(self.site_name).xpos.copy()

        self.welds = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{prefix}weld_{name}")
            for name in GRASPABLE2
        }
        self.held: str | None = None

    # -- observation -----------------------------------------------------
    def tcp(self) -> np.ndarray:
        return self.data.site(self.site_name).xpos.copy()

    def reach_from_base(self, x: float, y: float) -> float:
        return float(np.hypot(x - self.base_xy[0], y - self.base_xy[1]))

    # -- ArmBackend surface ----------------------------------------------
    def move_to(self, pos_m, yaw_rad: float = 0.0, timeout_s: float = 5.0,
                tolerance_m: float | None = None) -> bool:
        tolerance = MOVE_TOLERANCE_M if tolerance_m is None else tolerance_m
        # Re-sync the IK reference with the world: the other arm (and physics)
        # moved since this handle last solved.
        self.configuration.update(self.data.qpos)
        target_R = self._mink.SO3.from_z_radians(yaw_rad) @ self.home_rotation
        start = self.tcp()
        goal = np.asarray(pos_m, dtype=float)
        distance = float(np.linalg.norm(goal - start))
        steps = max(1, int(distance / (CARTESIAN_SPEED_MS * self.model.opt.timestep)))
        steps = min(steps, int(timeout_s / self.model.opt.timestep))
        for i in range(1, steps + 1):
            self._servo(start + (goal - start) * (i / steps), target_R)
        offset = np.zeros(3)
        for _ in range(CORRECTION_ROUNDS + 1):
            for _ in range(self.settle_steps):
                self._servo(goal + offset, target_R)
            err = goal - self.tcp()
            if float(np.linalg.norm(err)) <= tolerance:
                return True
            offset = offset + err
        return float(np.linalg.norm(self.tcp() - goal)) <= tolerance

    def open_gripper(self) -> None:
        self._drive_gripper(GRIPPER_OPEN)

    def close_gripper(self) -> None:
        self._drive_gripper(GRIPPER_CLOSED)

    def attach(self, obj_name: str) -> None:
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
        self.data.eq_active[self.welds[self.held]] = False
        self.held = None
        self._step_physics(self.settle_steps)

    def home(self) -> bool:
        return self.move_to(self.home_pos)

    # -- internals -------------------------------------------------------
    def _servo(self, position, rotation) -> None:
        target = self._mink.SE3.from_rotation_and_translation(rotation, np.asarray(position, dtype=float))
        self.eef_task.set_target(target)
        for _ in range(IK_MAX_ITERS):
            v = self._mink.solve_ik(self.configuration, self.tasks, self.model.opt.timestep,
                                    IK_SOLVER, damping=1e-3)
            self.configuration.integrate_inplace(v, self.model.opt.timestep)
            err = self.eef_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= IK_POS_TOL_M and np.linalg.norm(err[3:]) <= IK_ORI_TOL_RAD:
                break
        self.data.ctrl[self.act_ids[:7]] = self.configuration.q[self.qpos_ids[:7]]
        self._step_physics(1)

    def _drive_gripper(self, value: float, steps: int = 60) -> None:
        self.data.ctrl[self.act_ids[7]] = value
        self._step_physics(steps)

    def _step_physics(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.sync()

    def _relative_pose(self, body1: int, body2: int):
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, self.data.xquat[body1])
        d = self.data.xpos[body2] - self.data.xpos[body1]
        pos = np.zeros(3)
        mujoco.mju_rotVecQuat(pos, d, q1_inv)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q1_inv, self.data.xquat[body2])
        return pos, quat


class TwoArms:
    """Both handles over one shared world."""

    def __init__(self, model=None, data=None, sync=None):
        if model is None:
            model, data = build_two_arm_scene()
        self.model, self.data = model, data
        self.a = ArmHandle(model, data, ARMS["a"]["prefix"], sync)
        self.b = ArmHandle(model, data, ARMS["b"]["prefix"], sync)

    # -- routing ---------------------------------------------------------
    def for_arm(self, name: str | None):
        """The handle for a named arm. The executor calls this per step.

        None falls back to arm A rather than raising: a plan that omits `arm`
        has already been rejected by the precondition checks, so reaching here
        without one means a code path that predates two arms.
        """
        if name is None:
            return self.a
        try:
            return getattr(self, str(name))
        except AttributeError:
            raise KeyError(f"no arm called {name!r}; expected one of a, b") from None

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def nearest_arm(self, x: float, y: float) -> ArmHandle:
        return self.a if self.a.reach_from_base(x, y) <= self.b.reach_from_base(x, y) else self.b

    # -- ArmBackend passthrough (arm A), so un-routed callers still work -----
    def move_to(self, *a, **kw):
        return self.a.move_to(*a, **kw)

    def open_gripper(self) -> None:
        self.a.open_gripper()

    def close_gripper(self) -> None:
        self.a.close_gripper()

    def attach(self, obj_name: str) -> None:
        self.a.attach(obj_name)

    def detach(self) -> None:
        self.a.detach()

    def home(self) -> bool:
        return self.a.home()

    @property
    def sync(self):
        return self.a.sync

    @sync.setter
    def sync(self, fn) -> None:
        self.a.sync = fn
        self.b.sync = fn


__all__ = ["ArmHandle", "TwoArms"]

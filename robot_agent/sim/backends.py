"""Arm backends. One Protocol, three implementations, built in this order.

MockBackend        - Phase 2, no MuJoCo. Lets the agent loop be built and tested first.
PandaIKBackend     - Phase 4, mink differential IK driving the Panda position actuators.
FloatingGripperBackend - escape hatch, mocap body moved directly. Hard cutoff 2:15 PM.
"""

from __future__ import annotations

from typing import Protocol


class ArmBackend(Protocol):
    """Everything the executor is allowed to ask of a body."""

    def move_to(
        self, pos_m: list[float], yaw_rad: float = 0.0, timeout_s: float = 5.0
    ) -> bool:
        """Move the gripper to a world position. False if not within 1 cm in time."""
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
    """No simulator. Moves succeed, attach/detach flip a field in a state dict.

    Phase 2. Keeps the pytest suite instant and headless.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Phase 2")


class PandaIKBackend:
    """mink differential IK: FrameTask on attachment_site + PostureTask.

    Phase 4. Shape follows mink's own arm_panda.py example: iterate solve_ik and
    integrate_inplace to convergence, then data.ctrl = configuration.q[:8], mj_step.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Phase 4")


class FloatingGripperBackend:
    """Mocap body with two-finger geometry, moved directly. --backend floating."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 4 escape hatch")

"""Arm backends. One Protocol, three implementations, built in this order.

MockBackend        - Phase 2, no MuJoCo. Lets the agent loop be built and tested first.
PandaIKBackend     - Phase 4, mink differential IK driving the Panda position actuators.
FloatingGripperBackend - escape hatch, mocap body moved directly. Hard cutoff 2:15 PM.
"""

from __future__ import annotations

import math
from typing import Protocol

from .scene import ITEMS, footprint, half_height

# How close the empty gripper must pass to an object to shove it, in metres.
PUSH_CONTACT_M = 0.07


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
    def move_to(self, pos_m, yaw_rad: float = 0.0, timeout_s: float = 5.0) -> bool:
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
        x += self.drift_m
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

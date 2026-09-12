"""Phase 0 smoke test: the package imports and the contract is the agreed shape.

Headless. Must not import mujoco.viewer.
"""

import inspect

from robot_agent.actions.schema import ActionCall, ActionResult, Safety
from robot_agent.sim.backends import ArmBackend


def test_safety_classes():
    assert [s.value for s in Safety] == ["safe", "caution", "irreversible"]


def test_action_call_shape():
    call = ActionCall(name="pick", args={"object": "red_block"}, rationale="asked for")
    assert call.args["object"] == "red_block"
    assert ActionResult(ok=True, reason="", state_after={}).ok is True


def test_backend_protocol_surface():
    expected = {"move_to", "open_gripper", "close_gripper", "attach", "detach", "home"}
    actual = {n for n, _ in inspect.getmembers(ArmBackend, inspect.isfunction)}
    assert expected <= actual

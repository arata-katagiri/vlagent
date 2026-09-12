"""Headless two-arm hand-off: A carries the red block to the tray in the middle,
B takes it from the tray to its own side. Writes runs/two_arm_demo.mp4 plus
start/end stills through the repo's own renderer (ffmpeg pipe, no imageio).

Run: python -m robot_agent.sim.two_arm_demo
"""

from __future__ import annotations

import time

import numpy as np

from .render import RUNS_DIR, png, recorder
from .two_arm_backend import TwoArms
from .two_arm_scene import CAMERA_NAME, OBJECTS2, TRAY2, half_height, rest_z, settle

APPROACH, LIFT, GRASP_DEPTH, CLEAR_TOL = 0.12, 0.15, 0.025, 0.025


def pick(arm, arms, name):
    obj = OBJECTS2[name]
    p = arms.object_pos(name)
    top = p[2] + half_height(obj)
    arm.open_gripper()
    ok = arm.move_to([p[0], p[1], top + APPROACH], tolerance_m=CLEAR_TOL)
    ok &= arm.move_to([p[0], p[1], top - GRASP_DEPTH])
    grasp_offset = arm.tcp()[2] - arms.object_pos(name)[2]
    arm.close_gripper()
    arm.attach(name)
    ok &= arm.move_to([p[0], p[1], top + LIFT], tolerance_m=CLEAR_TOL)
    lifted = arms.object_pos(name)[2] - p[2]
    return ok and lifted > 0.03, grasp_offset


def place(arm, arms, name, xy, grasp_offset):
    tcp_z = rest_z(OBJECTS2[name]) + grasp_offset + 0.004
    ok = arm.move_to([xy[0], xy[1], tcp_z + LIFT], tolerance_m=CLEAR_TOL)
    ok &= arm.move_to([xy[0], xy[1], tcp_z])
    arm.detach()
    arm.open_gripper()
    ok &= arm.move_to([xy[0], xy[1], tcp_z + LIFT], tolerance_m=CLEAR_TOL)
    return ok


def main():
    t0 = time.time()
    arms = TwoArms()
    model, data = arms.model, arms.data
    settle(model, data, 0.3)
    RUNS_DIR.mkdir(exist_ok=True)
    png(model, data, RUNS_DIR / "two_arm_start.png", camera=CAMERA_NAME)

    log = []
    target = (0.88, 0.0)
    with recorder(model, RUNS_DIR / "two_arm_demo.mp4", camera=CAMERA_NAME) as rec:
        arms.a.sync = lambda: rec.capture(data)
        arms.b.sync = lambda: rec.capture(data)

        ok, off = pick(arms.a, arms, "red_block")
        log.append(f"A pick red_block: {'OK' if ok else 'FAIL'}")
        ok = place(arms.a, arms, "red_block", TRAY2["pos"][:2], off)
        log.append(f"A place on tray: {'OK' if ok else 'FAIL'}")
        log.append(f"A home: {'OK' if arms.a.home() else 'FAIL'}")

        ok, off = pick(arms.b, arms, "red_block")
        log.append(f"B pick red_block from tray: {'OK' if ok else 'FAIL'}")
        ok = place(arms.b, arms, "red_block", target, off)
        log.append(f"B place at {target}: {'OK' if ok else 'FAIL'}")
        log.append(f"B home: {'OK' if arms.b.home() else 'FAIL'}")
        settle(model, data, 0.5)

    final = arms.object_pos("red_block")
    err = float(np.hypot(final[0] - target[0], final[1] - target[1]))
    on_table = abs(final[2] - rest_z(OBJECTS2["red_block"])) < 0.01
    log.append(f"red_block final {np.round(final, 3).tolist()}  xy err {err:.3f} m  on table: {on_table}")
    png(model, data, RUNS_DIR / "two_arm_end.png", camera=CAMERA_NAME)
    print("\n".join(log))
    print(f"wall={time.time() - t0:.1f}s video={RUNS_DIR / 'two_arm_demo.mp4'}")
    print("TWO-ARM HANDOFF", "PASS" if (err < 0.02 and on_table) else "FAIL")


if __name__ == "__main__":
    main()

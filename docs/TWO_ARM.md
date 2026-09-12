# Two-arm scene (standalone, experimental)

Two Menagerie Pandas facing each other across one table, built 2026-09-12 after the
single-arm feature freeze. **Nothing in the single-arm path imports these files.**

```
robot_agent/sim/two_arm_scene.py    scene: empty world + Panda attached twice with prefixes a_/b_
robot_agent/sim/two_arm_backend.py  ArmHandle (prefix-parametrized PandaIKBackend) + TwoArms
robot_agent/sim/two_arm_demo.py     headless hand-off demo, writes runs/two_arm_demo.mp4
```

Run:
```bash
python -m robot_agent.sim.two_arm_demo      # ~25 s, prints TWO-ARM HANDOFF PASS
```

## How the second arm goes in

`scene.py` loads `panda.xml` as the root spec and adds the table into it, which cannot
be done twice. Here an empty `MjSpec` is the root and the Panda spec is **attached** into a
frame twice via `spec.attach(child, prefix=..., frame=...)`. Every body, joint, actuator,
site and keyframe comes out prefixed: `a_hand`, `b_hand`, `a_grasp_site`, `a_actuator1..8`,
`b_actuator1..8`. The grasp site is added to the child *before* attaching so it gets the
prefix too. Welds exist per (arm, object): `a_weld_cup`, `b_weld_cup`.

The parent's `option.integrator` must be set to `implicitfast` explicitly: on attach the
parent's options win and the Panda's own setting is dropped with a warning.

## Layout

| | arm A | arm B |
|---|---|---|
| base | origin, facing +x | (1.05, 0, 0), rotated 180 deg |
| home | Franka "ready" pose (TCP 0.31 m out) | same |
| reach | 0.25-0.80 m from base | same |

Table x 0.30-0.75, y +-0.38, top z 0.40. Both envelopes cover the whole centre line, so
the tray in the middle (0.525, 0) is the hand-off zone. Blocks start on A's side, cups on
B's side.

**Why not the Menagerie `home` pose:** with the bases 1.05 m apart it puts the two hands
6 cm from each other at reset. The ready pose keeps the wrists retracted (44 cm apart)
and the gripper still points straight down, so `home_rotation` works unchanged.

## What is NOT done (next steps, in order)

1. `world_state` for two grippers (currently the single-arm `get_world_state` assumes one).
2. Safety reach check per arm (`ArmHandle.reach_from_base`) instead of distance from origin.
3. Planner routing: which arm executes a step (`TwoArms.nearest_arm` is a start).
4. Collision between arms: none. Rule for now is one arm goes home before the other moves.
5. A `--two-arm` flag on the app; chess board scene + python-chess on top of this.

## Known deviations from CLAUDE.md

- The machine this was built on has no system `ffmpeg`; `imageio-ffmpeg` was pip-installed
  and its bundled binary symlinked to `.venv/bin/ffmpeg` so `--record` and this demo work.
  Not added to `requirements.txt`.

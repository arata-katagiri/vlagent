# Two-arm scene

Two Menagerie Pandas facing each other across one table, built 2026-09-12. The agent
knows there are two arms: every physical step names the arm that performs it, reach is
measured from each arm's own base, and moving an object across the table requires a
planned hand-off. Single-arm behaviour is unchanged and is still the default.

```bash
mjpython -m robot_agent.app.main --scene two_arm     # the agent, two arms
python  -m robot_agent.sim.two_arm_demo              # scripted hand-off, headless
```

```
robot_agent/sim/two_arm_scene.py        empty world + Panda attached twice with prefixes a_/b_
robot_agent/sim/two_arm_backend.py      ArmHandle (prefix-parametrized) + TwoArms with for_arm()
robot_agent/sim/two_arm_world_state.py  ground truth -> state dict with an "arms" key
robot_agent/sim/two_arm_demo.py         headless scripted hand-off, writes runs/two_arm_demo.mp4
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
| base | origin, facing +x | (1.25, 0, 0), rotated 180 deg |
| home | Franka "ready" pose (TCP 0.31 m out) | same |
| reach | 0.25-0.80 m from base | same |

Table x 0.30-0.95, y +-0.38, top z 0.40, tray at (0.62, 0).

**The bases are 1.25 m apart, not 1.05.** Each arm reaches 0.80 m from its own base, so at
a closer spacing both arms reach the whole table and the choice of arm never matters. At
1.25 m the reach zones are exclusive:

| object | reachable by |
|---|---|
| red_block, green_block, blue_block | arm a only |
| cup, glass | arm b only |
| tray | both |

So moving anything across the table *requires* a hand-off through the tray, and the agent
plans it: pick with a, place on tray, send a home, pick with b, place on b's side.

**Why not the Menagerie `home` pose:** with the bases 1.05 m apart it puts the two hands
6 cm from each other at reset. The ready pose keeps the wrists retracted (44 cm apart)
and the gripper still points straight down, so `home_rotation` works unchanged.

## How the agent knows

| layer | change |
|---|---|
| `schema.set_arms(names)` | called once at startup. With >1 arm, pick/place_on/place_at/push/home gain a required `arm` argument, pinned to an enum. `arm` is in `ARG_KEYS` so the parser keeps it |
| `two_arm_world_state` | state gains `"arms": {"a": {...}, "b": {...}}` with each arm's gripper position, what it holds, and its base. `"gripper"` stays as a compatibility mirror |
| `safety` | `holding_of(state, arm)` and `base_of(state, arm)` replace direct gripper reads; `reachable(x, y, base)` measures from the arm's own base. Out-of-reach messages name the other arm when it could do the job |
| `executor` | routes each step through `backend.for_arm(args["arm"])` |
| `llm` | the prompt gains a two-arm section; the scene listing tags every object with which arms can reach it |
| `ui` | an `arm` column appears in the plan table only when steps carry one |

Describing `arm` in the prompt was not enough on its own: gpt-4o proposed a correct
five-step hand-off three times and omitted the arm every time. Listing `arm` in the tool
schema's `required` array is what made it appear.

## Still not done

1. **No collision avoidance between the arms.** They share one workspace and one physics
   step. The prompt tells the planner to send an arm `home` before the other reaches into
   the middle, and gpt-4o does, but nothing enforces it.
2. `--backend mock` is not supported for this scene; it needs the real simulator.
3. Chess: board scene plus a move engine on top of this.

## Known deviations from CLAUDE.md

- The machine this was built on has no system `ffmpeg`; `imageio-ffmpeg` was pip-installed
  and its bundled binary symlinked to `.venv/bin/ffmpeg` so `--record` and this demo work.
  Not added to `requirements.txt`.

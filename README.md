# Language-to-Action Robot Arm Agent

A home-assistant robot arm you talk to in plain English. It reads the actual scene before acting,
turns a vague request into a **visible plan** of typed physical actions, checks preconditions,
**asks before anything irreversible**, and **verifies that each action actually worked**.

> Built during the AI Tinkerers Hong Kong hackathon.

```
you > Push the glass out of the way to the left.

  #  action  arguments                                   safety
  1  push    object=glass, direction=left, distance_m=0.3  irreversible

  Step 1 is irreversible.
  predicted final position is 0.14 m past the table edge; glass would fall
  Type yes to allow this one step (no): no

  1. push skipped; asking for a safer alternative
  1. pick     OK  holding glass, lifted 0.149 m
  2. place_on OK  glass rests on tray, 0.002 m from the intended spot
```

## Why this cannot be a chatbox

1. **Grounding.** Vague language is resolved against live scene state. "Put it somewhere safe"
   becomes `place_on(tray)` because the agent can see where the tray is and what is already on it.
2. **Physical consequences.** The agent reasons about *irreversibility* — a category that only
   exists once you have a body. Pushing a glass 0.30 m left is fine or catastrophic depending on
   where the table ends.
3. **Closed loop.** Every action is verified against ground truth, and the agent replans when
   reality disagrees with the plan.

**Safety is deterministic code, not model judgement.** The planner proposes; `actions/safety.py`
decides. The model never sees that code run — it only receives the verdict, so it cannot approve,
downgrade, or argue past a refusal. Irreversible steps need a per-step `yes` from the user, the
default is no, and a confirmation is never reused for another step.

## The loop

```
observe → plan → validate → show → confirm → execute → verify → report
           ↑                                              │
           └──────────── replan on failure ───────────────┘
```

| Stage | What happens |
|---|---|
| **observe** | `get_world_state` turns MuJoCo ground truth into a plain dict: positions, what supports what, edge proximity, what is held |
| **plan** | The LLM may only answer with `propose_plan(steps, summary)` or `ask_user(question)`. Action names are pinned to an enum |
| **validate** | Preconditions are checked symbolically, simulating the state forward so multi-step plans check correctly. Failures go back to the model as feedback, at most twice |
| **show** | A table of every step with its arguments, the model's rationale, and a safety badge |
| **confirm** | Irreversible steps stop and ask, quoting the measured consequence |
| **execute** | One step at a time through the `ArmBackend` protocol |
| **verify** | Each action's postcondition is checked against ground truth, not against what the plan expected |
| **report** | One paragraph, plus a JSONL log of every plan, refusal, confirmation and result |

## Actions

| Tool | Preconditions | Verified postcondition | Safety |
|---|---|---|---|
| `pick(object)` | exists, graspable, nothing on top, gripper empty | gripper holds it, object lifted clear | safe · caution if fragile |
| `place_on(target)` | holding something, target flat and not fragile | rests on target within 2 cm of the intended spot | safe |
| `place_at(x_m, y_m)` | holding something, point reachable | rests within 2 cm of the point | caution near edge · **irreversible off table** |
| `push(object, direction, distance_m)` | exists, gripper empty, 0 < distance ≤ 0.3 m | moved at least half the requested travel | **irreversible if predicted to fall** · caution if fragile |
| `home()` | none | arm at rest pose | safe |
| `ask_user(question)` | none | none | safe |

## Setup

macOS on Apple Silicon, Python 3.13. CPU only — no GPU required.

```bash
python3 -m venv vlagent && source vlagent/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add an OpenAI-compatible key
```

The Panda model is fetched by `robot_descriptions` and cached under `~/.cache`, so it loads offline
after the first run.

## Run

`mujoco.viewer.launch_passive` needs `mjpython` on macOS:

```bash
mjpython -m robot_agent.app.main                      # full loop, live viewer
mjpython -m robot_agent.app.main --demo               # the scripted demo
python  -m robot_agent.app.main --backend mock --mock-llm   # no simulator, no network
python  -m robot_agent.app.main --no-viewer --record  # headless, writes runs/*.mp4
```

| Flag | Effect |
|---|---|
| `--mock-llm` | Scripted offline planner. No network |
| `--backend mock` | Symbolic arm, no MuJoCo. Runs under plain `python` |
| `--demo` | Runs the four scripted scenarios |
| `--record` | Writes an mp4 via ffmpeg from the fixed `demo` camera |
| `--inject-failure` | Nudges an object on release, to exercise verification and replanning |
| `--no-viewer` | Headless simulation |
| `--no-log` | Skip the JSONL run log |

Tests are headless and never open the viewer:

```bash
pytest          # 62 tests
```

## Architecture

```
robot_agent/
  sim/scene.py        Panda + table + objects + inactive grasp welds, fixed camera
  sim/world_state.py  MuJoCo ground truth -> plain state dict
  sim/backends.py     ArmBackend protocol; MockBackend and PandaIKBackend
  sim/render.py       offscreen stills, filmstrips, mp4 recorder
  actions/schema.py   six typed actions, bounds with units, the LLM tool schema
  actions/safety.py   preconditions and irreversibility — deterministic, no LLM
  actions/executor.py plan validation, execution, postcondition verification
  agent/llm.py        OpenAI-compatible client + offline MockLLM
  agent/planner.py    command -> validated plan, clarification, replanning
  app/ui.py           rich rendering: scene, plan table, badges, confirmations
  app/main.py         single-threaded loop, REPL, demo mode, JSONL run log
```

Two rules hold the design together:

- **Nothing above `sim/backends.py` imports MuJoCo.** Safety, the executor and the planner operate
  on plain dicts, so they were built and tested before the arm existed and the whole agent runs
  under `--backend mock` with no simulator at all.
- **There is no threading.** `launch_passive` renders on its own UI thread, so blocking on `input()`
  or an LLM call never freezes the window, and the arm is idle during both. Physics steps only
  inside backend calls. No queues, no lock on `data`.

### Notes on the simulation

Grasping is faked with MuJoCo **weld equalities** — one per graspable object, inactive at rest.
`attach()` writes the live relative pose into `eq_data` and flips the weld on; `detach()` flips it
off, so the object keeps real velocity and settles naturally instead of popping.

The arm is driven by [mink](https://github.com/kevinzakka/mink) differential IK: a `FrameTask` on
the grasp site plus a `PostureTask`, iterated to convergence, written to the position actuators.
The IK reference is accurate to ~0.3 mm, but the actuators hold about 1.3 cm of steady-state droop
against gravity, so `move_to` closes the loop on the *observed* tool centre point rather than
trusting the reference.

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco) — physics
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Franka Emika Panda model
- [mink](https://github.com/kevinzakka/mink) — differential inverse kinematics

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

## Skills the robot writes for itself

The six actions above are the starting vocabulary, not the limit. Ask for a motion the arm
cannot do yet and the planner **writes a new skill** instead of refusing:

```
you > rotate the red block 90 degrees

  new skill: rotate(object, angle_deg)          <- the model's code, shown before anything moves
  rehearsal of rotate (world restored)
    PASSED  only the intended objects moved     rotated: red_block +90 deg   safe
  Keep rotate as a skill and run it for real? (yes/no): yes
  1. rotate OK red_block turned 90 degrees

you > now rotate the green block by 45 degrees
  1. rotate OK green_block turned 45 degrees     <- no code this time: the robot learned it
```

Three layers sit above the joints:

| layer | file | what it is |
|---|---|---|
| 1 motion | `skills/body.py` | the `Arm` the model codes against: `move_to(xyz, yaw, tilt, seconds)`, `follow(keyframes)` for timed trajectories (throws, sweeps, taps), grasp/release, perception |
| 2 skills | `skills/registry.py` | task-level functions, built-in or model-written, persisted as `skills/<name>.json` with their code, declared effect and rehearsal record |
| 3 plans | `agent/planner.py` | sequences of actions and skills for one request |

**Every new skill is rehearsed before it is kept.** `skills/rehearse.py` snapshots the MuJoCo state,
runs the code in a sandbox (`skills/sandbox.py`: no imports, no file or network access, a budget of
simulated time), measures what happened, and restores the world. The measurement, not the model,
decides the safety class: anything that left the table is **irreversible** and needs a per-step
`yes`; a fragile object touched is **caution**. If the skill's own `check()` says the declared effect
did not happen, the metrics go back to the model and it revises the code, at most three times.

```
you > knock the cup over
  new skill: knock_over(object)  ... rehearsal FAILED: declared effect not achieved (revision 2, 3)
  rehearsal of knock_over (world restored)
    PASSED  cup left the table   irreversible     tipped over: cup   side effects: red_block
  Step 1 is irreversible. Type yes to allow this one step (no):
```

`skills` lists the library; `forget <name>` removes one; `--skills-dir` chooses where they live.
For a live picture, run `python -m robot_agent.app.graph_window` in a second terminal: a small window that redraws whenever a skill is added, revised or forgotten. Cosmetic only; it never touches the simulator.
Learned skills can call each other through `arm.skill(name, **args)`.

## The browser UI

```bash
python -m robot_agent.app.main --web --safety            # then open http://127.0.0.1:8000
python -m robot_agent.app.main --web --scene two_arm     # any scene works; pick the camera on the page
```

Four boxes. The **simulation**, streamed from the same render hook the recorder uses, with a camera
picker and a Stop button (Ctrl-C for the running step; a rehearsal restores the world). The
**library**, switchable between learned actions and learned safety rules, each row with its level and
whether the world or the model verified it; click a row to see its code, ✕ to forget it. **Talk to the
arm**, where the agent's plan summaries, questions, answers and results appear and every confirmation
is a pair of single-use buttons that default to no; a voice toggle uses the browser's own speech
recognition, or dictate into the box with Wispr Flow; a speak toggle reads replies aloud. And **what
the agent is doing**: the terminal's own rendering, captured as HTML, so plan tables, generated code
and rehearsal verdicts look exactly as they do in the terminal.

The server runs on one daemon thread that only reads the latest JPEG and moves JSON between queues;
the agent loop, physics and every prompt stay on the main thread, as in the REPL. Needs `fastapi` and
`uvicorn[standard]`; the terminal UI does not.

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
| `--safety` | Enable the policy checks and the confirmation gate (fragile, stability, irreversible). **Off by default** |
| `--skills-dir` | Where learned skills are kept (default `./skills`) |

Tests are headless and never open the viewer:

```bash
pytest          # 62 tests
```

## Safety is opt-in

Without `--safety` the policy layer is off: nothing is refused for being unwise,
unstable or irreversible, and nothing is confirmed. The banner turns red so the mode is never
ambiguous. Checks that describe what is *impossible* still apply — an object must exist, the gripper
holds one thing, a target must be within reach — because without those the executor has nothing to
act on.

It is worth running once to see why the policy exists. Told to stack onto the fragile glass, the arm
does it: every placement passes verification at the moment of release, and then the blue block falls
off the rim onto the table while the cup stays perched on top. The policy was not being timid; it
was predicting that.

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

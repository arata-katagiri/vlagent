# CLAUDE.md: Language-to-Action Robot Arm Agent

> Put this file at the root of the repo as `CLAUDE.md`. Claude Code reads it automatically.
> **We build on ONE machine, in ONE Claude Code session.** Phases run in sequence.
>
> *Revised before Phase 0: single-threaded app loop (was worker threads + locks), weld-based grasp
> (was per-step pose override), agent loop built before the arm (was after), offscreen filmstrip
> rendering for visual checks (was screencapture). Rationale is inline in each section.*

## 1. Context and hard constraints

Built at the AI Tinkerers Hong Kong hackathon. The global challenge: "Build an agent for a place
people already work, talk, or live, then make it meaningfully more useful because of that context."
Our environment is robotics: a simulated home-assistant robot arm.

Hard constraints (never violate these):

- **About 3 hours of real build time, one machine.** A working end-to-end loop beats more features.
  Ruthless simplicity matters more than usual.
- **Net-new build.** Everything here is written during the event. Libraries, model files and templates
  are allowed. Commit at the end of every phase; the git history is our proof.
  - Therefore: **no third-party source trees inside this repo.** `mink` is a PyPI dependency, not a
    vendored clone. The Panda model comes from the `robot_descriptions` package at runtime.
- **Public GitHub repo.** No secrets. `.env` in `.gitignore`, provide `.env.example`.
- **macOS on Apple Silicon, no NVIDIA GPU.** Native arm64 Python 3.13 (venv at `vlagent/`), CPU only.
- **Slow venue Wi-Fi and possible LLM region issues.** The pipeline must also run offline with a mock
  LLM (`--mock-llm`). The provider is set by env var, never hardcoded. The MuJoCo Menagerie clone is
  already cached at `~/.cache/robot_descriptions/`, so model loading works with Wi-Fi down.
- **2-minute demo video.** We need a deterministic scripted demo mode (`--demo`) and an offscreen
  recorder (`--record`) so the video does not depend on a screen capture working.
- **Feature freeze at 2:30 PM.** After that: bug fixes, README, recording, submission only.

Sequential by default. The headless phases (2 and 3) touch no simulator and *can* be parallelized
across subagents if we fall behind — ask the human before spawning any.

## 2. The product

**Pitch:** A home-assistant robot arm you talk to in plain language. It reads the actual scene before
acting, turns vague requests into a visible plan of typed physical actions, checks preconditions,
asks before anything irreversible, and verifies each action actually worked.

**User:** someone at home who cannot easily reach or handle objects, with an arm at their kitchen table.

**Why this cannot be a chatbox (put this in the README and the demo):**
1. Grounding: vague language resolved against live scene state ("the cup", "somewhere safe").
2. Physical consequences: the agent reasons about irreversibility, which only exists with a body.
3. Closed loop: actions verified against ground truth, replanning on failure.

## 3. Judging rubric (optimize for these)

| Criterion | What earns a 4 to 5 |
|---|---|
| Core functionality | The loop (command, plan, confirm, execute, verify) works reliably |
| Innovation and theme | The environment shapes the workflow; the safety pattern is impossible in a chatbox |
| Technical execution | Clean architecture, typed tools, deterministic safety, failure handling |
| Usefulness and agentic UX | Clear user, clear value, user in control (visible plan, confirmations) |

The rubric scores **the loop and the safety pattern**, not arm realism. Build in that order.

## 4. Scene

- MuJoCo with the Franka Emika Panda from MuJoCo Menagerie. Load via the `robot_descriptions` package
  (`panda_mj_description`), falling back to a local `mujoco_menagerie` clone. Load the model's `home`
  keyframe at startup. **Verified in Phase 0** against the cached Menagerie copy, so do not re-guess:
  - `MJCF_PATH` resolves to `franka_emika_panda/panda.xml`, which **has no sites at all**. The
    `attachment_site` frame used by mink's `arm_panda.py` lives in `panda_nohand.xml`; `mjx_panda.xml`
    calls its equivalent `gripper`. Our end-effector frame must therefore be either a site we add
    ourselves via `mujoco.MjSpec` (preferred: visible in the viewer, easier to debug) or the `hand`
    body with a baked TCP offset (fallback, no MJCF surgery).
  - Bodies of interest: `hand`, `left_finger`, `right_finger`. Weld to `hand`.
  - **TCP offset measured at the `home` keyframe: +0.0584 m along the hand frame's local z**
    (finger-body midpoint relative to `hand`). Start Phase 4 from this number.
  - `nq=9, nv=9, nu=8`: actuators are `actuator1..actuator7` (arm) and `actuator8` (gripper,
    range 0-255, 255 = open at `home`). Home pose puts the hand at about x=0.55, z=0.62.
- A table in front of the arm. Define `TABLE_BOUNDS_M` (x/y min/max, top z) in one place.
- Objects (free bodies, simple primitive geoms, distinct colors, all named):
  `red_block`, `green_block`, `blue_block` (boxes ~4 cm), `cup` (cylinder ~7 cm),
  `glass` (tall thin cylinder, semi-transparent, `fragile=True`), `tray` (fixed flat box, the "safe place").
- One object registry (Python dict): name, aliases, color, fragile, graspable, size.
- One **inactive `<weld>` equality per graspable object**, hand body to object body. See section 8.
- A fixed camera angle that looks good on video (arm and whole table visible). Declare it in the MJCF
  as a named `<camera>` so the viewer, the offscreen renderer and the recorder all share it.

## 5. Architecture

```
robot_agent/
  sim/scene.py            # builds/loads MJCF: Panda + table + objects + welds
  sim/world_state.py      # ground truth -> structured dict
  sim/backends.py         # ArmBackend protocol, MockBackend, PandaIKBackend, FloatingGripperBackend
  sim/render.py           # offscreen PNG, filmstrip, mp4 recorder (no viewer)
  actions/schema.py       # typed action definitions (params with units, safety class)
  actions/safety.py       # preconditions, irreversibility prediction (deterministic code)
  actions/executor.py     # plan validation, execution, postcondition verification
  agent/llm.py            # OpenAI-compatible client + MockLLM
  agent/planner.py        # command -> plan, clarification, replanning
  app/main.py             # single-threaded loop, viewer, terminal REPL, demo mode
tests/
runs/                     # JSONL logs and screenshots (gitignored)
```

### Interface contract (write these stubs in Phase 0, before any logic)

```python
# sim/world_state.py
def get_world_state(model, data) -> dict:
    """{"objects": {"<name>": {"position_m": [x,y,z], "color": str, "fragile": bool,
                               "on_table": bool, "on_top_of": str|None, "supporting": [str],
                               "near_table_edge": bool, "held": bool}},
        "gripper": {"position_m": [x,y,z], "holding": str|None},
        "table_bounds_m": {"x_min":..., "x_max":..., "y_min":..., "y_max":..., "top_z":...}}"""

# sim/backends.py
class ArmBackend(Protocol):
    def move_to(self, pos_m: list[float], yaw_rad: float = 0.0, timeout_s: float = 5.0) -> bool: ...
    def open_gripper(self) -> None: ...
    def close_gripper(self) -> None: ...
    def attach(self, obj_name: str) -> None: ...
    def detach(self) -> None: ...
    def home(self) -> bool: ...

# actions/schema.py
class Safety(StrEnum):          # never a bare string: the badge, the gate and the log all key off this
    SAFE = "safe"; CAUTION = "caution"; IRREVERSIBLE = "irreversible"

@dataclass
class ActionCall:
    name: str; args: dict; rationale: str

@dataclass
class ActionResult:
    ok: bool; reason: str; state_after: dict

# actions/executor.py
def validate_plan(plan: list[ActionCall], state: dict) -> list[str]: ...   # empty = valid
def classify(call: ActionCall, state: dict) -> tuple[Safety, str]: ...     # (class, reason)
def execute(call: ActionCall, backend: ArmBackend, get_state) -> ActionResult: ...
```

Write these stubs first and keep them stable, so each module stays testable on its own.
**Nothing above `sim/backends.py` may import MuJoCo.** Safety, executor and planner operate on plain
state dicts; that is what lets phases 2 and 3 be built and tested before the arm exists.

### Threading: there is none (macOS)

`mujoco.viewer.launch_passive` requires `mjpython` on macOS. It renders on its **own** UI thread, so
blocking our loop in `input()` or a 2-second LLM call does not freeze the window — it just stops
feeding new physics, and the arm is idle during both anyway. So the app is one thread:

```python
while True:
    cmd  = input("> ")                      # blocks; arm idle; fine
    plan = planner.plan(cmd, state)         # blocks; arm idle; fine
    for step in plan:
        executor.execute(step, backend)     # this is what runs mj_step + viewer.sync()
```

No worker threads, no queues, no lock on `data`. Physics steps **only** inside backend calls, at a
`RateLimiter` so motion looks real-time. If we later want the scene physically live while the user
types, use non-blocking stdin (`select.select`), still with no lock. Do not reintroduce threads.

## 6. Actions (the tools the LLM can call)

All parameters carry units in their names and have explicit bounds.

| Tool | Preconditions | Postcondition verified | Safety |
|---|---|---|---|
| `pick(object)` | exists, graspable, nothing on top, gripper empty | gripper holds it, object lifted | safe (caution if fragile) |
| `place_on(target)` | holding something, target is `tray` or a block, target not fragile | held object rests on target (xy within 2 cm) | safe |
| `place_at(x_m, y_m)` | holding something, point reachable | object rests at point (within 2 cm) | caution near edge, irreversible if off table |
| `push(object, direction, distance_m)` | exists, gripper empty, direction in {left,right,forward,back}, 0 < distance_m <= 0.3 | object moved roughly as predicted | irreversible if predicted final position is off the table; caution if fragile |
| `home()` | none | arm at home pose | safe |
| `ask_user(question)` | none | none | safe |

Safety rules:
- Classification is **deterministic code in `actions/safety.py`**, never LLM judgment. The LLM cannot
  downgrade it. `classify` returns a `Safety` enum, so a typo cannot silently become "safe".
- Irreversible steps require the user to type `yes` for that specific step. A confirmation is bound to
  one step and cannot be reused. Default is no.
- Log every refusal and confirmation.

## 7. Agent loop

1. **Observe:** get world state, serialize compactly for the prompt.
2. **Plan:** the LLM must call `propose_plan(steps=[{name, args, rationale}], summary)` or
   `ask_user(question)` when a reference is genuinely ambiguous.
3. **Validate:** check preconditions symbolically in order. If invalid, return the errors to the LLM
   and replan (max 2 times).
4. **Show plan:** a `rich` table with step number, action, args, rationale, safety badge.
5. **Confirm:** for irreversible steps show the concrete reason ("predicted final position is 0.12 m
   past the table edge; the glass would fall") and ask y/n. If refused, ask the LLM for a safer alternative.
6. **Execute and verify:** run steps one at a time, check postconditions from ground truth. On failure:
   stop, re-observe, replan the remaining goal (max 2 attempts), then give up gracefully and go home.
7. **Report:** a one-paragraph summary.

LLM system prompt must say: act only through tools; never invent objects; resolve references using the
scene state; ask instead of guessing when ambiguity matters; safety is enforced externally.

LLM client: OpenAI-compatible (`openai` SDK). `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` from
`.env`, so we can switch between the event's OpenAI credits, OpenRouter, or a local endpoint without
code changes. 30 s timeout, one retry. `MockLLM` maps the demo commands (section 9) to fixed plans so
everything runs offline. The installed `openai` SDK is 3.x — read its tool-calling signature before
writing the client rather than assuming.

## 8. Controller notes

Three backends behind one `ArmBackend` Protocol. Build them in this order:

- **`MockBackend` (Phase 2, no MuJoCo).** `move_to` returns True, `attach`/`detach` flip a field in a
  plain state dict. Lets the whole agent loop be built and tested before any arm works, and keeps the
  pytest suite instant and headless.
- **`PandaIKBackend` (Phase 4, the visual payoff).** `mink` differential IK: a `FrameTask` on our
  end-effector frame (see section 4 -- *not* `attachment_site`, it does not exist in `panda.xml`) plus a
  `PostureTask`, iterate `solve_ik` + `integrate_inplace` to convergence, then
  `data.ctrl = configuration.q[:8]` and `mj_step`. This is the shape of mink's own `arm_panda.py`; read
  it, do not reinvent it. A reference clone sits at `~/Documents/my_projects/mink` (outside this repo).
- **`FloatingGripperBackend` (escape hatch).** A mocap body with two-finger geometry moved directly,
  same interface, `--backend floating`. **Hard cutoff: if Panda IK is not doing a reliable
  pick-and-place by 2:15 PM, switch and move on.**

Grasping:

- **Fake it with a weld, not a pose override.** Each graspable object has an inactive `<weld>` equality
  to the hand body in the MJCF. `attach` sets `data.eq_active[weld_id] = True`, `detach` sets it False.
  Verified on mujoco 3.13: `data.eq_active` is a runtime-toggleable bool array and the weld holds the
  relative transform through stepping. Two lines, and on release the object keeps real velocity and
  settles naturally instead of popping — which is the "cup jitters after release" bug we would
  otherwise spend Phase 5 chasing. Do NOT attempt real frictional grasping.
- Fallback if the solver misbehaves: per-step pose override (set the held object's free-joint pose from
  the gripper pose plus the offset captured at attach, zero its velocity).
- **Pick:** approach 10 cm above, descend, close and attach, lift 15 cm.
  **Place:** move above target, descend to target top + half object height + 1 cm, detach, open, retreat.
- Top-down grasps only; only yaw varies.
- Interpolate waypoints so motion looks smooth at roughly real time.
- `move_to` returns False if position error is not under 1 cm within the timeout.

## 9. App and demo

- Run: `mjpython -m app.main` with flags `--mock-llm`, `--backend {panda,floating,mock}`, `--demo`,
  `--record`, `--inject-failure`.
- Terminal UI with `rich`: scene summary, plan table, safety badges, confirmations, results.
- Log each run to `runs/<timestamp>.jsonl`.
- `--record` writes frames from `mujoco.Renderer` at the named camera and pipes them to `ffmpeg`
  (already installed; no `imageio` dependency). The 2-minute video comes from this, not from a screen
  recording that a laggy window or a missing permission can ruin. Record the viewer too if it looks
  better, but `--record` is the one that must work.
- `--demo` runs this script with pauses, for recording:
  1. "Can you clear the red and green blocks onto the tray?" (multi-step, executed and verified)
  2. "Put the blue block on the glass." (rejected: fragile, not a stable surface; proposes the tray)
  3. "Push the glass out of the way to the left." (predicted to fall, flagged irreversible, user says no,
     agent proposes picking it up onto the tray instead)
  4. With `--inject-failure`: an object is nudged after placement, verification catches it, agent replans.

## 10. Build phases and checkpoints (single machine, sequential)

Phases 2 and 3 build the agent **before** the arm, against `MockBackend`. That is deliberate: the loop
and the safety pattern are what the rubric scores, they depend on no simulator, and it means we have a
shippable demo at 2:00 instead of gambling the afternoon on IK convergence.

One person drives the keyboard. **At every checkpoint: run it, show the output, commit, then STOP and
wait for human confirmation.** Never claim something works without having run it.

| Phase | Clock | Deliverable |
|---|---|---|
| 0 | 11:15–11:30 | Repo reset and skeleton, `requirements.txt`, `.gitignore`, `.env.example`, README stub, interface stubs from section 5, environment verified (model loads offline). Commit. |
| 1 | 11:30–12:00 | Scene loads in the viewer with arm, table, objects, welds; named camera; `get_world_state` prints correct JSON; offscreen PNG renders. Commit. |
| 2 | 12:45–1:15 | `actions/schema.py`, `safety.py`, `executor.py`, `MockBackend`. pytest green and headless: push glass off table is irreversible, place on glass is rejected, pick with something on top fails, postcondition failure is caught. Commit. |
| 3 | 1:15–1:45 | `agent/llm.py`, `MockLLM`, `planner.py`, rich UI. **Full loop end to end on `MockBackend`.** Tag `demo-v0` — from here we always have something to show. Commit. |
| 4 | 1:45–2:15 | `PandaIKBackend` with weld grasp: scripted pick of `red_block` onto `tray` three times in a row, then wired into the executor so the real loop drives the real arm. **Hard cutoff 2:15: switch to `--backend floating`.** Commit. |
| 5 | 2:15–2:30 | Confirmation flow polish, `--demo`, `--record`, `--inject-failure` if time. **Feature freeze at 2:30.** Commit. |
| 6 | 2:30–3:00 | README (setup, architecture, "Built during the hackathon", credits for MuJoCo, Menagerie, mink), recording, submission. |

Phase order is priority order. If a phase overruns, cut from Phase 5 first, then Phase 4 (ship on
`--backend floating`, or in the worst case `mock`). Phase 3 must happen: without it there is no agent.

## 11. How the three of us work on one machine

- **Driver** (keyboard, the single Claude Code session): runs the phases above.
- **Navigator** (sitting beside): watches the viewer and the diffs, reads what Claude Code writes,
  spots wrong frames, wrong names, floating or intersecting objects. Claude Code can render stills and
  filmstrips and read them itself, but only the navigator sees live motion and timing feel, so report
  precisely what went wrong ("the gripper stops 5 cm above the block", "the cup jitters after release").
- **Third person** works away from the keyboard on things that need no repo access: demo wording, the
  2-minute pitch, project title, written description, social post draft, partner handles, testing
  screen recording, checking the submission portal requirements.
- **Rotate driver and navigator at each phase boundary** so everyone understands the code. The rules say
  we may be asked which parts we built; all three of us must be able to answer.
- The third person joins the keyboard pair whenever a phase is behind schedule.

## 12. Working rules for Claude Code

- Simplest thing that works. No premature abstractions, no web UI, no extra frameworks.
- Dependencies: `mujoco`, `mink` plus its QP solver, `numpy`, `openai`, `rich`, `python-dotenv`,
  `pytest`, `robot_descriptions`, `loop-rate-limiters`. All are already installed in `vlagent/`.
  Ask before adding others.
- If an installed library's API differs from these notes, read the installed source or docs and adapt.
  The architecture matters, not the exact calls.
- **Check the simulation visually at every visual checkpoint.** Render offscreen to `runs/` with
  `mujoco.Renderer` and read the PNG directly — it is deterministic, headless and needs no macOS
  permission. To check *motion*, render N frames of a trajectory and tile them into a single filmstrip
  PNG (3x2 grid); that shows approach, descent, close and lift in one image. Use `screencapture` only
  if the navigator asks for it. Do not assume a phase passed because the code ran without errors.
- Headless tests must not open the viewer and must not import `mujoco.viewer`.
- Commit at every phase boundary: `feat(sim): ...`, `fix(executor): ...`, `docs: ...`.
- Before starting any Phase 5 extra, make sure the last commit is a known-good demo we can revert to.
- Never hardcode or print API keys.

## 13. Stretch goals (only if the core demo is solid and verified before 2:15)

1. Verification and replanning with `--inject-failure` (highest value).
2. One harder grounding command ("put the cup somewhere safe" when the tray is occupied).
3. Voice input via speech-to-text, keeping typed input as the fallback.

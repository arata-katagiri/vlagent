# Plan: a robot that grows its own primitives

*Written 2026-09-12 ~14:00 HKT. Deadline 16:30. Show & tell 15:30.*

## The idea in one line

Vibe coding, but the output is a robot skill: describe a motion the robot cannot do yet,
the agent writes it, rehearses it in the simulator, shows you the result, and keeps it if
you approve. Nothing task-level is hardcoded. The catalogue grows during the session.

## Why this is the right shape

The current repo has six verbs (pick, place_on, place_at, push, home, ask_user) with
per-verb safety rules and per-verb execution code. Every new capability today means a
human editing three files. The change is to make **everything a skill**, including the six
we already have, so "built in" and "learned five minutes ago" are the same kind of thing.

Only the *body* is fixed. Everything above it is generated, measured, and kept or discarded.

## Layers

```
 request ──> planner ──> catalogue hit? ──yes──> plan with skills ──> rehearse ──> confirm ──> execute ──> verify
                │                                                       ▲
                └── no ──> define_skill(code) ──> sandbox ──> rehearse ──┘   (metrics back to the model, ≤3 revisions)
                                                                        │
                                                            keep ──> library on disk
```

### 1. Body API (fixed, tiny)  `robot_agent/skills/body.py`
What the arm physically offers and nothing else. Wraps the existing `PandaIKBackend`.

| call | notes |
|---|---|
| `move_to(xyz, yaw=0, tilt=0, speed=None)` | tilt and speed are NEW args on the backend; yaw exists today |
| `move_through(waypoints, speed, release_at=None)` | for throws: detach at waypoint index |
| `open()`, `close()`, `attach(name)`, `detach()` | exist today |
| `state()` | poses, what is held, table bounds, contacts |
| `home()` | exists |

### 2. Skill registry (generated, growing)  `robot_agent/skills/registry.py`
A skill is: `name`, `args` (typed), `doc`, `code` (a `run(arm, state, **args)` function), a declared
**effect** the code claims to produce (e.g. `yaw_changed(object) >= 80deg`, `z(object) < table_top`),
and provenance (author = model or human, created, rehearsal metrics, uses, failures).
Persisted as `skills/<name>.py` + `skills/<name>.json`. Loaded at launch. The six existing verbs are
migrated into this format so the mechanism is the same for all of them.

### 3. Sandbox  `robot_agent/skills/sandbox.py`
`exec` with a restricted namespace: the `arm` object, `state`, `math`, `numpy`, a step budget
(max 10 s of sim time), no imports, no file or network access. Exceptions become feedback.

### 4. Rehearsal and effects  `robot_agent/skills/rehearse.py`, `effects.py`
Snapshot MuJoCo (`qpos, qvel, ctrl, act, eq_active`, backend `held`), run the skill, **measure**,
restore. Measurements are the safety input:

| measured | classification |
|---|---|
| anything left the table | irreversible → needs a yes |
| fragile object displaced or touched | caution |
| contact speed above threshold | caution |
| declared effect not achieved | rehearsal failed → metrics go back to the model to revise |
| objects moved that the request did not mention | reported as side effects |

This replaces per-verb safety rules with **outcome-based** rules. Safety stays deterministic code,
which was the team's principle; it just judges consequences instead of verb names.

### 5. Planner changes  `agent/llm.py`, `agent/planner.py`
- The tool catalogue is built from the registry at call time, not a fixed enum.
- New tool `define_skill(name, args, doc, effect, code)`.
- New loop: rehearsal metrics → model → revised code, at most 3 rounds.
- Composition: a skill may call other skills, so "clear the table" can call "sweep".

### 6. UI  `app/ui.py`, `app/main.py`
Code panel (rich Syntax), rehearsal verdict panel, "skill saved" note, `skills` command lists the
library with provenance. `--skills-dir` flag.

## What is buildable today vs. not

| skill | needs | today? |
|---|---|---|
| rotate (yaw) | nothing new | yes |
| knock over (push high on the object) | nothing new | yes |
| sweep (open gripper dragged along the surface) | nothing new | yes |
| pull (grasp, draw toward base, release) | nothing new | yes |
| stack / line up / sort (compositions) | nothing new | yes |
| throw | `speed` + `release_at` in the body | stretch, ~30 min, physics may look bad |
| lay on its side / tilt | `tilt` in the IK target | stretch, ~20 min, collisions near the table |
| break | MuJoCo has no rigid-body fracture; reinterpret as "knock a stack apart" | reframe only |
| two arms playing chess | second Panda in scene, chess set, python-chess for legality, turn orchestrator, two planners | **not today**: 4-6 hours. This is the "what's next" slide |

## Timeline for the rest of today

| time | who | what |
|---|---|---|
| 14:00-14:25 | Claude | body.py, registry.py, sandbox.py (new files, no conflicts) |
| 14:00-14:25 | teammate A | `speed` and `tilt` args on `PandaIKBackend.move_to`, `move_through` with `release_at` |
| 14:25-14:50 | Claude | rehearse.py, effects.py, outcome-based safety fallback |
| 14:50-15:10 | Claude | define_skill tool, revision loop, UI panels, persistence |
| 14:50-15:10 | teammate B | script the video, test commands live, prep social post + description |
| 15:10-15:25 | all | live test: rotate, knock over, sweep. Pick the two that work cleanly |
| 15:25-15:45 | B presents at show & tell (15:30) while A records the video |
| 15:45-16:20 | all | README, written description, social post, submit |

## Demo script (2 minutes)

1. "Rotate the red block 90 degrees" → no skill → code appears → rehearsal → yes → done → saved.
2. "Now the green one" → instant. **The robot learned it.**
3. "Knock the cup over" → new skill → rehearsal shows the cup leaving the table → irreversible → refused, safer version proposed.
4. `skills` → the library, with two entries that did not exist two minutes ago.

## Risks

- Generated code that fights the IK (unreachable poses) → sandbox step budget + clear error feedback.
- Rehearsal not perfectly restoring state → snapshot the full `MjData` fields listed above and `mj_forward` after.
- Time. If 15:10 arrives and rehearsal is not working, ship define_skill + confirm-always without rehearsal and say so honestly.

# Language-to-Action Robot Arm Agent

A home-assistant robot arm you talk to in plain language. It reads the actual scene before acting,
turns a vague request into a visible plan of typed physical actions, checks preconditions, asks before
anything irreversible, and verifies that each action actually worked.

> Built during the AI Tinkerers Hong Kong hackathon. Work in progress — see `CLAUDE.md` for the
> build plan and architecture.

## Why this cannot be a chatbox

1. **Grounding** — vague language is resolved against live scene state ("the cup", "somewhere safe").
2. **Physical consequences** — the agent reasons about irreversibility, which only exists with a body.
3. **Closed loop** — actions are verified against ground truth, and the agent replans on failure.

Safety classification is deterministic code, not LLM judgement: the planner proposes, but it cannot
approve or downgrade an unsafe action. Irreversible steps need a per-step `yes` from the user.

## Setup

macOS on Apple Silicon, Python 3.13.

```bash
python3 -m venv vlagent
source vlagent/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your OpenAI-compatible credentials
```

## Run

`mujoco.viewer.launch_passive` needs `mjpython` on macOS:

```bash
mjpython -m robot_agent.app.main                    # full loop
mjpython -m robot_agent.app.main --mock-llm         # fully offline
mjpython -m robot_agent.app.main --demo --record    # scripted demo, writes an mp4
```

Tests are headless and never open the viewer:

```bash
pytest
```

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco) — physics
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Franka Emika Panda model
- [mink](https://github.com/kevinzakka/mink) — differential inverse kinematics

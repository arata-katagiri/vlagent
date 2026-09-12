# Chess scenario (third scene)

*Added 2026-09-12 ~15:40 HKT; wired in at ~16:05.*

```bash
python .venv/bin/mjpython -m robot_agent.app.main --scene chess --safety              # one arm
python .venv/bin/mjpython -m robot_agent.app.main --scene chess_two_arm --safety      # arm a = white side, arm b = black side
python .venv/bin/mjpython -m robot_agent.app.main --scene chess --fen opening --safety # 29-piece "tenth turn" position
```

`--fen` takes a preset (`endgame`, `opening`) or any FEN string. The system prompt gets a table of
every square's centre coordinates, so the planner copies numbers instead of computing them.

Verified headless with gpt-4o: "move the white queen to d5" executes; "move the white king to e3" is
refused with the legal destinations listed. Two arms: "move the black king to d8" ends with arm b
doing it after arm a's attempt was out of reach and the loop replanned.

## Scene-editor examples (assets/scene_editor, MIT)

`scene_chess.xml` and `scene_chemistry_lab.xml` from
[markusgrotz/mujoco-scene-editor](https://github.com/markusgrotz/mujoco-scene-editor) are LLM-generated
MJCF made of primitives (no meshes, no textures). The chess one is a 31-body "tenth turn" position;
its FEN is the `opening` preset here (minus a knight the generator put on the same square as a pawn).
The chemistry one is a whole room: fume hood, cabinets, bench at 0.98 m, glassware as free bodies.
Neither is loaded yet; they are set dressing and object shapes for later (MjSpec `attach` can graft
their bodies into ours, with a registry entry per free body derived from its geoms).

## What it is

A chessboard on the table, pieces placed from a FEN string, and a legality rule: a piece may only
be set down on a square that is a legal chess move for it, captures go to the tray first, and
off-board points are refused. Same arm, same table, same primitives.

## Assets

`robot_agent/sim/assets/chess/` holds six Staunton piece meshes (`pawn.stl` ... `king.stl`) and one
board texture from [clarkerubber/Staunton-Pieces](https://github.com/clarkerubber/Staunton-Pieces),
MIT licence (`LICENSE.staunton-pieces.txt` alongside). Meshes are millimetres, height along +y,
base at y=0, 30 mm across, king 78 mm tall; the scene scales by 0.001 and rotates y->z.

Fallback if a different look is wanted: [quaternionmedia/scad-chess](https://github.com/quaternionmedia/scad-chess)
(CC-BY-4.0, parametric OpenSCAD, ships a 3MF).

Physics never touches the meshes: each piece is a plain cylinder collision body (radius 15 mm,
12 mm for pawns) hidden in render group 3, with the mesh as a visual-only geom on top. That is why
the existing grasp tuning works unchanged.

## Try it

```bash
source .venv/bin/activate
pytest tests/test_chess_scene.py -q            # 8 tests, headless
python -m robot_agent.sim.chess_demo           # build, render, one legal move
python -m robot_agent.sim.chess_demo --piece white_pawn_e2 --to e4
```

The demo writes `runs/chess_board.png` and `runs/chess_after_move.png`, prints every piece's
square, the legal destinations of the chosen piece, and runs `pick` + `place_at` through the same
`check_preconditions` / `execute` path the agent uses. Verified: white queen g2 -> d5, placed within
4 mm, landed on the right square.

## Module surface (`robot_agent/sim/chess_scene.py`)

| name | what |
|---|---|
| `SQUARE_M`, `BOARD_CENTER_M`, `BOARD_HALF_M` | 40 mm squares, board centred 0.50 m out, 0.32 m across |
| `square_xy("e4")` / `nearest_square(x, y)` | square <-> world; rank 1 nearest the robot, a-file on its left (+y) |
| `pieces_from_fen(fen)` | item specs; unique pieces are `white_king`, duplicates `black_pawn_e7` |
| `board_from_state(state, moving)` | python-chess `Board` reconstructed from where pieces stand |
| `legal_destinations(state, piece)` | squares the piece may go to right now |
| `precondition_hook(call, state)` | the legality rule, for `safety.PRECONDITION_HOOKS` |
| `build_spec()` | default builder + board + meshes; installed as `scene.build_spec` on activate |
| `activate(fen) -> restore` | swap registries, install hook, rules and builder; `restore()` undoes it |

## Integrating later

1. `main.py`: `--scene chess` -> `chess_scene.activate()` in the same place `--scene lab` activates,
   before `REGISTRY.load()` and before the world is built.
2. `DEFAULT_FEN` is a 7-piece endgame to keep the sim light. A full 32-piece board works (each piece
   is one free body and one convex hull) but slows compile and settle.
3. Adjacent occupied squares: the gripper opens to 80 mm and squares are 40 mm, so a top-down grasp
   between two occupied neighbours will touch them. Fix when needed: yaw the grasp so the fingers
   close along the free diagonal (the backend already takes `yaw_rad`), or pre-close to ~50 mm.
4. Turn tracking is by inference: `board_from_state` puts the moving piece on its last observed
   square and sets the side to move from its colour. A proper game loop should own a `chess.Board`
   and push moves after each verified placement.
5. A "play against the robot" demo needs a move chooser: `board.legal_moves` plus any scoring, or
   Stockfish via `chess.engine` if installed.

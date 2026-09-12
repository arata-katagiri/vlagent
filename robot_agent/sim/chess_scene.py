"""Third scenario: chess. Separate for now: nothing in the single-arm path
imports this, and there is no --scene flag for it yet.

What it provides
- An 8x8 board on the table, pieces placed from a FEN string, named so the
  planner can refer to them ("white_queen", "black_pawn_e7").
- Real Staunton meshes for looks (assets/chess/*.stl, MIT, clarkerubber), over
  plain cylinder collision bodies so the existing grasp tuning applies.
- Square <-> world coordinate helpers, and a nearest-square lookup.
- A chess-legality precondition hook (python-chess) for place_at / place_on:
  a piece may only land on a square that is a legal move for it, captures go
  to the tray first, off-board points are refused.
- activate(fen) that swaps the scene registries, installs the hooks and the
  board builder, and returns restore().

Wired in main.py as `--scene chess` (one arm) and `--scene chess_two_arm`
(two Pandas facing each other, the board between them: arm a plays from the
white side, arm b from the black side). `--fen` picks the position.
"""

from __future__ import annotations

import copy
import math
import struct
import sys
from pathlib import Path

import numpy as np

from ..actions import safety
from ..actions.schema import ActionCall
from ..agent import llm as _llm
from . import scene

try:  # optional: legality checks need python-chess
    import chess as _chess
except ImportError:  # pragma: no cover
    _chess = None

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "chess"

# ------------------------------------------------------------------- board

# 40 mm squares, centred 0.50 m out: the far corner is then 0.66 m from the
# base, inside the Panda's top-down reach (a 45 mm board reached a8 at 0.73 m
# and the approach pose there was unreachable).
SQUARE_M = 0.040
BOARD_CENTER_M = (0.50, 0.0)
BOARD_HALF_M = 4 * SQUARE_M          # 0.16
BOARD_LIFT_M = 0.0006                # board top sits this far above the table top
FILES = "abcdefgh"

# Real Staunton proportions (the meshes are 30 mm across, king 78 mm tall).
PIECE_HEIGHT_M = {
    "pawn": 0.043, "rook": 0.049, "knight": 0.059,
    "bishop": 0.059, "queen": 0.068, "king": 0.078,
}
PIECE_RADIUS_M = {"pawn": 0.012}
DEFAULT_RADIUS_M = 0.015
SIDE_RGBA = {
    "white": (0.93, 0.90, 0.82, 1.0),
    "black": (0.18, 0.15, 0.13, 1.0),
}
SYMBOL_TO_PIECE = {"p": "pawn", "r": "rook", "n": "knight", "b": "bishop", "q": "queen", "k": "king"}

# A small endgame: enough pieces to be a board, few enough to keep the sim light.
DEFAULT_FEN = "4k3/3pp3/8/8/8/8/3PP1Q1/4K3 w - - 0 1"
# The "around the tenth turn" position from mujoco-scene-editor's generated
# scene_chess.xml (assets/scene_editor, MIT), transcribed to FEN. The generated
# scene put a black knight and a white pawn both on e4; the knight is dropped.
# 29 pieces: fine for a still or a video, slow to settle, crowded to grasp.
OPENING_FEN = "r2qkb1r/pppp1ppp/2n2b2/3P4/2B1P3/2N2N2/PP3PPP/R1BQR1K1 b kq - 0 10"
PRESETS = {"endgame": DEFAULT_FEN, "opening": OPENING_FEN}

# Filled by activate(): name -> item spec of every piece on the board.
PIECES: dict[str, dict] = {}

TRAY_ALIASES = ("the tray", "captured pieces", "the graveyard", "somewhere safe", "safe place")


def square_xy(square: str) -> tuple[float, float]:
    """World (x, y) of a square's centre. Rank 1 is nearest the robot; the
    a-file is on the robot's left (+y)."""
    f, r = FILES.index(square[0]), int(square[1]) - 1
    x = BOARD_CENTER_M[0] - BOARD_HALF_M + (r + 0.5) * SQUARE_M
    y = BOARD_CENTER_M[1] + BOARD_HALF_M - (f + 0.5) * SQUARE_M
    return x, y


def nearest_square(x: float, y: float) -> str | None:
    """Square containing world (x, y), or None when off the board."""
    r = (x - (BOARD_CENTER_M[0] - BOARD_HALF_M)) / SQUARE_M
    f = ((BOARD_CENTER_M[1] + BOARD_HALF_M) - y) / SQUARE_M
    if not (0 <= r < 8 and 0 <= f < 8):
        return None
    return f"{FILES[int(f)]}{int(r) + 1}"


# ------------------------------------------------------------------ pieces

def pieces_from_fen(fen: str = DEFAULT_FEN) -> dict[str, dict]:
    """Scene item specs for every piece in a FEN placement.

    Names are stable identities: unique pieces are "white_king"; duplicated
    ones carry their starting square, "black_pawn_e7". Extra keys (piece, side,
    square, mesh) are ignored by world_state and used by the board builder.
    """
    placement = fen.split()[0]
    found: list[tuple[str, str, str]] = []  # (side, piece, square)
    for rank_idx, row in enumerate(placement.split("/")):
        rank = 8 - rank_idx
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            side = "white" if ch.isupper() else "black"
            found.append((side, SYMBOL_TO_PIECE[ch.lower()], f"{FILES[file_idx]}{rank}"))
            file_idx += 1

    counts: dict[tuple[str, str], int] = {}
    for side, piece, _ in found:
        counts[(side, piece)] = counts.get((side, piece), 0) + 1

    items: dict[str, dict] = {}
    for side, piece, sq in found:
        name = f"{side}_{piece}" if counts[(side, piece)] == 1 else f"{side}_{piece}_{sq}"
        x, y = square_xy(sq)
        h = PIECE_HEIGHT_M[piece]
        r = PIECE_RADIUS_M.get(piece, DEFAULT_RADIUS_M)
        items[name] = dict(
            kind="cylinder", half=(r, h / 2), rgba=SIDE_RGBA[side],
            pos=(x, y, 0.0), color=side, fragile=False, graspable=True,
            aliases=(f"{side} {piece}", f"the {side} {piece}", f"{piece} on {sq}", f"the {piece} on {sq}"),
            piece=piece, side=side, square=sq, mesh=piece,
        )
    return items


# ----------------------------------------------------------------- policy

CHESS_RULES = """\
- Chess scenario: the objects are chess pieces on an 8x8 board (squares a1-h8;
  rank 1 is nearest the robot, the a-file is on the robot's left). Move a piece
  with pick then place_at the destination square's coordinates. Only legal chess
  moves are allowed; the checker verifies legality. To capture, first pick the
  captured piece and place_on the tray, then move the capturing piece. Never
  set a piece down off the board except on the tray.
"""

CHESS_RULES_TWO_ARM = """\
- Two arms play: arm "a" sits on the white side (ranks 1-4 are nearest it),
  arm "b" on the black side (ranks 5-8 nearest it). Each arm reaches about
  four ranks comfortably; pick the arm whose side the destination is on. The
  captured-pieces tray is beside the board on the a-file side.
"""

_last_square: dict[str, str] = {}


def square_table_rule() -> str:
    """Every square's centre, so the planner copies coordinates instead of
    computing them (it got d6 for d5 and d7 for d8 when left to arithmetic)."""
    rows = []
    for r in range(8, 0, -1):
        cells = []
        for f in FILES:
            x, y = square_xy(f"{f}{r}")
            cells.append(f"{f}{r}=({x:.3f},{y:.3f})")
        rows.append("  " + " ".join(cells))
    return "- Square centres in metres (x, y); use these exact numbers for place_at:\n" + "\n".join(rows) + "\n"


def _observe(state: dict) -> None:
    for name, obj in state["objects"].items():
        if name == "tray" or obj["held"] or not obj["on_table"]:
            continue
        sq = nearest_square(*obj["position_m"][:2])
        if sq:
            _last_square[name] = sq


def board_from_state(state: dict, moving: str | None = None):
    """Reconstruct a python-chess Board from where the pieces stand. The held
    piece (if any) is placed on its last observed square."""
    if _chess is None:
        return None
    board = _chess.Board(None)
    for name, obj in state["objects"].items():
        item = PIECES.get(name)
        if item is None:
            continue
        if obj["held"] or not obj["on_table"]:
            sq = _last_square.get(name) if name == moving else None
        else:
            sq = nearest_square(*obj["position_m"][:2])
        if sq is None:
            continue
        symbol = item["piece"][0] if item["piece"] != "knight" else "n"
        symbol = symbol.upper() if item["side"] == "white" else symbol
        board.set_piece_at(_chess.parse_square(sq), _chess.Piece.from_symbol(symbol))
    if moving in PIECES:
        board.turn = _chess.WHITE if PIECES[moving]["side"] == "white" else _chess.BLACK
    return board


def legal_destinations(state: dict, piece_name: str) -> list[str]:
    board = board_from_state(state, moving=piece_name)
    origin = _last_square.get(piece_name)
    if board is None or origin is None:
        return []
    src = _chess.parse_square(origin)
    return sorted({_chess.square_name(m.to_square) for m in board.legal_moves if m.from_square == src})


def precondition_hook(call: ActionCall, state: dict) -> list[str]:
    """Chess legality on the predicted landing square of a built-in action."""
    _observe(state)
    name, args = call.name, call.args
    holding = safety.holding_of(state, args.get("arm"))  # two-arm aware
    if holding not in PIECES:
        return []
    if name == "place_on":
        return [] if args.get("target") == "tray" else [
            f"{holding} can only be set down on a board square (place_at) or on the tray"
        ]
    if name != "place_at":
        return []
    try:
        x, y = float(args["x_m"]), float(args["y_m"])
    except (KeyError, TypeError, ValueError):
        return []
    dest = nearest_square(x, y)
    if dest is None:
        return [f"({x:.2f}, {y:.2f}) m is off the board; captured pieces go on the tray"]
    cx, cy = square_xy(dest)
    if math.dist((x, y), (cx, cy)) > SQUARE_M * 0.35:
        return [f"({x:.2f}, {y:.2f}) m is not centred on a square; {dest} is at ({cx:.3f}, {cy:.3f})"]
    for other, obj in state["objects"].items():
        if other != holding and obj["on_table"] and not obj["held"] and nearest_square(*obj["position_m"][:2]) == dest:
            if PIECES.get(other, {}).get("side") == PIECES[holding]["side"]:
                return [f"{dest} is occupied by {other}, a piece of the same side; not a legal destination"]
            return [f"{dest} is occupied by {other}; to capture it, first pick {other} and place_on the tray, then move {holding}"]
    if _chess is None:
        return []
    origin = _last_square.get(holding)
    if origin is None:
        return []
    board = board_from_state(state, moving=holding)
    move = _chess.Move(_chess.parse_square(origin), _chess.parse_square(dest))
    if move not in board.legal_moves:
        legal = ", ".join(legal_destinations(state, holding)) or "nowhere"
        return [f"{holding} cannot move from {origin} to {dest}; legal destinations: {legal}"]
    return []


# ------------------------------------------------------------- board build

def _stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    arr = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    v = arr["v"].reshape(-1, 3)
    return v.min(0), v.max(0)


def build_spec():
    """The default builder (with the swapped-in pieces as cylinders), plus the
    board and a Staunton mesh over each piece."""
    spec = _default_build_spec()
    _add_board_and_meshes(spec, scene.OBJECTS, scene.TABLE_BOUNDS_M["top_z"])
    return spec


def build_two_arm_spec():
    """The two-arm builder (with the pieces as cylinders), plus board and meshes."""
    from . import two_arm_scene as ta  # noqa: PLC0415

    spec = _default_two_arm_build_spec()
    _add_board_and_meshes(spec, ta.OBJECTS2, ta.TABLE_BOUNDS_M["top_z"])
    return spec


def _add_board_and_meshes(spec, objects: dict, top_z: float) -> None:
    import mujoco  # noqa: PLC0415

    world = spec.worldbody

    tex = spec.add_texture()
    tex.name = "chess_checker"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.88, 0.80, 0.64]
    tex.rgb2 = [0.38, 0.25, 0.16]
    tex.width = tex.height = 512
    mat = spec.add_material()
    mat.name = "chessboard"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chess_checker"
    mat.texrepeat = [4.0, 4.0]        # the builtin checker is 2x2 per tile
    mat.texuniform = False

    half_t = 0.003
    board = world.add_body()
    board.name = "chessboard"
    board.pos = [BOARD_CENTER_M[0], BOARD_CENTER_M[1], top_z + BOARD_LIFT_M - half_t]
    bg = board.add_geom()
    bg.name = "chessboard_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = [BOARD_HALF_M, BOARD_HALF_M, half_t]
    bg.material = "chessboard"
    frame = board.add_geom()
    frame.name = "chessboard_frame"
    frame.type = mujoco.mjtGeom.mjGEOM_BOX
    frame.size = [BOARD_HALF_M + 0.012, BOARD_HALF_M + 0.012, half_t - 0.0005]
    frame.pos = [0.0, 0.0, -0.0005]
    frame.rgba = [0.25, 0.16, 0.10, 1.0]
    frame.contype = frame.conaffinity = 0

    # Absolute mesh paths: setting spec.meshdir would redirect the Panda's own
    # link meshes away from the Menagerie assets directory.
    loaded: set[str] = set()
    for name, item in objects.items():
        piece = item.get("mesh")
        if not piece:
            continue
        stl = ASSET_DIR / f"{piece}.stl"
        if not stl.exists():
            continue
        if piece not in loaded:
            m = spec.add_mesh()
            m.name = f"mesh_{piece}"
            m.file = stl.as_posix()
            m.scale = [0.001, 0.001, 0.001]
            m.maxhullvert = 32
            loaded.add(piece)
        body = spec.body(name)
        # The cylinder stays as the collision body but is hidden from the render.
        for g in body.geoms:
            g.group = 3
        lo, hi = _stl_bounds(stl)
        vis = body.add_geom()
        vis.name = f"{name}_mesh"
        vis.type = mujoco.mjtGeom.mjGEOM_MESH
        vis.meshname = f"mesh_{piece}"
        vis.contype = vis.conaffinity = 0
        vis.rgba = list(item["rgba"])
        # STL is millimetres, height along +y, base at y=0: rotate y->z, centre
        # x/z, and drop the base to the bottom of the body (origin at mid-height).
        s = 0.001
        vis.quat = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
        cx, cz = 0.5 * (lo[0] + hi[0]) * s, 0.5 * (lo[2] + hi[2]) * s
        vis.pos = [-cx, cz, -item["half"][1] - lo[1] * s]


_default_build_spec = scene.build_spec
_default_two_arm_build_spec = None  # bound on first two-arm activation

# -------------------------------------------------------------- activation


def activate(fen: str = DEFAULT_FEN, two_arm: bool = False):
    """Swap the chess pieces into the live scene registries, install the
    legality hook, prompt rules and board builder. Returns restore().

    two_arm=True targets robot_agent.sim.two_arm_scene instead: the board is
    centred between the arms, the tray moves beside it, and the two-arm builder
    is wrapped to add the board and meshes.
    """
    global BOARD_CENTER_M, BOARD_HALF_M, _default_two_arm_build_spec
    fen = PRESETS.get(fen, fen)
    saved = dict(
        center=BOARD_CENTER_M,
        stackable=safety.STACKABLE_TARGETS,
        rules=[CHESS_RULES] + ([CHESS_RULES_TWO_ARM] if two_arm else []),
    )
    if two_arm:
        from . import two_arm_scene as ta  # noqa: PLC0415

        BOARD_CENTER_M = (ta._CX, ta._CY)
        saved.update(
            objects2=copy.deepcopy(ta.OBJECTS2), items2=copy.deepcopy(ta.ITEMS2),
            graspable2=ta.GRASPABLE2, tray2=copy.deepcopy(ta.TRAY2),
            build2=ta.build_two_arm_spec,
        )
    pieces = pieces_from_fen(fen)
    PIECES.clear()
    PIECES.update(pieces)

    if two_arm:
        top_z = ta.TABLE_BOUNDS_M["top_z"]
        # Registries are mutated in place: two_arm_world_state holds ITEMS2.
        ta.OBJECTS2.clear()
        ta.OBJECTS2.update(pieces)
        ta.GRASPABLE2 = tuple(pieces)
        backend = sys.modules.get("robot_agent.sim.two_arm_backend")
        if backend is not None:  # it binds GRASPABLE2 at import
            backend.GRASPABLE2 = ta.GRASPABLE2
        # Tray beside the board on the a-file side, out of both arms' way.
        ta.TRAY2.update(half=(0.07, 0.07, 0.006), pos=(BOARD_CENTER_M[0], 0.29, top_z + 0.006))
        tray = dict(saved["items2"]["tray"], half=ta.TRAY2["half"], pos=ta.TRAY2["pos"], aliases=TRAY_ALIASES)
        ta.ITEMS2.clear()
        ta.ITEMS2.update(pieces)
        ta.ITEMS2["tray"] = tray
        if _default_two_arm_build_spec is None:
            _default_two_arm_build_spec = ta.build_two_arm_spec
        ta.build_two_arm_spec = build_two_arm_spec
    else:
        saved.update(
            objects=copy.deepcopy(scene.OBJECTS), items=copy.deepcopy(scene.ITEMS),
            graspable=scene.GRASPABLE, build_spec=scene.build_spec,
        )
        scene.OBJECTS.clear()
        scene.OBJECTS.update(pieces)
        scene.GRASPABLE = tuple(pieces)
        tray = dict(saved["items"]["tray"], aliases=TRAY_ALIASES)
        scene.ITEMS.clear()
        scene.ITEMS.update(scene.OBJECTS)
        scene.ITEMS["tray"] = tray
        scene.build_spec = build_spec

    safety.STACKABLE_TARGETS = ("tray",)
    safety.PRECONDITION_HOOKS.append(precondition_hook)
    saved["rules"].append(square_table_rule())
    _llm.SCENARIO_RULES.extend(saved["rules"])
    _last_square.clear()
    _last_square.update({n: it["square"] for n, it in pieces.items()})

    def restore():
        global BOARD_CENTER_M
        BOARD_CENTER_M = saved["center"]
        safety.STACKABLE_TARGETS = saved["stackable"]
        if two_arm:
            ta.OBJECTS2.clear(); ta.OBJECTS2.update(saved["objects2"])
            ta.ITEMS2.clear(); ta.ITEMS2.update(saved["items2"])
            ta.GRASPABLE2 = saved["graspable2"]
            backend = sys.modules.get("robot_agent.sim.two_arm_backend")
            if backend is not None:
                backend.GRASPABLE2 = saved["graspable2"]
            ta.TRAY2.clear(); ta.TRAY2.update(saved["tray2"])
            ta.build_two_arm_spec = saved["build2"]
        else:
            scene.OBJECTS.clear(); scene.OBJECTS.update(saved["objects"])
            scene.ITEMS.clear(); scene.ITEMS.update(saved["items"])
            scene.GRASPABLE = saved["graspable"]
            scene.build_spec = saved["build_spec"]
        if precondition_hook in safety.PRECONDITION_HOOKS:
            safety.PRECONDITION_HOOKS.remove(precondition_hook)
        for r in saved["rules"]:
            if r in _llm.SCENARIO_RULES:
                _llm.SCENARIO_RULES.remove(r)
        PIECES.clear()
        _last_square.clear()

    return restore


__all__ = [
    "SQUARE_M", "BOARD_CENTER_M", "BOARD_HALF_M", "DEFAULT_FEN", "OPENING_FEN", "PRESETS",
    "CHESS_RULES", "CHESS_RULES_TWO_ARM", "PIECES", "build_two_arm_spec",
    "square_xy", "nearest_square", "pieces_from_fen", "board_from_state",
    "legal_destinations", "precondition_hook", "build_spec", "activate",
]

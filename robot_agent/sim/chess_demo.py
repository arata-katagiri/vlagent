"""Headless chess-scene check: build the board, render it, and have the Panda
make one legal move. Run: python -m robot_agent.sim.chess_demo [--fen FEN]

Writes runs/chess_board.png (demo camera) and prints where the piece landed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ..actions.schema import ActionCall
from ..actions.safety import check_preconditions, set_policy_enabled
from . import chess_scene, scene
from .backends import PandaIKBackend
from .world_state import get_world_state

RUNS = Path(__file__).resolve().parents[2] / "runs"


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer (stdlib only) so the demo needs no image library."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=chess_scene.DEFAULT_FEN)
    ap.add_argument("--piece", default="white_queen")
    ap.add_argument("--to", default=None, help="destination square (default: first legal)")
    args = ap.parse_args()

    set_policy_enabled(True)
    restore = chess_scene.activate(args.fen)
    try:
        model, data = scene.build_scene()
        scene.settle(model, data, 0.5)
        RUNS.mkdir(exist_ok=True)
        r = mujoco.Renderer(model, 720, 960)
        r.update_scene(data, camera=scene.CAMERA_NAME)
        _write_png(RUNS / "chess_board.png", r.render())
        print(f"rendered {RUNS / 'chess_board.png'}")

        get = lambda: get_world_state(model, data)  # noqa: E731
        state = get()
        squares = {n: chess_scene.nearest_square(*o["position_m"][:2])
                   for n, o in state["objects"].items() if n != "tray"}
        print("pieces:", squares)

        legal = chess_scene.legal_destinations(state, args.piece)
        print(f"legal destinations for {args.piece}: {legal}")
        # Default to the legal square nearest the board centre: the arm's sweet spot.
        cx, cy = chess_scene.BOARD_CENTER_M
        dest = args.to or (min(legal, key=lambda sq: np.hypot(*(np.subtract(chess_scene.square_xy(sq), (cx, cy)))))
                           if legal else None)
        if dest is None:
            print("no move to make"); return 1
        x, y = chess_scene.square_xy(dest)

        # The same two steps the agent would run, through the same checks.
        backend = PandaIKBackend(model, data, sync=lambda: None)
        from ..actions.executor import execute  # noqa: PLC0415

        for call in (ActionCall("pick", {"object": args.piece}, "demo"),
                     ActionCall("place_at", {"x_m": x, "y_m": y}, "demo")):
            problems = check_preconditions(call, get())
            if problems:
                print(f"{call.name} refused: {problems}"); return 1
            res = execute(call, backend, get)
            print(f"{call.name}: {'OK' if res.ok else 'FAILED'} {res.reason}")
            if not res.ok:
                return 1
        landed = chess_scene.nearest_square(*get()["objects"][args.piece]["position_m"][:2])
        print(f"{args.piece} now on {landed} (wanted {dest})")
        r.update_scene(data, camera=scene.CAMERA_NAME)
        _write_png(RUNS / "chess_after_move.png", r.render())
        return 0 if landed == dest else 1
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(main())

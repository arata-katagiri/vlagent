"""Offscreen rendering. Never imports mujoco.viewer, so tests can use it.

Three jobs:
  png       - one still, for a visual checkpoint
  filmstrip - N frames of a trajectory tiled into one image, so motion can be
              checked from a single picture (approach, descend, close, lift)
  recorder  - frames piped to ffmpeg for the 2-minute submission video

PNG is encoded with the standard library (zlib + struct). That is a dozen lines
and saves adding Pillow, which is not in the venv.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
WIDTH, HEIGHT = 960, 720
FILMSTRIP_GRID = (3, 2)  # cols, rows


def write_png(rgb: np.ndarray, path: str | Path) -> Path:
    """Write an (H, W, 3) uint8 array as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    # Each scanline is prefixed with filter type 0 (None).
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


class Camera:
    """Reusable offscreen renderer bound to one named camera."""

    def __init__(self, model, camera: str | None = None, width=WIDTH, height=HEIGHT):
        self._renderer = mujoco.Renderer(model, height, width)
        self._camera = camera if camera is not None else -1

    def frame(self, data) -> np.ndarray:
        self._renderer.update_scene(data, camera=self._camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def png(model, data, path: str | Path, camera: str | None = None) -> Path:
    """Render one frame to path."""
    with Camera(model, camera) as cam:
        return write_png(cam.frame(data), path)


def filmstrip(
    frames: list[np.ndarray], path: str | Path, grid: tuple[int, int] = FILMSTRIP_GRID
) -> Path:
    """Tile frames into a single PNG, left to right then top to bottom.

    This is how motion gets checked without watching the viewer: six frames of a
    pick show approach, descent, close and lift in one image.
    """
    if not frames:
        raise ValueError("no frames to tile")
    cols, rows = grid
    h, w, _ = frames[0].shape
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames[: cols * rows]):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = f
        sheet[r * h : r * h + 2, c * w : (c + 1) * w] = 255  # divider
        sheet[r * h : (r + 1) * h, c * w : c * w + 2] = 255
    return write_png(sheet, path)


def recorder(path: str | Path, fps: int = 30):
    """Context manager writing frames to an mp4 through ffmpeg."""
    raise NotImplementedError("Phase 5")


__all__ = ["Camera", "png", "filmstrip", "write_png", "recorder", "RUNS_DIR", "WIDTH", "HEIGHT"]

"""Offscreen rendering. Never imports mujoco.viewer, so tests can use it.

Three jobs:
  png       - one still, for a visual checkpoint
  filmstrip - N frames of a trajectory tiled into one image, so motion can be
              checked from a single picture (approach, descend, close, lift)
  recorder  - frames piped to ffmpeg for the 2-minute submission video
"""

from __future__ import annotations

from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
WIDTH, HEIGHT = 960, 720
FILMSTRIP_GRID = (3, 2)  # cols, rows


def png(model, data, path: Path, camera: str | None = None) -> Path:
    """Render one frame to path."""
    raise NotImplementedError("Phase 1")


def filmstrip(frames: list, path: Path, grid: tuple[int, int] = FILMSTRIP_GRID) -> Path:
    """Tile captured frames into a single PNG."""
    raise NotImplementedError("Phase 1")


def recorder(path: Path, fps: int = 30):
    """Context manager writing frames to an mp4 through ffmpeg."""
    raise NotImplementedError("Phase 5")

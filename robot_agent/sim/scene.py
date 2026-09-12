"""Builds the scene MJCF: Panda + table + objects + inactive welds.

The Panda comes from the robot_descriptions package (panda_mj_description),
which reads from the MuJoCo Menagerie clone cached under ~/.cache. Falls back to
a local mujoco_menagerie checkout via the MENAGERIE_PATH env var, so the scene
still loads with the venue Wi-Fi down.

Read the real body / site / joint / actuator names out of the MJCF. Do not guess.
"""

from __future__ import annotations

import os
from pathlib import Path

# Named camera used by the viewer, the offscreen renderer and the recorder, so
# every image of this project is framed identically.
CAMERA_NAME = "demo"


def panda_xml_path() -> Path:
    """Locate the Menagerie Panda MJCF, preferring the local cache."""
    override = os.environ.get("MENAGERIE_PATH")
    if override:
        candidate = Path(override) / "franka_emika_panda" / "scene.xml"
        if candidate.exists():
            return candidate
    from robot_descriptions import panda_mj_description  # noqa: PLC0415

    return Path(panda_mj_description.MJCF_PATH)


def build_scene():
    """Return (model, data) for the full scene."""
    raise NotImplementedError("Phase 1")

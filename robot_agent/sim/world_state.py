"""Ground truth MuJoCo state -> plain structured dict.

This is the only place that reads simulator internals for the agent. Everything
downstream (safety, executor, planner) consumes the dict returned here, which is
what lets those layers be built and tested without a simulator.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for the table extents, in metres. Filled in Phase 1
# once the scene MJCF exists.
TABLE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.0,
    "x_max": 0.0,
    "y_min": 0.0,
    "y_max": 0.0,
    "top_z": 0.0,
}

# name -> {aliases, color, fragile, graspable, size_m}. Filled in Phase 1.
OBJECTS: dict[str, dict[str, Any]] = {}


def get_world_state(model, data) -> dict:
    """Return the full scene state as a plain dict.

    Shape::

        {"objects": {"<name>": {"position_m": [x, y, z], "color": str, "fragile": bool,
                                "on_table": bool, "on_top_of": str | None,
                                "supporting": [str], "near_table_edge": bool,
                                "held": bool}},
         "gripper": {"position_m": [x, y, z], "holding": str | None},
         "table_bounds_m": {"x_min": ..., "x_max": ..., "y_min": ...,
                            "y_max": ..., "top_z": ...}}
    """
    raise NotImplementedError("Phase 1")

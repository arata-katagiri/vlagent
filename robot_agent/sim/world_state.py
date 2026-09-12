"""Ground truth MuJoCo state -> plain structured dict.

This is the only place that reads simulator internals for the agent. Everything
downstream (safety, executor, planner) consumes the dict returned here, which is
what lets those layers be built and tested without a simulator.
"""

from __future__ import annotations

import numpy as np

from .scene import (
    GRASP_SITE,
    ITEMS,
    TABLE_BOUNDS_M,
    footprint,
    half_height,
)

# How close to the table edge counts as "near the edge", in metres.
EDGE_MARGIN_M = 0.06

# Vertical slack when deciding whether A rests on B, in metres.
SUPPORT_TOL_M = 0.012


def _on_table(pos: np.ndarray) -> bool:
    t = TABLE_BOUNDS_M
    return bool(
        t["x_min"] <= pos[0] <= t["x_max"]
        and t["y_min"] <= pos[1] <= t["y_max"]
        and pos[2] > t["top_z"] - 0.05
    )


def _near_edge(pos: np.ndarray) -> bool:
    t = TABLE_BOUNDS_M
    if not _on_table(pos):
        return False
    return bool(
        min(
            pos[0] - t["x_min"],
            t["x_max"] - pos[0],
            pos[1] - t["y_min"],
            t["y_max"] - pos[1],
        )
        <= EDGE_MARGIN_M
    )


def _resting_on(name: str, pos: np.ndarray, positions: dict[str, np.ndarray]) -> str | None:
    """Name of the item directly supporting `name`, or None for the table/air."""
    bottom = pos[2] - half_height(ITEMS[name])
    best: tuple[float, str] | None = None
    for other, opos in positions.items():
        if other == name:
            continue
        top = opos[2] + half_height(ITEMS[other])
        if abs(bottom - top) > SUPPORT_TOL_M:
            continue
        hx, hy = footprint(ITEMS[other])
        if abs(pos[0] - opos[0]) <= hx + 0.01 and abs(pos[1] - opos[1]) <= hy + 0.01:
            gap = abs(bottom - top)
            if best is None or gap < best[0]:
                best = (gap, other)
    return best[1] if best else None


def held_object(model, data) -> str | None:
    """Which object an active weld is currently holding, if any."""
    for i in range(model.neq):
        if not data.eq_active[i]:
            continue
        name = model.equality(i).name
        if name.startswith("weld_"):
            return name[len("weld_") :]
    return None


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
    positions = {name: np.asarray(data.body(name).xpos, dtype=float) for name in ITEMS}
    holding = held_object(model, data)

    objects: dict[str, dict] = {}
    for name, pos in positions.items():
        item = ITEMS[name]
        objects[name] = {
            "position_m": [round(float(v), 4) for v in pos],
            "color": item["color"],
            "fragile": bool(item["fragile"]),
            "graspable": bool(item["graspable"]),
            "on_table": _on_table(pos),
            "on_top_of": _resting_on(name, pos, positions),
            "supporting": [],
            "near_table_edge": _near_edge(pos),
            "held": name == holding,
        }

    # supporting is the inverse of on_top_of, filled in once all entries exist.
    for name, entry in objects.items():
        parent = entry["on_top_of"]
        if parent is not None:
            objects[parent]["supporting"].append(name)

    return {
        "objects": objects,
        "gripper": {
            "position_m": [
                round(float(v), 4) for v in np.asarray(data.site(GRASP_SITE).xpos)
            ],
            "holding": holding,
        },
        "table_bounds_m": dict(TABLE_BOUNDS_M),
    }


__all__ = ["get_world_state", "held_object", "EDGE_MARGIN_M", "TABLE_BOUNDS_M", "ITEMS"]

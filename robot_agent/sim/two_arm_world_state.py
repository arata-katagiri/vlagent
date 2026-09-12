"""Ground truth of the two-arm scene -> plain structured dict.

Same shape as world_state.get_world_state, plus an "arms" key::

    {"objects": {...},                       # identical shape to the single-arm scene
     "arms": {"a": {"position_m": [x,y,z], "holding": str|None, "base_m": [x,y]},
              "b": {...}},
     "gripper": {...},                       # compatibility view, see below
     "table_bounds_m": {...}}

`gripper` is kept so that code which predates two arms -- scenario hooks, the
skills sandbox -- keeps working. It mirrors whichever arm is currently holding
something, else the first arm. Anything that actually cares which arm did what
must read "arms".
"""

from __future__ import annotations

import numpy as np

from .two_arm_scene import ARMS, GRASP_SITE, ITEMS2, TABLE_BOUNDS_M, footprint, half_height

EDGE_MARGIN_M = 0.06
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
        min(pos[0] - t["x_min"], t["x_max"] - pos[0],
            pos[1] - t["y_min"], t["y_max"] - pos[1]) <= EDGE_MARGIN_M
    )


def _resting_on(name: str, pos: np.ndarray, positions: dict[str, np.ndarray]) -> str | None:
    bottom = pos[2] - half_height(ITEMS2[name])
    best: tuple[float, str] | None = None
    for other, opos in positions.items():
        if other == name:
            continue
        top = opos[2] + half_height(ITEMS2[other])
        if abs(bottom - top) > SUPPORT_TOL_M:
            continue
        hx, hy = footprint(ITEMS2[other])
        if abs(pos[0] - opos[0]) <= hx + 0.01 and abs(pos[1] - opos[1]) <= hy + 0.01:
            gap = abs(bottom - top)
            if best is None or gap < best[0]:
                best = (gap, other)
    return best[1] if best else None


def held_by(model, data, prefix: str) -> str | None:
    """Which object this arm's active weld is holding, if any."""
    tag = f"{prefix}weld_"
    for i in range(model.neq):
        if not data.eq_active[i]:
            continue
        name = model.equality(i).name
        if name.startswith(tag):
            return name[len(tag):]
    return None


def get_two_arm_world_state(model, data) -> dict:
    positions = {name: np.asarray(data.body(name).xpos, dtype=float) for name in ITEMS2}
    held = {key: held_by(model, data, arm["prefix"]) for key, arm in ARMS.items()}
    all_held = {v for v in held.values() if v}

    objects: dict[str, dict] = {}
    for name, pos in positions.items():
        item = ITEMS2[name]
        hx, hy = footprint(item)
        objects[name] = {
            "position_m": [round(float(v), 4) for v in pos],
            "half_extent_m": [round(hx, 4), round(hy, 4), round(half_height(item), 4)],
            "color": item["color"],
            "fragile": bool(item["fragile"]),
            "graspable": bool(item["graspable"]),
            "on_table": _on_table(pos),
            "on_top_of": _resting_on(name, pos, positions),
            "supporting": [],
            "near_table_edge": _near_edge(pos),
            "held": name in all_held,
            "held_by": next((k for k, v in held.items() if v == name), None),
        }
    for name, entry in objects.items():
        parent = entry["on_top_of"]
        if parent is not None:
            objects[parent]["supporting"].append(name)

    arms: dict[str, dict] = {}
    for key, arm in ARMS.items():
        site = data.site(f"{arm['prefix']}{GRASP_SITE}").xpos
        arms[key] = {
            "position_m": [round(float(v), 4) for v in np.asarray(site)],
            "holding": held[key],
            "base_m": [round(float(v), 3) for v in arm["base"][:2]],
        }

    busy = next((k for k in arms if arms[k]["holding"]), next(iter(arms)))
    return {
        "objects": objects,
        "arms": arms,
        "gripper": dict(arms[busy]),
        "table_bounds_m": dict(TABLE_BOUNDS_M),
    }


__all__ = ["get_two_arm_world_state", "held_by", "EDGE_MARGIN_M", "TABLE_BOUNDS_M", "ITEMS2"]

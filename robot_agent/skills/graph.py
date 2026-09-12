"""The skill graph: measured similarity, composition edges, and the model's opinion.

Three kinds of edge, in order of trust:
  calls         which skills a skill invoked while it ran (recorded, not declared)
  nearest       motion-signature distance: where the tool centre point went relative
                to the object, what the fingers did, and what changed (all measured)
  derived_from  the model's own opinion when it wrote the skill (a label, never a judge)
"""

from __future__ import annotations

import math

N_PTS = 16
W_PATH, W_GRIP, W_EFFECT = 1.0, 0.05, 0.10


def _resample(seq: list, n: int) -> list:
    if not seq:
        return [seq[0] if seq else 0.0] * n if seq else [0.0] * n
    if len(seq) == 1:
        return [seq[0]] * n
    out = []
    for i in range(n):
        t = i * (len(seq) - 1) / (n - 1)
        lo, hi = int(math.floor(t)), min(int(math.ceil(t)), len(seq) - 1)
        a, b = seq[lo], seq[hi]
        f = t - lo
        if isinstance(a, (list, tuple)):
            out.append([a[k] + (b[k] - a[k]) * f for k in range(len(a))])
        else:
            out.append(a + (b - a) * f)
    return out


def motion_signature(path: list, grip: list, anchor, metrics: dict) -> list[float]:
    """A fixed-length vector recorded from what the arm actually did.

    path: tool centre point samples; grip: 0/1 closed flags aligned with path;
    anchor: the acted-on object's initial position, so the path is relative to it.
    """
    if not path:
        rel = [[0.0, 0.0, 0.0]] * N_PTS
    else:
        ax, ay, az = (anchor if anchor is not None else path[0])
        rel = _resample([[p[0] - ax, p[1] - ay, p[2] - az] for p in path], N_PTS)
    flat = [round(float(v), 4) for pt in rel for v in pt]
    g = _resample([float(v) for v in grip] if grip else [0.0], N_PTS)
    effect = [
        1.0 if metrics.get("moved") else 0.0,
        1.0 if metrics.get("rotated") else 0.0,
        1.0 if metrics.get("tipped") else 0.0,
        1.0 if metrics.get("touched") else 0.0,
        1.0 if metrics.get("off_table") else 0.0,
    ]
    return flat + [round(float(v), 3) for v in g] + effect


def distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    n_path = N_PTS * 3
    d_path = math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:n_path], b[:n_path])) / N_PTS)
    d_grip = sum(abs(x - y) for x, y in zip(a[n_path:n_path + N_PTS], b[n_path:n_path + N_PTS])) / N_PTS
    d_eff = sum(abs(x - y) for x, y in zip(a[n_path + N_PTS:], b[n_path + N_PTS:]))
    return round(W_PATH * d_path + W_GRIP * d_grip + W_EFFECT * d_eff, 4)


def nearest(skills: dict, vec: list[float], k: int = 2, exclude=()) -> list[tuple[str, float]]:
    scored = [(s.name, distance(vec, s.signature_vec)) for s in skills.values()
              if s.name not in exclude and s.signature_vec]
    scored = [(n, d) for n, d in scored if d != float("inf")]
    return sorted(scored, key=lambda t: t[1])[:k]


def graph_lines(skills: dict) -> list[str]:
    """One line per skill: what it calls, what the world says it resembles, what the model said."""
    lines = []
    for s in skills.values():
        parts = []
        if s.calls:
            parts.append("calls " + ", ".join(s.calls))
        near = nearest(skills, s.signature_vec, k=2, exclude={s.name}) if s.signature_vec else []
        if near:
            parts.append("resembles " + ", ".join(f"{n} ({d:.2f})" for n, d in near))
        if s.derived_from:
            parts.append(f"model says: from {s.derived_from}")
        if s.similar_to:
            parts.append("model says: like " + ", ".join(s.similar_to))
        if s.stale:
            parts.append("STALE (a skill it calls changed; rehearse again)")
        lines.append(f"{s.name}  " + ("  |  ".join(parts) if parts else "no edges yet"))
    return lines


__all__ = ["motion_signature", "distance", "nearest", "graph_lines", "N_PTS"]

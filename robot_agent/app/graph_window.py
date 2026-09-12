"""A small window showing the skill graph. Purely cosmetic.

Runs as its own process next to the app, watches the skills directory, and
redraws when a skill is added, revised or forgotten. Nothing in the app imports
it, and it never touches the simulator.

    python -m robot_agent.app.graph_window                 # window, updates live
    python -m robot_agent.app.graph_window --once --out g.png

Node colour is the skill's last measured safety level. Edges, most to least
trusted: solid = calls (recorded), dashed = resembles (measured motion
signature), dotted = model says (its own opinion when it wrote the skill).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from ..skills.graph import nearest
from ..skills.registry import DEFAULT_DIR, Registry

LEVEL_COLOUR = {"safe": "#2e7d32", "caution": "#b45309", "irreversible": "#b71c1c"}
FILL = {"safe": "#e8f5e9", "caution": "#fdebd0", "irreversible": "#fde0dc"}
POLL_S = 1.0


def _stamp(directory: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.json"))) if directory.exists() else ()


def _edges(reg: Registry) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for s in reg.skills.values():
        for c in s.calls:
            if c in reg.skills:
                out.append((s.name, c, "calls"))
        if s.signature_vec:
            for other, _d in nearest(reg.skills, s.signature_vec, k=1, exclude=(s.name,)):
                if (other, s.name, "resembles") not in out:
                    out.append((s.name, other, "resembles"))
        for other in ([s.derived_from] if s.derived_from else []) + list(s.similar_to):
            if other in reg.skills and other != s.name:
                out.append((s.name, other, "model says"))
    return out


def draw(ax, reg: Registry) -> None:
    ax.clear()
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
    names = list(reg.skills)
    ax.set_title(f"skills  ({len(names)})", fontsize=13, fontweight="bold", loc="left", color="#111")
    if not names:
        ax.text(0, 0, "no learned skills yet", ha="center", va="center", fontsize=12, color="#777")
        return
    pos = {}
    for i, n in enumerate(names):
        a = math.pi / 2 - 2 * math.pi * i / len(names)
        r = 0.0 if len(names) == 1 else 0.85
        pos[n] = (r * math.cos(a), r * math.sin(a))
    style = {"calls": ("-", 1.8), "resembles": ("--", 1.2), "model says": (":", 1.2)}
    for a, b, kind in _edges(reg):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ls, lw = style[kind]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", linestyle=ls, linewidth=lw, color="#666",
                                    shrinkA=22, shrinkB=22, mutation_scale=12))
    for n, (x, y) in pos.items():
        s = reg.skills[n]
        lvl = s.last_level
        ax.scatter([x], [y], s=1500, c=FILL.get(lvl, "#eee"), edgecolors=LEVEL_COLOUR.get(lvl, "#666"),
                   linewidths=2.0 if not s.stale else 1.0, zorder=3,
                   linestyle="-" if not s.stale else "--")
        ax.text(x, y, n.replace("_", "\n"), ha="center", va="center", fontsize=8.5, zorder=4, color="#111")
        ax.text(x, y - 0.24, f"{s.uses}×", ha="center", va="top", fontsize=7, color="#777")
    ax.text(-1.3, 1.28, "— calls   -- resembles   ·· model says", fontsize=7.5, color="#777", va="top")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skills-dir", default=str(DEFAULT_DIR))
    p.add_argument("--once", action="store_true", help="draw once and exit")
    p.add_argument("--out", default=None, help="with --once: save to this file instead of showing")
    args = p.parse_args()

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg = Registry(args.skills_dir)
    reg.load()
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    fig.patch.set_facecolor("white")
    try:
        fig.canvas.manager.set_window_title("skill graph")
    except Exception:  # noqa: BLE001
        pass
    draw(ax, reg)

    if args.once:
        if args.out:
            fig.savefig(args.out, facecolor="white")
            print(f"wrote {args.out}")
        else:
            plt.show()
        return

    plt.ion()
    plt.show()
    seen = _stamp(reg.dir)
    while plt.fignum_exists(fig.number):
        plt.pause(POLL_S)
        now = _stamp(reg.dir)
        if now != seen:
            seen = now
            reg.load()
            draw(ax, reg)
            fig.canvas.draw_idle()
    print("skill graph window closed")


if __name__ == "__main__":
    main()

"""Draw the architecture of the harness to docs/ARCHITECTURE.png and .pdf.

Hand-authored, not introspected: this is the picture we agreed on, and it is
edited here as the design changes.  Run:  python -m robot_agent.architecture
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ARCHITECTURE"

# Two colours and grey. Blue is the harness. Orange is where the world judges.
BLUE = ("#1f4e79", "#dce9f5")
ORANGE = ("#b45309", "#fdebd0")
GREY = ("#555555", "#ececec")
INK = "#111111"


def box(ax, x, y, w, h, text, colour=BLUE, fs=14, bold=False, lw=2.0):
    from matplotlib.patches import FancyBboxPatch

    edge, fill = colour
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, colour="#333333", lw=1.8, label=None, fs=11, dx=0.008):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=colour, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=fs, color=colour, ha="left", va="center")


def heading(ax, x, y, text, fs=18):
    ax.text(x, y, text, fontsize=fs, fontweight="bold", color=INK, va="center")


def chain(ax, items, x0, x1, y, h, fs=16, gap=0.018):
    """A horizontal row of boxes with arrows between. Returns their x-centres."""
    n = len(items)
    w = (x1 - x0 - gap * (n - 1)) / n
    centres = []
    for i, (text, colour) in enumerate(items):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, text, colour, fs=fs)
        centres.append(x + w / 2)
        if i < n - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, lw=2.2)
    return centres, w


def draw(out_base: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(17, 11), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ------------------------------------------------------------ title
    ax.text(0.03, 0.945, "The harness", fontsize=36, fontweight="bold", color=INK, va="center")
    ax.text(0.03, 0.895, "The model proposes.  The world disposes.", fontsize=20, color="#333333", va="center")

    # ------------------------------------------ diagram 1: one command
    heading(ax, 0.03, 0.83, "One command", fs=22)
    by, bh = 0.60, 0.18
    box(ax, 0.03, by, 0.12, bh, "PERSON", GREY, fs=18, bold=True)
    box(ax, 0.19, by, 0.62, bh, "", BLUE, fs=18, bold=True, lw=3)
    ax.text(0.50, by + bh - 0.028, "THE HARNESS", fontsize=18, fontweight="bold", color=INK, ha="center", va="center")
    box(ax, 0.85, by, 0.12, bh, "WORLD", GREY, fs=18, bold=True)
    arrow(ax, 0.15, by + bh / 2, 0.19, by + bh / 2, lw=2.2)
    arrow(ax, 0.81, by + bh / 2, 0.85, by + bh / 2, lw=2.2)
    steps = [("Plan", BLUE), ("Check", BLUE), ("Confirm", BLUE), ("Execute", BLUE), ("Verify", ORANGE)]
    cx, w = chain(ax, steps, 0.21, 0.79, by + 0.055, 0.062, fs=16, gap=0.016)
    # replan loop under the chain, inside the harness
    ly = by + 0.028
    ax.add_patch(FancyArrowPatch((cx[-1], by + 0.055), (cx[-1], ly), arrowstyle="-", linewidth=1.8, color="#333333"))
    ax.add_patch(FancyArrowPatch((cx[-1], ly), (cx[0], ly), arrowstyle="-", linewidth=1.8, color="#333333"))
    arrow(ax, cx[0], ly, cx[0], by + 0.055, lw=1.8)
    ax.text(0.50, ly - 0.012, "replan", fontsize=13, color="#333333", ha="center", va="center")
    # the model, below
    my, mh = 0.455, 0.07
    box(ax, 0.41, my, 0.18, mh, "MODEL", GREY, fs=18, bold=True)
    arrow(ax, 0.53, my + mh, 0.53, by, colour=BLUE[0], lw=2.4)
    ax.text(0.545, (my + mh + by) / 2, "proposes", fontsize=14, color=BLUE[0], ha="left", va="center")
    arrow(ax, 0.47, by, 0.47, my + mh, colour=BLUE[0], lw=2.4)
    ax.text(0.455, (my + mh + by) / 2, "scene, catalogue, failures", fontsize=14, color=BLUE[0], ha="right", va="center")

    # ------------------------------------------ diagram 2: gaining a skill
    heading(ax, 0.03, 0.365, "Gaining a skill", fs=22)
    items = [
        ("Can't do it yet", GREY),
        ("Model writes code", BLUE),
        ("Sandbox", BLUE),
        ("Rehearse, restore", ORANGE),
        ("World verifies", ORANGE),
        ("Person keeps it", GREY),
    ]
    sy, sh = 0.215, 0.095
    cx, w = chain(ax, items, 0.03, 0.97, sy, sh, fs=16, gap=0.02)
    # failed -> revise, back to the code
    ly = sy - 0.035
    ax.add_patch(FancyArrowPatch((cx[4], sy), (cx[4], ly), arrowstyle="-", linewidth=1.8, color=ORANGE[0]))
    ax.add_patch(FancyArrowPatch((cx[4], ly), (cx[1], ly), arrowstyle="-", linewidth=1.8, color=ORANGE[0]))
    arrow(ax, cx[1], ly, cx[1], sy, colour=ORANGE[0], lw=1.8)
    ax.text((cx[1] + cx[4]) / 2, ly - 0.014, "fails?  revise, up to 3 times", fontsize=13, color=ORANGE[0], ha="center", va="center")
    ax.text(0.03, 0.085, "Rules:  same loop.  They can only tighten.", fontsize=16, color="#333333", va="center")

    outs = []
    for ext in ("png", "pdf"):
        path = out_base.with_suffix("." + ext)
        fig.savefig(path, dpi=150 if ext == "png" else None, facecolor="white")
        outs.append(path)
    plt.close(fig)
    return outs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT), help="output basename (png and pdf are written)")
    args = p.parse_args()
    for path in draw(Path(args.out)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

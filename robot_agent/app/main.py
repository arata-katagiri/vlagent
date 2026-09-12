"""Single-threaded app loop: viewer, terminal REPL, demo mode.

There is no threading here, by design (see CLAUDE.md section 5). launch_passive
renders on its own UI thread, so blocking in input() or an LLM call does not
freeze the window -- and the arm is idle during both. Physics steps only inside
backend calls. Do not add a worker thread, a queue, or a lock on data.

Run: mjpython -m robot_agent.app.main [--mock-llm] [--backend panda|floating|mock]
                                      [--demo] [--record] [--inject-failure]
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robot_agent")
    p.add_argument("--mock-llm", action="store_true", help="offline scripted planner")
    p.add_argument(
        "--backend", choices=("panda", "floating", "mock"), default="panda"
    )
    p.add_argument("--demo", action="store_true", help="run the scripted demo script")
    p.add_argument("--record", action="store_true", help="write an mp4 via ffmpeg")
    p.add_argument(
        "--inject-failure", action="store_true", help="nudge an object after placement"
    )
    return p


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("Phase 3")


if __name__ == "__main__":
    main()

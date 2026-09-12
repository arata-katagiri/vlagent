"""Single-threaded app loop: viewer, terminal REPL, demo mode.

There is no threading here, by design (see CLAUDE.md section 5). launch_passive
renders on its own UI thread, so blocking in input() or an LLM call does not
freeze the window -- and the arm is idle during both. Physics steps only inside
backend calls. Do not add a worker thread, a queue, or a lock on data.

Run: mjpython -m robot_agent.app.main [--mock-llm] [--backend panda|floating|mock]
                                     [--demo] [--record] [--inject-failure]

--backend mock needs no simulator and no viewer, so it runs under plain python.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from ..actions.executor import classify_plan, execute
from ..actions.safety import set_policy_enabled
from ..actions.schema import ActionCall, Safety
from ..agent import planner
from ..agent.llm import build_llm
from . import ui

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

DEMO_SCRIPT = [
    "Can you clear the red and green blocks onto the tray?",
    "Put the blue block on the glass.",
    "Push the glass out of the way to the left.",
]


class RunLog:
    """One JSONL file per run: every plan, refusal, confirmation and result."""

    def __init__(self, enabled: bool = True):
        self.path = RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        self.enabled = enabled
        if enabled:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields) -> None:
        if not self.enabled:
            return
        record = {"t": time.time(), "kind": kind, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robot_agent")
    p.add_argument("--mock-llm", action="store_true", help="offline scripted planner")
    p.add_argument("--backend", choices=("panda", "mock"), default="panda")
    p.add_argument("--no-viewer", action="store_true",
                   help="run the simulator headless (for recording and tests)")
    p.add_argument("--demo", action="store_true", help="run the scripted demo script")
    p.add_argument("--record", action="store_true", help="write an mp4 via ffmpeg")
    p.add_argument("--inject-failure", action="store_true",
                   help="nudge an object after placement, to exercise verification")
    p.add_argument("--no-log", action="store_true", help="do not write runs/*.jsonl")
    p.add_argument(
        "--no-safety", action="store_true",
        help="disable the policy checks and the confirmation gate: the agent "
             "executes whatever it plans, and physics decides the outcome",
    )
    return p


def build_world(args, stack):
    """Return (get_state, backend). `mock` needs no simulator and no viewer."""
    drift = 0.08 if args.inject_failure else 0.0

    if args.backend == "mock":
        from ..sim.backends import MockBackend, initial_state  # noqa: PLC0415

        state = initial_state()
        return (lambda: state), MockBackend(state, drift_m=drift)

    from ..sim.backends import PandaIKBackend  # noqa: PLC0415
    from ..sim.render import RUNS_DIR, recorder  # noqa: PLC0415
    from ..sim.scene import CAMERA_NAME, build_scene, settle  # noqa: PLC0415
    from ..sim.world_state import get_world_state  # noqa: PLC0415

    model, data = build_scene()
    settle(model, data, 0.3)

    hooks = []
    if not args.no_viewer:
        import mujoco.viewer  # noqa: PLC0415

        viewer = stack.enter_context(
            mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            )
        )
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        hooks.append(viewer.sync)

    if args.record:
        path = RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.mp4"
        rec = stack.enter_context(recorder(model, path, camera=CAMERA_NAME))
        hooks.append(lambda: rec.capture(data))
        ui.note(f"recording to {path}")

    def sync():
        for hook in hooks:
            hook()

    backend = PandaIKBackend(model, data, sync=sync, drift_m=drift)
    return (lambda: get_world_state(model, data)), backend


def run_command(command: str, get_state, backend, llm, log: RunLog) -> None:
    """One full turn: observe, plan, validate, show, confirm, execute, verify, report."""
    state = get_state()
    log.write("command", text=command)

    outcome = planner.plan(command, state, llm)
    log.write("plan", attempts=outcome.attempts, ok=outcome.ok,
              steps=[(s.name, s.args) for s in outcome.steps],
              summary=outcome.summary, problems=outcome.problems,
              answer=outcome.answer, question=outcome.question)

    for steps, problems in outcome.rejected:
        ui.rejected(steps, problems)
        log.write("plan_rejected", steps=[s.name for s in steps], problems=problems)

    if outcome.answer:
        ui.answer(outcome.answer)
        log.write("answer", text=outcome.answer)
        return

    if outcome.question:
        ui.question(outcome.question)
        return

    if not outcome.ok:
        ui.report("I could not find a safe plan for that.\n" + "\n".join(outcome.problems), ok=False)
        return

    levels = classify_plan(outcome.steps, state)
    ui.plan(outcome.steps, levels, outcome.summary)

    executed, refused = [], False
    for i, (step, (level, reason)) in enumerate(zip(outcome.steps, levels), start=1):
        # ask_user can arrive as a plan step rather than as the ask_user tool.
        # It is a question for the human, not an action to execute and tick off.
        if step.name == "ask_user":
            ui.question(str(step.args.get("question", "Could you clarify?")))
            log.write("question", n=i, text=step.args.get("question"))
            return

        if level is Safety.IRREVERSIBLE:
            # Bound to this one step: a yes here is never reused for another.
            if not ui.confirm(i, step, reason):
                log.write("refused", step=step.name, args=step.args, reason=reason)
                ui.note(f"  {i}. {step.name} skipped; asking for a safer alternative")
                refused = True
                break
            log.write("confirmed", step=step.name, args=step.args, reason=reason)

        result = execute(step, backend, get_state)
        executed.append((step, result))
        ui.step_result(i, step, result.ok, result.reason)
        log.write("step", n=i, action=step.name, args=step.args,
                  safety=str(level), ok=result.ok, reason=result.reason)

        if not result.ok:
            _recover(command, step, result, get_state, backend, llm, log)
            return

    if refused:
        _offer_alternative(command, get_state, backend, llm, log, already=executed)
        return

    ui.report(_summarize(command, executed))


def _recover(command, failed_step, result, get_state, backend, llm, log) -> None:
    """Re-observe and replan the remaining goal after a verification failure."""
    ui.note(f"  verification failed: {result.reason}")
    ui.note("  re-observing the scene and replanning")
    log.write("replan_triggered", after=failed_step.name, reason=result.reason)

    for attempt in range(1, planner.MAX_REPLAN_ATTEMPTS + 1):
        state = get_state()
        outcome = planner.replan_after_failure(command, state, llm, result.reason)
        if not outcome.ok:
            continue
        levels = classify_plan(outcome.steps, state)
        ui.plan(outcome.steps, levels, outcome.summary or f"recovery attempt {attempt}")
        done = []
        for i, (step, (level, reason)) in enumerate(zip(outcome.steps, levels), start=1):
            if level is Safety.IRREVERSIBLE and not ui.confirm(i, step, reason):
                log.write("refused", step=step.name, args=step.args, reason=reason)
                break
            r = execute(step, backend, get_state)
            done.append((step, r))
            ui.step_result(i, step, r.ok, r.reason)
            log.write("step", n=i, action=step.name, args=step.args,
                      safety=str(level), ok=r.ok, reason=r.reason, recovery=attempt)
            if not r.ok:
                break
        else:
            ui.report(_summarize(command, done))
            return

    ui.note("  giving up on this request and returning the arm home")
    log.write("gave_up", command=command)
    backend.home()
    ui.report("I could not complete that safely, so I stopped and returned home.", ok=False)


def _offer_alternative(command, get_state, backend, llm, log, already=None) -> None:
    """After a refusal, ask the planner for a safer way to achieve the same goal."""
    state = get_state()
    outcome = planner.replan_after_failure(
        command, state, llm,
        "the user refused the irreversible step; propose a reversible alternative",
    )
    if not outcome.ok:
        ui.report("Understood, I will leave it as it is.")
        return
    levels = classify_plan(outcome.steps, state)
    ui.plan(outcome.steps, levels, outcome.summary or "a safer alternative")
    done = list(already or [])
    for i, (step, (level, reason)) in enumerate(zip(outcome.steps, levels), start=1):
        if level is Safety.IRREVERSIBLE and not ui.confirm(i, step, reason):
            ui.report("Understood, I will leave it as it is.")
            return
        r = execute(step, backend, get_state)
        done.append((step, r))
        ui.step_result(i, step, r.ok, r.reason)
        log.write("step", n=i, action=step.name, args=step.args,
                  safety=str(level), ok=r.ok, reason=r.reason, alternative=True)
        if not r.ok:
            ui.report("The alternative did not work either.", ok=False)
            return
    ui.report(_summarize(command, done))


def _summarize(command: str, executed: list[tuple[ActionCall, object]]) -> str:
    if not executed:
        return "Nothing was done."
    names = ", ".join(s.name for s, _ in executed)
    ok = all(r.ok for _, r in executed)  # type: ignore[attr-defined]
    verb = "Done" if ok else "Partly done"
    return (
        f"{verb}. For \"{command}\" I ran {len(executed)} step(s): {names}. "
        + ("Every step was verified against the scene." if ok
           else "At least one step did not verify.")
    )


def main() -> None:
    args = build_parser().parse_args()

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass

    if args.no_safety:
        set_policy_enabled(False)

    llm = build_llm(args.mock_llm)
    model = "mock" if args.mock_llm else os.environ.get("OPENAI_MODEL", "unset")
    log = RunLog(enabled=not args.no_log)
    log.write("session", backend=args.backend, model=model, safety=not args.no_safety)

    with ExitStack() as stack:
        get_state, backend = build_world(args, stack)

        ui.banner(args.backend, model, safety=not args.no_safety)
        ui.scene(get_state())

        if args.demo:
            for command in DEMO_SCRIPT:
                ui.console.rule(f"[bold]{command}")
                run_command(command, get_state, backend, llm, log)
                ui.scene(get_state())
            ui.note(f"demo complete; log written to {log.path}")
            return

        while True:
            try:
                command = ui.console.input("\n[bold cyan]you >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                continue
            if command.lower() in ("quit", "exit"):
                break
            run_command(command, get_state, backend, llm, log)

    ui.note(f"log written to {log.path}")


if __name__ == "__main__":
    main()

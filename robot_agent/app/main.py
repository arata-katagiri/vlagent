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
from ..agent.llm import SkillProposal, build_llm
from ..skills.registry import REGISTRY, SKILL_ARG_KEYS, Skill
from ..skills.rehearse import rehearse

MAX_SKILL_REVISIONS = 3
from . import ui

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

DEMO_SCRIPT = [
    "Can you clear the red and green blocks onto the tray?",
    "Put the blue block on the glass.",
    "Push the glass out of the way to the left.",
]


class RunLog:
    """One JSONL file per run: every plan, refusal, confirmation and result.

    The same events, rendered as short lines, are the planner's conversation
    memory (`transcript`), so "now the green one" and "why did that fail?" can
    be answered from what actually happened. Kept in memory even with --no-log.
    """

    def __init__(self, enabled: bool = True):
        self.path = RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        self.enabled = enabled
        self.transcript: list[dict] = []
        if enabled:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields) -> None:
        turn = _as_turn(kind, fields)
        if turn is not None:
            self.transcript.append(turn)
        if not self.enabled:
            return
        record = {"t": time.time(), "kind": kind, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _fmt_steps(steps) -> str:
    out = []
    for item in steps:
        name, args = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, {})
        argtxt = ", ".join(f"{k}={v}" for k, v in dict(args or {}).items())
        out.append(f"{name}({argtxt})")
    return ", ".join(out)


def _as_turn(kind: str, f: dict) -> dict | None:
    """Render one log record as a conversation turn for the planner, or None."""
    if kind == "command":
        return {"role": "user", "content": str(f.get("text", ""))}
    if kind == "plan":
        if f.get("answer") or f.get("question"):
            return None  # recorded by their own events below
        if f.get("ok"):
            return {"role": "assistant",
                    "content": f"Proposed plan: {f.get('summary') or ''} [{_fmt_steps(f.get('steps', []))}]"}
        if f.get("problems"):
            return {"role": "assistant",
                    "content": "Could not find a safe plan: " + "; ".join(map(str, f["problems"]))}
        return None
    if kind == "plan_rejected":
        return {"role": "assistant",
                "content": "A proposed plan was rejected by the safety checker: "
                           + "; ".join(map(str, f.get("problems", [])))}
    if kind == "answer":
        return {"role": "assistant", "content": f"Answered: {f.get('text', '')}"}
    if kind == "question":
        return {"role": "assistant", "content": f"Asked the user: {f.get('text', '')}"}
    if kind == "refused":
        return {"role": "assistant",
                "content": f"The user refused irreversible step {f.get('step')} ({f.get('reason', '')})"}
    if kind == "confirmed":
        return {"role": "assistant", "content": f"The user confirmed irreversible step {f.get('step')}"}
    if kind == "step":
        status = "OK" if f.get("ok") else "FAILED"
        return {"role": "assistant",
                "content": f"Executed {_fmt_steps([(f.get('action'), f.get('args'))])}: {status}, {f.get('reason', '')}"}
    if kind == "replan_triggered":
        return {"role": "assistant",
                "content": f"Verification failed after {f.get('after')}: {f.get('reason', '')}. Re-observed the scene and replanned."}
    if kind == "gave_up":
        return {"role": "assistant",
                "content": "Gave up on that request after repeated failures and returned the arm home."}
    # -- learned skills: the model must remember what it wrote and why it failed
    if kind == "skill_proposed":
        return {"role": "assistant",
                "content": f"Wrote skill {f.get('name')}({', '.join(f.get('args') or [])}) attempt {f.get('attempt')}, "
                           f"declared effect: {f.get('effect', '')}"}
    if kind == "rehearsal":
        m = f.get("metrics") or {}
        facts = []
        if m.get("moved"):
            facts.append("moved " + ", ".join(f"{k} {v:.2f} m" for k, v in m["moved"].items()))
        if m.get("rotated"):
            facts.append("rotated " + ", ".join(f"{k} {v:+.0f} deg" for k, v in m["rotated"].items()))
        if m.get("touched"):
            facts.append("touched " + ", ".join(m["touched"]))
        if m.get("off_table"):
            facts.append("left the table: " + ", ".join(m["off_table"]))
        if m.get("edge_m"):
            facts.append("near edge: " + ", ".join(f"{k} {v:.3f} m inside" for k, v in m["edge_m"].items()))
        status = "PASSED" if f.get("ok") else "FAILED"
        who = "world-verified" if f.get("verified") else "self-reported"
        return {"role": "assistant",
                "content": f"Rehearsed {f.get('name')} attempt {f.get('attempt')}: {status} ({who}), "
                           f"{f.get('reason', '')}. " + ("; ".join(facts) if facts else "")}
    if kind == "skill_kept":
        return {"role": "assistant", "content": f"The user kept skill {f.get('name')} ({f.get('level')}); it is now in the catalogue."}
    if kind == "skill_discarded":
        return {"role": "assistant", "content": f"The user discarded skill {f.get('name')} after rehearsal."}
    if kind == "skill_stopped":
        return {"role": "assistant", "content": f"The user stopped work on skill {f.get('name')} at attempt {f.get('attempt')}."}
    if kind == "skill_abandoned":
        return {"role": "assistant", "content": f"Gave up on skill {f.get('name')} after attempt {f.get('attempt')}; nothing was changed."}
    if kind == "skill_stale":
        return {"role": "assistant", "content": f"Revising {f.get('name')} made these dependent skills stale: {', '.join(f.get('dependents') or [])}."}
    if kind == "interrupted":
        return {"role": "assistant", "content": "The user interrupted the command with Ctrl-C."}
    return None


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
    p.add_argument("--voice", action="store_true",
                   help="speak commands instead of typing (local Whisper + say); "
                        "falls back to typing if the microphone is unavailable")
    p.add_argument("--scene", choices=("default", "lab"), default="default",
                   help="which scenario to load: the kitchen table (default) or the lab bench")
    p.add_argument("--skills-dir", default=None,
                   help="where learned skills are kept (default: ./skills)")
    p.add_argument(
        "--safety", action="store_true",
        help="enable the policy checks and the confirmation gate (fragile, "
             "stability, irreversible). Off by default: the agent executes "
             "whatever it plans, and physics decides the outcome",
    )
    p.add_argument("--no-safety", action="store_true", help=argparse.SUPPRESS)  # legacy, now the default
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

    if outcome.skill is not None:
        learned = _learn_skill(outcome.skill, command, get_state, backend, llm, log)
        if learned is None:
            return
        state = get_state()
        outcome = planner.plan(
            command, state, llm,
            feedback=[f"the skill '{learned.name}' is now available and rehearsed; use it"],
        )
        log.write("plan", attempts=outcome.attempts, ok=outcome.ok,
                  steps=[(s.name, s.args) for s in outcome.steps],
                  summary=outcome.summary, problems=outcome.problems,
                  answer=outcome.answer, question=outcome.question, after_skill=learned.name)

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

    ui.report(_summarize(command, executed), spoken=_spoken(executed))


def _learn_skill(proposal: SkillProposal, command: str, get_state, backend, llm, log: RunLog):
    """The vibe loop: show the code, rehearse it in a snapshot, measure, revise
    on failure, ask the user to keep it, save it. Returns the Skill or None."""
    for attempt in range(1, MAX_SKILL_REVISIONS + 1):
        skill = Skill(name=proposal.name, doc=proposal.doc, effect=proposal.effect,
                      args=proposal.args, code=proposal.code,
                      effect_kind=proposal.effect_kind, effect_of=proposal.effect_of,
                      effect_value=proposal.effect_value)
        ui.code(skill, attempt)
        log.write("skill_proposed", name=skill.name, attempt=attempt, args=skill.args,
                  effect=skill.effect, code=skill.code, example_args=proposal.example_args)

        bad = [a for a in skill.args if a not in SKILL_ARG_KEYS]
        if bad or not skill.name.isidentifier():
            verdict_reason = (f"argument names must come from {', '.join(SKILL_ARG_KEYS)}; got {bad}"
                              if bad else f"'{skill.name}' is not a valid skill name")
            feedback = [f"rehearsal skipped: {verdict_reason}"]
        else:
            ui.note(f"  rehearsing {skill.name}{proposal.example_args} in a snapshot of the world")
            verdict = rehearse(skill, proposal.example_args, backend, get_state, registry=REGISTRY)
            ui.rehearsal(skill, verdict)
            log.write("rehearsal", name=skill.name, attempt=attempt, **verdict.as_record())
            if verdict.error == "stopped":
                log.write("skill_stopped", name=skill.name, attempt=attempt)
                ui.note(f"  stopped; {skill.name} was not kept and the world is as it was")
                return None
            if verdict.ok:
                skill.rehearsals.append(verdict.as_record())
                if not ui.confirm_keep(skill):
                    log.write("skill_discarded", name=skill.name)
                    ui.note(f"  {skill.name} discarded")
                    return None
                skill.calls = list(verdict.metrics.get("calls", []))
                skill.signature_vec = list(verdict.metrics.get("signature", []))
                skill.derived_from = proposal.derived_from
                skill.similar_to = [s for s in proposal.similar_to if s in REGISTRY and s != skill.name]
                revised = skill.name in REGISTRY
                REGISTRY.register(skill)
                log.write("skill_kept", name=skill.name, level=verdict.level, calls=skill.calls,
                          derived_from=skill.derived_from, similar_to=skill.similar_to)
                ui.note(f"  {skill.name} saved to {REGISTRY.dir / (skill.name + '.json')}")
                if revised:
                    stale = REGISTRY.mark_stale_dependents(skill.name)
                    if stale:
                        log.write("skill_stale", name=skill.name, dependents=stale)
                        ui.note(f"  {', '.join(stale)} built on {skill.name} and must be rehearsed again")
                return skill
            shown = {k: v for k, v in verdict.metrics.items() if k not in ("signature",)}
            feedback = [f"rehearsal of {skill.name} failed: {verdict.reason}",
                        f"rehearsal metrics: {json.dumps(shown)}"]
            # Retrieval: the learned skills whose measured motion was closest, plus
            # the one the model said it built on. Their code is the best starting point.
            near = REGISTRY.nearest(verdict.metrics.get("signature", []), k=2, exclude={skill.name})
            names = [n for n, _ in near]
            if proposal.derived_from in REGISTRY and proposal.derived_from not in names:
                names.insert(0, proposal.derived_from)
            for n in names[:3]:
                other = REGISTRY.get(n)
                d = dict(near).get(n)
                how = f"measured motion distance {d:.2f}" if d is not None else "you said it was derived from this"
                feedback.append(f"rehearsal hint: existing skill {other.signature()} ({how}) passed before; its code:\n{other.code}")

        if attempt == MAX_SKILL_REVISIONS:
            break
        if not ui.confirm_revise(skill, attempt, MAX_SKILL_REVISIONS - attempt):
            log.write("skill_stopped", name=skill.name, attempt=attempt)
            ui.note(f"  stopped; {skill.name} was not kept")
            return None
        ui.note("  asking the planner to revise the skill")
        try:
            proposal2 = llm.propose(command, get_state(), feedback)
        except KeyboardInterrupt:
            ui.note(f"  stopped while the planner was revising; {skill.name} was not kept")
            return None
        if not isinstance(proposal2, SkillProposal):
            text = getattr(proposal2, "text", None) or getattr(proposal2, "summary", "")
            ui.report(f"I could not get {skill.name} to pass rehearsal. {text}".strip(), ok=False)
            log.write("skill_abandoned", name=skill.name, attempt=attempt)
            return None
        proposal = proposal2

    ui.report(f"{proposal.name} did not pass rehearsal after {MAX_SKILL_REVISIONS} attempts; nothing was changed.", ok=False)
    log.write("skill_abandoned", name=proposal.name, attempt=MAX_SKILL_REVISIONS)
    return None


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
            ui.report(_summarize(command, done), spoken=_spoken(done))
            return

    ui.note("  giving up on this request and returning the arm home")
    log.write("gave_up", command=command)
    backend.home()
    ui.report(
        f"I could not complete that after {planner.MAX_REPLAN_ATTEMPTS} recovery attempts; "
        "the arm is back home. Tell me a different way to do it, or where things should go.",
        ok=False,
    )


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
    ui.report(_summarize(command, done), spoken=_spoken(done))


def _spoken(executed: list[tuple[ActionCall, object]]) -> str:
    """Short form of the result for talk-back: no quoted command, no underscores."""
    if not executed:
        return "Nothing was done."
    n = len(executed)
    failed = sum(1 for _, r in executed if not r.ok)  # type: ignore[attr-defined]
    steps = "one step" if n == 1 else f"{n} steps"
    if failed == 0:
        return f"Done. {steps.capitalize()}, all verified."
    return f"Partly done. {failed} of {steps} did not verify."


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

    set_policy_enabled(bool(args.safety))

    if args.scene == "lab":
        from ..sim.lab_scene import activate as activate_lab  # noqa: PLC0415

        activate_lab()
        ui.note("scene: lab bench (acid_bottle, water_flask, three samples, containment tray)")

    if args.skills_dir:
        REGISTRY.dir = Path(args.skills_dir)
    learned_count = REGISTRY.load()

    llm = build_llm(args.mock_llm)

    if args.voice:
        from .voice import VoiceSession  # noqa: PLC0415  # optional deps, only under --voice

        ui.voice = VoiceSession()
    model = "mock" if args.mock_llm else os.environ.get("OPENAI_MODEL", "unset")
    log = RunLog(enabled=not args.no_log)
    llm.transcript = log.transcript  # conversation memory, shared by reference
    log.write("session", backend=args.backend, model=model, safety=bool(args.safety))

    with ExitStack() as stack:
        get_state, backend = build_world(args, stack)

        ui.banner(args.backend, model, safety=bool(args.safety))
        ui.note(f"{learned_count} learned skill(s) in {REGISTRY.dir}; type 'skills' to list them")
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
                if ui.voice is not None:
                    command = ui.voice.ask().strip()
                    ui.console.print(f"\n[bold cyan]you >[/bold cyan] {command}")
                else:
                    command = ui.console.input("\n[bold cyan]you >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                continue
            if command.lower() in ("quit", "exit"):
                break
            if command.lower() == "skills":
                ui.skills(REGISTRY)
                continue
            if command.lower() == "graph":
                ui.graph(REGISTRY)
                continue
            if command.lower().startswith("forget "):
                name = command.split(None, 1)[1].strip()
                ui.note(f"forgot {name}" if REGISTRY.forget(name) else f"no skill called {name}")
                continue
            try:
                run_command(command, get_state, backend, llm, log)
            except KeyboardInterrupt:
                ui.note("stopped by Ctrl-C; the arm stays where it is (type 'home' to reset it)")
                log.write("interrupted", command=command)

    ui.note(f"log written to {log.path}")


if __name__ == "__main__":
    main()

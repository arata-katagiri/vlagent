"""The browser UI: --web [PORT].

Four boxes. The simulation streamed as MJPEG from the same render hook the
recorder uses; the library (skills and rules); the chat, where the agent talks
to the person and every confirmation is a button; and the output, which is the
terminal's own rendering captured as HTML, so plans, code and rehearsal verdicts
look exactly as they do in the terminal.

Threads, deliberately and narrowly: uvicorn runs on a daemon thread and only
ever reads the latest JPEG bytes and moves JSON between queues. The agent loop,
the physics and every prompt stay on the main thread, exactly as in the REPL.
"""

from __future__ import annotations

import _thread
import asyncio
import collections
import io
import itertools
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

try:  # module-level so FastAPI can resolve the (postponed) annotations on the handlers
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, StreamingResponse
except ImportError:  # the sink and the grabber work without the server installed
    FastAPI = WebSocket = WebSocketDisconnect = HTMLResponse = StreamingResponse = None  # type: ignore

# ---------------------------------------------------------------- frames
GRABBER = None


class FrameGrabber:
    """Renders the named camera to JPEG, throttled, from the physics thread."""

    def __init__(self, model, data, camera: str, width: int = 960, height: int = 540, fps: float = 20.0):
        import mujoco  # noqa: PLC0415

        self.model, self.data = model, data
        w = min(width, int(model.vis.global_.offwidth) or width)
        h = min(height, int(model.vis.global_.offheight) or height)
        self.renderer = mujoco.Renderer(model, h, w)
        self.cameras = [model.camera(i).name for i in range(model.ncam)] or []
        self.camera = camera if camera in self.cameras else (self.cameras[0] if self.cameras else None)
        self.min_dt = 1.0 / fps
        self._last = 0.0
        self._lock = threading.Lock()
        self.latest: bytes = b""
        self.want_refresh = False

    def capture(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and not self.want_refresh and now - self._last < self.min_dt:
            return
        self._last = now
        self.want_refresh = False
        from PIL import Image  # noqa: PLC0415

        if self.camera is not None:
            self.renderer.update_scene(self.data, camera=self.camera)
        else:
            self.renderer.update_scene(self.data)
        rgb = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
        with self._lock:
            self.latest = buf.getvalue()

    def frame(self) -> bytes:
        with self._lock:
            return self.latest


def install_grabber(model, data, camera: str):
    """Called from main's world builders when --web is on. Returns the sync hook."""
    global GRABBER
    GRABBER = FrameGrabber(model, data, camera)
    GRABBER.capture(force=True)
    return GRABBER.capture


# ------------------------------------------------------------------ sink
CHAT_FUNCS = ("question", "answer", "report")
RENDER_FUNCS = ("banner", "scene", "plan", "rejected", "step_result", "question", "answer", "report",
                "note", "code", "rehearsal", "rule_code", "rule_dry_run", "rules", "graph", "skills")
PROMPT_FUNCS = ("confirm", "confirm_keep", "confirm_revise", "confirm_keep_rule")

SINK = None


class WebSink:
    """Routes the terminal UI into the page: rendering to box 4, talk to box 3,
    prompts to buttons. Installs itself over the functions in app/ui.py."""

    def __init__(self):
        from rich.console import Console  # noqa: PLC0415

        self.console = Console(file=io.StringIO(), record=True, width=112, force_terminal=True,
                               color_system="truecolor", legacy_windows=False)
        self.answers: queue.Queue = queue.Queue()
        self.history: collections.deque = collections.deque(maxlen=300)
        # One queue per connected page, so every client sees every event
        # (a shared queue would split them between tabs).
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self.speak = False
        self._ids = itertools.count(1)
        self._orig: dict = {}

    # -- transport -------------------------------------------------------
    def push(self, event: dict) -> None:
        self.history.append(event)
        with self._sub_lock:
            for q in self._subscribers:
                q.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def flush(self) -> None:
        html = self.console.export_html(
            inline_styles=True, clear=True, code_format='<pre class="rich">{code}</pre>')
        if html.strip() and html.strip() != '<pre class="rich"></pre>':
            self.push({"type": "output", "html": html})

    def chat(self, role: str, text: str, ok: bool | None = None) -> None:
        self.push({"type": "chat", "role": role, "text": text, "ok": ok})
        if role == "agent" and self.speak and sys.platform == "darwin":
            try:
                subprocess.Popen(["say", "-v", "Samantha", text[:400]])
            except OSError:
                pass

    def ask(self, text: str, options: list[str], default: str) -> str:
        pid = next(self._ids)
        self.push({"type": "prompt", "id": pid, "text": text, "options": options, "default": default})
        while True:
            got_id, value = self.answers.get()
            if got_id == pid:
                self.push({"type": "prompt_done", "id": pid, "value": value})
                return str(value)

    # -- installation over ui.py -----------------------------------------
    def install(self) -> None:
        global SINK
        from . import ui  # noqa: PLC0415

        SINK = self
        self._orig = {n: getattr(ui, n) for n in set(RENDER_FUNCS) | set(PROMPT_FUNCS)}
        ui.console = self.console
        for name in RENDER_FUNCS:
            setattr(ui, name, self._wrap(name, self._orig[name]))
        ui.confirm = self._confirm
        ui.confirm_keep = self._confirm_keep
        ui.confirm_revise = self._confirm_revise
        ui.confirm_keep_rule = self._confirm_keep_rule

    def _wrap(self, name: str, orig):
        def wrapped(*a, **k):
            result = orig(*a, **k)
            self.flush()
            if name == "plan" and len(a) >= 3 and a[2]:
                self.chat("agent", f"Plan: {a[2]}")
            elif name == "question" and a:
                self.chat("agent", str(a[0]))
            elif name == "answer" and a:
                self.chat("agent", str(a[0]))
            elif name == "report" and a:
                self.chat("agent", str(a[0]), ok=k.get("ok", a[1] if len(a) > 1 else True))
            return result
        wrapped.__name__ = name
        return wrapped

    def _confirm(self, step_number: int, step, reason: str) -> bool:
        self._orig["confirm"].__globals__  # noqa: B018  (keep linter honest about the swap)
        args = ", ".join(f"{k}={v}" for k, v in step.args.items())
        text = f"Step {step_number} is irreversible: {step.name}({args}). {reason} Allow this one step?"
        self.chat("agent", text)
        return self.ask(text, ["yes", "no"], "no") == "yes"

    def _confirm_keep(self, skill) -> bool:
        text = f"Rehearsal passed. Keep {skill.name} as a skill and run it for real?"
        return self.ask(text, ["yes", "no"], "yes") == "yes"

    def _confirm_revise(self, skill, attempt: int, remaining: int) -> bool:
        text = f"Rehearsal failed. Let the planner revise {skill.name} and rehearse again? ({remaining} attempt(s) left)"
        return self.ask(text, ["revise", "stop"], "revise") == "revise"

    def _confirm_keep_rule(self, rule) -> bool:
        text = f"Keep {rule.name} as a standing safety rule?"
        return self.ask(text, ["yes", "no"], "yes") == "yes"


def library_event() -> dict:
    from ..skills.registry import REGISTRY  # noqa: PLC0415
    from ..skills.rules import RULES  # noqa: PLC0415

    return {
        "type": "library",
        "skills": [
            {"name": s.name, "signature": s.signature(), "effect": s.effect, "measured": s.effect_label(),
             "level": s.last_level, "verified": s.world_verified, "stale": s.stale, "calls": s.calls,
             "uses": s.uses, "args": s.args, "code": s.code}
            for s in REGISTRY.skills.values()
        ],
        "rules": [{"name": r.name, "doc": r.doc, "blocks": r.blocks, "code": r.code, "dry_run": r.dry_run}
                  for r in RULES.rules.values()],
        "tags": dict(RULES.tags),
    }


# ---------------------------------------------------------------- server
def make_app(info: dict):
    if FastAPI is None:
        raise RuntimeError("the web UI needs fastapi and uvicorn: pip install fastapi 'uvicorn[standard]'")
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse((STATIC / "index.html").read_text(),
                            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})

    @app.get("/stream")
    async def stream():
        async def gen():
            last = None
            while True:
                frame = GRABBER.frame() if GRABBER else b""
                if frame and frame is not last:
                    last = frame
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                await asyncio.sleep(0.05)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        hello = dict(info)
        hello.update({"type": "hello", "cameras": GRABBER.cameras if GRABBER else [],
                      "camera": GRABBER.camera if GRABBER else None, "speak": SINK.speak,
                      "scenes": list(info.get("scenes", []))})
        await sock.send_text(json.dumps(hello))
        inbox = SINK.subscribe()          # subscribe first, then replay: nothing is lost in between
        for ev in list(SINK.history):
            await sock.send_text(json.dumps(ev))
        await sock.send_text(json.dumps(library_event()))
        loop = asyncio.get_event_loop()

        async def writer():
            while True:
                ev = await loop.run_in_executor(None, _poll, inbox)
                if ev is not None:
                    await sock.send_text(json.dumps(ev))

        task = asyncio.create_task(writer())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                kind = msg.get("type")
                if kind == "command":
                    INBOX.put(str(msg.get("text", "")))
                elif kind == "scene":
                    RESTART_WITH["scene"] = str(msg.get("name", "default"))
                    INBOX.put("__restart__")
                elif kind == "answer":
                    SINK.answers.put((int(msg.get("id", 0)), str(msg.get("value", ""))))
                elif kind == "camera" and GRABBER and msg.get("name") in GRABBER.cameras:
                    GRABBER.camera = msg["name"]
                    GRABBER.want_refresh = True
                elif kind == "speak":
                    SINK.speak = bool(msg.get("on"))
                elif kind == "stop":
                    _thread.interrupt_main()
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            SINK.unsubscribe(inbox)

    return app


INBOX: queue.Queue = queue.Queue()
RESTART_WITH: dict = {}          # set by the page's scene picker; serve() re-execs the process


def _relaunch_argv(scene: str) -> list[str]:
    """The current command line with --scene replaced (and --fen dropped)."""
    argv = [sys.executable, "-m", "robot_agent.app.main"]
    skip = 0
    for a in sys.argv[1:]:
        if skip:
            skip -= 1
            continue
        if a in ("--scene", "--fen"):
            skip = 1
            continue
        if a.startswith("--scene=") or a.startswith("--fen="):
            continue
        argv.append(a)
    return argv + ["--scene", scene]


def _poll(q: queue.Queue, timeout: float = 0.5):
    """A queue read that returns None on timeout, so a cancelled writer never
    leaves a pool thread parked forever (which would stop the process exiting)."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def serve(args, get_state, backend, llm, log, handle_command, info: dict) -> None:
    """The web main loop. Runs on the main thread; the server on a daemon thread."""
    import uvicorn  # noqa: PLC0415

    port = int(args.web)
    app = make_app(info)

    class _Server(uvicorn.Server):
        def install_signal_handlers(self) -> None:  # not the main thread
            pass

    server = _Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="web").start()
    url = f"http://127.0.0.1:{port}"
    print(f"web UI at {url}   (Ctrl-C in this terminal quits)")
    SINK.chat("agent", "Ready. Tell me what to do with the arm, or state a rule.")
    SINK.push(library_event())

    while True:
        try:
            command = INBOX.get(timeout=0.5)
        except queue.Empty:
            if GRABBER and time.monotonic() - GRABBER._last > 2.0:
                GRABBER.capture(force=True)
            continue
        except KeyboardInterrupt:
            SINK.chat("agent", "Nothing was running.")
            continue
        command = command.strip()
        if not command:
            continue
        if command == "__restart__":
            scene = RESTART_WITH.get("scene", "default")
            SINK.push({"type": "restarting", "scene": scene})
            SINK.chat("agent", f"Switching to the {scene} scene; the page will reconnect in a few seconds.")
            log.write("scene_switch", scene=scene)
            time.sleep(0.5)                      # let the message reach the page
            server.should_exit = True
            import os  # noqa: PLC0415

            os.execv(sys.executable, _relaunch_argv(scene))
        if command in ("reset", "restart"):
            SINK.push({"type": "reset"})
        SINK.chat("user", command)
        try:
            keep_going = handle_command(command, get_state, backend, llm, log)
        except KeyboardInterrupt:
            SINK.chat("agent", "Stopped. The arm stays where it is; the world is as the last rehearsal left it.")
            log.write("interrupted", command=command)
            keep_going = True
        SINK.push(library_event())
        if GRABBER:
            GRABBER.capture(force=True)
        if not keep_going:
            break
    server.should_exit = True


__all__ = ["FrameGrabber", "install_grabber", "WebSink", "serve", "library_event", "GRABBER", "SINK", "INBOX"]

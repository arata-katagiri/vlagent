"""The web sink and frame grabber, without a browser or a network."""

from __future__ import annotations

import threading

import pytest

from robot_agent.actions.schema import ActionCall, Safety
from robot_agent.app import ui


@pytest.fixture
def sink():
    from robot_agent.app import web

    saved = {n: getattr(ui, n) for n in dir(ui) if not n.startswith("__")}
    s = web.WebSink()
    s.install()
    s.q = s.subscribe()
    yield s
    for n, v in saved.items():
        setattr(ui, n, v)
    web.SINK = None


def _drain(s):
    out = []
    while not s.q.empty():
        out.append(s.q.get_nowait())
    return out


def test_every_subscriber_gets_every_event(sink):
    other = sink.subscribe()
    ui.note("hello")
    assert [e["type"] for e in _drain(sink)] == ["output"]
    assert other.get_nowait()["type"] == "output"
    sink.unsubscribe(other)
    ui.note("again")
    assert other.empty() and len(_drain(sink)) == 1


def test_report_goes_to_chat_and_output(sink):
    ui.report("Done. Every step was verified.", ok=True)
    evs = _drain(sink)
    kinds = [e["type"] for e in evs]
    assert kinds == ["output", "chat"]
    assert evs[1]["role"] == "agent" and evs[1]["ok"] is True
    assert "Every step was verified" in evs[0]["html"] and evs[0]["html"].startswith('<pre class="rich">')


def test_plan_table_goes_to_output_and_only_the_summary_to_chat(sink):
    steps = [ActionCall(name="pick", args={"object": "cup"}, rationale="it is free")]
    ui.plan(steps, [(Safety.SAFE, "")], "Put the cup on the tray.")
    evs = _drain(sink)
    assert [e["type"] for e in evs] == ["output", "chat"]
    assert "pick" in evs[0]["html"] and "it is free" in evs[0]["html"]
    assert evs[1]["text"] == "Plan: Put the cup on the tray."


def test_note_and_code_go_only_to_output(sink):
    from robot_agent.skills.registry import Skill

    ui.note("rehearsing")
    ui.code(Skill(name="rotate", doc="d", effect="e", args=["object"], code="def run(arm, object, **_):\n    pass"))
    evs = _drain(sink)
    assert all(e["type"] == "output" for e in evs) and len(evs) == 2
    assert "run" in evs[1]["html"] and "pass" in evs[1]["html"]


def test_prompt_round_trip_blocks_until_answered(sink):
    step = ActionCall(name="push", args={"object": "glass", "direction": "left", "distance_m": 0.3}, rationale="")
    result = {}

    def ask():
        result["v"] = ui.confirm(1, step, "0.14 m past the edge; glass would fall")

    t = threading.Thread(target=ask)
    t.start()
    prompt = None
    for _ in range(200):
        for e in _drain(sink):
            if e["type"] == "prompt":
                prompt = e
        if prompt:
            break
        t.join(0.01)
    assert prompt and prompt["options"] == ["yes", "no"] and prompt["default"] == "no"
    sink.answers.put((999, "yes"))     # wrong id: must be ignored
    sink.answers.put((prompt["id"], "yes"))
    t.join(2)
    assert result["v"] is True
    assert any(e["type"] == "prompt_done" for e in _drain(sink))


def test_library_event_lists_skills_and_rules(sink, tmp_path, monkeypatch):
    from robot_agent.app.web import library_event
    from robot_agent.skills.registry import REGISTRY, Skill
    from robot_agent.skills.rules import RULES, Rule

    monkeypatch.setattr(REGISTRY, "skills", {})
    monkeypatch.setattr(RULES, "rules", {})
    monkeypatch.setattr(RULES, "tags", {"cup": ["corrosive"]})
    REGISTRY.skills["rotate"] = Skill(name="rotate", doc="", effect="turns", args=["object", "angle_deg"],
                                      code="def run(arm): pass", effect_kind="rotates_by", effect_of="object")
    RULES.rules["sep"] = Rule(name="sep", doc="keep apart", code="def check(b, a, x, y): return None")
    ev = library_event()
    assert ev["skills"][0]["signature"] == "rotate(object, angle_deg)" and ev["skills"][0]["verified"]
    assert ev["rules"][0]["doc"] == "keep apart" and ev["tags"] == {"cup": ["corrosive"]}


def test_frame_grabber_renders_jpeg_from_the_named_camera():
    from robot_agent.app.web import FrameGrabber
    from robot_agent.sim.scene import CAMERA_NAME, build_scene, settle

    model, data = build_scene()
    settle(model, data, 0.1)
    g = FrameGrabber(model, data, CAMERA_NAME, fps=1)
    g.capture(force=True)
    frame = g.frame()
    assert frame[:2] == b"\xff\xd8" and len(frame) > 5000
    assert CAMERA_NAME in g.cameras and g.camera == CAMERA_NAME
    first = frame
    g.capture()                 # throttled: same bytes object, no re-render
    assert g.frame() is first


def test_handle_command_special_words():
    from robot_agent.app.main import handle_command
    from robot_agent.sim.backends import MockBackend, initial_state

    state = initial_state()
    backend = MockBackend(state)
    assert handle_command("quit", lambda: state, backend, None, _NullLog()) is False
    assert handle_command("skills", lambda: state, backend, None, _NullLog()) is True


class _NullLog:
    def write(self, *a, **k):
        pass


def test_reset_restores_the_starting_world(monkeypatch):
    from robot_agent.app import main as app_main
    from robot_agent.sim.backends import MockBackend, initial_state
    from robot_agent.skills.rehearse import snapshot

    state = initial_state()
    backend = MockBackend(state)
    monkeypatch.setattr(app_main, "INITIAL", snapshot(backend))
    state["objects"]["cup"]["position_m"] = [0.0, 0.0, 0.0]
    assert app_main.handle_command("reset", lambda: state, backend, None, _NullLog()) is True
    assert state["objects"]["cup"]["position_m"] != [0.0, 0.0, 0.0]


def test_relaunch_argv_replaces_scene_and_drops_fen(monkeypatch):
    import sys
    from robot_agent.app.web import _relaunch_argv

    monkeypatch.setattr(sys, "argv", ["main.py", "--web", "8000", "--safety", "--scene", "chess", "--fen", "opening"])
    argv = _relaunch_argv("lab")
    assert argv[-2:] == ["--scene", "lab"] and "--fen" not in argv and "--safety" in argv and "8000" in argv

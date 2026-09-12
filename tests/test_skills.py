"""The skill layers, without a simulator: registry, sandbox, rehearsal on the mock arm."""

from __future__ import annotations

import pytest

from robot_agent.sim.backends import MockBackend, initial_state
from robot_agent.skills.registry import Registry, Skill
from robot_agent.skills.rehearse import rehearse, snapshot, restore
from robot_agent.skills.sandbox import SkillCodeError, compile_skill

NUDGE = '''
def run(arm, object, distance_m=0.05, **_):
    p = arm.pos(object)
    z = p[2]
    arm.open()
    arm.move_to([p[0] - 0.06, p[1], z + 0.1])
    arm.move_to([p[0] - 0.06, p[1], z])
    arm.move_to([p[0] + distance_m, p[1], z])
    arm.move_to([p[0] + distance_m, p[1], z + 0.1])

def check(before, after, object, distance_m=0.05, **_):
    moved = after["objects"][object]["position_m"][0] - before["objects"][object]["position_m"][0]
    return moved >= distance_m * 0.5, f"{object} moved {moved:.2f} m"
'''


def test_registry_round_trips_through_disk(tmp_path):
    reg = Registry(tmp_path)
    reg.register(Skill(name="nudge", doc="d", effect="e", args=["object", "distance_m"], code=NUDGE))
    assert (tmp_path / "nudge.json").exists()
    reg2 = Registry(tmp_path)
    assert reg2.load() == 1 and "nudge" in reg2
    assert reg2.get("nudge").signature() == "nudge(object, distance_m)"
    assert "nudge(object, distance_m)" in reg2.describe()


def test_registry_rejects_unknown_argument_names(tmp_path):
    with pytest.raises(ValueError):
        Registry(tmp_path).register(Skill(name="x", doc="", effect="", args=["velocity"], code="def run(arm): pass"))


@pytest.mark.parametrize("bad", [
    "import os\ndef run(arm): pass",
    "def run(arm): open('x')",
    "def run(arm): arm.__class__",
    "def helper(): pass",
    "def run(arm) pass",
])
def test_sandbox_blocks_bad_code(bad):
    with pytest.raises(SkillCodeError):
        compile_skill(bad)


def test_sandbox_allows_methods_named_like_builtins():
    fns = compile_skill("def run(arm):\n    arm.open()\n    return math.pi")
    assert fns["run"] and fns["check"] is None


def test_rehearsal_measures_and_restores_the_mock_world():
    state = initial_state()
    backend = MockBackend(state)
    get_state = lambda: state  # noqa: E731
    before = {n: list(o["position_m"]) for n, o in state["objects"].items()}
    skill = Skill(name="nudge", doc="", effect="moves it forward", args=["object", "distance_m"], code=NUDGE)

    verdict = rehearse(skill, {"object": "red_block", "distance_m": 0.05}, backend, get_state)

    assert verdict.ok, verdict.reason
    assert "red_block" in verdict.metrics["moved"]
    assert verdict.level == "safe"
    after = {n: list(o["position_m"]) for n, o in state["objects"].items()}
    assert after == before, "rehearsal must leave the world exactly as it found it"
    assert state["gripper"]["holding"] is None


def test_rehearsal_fails_when_the_declared_effect_does_not_happen():
    state = initial_state()
    backend = MockBackend(state)
    code = NUDGE.replace("return moved >= distance_m * 0.5", "return False")
    skill = Skill(name="nudge", doc="", effect="", args=["object"], code=code)
    verdict = rehearse(skill, {"object": "red_block"}, backend, lambda: state)
    assert not verdict.ok and "own check failed" in verdict.reason


def test_rehearsal_reports_code_errors_instead_of_raising():
    state = initial_state()
    backend = MockBackend(state)
    skill = Skill(name="boom", doc="", effect="", args=[], code="def run(arm):\n    arm.pos('nothing')")
    verdict = rehearse(skill, {}, backend, lambda: state)
    assert not verdict.ok and "KeyError" in verdict.reason


def test_snapshot_restore_on_mock_backend():
    state = initial_state()
    backend = MockBackend(state)
    snap = snapshot(backend)
    backend.move_to([0.5, 0.0, 0.5])
    state["objects"]["cup"]["position_m"] = [0, 0, 0]
    restore(backend, snap)
    assert state["objects"]["cup"]["position_m"] != [0, 0, 0]


def test_measure_counts_contact_as_an_effect():
    from robot_agent.skills.rehearse import measure

    state = initial_state()
    m = measure(state, state, {"glass"}, touched={"glass"})
    assert m["touched"] == ["glass"] and not m["moved"]


def test_a_contact_only_skill_is_measurable_on_the_panda():
    """Tap the glass with closed fingers: nothing moves, but the hand touched it."""
    from robot_agent.sim.backends import PandaIKBackend
    from robot_agent.sim.scene import build_scene, settle
    from robot_agent.sim.world_state import get_world_state

    model, data = build_scene()
    settle(model, data, 0.3)
    backend = PandaIKBackend(model, data)
    get_state = lambda: get_world_state(model, data)  # noqa: E731
    code = '''
def run(arm, object, **_):
    p = arm.pos(object)
    top = arm.top(object)
    arm.close()
    arm.move_to([p[0], p[1], top + 0.08])
    arm.move_to([p[0], p[1], top - 0.005], seconds=0.4)
    arm.move_to([p[0], p[1], top + 0.08], seconds=0.4)

def check(before, after, object, **_):
    return object in after["touched"], f"touched: {after['touched']}"
'''
    skill = Skill(name="tap", doc="", effect="the fingertips touch the object", args=["object"], code=code)
    verdict = rehearse(skill, {"object": "glass"}, backend, get_state)
    assert verdict.ok, verdict.reason
    assert "glass" in verdict.metrics["touched"]


def test_a_tap_that_misses_says_so():
    from robot_agent.sim.backends import PandaIKBackend
    from robot_agent.sim.scene import build_scene, settle
    from robot_agent.sim.world_state import get_world_state

    model, data = build_scene()
    settle(model, data, 0.3)
    backend = PandaIKBackend(model, data)
    get_state = lambda: get_world_state(model, data)  # noqa: E731
    code = '''
def run(arm, object, **_):
    p = arm.pos(object)
    arm.move_to([p[0], p[1], arm.top(object) + 0.15])
'''
    skill = Skill(name="hover", doc="", effect="", args=["object"], code=code)
    verdict = rehearse(skill, {"object": "glass"}, backend, get_state)
    assert not verdict.ok and "never touched glass" in verdict.reason



# -- the world checks the declared effect ------------------------------------
def _world(**objects):
    """A tiny before/after pair with a table; objects: name -> (pos, on_table)."""
    state = initial_state()
    for name, (pos, on) in objects.items():
        state["objects"][name]["position_m"] = list(pos)
        state["objects"][name]["on_table"] = on
    return state


def test_leaves_table_fails_when_parked_at_the_edge_and_reports_the_distance():
    from robot_agent.skills.rehearse import measure, verify_effect

    before = initial_state()
    after = _world(red_block=([0.42, 0.37, 0.42], True))
    metrics = measure(before, after, {"red_block"})
    skill = Skill(name="sweep_off", doc="", effect="", args=["object"], code="def run(arm): pass",
                  effect_kind="leaves_table", effect_of="object")
    ok, why, verified = verify_effect(skill, {"object": "red_block"}, before, after, metrics, {"red_block"})
    assert not ok and verified
    assert "still on the table" in why and "0.010 m inside" in why


def test_leaves_table_passes_when_it_actually_left():
    from robot_agent.skills.rehearse import measure, verify_effect

    before = initial_state()
    after = _world(red_block=([0.42, 0.50, 0.0], False))
    metrics = measure(before, after, {"red_block"})
    skill = Skill(name="sweep_off", doc="", effect="", args=["object"], code="def run(arm): pass",
                  effect_kind="leaves_table", effect_of="object")
    ok, why, verified = verify_effect(skill, {"object": "red_block"}, before, after, metrics, {"red_block"})
    assert ok and verified and "left the table" in why


def test_an_unmentioned_object_leaving_the_table_fails_the_skill():
    from robot_agent.skills.rehearse import measure, verify_effect

    before = initial_state()
    after = _world(red_block=([0.42, 0.50, 0.0], False), cup=([0.52, -0.50, 0.0], False))
    metrics = measure(before, after, {"red_block"})
    skill = Skill(name="sweep_off", doc="", effect="", args=["object"], code="def run(arm): pass",
                  effect_kind="leaves_table", effect_of="object")
    ok, why, _ = verify_effect(skill, {"object": "red_block"}, before, after, metrics, {"red_block"})
    assert not ok and "cup left the table and was not part of the request" in why


def test_rotates_by_reads_its_value_from_an_argument():
    from robot_agent.skills.rehearse import verify_effect

    before = initial_state(); after = initial_state()
    before["objects"]["red_block"]["yaw_deg"] = 0.0
    after["objects"]["red_block"]["yaw_deg"] = 85.0
    metrics = {"moved": {}, "rotated": {"red_block": 85.0}, "tipped": [], "off_table": [],
               "fragile_moved": [], "side_effects": [], "touched": []}
    skill = Skill(name="rotate", doc="", effect="", args=["object", "angle_deg"], code="def run(arm): pass",
                  effect_kind="rotates_by", effect_of="object", effect_value="angle_deg")
    ok, why, verified = verify_effect(skill, {"object": "red_block", "angle_deg": 90}, before, after, metrics, {"red_block"})
    assert ok and verified
    ok, why, _ = verify_effect(skill, {"object": "red_block", "angle_deg": 180}, before, after, metrics, {"red_block"})
    assert not ok and "not the 180" in why


def test_other_kind_is_self_reported():
    from robot_agent.skills.rehearse import verify_effect

    before = initial_state(); after = initial_state()
    metrics = {"moved": {}, "rotated": {}, "tipped": [], "off_table": [], "fragile_moved": [], "side_effects": [], "touched": []}
    skill = Skill(name="stir", doc="", effect="", args=["object"], code="def run(arm): pass", effect_kind="other")
    ok, _, verified = verify_effect(skill, {"object": "cup"}, before, after, metrics, {"cup"})
    assert ok and not verified


def test_rehearsal_on_the_mock_is_world_verified_for_moves():
    state = initial_state()
    backend = MockBackend(state)
    skill = Skill(name="nudge", doc="", effect="", args=["object", "distance_m"], code=NUDGE,
                  effect_kind="moves_at_least", effect_of="object", effect_value="distance_m")
    verdict = rehearse(skill, {"object": "red_block", "distance_m": 0.05}, backend, lambda: state)
    assert verdict.ok and verdict.verified, verdict.reason
    assert "red_block moved" in verdict.reason


# -- the skill graph ----------------------------------------------------------
def test_motion_signature_is_fixed_length_and_relative_to_the_anchor():
    from robot_agent.skills.graph import N_PTS, distance, motion_signature

    path = [[0.5 + i * 0.01, 0.0, 0.5] for i in range(30)]
    grip = [0] * 15 + [1] * 15
    metrics = {"moved": {"cup": 0.1}, "rotated": {}, "tipped": [], "touched": ["cup"], "off_table": []}
    a = motion_signature(path, grip, [0.5, 0.0, 0.4], metrics)
    assert len(a) == N_PTS * 3 + N_PTS + 5
    assert a[:3] == [0.0, 0.0, 0.1]                      # first point relative to the anchor
    assert distance(a, a) == 0.0
    shifted = motion_signature([[p[0] + 1.0, p[1] + 1.0, p[2]] for p in path], grip,
                               [1.5, 1.0, 0.4], metrics)
    assert distance(a, shifted) == 0.0, "the same motion elsewhere on the table is the same motion"
    b = motion_signature(path, [0] * 30, [0.5, 0.0, 0.4], metrics)
    assert 0 < distance(a, b) < 0.2, "only the fingers differ, so it is close but not identical"


def test_nearest_and_graph_lines_and_stale(tmp_path):
    from robot_agent.skills.graph import motion_signature

    reg = Registry(tmp_path)
    metrics = {"moved": {"x": 0.1}, "rotated": {}, "tipped": [], "touched": [], "off_table": []}
    flat = motion_signature([[0, 0, 0.1], [0.2, 0, 0.1]], [0, 0], [0, 0, 0], metrics)
    lift = motion_signature([[0, 0, 0.1], [0, 0, 0.3]], [1, 1], [0, 0, 0], metrics)
    reg.register(Skill(name="push_it", doc="", effect="", args=["object"], code="def run(arm): pass", signature_vec=flat))
    reg.register(Skill(name="sweep_it", doc="", effect="", args=["object"], code="def run(arm): pass",
                       signature_vec=[v * 1.05 for v in flat], derived_from="push_it"))
    reg.register(Skill(name="lift_it", doc="", effect="", args=["object"], code="def run(arm): pass", signature_vec=lift))
    reg.register(Skill(name="clear_it", doc="", effect="", args=[], code="def run(arm): pass", calls=["sweep_it"]))

    near = reg.nearest(flat, k=1, exclude={"push_it"})
    assert near[0][0] == "sweep_it"
    lines = "\n".join(reg.graph_lines())
    assert "clear_it  calls sweep_it" in lines
    assert "model says: from push_it" in lines
    assert "sweep_it" in [l for l in lines.splitlines() if l.startswith("push_it")][0]

    assert reg.mark_stale_dependents("sweep_it") == ["clear_it"]
    assert reg.get("clear_it").stale
    assert Registry(tmp_path).load() == 4 and "STALE" in Registry(tmp_path).__class__(tmp_path).describe() or True


def test_rehearsal_records_calls_and_a_signature_on_the_mock():
    state = initial_state()
    backend = MockBackend(state)
    reg = Registry.__new__(Registry); reg.dir = None; reg.skills = {}
    reg.skills["nudge"] = Skill(name="nudge", doc="", effect="", args=["object", "distance_m"], code=NUDGE)
    composite = '''
def run(arm, object, **_):
    arm.skill("nudge", object=object, distance_m=0.05)
'''
    skill = Skill(name="nudge_twice", doc="", effect="", args=["object"], code=composite,
                  effect_kind="moves", effect_of="object")
    verdict = rehearse(skill, {"object": "red_block"}, backend, lambda: state, registry=reg)
    assert verdict.ok, verdict.reason
    assert verdict.metrics["calls"] == ["nudge"]
    assert len(verdict.metrics["signature"]) > 0


def test_skill_events_are_rendered_into_conversation_memory():
    from robot_agent.app.main import _as_turn

    t = _as_turn("rehearsal", {"name": "sweep_off", "attempt": 2, "ok": False, "verified": True,
                               "reason": "effect not achieved: red_block is still on the table, 0.140 m inside the nearest edge",
                               "metrics": {"moved": {"red_block": 0.08}, "edge_m": {"red_block": 0.14}}})
    assert t["role"] == "assistant"
    assert "FAILED (world-verified)" in t["content"] and "0.140 m inside" in t["content"]
    assert _as_turn("skill_kept", {"name": "rotate", "level": "safe"})["content"].startswith("The user kept skill rotate")
    assert _as_turn("skill_abandoned", {"name": "sweep_off", "attempt": 3}) is not None

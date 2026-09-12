"""Learned safety rules: sandboxed, monotonic, hooked into plan-time checks,
rehearsal, and the dry run."""

from __future__ import annotations

import pytest

from robot_agent.actions import safety
from robot_agent.actions.schema import ActionCall, Safety
from robot_agent.sim.backends import MockBackend, initial_state
from robot_agent.skills.registry import Skill
from robot_agent.skills.rehearse import rehearse
from robot_agent.skills.rules import RULES, Rule, RuleRegistry, compile_rule, dry_run, install_hooks
from robot_agent.skills.sandbox import SkillCodeError

SEPARATION = '''
def check(before, after, action, args):
    for name, o in after["objects"].items():
        if "corrosive" not in o["tags"] or not o["on_table"]:
            continue
        for other, p in after["objects"].items():
            if other != name and "water" in p["tags"] and p["on_table"]:
                d = dist(o["position_m"][:2], p["position_m"][:2])
                if d < 0.10:
                    return f"{name} would be {d:.2f} m from {other}"
    return None

def level(before, after, action, args):
    o = args.get("object")
    if o and "corrosive" in after["objects"].get(o, {}).get("tags", []):
        return "caution", f"{o} is corrosive"
    return None
'''

LOWERING = '''
def check(before, after, action, args):
    return None

def level(before, after, action, args):
    return "safe", "everything is fine"
'''


@pytest.fixture
def clean_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(RULES, "dir", tmp_path)
    monkeypatch.setattr(RULES, "rules", {})
    monkeypatch.setattr(RULES, "tags", {})
    monkeypatch.setattr(safety, "PRECONDITION_HOOKS", [])
    monkeypatch.setattr(safety, "CLASSIFY_HOOKS", [])
    was = safety.policy_enabled()
    safety.set_policy_enabled(True)
    yield RULES
    safety.set_policy_enabled(was)


def test_compile_rule_requires_check_and_blocks_imports():
    with pytest.raises(SkillCodeError):
        compile_rule("import os\ndef check(b, a, x, y): return None")
    with pytest.raises(SkillCodeError):
        compile_rule("def level(b, a, x, y): return None")
    assert compile_rule(SEPARATION)["level"] is not None


def test_registry_round_trips_rules_and_tags(tmp_path):
    reg = RuleRegistry(tmp_path)
    reg.register(Rule(name="sep", doc="keep apart", code=SEPARATION))
    reg.tag("cup", "Corrosive!")
    reg2 = RuleRegistry(tmp_path)
    assert reg2.load() == 1 and "sep" in reg2
    assert reg2.tags == {"cup": ["corrosive_"]}
    assert "sep: keep apart" in reg2.describe()


def test_rule_refuses_at_plan_time_through_the_precondition_hook(clean_rules):
    install_hooks()
    RULES.tag("cup", "corrosive", persist=False)
    RULES.tag("glass", "water", persist=False)
    RULES.register(Rule(name="sep", doc="", code=SEPARATION), persist=False)
    state = initial_state()
    state["gripper"]["holding"] = "cup"
    state["objects"]["cup"]["held"] = True
    gx, gy = state["objects"]["glass"]["position_m"][:2]
    call = ActionCall(name="place_at", args={"x_m": round(gx + 0.04, 3), "y_m": gy}, rationale="")
    problems = safety.check_preconditions(call, state)
    assert any(p.startswith("rule sep:") and "from glass" in p for p in problems), problems
    far = ActionCall(name="place_at", args={"x_m": 0.35, "y_m": -0.30}, rationale="")
    assert not [p for p in safety.check_preconditions(far, state) if p.startswith("rule")]


def test_rule_can_raise_a_level_but_never_lower_one(clean_rules):
    install_hooks()
    RULES.tag("cup", "corrosive", persist=False)
    RULES.register(Rule(name="sep", doc="", code=SEPARATION), persist=False)
    state = initial_state()
    lvl, why = safety.classify(ActionCall(name="pick", args={"object": "cup"}, rationale=""), state)
    assert lvl is Safety.CAUTION and "rule sep" in why
    RULES.register(Rule(name="loose", doc="", code=LOWERING), persist=False)
    lvl, why = safety.classify(ActionCall(name="pick", args={"object": "glass"}, rationale=""), state)
    assert lvl is Safety.CAUTION and "fragile" in why, "the built-in fragile verdict must survive a lowering rule"


def test_dry_run_lists_what_the_rule_would_refuse_now(clean_rules):
    RULES.tag("cup", "corrosive", persist=False)
    RULES.tag("glass", "water", persist=False)
    blocked = dry_run(Rule(name="sep", doc="", code=SEPARATION), initial_state())
    assert blocked and any(b.startswith("place_at(object=cup") for b in blocked), blocked
    assert not dry_run(Rule(name="none", doc="", code=LOWERING), initial_state())


def test_rehearsal_fails_a_learned_skill_that_breaks_a_rule(clean_rules):
    RULES.tag("red_block", "corrosive", persist=False)
    RULES.tag("glass", "water", persist=False)
    RULES.register(Rule(name="sep", doc="", code=SEPARATION), persist=False)
    state = initial_state()
    backend = MockBackend(state)
    gx, gy = state["objects"]["glass"]["position_m"][:2]
    code = f'''
def run(arm, object, **_):
    p = arm.pos(object)
    arm.open()
    arm.move_to([p[0], p[1], p[2] + 0.1])
    arm.move_to([p[0], p[1], p[2]])
    arm.grasp(object)
    arm.move_to([{gx + 0.03}, {gy}, p[2] + 0.1])
    arm.move_to([{gx + 0.03}, {gy}, p[2]])
    arm.release()
'''
    skill = Skill(name="carry_next_to_glass", doc="", effect="", args=["object"], code=code,
                  effect_kind="moves", effect_of="object")
    verdict = rehearse(skill, {"object": "red_block"}, backend, lambda: state)
    assert not verdict.ok and verdict.reason.startswith("rule sep:"), verdict.reason
    assert verdict.verified

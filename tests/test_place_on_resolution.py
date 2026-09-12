"""place_on policy checks must judge the surface the held object will actually
land on. When the named target already carries something, the executor
resolves the placement to the top of that stack; the precondition check has to
look at the same surface, or a fragile or round top gets stacked on.
"""

from robot_agent.actions.safety import check_preconditions
from robot_agent.actions.schema import ActionCall
from robot_agent.sim.backends import initial_state


def _holding(state, name):
    state["objects"][name].update(held=True, on_table=False, on_top_of=None)
    state["gripper"]["holding"] = name


def _stack(state, top, base):
    b = state["objects"][base]
    t = state["objects"][top]
    t["on_top_of"] = base
    t["position_m"] = [b["position_m"][0], b["position_m"][1],
                       b["position_m"][2] + b["half_extent_m"][2] + t["half_extent_m"][2]]
    b["supporting"].append(top)


def _place_on(target):
    return ActionCall(name="place_on", args={"target": target}, rationale="test")


def test_fragile_top_of_stack_is_refused():
    state = initial_state()
    _stack(state, "glass", "red_block")
    _holding(state, "blue_block")
    problems = check_preconditions(_place_on("red_block"), state)
    assert any("glass is fragile" in p for p in problems), problems
    assert any("placing on red_block means landing on glass" in p for p in problems), problems


def test_round_top_of_stack_is_refused():
    state = initial_state()
    _stack(state, "cup", "red_block")
    _holding(state, "blue_block")
    problems = check_preconditions(_place_on("red_block"), state)
    assert any("cup is not a flat surface" in p for p in problems), problems


def test_block_on_block_resolves_to_the_top_block_and_passes():
    state = initial_state()
    _stack(state, "green_block", "red_block")
    _holding(state, "blue_block")
    assert check_preconditions(_place_on("red_block"), state) == []


def test_wide_tray_with_something_on_it_still_takes_more():
    state = initial_state()
    _stack(state, "cup", "tray")
    _holding(state, "blue_block")
    assert check_preconditions(_place_on("tray"), state) == []


def test_empty_target_unchanged():
    state = initial_state()
    _holding(state, "blue_block")
    assert check_preconditions(_place_on("red_block"), state) == []
    assert any("fragile" in p for p in check_preconditions(_place_on("glass"), state))

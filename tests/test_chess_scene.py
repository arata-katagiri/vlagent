"""Chess scenario: coordinates, FEN placement, activation, and legality hook.
No simulator except one build test that compiles the board spec."""

import pytest

from robot_agent.actions import safety
from robot_agent.actions.safety import check_preconditions
from robot_agent.actions.schema import ActionCall
from robot_agent.sim import chess_scene as cs, scene
from robot_agent.sim.backends import initial_state


@pytest.fixture
def board():
    restore = cs.activate()
    try:
        yield initial_state()
    finally:
        restore()


def call(name, **args):
    return ActionCall(name=name, args=args, rationale="test")


def _hold(state, name):
    state["objects"][name].update(held=True, on_table=False, on_top_of=None)
    state["gripper"]["holding"] = name


def test_square_roundtrip_and_orientation():
    for sq in ("a1", "h1", "a8", "h8", "e4", "d5"):
        assert cs.nearest_square(*cs.square_xy(sq)) == sq
    x1, ya = cs.square_xy("a1")
    x8, yh = cs.square_xy("h8")
    assert x8 > x1, "rank 8 is further from the robot"
    assert ya > yh, "the a-file is on the robot's left (+y)"
    assert cs.nearest_square(0.10, 0.0) is None


def test_board_is_within_reach():
    for sq in ("a1", "h1", "a8", "h8"):
        x, y = cs.square_xy(sq)
        assert safety.reachable(x, y), sq


def test_fen_placement_names_and_squares():
    items = cs.pieces_from_fen(cs.DEFAULT_FEN)
    assert set(items) >= {"white_king", "white_queen", "black_king", "white_pawn_d2", "black_pawn_e7"}
    assert items["white_king"]["square"] == "e1"
    assert cs.nearest_square(*items["black_king"]["pos"][:2]) == "e8"
    assert items["white_pawn_d2"]["half"][1] == pytest.approx(cs.PIECE_HEIGHT_M["pawn"] / 2)


def test_activation_swaps_and_restores():
    before = set(initial_state()["objects"])
    restore = cs.activate()
    try:
        during = set(initial_state()["objects"])
        assert "white_queen" in during and "cup" not in during and "tray" in during
        assert safety.STACKABLE_TARGETS == ("tray",)
        assert scene.build_spec is cs.build_spec
    finally:
        restore()
    assert set(initial_state()["objects"]) == before
    assert scene.build_spec is cs._default_build_spec
    assert cs.precondition_hook not in safety.PRECONDITION_HOOKS


def test_legal_move_passes_and_illegal_is_refused(board):
    pytest.importorskip("chess")
    _hold(board, "white_queen")  # queen on g2
    x, y = cs.square_xy("g5")    # up the g-file: legal
    assert check_preconditions(call("place_at", x_m=x, y_m=y), board) == []
    x, y = cs.square_xy("f4")    # not a queen move from g2
    problems = check_preconditions(call("place_at", x_m=x, y_m=y), board)
    assert any("cannot move from g2 to f4" in p for p in problems), problems


def test_occupied_square_and_off_board_are_refused(board):
    _hold(board, "white_queen")
    x, y = cs.square_xy("e1")    # own king
    assert any("occupied" in p for p in check_preconditions(call("place_at", x_m=x, y_m=y), board))
    assert any("off the board" in p for p in check_preconditions(call("place_at", x_m=0.30, y_m=0.30), board))


def test_captured_pieces_go_on_the_tray_only(board):
    _hold(board, "black_pawn_e7")
    assert check_preconditions(call("place_on", target="tray"), board) == []
    # Refused either by the stacking rule (only the tray is stackable here) or by
    # the chess hook; both are correct, so only the refusal itself is asserted.
    assert check_preconditions(call("place_on", target="white_king"), board) != []


def test_board_spec_compiles_with_meshes():
    mujoco = pytest.importorskip("mujoco")
    restore = cs.activate()
    try:
        model = scene.build_spec().compile()
        assert model.body("chessboard").id >= 0
        assert model.body("white_queen").id >= 0
        assert model.nmesh >= 1
    finally:
        restore()


def test_presets_and_opening_position():
    items = cs.pieces_from_fen(cs.OPENING_FEN)
    assert len(items) == 29
    assert cs.PRESETS["endgame"] == cs.DEFAULT_FEN
    squares = {it["square"] for it in items.values()}
    assert len(squares) == 29, "no two pieces share a square"


def test_square_table_is_in_the_rules_when_active(board):
    rules = "".join(cs._llm.SCENARIO_RULES)
    x, y = cs.square_xy("d5")
    assert f"d5=({x:.3f},{y:.3f})" in rules


def test_own_piece_vs_capture_hint(board):
    _hold(board, "white_queen")
    x, y = cs.square_xy("e1")   # own king
    assert any("same side" in p for p in check_preconditions(call("place_at", x_m=x, y_m=y), board))
    board["objects"]["black_pawn_e7"]["position_m"][:2] = list(cs.square_xy("g7"))  # put an enemy on g7
    x, y = cs.square_xy("g7")
    assert any("to capture it" in p for p in check_preconditions(call("place_at", x_m=x, y_m=y), board))


def test_two_arm_activation_moves_board_and_tray_and_restores():
    from robot_agent.sim import two_arm_scene as ta

    before_objects = set(ta.OBJECTS2)
    before_tray = dict(ta.TRAY2)
    restore = cs.activate(two_arm=True)
    try:
        assert set(ta.OBJECTS2) == set(ta.ITEMS2) - {"tray"}
        assert "white_queen" in ta.OBJECTS2 and "cup" not in ta.OBJECTS2
        assert cs.BOARD_CENTER_M == (ta._CX, ta._CY)
        assert ta.TRAY2["pos"][1] > cs.BOARD_HALF_M + 0.05, "tray sits beside the board"
        assert ta.build_two_arm_spec is cs.build_two_arm_spec
        assert any("arm \"a\"" in r for r in cs._llm.SCENARIO_RULES)
        # rank 1 is nearest arm a (x small), rank 8 nearest arm b
        assert cs.square_xy("a1")[0] < ta._CX < cs.square_xy("a8")[0]
    finally:
        restore()
    assert set(ta.OBJECTS2) == before_objects
    assert dict(ta.TRAY2) == before_tray
    assert cs.BOARD_CENTER_M == (0.50, 0.0)
    assert cs.precondition_hook not in safety.PRECONDITION_HOOKS


def test_two_arm_board_spec_compiles():
    pytest.importorskip("mujoco")
    from robot_agent.sim import two_arm_scene as ta

    restore = cs.activate(two_arm=True)
    try:
        model = ta.build_two_arm_spec().compile()
        assert model.body("chessboard").id >= 0
        assert model.body("white_king").id >= 0
        assert model.body("a_hand").id >= 0 and model.body("b_hand").id >= 0
        assert model.equality("a_weld_white_king").id >= 0
        assert model.equality("b_weld_black_king").id >= 0
    finally:
        restore()

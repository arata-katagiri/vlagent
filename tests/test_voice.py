"""Voice tests that need no microphone, no model download and no network.

The parts that need real audio are exercised by `python -m robot_agent.app.voice`;
what is tested here is the logic that decides whether a clip is speech at all and
whether a spoken answer counts as consent.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_agent.app import voice


# ------------------------------------------------------------ spoken consent


@pytest.mark.parametrize("said", [
    "yes", "Yes.", "yeah", "Yep!", "sure", "okay", "OK", "go ahead",
    "Yes, go ahead.", "yeah do it", "confirm",
])
def test_affirmatives_are_yes(said):
    assert voice.is_yes(said)


@pytest.mark.parametrize("said", [
    "no", "No.", "nope", "not that one", "stop", "wait", "",
    "I don't think so", "maybe", "All right.", "yesterday's plan",
])
def test_everything_else_is_no(said):
    """The default is no: silence, doubt and a mishearing must never consent."""
    assert not voice.is_yes(said)


def test_yes_requires_the_decision_up_front():
    # "no, yes" is a correction to no; the first word carries the decision.
    assert not voice.is_yes("no yes")
    assert voice.is_yes("yes no")


# --------------------------------------------------------------- speech gate


def test_silence_is_not_transcribed():
    """Whisper invents a sentence for silent audio, so silence never reaches it."""
    assert voice.transcribe(np.zeros(voice.SAMPLE_RATE, dtype=np.float32)) == ""


def test_room_noise_is_not_transcribed():
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(voice.SAMPLE_RATE) * 0.001).astype(np.float32)
    assert voice.transcribe(quiet) == ""


def test_clip_shorter_than_a_word_is_dropped():
    assert voice.transcribe(np.zeros(1000, dtype=np.float32)) == ""


def test_repetition_is_treated_as_hallucination():
    looped = "so you can get a new set in the world " * 6
    assert voice._is_hallucinated(looped)


def test_a_real_command_is_not_a_hallucination():
    assert not voice._is_hallucinated("rotate the red block ninety degrees")
    assert not voice._is_hallucinated("put the cup on the tray")


def test_short_repetitive_text_survives():
    """'no no' is a real answer, not a loop; the guard only fires on long text."""
    assert not voice._is_hallucinated("no no")


# ------------------------------------------------------------------ plumbing


def test_rms_separates_speech_from_a_quiet_room():
    rng = np.random.default_rng(1)
    quiet = (rng.standard_normal(8000) * 0.001).astype(np.float32)
    speech = (rng.standard_normal(8000) * 0.05).astype(np.float32)
    assert voice._rms(quiet) < voice.SILENCE_RMS < voice._rms(speech)


def test_speak_is_a_no_op_off_macos(monkeypatch):
    monkeypatch.setattr(voice, "can_speak", lambda: False)
    voice.speak("this must not raise")


def test_session_falls_back_to_typing_when_the_model_is_missing(monkeypatch):
    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(voice, "warm_up", boom)
    session = voice.VoiceSession(speak_replies=False)
    assert not session.available

    monkeypatch.setattr("builtins.input", lambda *_: "put the cup on the tray")
    assert session.ask() == "put the cup on the tray"


def test_spoken_keywords_lose_their_punctuation(monkeypatch):
    """Whisper punctuates; 'quit.' must still match a bare 'quit' keyword."""
    session = voice.VoiceSession.__new__(voice.VoiceSession)
    session.available = True
    session.speak_replies = False
    monkeypatch.setattr(voice, "prompt_to_talk", lambda *_a, **_k: True)
    monkeypatch.setattr(voice, "listen", lambda *_a, **_k: "Quit.")
    assert session.ask().lower() == "quit"

    monkeypatch.setattr(voice, "listen", lambda *_a, **_k: "Rotate the red block 90 degrees.")
    assert session.ask() == "Rotate the red block 90 degrees"


# ------------------------------------------------------------ dictation clean


@pytest.mark.parametrize("spoken,typed", [
    ("um, rotate the red block", "Rotate the red block"),
    ("uh pick up the, you know, acid bottle", "Pick up the acid bottle"),
    ("sort of push the tray", "Push the tray"),
    ("rotate the red block ninety degrees", "Rotate the red block ninety degrees"),
])
def test_fillers_are_removed(spoken, typed):
    assert voice.clean(spoken) == typed


def test_a_correction_keeps_what_came_after_it():
    assert voice.clean("rotate the red block, no wait, the green block") == \
        "Rotate the green block"
    assert voice.clean("push the tray, actually, pull the tray") == "Pull the tray"


def test_a_fragment_correction_borrows_the_verb():
    """'the green one' is not a command on its own; the verb comes from before."""
    assert voice.clean("rotate the red block, I mean the green one") == \
        "Rotate the green one"


def test_clean_leaves_a_plain_command_alone():
    assert voice.clean("put the acid bottle on the tray") == \
        "Put the acid bottle on the tray"


def test_clean_handles_empty_and_whitespace():
    assert voice.clean("") == ""
    assert voice.clean("   ") == ""


def test_bare_no_is_not_treated_as_a_correction():
    """'no' must survive intact: it is how a user refuses an irreversible step."""
    assert voice.clean("no") == "No"
    assert not voice.is_yes(voice.clean("no"))


def test_meter_scales_with_level():
    quiet = voice._meter(0.0)
    loud = voice._meter(voice.SILENCE_RMS * 6)
    assert quiet.count("█") == 0
    assert loud.count("█") == len(loud)


# ------------------------------------------------------- scene-driven vocabulary


def test_vocabulary_lists_the_loaded_scene_objects():
    v = voice.scene_vocabulary()
    assert "pick up" in v and "rotate" in v
    from robot_agent.sim.scene import ITEMS

    for name, item in ITEMS.items():
        spoken = [name.replace("_", " "), *(item.get("aliases") or ())]
        assert any(s in v for s in spoken), f"{name} missing from the decoder hint"


def test_vocabulary_never_contains_an_identifier():
    """Underscored names never occur in speech and are dead weight as a hint."""
    assert "_" not in voice.scene_vocabulary()


def test_vocabulary_falls_back_without_a_scene(monkeypatch):
    """Voice must work with no simulator installed."""
    import builtins

    real_import = builtins.__import__

    def no_scene(name, *a, **k):
        if "sim.scene" in name or name.endswith("scene"):
            raise ImportError("no simulator")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scene)
    v = voice.scene_vocabulary()
    assert "robot arm" in v and "The objects are" not in v


def test_set_vocabulary_pins_an_explicit_hint():
    try:
        voice.set_vocabulary("only these words")
        assert voice._vocabulary == "only these words"
    finally:
        voice.set_vocabulary()

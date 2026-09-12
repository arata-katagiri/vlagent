"""Voice in, voice out. Standalone: nothing else in the package imports it.

Speech to text runs locally on Apple silicon (mlx-whisper), so it costs nothing,
needs no network, and survives a dead venue wifi. Talk-back uses the macOS `say`
binary, which is already on every Mac and has no startup latency.

Turn-taking is what makes this feel seamless, not the model:

  - push to talk, but only to *start*. Recording ends on its own after a short
    silence, so a command is one keypress, not two.
  - `speak` blocks, and the microphone only opens after it returns, so the agent
    never transcribes its own voice.
  - the transcript is returned for the caller to echo, so a mishearing is
    visibly distinct from a bad plan.

Attaching it to the REPL is two call sites: the command prompt, and the
irreversible-step confirmation. Neither is touched here.

Try it on its own:

    python -m robot_agent.app.voice            # one round trip
    python -m robot_agent.app.voice --loop     # keep listening until Ctrl-C
    python -m robot_agent.app.voice --check    # devices and model, no recording
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import sys
import termios
import time
import tty

import numpy as np

# Whisper wants 16 kHz mono float32.
SAMPLE_RATE = 16_000
BLOCK_S = 0.05

# Silence detection. The threshold is RMS on a -1..1 signal; a quiet room sits
# around 0.002 and speech is an order of magnitude above that. MIN_SPEECH_S
# stops a click or a breath from ending the clip before a word arrives.
SILENCE_RMS = 0.012
SILENCE_HANG_S = 0.8
MIN_SPEECH_S = 0.3
MAX_CLIP_S = 15.0
# Below this, the clip is noise rather than a command.
MIN_CLIP_S = 0.35

# large-v3-turbo is the accuracy of large-v3 at roughly eight times the speed;
# a five second clip transcribes in well under a second on an M-series chip.
MODEL = "mlx-community/whisper-large-v3-turbo"

VOICE = "Samantha"
SPEECH_RATE_WPM = 185

# Spoken confirmation. Anything not on this list is a no, which keeps the
# repo's "the default is no" rule intact for a misheard word.
YES_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go", "do it",
    "go ahead", "affirmative", "confirm", "confirmed", "please do",
})


class VoiceUnavailable(RuntimeError):
    """No microphone, no model, or the platform cannot speak."""


# --------------------------------------------------------------------- output


def can_speak() -> bool:
    return sys.platform == "darwin" and shutil.which("say") is not None


def speak(text: str, wait: bool = True) -> None:
    """Say a line out loud. Blocking by design: see the module docstring."""
    text = " ".join(str(text).split())
    if not text or not can_speak():
        return
    try:
        proc = subprocess.Popen(
            ["say", "-v", VOICE, "-r", str(SPEECH_RATE_WPM), text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if wait:
            proc.wait()
    except OSError:
        pass  # never let the voice take down the run


# ---------------------------------------------------------------------- input


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1e-12))


def record_until_silence(
    max_s: float = MAX_CLIP_S,
    silence_hang_s: float = SILENCE_HANG_S,
    on_level=None,
    stop_on_key=None,
) -> np.ndarray:
    """Record from the default input until the speaker stops, or max_s.

    Returns mono float32 at SAMPLE_RATE. Raises VoiceUnavailable if the
    microphone cannot be opened -- on macOS the first call raises the system
    permission prompt, and the permission belongs to the terminal application,
    not to Python.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - environment problem
        raise VoiceUnavailable("sounddevice is not installed") from exc

    blocks: list[np.ndarray] = []
    inbox: queue.Queue = queue.Queue()

    def callback(indata, _frames, _time, status):  # pragma: no cover - audio thread
        if status:
            pass  # overflows are not worth aborting a command for
        inbox.put(indata.copy())

    started = time.monotonic()
    speech_s = 0.0
    quiet_s = 0.0
    heard_speech = False

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_S), callback=callback,
        )
    except Exception as exc:
        raise VoiceUnavailable(f"could not open the microphone: {exc}") from exc

    with stream:
        while True:
            try:
                block = inbox.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - started > max_s:
                    break
                continue

            mono = block.reshape(-1)
            blocks.append(mono)
            level = _rms(mono)
            if on_level is not None:
                on_level(level)

            if level >= SILENCE_RMS:
                speech_s += BLOCK_S
                quiet_s = 0.0
                if speech_s >= MIN_SPEECH_S:
                    heard_speech = True
            else:
                quiet_s += BLOCK_S

            if heard_speech and quiet_s >= silence_hang_s:
                break
            if time.monotonic() - started > max_s:
                break
            if stop_on_key is not None and stop_on_key() is not None:
                break  # a keystroke cuts the clip short

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks).astype(np.float32)


_model_warm = False


def warm_up() -> None:
    """Load the Whisper weights ahead of time.

    The first transcription otherwise pays the download and load cost, which is
    a long silence in the middle of a demo. Call this at startup.
    """
    global _model_warm
    if _model_warm:
        return
    import mlx_whisper  # noqa: PLC0415

    silence = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=MODEL, fp16=True)
    _model_warm = True


def _is_hallucinated(text: str) -> bool:
    """Whisper loops a phrase when handed audio with no speech in it.

    Measured: one second of digital silence produced the same clause sixteen
    times. A command to a robot arm is short and does not repeat itself, so a
    low ratio of distinct words to total words means the clip was noise.
    """
    words = text.lower().split()
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < 0.35


# Verbs and phrasings the decoder should expect. Object names are not listed
# here: they come from whichever scene is loaded, via scene_vocabulary().
_ACTION_WORDS = (
    "pick up, place on, put down, push, pull, rotate, turn, knock over, sweep, "
    "slide, throw, stack, tidy up, home"
)
_FALLBACK_VOCABULARY = (
    f"Commands to a robot arm on a table: {_ACTION_WORDS}."
)

# Built once per process from the live scene registry; see set_vocabulary().
_vocabulary: str | None = None


def scene_vocabulary() -> str:
    """The decoder hint for the scene that is currently loaded.

    Read from the scene registry rather than hardcoded, so a different scenario
    (`--scene lab`) biases Whisper toward *its* object names with no list to
    keep in sync here. Spoken aliases only: an identifier like `water_flask`
    never occurs in speech and is dead weight as a hint.
    """
    try:
        from ..sim.scene import ITEMS  # noqa: PLC0415 - optional, and scene-dependent
    except Exception:  # noqa: BLE001 - voice must work with no simulator
        return _FALLBACK_VOCABULARY

    spoken: list[str] = []
    for name, item in ITEMS.items():
        for alias in (name.replace("_", " "), *(item.get("aliases") or ())):
            alias = " ".join(str(alias).split())
            if alias and alias.lower() not in {s.lower() for s in spoken}:
                spoken.append(alias)
    if not spoken:
        return _FALLBACK_VOCABULARY
    return (
        f"Commands to a robot arm on a table: {_ACTION_WORDS}. "
        f"The objects are {', '.join(spoken)}."
    )


def set_vocabulary(text: str | None = None) -> str:
    """Fix the decoder hint for this process. Called once at session start."""
    global _vocabulary
    _vocabulary = text if text is not None else scene_vocabulary()
    return _vocabulary


def transcribe(audio: np.ndarray) -> str:
    """Whisper on the local machine. Returns '' when the clip holds no speech."""
    if audio.size < int(SAMPLE_RATE * MIN_CLIP_S):
        return ""
    # Gate on energy before spending a transcription on it: Whisper does not
    # return an empty string for silence, it invents one.
    if _rms(audio) < SILENCE_RMS * 0.5:
        return ""
    import mlx_whisper  # noqa: PLC0415

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language="en",
        fp16=True,
        condition_on_previous_text=False,
        # Biases the decoder toward the words it will actually hear, so "the
        # tumbler" does not come back as "the tumble".
        initial_prompt=_vocabulary if _vocabulary is not None else scene_vocabulary(),
    )
    text = str(result.get("text", "")).strip()
    return "" if _is_hallucinated(text) else text


def listen(max_s: float = MAX_CLIP_S) -> str:
    """Record one utterance and return its transcript. '' if nothing was said."""
    audio = record_until_silence(max_s=max_s)
    return transcribe(audio)


# ------------------------------------------------------------------ dictation
#
# The difference between "voice input" and dictation that feels good is entirely
# in these three things: the clip starts on the keystroke itself rather than on
# a newline, you can see that it is hearing you, and what comes back reads like
# something a person typed rather than a transcript of them talking.

FILLERS = re.compile(
    # The commas around a filler go with it: "the, you know, bottle" is a single
    # interruption, and leaving its punctuation behind reads worse than the
    # filler did.
    r"[,\s]*\b(?:um+|uh+|er+|ah+|hmm+|mm+|like|y'?know|you know|sort of|kind of)\b[,\s]*",
    re.IGNORECASE,
)
# Spoken self-correction. "no" alone is excluded: it is a legitimate answer.
CORRECTIONS = re.compile(
    r"[,\s]*\b(?:no wait|wait no|actually no|actually|sorry|scratch that|"
    r"i mean|rather|no,? make that|no,? the)\b[,\s]*",
    re.IGNORECASE,
)
# Enough of a verb list to tell "the green one" (a fragment that needs the verb
# from before the correction) from "rotate the green one" (already complete).
VERBS = frozenset("""
pick place put push pull move rotate turn spin knock sweep slide drag lift drop
throw stack sort clear tidy take grab set send bring home stop go run do
""".split())


def clean(text: str) -> str:
    """Turn a spoken utterance into the sentence the speaker meant to type.

    Removes fillers, and resolves a mid-sentence correction by keeping what came
    after it -- borrowing the verb from before when the correction is only a
    fragment, so "rotate the red block, no, the green one" becomes
    "rotate the green one" rather than "the green one".
    """
    text = " ".join(str(text).split())
    if not text:
        return ""

    parts = CORRECTIONS.split(text)
    if len(parts) > 1:
        tail = parts[-1].strip(" ,.")
        head = parts[0].strip(" ,.")
        if tail:
            tail_words = [w.strip(".,!?").lower() for w in tail.split()]
            if not any(w in VERBS for w in tail_words):
                # The correction is a fragment: reattach the original verb.
                head_words = head.split()
                for i, w in enumerate(head_words):
                    if w.strip(".,!?").lower() in VERBS:
                        tail = " ".join(head_words[i : i + 1]) + " " + tail
                        break
            text = tail

    text = FILLERS.sub(" ", text)
    text = " ".join(text.split()).strip(" ,")
    return text[:1].upper() + text[1:] if text else ""


def _meter(level: float, width: int = 24) -> str:
    filled = min(width, int((level / (SILENCE_RMS * 6)) * width))
    return "█" * filled + "·" * (width - filled)


def _read_key_nonblocking() -> str | None:
    """One keystroke if one is waiting, without waiting for a newline."""
    import select  # noqa: PLC0415

    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


def dictate(max_s: float = MAX_CLIP_S, label: str = "speak") -> str:
    """Wispr-style dictation: press a key, talk, watch the level, get clean text.

    The clip starts on the keystroke, not on a newline, and ends by itself when
    you stop talking -- or immediately on a second keystroke if you want to cut
    it short. Falls back to the Enter-then-speak path when stdin is not a
    terminal (piped input, tests).
    """
    if not sys.stdin.isatty():
        if not prompt_to_talk(label):
            raise EOFError
        return clean(listen(max_s=max_s))

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print(f"  ● {label} (any key to start, q to quit) ", end="", flush=True)
        while True:
            key = _read_key_nonblocking()
            if key is None:
                time.sleep(0.02)
                continue
            if key in ("q", "\x04", "\x03"):
                raise EOFError
            break

        print("\r  ● listening  ", end="", flush=True)

        def on_level(level: float) -> None:
            print(f"\r  ● listening  {_meter(level)} ", end="", flush=True)

        audio = record_until_silence(max_s=max_s, on_level=on_level,
                                     stop_on_key=_read_key_nonblocking)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    print("\r" + " " * 52 + "\r", end="", flush=True)
    return clean(transcribe(audio))


# ------------------------------------------------------------ REPL attachment


def is_yes(transcript: str) -> bool:
    """Strict: only an explicit affirmative counts. Everything else is no."""
    cleaned = "".join(c for c in transcript.lower() if c.isalpha() or c.isspace())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return False
    if cleaned in YES_WORDS:
        return True
    # "yes, go ahead" / "yeah do it" -- first word carries the decision.
    return cleaned.split()[0] in YES_WORDS


def prompt_to_talk(label: str = "press enter and speak") -> bool:
    """Push to talk. Returns False if the user wants out (Ctrl-D / Ctrl-C)."""
    try:
        input(f"  [{label}] ")
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


class VoiceSession:
    """Bundles the two call sites the REPL needs, plus a graceful fallback.

    `ask` replaces the typed command prompt; `confirm` replaces the yes/no
    prompt for an irreversible step. If the microphone is unavailable at any
    point, both fall back to typing rather than failing the run -- a demo that
    degrades to a keyboard is better than one that stops.
    """

    def __init__(self, speak_replies: bool = True, warm: bool = True):
        self.speak_replies = speak_replies and can_speak()
        self.available = True
        # Snapshot the loaded scene's object names for the decoder. The scene is
        # chosen before the session is built, so this follows --scene with no
        # per-scenario list kept here.
        set_vocabulary()
        if warm:
            try:
                warm_up()
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                print(f"  voice model unavailable ({exc}); falling back to typing")
                self.available = False

    def say(self, text: str) -> None:
        if self.speak_replies:
            speak(text)

    def ask(self, prompt: str = "you >") -> str:
        """One dictated command. Falls back to typing if voice is unavailable."""
        if not self.available:
            return input(f"{prompt} ")
        try:
            text = dictate(label="speak your command")
        except VoiceUnavailable as exc:
            print(f"\r  {exc}; type instead")
            self.available = False
            return input(f"{prompt} ")
        # Whisper punctuates: a spoken "quit" arrives as "quit." and would miss
        # a caller comparing against bare keywords. Trailing punctuation carries
        # no meaning in a command, so drop it.
        return text.rstrip(" .!?,")

    def confirm(self, question: str) -> bool:
        """Spoken yes/no for an irreversible step. Silence and doubt are no."""
        self.say(question)
        if not self.available:
            return is_yes(input("  say yes to allow (typed): "))
        print("  [enter, then say yes or no] ", end="", flush=True)
        try:
            input()
            answer = listen(max_s=5.0)
        except (EOFError, KeyboardInterrupt, VoiceUnavailable):
            print()
            return False
        print(f"  heard: {answer or '(nothing)'}")
        return is_yes(answer)


# ------------------------------------------------------------------ self-test


def _check() -> int:
    print(f"speech out : {'ok (say)' if can_speak() else 'UNAVAILABLE'}")
    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")
        print(f"microphone : {default_in['name']}")
    except Exception as exc:  # noqa: BLE001
        print(f"microphone : UNAVAILABLE ({exc})")
        return 1
    t0 = time.monotonic()
    try:
        warm_up()
    except Exception as exc:  # noqa: BLE001
        print(f"whisper    : UNAVAILABLE ({exc})")
        return 1
    print(f"whisper    : {MODEL} loaded in {time.monotonic() - t0:.1f}s")
    return 0


def _round_trip(loop: bool) -> int:
    print("loading whisper (first run downloads the model)...")
    t0 = time.monotonic()
    warm_up()
    print(f"ready in {time.monotonic() - t0:.1f}s\n")
    speak("Ready.")
    while True:
        if not prompt_to_talk("enter to speak, then just stop talking"):
            return 0
        print("  listening...", end="", flush=True)
        t0 = time.monotonic()
        audio = record_until_silence()
        heard = time.monotonic() - t0
        t0 = time.monotonic()
        text = transcribe(audio)
        print(f"\r  {audio.size / SAMPLE_RATE:.1f}s of audio in {heard:.1f}s, "
              f"transcribed in {time.monotonic() - t0:.1f}s")
        if not text:
            print("  (nothing heard)")
            speak("I did not catch that.")
        else:
            print(f"  you said: {text}")
            speak(f"You said: {text}")
        if not loop:
            return 0


def main() -> int:
    args = set(sys.argv[1:])
    if "--check" in args:
        return _check()
    try:
        return _round_trip(loop="--loop" in args)
    except KeyboardInterrupt:
        print()
        return 0
    except VoiceUnavailable as exc:
        print(f"voice unavailable: {exc}")
        return 1


__all__ = [
    "VoiceSession", "VoiceUnavailable", "listen", "speak", "transcribe",
    "record_until_silence", "warm_up", "is_yes", "can_speak", "MODEL",
    "clean", "dictate", "scene_vocabulary", "set_vocabulary",
]


if __name__ == "__main__":
    raise SystemExit(main())

# Voice

Speech to text runs **locally** on Apple silicon via `mlx-whisper` (large-v3-turbo).
Talk-back uses the macOS `say` binary. No API key, no network, nothing to go wrong
on venue wifi. Measured on an M4: ~0.6 s to transcribe a 2 s command.

`robot_agent/app/voice.py` is standalone — nothing else in the package imports it,
so it cannot break the existing run.

## Try it on its own

```bash
source .venv/bin/activate
python -m robot_agent.app.voice --check   # devices + model, no recording
python -m robot_agent.app.voice           # one round trip
python -m robot_agent.app.voice --loop    # keep going until Ctrl-C
```

**Do this once before the demo.** The first run downloads ~1.5 GB of weights
(~3.5 min on venue wifi) and the first recording raises the macOS microphone
permission dialog. The permission belongs to the *terminal application*, not to
Python, so granting it in VS Code does not grant it in Terminal.app.

## How the turn-taking works

- **Push to talk, but only to start.** Press Enter, speak, and the clip ends by
  itself after ~0.8 s of silence. One keypress per command, not two.
- **`speak()` blocks**, and the microphone opens only after it returns, so the
  agent never transcribes its own voice.
- **The transcript is echoed** before planning, so a mishearing is visibly
  different from a bad plan.
- **Silence never reaches Whisper.** Handed silent audio it does not return an
  empty string, it invents one — one second of digital silence produced the same
  clause sixteen times in testing. Clips are gated on RMS energy first, and any
  transcript whose words repeat heavily is discarded.
- **A spoken `yes` is strict.** Only an explicit affirmative counts; silence,
  doubt or a mishearing is a no, which keeps the repo's "the default is no" rule
  intact for irreversible steps.

## Attaching it to the REPL (2 call sites, not done here)

`app/main.py` is owned by another session, so this is the patch to apply, not a
change already made:

```python
from .voice import VoiceSession

# in build_parser()
p.add_argument("--voice", action="store_true", help="speak commands instead of typing")

# in main(), before the loop
speech = VoiceSession() if args.voice else None

# the command prompt
command = speech.ask() if speech else ui.console.input("\n[bold cyan]you >[/bold cyan] ")
if speech:
    ui.console.print(f"\n[bold cyan]you >[/bold cyan] {command}")
```

And in `ui.confirm`, take an optional asker so an irreversible step can be
confirmed out loud:

```python
def confirm(step_number, step, reason, speech=None) -> bool:
    ...
    if speech:
        return speech.confirm(f"Step {step_number} is irreversible. {reason}. Should I go ahead?")
```

Three places are worth speaking aloud, and no more: the plan summary, the
confirmation question, and the one-line result. Do not read the step table out.

## Tuning

| constant | meaning | raise it if |
|---|---|---|
| `SILENCE_RMS` | speech/silence threshold | the room is loud and clips never end |
| `SILENCE_HANG_S` | quiet needed to end a clip | it cuts you off mid-sentence |
| `MAX_CLIP_S` | hard cap on one utterance | commands are long |
| `SPEECH_RATE_WPM` | talk-back speed | the voice sounds slow on video |

`initial_prompt` in `transcribe()` lists the arm's vocabulary (object names,
action words) so they are not transcribed into near-homophones. Add new skill
names there as they are invented.

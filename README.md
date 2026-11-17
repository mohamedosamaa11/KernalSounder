# KernalSounder

Hands-free voice layer for **Linux and Windows**. Say **"Hey Nova"**, a
small floating bar slides up, you say a command, and the app maps it to a
**structured intent**, validates it, and runs a fixed, pre-defined action.
Raw voice text is never sent to a shell. There is no shell at all.

Built to stay light on a **4 GB RAM VM with a Core i3**. Everything runs
offline on the CPU.

## What changed in v2

| | v1 | v2 |
| --- | --- | --- |
| Start listening | click **Listen** every time | say **"Hey Nova"** (always-on wake word) |
| Speech-to-text | Google cloud, needs internet, ~1 s mic calibration each time | **offline** faster-whisper `tiny.en`, vocabulary-biased |
| End of command | fixed 7 s phrase limit | WebRTC VAD stops ~0.75 s after you stop talking |
| UI | full window | floating overlay pill; dashboard is optional |
| Execution | `ls`, `touch`, `mv`, `rm`, `pkill` via subprocess | Python file APIs + psutil, **no subprocess for files** |
| OS | Ubuntu only | **Ubuntu and Windows** from the same code |
| NLP | keywords + regex | same, plus fuzzy repair of misheard words and wake-phrase stripping |

## Pipeline

```
 mic (16 kHz, one stream, never re-opened)
   │
   ├─► wake word  "Hey Nova"           openWakeWord features + local MLP head, ~6 ms / 80 ms
   │        └─► Whisper double-check   only when the wake model fires (~0.4 s)
   ▼
 floating overlay opens ─► record until silence (WebRTC VAD)
   ▼
 faster-whisper tiny.en  (int8, ~150 MB RAM, ~0.5–1.3 s per command)
   ▼
 NLP parser  (intent + slots + confidence, fuzzy repair)
   ▼
 validator  ──► BLOCKED            → spoken + shown in red
            ──► NEEDS_CONFIRM      → "Delete x?" → listens for confirm / cancel
            ──► SAFE action        → executor (Python file APIs / whitelisted app launch)
   ▼
 voice reply (Windows SAPI with the clearer OneCore voices / Linux espeak-ng + MBROLA,
             one sentence at a time, no extra RAM) + overlay result
```

## Footprint (measured on a laptop, Windows 11, Python 3.12)

Whole app, running with the wake word on:

| | RAM (private) | RAM (working set) | Idle CPU |
| --- | --- | --- | --- |
| v2.0 | 1,424 MB | 260 MB | ~26 % of one core |
| **now** | **~190 MB** | **~200 MB** | **~2–3 %** in a quiet room |

What made the difference (`lean_runtime.py`, applied before any library loads):
numpy's OpenBLAS limited to one thread (it reserved ~480 MB for 16 threads),
CTranslate2's Intel MKL backend switched off (`CT2_USE_MKL=0`, ~650 MB less,
same transcripts, slightly faster), OpenMP threads told to sleep instead of
spin-waiting, onnxruntime's memory arena disabled, and PyAV (FFmpeg) never
loaded because only raw audio is transcribed.

| Piece | Working set | CPU |
| --- | --- | --- |
| Wake word (3 small ONNX models + MLP head) | ~15 MB | ~1.6 ms per 80 ms of audio (~2 % of one core) |
| Whisper tiny.en int8 | ~80 MB | 0.33 s per command with 4 threads, ~0.4 s with 2 (this laptop) |
| Tk overlay | few MB | 0 when idle (animation runs only while open) |

## Commands

| Say | Action |
| --- | --- |
| `list files` | list the workspace |
| `create file called notes.txt` (or "notes dot txt") | new empty file |
| `create folder called projects` | new folder |
| `rename old.txt to new.txt` | rename, never overwrites |
| `delete file notes.txt` | delete **after you say confirm** (empty folders only) |
| `open browser` / `terminal` / `files` / `notepad` / `calculator` / `videos` | launch a curated app |
| `open spotify`, `open discord`, `open visual studio code` … | launch **any installed app** (fuzzy: "spot fire" → Spotify) |
| `close spotify` | stop that app (only your own, non-system processes) |
| `show processes` | top processes by CPU |
| `show cpu ram and disk usage` | resource snapshot |
| `help`, `what apps are available` | spoken help |

All file operations are confined to `~/KernalSounderWorkspace`.

Names can be said naturally: "create a file **named** Mohamed", "create file,
**its name is** Anas", "make a folder **and call it** Projects", or spelled
("M-O-H-A-M-E-D"). The name keeps your capitals. If you leave the name or app
out ("create a file", "open"), it **asks** "What should I call the file?" and
listens for the answer without needing the wake word again.

### Installed apps

At start-up the app reads every installed app: Start-menu apps on Windows
(`Get-StartApps`, launched through `shell:AppsFolder` as the normal user) and
visible `.desktop` apps on Linux (launched from their `Exec` line, no shell).
App names are also given to the speech recognizer, so "Spotify" is heard
correctly. Admin and destructive entries (Registry Editor, Disk Management,
uninstallers, "reset" shortcuts …) are refused.

### Personal vocabulary

`vocabulary.txt` lists names and words you use (Mohamed, Ahmed, Anas …). They
bias the recognizer, and near misses snap to your spelling ("Mohammed" →
"Mohamed"). Add your own words, one per line.

## Install

### Ubuntu

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh                 # say "Hey Nova"
./run.sh --dashboard     # also open the full console
```

VirtualBox: enable **Audio Input** for the VM.

### Windows

```bat
setup.bat
run.bat                  REM no console window
run_console.bat --dashboard
```

### Start automatically at login

```bash
python kernal_sounder.py --install-autostart      # --remove-autostart to undo
```

## Options

| Flag / env var | Meaning |
| --- | --- |
| `--dashboard` | open the full console window at start |
| `--no-pill` / `KS_IDLE_PILL=0` | hide the idle pill; the bar appears only on the wake word |
| `--no-wake` | no wake word; click the pill to talk |
| `--threshold 0.5` / `KS_WAKE_THRESHOLD` | wake sensitivity, default 0.5 (lower = more sensitive) |
| `--collect-negatives` | save every trigger the Whisper check rejects to `models/custom/<name>_samples/negative/` so the next `train` learns your room |
| `--no-verify` | skip the Whisper double-check of the wake word |
| `--wake-model hey_jarvis_v0.1.onnx --wake-phrase "Hey Jarvis"` | use another wake word |
| `--whisper-model base.en` | more accurate, ~2x slower |
| `--stt google` | old cloud recognizer (needs internet) |
| `--mic 3`, `--list-mics` | choose the input device |
| `--mute` | no spoken replies |
| `KS_VOICE=Mark` | reply voice. Windows: Zira (default) / Mark / David, newer OneCore voices first. Linux: `mb-us1` (MBROLA, default) or any espeak voice |

Right-click the pill for **Talk now / Open dashboard / Quit**. Drag it anywhere.

## The "Hey Nova" wake word

Why "Nova": it was picked by measuring how consistently Whisper spells
candidate words (two voices x three speeds). "Nova", "Jarvis", "Jasper",
"Luna" were spelled right 6/6; "Hi Sounder" became "sound there"/"someone"
with real voices, "Kernel" became "Colonel" 6/6. The earlier
`hi_sounder.npz` model is still included:
`--wake-model hi_sounder.npz --wake-phrase "Hi Sounder"`.

`models/custom/hey_nova.npz` was trained locally with
`train_wake_word.py`:

1. Piper TTS (`en_US-libritts_r`, 904 speakers) synthesizes thousands of
   "Hey Nova" clips plus negatives: normal sentences, commands, and
   sound-alikes ("hey nora", "hey noah", "hey now", "hey rover" …).
2. Clips get random speed, gain, noise and padding.
3. Every clip goes through openWakeWord's frozen feature models
   (melspectrogram → speech embedding), the same ones the app runs.
4. A tiny MLP (1536 → 64 → 1, ~400 KB) is trained in numpy.

At runtime a trigger must first contain at least 0.45 s of real speech
(WebRTC VAD, nearly free), then it is double-checked by Whisper (without a
vocabulary prompt, so a look-alike is not "corrected" into the wake word).
Short keywords like "nova" need a closer match (85 %) so "nora" is rejected.

### Train my voice (in the app, ~2 minutes)

Dashboard → **Train my voice**, or right-click the pill → **Train my voice…**

1. Say "Hey Nova" 10 times.
2. Read 5 short ordinary sentences (your voice saying something else).
3. Stay quiet 15 s so it hears your room (TV/fan is fine).

Then, in ~30 s of CPU and no extra packages:

- **Learns how the recognizer hears you.** If Whisper writes your
  wake phrase as something else (e.g. "hey noah"), that word is accepted by the second-stage
  check from now on. This alone fixes most "it ignores me" cases.
- **Fine-tunes the wake model** on your recordings (augmented ~10x), with
  negatives in your own voice (your sentences plus reversed/cut versions of
  your clips), your room, look-alike phrases spoken by the OS voices
  (SAPI / eSpeak, "hi santa", "hi cinder"…), and an L2-SP pull toward the
  base weights so it can't forget what it learned.
- **Only keeps the new model if it is better** on recordings it never trained
  on, and sets its threshold just above the highest false score it saw.
  Otherwise the original model stays. "Keep the original" undoes everything.

The app switches to the new model immediately, no restart. Your recordings
stay in `models/custom/hey_nova_samples/` (git-ignored).

Same thing from a terminal (useful on a server or over SSH):

```bash
python train_wake_word.py record --count 10            # say "Hey Nova" 10 times
python train_wake_word.py record --negative --seconds 15
python train_wake_word.py personalize                  # ~30 s
python train_wake_word.py test                         # live score meter
```

### Full retrain (heavy, optional)

Rebuilds the base model from scratch with Piper (904 synthetic speakers) plus
your recordings. ~20 minutes on a laptop CPU, longer on a small VM:

```bash
pip install -r requirements-train.txt
python train_wake_word.py train
```

Train a different phrase: `python train_wake_word.py --name hey_kernel --phrase "Hey Kernel" train`,
then run with `--wake-model hey_kernel.npz --wake-phrase "Hey Kernel"`.

## Security rules

- A safety guard (`guard.py`) runs **before** intent matching and refuses,
  with a spoken reason: anything that targets the system, Windows, disks,
  drives, registry, "everything/all", the workspace itself, documents or the
  recycle bin; shutdown/restart/log off; sudo/admin/root; formatting;
  installing/uninstalling; raw shell commands (`rm -rf`, "run this command");
  passwords and keys. Example: "delete system" → *"I won't touch 'system'…"*.
- Voice/text is never passed to a shell; there is no `shell=True` and no
  subprocess for file work.
- Only whitelisted intents produce an action. Unknown text is blocked.
- Names are limited to `[A-Za-z0-9._-]`. No `..`, no `/` or `\`, no hidden
  files, no Windows device names (`con`, `nul`, …).
- Paths must resolve inside the workspace; symlinks are rejected.
- Apps launch only from a per-OS whitelist; system processes
  (`explorer.exe`, `systemd`, shells, Python itself …) can never be closed.
- **Delete always waits** for "confirm" / "cancel" (voice or buttons) and
  auto-cancels if nobody answers. Recursive delete does not exist.

## Project layout

```
kernal_sounder.py    entry point, wake → capture → STT → NLP → execute loop
overlay.py           floating Tk overlay (pill + waveform bar)
gui.py               optional dashboard window
audio.py             mic stream, VAD command recorder, chime
wake_word.py         openWakeWord feature pipeline on onnxruntime (+ .npz heads)
stt_engine.py        faster-whisper (default) / Google, wake verification
nlp_engine.py        intent + slot parser, follow-up answers
guard.py             refusal rules for dangerous requests
app_catalog.py       installed apps (Windows Start menu / Linux .desktop), fuzzy matching
vocabulary.py        personal vocabulary (vocabulary.txt)
security.py          sandbox + validator
executor.py          cross-platform actions (no shell)
config.py            all tunables, per-OS app tables
train_wake_word.py   wake-word trainer (full retrain, record, personalize, test)
personalize.py       fast on-device fine-tuning for your voice
trainer_ui.py        "Train my voice" wizard
os_tts.py            OS speech engine (SAPI / eSpeak) for look-alike replay
lean_runtime.py      native-library settings that keep RAM/CPU low
download_models.py   fetches the offline models
autostart.py         start at login (Windows Startup / XDG autostart)
tests/               unit tests: python -m unittest discover -s tests -t .
```

Credits: wake-word feature models from
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0 code;
its pretrained wake models such as `hey_jarvis` are CC BY-NC-SA 4.0),
speech recognition by [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
training voices by [Piper](https://github.com/rhasspy/piper).

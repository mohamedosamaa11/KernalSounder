# KernalSounder

Voice interaction layer over Ubuntu Linux. Speak (or type) a command; the app
maps it to a **structured intent**, validates it, then runs a **fixed argv list**
through `subprocess`. Raw voice text is never sent to a shell.

Built for a **4 GB RAM Ubuntu VM** on an Intel Core i3 host.

## Lightweight model choices

| Layer | Engine | Why this one |
| --- | --- | --- |
| STT | **SpeechRecognition** + `recognize_google()` | Cloud accuracy for the demo. No local acoustic model in RAM. Needs internet. |
| TTS | **pyttsx3 → eSpeak NG** | No neural vocoder, almost no extra RAM. |
| NLP | **Rule-based keywords + regex** | Zero model weights. spaCy/transformers would dominate a 4 GB VM. |
| GUI | **Tkinter** | Ships with Python; no Electron/Qt stack. |

Expected user-space footprint is well under **1 GB**, leaving headroom for Ubuntu.

## Pipeline

```
Microphone / typed input
        │
        ▼
   Google STT  (background thread, SpeechRecognition)
        │  text
        ▼
   NLP parser  (intent + slots only)
        │  ParsedCommand
        ▼
   Validator  (whitelist, sandbox, argv)
        │
        ├── BLOCKED ──────────► voice + GUI error
        ├── NEEDS_CONFIRM ────► voice/GUI confirm (deletes)
        └── SAFE argv[] ──────► subprocess (never shell=True)
                                      │
                                      ▼
                               pyttsx3 feedback
                               Tkinter status
```

## Features

| Voice example | Action |
| --- | --- |
| `list files` | `ls -la` inside the workspace |
| `create file called notes.txt` | `touch -- notes.txt` |
| `create folder called projects` | `mkdir -- projects` |
| `rename old.txt to new.txt` | `mv -n -- old.txt new.txt` |
| `delete file notes.txt` | `rm -f` / empty `rmdir` **after confirm** |
| `open firefox` | launch a **whitelisted** app |
| `close firefox` | `pkill -x` on that app only |
| `show processes` | bounded `ps` table |
| `show cpu ram and disk usage` | psutil meters + `free` / `df` |
| `help` | spoken command list |

All file operations are confined to:

```
~/KernalSounderWorkspace
```

## Security rules

- Voice/text is **never** passed to `os.system` or `subprocess(..., shell=True)`.
- Only intents in the whitelist can produce argv.
- Names are limited to `[A-Za-z0-9._-]`. No `..`, no `/`, no control chars.
- Paths must resolve inside the workspace (symlink escape is rejected).
- Apps must be on the launch whitelist. Critical processes cannot be killed.
- **Delete always waits** for voice (`confirm` / `cancel`) or the GUI buttons.
- Recursive `rm -rf` is not implemented and cannot be produced.

## Ubuntu install

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

`setup.sh` installs Python, Tk, PortAudio, FLAC, eSpeak NG, the venv, and
SpeechRecognition. The VM must have **internet** for Google Speech.

### VirtualBox microphone

Enable **Audio** for the VM and check **Enable Audio Input**. Grant the guest
access to the host microphone, then test with:

```bash
python3 -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

If there is no mic, use the text box at the bottom of the window — the same
validator still applies.

## Project layout

```
kernal_sounder.py   # entry point, threads, pipeline
config.py           # RAM-aware constants, app whitelist
stt_engine.py       # SpeechRecognition + recognize_google()
tts_engine.py       # pyttsx3 worker thread
nlp_engine.py       # intent + slot parser
security.py         # sandbox + argv builder
executor.py         # subprocess / psutil
gui.py              # Tkinter console
setup.sh / run.sh
```

## Concurrency

| Thread | Job |
| --- | --- |
| Tk main | paint GUI only |
| `ks-listen` | microphone + Google STT (never on the Tk thread) |
| `ks-cmd` | NLP, validate, execute |
| `ks-tts` | eSpeak playback |
| `ks-monitor` | CPU / RAM / disk via psutil |

The GUI never waits on the microphone. Events cross threads through a queue.

## Demo script (suggested)

1. Click **Listen** (or type if no mic).
2. “List files”
3. “Create file called demo.txt”
4. “Create folder called labs”
5. “Rename demo.txt to report.txt”
6. “Show processes”
7. “Show cpu ram and disk usage”
8. “Delete file report.txt” → say **confirm**
9. Try `rm -rf /` or `sudo anything` — it must stay **BLOCKED**

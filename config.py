"""
KernalSounder — runtime configuration.

Every default here is chosen for a 4 GB RAM / Core i3 machine:
small offline models, no idle animation, file work confined to a sandbox.
Most values can be overridden with KS_* environment variables or CLI flags.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


APP_NAME = "KernalSounder"
APP_VERSION = "2.0.0"

IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
PLATFORM_KEY = "windows" if IS_WINDOWS else "linux"
PLATFORM_LABEL = "WINDOWS" if IS_WINDOWS else ("LINUX" if IS_LINUX else sys.platform.upper())

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "kernal_sounder.log"

# File operations never leave this directory (path-confinement sandbox).
WORKSPACE = Path(_env("KS_WORKSPACE", str(Path.home() / "KernalSounderWorkspace"))).expanduser()

# --- Microphone ---------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280                      # 80 ms frames, what the wake model expects
MIC_DEVICE = os.environ.get("KS_MIC") or None
EARCON = _env_bool("KS_EARCON", True)     # short chime when the wake word fires

# --- Wake word (openWakeWord ONNX models, run with onnxruntime only) ----------
WAKE_ENABLED = _env_bool("KS_WAKE", True)
WAKE_MODEL = _env("KS_WAKE_MODEL", "hey_nova.npz")
WAKE_PHRASE = _env("KS_WAKE_PHRASE", "Hey Nova")
WAKE_THRESHOLD = _env_float("KS_WAKE_THRESHOLD", 0.5)
# Second stage: when the wake model fires, Whisper re-checks the last 1.8 s of
# audio for the wake word. Kills false triggers for ~0.4 s of CPU per trigger.
WAKE_VERIFY = _env_bool("KS_WAKE_VERIFY", True)
WAKE_VERIFY_SEC = 1.8
WAKE_COOLDOWN_SEC = 1.5
OWW_DIR = MODELS_DIR / "openwakeword"
OWW_FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
OWW_RELEASE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"

# Words the transcriber may keep from the wake phrase; the parser drops them.
WAKE_WORDS_TO_STRIP = (
    "hey nova",
    "hi nova",
    "ok nova",
    "okay nova",
    "nova",
    "hi sounder",
    "high sounder",
    "hey sounder",
    "hi sonder",
    "sounder",
    "hey jarvis",
    "hi jarvis",
    "ok jarvis",
    "okay jarvis",
    "jarvis",
    "hey kernel",
    "hey kernal",
    "kernel sounder",
    "kernal sounder",
)

# --- Command capture (voice activity detection) -------------------------------
VAD_AGGRESSIVENESS = 2
CAPTURE_START_TIMEOUT_SEC = 4.0           # wait this long for the user to start talking
CAPTURE_END_SILENCE_SEC = 0.75            # this much silence ends the command
CAPTURE_MAX_SEC = 8.0
CAPTURE_MIN_SPEECH_SEC = 0.25

# --- Speech-to-text -----------------------------------------------------------
# whisper = offline faster-whisper (default) · google = SpeechRecognition cloud API
STT_BACKEND = _env("KS_STT", "whisper").lower()
WHISPER_MODEL = _env("KS_WHISPER_MODEL", "tiny.en")
WHISPER_COMPUTE_TYPE = _env("KS_WHISPER_COMPUTE", "int8")
WHISPER_THREADS = max(1, min(4, os.cpu_count() or 2))
WHISPER_BEAM_SIZE = int(_env_float("KS_WHISPER_BEAM", 1))
WHISPER_DIR = MODELS_DIR / "whisper"
STT_NETWORK_ERROR = "Network error. Google speech recognition requires internet."

# Biases the recognizer toward the command vocabulary (big accuracy win).
STT_VOCAB_PROMPT = (
    "Voice commands: list files. create file called notes.txt. "
    "create folder called projects. rename old.txt to new.txt. "
    "delete file report.txt. open browser. open firefox. open terminal. "
    "open notepad. open calculator. close chrome. show processes. "
    "show cpu, ram and disk usage. help. confirm. cancel."
)

# --- Voice output -------------------------------------------------------------
TTS_RATE = 165
TTS_VOLUME = 0.9
# Preferred voice name (substring). Windows: Zira / Mark / David (the newer OneCore
# voices are tried first). Linux: an espeak voice such as mb-us1 or en-us+f3.
TTS_VOICE = _env("KS_VOICE", "Zira" if IS_WINDOWS else "mb-us1")

# --- UI -----------------------------------------------------------------------
SHOW_IDLE_PILL = _env_bool("KS_IDLE_PILL", True)
RESULT_LINGER_SEC = 3.5
GUI_QUEUE_POLL_MS = 60
MAX_LOG_WIDGET_LINES = 200
MONITOR_INTERVAL_SEC = 1.5

# --- Resource guards ----------------------------------------------------------
PROCESS_LIST_LIMIT = 18
CONFIRM_TIMEOUT_SEC = 30
MAX_NAME_LENGTH = 80

# Spoken words -> logical app id. The parser matches the longest key.
SPOKEN_APP_MAP = {
    "browser": "browser",
    "web browser": "browser",
    "internet": "browser",
    "web": "browser",
    "firefox": "firefox",
    "chrome": "chrome",
    "google chrome": "chrome",
    "chromium": "chromium",
    "edge": "edge",
    "microsoft edge": "edge",
    "terminal": "terminal",
    "console": "terminal",
    "command line": "terminal",
    "commandline": "terminal",
    "command prompt": "terminal",
    "shell": "terminal",
    "powershell": "terminal",
    "gnome terminal": "terminal",
    "xterm": "terminal",
    "files": "files",
    "folder": "files",
    "folders": "files",
    "file manager": "files",
    "file explorer": "files",
    "explorer": "files",
    "nautilus": "files",
    "home folder": "files",
    "videos": "videos",
    "video": "videos",
    "player": "videos",
    "movie": "videos",
    "movies": "videos",
    "video player": "videos",
    "media player": "videos",
    "media": "videos",
    "totem": "videos",
    "vlc": "vlc",
    "editor": "editor",
    "text editor": "editor",
    "notes editor": "editor",
    "gedit": "editor",
    "notepad": "editor",
    "calculator": "calculator",
    "calc": "calculator",
    "pdf": "pdf",
    "reader": "pdf",
    "evince": "pdf",
    "image viewer": "images",
    "pictures": "images",
    "photos": "images",
    "images": "images",
    "eog": "images",
    "paint": "paint",
}

# The security whitelist: nothing outside this table can ever be launched.
APP_WHITELIST = SPOKEN_APP_MAP

FRIENDLY_APP_NAMES = {
    "browser": "the browser",
    "firefox": "Firefox",
    "chrome": "Chrome",
    "chromium": "Chromium",
    "edge": "Edge",
    "terminal": "the terminal",
    "files": "the file manager",
    "videos": "the video player",
    "vlc": "VLC",
    "editor": "the text editor",
    "calculator": "the calculator",
    "pdf": "the document reader",
    "images": "the image viewer",
    "paint": "Paint",
}

# Executables tried in order. "uri:" entries are opened through the desktop shell.
APP_LAUNCH = {
    "linux": {
        "browser": ["firefox", "google-chrome", "chromium-browser", "chromium"],
        "firefox": ["firefox"],
        "chrome": ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
        "chromium": ["chromium-browser", "chromium"],
        "edge": ["microsoft-edge", "microsoft-edge-stable"],
        "terminal": ["gnome-terminal", "x-terminal-emulator", "xterm", "konsole"],
        "files": ["nautilus", "nemo", "thunar", "pcmanfm", "dolphin"],
        "videos": ["totem", "vlc", "mpv", "celluloid"],
        "vlc": ["vlc", "totem", "mpv"],
        "editor": ["gedit", "gnome-text-editor", "mousepad", "leafpad", "xed", "kate"],
        "calculator": ["gnome-calculator", "kcalc", "galculator"],
        "pdf": ["evince", "okular", "atril"],
        "images": ["eog", "gpicview", "gwenview", "eom"],
        "paint": ["pinta", "kolourpaint"],
    },
    "windows": {
        "browser": ["msedge.exe", "chrome.exe", "firefox.exe"],
        "firefox": ["firefox.exe"],
        "chrome": ["chrome.exe"],
        "chromium": ["chrome.exe", "msedge.exe"],
        "edge": ["msedge.exe"],
        "terminal": ["wt.exe", "powershell.exe", "cmd.exe"],
        "files": ["explorer.exe"],
        "videos": ["vlc.exe", "wmplayer.exe", "uri:mswindowsvideo:"],
        "vlc": ["vlc.exe"],
        "editor": ["notepad.exe"],
        "calculator": ["calc.exe"],
        "pdf": ["AcroRd32.exe", "Acrobat.exe", "msedge.exe"],
        "images": ["uri:ms-photos:", "mspaint.exe"],
        "paint": ["mspaint.exe"],
    },
}

# Apps that should open at the workspace folder.
APPS_OPENED_AT_WORKSPACE = frozenset({"files"})

# Process names that "close <app>" may stop. The first name found running wins.
APP_PROCESS_NAMES = {
    "linux": {
        "browser": ["firefox", "firefox-bin", "chrome", "chromium"],
        "firefox": ["firefox", "firefox-bin"],
        "chrome": ["chrome", "google-chrome"],
        "chromium": ["chromium", "chromium-browser"],
        "edge": ["msedge", "microsoft-edge"],
        "terminal": ["gnome-terminal-server", "xterm", "konsole"],
        "files": ["nautilus", "nemo", "thunar", "pcmanfm", "dolphin"],
        "videos": ["totem", "vlc", "mpv", "celluloid"],
        "vlc": ["vlc"],
        "editor": ["gedit", "gnome-text-editor", "mousepad", "leafpad", "xed", "kate"],
        "calculator": ["gnome-calculator", "kcalc", "galculator"],
        "pdf": ["evince", "okular", "atril"],
        "images": ["eog", "gpicview", "gwenview", "eom"],
        "paint": ["pinta", "kolourpaint"],
    },
    "windows": {
        "browser": ["msedge.exe", "chrome.exe", "firefox.exe"],
        "firefox": ["firefox.exe"],
        "chrome": ["chrome.exe"],
        "chromium": ["chrome.exe"],
        "edge": ["msedge.exe"],
        "terminal": ["WindowsTerminal.exe"],
        "files": [],
        "videos": ["vlc.exe", "wmplayer.exe", "Microsoft.Media.Player.exe"],
        "vlc": ["vlc.exe"],
        "editor": ["notepad.exe"],
        "calculator": ["CalculatorApp.exe", "calc.exe"],
        "pdf": ["AcroRd32.exe", "Acrobat.exe"],
        "images": ["Photos.exe", "Microsoft.Photos.exe"],
        "paint": ["mspaint.exe"],
    },
}

# Never stopped by voice, on any platform (compared case-insensitively).
PROTECTED_PROCESS_NAMES = frozenset(
    name.lower()
    for name in (
        "systemd", "init", "kthreadd", "Xorg", "X", "gnome-shell", "gdm", "gdm3",
        "lightdm", "sddm", "dbus-daemon", "pulseaudio", "pipewire", "sshd", "kernel",
        "kernal_sounder", "python", "python3", "python.exe", "pythonw.exe",
        "explorer.exe", "csrss.exe", "winlogon.exe", "services.exe", "lsass.exe",
        "svchost.exe", "dwm.exe", "smss.exe", "wininit.exe", "system", "registry",
        "cmd.exe", "powershell.exe", "pwsh.exe", "conhost.exe", "taskmgr.exe",
    )
)

"""
KernalSounder — runtime configuration.

Every choice here is driven by the 4 GB RAM / Core i3 constraint:
keep models tiny, bound subprocesses, and confine file work to a sandbox.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "KernalSounder"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vosk-model-small-en-us-0.15"
LOG_FILE = BASE_DIR / "kernal_sounder.log"

# File operations never leave this directory (path-confinement sandbox).
WORKSPACE = Path.home() / "KernalSounderWorkspace"

# --- Speech (Vosk small English model ~40 MB on disk, ~180–300 MB RAM) ---
SAMPLE_RATE = 16000
STT_BLOCKSIZE = 4000
STT_PHRASE_PAUSE_SEC = 1.15

# --- Voice (espeak via pyttsx3 — no neural TTS weights in RAM) ---
TTS_RATE = 160
TTS_VOLUME = 0.9

# --- Resource guards ---
SUBPROCESS_TIMEOUT_SEC = 12
PROCESS_LIST_LIMIT = 18
CONFIRM_TIMEOUT_SEC = 30
MAX_NAME_LENGTH = 80
MONITOR_INTERVAL_SEC = 1.5
GUI_QUEUE_POLL_MS = 80
MAX_LOG_WIDGET_LINES = 200

# Applications the voice layer is allowed to launch or stop.
APP_WHITELIST = {
    "firefox": "firefox",
    "browser": "firefox",
    "chrome": "google-chrome",
    "google chrome": "google-chrome",
    "chromium": "chromium-browser",
    "terminal": "gnome-terminal",
    "console": "gnome-terminal",
    "gnome terminal": "gnome-terminal",
    "xterm": "xterm",
    "files": "nautilus",
    "file manager": "nautilus",
    "nautilus": "nautilus",
    "editor": "gedit",
    "text editor": "gedit",
    "gedit": "gedit",
    "calculator": "gnome-calculator",
    "calc": "gnome-calculator",
    "evince": "evince",
    "reader": "evince",
    "image viewer": "eog",
    "eog": "eog",
    "vlc": "vlc",
}

CLOSE_NAME_ALIASES = {
    "gnome-terminal": ["gnome-terminal-server", "gnome-terminal"],
    "firefox": ["firefox-bin", "firefox"],
    "google-chrome": ["chrome", "google-chrome"],
    "chromium-browser": ["chromium", "chromium-browser"],
    "nautilus": ["nautilus"],
    "gedit": ["gedit"],
    "vlc": ["vlc"],
}

APP_FALLBACKS = {
    "gnome-terminal": ["gnome-terminal", "x-terminal-emulator", "xterm", "konsole"],
    "nautilus": ["nautilus", "nemo", "thunar", "pcmanfm", "dolphin"],
    "gedit": ["gedit", "mousepad", "leafpad", "xed", "kate"],
    "google-chrome": ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
    "chromium-browser": ["chromium-browser", "chromium"],
    "gnome-calculator": ["gnome-calculator", "kcalc", "galculator"],
}

PROTECTED_PROCESS_NAMES = frozenset(
    {
        "systemd",
        "init",
        "kthreadd",
        "Xorg",
        "X",
        "gnome-shell",
        "gdm",
        "gdm3",
        "lightdm",
        "sddm",
        "dbus-daemon",
        "pulseaudio",
        "pipewire",
        "sshd",
        "kernel",
        "kernal_sounder",
        "python",
        "python3",
    }
)

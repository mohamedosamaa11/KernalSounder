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
LOG_FILE = BASE_DIR / "kernal_sounder.log"

# File operations never leave this directory (path-confinement sandbox).
WORKSPACE = Path.home() / "KernalSounderWorkspace"

# --- Speech (Google Web Speech via SpeechRecognition — no local model RAM) ---
STT_AMBIENT_DURATION_SEC = 1
STT_PHRASE_LIMIT_SEC = 7
STT_NETWORK_ERROR = "Network error. Speech recognition requires internet."

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

# Generic spoken words -> Ubuntu binaries. NLP matches the longest key.
SPOKEN_APP_MAP = {
    "firefox": "firefox",
    "browser": "firefox",
    "internet": "firefox",
    "web": "firefox",
    "web browser": "firefox",
    "chrome": "google-chrome",
    "google chrome": "google-chrome",
    "chromium": "chromium-browser",
    "terminal": "gnome-terminal",
    "console": "gnome-terminal",
    "command line": "gnome-terminal",
    "commandline": "gnome-terminal",
    "shell": "gnome-terminal",
    "gnome terminal": "gnome-terminal",
    "xterm": "xterm",
    "files": "nautilus",
    "folder": "nautilus",
    "folders": "nautilus",
    "file manager": "nautilus",
    "explorer": "nautilus",
    "nautilus": "nautilus",
    "home folder": "nautilus",
    "videos": "totem",
    "video": "totem",
    "player": "totem",
    "movie": "totem",
    "movies": "totem",
    "video player": "totem",
    "media player": "totem",
    "media": "totem",
    "totem": "totem",
    "vlc": "vlc",
    "editor": "gedit",
    "text editor": "gedit",
    "gedit": "gedit",
    "notes editor": "gedit",
    "calculator": "gnome-calculator",
    "calc": "gnome-calculator",
    "evince": "evince",
    "reader": "evince",
    "pdf": "evince",
    "image viewer": "eog",
    "pictures": "eog",
    "photos": "eog",
    "images": "eog",
    "eog": "eog",
}

# Same mapping used by the security whitelist (never launch outside this table).
APP_WHITELIST = SPOKEN_APP_MAP

FRIENDLY_APP_NAMES = {
    "firefox": "the browser",
    "google-chrome": "Chrome",
    "chromium-browser": "Chromium",
    "gnome-terminal": "the terminal",
    "xterm": "the terminal",
    "nautilus": "the file manager",
    "totem": "the video player",
    "vlc": "VLC",
    "gedit": "the text editor",
    "gnome-calculator": "the calculator",
    "evince": "the document reader",
    "eog": "the image viewer",
}

CLOSE_NAME_ALIASES = {
    "gnome-terminal": ["gnome-terminal-server", "gnome-terminal"],
    "firefox": ["firefox-bin", "firefox"],
    "google-chrome": ["chrome", "google-chrome"],
    "chromium-browser": ["chromium", "chromium-browser"],
    "nautilus": ["nautilus"],
    "gedit": ["gedit"],
    "vlc": ["vlc", "totem"],
    "totem": ["totem", "vlc"],
    "eog": ["eog"],
}

APP_FALLBACKS = {
    "gnome-terminal": ["gnome-terminal", "x-terminal-emulator", "xterm", "konsole"],
    "nautilus": ["nautilus", "nemo", "thunar", "pcmanfm", "dolphin"],
    "gedit": ["gedit", "mousepad", "leafpad", "xed", "kate"],
    "google-chrome": ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
    "chromium-browser": ["chromium-browser", "chromium"],
    "gnome-calculator": ["gnome-calculator", "kcalc", "galculator"],
    "totem": ["totem", "vlc", "mpv", "celluloid"],
    "vlc": ["vlc", "totem", "mpv"],
    "eog": ["eog", "gpicview", "gwenview", "eom"],
    "evince": ["evince", "okular", "atril"],
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

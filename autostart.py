"""
Start KernalSounder automatically at login, so the wake word always works.

    python kernal_sounder.py --install-autostart
    python kernal_sounder.py --remove-autostart

Windows: a shortcut in the user's Startup folder that runs pythonw (no console).
Linux:   ~/.config/autostart/kernalsounder.desktop that runs run.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import APP_NAME, BASE_DIR, IS_WINDOWS


def _windows_startup_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _pythonw() -> str:
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return str(candidate if candidate.exists() else exe)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def install() -> str:
    if IS_WINDOWS:
        link = _windows_startup_dir() / f"{APP_NAME}.lnk"
        script = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps_quote(str(link)) + ");"
            "$s.TargetPath = " + _ps_quote(_pythonw()) + ";"
            "$s.Arguments = " + _ps_quote(f'"{BASE_DIR / "kernal_sounder.py"}"') + ";"
            "$s.WorkingDirectory = " + _ps_quote(str(BASE_DIR)) + ";"
            "$s.Description = " + _ps_quote(f"{APP_NAME} voice assistant") + ";"
            "$s.Save()"
        )
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True)
        return str(link)

    target = Path.home() / ".config" / "autostart" / "kernalsounder.desktop"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=Voice assistant with wake word\n"
        f"Exec=\"{BASE_DIR / 'run.sh'}\"\n"
        f"Path={BASE_DIR}\n"
        "X-GNOME-Autostart-enabled=true\n"
        "X-GNOME-Autostart-Delay=5\n",
        encoding="utf-8",
    )
    return str(target)


def remove() -> str:
    target = (_windows_startup_dir() / f"{APP_NAME}.lnk") if IS_WINDOWS else (
        Path.home() / ".config" / "autostart" / "kernalsounder.desktop"
    )
    target.unlink(missing_ok=True)
    return str(target)

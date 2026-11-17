"""
KernalSounder — catalog of the apps installed on this computer.

Windows: every Start-menu app (classic and Store apps) from Get-StartApps,
         launched through the shell as the normal user:
         explorer.exe shell:AppsFolder\\<AppID>
Linux:   every visible .desktop application (system, user, flatpak, snap),
         launched from its Exec line as an argument list (no shell).

Spoken names are matched fuzzily ("spot fire" -> Spotify). Administrative
and destructive entries (Registry Editor, Disk Management, uninstallers,
"reset" shortcuts ...) are in the catalog only so they can be refused with a
clear message; they are never launched.
"""

from __future__ import annotations

import configparser
import difflib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from config import IS_WINDOWS

DENY_PATTERNS = re.compile(
    r"\b(uninstall\w*|reset|repair|remove|clean ?up|cleaner|delete|wipe|format|recovery|"
    r"registry|regedit|diskpart|disk management|partition|computer management|group policy|"
    r"gpedit|services|task scheduler|system configuration|msconfig|local security|"
    r"security policy|bitlocker|administrative|windows tools|odbc|iscsi|component services|"
    r"defragment|optimi[sz]e drives|disk cleanup|memory diagnostic|\(admin\)|as administrator|"
    r"firmware|bios|installer|setup)\b",
    re.IGNORECASE,
)

_GENERIC_WORDS = {"microsoft", "app", "application", "the", "desktop", "for", "windows"}


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9+ ]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _key(text: str) -> str:
    words = [w for w in _norm(text).split() if w not in _GENERIC_WORDS]
    return " ".join(words) or _norm(text)


def _compact(text: str) -> str:
    return _key(text).replace(" ", "").replace("ph", "f")


def _skeleton(compact: str) -> str:
    """Consonant skeleton: tolerant to the vowel mistakes speech recognizers make."""
    if not compact:
        return ""
    rest = re.sub(r"[aeiouy]", "", compact[1:])
    return compact[0] + re.sub(r"(.)+", r"", rest)


ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "regedit": "registry editor",
    "control panel": "control panel",
    "task manager": "task manager",
    "word": "word",
    "excel": "excel",
    "powerpoint": "powerpoint",
    "whatsapp": "whatsapp",
}


@dataclass
class AppEntry:
    name: str
    launch: List[str]
    process_names: List[str] = field(default_factory=list)
    denied: bool = False

    @property
    def key(self) -> str:
        return _key(self.name)


def _windows_apps() -> List[AppEntry]:
    script = "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=30, creationflags=flags,
    ).stdout
    rows = json.loads(out) if out.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    apps = []
    for row in rows:
        name, app_id = (row.get("Name") or "").strip(), (row.get("AppID") or "").strip()
        if not name or not app_id or app_id.lower().startswith(("http:", "https:")):
            continue
        procs = []
        exe = re.search(r"([^\\/]+\.exe)$", app_id, re.IGNORECASE)
        if exe:
            procs.append(exe.group(1))
        apps.append(AppEntry(name=name, launch=["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                             process_names=procs))
    return apps


_DESKTOP_DIRS = (
    "/usr/share/applications",
    "/usr/local/share/applications",
    "~/.local/share/applications",
    "/var/lib/flatpak/exports/share/applications",
    "~/.local/share/flatpak/exports/share/applications",
    "/var/lib/snapd/desktop/applications",
)


def _linux_apps() -> List[AppEntry]:
    apps = []
    seen = set()
    for folder in _DESKTOP_DIRS:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.desktop")):
            parser = configparser.RawConfigParser(strict=False, interpolation=None)
            try:
                parser.read(path, encoding="utf-8")
                entry = parser["Desktop Entry"]
            except Exception:
                continue
            if entry.get("Type", "Application") != "Application":
                continue
            if entry.get("NoDisplay", "false").lower() == "true" or entry.get("Hidden", "false").lower() == "true":
                continue
            if entry.get("Terminal", "false").lower() == "true":
                continue
            name, exec_line = entry.get("Name", "").strip(), entry.get("Exec", "").strip()
            if not name or not exec_line or name.lower() in seen:
                continue
            exec_line = re.sub(r"%[fFuUdDnNickvm]", "", exec_line).replace("%%", "%")
            try:
                argv = shlex.split(exec_line)
            except ValueError:
                continue
            if not argv:
                continue
            seen.add(name.lower())
            base = os.path.basename(argv[0])
            procs = [base] if base not in {"env", "flatpak", "snap", "sh", "bash"} else []
            if base == "flatpak" and len(argv) > 2:
                procs.append(argv[-1].split(".")[-1].lower())
            apps.append(AppEntry(name=name, launch=argv, process_names=procs))
    return apps


class AppCatalog:
    def __init__(self) -> None:
        self._apps: List[AppEntry] = []
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        self._loaded_at = 0.0

    def refresh(self) -> int:
        try:
            apps = _windows_apps() if IS_WINDOWS else _linux_apps()
        except Exception as exc:
            logging.warning("App catalog refresh failed: %s", exc)
            apps = []
        for app in apps:
            app.denied = bool(DENY_PATTERNS.search(app.name))
        with self._lock:
            if apps or not self._apps:
                self._apps = apps
            self._loaded_at = time.monotonic()
        self._loaded.set()
        return len(apps)

    def wait(self, timeout: float = 5.0) -> bool:
        return self._loaded.wait(timeout)

    def allowed_names(self) -> List[str]:
        with self._lock:
            return [a.name for a in self._apps if not a.denied]

    def find(self, spoken: str, min_score: float = 0.72) -> Optional[Tuple[AppEntry, float]]:
        best = self._best(spoken)
        if (best is None or best[1] < min_score) and time.monotonic() - self._loaded_at > 60:
            self.refresh()
            best = self._best(spoken)
        if best is None or best[1] < min_score:
            return None
        return best

    def _best(self, spoken: str) -> Optional[Tuple[AppEntry, float]]:
        spoken = ALIASES.get(_norm(spoken), spoken)
        query, compact = _key(spoken), _compact(spoken)
        if not query:
            return None
        skel = _skeleton(compact)
        q_words = query.split()
        best: Optional[Tuple[AppEntry, float]] = None
        with self._lock:
            apps = list(self._apps)
        for app in apps:
            key, ckey = app.key, _compact(app.name)
            words = key.split()
            if query == key or compact == ckey:
                score = 1.0
            elif all(w in words for w in q_words):
                score = 0.9 - 0.02 * (len(words) - len(q_words))
            elif key.startswith(query) or ckey.startswith(compact):
                score = 0.86
            else:
                score = difflib.SequenceMatcher(None, compact, ckey).ratio() * 0.95
                if len(skel) >= 3:
                    score = max(score, difflib.SequenceMatcher(None, skel, _skeleton(ckey)).ratio() * 0.92)
                for word in words:
                    if len(word) >= 3:
                        score = max(score, difflib.SequenceMatcher(None, compact, word).ratio() * 0.9)
                        if len(skel) >= 3 and len(word) >= 4:
                            score = max(score, difflib.SequenceMatcher(None, skel, _skeleton(word)).ratio() * 0.88)
            if app.denied:
                score += 0.001
            if best is None or score > best[1] + 1e-9 or (
                abs(score - best[1]) < 1e-9 and len(app.name) < len(best[0].name)
            ):
                best = (app, score)
        return best


CATALOG = AppCatalog()

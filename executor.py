"""
KernalSounder — command executor.

Acts ONLY on results produced by security.validate(). There is no shell at
all: file work uses Python's file APIs, system info comes from psutil, and
apps are started from a fixed whitelist with an argument list. The same code
runs on Linux and Windows.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil

from config import (
    APP_LAUNCH,
    APPS_OPENED_AT_WORKSPACE,
    FRIENDLY_APP_NAMES,
    IS_WINDOWS,
    PLATFORM_KEY,
    PROCESS_LIST_LIMIT,
    PROTECTED_PROCESS_NAMES,
    WORKSPACE,
)
from security import (
    Intent,
    ParsedCommand,
    ValidationResult,
    ValidationState,
    closable_process_names,
    ensure_workspace,
)


@dataclass
class ExecutionResult:
    ok: bool
    spoken: str
    details: str
    display_command: str


def _friendly(app_id: Optional[str]) -> str:
    return FRIENDLY_APP_NAMES.get(app_id or "", app_id or "that app")


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _list_workspace(root: Path) -> List[str]:
    rows = []
    entries = sorted(os.scandir(root), key=lambda e: (not e.is_dir(), e.name.lower()))
    for entry in entries:
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        kind = "dir " if stat.S_ISDIR(info.st_mode) else "file"
        stamp = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
        size = "-" if kind == "dir " else _human_size(info.st_size)
        rows.append(f"{kind}  {size:>9}  {stamp}  {entry.name}")
    return rows


def _windows_app_path(exe: str) -> Optional[str]:
    try:
        import winreg
    except ImportError:
        return None
    sub = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, sub) as key:
                value, _ = winreg.QueryValueEx(key, None)
                value = os.path.expandvars(str(value).strip('"'))
                if value and Path(value).exists():
                    return value
        except OSError:
            continue
    return None


def _resolve_executable(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    if IS_WINDOWS:
        return _windows_app_path(name)
    return None


def _launch(app_id: str) -> Optional[str]:
    """Start the first available candidate. Returns what was started, or None."""
    candidates = APP_LAUNCH.get(PLATFORM_KEY, {}).get(app_id, [])
    extra_args: List[str] = [str(WORKSPACE)] if app_id in APPS_OPENED_AT_WORKSPACE else []
    flags = 0
    if IS_WINDOWS:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
    for candidate in candidates:
        if candidate.startswith("uri:"):
            shell = _resolve_executable("explorer.exe" if IS_WINDOWS else "xdg-open")
            if not shell:
                continue
            argv = [shell, candidate[4:]]
        else:
            exe = _resolve_executable(candidate)
            if not exe:
                continue
            argv = [exe, *extra_args]
        console_app = IS_WINDOWS and Path(argv[0]).name.lower() in {"powershell.exe", "cmd.exe"}
        kwargs = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(WORKSPACE),
            close_fds=True,
        )
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE if console_app else flags
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(argv, **kwargs)
        except OSError:
            if IS_WINDOWS and not console_app:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                try:
                    subprocess.Popen(argv, **kwargs)
                except OSError:
                    continue
            else:
                continue
        return Path(argv[0]).name
    return None


def _protected_pids() -> set:
    pids = {os.getpid()}
    try:
        for parent in psutil.Process().parents():
            pids.add(parent.pid)
    except psutil.Error:
        pass
    return pids


def _own_processes():
    try:
        me = psutil.Process().username()
    except psutil.Error:
        me = None
    skip = _protected_pids()
    for proc in psutil.process_iter(["name", "username"]):
        try:
            pname = (proc.info.get("name") or "").lower()
            if not pname or proc.pid in skip or pname in PROTECTED_PROCESS_NAMES:
                continue
            if me and proc.info.get("username") and proc.info["username"] != me:
                continue
            yield proc, pname
        except psutil.Error:
            continue


def _running_name_like(spoken: str) -> Optional[str]:
    """Best running process name for a spoken app name ("spotify" -> spotify.exe)."""
    want = re.sub(r"[^a-z0-9]", "", (spoken or "").lower())
    if len(want) < 3:
        return None
    best, best_score = None, 0.0
    for _proc, pname in _own_processes():
        stem = re.sub(r"[^a-z0-9]", "", re.sub(r"\.exe$", "", pname))
        score = 1.0 if stem == want else difflib.SequenceMatcher(None, want, stem).ratio()
        if score > best_score:
            best, best_score = pname, score
    return best if best_score >= 0.85 else None


def _terminate_names(names: List[str]) -> Optional[str]:
    """Stop the user's own processes for the first name found running."""
    for name in names:
        target = name.lower()
        if target in PROTECTED_PROCESS_NAMES:
            continue
        victims = [proc for proc, pname in _own_processes() if pname == target]
        if not victims:
            continue
        for proc in victims:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(victims, timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        return name
    return None


def _terminate(validation: ValidationResult) -> Optional[str]:
    names: List[str] = []
    if validation.app_id:
        names += closable_process_names(validation.app_id)
    entry = validation.catalog_app
    if entry is not None:
        names += list(getattr(entry, "process_names", []))
    stopped = _terminate_names(names) if names else None
    if stopped:
        return stopped
    query = getattr(entry, "name", None) or validation.close_query or validation.app_id or ""
    guess = _running_name_like(query)
    return _terminate_names([guess]) if guess else None


def _launch_installed(entry) -> Optional[str]:
    argv = list(entry.launch)
    exe = _resolve_executable(argv[0])
    if not exe:
        return None
    argv[0] = exe
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  cwd=str(WORKSPACE), close_fds=True)
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(argv, **kwargs)
    return entry.name


def _usage_snapshot() -> str:
    cpu = psutil.cpu_percent(interval=0.15)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home().anchor or "/"))
    gb = 1024 ** 3
    return (
        f"CPU {cpu:.0f} percent. "
        f"RAM {mem.used / gb:.1f} of {mem.total / gb:.1f} gigabytes, {mem.percent:.0f} percent. "
        f"Disk {disk.used / gb:.1f} of {disk.total / gb:.1f} gigabytes, {disk.percent:.0f} percent."
    )


def _process_table(limit: int = PROCESS_LIST_LIMIT) -> str:
    procs = []
    for proc in psutil.process_iter(["pid", "name", "username", "memory_percent"]):
        try:
            proc.cpu_percent(None)
            procs.append(proc)
        except psutil.Error:
            continue
    time.sleep(0.4)
    cores = psutil.cpu_count() or 1
    rows = []
    for proc in procs:
        try:
            cpu = proc.cpu_percent(None) / cores
            user = (proc.info.get("username") or "?").split("\\")[-1]
            rows.append((cpu, proc.pid, user[:12], (proc.info.get("name") or "?")[:28],
                         proc.info.get("memory_percent") or 0.0))
        except psutil.Error:
            continue
    rows.sort(key=lambda r: (r[0], r[4]), reverse=True)
    lines = [f"{'PID':>7}  {'USER':<12}  {'NAME':<28}  {'%CPU':>5}  {'%MEM':>5}"]
    for cpu, pid, user, name, mem in rows[:limit]:
        lines.append(f"{pid:>7}  {user:<12}  {name:<28}  {cpu:>5.1f}  {mem:>5.1f}")
    return "\n".join(lines)


def execute(command: ParsedCommand, validation: ValidationResult) -> ExecutionResult:
    display = validation.action
    if validation.state != ValidationState.SAFE:
        return ExecutionResult(False, validation.reason, validation.reason, display)

    intent = command.intent
    path = validation.resolved_path

    try:
        if intent in (Intent.HELP, Intent.LIST_APPS):
            from nlp_engine import APPS_HELP_TEXT, HELP_TEXT

            spoken = APPS_HELP_TEXT if intent == Intent.LIST_APPS else HELP_TEXT
            return ExecutionResult(True, spoken, spoken, display)

        if intent == Intent.SHOW_USAGE:
            snapshot = _usage_snapshot()
            return ExecutionResult(True, snapshot, snapshot, display)

        if intent == Intent.SHOW_PROCESSES:
            table = _process_table()
            spoken = f"Showing the top {PROCESS_LIST_LIMIT} processes by CPU."
            return ExecutionResult(True, spoken, table, display)

        if intent == Intent.LIST_FILES:
            rows = _list_workspace(ensure_workspace())
            if not rows:
                return ExecutionResult(True, "The workspace is empty.", "(empty)", display)
            noun = "item" if len(rows) == 1 else "items"
            spoken = f"The workspace has {len(rows)} {noun}."
            return ExecutionResult(True, spoken, "\n".join(rows), display)

        if intent == Intent.CREATE_FILE and path is not None:
            with open(path, "x", encoding="utf-8"):
                pass
            return ExecutionResult(True, f"Created file {path.name}.", str(path), display)

        if intent == Intent.CREATE_FOLDER and path is not None:
            path.mkdir()
            return ExecutionResult(True, f"Created folder {path.name}.", str(path), display)

        if intent == Intent.RENAME_FILE and path is not None and validation.extra_path is not None:
            dst = validation.extra_path
            if dst.exists():
                msg = f"{dst.name} already exists, so nothing was renamed."
                return ExecutionResult(False, msg, msg, display)
            os.rename(path, dst)
            spoken = f"Renamed {path.name} to {dst.name}."
            return ExecutionResult(True, spoken, spoken, display)

        if intent == Intent.DELETE_FILE and path is not None:
            if path.is_symlink():
                msg = "Symbolic links are not deleted."
                return ExecutionResult(False, msg, msg, display)
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    msg = f"{path.name} is not empty. Recursive delete is disabled."
                    return ExecutionResult(False, f"Could not delete {path.name}.", msg, display)
            else:
                path.unlink()
            return ExecutionResult(True, f"Deleted {path.name}.", str(path), display)

        if intent == Intent.OPEN_APP:
            entry = validation.catalog_app
            if entry is not None:
                started = _launch_installed(entry)
                label = entry.name
            else:
                started = _launch(validation.app_id or "")
                label = _friendly(validation.app_id)
            if not started:
                msg = f"I could not start {label} on this computer."
                return ExecutionResult(False, msg, msg, display)
            spoken = f"Opening {label}."
            return ExecutionResult(True, spoken, f"{spoken} ({started})", display)

        if intent == Intent.CLOSE_APP:
            entry = validation.catalog_app
            if entry is not None:
                label = entry.name
            elif validation.app_id:
                label = _friendly(validation.app_id)
            else:
                label = validation.close_query or "that app"
            stopped = _terminate(validation)
            if not stopped:
                msg = f"{label} is not running."
                return ExecutionResult(False, msg, msg, display)
            spoken = f"Closed {label}."
            return ExecutionResult(True, spoken, f"{spoken} ({stopped})", display)

        msg = "No executable action was produced."
        return ExecutionResult(False, msg, msg, display)

    except FileExistsError:
        msg = "That name already exists."
        return ExecutionResult(False, msg, msg, display)
    except FileNotFoundError:
        msg = "That item no longer exists."
        return ExecutionResult(False, msg, msg, display)
    except PermissionError:
        msg = "Permission denied by the operating system."
        return ExecutionResult(False, msg, msg, display)
    except Exception as exc:
        return ExecutionResult(
            False, "Something went wrong while running that action.", f"Execution error: {exc}", display
        )

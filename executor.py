"""
KernalSounder — command executor.

Runs ONLY argv lists produced by security.validate(). shell=True is never used.
stdout/stderr are captured with a hard timeout so a hung tool cannot freeze
the 4 GB VM.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import psutil

from config import (
    APP_FALLBACKS,
    CLOSE_NAME_ALIASES,
    PROCESS_LIST_LIMIT,
    SUBPROCESS_TIMEOUT_SEC,
    WORKSPACE,
)
from security import Intent, ParsedCommand, ValidationResult, ValidationState, ensure_workspace


@dataclass
class ExecutionResult:
    ok: bool
    spoken: str
    details: str
    display_command: str


def _run(argv: Sequence[str], *, timeout: int = SUBPROCESS_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(ensure_workspace()),
    )


def _trim(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]"


def _resolve_launch_binary(preferred: str) -> Optional[str]:
    candidates: List[str] = [preferred]
    candidates.extend(APP_FALLBACKS.get(preferred, []))
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = shutil.which(name)
        if path:
            return path
    return None


def _usage_snapshot() -> str:
    cpu = psutil.cpu_percent(interval=0.15)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    ram_used = mem.used / (1024 ** 3)
    ram_total = mem.total / (1024 ** 3)
    disk_used = disk.used / (1024 ** 3)
    disk_total = disk.total / (1024 ** 3)
    return (
        f"CPU {cpu:.0f} percent. "
        f"RAM {ram_used:.1f} of {ram_total:.1f} gigabytes, {mem.percent:.0f} percent. "
        f"Disk {disk_used:.1f} of {disk_total:.1f} gigabytes, {disk.percent:.0f} percent."
    )


def _process_table(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "No process data."
    header, body = lines[0], lines[1:]
    clipped = body[:PROCESS_LIST_LIMIT]
    return "\n".join([header, *clipped])


def execute(command: ParsedCommand, validation: ValidationResult) -> ExecutionResult:
    if validation.state not in (ValidationState.SAFE,):
        return ExecutionResult(
            ok=False,
            spoken=validation.reason,
            details=validation.reason,
            display_command=validation.display_command or "",
        )

    intent = command.intent
    display = validation.display_command

    try:
        if intent == Intent.HELP:
            from nlp_engine import HELP_TEXT

            return ExecutionResult(True, HELP_TEXT, HELP_TEXT, display)

        if intent == Intent.SHOW_USAGE:
            snapshot = _usage_snapshot()
            extras: List[str] = [snapshot]
            try:
                free = _run(["free", "-h"])
                extras.append(_trim(free.stdout, 400))
            except Exception:
                pass
            try:
                df = _run(["df", "-h", "/"])
                extras.append(_trim(df.stdout, 400))
            except Exception:
                pass
            details = "\n\n".join(part for part in extras if part)
            return ExecutionResult(True, snapshot, details, display)

        if intent == Intent.OPEN_APP:
            binary = _resolve_launch_binary(validation.app_binary or "")
            if not binary:
                msg = f"I could not find {validation.app_binary} on this system."
                return ExecutionResult(False, msg, msg, display)
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(WORKSPACE),
            )
            spoken = f"Opening {validation.app_binary}."
            return ExecutionResult(True, spoken, spoken, display)

        if intent == Intent.CLOSE_APP:
            app = validation.app_binary or "application"
            base = Path(app).name
            names = CLOSE_NAME_ALIASES.get(base, [base])
            if base not in names:
                names = [base, *names]
            last_err = ""
            for name in names:
                completed = _run(["pkill", "-x", name])
                if completed.returncode == 0:
                    spoken = f"Closed {app}."
                    return ExecutionResult(True, spoken, spoken, display)
                last_err = _trim(completed.stderr)
            spoken = f"No running window named {app} was found."
            return ExecutionResult(False, spoken, last_err or spoken, display)

        if validation.argv is None:
            msg = "No validated command list was produced."
            return ExecutionResult(False, msg, msg, display)

        completed = _run(validation.argv)

        if intent == Intent.LIST_FILES:
            listing = _trim(completed.stdout or "Workspace is empty.")
            spoken = f"Listed files in the workspace. {len(listing.splitlines())} lines."
            return ExecutionResult(completed.returncode == 0, spoken, listing, display)

        if intent == Intent.CREATE_FILE:
            name = validation.resolved_path.name if validation.resolved_path else "file"
            if completed.returncode == 0:
                spoken = f"Created file {name}."
                return ExecutionResult(True, spoken, spoken, display)
            err = _trim(completed.stderr or "touch failed.")
            return ExecutionResult(False, f"Could not create file {name}.", err, display)

        if intent == Intent.CREATE_FOLDER:
            name = validation.resolved_path.name if validation.resolved_path else "folder"
            if completed.returncode == 0:
                spoken = f"Created folder {name}."
                return ExecutionResult(True, spoken, spoken, display)
            err = _trim(completed.stderr or "mkdir failed.")
            return ExecutionResult(False, f"Could not create folder {name}.", err, display)

        if intent == Intent.RENAME_FILE:
            src = validation.resolved_path.name if validation.resolved_path else "file"
            dst = validation.extra_path.name if validation.extra_path else "file"
            if completed.returncode == 0:
                spoken = f"Renamed {src} to {dst}."
                return ExecutionResult(True, spoken, spoken, display)
            err = _trim(completed.stderr or "mv failed.")
            return ExecutionResult(False, f"Could not rename {src}.", err, display)

        if intent == Intent.DELETE_FILE:
            name = validation.resolved_path.name if validation.resolved_path else "item"
            if completed.returncode == 0:
                spoken = f"Deleted {name}."
                return ExecutionResult(True, spoken, spoken, display)
            err = _trim(completed.stderr or "delete failed.")
            hint = err
            if validation.resolved_path and validation.resolved_path.is_dir():
                hint += "\nFolders must be empty. Recursive delete is disabled."
            return ExecutionResult(False, f"Could not delete {name}.", hint, display)

        if intent == Intent.SHOW_PROCESSES:
            table = _process_table(completed.stdout)
            spoken = f"Showing the top {PROCESS_LIST_LIMIT} processes by CPU."
            details = table if completed.returncode == 0 else _trim(completed.stderr)
            return ExecutionResult(completed.returncode == 0, spoken, details, display)

        err = _trim(completed.stderr or completed.stdout or "Command finished.")
        ok = completed.returncode == 0
        spoken = "Done." if ok else "The command failed."
        return ExecutionResult(ok, spoken, err, display)

    except subprocess.TimeoutExpired:
        msg = "The command timed out and was stopped to protect the VM."
        return ExecutionResult(False, msg, msg, display)
    except FileNotFoundError:
        msg = "A required Linux tool is not installed on this VM."
        return ExecutionResult(False, msg, msg, display)
    except Exception as exc:
        msg = f"Execution error: {exc}"
        return ExecutionResult(False, "Something went wrong while running that action.", msg, display)

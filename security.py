"""
KernalSounder — command validation and sandboxing.

Voice text is NEVER passed to a shell. The NLP layer produces a structured
intent; this module decides whether that intent may be turned into a
subprocess argument list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    APP_WHITELIST,
    MAX_NAME_LENGTH,
    PROTECTED_PROCESS_NAMES,
    WORKSPACE,
)

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
HIDDEN_OR_SPECIAL = re.compile(r"(^\.|\.\.|[\x00-\x1f])")

BLOCKED_NAME_FRAGMENTS = (
    "sudo",
    "passwd",
    "shadow",
    "bashrc",
    "profile",
    "authorized_keys",
    "id_rsa",
    "id_ed25519",
)


class ValidationState(str, Enum):
    SAFE = "SAFE"
    BLOCKED = "BLOCKED"
    NEEDS_CONFIRM = "NEEDS_CONFIRM"
    ERROR = "ERROR"


class Intent(str, Enum):
    LIST_FILES = "LIST_FILES"
    CREATE_FILE = "CREATE_FILE"
    CREATE_FOLDER = "CREATE_FOLDER"
    RENAME_FILE = "RENAME_FILE"
    DELETE_FILE = "DELETE_FILE"
    OPEN_APP = "OPEN_APP"
    CLOSE_APP = "CLOSE_APP"
    SHOW_PROCESSES = "SHOW_PROCESSES"
    SHOW_USAGE = "SHOW_USAGE"
    HELP = "HELP"
    LIST_APPS = "LIST_APPS"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedCommand:
    intent: Intent
    slots: Dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class ValidationResult:
    state: ValidationState
    reason: str
    argv: Optional[List[str]] = None
    display_command: str = ""
    resolved_path: Optional[Path] = None
    extra_path: Optional[Path] = None
    app_binary: Optional[str] = None


class SecurityError(Exception):
    """Raised when a voice-derived action fails policy checks."""


def ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE.resolve()


def _is_inside_workspace(path: Path) -> bool:
    root = WORKSPACE.resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def sanitize_name(name: str, *, allow_extension: bool = True) -> str:
    if name is None:
        raise SecurityError("Missing name.")
    cleaned = name.strip().strip("\"'").replace(" ", "_")
    if not cleaned:
        raise SecurityError("Empty name is not allowed.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise SecurityError(f"Name exceeds {MAX_NAME_LENGTH} characters.")
    if HIDDEN_OR_SPECIAL.search(cleaned):
        raise SecurityError("Hidden, traversal, or control characters are blocked.")
    if "/" in cleaned or "\\" in cleaned:
        raise SecurityError("Path separators are not allowed in names.")
    if not SAFE_NAME_RE.match(cleaned):
        raise SecurityError("Name contains unsupported characters.")
    lower = cleaned.lower()
    for fragment in BLOCKED_NAME_FRAGMENTS:
        if fragment in lower:
            raise SecurityError("That name is blocked for safety.")
    if not allow_extension and "." in cleaned:
        raise SecurityError("Folder names may not contain a dot extension.")
    return cleaned


def resolve_workspace_path(name: str, *, must_exist: Optional[bool] = None) -> Path:
    safe = sanitize_name(name)
    root = ensure_workspace()
    path = (root / safe).resolve()
    if not _is_inside_workspace(path):
        raise SecurityError("Path escapes the KernalSounder workspace.")
    if path.is_symlink():
        raise SecurityError("Symbolic links are not allowed.")
    if must_exist is True and not path.exists():
        raise SecurityError(f"'{safe}' does not exist in the workspace.")
    if must_exist is False and path.exists():
        raise SecurityError(f"'{safe}' already exists.")
    return path


def resolve_app(spoken_app: str) -> str:
    key = spoken_app.strip().lower()
    if key in APP_WHITELIST:
        return APP_WHITELIST[key]
    contained = [name for name in APP_WHITELIST if name in key]
    if contained:
        best = max(contained, key=len)
        return APP_WHITELIST[best]
    raise SecurityError(
            f"I don't know the app '{spoken_app}'. "
            "Try browser, files, terminal, videos, editor, or calculator."
        )


def validate(command: ParsedCommand) -> ValidationResult:
    """
    Map a parsed intent to a frozen argv list.

    Returns BLOCKED / NEEDS_CONFIRM / SAFE. The executor may only run argv
    from a SAFE (or confirmed) result — never command.raw_text.
    """
    intent = command.intent
    slots = command.slots

    try:
        if intent in (Intent.UNKNOWN,):
            return ValidationResult(
                state=ValidationState.ERROR,
                reason=(
                    "Command not recognized. You can ask me to list files, "
                    "open firefox, or show cpu usage."
                ),
            )

        if intent in (Intent.HELP, Intent.LIST_APPS):
            label = "app catalog" if intent == Intent.LIST_APPS else "voice help"
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Help does not run a shell command.",
                display_command=f"({label})",
            )

        if intent == Intent.LIST_FILES:
            root = ensure_workspace()
            argv = ["ls", "-la", "--", str(root)]
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Listing is confined to the workspace.",
                argv=argv,
                display_command="ls -la -- <workspace>",
                resolved_path=root,
            )

        if intent == Intent.CREATE_FILE:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=False)
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Create file is confined and uses touch.",
                argv=["touch", "--", str(path)],
                display_command=f"touch -- {path.name}",
                resolved_path=path,
            )

        if intent == Intent.CREATE_FOLDER:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=False)
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Create folder is confined and uses mkdir.",
                argv=["mkdir", "--", str(path)],
                display_command=f"mkdir -- {path.name}",
                resolved_path=path,
            )

        if intent == Intent.RENAME_FILE:
            src = resolve_workspace_path(slots.get("source", ""), must_exist=True)
            dst = resolve_workspace_path(slots.get("destination", ""), must_exist=False)
            if src == dst:
                raise SecurityError("Source and destination are the same.")
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Rename stays inside the workspace.",
                argv=["mv", "-n", "--", str(src), str(dst)],
                display_command=f"mv -n -- {src.name} {dst.name}",
                resolved_path=src,
                extra_path=dst,
            )

        if intent == Intent.DELETE_FILE:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=True)
            if path.is_dir():
                argv = ["rmdir", "--", str(path)]
                display = f"rmdir -- {path.name}"
                extra = " Empty folders only — recursive delete is forbidden."
            else:
                argv = ["rm", "-f", "--", str(path)]
                display = f"rm -f -- {path.name}"
                extra = ""
            return ValidationResult(
                state=ValidationState.NEEDS_CONFIRM,
                reason="Delete requires explicit confirmation." + extra,
                argv=argv,
                display_command=display,
                resolved_path=path,
            )

        if intent == Intent.OPEN_APP:
            binary = resolve_app(slots.get("app", ""))
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Application is on the launch whitelist.",
                argv=[binary],
                display_command=binary,
                app_binary=binary,
            )

        if intent == Intent.CLOSE_APP:
            binary = resolve_app(slots.get("app", ""))
            base = Path(binary).name
            if base.lower() in {p.lower() for p in PROTECTED_PROCESS_NAMES}:
                raise SecurityError("That process is protected and cannot be closed.")
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Close is limited to a whitelisted, non-critical process.",
                argv=["pkill", "-x", base],
                display_command=f"pkill -x {base}",
                app_binary=binary,
            )

        if intent == Intent.SHOW_PROCESSES:
            argv = [
                "ps",
                "-eo",
                "pid,user,comm,%cpu,%mem",
                "--sort=-%cpu",
            ]
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Process listing uses a fixed ps format.",
                argv=argv,
                display_command="ps -eo pid,user,comm,%cpu,%mem --sort=-%cpu",
            )

        if intent == Intent.SHOW_USAGE:
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Usage is read from psutil plus bounded df/free calls.",
                argv=["free", "-h"],
                display_command="free -h && df -h / && cpu snapshot",
            )

        return ValidationResult(
            state=ValidationState.BLOCKED,
            reason="Intent is not in the executable whitelist.",
        )
    except SecurityError as exc:
        return ValidationResult(state=ValidationState.BLOCKED, reason=str(exc))
    except Exception as exc:
        return ValidationResult(
            state=ValidationState.ERROR,
            reason=f"Validation failed: {exc}",
        )

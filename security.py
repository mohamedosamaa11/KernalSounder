"""
KernalSounder — command validation and sandboxing.

Voice text is NEVER passed to a shell. The NLP layer produces a structured
intent; this module decides whether that intent may become a concrete,
pre-defined action (a Python file API call, a whitelisted app launch, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from config import (
    APP_PROCESS_NAMES,
    APP_WHITELIST,
    MAX_NAME_LENGTH,
    PLATFORM_KEY,
    PROTECTED_PROCESS_NAMES,
    WORKSPACE,
)

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
HIDDEN_OR_SPECIAL = re.compile(r"(^\.|\.\.|[\x00-\x1f])")
WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)

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
    action: str = ""
    resolved_path: Optional[Path] = None
    extra_path: Optional[Path] = None
    app_id: Optional[str] = None


class SecurityError(Exception):
    """Raised when a voice-derived action fails policy checks."""


def ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE.resolve()


def _is_inside_workspace(path: Path) -> bool:
    root = WORKSPACE.resolve()
    try:
        path.resolve().relative_to(root)
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
    if not SAFE_NAME_RE.match(cleaned) or cleaned.endswith("."):
        raise SecurityError("Name contains unsupported characters.")
    if WINDOWS_RESERVED.match(cleaned):
        raise SecurityError("That is a reserved device name.")
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
    candidate = root / safe
    if candidate.is_symlink():
        raise SecurityError("Symbolic links are not allowed.")
    path = candidate.resolve()
    if not _is_inside_workspace(path):
        raise SecurityError("Path escapes the KernalSounder workspace.")
    if must_exist is True and not path.exists():
        raise SecurityError(f"'{safe}' does not exist in the workspace.")
    if must_exist is False and path.exists():
        raise SecurityError(f"'{safe}' already exists.")
    return path


def resolve_app(spoken_app: str) -> str:
    """Map a spoken app phrase to a whitelisted logical app id."""
    key = (spoken_app or "").strip().lower()
    if key in APP_WHITELIST:
        return APP_WHITELIST[key]
    padded = f" {key} "
    contained = [name for name in APP_WHITELIST if f" {name} " in padded]
    if contained:
        return APP_WHITELIST[max(contained, key=len)]
    raise SecurityError(
        f"I don't know the app '{spoken_app}'. "
        "Try browser, files, terminal, videos, editor, or calculator."
    )


def closable_process_names(app_id: str) -> list:
    names = APP_PROCESS_NAMES.get(PLATFORM_KEY, {}).get(app_id, [])
    return [n for n in names if n.lower() not in PROTECTED_PROCESS_NAMES]


def validate(command: ParsedCommand) -> ValidationResult:
    """
    Map a parsed intent to a frozen, pre-defined action.

    Returns BLOCKED / NEEDS_CONFIRM / SAFE / ERROR. The executor only acts on
    SAFE (or user-confirmed) results and never sees command.raw_text.
    """
    intent = command.intent
    slots = command.slots

    try:
        if intent == Intent.UNKNOWN:
            return ValidationResult(
                state=ValidationState.ERROR,
                reason=(
                    "Command not recognized. You can ask me to list files, "
                    "open the browser, or show cpu usage."
                ),
            )

        if intent in (Intent.HELP, Intent.LIST_APPS):
            label = "app catalog" if intent == Intent.LIST_APPS else "voice help"
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Help only speaks; nothing is executed.",
                action=f"speak({label})",
            )

        if intent == Intent.LIST_FILES:
            root = ensure_workspace()
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Listing is confined to the workspace.",
                action="list_dir(<workspace>)",
                resolved_path=root,
            )

        if intent == Intent.CREATE_FILE:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=False)
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="New empty file inside the workspace.",
                action=f"create_file({path.name})",
                resolved_path=path,
            )

        if intent == Intent.CREATE_FOLDER:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=False)
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="New folder inside the workspace.",
                action=f"create_folder({path.name})",
                resolved_path=path,
            )

        if intent == Intent.RENAME_FILE:
            src = resolve_workspace_path(slots.get("source", ""), must_exist=True)
            dst = resolve_workspace_path(slots.get("destination", ""), must_exist=False)
            if src == dst:
                raise SecurityError("Source and destination are the same.")
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Rename stays inside the workspace and never overwrites.",
                action=f"rename({src.name} -> {dst.name})",
                resolved_path=src,
                extra_path=dst,
            )

        if intent == Intent.DELETE_FILE:
            path = resolve_workspace_path(slots.get("name", ""), must_exist=True)
            if path.is_dir():
                action = f"remove_empty_folder({path.name})"
                extra = " Empty folders only. Recursive delete is forbidden."
            else:
                action = f"delete_file({path.name})"
                extra = ""
            return ValidationResult(
                state=ValidationState.NEEDS_CONFIRM,
                reason="Delete requires explicit confirmation." + extra,
                action=action,
                resolved_path=path,
            )

        if intent == Intent.OPEN_APP:
            app_id = resolve_app(slots.get("app", ""))
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Application is on the launch whitelist.",
                action=f"launch({app_id})",
                app_id=app_id,
            )

        if intent == Intent.CLOSE_APP:
            app_id = resolve_app(slots.get("app", ""))
            if not closable_process_names(app_id):
                raise SecurityError("That app is part of the system and cannot be closed by voice.")
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Close is limited to a whitelisted, non-critical process.",
                action=f"terminate({app_id})",
                app_id=app_id,
            )

        if intent == Intent.SHOW_PROCESSES:
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Read-only process snapshot.",
                action="process_snapshot(top by cpu)",
            )

        if intent == Intent.SHOW_USAGE:
            return ValidationResult(
                state=ValidationState.SAFE,
                reason="Read-only CPU, RAM and disk snapshot.",
                action="usage_snapshot(cpu, ram, disk)",
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

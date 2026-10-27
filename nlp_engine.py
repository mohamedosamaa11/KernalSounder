"""
KernalSounder — command understanding.

No neural NLP model is loaded. A keyword scorer plus regex slot extractors
map noisy STT transcripts onto a closed intent set. RAM cost is effectively
zero compared with spaCy/transformers, which is required on a 4 GB VM.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from security import Intent, ParsedCommand

_WORD_DOT = re.compile(
    r"\b(dot|period)\b",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")

# Spoken punctuation / separators that Vosk often emits as words.
_SPOKEN_REPLACEMENTS = (
    (" under score ", "_"),
    (" underscore ", "_"),
    (" dash ", "-"),
    (" hyphen ", "-"),
    (" minus ", "-"),
)

INTENT_PHRASES: Dict[Intent, Tuple[str, ...]] = {
    Intent.LIST_FILES: (
        "list files",
        "list the files",
        "show files",
        "show the files",
        "list directory",
        "list folder",
        "what files",
        "what's in the workspace",
        "show workspace",
        "list contents",
    ),
    Intent.CREATE_FILE: (
        "create file",
        "create a file",
        "make file",
        "make a file",
        "new file",
        "touch file",
    ),
    Intent.CREATE_FOLDER: (
        "create folder",
        "create a folder",
        "create directory",
        "create a directory",
        "make folder",
        "make a folder",
        "make directory",
        "new folder",
        "mkdir",
    ),
    Intent.RENAME_FILE: (
        "rename file",
        "rename",
        "change name",
        "move file",
    ),
    Intent.DELETE_FILE: (
        "delete file",
        "delete the file",
        "delete folder",
        "delete the folder",
        "remove file",
        "remove the file",
        "remove folder",
        "erase file",
        "unlink",
    ),
    Intent.OPEN_APP: (
        "open",
        "launch",
        "start",
        "run application",
        "run app",
    ),
    Intent.CLOSE_APP: (
        "close",
        "quit",
        "kill",
        "stop application",
        "exit app",
    ),
    Intent.SHOW_PROCESSES: (
        "show processes",
        "list processes",
        "running processes",
        "show running programs",
        "list programs",
        "task list",
        "what is running",
    ),
    Intent.SHOW_USAGE: (
        "show usage",
        "system usage",
        "cpu usage",
        "ram usage",
        "memory usage",
        "disk usage",
        "show cpu",
        "show ram",
        "show memory",
        "show disk",
        "system status",
        "resource usage",
        "how much memory",
        "how much ram",
        "how much cpu",
    ),
    Intent.HELP: (
        "help",
        "what can you do",
        "available commands",
        "list commands",
        "how to use",
    ),
    Intent.CONFIRM: (
        "confirm",
        "yes",
        "yeah",
        "yep",
        "do it",
        "proceed",
        "okay",
        "ok",
        "sure",
        "affirmative",
    ),
    Intent.CANCEL: (
        "cancel",
        "no",
        "nope",
        "stop",
        "abort",
        "never mind",
        "don't",
        "do not",
        "negative",
    ),
}

_CREATE_FILE_RE = re.compile(
    r"(?:create|make|new|touch)\s+(?:a\s+)?(?:file|document)\s+"
    r"(?:called|named|with name|with the name)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_CREATE_FOLDER_RE = re.compile(
    r"(?:create|make|new)\s+(?:a\s+)?(?:folder|directory|dir)\s+"
    r"(?:called|named|with name|with the name)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_RENAME_RE = re.compile(
    r"(?:rename|change name of|change the name of|move)\s+"
    r"(?:the\s+)?(?:file|folder|directory)?\s*"
    r"(?P<source>[\w.\-]+)\s+(?:to|into)\s+(?P<destination>[\w.\-]+)",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    r"(?:delete|remove|erase|unlink)\s+(?:the\s+)?(?:file|folder|directory)?\s*"
    r"(?:called|named)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_OPEN_RE = re.compile(
    r"(?:open|launch|start|run)\s+(?:the\s+)?(?:app|application|program)?\s*"
    r"(?P<app>.+)$",
    re.IGNORECASE,
)
_CLOSE_RE = re.compile(
    r"(?:close|quit|kill|stop|exit)\s+(?:the\s+)?(?:app|application|program)?\s*"
    r"(?P<app>.+)$",
    re.IGNORECASE,
)

_TRAILING_PLEASE = re.compile(
    r"\b(please|now|for me|thanks|thank you)\b",
    re.IGNORECASE,
)


def normalize_speech(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = _WORD_DOT.sub(".", lowered)
    for src, dst in _SPOKEN_REPLACEMENTS:
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"\s*\.\s*", ".", lowered)
    lowered = _TRAILING_PLEASE.sub(" ", lowered)
    lowered = _MULTI_SPACE.sub(" ", lowered).strip(" .,!?")
    return lowered


def _clean_name(value: str) -> str:
    token = (value or "").strip().strip("\"'")
    token = token.replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9._\-]", "", token)
    return token.strip("._-")


def _score_intent(text: str) -> Tuple[Intent, float]:
    best = Intent.UNKNOWN
    best_score = 0.0
    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            if text == phrase:
                return intent, 1.0
            if phrase in text:
                score = min(0.97, 0.55 + len(phrase) / max(len(text), 1) * 0.5)
                if score > best_score:
                    best, best_score = intent, score
    return best, best_score


def parse_command(raw_text: str) -> ParsedCommand:
    text = normalize_speech(raw_text)
    if not text:
        return ParsedCommand(intent=Intent.UNKNOWN, raw_text=raw_text, confidence=0.0)

    intent, confidence = _score_intent(text)
    slots: Dict[str, str] = {}

    if intent == Intent.CREATE_FILE:
        match = _CREATE_FILE_RE.search(text)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent == Intent.CREATE_FOLDER:
        match = _CREATE_FOLDER_RE.search(text)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent == Intent.RENAME_FILE:
        match = _RENAME_RE.search(text)
        if match:
            slots["source"] = _clean_name(match.group("source"))
            slots["destination"] = _clean_name(match.group("destination"))
    elif intent == Intent.DELETE_FILE:
        match = _DELETE_RE.search(text)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent == Intent.OPEN_APP:
        match = _OPEN_RE.search(text)
        if match:
            slots["app"] = match.group("app").strip()
    elif intent == Intent.CLOSE_APP:
        match = _CLOSE_RE.search(text)
        if match:
            slots["app"] = match.group("app").strip()

    if intent in (Intent.CREATE_FILE, Intent.CREATE_FOLDER, Intent.DELETE_FILE):
        if not slots.get("name"):
            confidence = min(confidence, 0.4)
    if intent == Intent.RENAME_FILE and (
        not slots.get("source") or not slots.get("destination")
    ):
        confidence = min(confidence, 0.4)
    if intent in (Intent.OPEN_APP, Intent.CLOSE_APP) and not slots.get("app"):
        confidence = min(confidence, 0.4)

    # Ambiguous short words like "open" with no target stay unknown-ish.
    if intent in (Intent.OPEN_APP, Intent.CLOSE_APP) and confidence < 0.5:
        intent = Intent.UNKNOWN

    return ParsedCommand(
        intent=intent,
        slots=slots,
        raw_text=raw_text,
        confidence=confidence,
    )


HELP_TEXT = (
    "You can say: list files; create file called notes.txt; "
    "create folder called projects; rename old.txt to new.txt; "
    "delete file notes.txt; open firefox; close firefox; "
    "show processes; or show cpu ram and disk usage."
)

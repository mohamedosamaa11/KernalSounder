"""
KernalSounder — command understanding.

Still no neural NLP: a filler stripper, verb rules, and a spoken-app dictionary
keep RAM near zero while accepting conversational phrasing.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from config import SPOKEN_APP_MAP
from security import Intent, ParsedCommand

_WORD_DOT = re.compile(r"\b(dot|period)\b", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")
_NON_NAME = re.compile(r"[^A-Za-z0-9._\-]")

_SPOKEN_REPLACEMENTS = (
    (" under score ", "_"),
    (" underscore ", "_"),
    (" dash ", "-"),
    (" hyphen ", "-"),
    (" minus ", "-"),
)

# Longest phrases first so "could you please" wins over "could you".
FILLER_PHRASES = (
    "i would like you to",
    "i would like to",
    "i want you to",
    "i need you to",
    "i'd like you to",
    "i'd like to",
    "could you please",
    "would you please",
    "can you please",
    "if you could",
    "if you can",
    "go ahead and",
    "would you mind",
    "do you mind",
    "i want to",
    "i need to",
    "i wanna",
    "could you",
    "would you",
    "can you",
    "show me the",
    "show me",
    "tell me the",
    "tell me",
    "give me the",
    "give me",
    "what is the",
    "what are the",
    "what's the",
    "whats the",
    "what is",
    "what are",
    "what's",
    "whats",
    "are there any",
    "is there any",
    "are there",
    "is there",
    "do we have",
    "do i have",
    "right now",
    "real quick",
    "for me please",
    "for me",
    "thank you",
    "thanks",
    "please",
    "kindly",
)

FILLER_WORDS = frozenset(
    {
        "please",
        "just",
        "kindly",
        "maybe",
        "actually",
        "simply",
        "really",
        "quickly",
        "hey",
        "hi",
        "hello",
        "yo",
        "um",
        "uh",
        "the",
        "a",
        "an",
        "some",
        "my",
        "your",
        "this",
        "that",
        "those",
        "these",
        "me",
        "now",
        "like",
        "also",
        "then",
        "currently",
        "available",
    }
)

# Do not drop these while stripping leftover words — they carry the intent.
INTENT_KEEP_WORDS = frozenset(
    {
        "list",
        "show",
        "display",
        "create",
        "make",
        "new",
        "touch",
        "rename",
        "move",
        "delete",
        "remove",
        "erase",
        "open",
        "launch",
        "start",
        "run",
        "play",
        "watch",
        "close",
        "quit",
        "kill",
        "stop",
        "exit",
        "help",
        "cpu",
        "ram",
        "memory",
        "disk",
        "usage",
        "process",
        "processes",
        "files",
        "file",
        "folder",
        "folders",
        "directory",
        "apps",
        "applications",
        "confirm",
        "cancel",
        "yes",
        "no",
        "to",
    }
)

OPEN_VERBS = ("open", "launch", "start", "run", "play", "watch")
CLOSE_VERBS = ("close", "quit", "kill", "exit")
CREATE_VERBS = ("create", "make", "new", "touch")
DELETE_VERBS = ("delete", "remove", "erase", "unlink")

INTENT_PHRASES: Dict[Intent, Tuple[str, ...]] = {
    Intent.LIST_FILES: (
        "list files",
        "list the files",
        "show files",
        "show the files",
        "list directory",
        "list folder",
        "list contents",
        "what files",
        "what's in the workspace",
        "whats in the workspace",
        "what's in here",
        "whats in here",
        "show workspace",
        "workspace files",
        "files here",
        "contents of workspace",
    ),
    Intent.CREATE_FILE: (
        "create file",
        "create a file",
        "make file",
        "make a file",
        "new file",
        "touch file",
        "make document",
        "create document",
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
        "change the name",
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
        "play",
        "watch",
    ),
    Intent.CLOSE_APP: (
        "close",
        "quit",
        "kill",
        "stop application",
        "exit app",
        "exit application",
    ),
    Intent.SHOW_PROCESSES: (
        "show processes",
        "list processes",
        "running processes",
        "show running programs",
        "running programs",
        "list programs",
        "task list",
        "what is running",
        "what's running",
        "whats running",
        "what is running now",
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
        "how much disk",
        "cpu ram",
        "resources",
    ),
    Intent.HELP: (
        "help",
        "what can you do",
        "what do you do",
        "available commands",
        "list commands",
        "how to use",
        "how do i use you",
        "what commands",
    ),
    Intent.LIST_APPS: (
        "what apps are available",
        "what applications are available",
        "what apps",
        "which apps",
        "available apps",
        "list apps",
        "show apps",
        "what can i open",
        "what can you open",
        "apps available",
        "applications available",
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
        "go ahead",
    ),
    Intent.CANCEL: (
        "cancel",
        "no",
        "nope",
        "abort",
        "never mind",
        "nevermind",
        "don't",
        "do not",
        "negative",
        "stop it",
    ),
}

_CREATE_FILE_RE = re.compile(
    r"(?:create|make|new|touch)\s+(?:file|document)\s+"
    r"(?:called|named|with name|with the name)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_CREATE_FOLDER_RE = re.compile(
    r"(?:create|make|new)\s+(?:folder|directory|dir)\s+"
    r"(?:called|named|with name|with the name)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_RENAME_RE = re.compile(
    r"(?:rename|change name of|change the name of|move)\s+"
    r"(?:file|folder|directory)?\s*"
    r"(?P<source>[\w.\-]+)\s+to\s+(?P<destination>[\w.\-]+)",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    r"(?:delete|remove|erase|unlink)\s+(?:file|folder|directory)?\s*"
    r"(?:called|named)?\s*(?P<name>.+)$",
    re.IGNORECASE,
)
_OPEN_RE = re.compile(
    r"(?:open|launch|start|run|play|watch)\s+"
    r"(?:app|application|program|the)?\s*"
    r"(?P<app>.+)$",
    re.IGNORECASE,
)
_CLOSE_RE = re.compile(
    r"(?:close|quit|kill|stop|exit)\s+"
    r"(?:app|application|program|the)?\s*"
    r"(?P<app>.+)$",
    re.IGNORECASE,
)

_APPS_QUERY_RE = re.compile(
    r"\b(what|which|list|show|available)\b.+\b(apps|applications|software)\b"
    r"|\b(apps|applications)\b.+\b(available|open|use|list)\b"
    r"|\bwhat can i open\b"
    r"|\bwhat can you open\b",
    re.IGNORECASE,
)

_APP_KEYS = tuple(sorted(SPOKEN_APP_MAP.keys(), key=len, reverse=True))

FALLBACK_TEXT = (
    "Command not recognized. You can ask me to list files, open firefox, "
    "or show cpu usage."
)

HELP_TEXT = (
    "You can say: list files; create file called notes.txt; "
    "create folder called projects; rename old.txt to new.txt; "
    "delete file notes.txt; open browser or open videos; close firefox; "
    "show processes; or show cpu ram and disk usage. "
    "Ask what apps are available to hear the app names I know."
)

APPS_HELP_TEXT = (
    "You can open a browser or internet for Firefox, "
    "files or folder for the file manager, "
    "terminal for the command line, "
    "and videos or player for the video player. "
    "Try saying open browser, open files, open terminal, or open videos."
)

_LEFTOVER_INTENTS = {
    "files": Intent.LIST_FILES,
    "contents": Intent.LIST_FILES,
    "workspace": Intent.LIST_FILES,
    "directory": Intent.LIST_FILES,
    "cpu": Intent.SHOW_USAGE,
    "ram": Intent.SHOW_USAGE,
    "memory": Intent.SHOW_USAGE,
    "disk": Intent.SHOW_USAGE,
    "usage": Intent.SHOW_USAGE,
    "status": Intent.SHOW_USAGE,
    "resources": Intent.SHOW_USAGE,
    "processes": Intent.SHOW_PROCESSES,
    "process": Intent.SHOW_PROCESSES,
    "tasks": Intent.SHOW_PROCESSES,
    "help": Intent.HELP,
    "commands": Intent.HELP,
    "apps": Intent.LIST_APPS,
    "applications": Intent.LIST_APPS,
    "programs": Intent.SHOW_PROCESSES,
}


def _has_any(text: str, words: Tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def normalize_speech(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = lowered.replace("'", "'")
    lowered = lowered.replace("what's", "what is").replace("whats", "what is")
    lowered = lowered.replace("i'd", "i would")
    lowered = _WORD_DOT.sub(".", lowered)
    for src, dst in _SPOKEN_REPLACEMENTS:
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"\s*\.\s*", ".", lowered)
    lowered = re.sub(r"[?!,:;]+", " ", lowered)
    lowered = _MULTI_SPACE.sub(" ", lowered).strip(" .,")
    return lowered


def strip_fillers(text: str) -> str:
    cleaned = f" {text} "
    for phrase in FILLER_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
    tokens = []
    for token in cleaned.split():
        if token in FILLER_WORDS and token not in INTENT_KEEP_WORDS:
            continue
        tokens.append(token)
    return _MULTI_SPACE.sub(" ", " ".join(tokens)).strip()


def match_spoken_app(text: str) -> Optional[str]:
    """Return the longest generic app phrase found in spoken text."""
    blob = f" {normalize_speech(text)} "
    for key in _APP_KEYS:
        if f" {key} " in blob or blob.strip() == key:
            return key
    return None


def _clean_name(value: str) -> str:
    token = (value or "").strip().strip("\"'")
    token = strip_fillers(token)
    token = token.replace(" ", "_")
    token = _NON_NAME.sub("", token)
    return token.strip("._-")


def _extract_app_slot(text: str) -> str:
    match = _OPEN_RE.search(text) or _CLOSE_RE.search(text)
    candidate = match.group("app") if match else text
    candidate = strip_fillers(candidate)
    candidate = re.sub(
        r"\b(open|launch|start|run|play|watch|close|quit|kill|stop|exit|"
        r"app|application|program)\b",
        " ",
        candidate,
    )
    candidate = _MULTI_SPACE.sub(" ", candidate).strip()
    alias = match_spoken_app(candidate) or match_spoken_app(text)
    return alias or candidate


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


def _rule_intent(stripped: str, original: str) -> Optional[Tuple[Intent, float]]:
    if _APPS_QUERY_RE.search(original) or _APPS_QUERY_RE.search(stripped):
        if not _has_any(stripped, ("close", "kill", "quit", "running", "process", "processes")):
            return Intent.LIST_APPS, 0.96

    if _has_any(stripped, CREATE_VERBS) and _has_any(stripped, ("file", "document")):
        return Intent.CREATE_FILE, 0.93
    if _has_any(stripped, CREATE_VERBS) and _has_any(stripped, ("folder", "directory", "dir")):
        return Intent.CREATE_FOLDER, 0.93
    if _has_any(stripped, ("rename",)) or (
        "change" in stripped and "name" in stripped
    ):
        return Intent.RENAME_FILE, 0.9
    if _has_any(stripped, DELETE_VERBS):
        return Intent.DELETE_FILE, 0.92

    if _has_any(stripped, OPEN_VERBS):
        return Intent.OPEN_APP, 0.9
    if _has_any(stripped, CLOSE_VERBS):
        return Intent.CLOSE_APP, 0.9

    if _has_any(stripped, ("process", "processes", "running")) and _has_any(
        stripped, ("show", "list", "running", "task", "tasks")
    ):
        return Intent.SHOW_PROCESSES, 0.9
    if _has_any(stripped, ("cpu", "ram", "memory", "disk", "usage", "resource", "resources")):
        return Intent.SHOW_USAGE, 0.88
    if _has_any(stripped, ("list", "show", "display")) and _has_any(
        stripped, ("file", "files", "folder", "folders", "directory", "contents", "workspace")
    ):
        return Intent.LIST_FILES, 0.9
    return None


def _fill_slots(intent: Intent, stripped: str) -> Dict[str, str]:
    slots: Dict[str, str] = {}
    if intent == Intent.CREATE_FILE:
        match = _CREATE_FILE_RE.search(stripped)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent == Intent.CREATE_FOLDER:
        match = _CREATE_FOLDER_RE.search(stripped)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent == Intent.RENAME_FILE:
        match = _RENAME_RE.search(stripped)
        if match:
            slots["source"] = _clean_name(match.group("source"))
            slots["destination"] = _clean_name(match.group("destination"))
    elif intent == Intent.DELETE_FILE:
        match = _DELETE_RE.search(stripped)
        if match:
            slots["name"] = _clean_name(match.group("name"))
    elif intent in (Intent.OPEN_APP, Intent.CLOSE_APP):
        app = _extract_app_slot(stripped)
        if app:
            slots["app"] = app
    return slots


def parse_command(raw_text: str) -> ParsedCommand:
    original = normalize_speech(raw_text)
    if not original:
        return ParsedCommand(intent=Intent.UNKNOWN, raw_text=raw_text, confidence=0.0)

    stripped = strip_fillers(original) or original

    # Confirm / cancel must win over leftover filler matches like "ok".
    if stripped in {"confirm", "yes", "yeah", "yep", "ok", "okay", "sure", "do it", "proceed", "go ahead"}:
        return ParsedCommand(intent=Intent.CONFIRM, raw_text=raw_text, confidence=1.0)
    if stripped in {"cancel", "no", "nope", "abort", "never mind", "nevermind", "stop it", "negative"}:
        return ParsedCommand(intent=Intent.CANCEL, raw_text=raw_text, confidence=1.0)

    ruled = _rule_intent(stripped, original)
    scored_intent, scored_conf = _score_intent(stripped)
    if scored_intent is Intent.UNKNOWN:
        scored_intent, scored_conf = _score_intent(original)

    if ruled and (scored_intent is Intent.UNKNOWN or ruled[1] >= scored_conf):
        intent, confidence = ruled
    else:
        intent, confidence = scored_intent, scored_conf

    # Bare leftover after "show me the …" / "what is …"
    if intent is Intent.UNKNOWN or confidence < 0.5:
        leftover = stripped
        if leftover in _LEFTOVER_INTENTS:
            intent = _LEFTOVER_INTENTS[leftover]
            confidence = 0.82
        else:
            alias = match_spoken_app(leftover)
            if alias and leftover not in {"files", "folder", "folders"}:
                intent = Intent.OPEN_APP
                confidence = 0.8
                stripped = f"open {alias}"

    slots = _fill_slots(intent, stripped)

    if intent in (Intent.CREATE_FILE, Intent.CREATE_FOLDER, Intent.DELETE_FILE):
        if not slots.get("name"):
            confidence = min(confidence, 0.4)
    if intent == Intent.RENAME_FILE and (
        not slots.get("source") or not slots.get("destination")
    ):
        confidence = min(confidence, 0.4)
    if intent in (Intent.OPEN_APP, Intent.CLOSE_APP):
        if slots.get("app"):
            confidence = max(confidence, 0.88)
        else:
            confidence = min(confidence, 0.4)
            intent = Intent.UNKNOWN

    return ParsedCommand(
        intent=intent,
        slots=slots,
        raw_text=raw_text,
        confidence=confidence,
    )

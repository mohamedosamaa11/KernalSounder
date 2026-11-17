"""
KernalSounder — personal vocabulary (vocabulary.txt).

Two uses:
  * words go into the speech recognizer's prompt so it spells them your way;
  * file/folder names that are a near miss ("Mohammed") snap to your spelling ("Mohamed").
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Tuple

from config import BASE_DIR

VOCAB_FILE = BASE_DIR / "vocabulary.txt"


@lru_cache(maxsize=1)
def load_vocabulary() -> Tuple[str, ...]:
    try:
        lines = VOCAB_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    words = []
    for line in lines:
        word = line.split("#", 1)[0].strip()
        if word and word.lower() not in {w.lower() for w in words}:
            words.append(word)
    return tuple(words)


def snap_to_vocabulary(name: str, cutoff: float = 0.84) -> str:
    """Replace each part of a file name with the vocabulary spelling when it is a close match."""
    vocab = load_vocabulary()
    if not name or not vocab:
        return name
    lower_map = {w.lower(): w for w in vocab}
    stem, dot, ext = name.rpartition(".") if "." in name else (name, "", "")
    parts = re.split(r"([_\-.])", stem)
    out = []
    for part in parts:
        low = part.lower()
        if len(part) >= 3 and part.isalpha():
            if low in lower_map:
                out.append(lower_map[low])
                continue
            match = difflib.get_close_matches(low, list(lower_map), n=1, cutoff=cutoff)
            if match:
                out.append(lower_map[match[0]])
                continue
        out.append(part)
    snapped = "".join(out)
    return f"{snapped}{dot}{ext}" if dot else snapped

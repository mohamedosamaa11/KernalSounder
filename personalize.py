"""
KernalSounder — make the wake word fit YOUR voice in about a minute.

No new synthetic data and no GPU: the shipped wake-word head ("Hey Nova") is
fine-tuned on
  * ~12 recordings of you saying the phrase (each augmented ~10x),
  * ~15 s of your room's normal sound (TV, fan, people talking),
  * triggers the Whisper check rejected earlier (if collected),
while an L2-SP penalty keeps the weights close to the base model, so a few
personal samples cannot make it forget everything else it learned.

It also learns how Whisper hears YOUR wake phrase (e.g. "hey noah") and
stores those words as aliases for the second-stage check.

Used by the "Train my voice" wizard in the app and by
`python train_wake_word.py personalize`.
"""

from __future__ import annotations

import lean_runtime  # noqa: F401

import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from audio import CommandRecorder, rms_level
from config import SAMPLE_RATE

_ANCHOR = 3e-2
_EPOCHS = 25

GREETING_WORDS = {"hi", "hey", "high", "hello", "i", "a", "okay", "ok", "oh", "um", "uh", "so",
                  "and", "the", "eye", "hay", "hai"}


def custom_dir() -> Path:
    from train_wake_word import CUSTOM_DIR

    return CUSTOM_DIR


def personal_model_path(name: str) -> Path:
    return custom_dir() / f"{name}_personal.npz"


def _load_head(path: Path):
    with np.load(path) as data:
        params = [data[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2")]
        mu, sigma = data["mu"].astype(np.float32), data["sigma"].astype(np.float32)
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return params, mu, sigma, meta


def _clip_scores(params, mu, sigma, embedder, clips: Sequence[np.ndarray], rng, seed0: int) -> List[float]:
    from train_wake_word import FRAMES, _augment, _positive_windows, _predict

    scores = []
    for i, clip in enumerate(clips):
        audio, end = _augment(clip, rng, seed0 + i, pad_before=(1.8, 1.8), pad_after=(0.5, 0.5))
        pos, _early = _positive_windows(embedder.embeddings(audio), end)
        if len(pos) == 0:
            scores.append(0.0)
            continue
        scores.append(float(_predict(params, mu, sigma, pos.reshape(-1, FRAMES * 96)).max()))
    return scores


def _segments(audio: np.ndarray, seconds: float = 3.0) -> List[np.ndarray]:
    step = int(seconds * SAMPLE_RATE)
    if len(audio) <= step:
        return [audio] if len(audio) > SAMPLE_RATE // 2 else []
    return [audio[s:s + step] for s in range(0, len(audio) - step + 1, step // 2)]


def learn_aliases(stt, clips: Sequence[np.ndarray], phrase: str) -> List[str]:
    """Words Whisper uses for the user's own wake phrase that the keyword check would miss."""
    if stt is None:
        return []
    counts: Counter = Counter()
    misses = 0
    for clip in clips:
        padded = np.concatenate([np.zeros(SAMPLE_RATE // 4, np.int16), clip, np.zeros(SAMPLE_RATE // 4, np.int16)])
        ok, text = stt.verify_keyword(padded, phrase)
        if ok is None:
            return []
        if ok:
            continue
        misses += 1
        words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in GREETING_WORDS and len(w) >= 3]
        counts.update(set(words))
    need = 2 if len(clips) >= 6 else 1
    return sorted(w for w, c in counts.items() if c >= need)


def _same_voice_negatives(clip: np.ndarray, k: int) -> List[np.ndarray]:
    """Speaker-matched negatives made from a positive clip: the same voice, not the phrase.

    Without them a few personal samples teach the model "this voice" instead of
    "this phrase" and it fires on anything the user says.
    """
    clip = np.asarray(clip, dtype=np.int16)
    n = len(clip)
    out = [clip[::-1].copy(), clip[: int(n * 0.42)].copy(), clip[int(n * 0.38):].copy()]
    piece = int(0.12 * SAMPLE_RATE)
    parts = [clip[i:i + piece] for i in range(0, n, piece)]
    if len(parts) >= 3:
        order = list(range(len(parts)))
        random.Random(k).shuffle(order)
        if order == sorted(order):
            order.reverse()
        out.append(np.concatenate([parts[i] for i in order]))
    doubled = np.concatenate([clip[: int(n * 0.42)], clip[: int(n * 0.42)]])
    out.append(doubled)
    return [o for o in out if len(o) > SAMPLE_RATE // 10]


def replay_clips(phrase: str, exclude: Sequence[str] = (), name: str = "") -> tuple:
    """Look-alike negatives and phrase positives in the OS voices, cached per machine."""
    from os_tts import synthesize
    from train_wake_word import COMMANDS, soundalikes_for

    skip = {e.lower() for e in exclude}
    negatives = [t for t in soundalikes_for(name or re.sub(r"[^a-z0-9]+", "_", phrase.lower()))
                 + COMMANDS[:8] if t.lower() not in skip]
    key = re.sub(r"[^a-z0-9]+", "_", (phrase + "_" + "_".join(sorted(skip))).lower())[:80]
    cache = custom_dir() / f"replay_{key}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            pos = [data[k] for k in sorted(data.files) if k.startswith("p")]
            neg = [data[k] for k in sorted(data.files) if k.startswith("n")]
        if pos and neg:
            return pos, neg
    jobs_neg = [(t, v, r) for t in negatives for v in (0, 1, 2) for r in (-3, 2)]
    jobs_pos = [(phrase, v, r) for v in (0, 1, 2) for r in (-4, -2, 0, 2, 4)]
    clips = synthesize(jobs_pos + jobs_neg)
    if len(clips) != len(jobs_pos) + len(jobs_neg):
        return [], []
    pos, neg = clips[:len(jobs_pos)], clips[len(jobs_pos):]
    custom_dir().mkdir(parents=True, exist_ok=True)
    arrays = {f"p{i:04d}": c for i, c in enumerate(pos)}
    arrays.update({f"n{i:04d}": c for i, c in enumerate(neg)})
    np.savez_compressed(cache, **arrays)
    return pos, neg


def personalize(name: str, phrase: str, positives: Sequence[np.ndarray], background: Sequence[np.ndarray],
                stt=None, progress: Callable[[float, str], None] = lambda p, m: None,
                copies: int = 10, epochs: int = _EPOCHS, seed: int = 5,
                extra_negatives: Sequence[np.ndarray] = (), replay: bool = True,
                replay_exclude: Sequence[str] = ()) -> dict:
    """Fine-tune <name>.npz on the user's samples; writes <name>_personal.npz. Returns a report."""
    from train_wake_word import FRAMES, Embedder, _augment, _positive_windows, _predict, _train_mlp, _windows

    started = time.monotonic()
    base_path = custom_dir() / f"{name}.npz"
    params0, mu, sigma, base_meta = _load_head(base_path)
    rng = random.Random(seed)
    embedder = Embedder()

    clips = [np.asarray(c, dtype=np.int16) for c in positives if len(c) > SAMPLE_RATE // 4]
    if len(clips) < 4:
        raise ValueError("Need at least 4 recordings of the wake phrase.")
    held = clips[::4]
    train_clips = [c for i, c in enumerate(clips) if i % 4 != 0]

    room = np.concatenate([np.asarray(b, dtype=np.int16) for b in background]) if background else np.zeros(0, np.int16)
    cut = int(len(room) * 0.75)
    room_train, room_val = _segments(room[:cut]), _segments(room[cut:])
    hard = []
    folder = custom_dir() / f"{name}_samples" / "negative"
    if folder.is_dir():
        hard = [np.load(p).astype(np.int16) for p in sorted(folder.glob("*.npy"))[-300:]]

    progress(0.05, "Listening to your recordings…")
    pos_x, neg_x, neg_w = [], [], []
    seed_n = 100
    for k, clip in enumerate(train_clips):
        for _ in range(copies):
            seed_n += 1
            audio, end = _augment(clip, rng, seed_n)
            pos, early = _positive_windows(embedder.embeddings(audio), end)
            pos_x.append(pos)
            if early.size:
                neg_x.append(early)
                neg_w.append(np.full(len(early), 0.5, np.float32))
        progress(0.05 + 0.35 * (k + 1) / len(train_clips), "Listening to your recordings…")

    progress(0.38, "Learning what is NOT the phrase in your voice…")
    extra = [np.asarray(c, dtype=np.int16) for c in extra_negatives]
    extra_val = extra[::3]
    extra_train = [c for i, c in enumerate(extra) if i % 3 != 0]
    for k, clip in enumerate(train_clips + extra_train):
        for variant in _same_voice_negatives(clip, k):
            seed_n += 1
            audio, _ = _augment(variant, rng, seed_n, pad_before=(1.6, 2.2), pad_after=(0.2, 0.5))
            wins = _windows(embedder.embeddings(audio))
            neg_x.append(wins)
            neg_w.append(np.full(len(wins), 2.0, np.float32))

    if replay:
        progress(0.40, "Reminding the model of look-alike phrases…")
        rep_pos, rep_neg = replay_clips(phrase, replay_exclude, name)
        rep_val = rep_neg[::5]
        for clip in [c for i, c in enumerate(rep_neg) if i % 5 != 0]:
            seed_n += 1
            audio, _ = _augment(clip, rng, seed_n, pad_before=(1.6, 2.2), pad_after=(0.2, 0.5))
            wins = _windows(embedder.embeddings(audio))
            neg_x.append(wins)
            neg_w.append(np.full(len(wins), 2.0, np.float32))
        for clip in rep_pos:
            for _ in range(6):
                seed_n += 1
                audio, end = _augment(clip, rng, seed_n)
                pos, early = _positive_windows(embedder.embeddings(audio), end)
                pos_x.append(pos)

    progress(0.42, "Learning your room's sound…")
    for seg in room_train + hard:
        for _ in range(2):
            seed_n += 1
            audio, _ = _augment(seg, rng, seed_n, pad_before=(1.2, 2.0), pad_after=(0.2, 0.5))
            wins = _windows(embedder.embeddings(audio))
            neg_x.append(wins)
            neg_w.append(np.full(len(wins), 1.5, np.float32))
    for k in range(20):
        noise = np.asarray(rng.choice(room_train) if room_train else np.zeros(SAMPLE_RATE * 3, np.int16))
        quiet = (noise.astype(np.float32) * rng.uniform(0.05, 0.4)).astype(np.int16)
        wins = _windows(embedder.embeddings(np.concatenate([np.zeros(SAMPLE_RATE * 2, np.int16), quiet])))
        neg_x.append(wins)
        neg_w.append(np.ones(len(wins), np.float32))

    xp = np.concatenate(pos_x).reshape(-1, FRAMES * 96)
    xn = np.concatenate(neg_x).reshape(-1, FRAMES * 96) if neg_x else np.zeros((0, FRAMES * 96), np.float32)
    wn = np.concatenate(neg_w) if neg_w else np.zeros(0, np.float32)
    x = np.concatenate([xp, xn]).astype(np.float32)
    y = np.concatenate([np.ones(len(xp), np.float32), np.zeros(len(xn), np.float32)])
    w = np.concatenate([np.full(len(xp), max(1.0, len(xn) / max(1, len(xp))) * 0.6, np.float32), wn])

    progress(0.55, "Training…")
    params = _train_mlp((x - mu) / sigma, y, w, hidden=params0[0].shape[1], epochs=epochs, seed=seed,
                        init=params0, anchor=_ANCHOR, lr=4e-4, verbose=False,
                        progress=lambda f: progress(0.55 + 0.3 * f, "Training…"))

    progress(0.86, "Checking the result on recordings it never trained on…")
    before = _clip_scores(params0, mu, sigma, embedder, held, random.Random(9), 900)
    after = _clip_scores(params, mu, sigma, embedder, held, random.Random(9), 900)
    val_negs = list(extra_val) + (list(rep_val) if replay else []) + list(room_val)

    def neg_maxima(prm) -> List[float]:
        out = []
        for clip in val_negs:
            wins = _windows(embedder.embeddings(np.concatenate([np.zeros(SAMPLE_RATE * 2, np.int16), clip])))
            out.append(float(_predict(prm, mu, sigma, wins.reshape(-1, FRAMES * 96)).max()) if len(wins) else 0.0)
        return out

    neg_before, neg_after = neg_maxima(params0), neg_maxima(params)
    threshold = min(0.97, max(0.5, (max(neg_after) + 0.02) if neg_after else 0.5))
    recall_before = int(sum(s >= 0.5 for s in before))
    recall_after = int(sum(s >= threshold for s in after))
    false_before = int(sum(s >= 0.5 for s in neg_before))
    false_after = int(sum(s >= threshold for s in neg_after))
    use_personal = recall_after > recall_before or (recall_after == recall_before and false_after < false_before)

    progress(0.93, "Learning how the recognizer hears you…")
    aliases = learn_aliases(stt, clips, phrase)
    save_aliases(name, aliases)

    report = {
        "held_out_clips": len(held),
        "detected_before": recall_before,
        "detected_after": recall_after,
        "validation_negatives": len(val_negs),
        "false_before": false_before,
        "false_after": false_after,
        "threshold": round(threshold, 3),
        "used_personal_model": bool(use_personal),
        "aliases": aliases,
        "seconds": round(time.monotonic() - started, 1),
    }
    out = personal_model_path(name)
    if use_personal:
        meta = dict(base_meta)
        meta.update({"phrase": phrase, "personalized": time.strftime("%Y-%m-%d %H:%M"),
                     "user_samples": len(clips), "threshold": round(threshold, 3), "report": report,
                     "base": base_path.name})
        w1, b1, w2, b2 = params
        np.savez_compressed(out, w1=w1, b1=b1, w2=w2, b2=b2, mu=mu, sigma=sigma,
                            meta=np.array(json.dumps(meta)))
        report["path"] = str(out)
    else:
        out.unlink(missing_ok=True)
        report["path"] = str(base_path)
    progress(1.0, "Done.")
    return report


def aliases_path(name: str) -> Path:
    return custom_dir() / f"{name}_aliases.json"


def save_aliases(name: str, aliases: Sequence[str]) -> None:
    custom_dir().mkdir(parents=True, exist_ok=True)
    aliases_path(name).write_text(json.dumps(sorted(set(aliases))), encoding="utf-8")


def load_aliases(name: str) -> List[str]:
    try:
        return list(json.loads(aliases_path(name).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def save_samples(name: str, positives: Sequence[np.ndarray], background: Sequence[np.ndarray],
                 others: Sequence[np.ndarray] = ()) -> Path:
    """Keep the user's recordings locally (git-ignored) so a full `train` can use them later."""
    base = custom_dir() / f"{name}_samples"
    stamp = int(time.time() * 1000)
    (base / "positive").mkdir(parents=True, exist_ok=True)
    (base / "negative").mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(positives):
        np.save(base / "positive" / f"you_{stamp}_{i:02d}.npy", np.asarray(clip, np.int16))
    if background:
        np.save(base / "negative" / f"room_{stamp}.npy", np.concatenate([np.asarray(b, np.int16) for b in background]))
    for i, clip in enumerate(others):
        np.save(base / "negative" / f"said_{stamp}_{i:02d}.npy", np.asarray(clip, np.int16))
    return base


OTHER_SENTENCES = (
    "Hi there",
    "Good morning",
    "Open the browser",
    "Hello, how are you?",
    "Sounds good to me",
)


class EnrollSession:
    """Collects the user's recordings from the app's live mic stream.

    Three steps: the wake phrase N times, a few ordinary sentences (same voice,
    not the phrase), then a few seconds of the room. feed() is called from the
    voice thread; the wizard polls the public fields from the Tk thread.
    """

    def __init__(self, target: int = 10, background_sec: float = 15.0) -> None:
        self.target = target
        self.background_sec = background_sec
        self.mode = "idle"
        self.clips: List[np.ndarray] = []
        self.others: List[np.ndarray] = []
        self.background: List[np.ndarray] = []
        self.level = 0.0
        self.message = ""
        self._kind = "phrases"
        self._max_sec = 2.6
        self._recorder = CommandRecorder()
        self._lock = threading.Lock()

    def start_phrases(self) -> None:
        self._start("phrases", 2.6)

    def start_sentences(self) -> None:
        self._start("sentences", 4.5)

    def _start(self, kind: str, max_sec: float) -> None:
        with self._lock:
            self._recorder.reset(start_timeout=8.0)
            self._kind, self._max_sec = kind, max_sec
            self.message = ""
            self.mode = kind

    def start_background(self) -> None:
        with self._lock:
            self.background = []
            self.mode = "background"

    def stop(self) -> None:
        with self._lock:
            self.mode = "idle"

    @property
    def background_progress(self) -> float:
        got = sum(len(b) for b in self.background) / SAMPLE_RATE
        return min(1.0, got / self.background_sec)

    def feed(self, chunk: np.ndarray) -> None:
        self.level = max(self.level * 0.7, rms_level(chunk))
        with self._lock:
            if self.mode == "background":
                self.background.append(chunk.copy())
                if self.background_progress >= 1.0:
                    self.mode = "background_done"
                return
            if self.mode not in ("phrases", "sentences"):
                return
            status = self._recorder.feed(chunk)
            if status == "done":
                from train_wake_word import _trim

                clip = _trim(self._recorder.audio())
                seconds = len(clip) / SAMPLE_RATE
                if seconds < 0.35:
                    self.message = "Too short. Say the whole thing."
                elif seconds > self._max_sec:
                    self.message = "Too long. Say only what is shown."
                else:
                    bucket = self.clips if self._kind == "phrases" else self.others
                    bucket.append(clip)
                    self.message = "Got it ✓"
                    goal = self.target if self._kind == "phrases" else len(OTHER_SENTENCES)
                    if len(bucket) >= goal:
                        self.mode = f"{self._kind}_done"
                        return
                self._recorder.reset(start_timeout=8.0)
            elif status == "timeout":
                self.message = "I didn't hear anything."
                self._recorder.reset(start_timeout=8.0)

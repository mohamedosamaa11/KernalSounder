"""
Train a custom wake word ("Hi Sounder" by default) on this machine. No GPU, no cloud.

How it works
------------
1. Synthesize thousands of positive clips with Piper's multi-speaker voice
   (904 speakers) and negative clips (normal sentences, commands and
   sound-alikes such as "hi sandra", "hi thunder").
2. Optionally add YOUR voice: `record` saves real samples from the mic.
3. Turn every clip into openWakeWord speech embeddings (the same frozen
   feature models the app runs) and train a tiny MLP head in numpy.
4. Save models/custom/<name>.npz — wake_word.py loads it like any .onnx model.

Usage
-----
    pip install -r requirements-train.txt
    python train_wake_word.py record --count 15          # optional, your voice
    python train_wake_word.py record --negative --seconds 60
    python train_wake_word.py train                      # ~10-15 min on a laptop
    python train_wake_word.py test                       # live mic score meter
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from config import MODELS_DIR, SAMPLE_RATE

CUSTOM_DIR = MODELS_DIR / "custom"
PIPER_DIR = MODELS_DIR / "piper"
PIPER_VOICE = "en_US-libritts_r-medium"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/"

FRAMES = 16
EMB_HOP_SEC = 0.08
EMB_CONTEXT_SEC = 0.76

SOUNDALIKES = [
    "hi sandra", "hi wonder", "hi thunder", "hi founder", "hi sunday", "hi soldier",
    "hi sound", "hi sonny", "hi sander", "high sun", "hi summer", "hi senator", "hi saunter",
    "hi santa", "hi sender", "hi cinder", "hi sounding", "hi sunder land", "sounder",
    "sound good", "hi there", "hi", "hello", "hey sounds good", "my founder", "the sounder",
    "hi wanda", "hi lender", "hi slender", "hi alexander", "why sound", "hi thunderbird",
    "hey jarvis", "alexa", "okay google", "hey siri", "hi kernel", "hi son", "i sound older",
]

COMMANDS = [
    "list files", "create file called notes dot txt", "create folder called projects",
    "rename old dot txt to new dot txt", "delete file report dot txt", "open the browser",
    "open firefox", "open terminal", "close chrome", "show processes", "show cpu usage",
    "what apps are available", "confirm", "cancel", "help", "open notepad", "open calculator",
]

_SUBJECTS = ["I", "we", "my friend", "the teacher", "our team", "she", "he", "they", "the system",
             "my brother", "the doctor", "everyone", "the manager", "this computer", "the student"]
_VERBS = ["finished", "started", "opened", "checked", "sent", "found", "moved", "watched",
          "fixed", "planned", "cleaned", "bought", "called", "visited", "wrote"]
_OBJECTS = ["the report", "a new folder", "the meeting notes", "the project", "my email",
            "the window", "the kitchen", "a short video", "the lecture", "the homework",
            "the server logs", "a long book", "the website", "the car", "the documents"]
_TAILS = ["this morning", "yesterday", "after lunch", "before the exam", "on the weekend",
          "in the office", "at home", "last night", "right now", "very quickly", "", "", ""]


def _sentences(rng: random.Random, n: int) -> List[str]:
    out = []
    for _ in range(n):
        tail = rng.choice(_TAILS)
        out.append(f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {tail}".strip())
    return out


def _noise(n: int, seed: int, color: str = "white") -> np.ndarray:
    idx = np.arange(n, dtype=np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) % (1 << 64))
    x = idx * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    x ^= x >> np.uint64(29)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(32)
    white = (x % np.uint64(20001)).astype(np.float64) / 10000.0 - 1.0
    if color == "brown":
        brown = np.cumsum(white)
        brown -= np.convolve(brown, np.ones(400) / 400, mode="same")
        return brown / (np.max(np.abs(brown)) + 1e-9)
    return white


def _augment(clip: np.ndarray, rng: random.Random, seed: int, pad_before=(1.6, 2.4),
             pad_after=(0.25, 0.7)) -> Tuple[np.ndarray, float]:
    """Speed/gain/noise augmentation. Returns (audio, speech_end_seconds)."""
    x = clip.astype(np.float64)
    speed = rng.uniform(0.9, 1.12)
    if abs(speed - 1.0) > 0.01:
        n = int(len(x) / speed)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak * (10 ** (rng.uniform(-20, -2) / 20) * 32767)
    before = int(rng.uniform(*pad_before) * SAMPLE_RATE)
    after = int(rng.uniform(*pad_after) * SAMPLE_RATE)
    total = before + len(x) + after
    noise = _noise(total, seed, rng.choice(["white", "brown", "brown"]))
    speech_rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    snr_db = rng.uniform(6, 35)
    noise *= speech_rms / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
    out = noise
    out[before:before + len(x)] += x
    return np.clip(out, -32768, 32767).astype(np.int16), (before + len(x)) / SAMPLE_RATE


def _trim(clip: np.ndarray) -> np.ndarray:
    a = np.abs(clip.astype(np.float32))
    if a.size == 0:
        return clip
    thr = max(200.0, 0.05 * float(a.max()))
    loud = np.where(a > thr)[0]
    if loud.size == 0:
        return clip
    s = max(0, loud[0] - 800)
    e = min(len(clip), loud[-1] + 800)
    return clip[s:e]


def _to_16k(data: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return data.astype(np.int16)
    n = int(len(data) * SAMPLE_RATE / rate)
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.int16)


def _ensure_piper() -> Path:
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        dest = PIPER_DIR / f"{PIPER_VOICE}{suffix}"
        if not dest.is_file():
            print(f"  downloading {dest.name} ...")
            urllib.request.urlretrieve(PIPER_URL + dest.name, dest)
    return PIPER_DIR / f"{PIPER_VOICE}.onnx"


class PiperSynth:
    def __init__(self) -> None:
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(_ensure_piper()))
        self.speakers = self.voice.config.num_speakers
        self.rate = self.voice.config.sample_rate

    def say(self, text: str, speaker: int, rng: random.Random) -> np.ndarray:
        from piper.config import SynthesisConfig

        cfg = SynthesisConfig(
            speaker_id=speaker,
            length_scale=rng.uniform(0.8, 1.3),
            noise_scale=rng.uniform(0.5, 0.9),
            noise_w_scale=rng.uniform(0.6, 1.0),
        )
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, syn_config=cfg)]
        data = np.concatenate(chunks) if chunks else np.zeros(1, np.int16)
        return _trim(_to_16k(data, self.rate))


class Embedder:
    """Batch version of wake_word.WakeWordDetector's feature pipeline."""

    def __init__(self) -> None:
        from wake_word import WakeWordDetector

        self.det = WakeWordDetector.__new__(WakeWordDetector)
        import onnxruntime as ort
        from config import OWW_DIR

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.det._mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"), sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self.det._emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"), sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self.det._mel_in = self.det._mel.get_inputs()[0].name
        self.det._emb_in = self.det._emb.get_inputs()[0].name

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        return self.det._embed_clip(audio)


def _window_end_times(n_emb: int) -> np.ndarray:
    return np.arange(n_emb) * EMB_HOP_SEC + EMB_CONTEXT_SEC


def _windows(emb: np.ndarray) -> np.ndarray:
    if emb.shape[0] < FRAMES:
        return np.zeros((0, FRAMES, 96), np.float32)
    idx = np.arange(FRAMES)[None, :] + np.arange(emb.shape[0] - FRAMES + 1)[:, None]
    return emb[idx].astype(np.float32)


def _positive_windows(emb: np.ndarray, speech_end: float) -> Tuple[np.ndarray, np.ndarray]:
    wins = _windows(emb)
    ends = _window_end_times(emb.shape[0])[FRAMES - 1:]
    pos = wins[(ends >= speech_end - 0.08) & (ends <= speech_end + 0.45)]
    early = wins[ends <= speech_end - 0.45]
    return pos, early


def _sample_dir(name: str, negative: bool = False) -> Path:
    return CUSTOM_DIR / f"{name}_samples" / ("negative" if negative else "positive")


def cmd_record(args) -> int:
    import sounddevice as sd

    folder = _sample_dir(args.name, args.negative)
    folder.mkdir(parents=True, exist_ok=True)
    if args.negative:
        print(f"Talk normally (read anything aloud) for {args.seconds} seconds. Recording starts in 2 s ...")
        time.sleep(2)
        audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        path = folder / f"talk_{int(time.time())}.npy"
        np.save(path, audio[:, 0])
        print("Saved", path)
        return 0
    print(f"Say '{args.phrase}' once each time you see SPEAK. Vary tone, speed and distance.")
    for i in range(args.count):
        input(f"[{i + 1}/{args.count}] press Enter, then speak ...")
        print("   SPEAK")
        audio = sd.rec(int(2.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        clip = _trim(audio[:, 0])
        if np.max(np.abs(clip)) < 800:
            print("   too quiet, skipped")
            continue
        np.save(folder / f"you_{int(time.time() * 1000)}.npy", clip)
    print("Saved samples in", folder)
    return 0


def _load_user_samples(name: str, negative: bool) -> List[np.ndarray]:
    folder = _sample_dir(name, negative)
    return [np.load(p).astype(np.int16) for p in sorted(folder.glob("*.npy"))] if folder.is_dir() else []


def _train_mlp(x: np.ndarray, y: np.ndarray, w: np.ndarray, hidden: int, epochs: int, seed: int):
    rng = random.Random(seed)
    n, d = x.shape
    scale1 = np.sqrt(2.0 / d)
    noise = _noise(d * hidden, seed).reshape(d, hidden)
    w1 = (noise * scale1 * np.sqrt(3)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    w2 = (_noise(hidden, seed + 1) * np.sqrt(3.0 / hidden)).astype(np.float32).reshape(hidden, 1)
    b2 = np.zeros(1, np.float32)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps, l2 = 1e-3, 0.9, 0.999, 1e-8, 1e-4
    step = 0
    order = list(range(n))
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for start in range(0, n, 512):
            idx = np.array(order[start:start + 512])
            xb, yb, wb = x[idx], y[idx], w[idx]
            h = xb @ w1 + b1
            a = np.maximum(h, 0)
            logits = (a @ w2 + b2)[:, 0]
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            loss = -np.sum(wb * (yb * np.log(p + 1e-7) + (1 - yb) * np.log(1 - p + 1e-7))) / wb.sum()
            total += loss * len(idx)
            g = (wb * (p - yb) / wb.sum())[:, None].astype(np.float32)
            gw2 = a.T @ g + l2 * w2
            gb2 = g.sum(0)
            ga = g @ w2.T
            gh = ga * (h > 0)
            gw1 = xb.T @ gh + l2 * w1
            gb1 = gh.sum(0)
            step += 1
            for i, grad in enumerate((gw1, gb1, gw2, gb2)):
                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * grad * grad
                mh = m[i] / (1 - beta1 ** step)
                vh = v[i] / (1 - beta2 ** step)
                params[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        if epoch % 5 == 4 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss {total / n:.4f}")
    return params


def _predict(params, mu, sigma, x: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2 = params
    z = (x - mu) / sigma
    logits = (np.maximum(z @ w1 + b1, 0) @ w2 + b2)[:, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def cmd_train(args) -> int:
    rng = random.Random(args.seed)
    synth = PiperSynth()
    embedder = Embedder()
    speakers = list(range(synth.speakers))
    rng.shuffle(speakers)
    val_speakers = set(speakers[:120])
    train_speakers = speakers[120:]
    phrase_variants = [args.phrase, args.phrase + ".", args.phrase + "!",
                       args.phrase.replace(" ", ", ", 1), args.phrase.lower()]

    pos_x: List[np.ndarray] = []
    neg_x: List[np.ndarray] = []
    neg_w: List[np.ndarray] = []
    val_pos: List[np.ndarray] = []
    val_neg: List[np.ndarray] = []
    neg_seconds = 0.0
    seed = 1000
    t0 = time.monotonic()

    def add_positive(clip: np.ndarray, copies: int, val: bool) -> None:
        nonlocal seed
        for _ in range(copies):
            seed += 1
            audio, end = _augment(clip, rng, seed)
            pos, early = _positive_windows(embedder.embeddings(audio), end)
            (val_pos if val else pos_x).append(pos)
            if not val and early.size:
                neg_x.append(early)
                neg_w.append(np.full(len(early), 0.5, np.float32))

    def add_negative(clip: np.ndarray, val: bool, weight: float = 1.0) -> None:
        nonlocal seed, neg_seconds
        seed += 1
        audio, _ = _augment(clip, rng, seed, pad_before=(1.2, 2.0), pad_after=(0.2, 0.6))
        wins = _windows(embedder.embeddings(audio))[::2]
        if val:
            val_neg.append(wins)
            neg_seconds += len(audio) / SAMPLE_RATE
        else:
            neg_x.append(wins)
            neg_w.append(np.full(len(wins), weight, np.float32))

    print(f"Synthesizing {args.positives} positive clips of '{args.phrase}' ...")
    for i in range(args.positives):
        val = i % 10 == 0
        spk = rng.choice(list(val_speakers) if val else train_speakers)
        add_positive(synth.say(rng.choice(phrase_variants), spk, rng), 1, val)
        if i % 250 == 249:
            print(f"  {i + 1} ({time.monotonic() - t0:.0f}s)")

    negatives = [(t, 2.0) for t in SOUNDALIKES] * max(1, args.negatives // (3 * len(SOUNDALIKES)))
    negatives += [(t, 1.0) for t in COMMANDS] * max(1, args.negatives // (6 * len(COMMANDS)))
    negatives += [(t, 1.0) for t in _sentences(rng, max(0, args.negatives - len(negatives)))]
    rng.shuffle(negatives)
    print(f"Synthesizing {len(negatives)} negative clips ...")
    for i, (text, weight) in enumerate(negatives):
        val = i % 10 == 0
        spk = rng.choice(list(val_speakers) if val else train_speakers)
        add_negative(synth.say(text, spk, rng), val, weight)
        if i % 500 == 499:
            print(f"  {i + 1} ({time.monotonic() - t0:.0f}s)")

    for k in range(args.noise_clips):
        seed += 1
        audio = (_noise(int(SAMPLE_RATE * 3), seed, "brown" if k % 2 else "white")
                 * rng.uniform(50, 3000)).astype(np.int16)
        neg_x.append(_windows(embedder.embeddings(audio))[::2])
        neg_w.append(np.ones(len(neg_x[-1]), np.float32))

    user_pos = _load_user_samples(args.name, False)
    user_neg = _load_user_samples(args.name, True)
    if user_pos:
        print(f"Adding {len(user_pos)} recordings of your voice (x{args.user_copies} augmented) ...")
        for i, clip in enumerate(user_pos):
            add_positive(clip, args.user_copies, val=False)
            add_positive(clip, 1, val=(i % 4 == 0))
    for clip in user_neg:
        step = SAMPLE_RATE * 3
        for s in range(0, max(1, len(clip) - step), step // 2):
            add_negative(clip[s:s + step], val=False, weight=2.0)

    xp = np.concatenate(pos_x).reshape(-1, FRAMES * 96)
    xn = np.concatenate(neg_x).reshape(-1, FRAMES * 96)
    wn = np.concatenate(neg_w)
    print(f"Training windows: {len(xp)} positive, {len(xn)} negative "
          f"({time.monotonic() - t0:.0f}s so far)")

    x = np.concatenate([xp, xn]).astype(np.float32)
    y = np.concatenate([np.ones(len(xp), np.float32), np.zeros(len(xn), np.float32)])
    w = np.concatenate([np.full(len(xp), len(xn) / max(1, len(xp)) * 0.5, np.float32), wn])
    mu = x.mean(0).astype(np.float32)
    sigma = (x.std(0) + 1e-3).astype(np.float32)
    params = _train_mlp((x - mu) / sigma, y, w, args.hidden, args.epochs, args.seed)

    vp = np.concatenate(val_pos).reshape(-1, FRAMES * 96)
    vn = np.concatenate(val_neg).reshape(-1, FRAMES * 96)
    sp, sn = _predict(params, mu, sigma, vp), _predict(params, mu, sigma, vn)
    clip_hits = [float(_predict(params, mu, sigma, c.reshape(-1, FRAMES * 96)).max()) if len(c) else 0.0
                 for c in val_pos]
    report = {}
    for thr in (0.3, 0.5, 0.7, 0.9):
        recall = float(np.mean(np.array(clip_hits) >= thr))
        fp = int((sn >= thr).sum())
        report[str(thr)] = {"clip_recall": round(recall, 3), "false_windows": fp,
                            "false_per_hour": round(fp / max(neg_seconds, 1) * 3600 / 4, 1)}
        print(f"  threshold {thr:.1f}: held-out speakers detected {recall * 100:5.1f}%  "
              f"false windows {fp} in {neg_seconds / 60:.1f} min of other speech")

    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    out = CUSTOM_DIR / f"{args.name}.npz"
    w1, b1, w2, b2 = params
    meta = {"phrase": args.phrase, "frames": FRAMES, "trained": time.strftime("%Y-%m-%d"),
            "positives": args.positives, "user_samples": len(user_pos), "validation": report}
    np.savez_compressed(out, w1=w1, b1=b1, w2=w2, b2=b2, mu=mu, sigma=sigma,
                        meta=np.array(json.dumps(meta)))
    print(f"Saved {out}  ({out.stat().st_size / 1024:.0f} KB, {time.monotonic() - t0:.0f}s total)")
    return 0


def cmd_test(args) -> int:
    from audio import MicStream
    from wake_word import WakeWordDetector

    det = WakeWordDetector(args.model, threshold=args.threshold)
    mic = MicStream()
    mic.start()
    print(f"Say '{args.phrase}'. Ctrl+C to stop.")
    try:
        while True:
            chunk = mic.read()
            if chunk is None:
                continue
            score = det.process(chunk)
            bar = "#" * int(score * 40)
            print(f"\r{score:5.2f} {bar:<40}{'  <-- WAKE' if score >= det.threshold else '          '}",
                  end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        mic.stop()
    return 0


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="hi_sounder")
    parser.add_argument("--phrase", default="Hi Sounder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record your own samples from the mic")
    rec.add_argument("--count", type=int, default=15)
    rec.add_argument("--negative", action="store_true", help="record normal talking instead")
    rec.add_argument("--seconds", type=int, default=60)
    tr = sub.add_parser("train", help="synthesize data and train the model")
    tr.add_argument("--positives", type=int, default=2000)
    tr.add_argument("--negatives", type=int, default=3000)
    tr.add_argument("--noise-clips", type=int, default=200)
    tr.add_argument("--user-copies", type=int, default=25)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--seed", type=int, default=11)
    te = sub.add_parser("test", help="live score meter from the mic")
    te.add_argument("--model", default="hi_sounder.npz")
    te.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return {"record": cmd_record, "train": cmd_train, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

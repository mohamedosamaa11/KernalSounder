"""
KernalSounder — always-on wake word.

Runs openWakeWord's three small ONNX models (about 4 MB in total) with
onnxruntime only, so the heavy openwakeword package (scipy, scikit-learn)
is not needed:

    80 ms audio -> melspectrogram.onnx -> embedding_model.onnx -> <wake>.onnx

Each 80 ms step costs a few milliseconds on one CPU core.
Any openWakeWord-compatible .onnx model works, and so does a .npz head
trained locally by train_wake_word.py (the default "Hey Nova" model).
"""

from __future__ import annotations

import lean_runtime  # noqa: F401

import json
from pathlib import Path
from typing import Optional

import numpy as np

from config import CHUNK_SAMPLES, OWW_DIR, OWW_FEATURE_MODELS, MODELS_DIR

_MEL_CONTEXT = 160 * 3
_EMB_WINDOW = 76
_EMB_STEP = 8
_MEL_KEEP = 10 * 97
_FEAT_KEEP = 120
_WARMUP_STEPS = 5


def _pseudo_noise(n: int, amplitude: int = 1000) -> np.ndarray:
    """Deterministic white-ish noise without numpy.random (blocked on some locked-down PCs)."""
    state = (np.arange(n, dtype=np.uint64) * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
    state ^= state >> np.uint64(29)
    state *= np.uint64(0xBF58476D1CE4E5B9)
    state ^= state >> np.uint64(32)
    return ((state % np.uint64(2 * amplitude)).astype(np.int64) - amplitude).astype(np.int16)


def find_wake_model(name_or_path: str) -> Optional[Path]:
    candidate = Path(name_or_path).expanduser()
    search = [candidate]
    if not candidate.is_absolute():
        search += [MODELS_DIR / candidate, OWW_DIR / candidate, MODELS_DIR / "custom" / candidate]
    for path in search:
        if path.is_file():
            return path.resolve()
    return None


def missing_feature_models() -> list:
    return [name for name in OWW_FEATURE_MODELS if not (OWW_DIR / name).is_file()]


class WakeWordDetector:
    def __init__(self, model: str, threshold: Optional[float] = None) -> None:
        import onnxruntime as ort

        missing = missing_feature_models()
        if missing:
            raise FileNotFoundError(
                f"Missing wake-word feature models: {', '.join(missing)}. Run download_models.py."
            )
        model_path = find_wake_model(model)
        if model_path is None:
            raise FileNotFoundError(f"Wake-word model not found: {model}. Run download_models.py.")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        providers = ["CPUExecutionProvider"]

        def session(path: Path):
            return ort.InferenceSession(str(path), sess_options=opts, providers=providers)

        self._mel = session(OWW_DIR / "melspectrogram.onnx")
        self._emb = session(OWW_DIR / "embedding_model.onnx")
        self._mel_in = self._mel.get_inputs()[0].name
        self._emb_in = self._emb.get_inputs()[0].name
        self._head = None
        self._ww = None
        self.phrase = None
        self.meta: dict = {}
        if model_path.suffix.lower() == ".npz":
            with np.load(model_path) as data:
                self._head = {k: data[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2", "mu", "sigma")}
                meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
            self.meta = meta
            self._n_frames = int(meta.get("frames", 16))
            self.phrase = meta.get("phrase")
        else:
            self._ww = session(model_path)
            self._ww_in = self._ww.get_inputs()[0].name
            frames = self._ww.get_inputs()[0].shape[1]
            self._n_frames = frames if isinstance(frames, int) and frames > 0 else 16

        self.model_path = model_path
        self.name = model_path.stem
        self.threshold = float(threshold) if threshold is not None else float(self.meta.get("threshold", 0.5))

        self._initial_features = self._embed_clip(_pseudo_noise(16000 * 4))
        self.reset()

    def _melspec(self, samples: np.ndarray) -> np.ndarray:
        x = samples.astype(np.float32)[None, :]
        out = self._mel.run(None, {self._mel_in: x})[0]
        return np.squeeze(out).reshape(-1, 32) / 10.0 + 2.0

    def _embed(self, windows: np.ndarray) -> np.ndarray:
        x = windows.astype(np.float32)[..., None]
        out = self._emb.run(None, {self._emb_in: x})[0]
        return out.reshape(-1, 96)

    def _embed_clip(self, samples: np.ndarray) -> np.ndarray:
        spec = self._melspec(samples)
        windows = [spec[i:i + _EMB_WINDOW] for i in range(0, spec.shape[0] - _EMB_WINDOW + 1, _EMB_STEP)]
        return self._embed(np.stack(windows))

    def reset(self) -> None:
        self._raw = np.zeros(CHUNK_SAMPLES + _MEL_CONTEXT, dtype=np.int16)
        self._pending = np.zeros(0, dtype=np.int16)
        self._mel_buf = np.ones((_EMB_WINDOW, 32), dtype=np.float32)
        self._features = self._initial_features.copy()
        self._steps = 0

    def _step(self, chunk: np.ndarray) -> float:
        self._raw = np.concatenate((self._raw, chunk))[-(CHUNK_SAMPLES + _MEL_CONTEXT):]
        self._mel_buf = np.vstack((self._mel_buf, self._melspec(self._raw)))[-_MEL_KEEP:]
        embedding = self._embed(self._mel_buf[-_EMB_WINDOW:][None, ...])
        self._features = np.vstack((self._features, embedding))[-_FEAT_KEEP:]
        self._steps += 1
        if self._steps < _WARMUP_STEPS:
            return 0.0
        x = self._features[-self._n_frames:][None, ...].astype(np.float32)
        if self._head is not None:
            h = self._head
            z = (x.reshape(1, -1) - h["mu"]) / h["sigma"]
            logit = float((np.maximum(z @ h["w1"] + h["b1"], 0.0) @ h["w2"] + h["b2"])[0, 0])
            return 1.0 / (1.0 + float(np.exp(-max(-30.0, min(30.0, logit)))))
        out = self._ww.run(None, {self._ww_in: x})[0]
        return float(np.ravel(out)[0])

    def process(self, chunk: np.ndarray) -> float:
        """Feed any amount of int16 audio; returns the best score seen."""
        self._pending = np.concatenate((self._pending, chunk.astype(np.int16)))
        best = 0.0
        while self._pending.shape[0] >= CHUNK_SAMPLES:
            step, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]
            best = max(best, self._step(step))
        return best

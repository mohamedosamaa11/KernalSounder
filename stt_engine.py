"""
KernalSounder — speech-to-text.

Default backend: faster-whisper (CTranslate2, int8) running offline.
tiny.en is ~75 MB on disk, ~150 MB in RAM and transcribes a short command in
well under a second on a laptop CPU. A vocabulary prompt biases it toward the
command words, which fixes most mishearings ("crate file" -> "create file").

Optional backend: Google Web Speech through SpeechRecognition (needs internet).
Both receive the same 16 kHz int16 audio from audio.MicStream.
"""

from __future__ import annotations

import lean_runtime  # noqa: F401

import difflib
import re
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import (
    SAMPLE_RATE,
    STT_BACKEND,
    STT_NETWORK_ERROR,
    STT_VOCAB_PROMPT,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DIR,
    WHISPER_MODEL,
    WHISPER_THREADS,
)


class STTNetworkError(Exception):
    """Raised when the cloud recognizer cannot be reached."""


_PROMPT_WORDS = re.sub(r"[^a-z ]", " ", STT_VOCAB_PROMPT.lower()).split()


def _looks_like_prompt_echo(text: str, prompt: str = STT_VOCAB_PROMPT) -> bool:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    if len(words) < 6:
        return False
    joined = " ".join(words)
    prompt_words = " ".join(re.sub(r"[^a-z ]", " ", prompt.lower()).split())
    return joined in prompt_words or joined in " ".join(_PROMPT_WORDS)


def _skip_pyav() -> None:
    """faster-whisper imports PyAV (FFmpeg) only to decode audio files; we pass raw
    arrays, so a placeholder module saves ~10 MB of RAM. Real PyAV wins if loaded."""
    import sys
    import types

    if "av" in sys.modules:
        return
    placeholder = types.ModuleType("av")

    def __getattr__(name):
        raise ImportError("PyAV is not loaded; KernalSounder only transcribes raw audio arrays.")

    placeholder.__getattr__ = __getattr__
    sys.modules["av"] = placeholder


class STTEngine:
    def __init__(self, backend: str = STT_BACKEND, model: str = WHISPER_MODEL,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self.backend = backend
        self.model_name = model
        self._on_status = on_status
        self._model = None
        self._load_lock = threading.Lock()
        self.ready = threading.Event()
        self.last_error: Optional[str] = None
        self._prompt = STT_VOCAB_PROMPT
        self._app_names: List[str] = []

    def set_app_names(self, names: List[str]) -> None:
        """Bias recognition toward the installed apps and the personal vocabulary."""
        self._app_names = list(names)
        self._prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        from vocabulary import load_vocabulary

        apps = sorted({n for n in self._app_names
                       if len(n) <= 24 and not re.search(r"[\d()&]", n)}, key=len)
        words = ", ".join(load_vocabulary())
        tail = (f"Names: {words}. " if words else "") + STT_VOCAB_PROMPT
        tokenizer = getattr(self._model, "hf_tokenizer", None)

        def size(text: str) -> int:
            if tokenizer is None:
                return len(text) // 4
            return len(tokenizer.encode(" " + text).ids)

        budget = 210
        while apps and size("Apps: " + ", ".join(apps) + ". " + tail) > budget:
            apps.pop()
        return ("Apps: " + ", ".join(apps) + ". " + tail) if apps else tail

    def _status(self, text: str) -> None:
        if self._on_status:
            self._on_status(text)

    def describe(self) -> str:
        if self.backend == "whisper":
            return f"offline whisper {self.model_name}"
        return "Google Web Speech (online)"

    def load(self) -> bool:
        """Load and warm up the model. Safe to call from a background thread."""
        with self._load_lock:
            if self.ready.is_set():
                return True
            if self.backend == "whisper":
                try:
                    self._load_whisper()
                except Exception as exc:
                    self.last_error = f"Offline speech model failed to load: {exc}"
                    if self._google_available():
                        self.backend = "google"
                        self.last_error += " Falling back to Google speech."
                    else:
                        return False
            elif not self._google_available():
                self.last_error = "SpeechRecognition is not installed."
                return False
            self.ready.set()
            return True

    def _load_whisper(self) -> None:
        _skip_pyav()
        from faster_whisper import WhisperModel

        local = WHISPER_DIR / self.model_name
        source = str(local) if (local / "model.bin").is_file() else self.model_name
        self._status(f"Loading speech model ({self.model_name})…")
        self._model = WhisperModel(
            source,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_THREADS,
            num_workers=1,
            download_root=str(WHISPER_DIR),
        )
        silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        list(self._model.transcribe(silence, language="en", beam_size=1, without_timestamps=True)[0])

    @staticmethod
    def _google_available() -> bool:
        try:
            import speech_recognition  # noqa: F401
            return True
        except Exception:
            return False

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE // 4:
            return ""
        if not self.ready.is_set() and not self.load():
            raise RuntimeError(self.last_error or "Speech recognition is unavailable.")
        if self.backend == "whisper":
            return self._transcribe_whisper(audio)
        return self._transcribe_google(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        samples = audio.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            samples,
            language="en",
            beam_size=WHISPER_BEAM_SIZE,
            temperature=0.0,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.75 and seg.avg_logprob < -0.8:
                continue
            parts.append(seg.text)
        text = " ".join(parts).strip()
        if _looks_like_prompt_echo(text, self._prompt):
            return ""
        return text

    def verify_keyword(self, audio: np.ndarray, phrase: str) -> Tuple[Optional[bool], str]:
        """Second-stage wake check. Returns (True/False, transcript) or (None, "") if it can't check.

        No vocabulary prompt here on purpose: a prompt makes the model "hear" the
        wake word in look-alikes such as "hi founder".
        """
        if self.backend != "whisper" or not self.ready.is_set() or self._model is None:
            return None, ""
        keyword = re.sub(r"[^a-z]", "", phrase.lower().split()[-1])
        samples = audio.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            samples, language="en", beam_size=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        for word in re.findall(r"[a-z]+", text.lower()):
            if word == keyword:
                return True, text
            if word[:1] == keyword[:1] and difflib.SequenceMatcher(None, word, keyword).ratio() >= 0.72:
                return True, text
        return False, text

    def _transcribe_google(self, audio: np.ndarray) -> str:
        import speech_recognition as sr

        data = sr.AudioData(audio.astype(np.int16).tobytes(), SAMPLE_RATE, 2)
        try:
            return (sr.Recognizer().recognize_google(data) or "").strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            self.last_error = STT_NETWORK_ERROR
            raise STTNetworkError(STT_NETWORK_ERROR) from exc

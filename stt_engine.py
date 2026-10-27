"""
KernalSounder — Speech-to-Text.

Engine: SpeechRecognition + Google Web Speech API
(recognizer.recognize_google). No local acoustic model is loaded, so the
4 GB VM keeps that RAM. Recognition runs only on a background thread.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from config import (
    STT_AMBIENT_DURATION_SEC,
    STT_NETWORK_ERROR,
    STT_PHRASE_LIMIT_SEC,
)


class STTNetworkError(Exception):
    """Raised when Google Speech cannot be reached (offline / API error)."""


class STTEngine:
    def __init__(self, on_partial: Optional[Callable[[str], None]] = None) -> None:
        self._on_partial = on_partial
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._recognizer = None
        self._microphone = None
        self._source = None
        self.last_error: Optional[str] = None

    def available(self) -> bool:
        try:
            import speech_recognition as sr  # noqa: F401
        except Exception as exc:
            self.last_error = f"SpeechRecognition is not installed: {exc}"
            return False
        try:
            import pyaudio  # noqa: F401
        except Exception as exc:
            self.last_error = (
                f"PyAudio is missing ({exc}). Install portaudio19-dev and rerun setup.sh."
            )
            return False
        return True

    def start(self) -> None:
        """
        Open the microphone and calibrate for VM background static.
        Must be called from a worker thread, never the Tk main loop.
        """
        if not self.available():
            raise RuntimeError(self.last_error or "STT is not available.")

        import speech_recognition as sr

        with self._lock:
            self._stop.clear()
            if self._source is not None:
                return
            self._recognizer = sr.Recognizer()
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8
            self._microphone = sr.Microphone()
            self._source = self._microphone.__enter__()
            if self._on_partial:
                self._on_partial("Calibrating microphone for ambient noise…")
            self._recognizer.adjust_for_ambient_noise(
                self._source, duration=STT_AMBIENT_DURATION_SEC
            )
            if self._on_partial:
                self._on_partial("Listening…")

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._close_microphone()

    def _close_microphone(self) -> None:
        if self._microphone is not None and self._source is not None:
            try:
                self._microphone.__exit__(None, None, None)
            except Exception:
                pass
        self._source = None
        self._microphone = None

    def listen_utterance(self, timeout: float = 6.0) -> str:
        """
        Capture one phrase and send it to Google. Blocks the caller thread
        only — the Tkinter GUI keeps pumping events on the main thread.
        """
        import speech_recognition as sr

        if self._stop.is_set():
            return ""

        with self._lock:
            recognizer = self._recognizer
            source = self._source
        if recognizer is None or source is None:
            return ""

        try:
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=STT_PHRASE_LIMIT_SEC,
            )
        except sr.WaitTimeoutError:
            return ""
        except OSError as exc:
            self.last_error = f"Microphone error: {exc}"
            raise

        if self._stop.is_set():
            return ""

        try:
            text = recognizer.recognize_google(audio)
            return (text or "").strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            self.last_error = STT_NETWORK_ERROR
            raise STTNetworkError(STT_NETWORK_ERROR) from exc
        except Exception as exc:
            message = str(exc).lower()
            if any(token in message for token in ("connection", "network", "timed out", "unreachable", "name or service")):
                self.last_error = STT_NETWORK_ERROR
                raise STTNetworkError(STT_NETWORK_ERROR) from exc
            raise

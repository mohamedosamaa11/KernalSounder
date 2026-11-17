"""
KernalSounder — text-to-speech.

Engine: pyttsx3 -> eSpeak NG on Linux, SAPI5 on Windows. No neural vocoder,
negligible RAM. A dedicated worker thread owns the engine because pyttsx3 is
not thread-safe. is_busy() lets the mic loop ignore the app's own voice.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from config import TTS_RATE, TTS_VOLUME

_ECHO_TAIL_SEC = 0.35


class TTSEngine:
    def __init__(self, on_start: Optional[Callable[[str], None]] = None,
                 on_done: Optional[Callable[[], None]] = None, enabled: bool = True) -> None:
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=16)
        self._on_start = on_start
        self._on_done = on_done
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._quiet_after = 0.0
        self._failed = not enabled
        self.last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._loop, name="ks-tts", daemon=True)
        self._thread.start()

    def is_busy(self) -> bool:
        with self._pending_lock:
            if self._pending > 0:
                return True
        return time.monotonic() < self._quiet_after

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned or self._failed:
            return
        with self._pending_lock:
            self._pending += 1
        try:
            self._queue.put_nowait(cleaned)
        except queue.Full:
            with self._pending_lock:
                self._pending -= 1

    def _loop(self) -> None:
        if self._failed:
            return
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
        except Exception as exc:
            self.last_error = f"Voice output is unavailable: {exc}"
            self._failed = True
            with self._pending_lock:
                self._pending = 0
            return

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                if self._on_start:
                    self._on_start(text)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                try:
                    engine.stop()
                except Exception:
                    pass
            finally:
                with self._pending_lock:
                    self._pending = max(0, self._pending - 1)
                    idle = self._pending == 0
                self._quiet_after = time.monotonic() + _ECHO_TAIL_SEC
                if idle and self._on_done:
                    self._on_done()

    def shutdown(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

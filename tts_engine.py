"""
KernalSounder — Text-to-Speech.

Engine: pyttsx3 -> eSpeak NG. No neural vocoder, no GPU, negligible RAM.
A dedicated worker thread owns the engine because pyttsx3 is not thread-safe.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from config import TTS_RATE, TTS_VOLUME


class TTSEngine:
    def __init__(self, on_start: Optional[Callable[[str], None]] = None,
                 on_done: Optional[Callable[[], None]] = None) -> None:
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=16)
        self._on_start = on_start
        self._on_done = on_done
        self._thread = threading.Thread(target=self._loop, name="ks-tts", daemon=True)
        self._ready = threading.Event()
        self._failed = False
        self.last_error: Optional[str] = None
        self._thread.start()

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned or self._failed:
            return
        try:
            self._queue.put_nowait(cleaned)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(cleaned)
            except queue.Full:
                pass

    def _loop(self) -> None:
        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
        except Exception as exc:
            self.last_error = f"TTS init failed: {exc}"
            self._failed = True
            self._ready.set()
            return

        self._ready.set()
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
                if self._on_done:
                    self._on_done()

    def shutdown(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

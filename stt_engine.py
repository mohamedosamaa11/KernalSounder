"""
KernalSounder — Speech-to-Text.

Engine: Vosk small English model (vosk-model-small-en-us-0.15).
It is a Kaldi-based offline recognizer, far lighter than Whisper, and stays
within a few hundred MB of RAM on a 4 GB Ubuntu VM.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Callable, Optional

from config import MODEL_DIR, SAMPLE_RATE, STT_BLOCKSIZE


class STTEngine:
    def __init__(self, on_partial: Optional[Callable[[str], None]] = None) -> None:
        self._on_partial = on_partial
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._model = None
        self._recognizer = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def available(self) -> bool:
        if not MODEL_DIR.is_dir():
            self.last_error = (
                f"Vosk model not found at {MODEL_DIR}. Run ./setup.sh first."
            )
            return False
        try:
            from vosk import KaldiRecognizer, Model  # noqa: F401
            import sounddevice as sd  # noqa: F401
        except Exception as exc:
            self.last_error = f"STT dependencies missing: {exc}"
            return False
        return True

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self._model = Model(str(MODEL_DIR))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.available():
            raise RuntimeError(self.last_error or "STT is not available.")
        from vosk import KaldiRecognizer

        self._load_model()
        self._recognizer = KaldiRecognizer(self._model, SAMPLE_RATE)
        self._recognizer.SetWords(False)
        self._stop.clear()
        self._drain_audio()
        with self._lock:
            self._pending = ""
        self._open_stream()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="ks-stt",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_stream()
        if self._thread:
            self._thread.join(timeout=1.5)
        self._drain_audio()

    def _open_stream(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
            if self._stop.is_set():
                return
            try:
                self._audio_q.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    _ = self._audio_q.get_nowait()
                except queue.Empty:
                    pass

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=STT_BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _drain_audio(self) -> None:
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if self._recognizer.AcceptWaveform(chunk):
                    payload = json.loads(self._recognizer.Result())
                    text = (payload.get("text") or "").strip()
                    if text:
                        with self._lock:
                            self._pending = text
                else:
                    if self._on_partial:
                        payload = json.loads(self._recognizer.PartialResult())
                        partial = (payload.get("partial") or "").strip()
                        if partial:
                            self._on_partial(partial)
            except Exception:
                continue

        # Flush remaining decoder state when stopping.
        try:
            payload = json.loads(self._recognizer.FinalResult())
            text = (payload.get("text") or "").strip()
            if text:
                with self._lock:
                    self._pending = text
        except Exception:
            pass

    def listen_utterance(self, timeout: float = 8.0) -> str:
        """
        Block the caller (never the Tk main thread) until Vosk emits a
        final phrase or the timeout elapses.
        """
        import time

        with self._lock:
            self._pending = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                if self._pending:
                    text = self._pending
                    self._pending = ""
                    return text
            time.sleep(0.05)
        return ""

    _pending = ""

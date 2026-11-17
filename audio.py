"""
KernalSounder — microphone stream and command capture.

One always-open 16 kHz mono stream feeds both the wake-word detector and the
command recorder, so the mic is opened once and never re-calibrated.
The recorder uses WebRTC VAD to know when the user stopped talking.
"""

from __future__ import annotations

import queue
from typing import List, Optional

import numpy as np

from config import (
    CAPTURE_END_SILENCE_SEC,
    CAPTURE_MAX_SEC,
    CAPTURE_MIN_SPEECH_SEC,
    CAPTURE_START_TIMEOUT_SEC,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
)


def rms_level(chunk: np.ndarray) -> float:
    """Loudness mapped to 0..1 (about -55 dBFS -> 0, -12 dBFS -> 1)."""
    if chunk.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
    if rms <= 1.0:
        return 0.0
    db = 20.0 * np.log10(rms / 32768.0)
    return float(min(1.0, max(0.0, (db + 55.0) / 43.0)))


class MicStream:
    def __init__(self, device: Optional[str] = None) -> None:
        self._device = int(device) if device and str(device).isdigit() else device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, _frames, _time, _status) -> None:
        chunk = indata[:, 0].copy()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def play_earcon(kind: str = "wake") -> None:
    """Tiny two-tone chime, generated in memory. Never raises."""
    try:
        import sounddevice as sd

        rate = 22050
        notes = (660.0, 990.0) if kind == "wake" else (520.0, 390.0)
        parts = []
        for freq in notes:
            t = np.arange(int(rate * 0.07)) / rate
            env = np.minimum(1.0, np.minimum(t, t[::-1]) * 120.0)
            parts.append(0.18 * env * np.sin(2 * np.pi * freq * t))
        sd.play(np.concatenate(parts).astype(np.float32), rate, blocking=False)
    except Exception:
        pass


class CommandRecorder:
    """Collects one spoken command and decides when it is finished."""

    FRAME = 320  # 20 ms at 16 kHz, a size WebRTC VAD accepts

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS) -> None:
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception:
            self._vad = None
        self.reset()

    @property
    def uses_webrtc(self) -> bool:
        return self._vad is not None

    def reset(self, start_timeout: float = CAPTURE_START_TIMEOUT_SEC) -> None:
        self._chunks: List[np.ndarray] = []
        self._first_voiced: Optional[int] = None
        self._speech_sec = 0.0
        self._silence_sec = 0.0
        self._elapsed = 0.0
        self._noise_floor: Optional[float] = None
        self._start_timeout = start_timeout

    @property
    def heard_speech(self) -> bool:
        return self._speech_sec >= CAPTURE_MIN_SPEECH_SEC

    def is_voiced(self, chunk: np.ndarray) -> bool:
        """Public speech/no-speech decision for one chunk (updates the noise floor)."""
        return self._voiced(chunk)

    def _voiced(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32))))) if chunk.size else 0.0
        if self._noise_floor is None:
            self._noise_floor = rms
        elif rms < self._noise_floor:
            self._noise_floor = 0.7 * self._noise_floor + 0.3 * rms
        else:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
        loud_enough = rms > max(180.0, self._noise_floor * 1.8)
        if self._vad is None:
            return rms > max(300.0, self._noise_floor * 3.0)
        voiced = 0
        frames = 0
        data = chunk.astype(np.int16)
        for start in range(0, len(data) - self.FRAME + 1, self.FRAME):
            frames += 1
            try:
                if self._vad.is_speech(data[start:start + self.FRAME].tobytes(), SAMPLE_RATE):
                    voiced += 1
            except Exception:
                return loud_enough
        return frames > 0 and voiced * 2 >= frames and loud_enough

    def feed(self, chunk: np.ndarray) -> str:
        """Returns 'waiting', 'speaking', 'done' or 'timeout'."""
        self._chunks.append(chunk)
        step = len(chunk) / SAMPLE_RATE
        self._elapsed += step
        if self._voiced(chunk):
            if self._first_voiced is None:
                self._first_voiced = len(self._chunks) - 1
            self._speech_sec += step
            self._silence_sec = 0.0
        elif self._first_voiced is not None:
            self._silence_sec += step

        if self._first_voiced is not None and not self.heard_speech \
                and self._silence_sec >= CAPTURE_END_SILENCE_SEC:
            self._first_voiced = None
            self._speech_sec = 0.0
            self._silence_sec = 0.0
        if self._first_voiced is None:
            return "timeout" if self._elapsed >= self._start_timeout else "waiting"
        if self.heard_speech and self._silence_sec >= CAPTURE_END_SILENCE_SEC:
            return "done"
        if self._elapsed >= CAPTURE_MAX_SEC:
            return "done" if self.heard_speech else "timeout"
        return "speaking"

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        start = 0
        if self._first_voiced is not None:
            pre_roll = max(1, int(0.3 * SAMPLE_RATE / max(1, len(self._chunks[0]))))
            start = max(0, self._first_voiced - pre_roll)
        return np.concatenate(self._chunks[start:]).astype(np.int16)

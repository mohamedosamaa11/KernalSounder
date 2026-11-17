"""
KernalSounder — text-to-speech.

Talks to the operating system's own speech engine, one sentence at a time:
  Windows: SAPI (SAPI.SpVoice through comtypes), synchronous Speak().
  Linux:   espeak-ng / espeak, run per sentence with an argument list.
  Other:   pyttsx3, re-initialized for every sentence.

pyttsx3's reusable engine is avoided on purpose: on Windows its loop stops
after the first runAndWait(), so every later reply was silent.
A dedicated worker thread owns the engine; is_busy() lets the mic loop
ignore the app's own voice.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

from config import IS_WINDOWS, TTS_RATE, TTS_VOICE, TTS_VOLUME

_ECHO_TAIL_SEC = 0.35


class _SapiVoice:
    def __init__(self) -> None:
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        self.name = self._pick_voice(comtypes.client)
        self._voice.Rate = max(-10, min(10, round((TTS_RATE - 165) / 12)))
        self._voice.Volume = int(max(0.0, min(1.0, TTS_VOLUME)) * 100)

    def _pick_voice(self, client) -> str:
        """Prefer the newer OneCore voices (clearer, same cost) and the configured name."""
        tokens = []
        try:
            cat = client.CreateObject("SAPI.SpObjectTokenCategory")
            cat.SetId(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", False)
            found = cat.EnumerateTokens()
            tokens += [found.Item(i) for i in range(found.Count)]
        except Exception:
            pass
        try:
            desktop = self._voice.GetVoices()
            tokens += [desktop.Item(i) for i in range(desktop.Count)]
        except Exception:
            pass
        english = [t for t in tokens if "english" in t.GetDescription().lower()] or tokens
        preferred = [t for t in english if TTS_VOICE.lower() in t.GetDescription().lower()]
        for token in preferred + english:
            try:
                self._voice.Voice = token
                return token.GetDescription()
            except Exception:
                continue
        return "default"

    def say(self, text: str) -> None:
        self._voice.Speak(text, 0)

    def stop(self) -> None:
        try:
            self._voice.Speak("", 2)
        except Exception:
            pass


class _EspeakVoice:
    def __init__(self) -> None:
        self._exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._exe:
            raise RuntimeError("espeak-ng is not installed (sudo apt install espeak-ng)")
        self._proc: Optional[subprocess.Popen] = None
        self.name = self._pick_voice()

    def _pick_voice(self) -> str:
        """MBROLA voices (apt: mbrola mbrola-us1) sound far less robotic than the
        default formant voice and still run as a tiny separate process."""
        wanted = [TTS_VOICE, "mb-us1", "mb-us2", "en-us+f3"]
        try:
            listing = subprocess.run([self._exe, "--voices=mb"], capture_output=True, text=True,
                                     timeout=10).stdout.lower()
        except Exception:
            listing = ""
        for voice in wanted:
            if not voice.startswith("mb-") or voice.lower() in listing:
                if voice.startswith("mb-") and not self._mbrola_works(voice):
                    continue
                return voice
        return "en-us+f3"

    def _mbrola_works(self, voice: str) -> bool:
        try:
            done = subprocess.run([self._exe, "-v", voice, "-q", "--", "test"], capture_output=True,
                                  timeout=10)
            return done.returncode == 0 and not done.stderr.strip()
        except Exception:
            return False

    def say(self, text: str) -> None:
        argv = [self._exe, "-v", self.name, "-s", str(int(TTS_RATE)),
                "-a", str(int(max(0.0, min(1.0, TTS_VOLUME)) * 200)), "--", text]
        self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()


class _Pyttsx3Voice:
    name = "pyttsx3 default"

    def say(self, text: str) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def stop(self) -> None:
        pass


def _make_voice():
    candidates = [_SapiVoice, _Pyttsx3Voice] if IS_WINDOWS else [_EspeakVoice, _Pyttsx3Voice]
    errors = []
    for factory in candidates:
        try:
            return factory()
        except Exception as exc:
            errors.append(f"{factory.__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


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
        self._voice = None
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

    def stop_speaking(self) -> None:
        """Cut the current sentence and drop queued ones (e.g. when the user starts talking)."""
        while True:
            try:
                if self._queue.get_nowait() is not None:
                    with self._pending_lock:
                        self._pending = max(0, self._pending - 1)
            except queue.Empty:
                break
        if self._voice is not None:
            self._voice.stop()

    def _loop(self) -> None:
        if self._failed:
            return
        try:
            self._voice = _make_voice()
        except Exception as exc:
            self.last_error = f"Voice output is unavailable: {exc}"
            logging.warning(self.last_error)
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
                self._voice.say(text)
            except Exception as exc:
                logging.warning("Speech output failed: %s", exc)
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

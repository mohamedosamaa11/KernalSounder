#!/usr/bin/env python3
"""
KernalSounder
=============
Voice interaction layer over Ubuntu Linux.

Pipeline (raw voice never reaches the shell):

    Microphone  ->  Google STT  ->  rule-based NLP  ->  validator
                                              |              |
                                              v              v
                                         Tkinter GUI    subprocess argv
                                                              |
                                                              v
                                                        pyttsx3 / eSpeak

Designed for a 4 GB RAM Ubuntu VM on an Intel Core i3 host:
  STT  : SpeechRecognition + recognize_google()  (cloud, no local model)
  TTS  : pyttsx3 + eSpeak NG                     (no neural vocoder)
  NLP  : keyword / regex matcher                 (no spaCy, no transformers)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional, Tuple

import psutil

from config import (
    APP_NAME,
    LOG_FILE,
    MONITOR_INTERVAL_SEC,
    STT_NETWORK_ERROR,
    WORKSPACE,
)
from executor import ExecutionResult, execute
from gui import KernelSounderGUI
from nlp_engine import FALLBACK_TEXT, parse_command
from security import (
    Intent,
    ParsedCommand,
    ValidationResult,
    ValidationState,
    ensure_workspace,
    validate,
)
from stt_engine import STTEngine, STTNetworkError
from tts_engine import TTSEngine

PendingDelete = Tuple[ParsedCommand, ValidationResult]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class KernelSounder:
    def __init__(self) -> None:
        ensure_workspace()
        self._stop = threading.Event()
        self._listening = threading.Event()
        self._speaking = threading.Event()
        self._busy = threading.Lock()
        self._pending: Optional[PendingDelete] = None
        self._listen_thread: Optional[threading.Thread] = None

        self.gui = KernelSounderGUI(
            on_toggle_listen=self.toggle_listen,
            on_confirm=lambda: self.submit_text("confirm", source="gui"),
            on_cancel=lambda: self.submit_text("cancel", source="gui"),
            on_typed=lambda text: self.submit_text(text, source="typed"),
        )

        self.tts = TTSEngine(
            on_start=lambda _t: self._speaking.set(),
            on_done=self._speaking.clear,
        )
        self.stt = STTEngine(on_partial=lambda text: self.gui.emit("partial", text=text))

        threading.Thread(target=self._monitor_loop, name="ks-monitor", daemon=True).start()
        self.gui.emit(
            "log",
            text=f"{APP_NAME} ready. Workspace: {WORKSPACE}",
            tag="info",
        )
        if not sys.platform.startswith("linux"):
            self.gui.emit(
                "log",
                text="This host is not Linux. Run the project on the Ubuntu VM for real OS commands.",
                tag="warn",
            )
        if not self.stt.available():
            self.gui.emit(
                "log",
                text=self.stt.last_error or "Speech recognition is unavailable.",
                tag="warn",
            )
            self.gui.emit(
                "status",
                text="Microphone STT is offline. You can still type commands.",
            )
        if self.tts.last_error:
            self.gui.emit("log", text=self.tts.last_error, tag="warn")

    def run(self) -> None:
        logging.info("Starting %s", APP_NAME)
        try:
            self.tts.speak(
                "Kernal Sounder ready. Press listen, or type a command."
            )
            self.gui.run()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        self._listening.clear()
        try:
            self.stt.stop()
        except Exception:
            pass
        self.tts.shutdown()
        logging.info("Shutdown complete")

    def toggle_listen(self) -> None:
        if self._listening.is_set():
            self._listening.clear()
            threading.Thread(target=self.stt.stop, name="ks-stt-stop", daemon=True).start()
            self.gui.emit("listen", active=False)
            self.gui.emit("status", text="Microphone stopped.")
            return

        if not self.stt.available():
            self.gui.emit(
                "error",
                text=self.stt.last_error or "Cannot start the microphone.",
            )
            return

        self._listening.set()
        self.gui.emit("listen", active=True)
        self.gui.emit("status", text="Listening… speak a command.")
        self._listen_thread = threading.Thread(
            target=self._listen_loop, name="ks-listen", daemon=True
        )
        self._listen_thread.start()

    def submit_text(self, text: str, source: str = "typed") -> None:
        threading.Thread(
            target=self._handle_text,
            args=(text, source),
            name="ks-cmd",
            daemon=True,
        ).start()

    def _listen_loop(self) -> None:
        try:
            self.stt.start()
        except Exception as exc:
            self._listening.clear()
            self.gui.emit("listen", active=False)
            self.gui.emit("error", text=f"Microphone error: {exc}")
            logging.exception("STT start failed")
            return

        self.tts.speak("Listening.")
        while self._listening.is_set() and not self._stop.is_set():
            if self._speaking.is_set():
                time.sleep(0.08)
                continue
            try:
                text = self.stt.listen_utterance(timeout=6.0)
            except STTNetworkError:
                self.gui.emit("error", text=STT_NETWORK_ERROR)
                self.gui.emit("status", text=STT_NETWORK_ERROR)
                self.gui.emit("log", text=STT_NETWORK_ERROR, tag="bad")
                self.tts.speak(STT_NETWORK_ERROR)
                time.sleep(4.0)
                continue
            except Exception as exc:
                logging.warning("STT utterance error: %s", exc)
                self.gui.emit("status", text="Voice recognition hiccup. Still listening.")
                time.sleep(0.3)
                continue
            if not text:
                continue
            if self._speaking.is_set() or not self._listening.is_set():
                continue
            self._handle_text(text, source="voice")

        try:
            self.stt.stop()
        except Exception:
            pass

    def _handle_text(self, text: str, source: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self.gui.emit("status", text="I did not catch that. Please try again.")
            if source == "voice":
                self.tts.speak("I did not catch that.")
            return

        if not self._busy.acquire(blocking=False):
            self.gui.emit("status", text="Still working on the previous command.")
            return

        try:
            self._process(cleaned, source)
        except Exception as exc:
            logging.exception("Command pipeline failed")
            self.gui.emit("validation", state=ValidationState.ERROR.value,
                          reason=str(exc), argv="")
            self.gui.emit("log", text=f"Pipeline error: {exc}", tag="bad")
            self.tts.speak("Something went wrong, but I am still running.")
        finally:
            self._busy.release()

    def _process(self, text: str, source: str) -> None:
        self.gui.emit("heard", text=text)
        self.gui.emit("status", text=f"Processing ({source})…")
        parsed = parse_command(text)
        slots = ", ".join(f"{k}={v}" for k, v in parsed.slots.items())
        self.gui.emit(
            "intent",
            intent=parsed.intent.value,
            confidence=parsed.confidence,
            slots=slots,
        )
        logging.info("Heard [%s]: %s -> %s %s", source, text, parsed.intent.value, parsed.slots)

        if self._pending is not None:
            if parsed.intent == Intent.CONFIRM:
                self._run_pending_delete()
                return
            if parsed.intent == Intent.CANCEL:
                self._clear_pending("Delete cancelled.")
                self.tts.speak("Delete cancelled.")
                return
            self.gui.emit(
                "status",
                text="A delete is waiting. Say confirm or cancel, or use the buttons.",
            )
            self.tts.speak("Please confirm or cancel the delete first.")
            return

        if parsed.intent in (Intent.CONFIRM, Intent.CANCEL):
            self.gui.emit("validation", state="IDLE", reason="Nothing is waiting for confirmation.", argv="")
            self.gui.emit("status", text="Nothing to confirm.")
            return

        if parsed.intent == Intent.UNKNOWN or parsed.confidence < 0.45:
            self.gui.emit(
                "validation",
                state=ValidationState.BLOCKED.value,
                reason=FALLBACK_TEXT,
                argv="(blocked — raw voice is never a shell command)",
            )
            self.gui.emit("log", text=f"Unmapped: {text!r}\n{FALLBACK_TEXT}", tag="warn")
            self.gui.emit("status", text=FALLBACK_TEXT)
            self.tts.speak(FALLBACK_TEXT)
            return

        result = validate(parsed)
        argv_preview = " ".join(result.argv) if result.argv else result.display_command
        self.gui.emit(
            "validation",
            state=result.state.value,
            reason=result.reason,
            argv=argv_preview,
        )

        if result.state == ValidationState.BLOCKED:
            self.gui.emit("log", text=f"BLOCKED {parsed.intent.value}: {result.reason}", tag="bad")
            self.gui.emit("status", text="Blocked by the security validator.")
            self.tts.speak(result.reason)
            return

        if result.state == ValidationState.ERROR:
            self.gui.emit("log", text=f"ERROR: {result.reason}", tag="bad")
            self.tts.speak(result.reason)
            return

        if result.state == ValidationState.NEEDS_CONFIRM:
            self._pending = (parsed, result)
            target = result.resolved_path.name if result.resolved_path else "that item"
            msg = (
                f"Delete {target}? Say confirm or cancel, or use the buttons. "
                "This is required before any delete."
            )
            self.gui.emit("status", text=msg)
            self.gui.emit("confirm_ready", ready=True)
            self.gui.emit("log", text=f"PENDING DELETE {target}", tag="warn")
            self.tts.speak(f"Please confirm deletion of {target}. Say confirm or cancel.")
            return

        self._run_validated(parsed, result)

    def _run_pending_delete(self) -> None:
        if not self._pending:
            return
        parsed, result = self._pending
        self._pending = None
        self.gui.emit("confirm_ready", ready=False)
        result.state = ValidationState.SAFE
        result.reason = "Delete confirmed by the user."
        argv_preview = " ".join(result.argv) if result.argv else result.display_command
        self.gui.emit(
            "validation",
            state=ValidationState.SAFE.value,
            reason=result.reason,
            argv=argv_preview,
        )
        self._run_validated(parsed, result)

    def _clear_pending(self, message: str) -> None:
        self._pending = None
        self.gui.emit("confirm_ready", ready=False)
        self.gui.emit("status", text=message)
        self.gui.emit("log", text=message, tag="warn")
        self.gui.emit(
            "validation",
            state="IDLE",
            reason=message,
            argv="(not executed)",
        )

    def _run_validated(self, parsed: ParsedCommand, result: ValidationResult) -> None:
        self.gui.emit("status", text="Executing validated argv via subprocess…")
        outcome: ExecutionResult = execute(parsed, result)
        tag = "ok" if outcome.ok else "bad"
        self.gui.emit(
            "log",
            text=f"{parsed.intent.value}  |  {outcome.display_command}\n{outcome.details}",
            tag=tag,
        )
        self.gui.emit(
            "status",
            text=outcome.spoken if outcome.ok else f"Failed: {outcome.spoken}",
        )
        self.tts.speak(outcome.spoken)

    def _monitor_loop(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            try:
                cpu = psutil.cpu_percent(interval=0.2)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                ram_used = mem.used / (1024 ** 3)
                ram_total = mem.total / (1024 ** 3)
                ram_text = f"{ram_used:.1f}/{ram_total:.1f} GB  {mem.percent:.0f}%"
                self.gui.emit(
                    "metrics",
                    cpu=float(cpu),
                    ram=float(mem.percent),
                    ram_text=ram_text,
                    disk=float(disk.percent),
                )
            except Exception as exc:
                logging.warning("Monitor error: %s", exc)
            time.sleep(MONITOR_INTERVAL_SEC)


def main() -> int:
    _setup_logging()
    try:
        KernelSounder().run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())

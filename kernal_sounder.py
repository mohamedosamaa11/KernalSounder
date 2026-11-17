#!/usr/bin/env python3
"""
KernalSounder
=============
Hands-free voice layer over Linux and Windows.

    "Hi Sounder" ──► wake word (locally trained head on openWakeWord features,
                     always on, ~6 ms per 80 ms) + Whisper double-check
         │
         ▼
    floating overlay opens, mic records until you stop talking (WebRTC VAD)
         │
         ▼
    offline speech-to-text (faster-whisper tiny.en, vocabulary-biased)
         │
         ▼
    rule-based NLP ──► validator ──► executor (no shell, ever) ──► voice reply

Nothing needs a click: the wake word opens the overlay, silence closes it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from typing import Optional, Tuple

import psutil

from config import (
    APP_NAME,
    CONFIRM_TIMEOUT_SEC,
    EARCON,
    LOG_FILE,
    MIC_DEVICE,
    MONITOR_INTERVAL_SEC,
    SHOW_IDLE_PILL,
    STT_BACKEND,
    WAKE_COOLDOWN_SEC,
    WAKE_ENABLED,
    WAKE_MODEL,
    WAKE_PHRASE,
    WAKE_THRESHOLD,
    WAKE_VERIFY,
    WAKE_VERIFY_SEC,
    WHISPER_MODEL,
    WORKSPACE,
)
import numpy as np

from audio import CommandRecorder, MicStream, play_earcon, rms_level
from config import CHUNK_SAMPLES, MODELS_DIR, SAMPLE_RATE
from executor import ExecutionResult, execute
from gui import KernelSounderGUI
from nlp_engine import FALLBACK_TEXT, answer_to_slot, missing_slot, parse_command
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
    handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


class KernelSounder:
    def __init__(self, args: argparse.Namespace) -> None:
        ensure_workspace()
        self.args = args
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._pending: Optional[PendingDelete] = None
        self._awaiting: Optional[dict] = None
        self._pending_at = 0.0
        self._state = "idle"
        self._capture_reason = ""
        self._manual = threading.Event()
        self._finish_now = threading.Event()
        self._abort = threading.Event()
        self._abort_quiet = False
        self._followup_after_tts = False
        self._wake_dirty = False
        self.mic: Optional[MicStream] = None
        self.wake = None
        self.recorder = CommandRecorder()
        self.speech_gate = CommandRecorder(aggressiveness=3)
        self.gate_rejects = 0
        self.wake_enabled = WAKE_ENABLED and not args.no_wake

        hint = f"Say “{args.wake_phrase}”" if self.wake_enabled else "Click to talk"
        self.gui = KernelSounderGUI(
            on_toggle_listen=self.toggle_listen,
            on_confirm=self.confirm_from_ui,
            on_cancel=self.cancel_from_ui,
            on_typed=lambda text: self.submit_text(text, source="typed"),
            overlay_callbacks=dict(
                on_talk=self.request_capture,
                on_stop=self.stop_from_ui,
                on_send=self.finish_capture,
                on_confirm=self.confirm_from_ui,
                on_cancel=self.cancel_from_ui,
                on_dashboard=self.gui_show_dashboard,
                on_quit=self.quit,
            ),
            start_hidden=not args.dashboard,
            hint=hint if self.wake_enabled else "Click to talk",
            show_idle_pill=not args.no_pill and SHOW_IDLE_PILL,
        )
        self.tts = TTSEngine(enabled=not args.mute)
        self.stt = STTEngine(
            backend=args.stt,
            model=args.whisper_model,
            on_status=lambda text: self.gui.emit("status", text=text),
        )

        threading.Thread(target=self._load_models, name="ks-models", daemon=True).start()
        threading.Thread(target=self._voice_loop, name="ks-voice", daemon=True).start()
        threading.Thread(target=self._monitor_loop, name="ks-monitor", daemon=True).start()
        self._log(f"{APP_NAME} ready. Workspace: {WORKSPACE}", "info")

    def _log(self, text: str, tag: str = "info") -> None:
        self.gui.emit("log", text=text, tag=tag)

    def _ui(self, phase: str, text: str = "", tone: str = "info") -> None:
        self.gui.emit("overlay", phase=phase, text=text, tone=tone)

    def gui_show_dashboard(self) -> None:
        self.gui.emit("show_dashboard")

    def run(self) -> None:
        logging.info("Starting %s", APP_NAME)
        try:
            self.gui.run()
        finally:
            self.shutdown()

    def quit(self) -> None:
        self.gui.emit("quit")

    def shutdown(self) -> None:
        self._stop.set()
        if self.mic is not None:
            self.mic.stop()
        self.tts.shutdown()
        logging.info("Shutdown complete")

    def _load_models(self) -> None:
        if self.wake_enabled:
            try:
                from wake_word import WakeWordDetector

                self.wake = WakeWordDetector(self.args.wake_model, threshold=self.args.threshold)
                self._log(f"Wake word ready: {self.wake.name} (threshold {self.wake.threshold:.2f})", "ok")
            except Exception as exc:
                self.wake = None
                self._log(f"Wake word unavailable: {exc}", "warn")
                self.gui.emit("hint", text="Click to talk")
        try:
            from app_catalog import CATALOG

            count = CATALOG.refresh()
            self._log(f"Found {count} installed apps.", "info")
        except Exception as exc:
            self._log(f"App catalog unavailable: {exc}", "warn")
        started = time.monotonic()
        if self.stt.load():
            try:
                from app_catalog import CATALOG

                self.stt.set_app_names(CATALOG.allowed_names())
            except Exception:
                pass
            self._log(f"Speech recognition ready: {self.stt.describe()} "
                      f"({time.monotonic() - started:.1f}s)", "ok")
            self.gui.emit("status", text="Ready. Say the wake word or press Talk.")
        if self.stt.last_error:
            self._log(self.stt.last_error, "warn")
        if self.tts.last_error:
            self._log(self.tts.last_error, "warn")

    def toggle_listen(self) -> None:
        if self._state == "capture":
            self.stop_from_ui()
        else:
            self.request_capture()

    def request_capture(self) -> None:
        if self.mic is None:
            self.gui.emit("error", text="No microphone available. Type the command in the dashboard.")
            self.gui.emit("show_dashboard")
            return
        self._manual.set()

    def finish_capture(self) -> None:
        if self._state == "capture":
            self._finish_now.set()

    def stop_from_ui(self) -> None:
        if self._state == "capture":
            self._abort_quiet = False
            self._abort.set()
        else:
            self._ui("idle")

    def confirm_from_ui(self) -> None:
        self._quiet_abort()
        self.submit_text("confirm", source="gui")

    def cancel_from_ui(self) -> None:
        self._quiet_abort()
        self.submit_text("cancel", source="gui")

    def _quiet_abort(self) -> None:
        if self._state == "capture":
            self._abort_quiet = True
            self._abort.set()

    def submit_text(self, text: str, source: str = "typed") -> None:
        threading.Thread(target=self._handle_text, args=(text, source), name="ks-cmd", daemon=True).start()

    def _voice_loop(self) -> None:
        try:
            self.mic = MicStream(self.args.mic)
            self.mic.start()
        except Exception as exc:
            self.mic = None
            self.gui.emit("error", text=f"Microphone unavailable ({exc}). You can still type commands.")
            self.gui.emit("hint", text="No microphone · right-click")
            return

        last_wake = 0.0
        window = max(1, int(WAKE_VERIFY_SEC * SAMPLE_RATE / CHUNK_SAMPLES))
        recent: deque = deque(maxlen=window)
        voiced: deque = deque(maxlen=window)
        while not self._stop.is_set():
            chunk = self.mic.read(timeout=0.5)
            if chunk is None:
                continue
            recent.append(chunk)
            if self._state == "capture":
                self._capture_step(chunk)
                continue
            if self._state == "processing":
                continue
            if self.tts.is_busy():
                self._wake_dirty = True
                continue
            if self._followup_after_tts:
                self._followup_after_tts = False
                if self._pending is not None or self._awaiting is not None:
                    self._begin_capture("followup")
                    continue
            if self._manual.is_set():
                self._manual.clear()
                self._begin_capture("button")
                continue
            wake = self.wake
            if wake is None:
                continue
            if self._wake_dirty:
                wake.reset()
                self._wake_dirty = False
            score = wake.process(chunk)
            voiced.append(self.speech_gate.is_voiced(chunk))
            now = time.monotonic()
            if score >= wake.threshold and now - last_wake > WAKE_COOLDOWN_SEC:
                last_wake = now
                wake.reset()
                speech_sec = sum(voiced) * CHUNK_SAMPLES / SAMPLE_RATE
                if speech_sec < 0.45:
                    self.gate_rejects += 1
                    logging.debug("Wake %.3f ignored: only %.2fs of speech", score, speech_sec)
                    continue
                if self.args.verify:
                    ok, heard = self.stt.verify_keyword(np.concatenate(list(recent)), self.args.wake_phrase)
                    if ok is False:
                        logging.info("Wake %.3f rejected by verifier: %r", score, heard)
                        if self.args.collect_negatives:
                            self._save_hard_negative(np.concatenate(list(recent)))
                        continue
                    logging.info("Wake %.3f verified: %r", score, heard)
                else:
                    logging.info("Wake word %.3f", score)
                self._begin_capture("wake")

    def _save_hard_negative(self, audio) -> None:
        try:
            from wake_word import find_wake_model

            model = find_wake_model(self.args.wake_model)
            name = model.stem if model else "wake"
            folder = MODELS_DIR / "custom" / f"{name}_samples" / "negative"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"auto_{int(time.time() * 1000)}.npy", audio.astype(np.int16))
        except Exception as exc:
            logging.warning("Could not save hard negative: %s", exc)

    def _begin_capture(self, reason: str) -> None:
        self.recorder.reset(start_timeout=6.0 if reason == "followup" else 4.0)
        self._abort.clear()
        self._finish_now.clear()
        self._capture_reason = reason
        self._state = "capture"
        self.gui.emit("listen", active=True)
        if reason == "wake" and EARCON:
            play_earcon("wake")
        if reason == "followup" and self._pending is not None:
            target = self._pending[1].resolved_path.name if self._pending[1].resolved_path else "it"
            self._ui("confirm", f"Delete {target}?  Say confirm or cancel.", "warn")
        elif reason == "followup" and self._awaiting is not None:
            self._ui("listening", self._awaiting["question"], "info")
        else:
            self._ui("listening", "Listening…", "muted")

    def _capture_step(self, chunk) -> None:
        self.gui.emit("level", value=rms_level(chunk))
        reason = self._capture_reason
        if self._abort.is_set():
            self._abort.clear()
            self._state = "idle"
            self.gui.emit("listen", active=False)
            if not self._abort_quiet:
                if reason == "followup" and self._pending is not None:
                    self.submit_text("cancel", source="gui")
                else:
                    self._awaiting = None
                    self._ui("idle")
            self._abort_quiet = False
            return

        status = self.recorder.feed(chunk)
        if self._finish_now.is_set():
            self._finish_now.clear()
            status = "done" if self.recorder.heard_speech else "timeout"

        if status == "done":
            audio = self.recorder.audio()
            self._state = "processing"
            self.gui.emit("listen", active=False)
            self._ui("thinking", "Working on it…", "muted")
            threading.Thread(target=self._handle_audio, args=(audio, reason),
                             name="ks-stt", daemon=True).start()
        elif status == "timeout":
            self._state = "idle"
            self.gui.emit("listen", active=False)
            if reason == "followup" and self._pending is not None:
                self._clear_pending("No answer, so the delete was cancelled.")
                self.tts.speak("Delete cancelled.")
            elif reason == "followup" and self._awaiting is not None:
                self._awaiting = None
                self._ui("result", "No answer, so I left it.", "muted")
            else:
                self._ui("result", "I didn't hear a command.", "muted")

    def _handle_audio(self, audio, reason: str) -> None:
        try:
            if not self.stt.ready.is_set():
                self._ui("thinking", "Loading the speech model…", "muted")
            started = time.monotonic()
            text = self.stt.transcribe(audio)
            logging.info("STT %.2fs for %.1fs audio: %r", time.monotonic() - started,
                         len(audio) / 16000, text)
        except STTNetworkError as exc:
            self._state = "idle"
            self._ui("result", str(exc), "bad")
            self._log(str(exc), "bad")
            self.tts.speak(str(exc))
            return
        except Exception as exc:
            self._state = "idle"
            logging.exception("Speech recognition failed")
            self._ui("result", f"Speech recognition failed: {exc}", "bad")
            return

        self._state = "idle"
        if not text:
            if reason == "followup" and self._pending is not None:
                self._ui("result", "Say confirm or cancel.", "warn")
                self.tts.speak("Please say confirm or cancel.")
                self._followup_after_tts = True
            else:
                self._awaiting = None
                self._ui("result", "Sorry, I didn't catch that.", "muted")
            return
        self._handle_text(text, source="voice")

    def _handle_text(self, text: str, source: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if not self._busy.acquire(blocking=False):
            self.gui.emit("status", text="Still working on the previous command.")
            return
        try:
            self._process(cleaned, source)
        except Exception as exc:
            logging.exception("Command pipeline failed")
            self.gui.emit("validation", state=ValidationState.ERROR.value, reason=str(exc), action="")
            self._log(f"Pipeline error: {exc}", "bad")
            self._ui("result", "Something went wrong, but I am still running.", "bad")
            self.tts.speak("Something went wrong, but I am still running.")
        finally:
            self._busy.release()

    def _process(self, text: str, source: str) -> None:
        self.gui.emit("heard", text=text)
        if source != "gui":
            self._ui("thinking", f"“{text}”", "info")
        parsed = parse_command(text)
        slots = ", ".join(f"{k}={v}" for k, v in parsed.slots.items())
        self.gui.emit("intent", intent=parsed.intent.value, confidence=parsed.confidence, slots=slots)
        logging.info("Heard [%s]: %s -> %s %s %.2f", source, text, parsed.intent.value,
                     parsed.slots, parsed.confidence)

        if self._pending is not None and time.monotonic() - self._pending_at > CONFIRM_TIMEOUT_SEC:
            self._clear_pending("The pending delete expired.")

        if self._pending is not None:
            if parsed.intent == Intent.CONFIRM:
                self._run_pending_delete()
                return
            if parsed.intent == Intent.CANCEL:
                self._clear_pending("Delete cancelled.")
                self.tts.speak("Delete cancelled.")
                return
            self.gui.emit("status", text="A delete is waiting. Say confirm or cancel, or use the buttons.")
            self._ui("confirm", "A delete is waiting. Say confirm or cancel.", "warn")
            self.tts.speak("Please confirm or cancel the delete first.")
            if source == "voice":
                self._followup_after_tts = True
            return

        if self._awaiting is not None and source != "gui":
            awaiting, self._awaiting = self._awaiting, None
            fresh = time.monotonic() - awaiting["at"] <= CONFIRM_TIMEOUT_SEC
            standalone = (parsed.intent not in (Intent.UNKNOWN, Intent.CONFIRM, Intent.CANCEL)
                          and parsed.confidence >= 0.85 and len(text.split()) >= 2
                          and missing_slot(parsed) is None)
            if fresh and parsed.intent == Intent.CANCEL:
                self._ui("result", "Okay, cancelled.", "muted")
                self.tts.speak("Okay, cancelled.")
                return
            if fresh and parsed.intent != Intent.DANGEROUS and not standalone:
                value = answer_to_slot(awaiting["slot"], text)
                if not value:
                    self._ui("result", "Sorry, I didn't catch that.", "muted")
                    return
                parsed = ParsedCommand(intent=awaiting["intent"], slots={awaiting["slot"]: value},
                                       raw_text=text, confidence=0.9)
                self.gui.emit("intent", intent=parsed.intent.value, confidence=parsed.confidence,
                              slots=f"{awaiting['slot']}={value}")
                logging.info("Follow-up answer %r -> %s %s", text, parsed.intent.value, parsed.slots)

        if parsed.intent in (Intent.CONFIRM, Intent.CANCEL):
            self.gui.emit("validation", state="IDLE", reason="Nothing is waiting for confirmation.", action="")
            self._ui("result", "Nothing is waiting for confirmation.", "muted")
            return

        if parsed.intent == Intent.DANGEROUS:
            reason = parsed.slots.get("reason", "That request is not allowed.")
            self.gui.emit("validation", state=ValidationState.BLOCKED.value, reason=reason,
                          action="(refused by the safety guard)")
            self._log(f"SECURITY refused {text!r}: {reason}", "bad")
            logging.warning("Security refusal: %r -> %s", text, reason)
            self._ui("result", f"Refused: {reason}", "bad")
            self.tts.speak(reason)
            return

        missing = missing_slot(parsed)
        if missing is not None and parsed.intent != Intent.UNKNOWN:
            slot, question = missing
            self._awaiting = {"intent": parsed.intent, "slot": slot, "question": question,
                              "at": time.monotonic()}
            self.gui.emit("status", text=question)
            self._ui("listening" if source == "voice" else "result", question, "info")
            self.tts.speak(question)
            if self.mic is not None and source == "voice":
                self._followup_after_tts = True
            return

        if parsed.intent == Intent.UNKNOWN or parsed.confidence < 0.45:
            self.gui.emit(
                "validation",
                state=ValidationState.BLOCKED.value,
                reason=FALLBACK_TEXT,
                action="(blocked — raw voice is never a command)",
            )
            self._log(f"Unmapped: {text!r}", "warn")
            self._ui("result", f"“{text}” is not a command I know.", "bad")
            self.tts.speak(FALLBACK_TEXT)
            return

        result = validate(parsed)
        self.gui.emit("validation", state=result.state.value, reason=result.reason, action=result.action)

        if result.state in (ValidationState.BLOCKED, ValidationState.ERROR):
            label = "BLOCKED" if result.state == ValidationState.BLOCKED else "ERROR"
            self._log(f"{label} {parsed.intent.value}: {result.reason}", "bad")
            self._ui("result", f"Blocked: {result.reason}", "bad")
            self.tts.speak(result.reason)
            return

        if result.state == ValidationState.NEEDS_CONFIRM:
            self._pending = (parsed, result)
            self._pending_at = time.monotonic()
            target = result.resolved_path.name if result.resolved_path else "that item"
            self.gui.emit("confirm_ready", ready=True)
            self._log(f"PENDING DELETE {target}", "warn")
            self._ui("confirm", f"Delete {target}?  Say confirm or cancel.", "warn")
            self.tts.speak(f"Delete {target}? Say confirm or cancel.")
            if self.mic is not None:
                self._followup_after_tts = True
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
        self.gui.emit("validation", state=result.state.value, reason=result.reason, action=result.action)
        self._run_validated(parsed, result)

    def _clear_pending(self, message: str) -> None:
        self._pending = None
        self._followup_after_tts = False
        self.gui.emit("confirm_ready", ready=False)
        self.gui.emit("status", text=message)
        self._log(message, "warn")
        self.gui.emit("validation", state="IDLE", reason=message, action="(not executed)")
        self._ui("result", message, "warn")

    def _run_validated(self, parsed: ParsedCommand, result: ValidationResult) -> None:
        outcome: ExecutionResult = execute(parsed, result)
        tag = "ok" if outcome.ok else "bad"
        self._log(f"{parsed.intent.value}  |  {outcome.display_command}\n{outcome.details}", tag)
        self.gui.emit("status", text=outcome.spoken if outcome.ok else f"Failed: {outcome.spoken}")
        shown = outcome.spoken
        if parsed.intent == Intent.LIST_FILES and outcome.ok and outcome.details != "(empty)":
            names = [row.split("  ")[-1].strip() for row in outcome.details.splitlines()]
            preview = ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
            shown = f"{outcome.spoken} {preview}"
        self._ui("result", shown, "ok" if outcome.ok else "bad")
        self.tts.speak(outcome.spoken)

    def _monitor_loop(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            if self.gui.dashboard_visible:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    disk = psutil.disk_usage(str(WORKSPACE.anchor or "/"))
                    gb = 1024 ** 3
                    self.gui.emit(
                        "metrics",
                        cpu=float(cpu),
                        ram=float(mem.percent),
                        ram_text=f"{mem.used / gb:.1f}/{mem.total / gb:.1f} GB  {mem.percent:.0f}%",
                        disk=float(disk.percent),
                    )
                except Exception as exc:
                    logging.warning("Monitor error: %s", exc)
            self._stop.wait(MONITOR_INTERVAL_SEC)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} — hands-free voice layer")
    parser.add_argument("--dashboard", action="store_true", help="open the full console window at start")
    parser.add_argument("--no-wake", action="store_true", help="disable the wake word (click the pill to talk)")
    parser.add_argument("--no-pill", action="store_true", help="hide the idle pill; overlay appears only on wake")
    parser.add_argument("--wake-model", default=WAKE_MODEL, help="openWakeWord .onnx model name or path")
    parser.add_argument("--wake-phrase", default=WAKE_PHRASE, help="phrase shown in the idle hint")
    parser.add_argument("--threshold", type=float, default=WAKE_THRESHOLD, help="wake sensitivity 0..1")
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=WAKE_VERIFY,
                        help="skip the whisper double-check of the wake word")
    parser.add_argument("--collect-negatives", action="store_true",
                        help="save rejected wake triggers locally for retraining")
    parser.add_argument("--stt", choices=("whisper", "google"), default=STT_BACKEND)
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help="tiny.en, base.en, small.en")
    parser.add_argument("--mute", action="store_true", help="no spoken replies")
    parser.add_argument("--mic", default=MIC_DEVICE, help="input device index or name")
    parser.add_argument("--list-mics", action="store_true", help="print input devices and exit")
    parser.add_argument("--install-autostart", action="store_true", help="start at login")
    parser.add_argument("--remove-autostart", action="store_true", help="stop starting at login")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.list_mics:
        import sounddevice as sd

        for index, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"{index:3d}  {dev['name']}")
        return 0
    if args.install_autostart or args.remove_autostart:
        import autostart

        path = autostart.install() if args.install_autostart else autostart.remove()
        print(("Installed: " if args.install_autostart else "Removed: ") + path)
        return 0

    _setup_logging()
    try:
        KernelSounder(args).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())

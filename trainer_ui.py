"""
KernalSounder — "Train my voice" wizard (about 2 minutes for the user, ~30 s of CPU).

1. Say the wake phrase 10 times.
2. Read 5 ordinary sentences (your voice saying something else).
3. 15 seconds of your room as it normally sounds.
4. personalize.personalize() runs in the background, then the result is shown
   and the app switches to the new model only if you press "Use it".
"""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from typing import Callable, Optional

from personalize import OTHER_SENTENCES, EnrollSession

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
SUCCESS = "#3fb950"
WARN = "#d29922"
DANGER = "#f85149"


def _font(size: int, weight: str = "normal"):
    return ("Segoe UI" if sys.platform.startswith("win") else "Ubuntu", size, weight)


class TrainerWizard:
    def __init__(self, root: tk.Tk, *, phrase: str, begin: Callable[[EnrollSession], bool],
                 end: Callable[[], None], train: Callable[[EnrollSession, Callable[[float, str], None]], dict],
                 apply: Callable[[bool], None]) -> None:
        self._begin, self._end, self._train, self._apply = begin, end, train, apply
        self.phrase = phrase
        self.session = EnrollSession()
        self._job: Optional[str] = None
        self._progress = (0.0, "")
        self._report: Optional[dict] = None
        self._error: Optional[str] = None
        self._recording = False

        self.win = tk.Toplevel(root)
        self.win.title("Train my voice")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        try:
            k = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except tk.TclError:
            k = 1.0
        self._k = k
        w, h = int(520 * k), int(360 * k)
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")

        self.step = tk.Label(self.win, bg=BG, fg=MUTED, font=_font(9, "bold"))
        self.step.pack(anchor="w", padx=24, pady=(20, 4))
        self.title = tk.Label(self.win, bg=BG, fg=TEXT, font=_font(17, "bold"), wraplength=int(470 * k),
                              justify="left")
        self.title.pack(anchor="w", padx=24)
        self.body = tk.Label(self.win, bg=BG, fg=TEXT, font=_font(10), wraplength=int(470 * k), justify="left")
        self.body.pack(anchor="w", padx=24, pady=(8, 10))
        self.meter = tk.Canvas(self.win, height=int(14 * k), bg=PANEL, highlightthickness=0, bd=0)
        self.meter.pack(fill=tk.X, padx=24, pady=(4, 6))
        self.status = tk.Label(self.win, bg=BG, fg=ACCENT, font=_font(10, "bold"))
        self.status.pack(anchor="w", padx=24)

        buttons = tk.Frame(self.win, bg=BG)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=20)
        self.secondary = self._button(buttons, "Cancel", self.close, MUTED)
        self.secondary.pack(side=tk.LEFT)
        self.primary = self._button(buttons, "Start", self._start_phrases, ACCENT)
        self.primary.pack(side=tk.RIGHT)
        self._intro()

    def _button(self, parent, text, command, color) -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg=PANEL, fg=color, activebackground="#30363d",
                         activeforeground=color, relief=tk.FLAT, font=_font(10, "bold"), padx=16, pady=8,
                         cursor="hand2", bd=0, highlightthickness=0)

    def _set(self, step: str, title: str, body: str = "", status: str = "", status_color: str = ACCENT) -> None:
        self.step.configure(text=step)
        self.title.configure(text=title)
        self.body.configure(text=body)
        self.status.configure(text=status, fg=status_color)

    def _bar(self, fraction: float, color: str = ACCENT) -> None:
        self.meter.delete("all")
        width = max(1, self.meter.winfo_width())
        self.meter.create_rectangle(0, 0, int(width * max(0.0, min(1.0, fraction))), int(14 * self._k),
                                    fill=color, outline="")

    def _intro(self) -> None:
        self._set("TRAIN MY VOICE", f"Teach me how you say “{self.phrase}”",
                  "About two minutes, three short steps:\n"
                  f"  1. Say “{self.phrase}” {self.session.target} times.\n"
                  f"  2. Read {len(OTHER_SENTENCES)} short sentences.\n"
                  f"  3. Stay quiet for {int(self.session.background_sec)} seconds.\n\n"
                  "Then I adapt the wake word to your voice and room (about 30 seconds). "
                  "Your recordings stay on this computer.")
        self.primary.configure(text="Start", command=self._start_phrases, state=tk.NORMAL)

    def _start_phrases(self) -> None:
        if not self._recording:
            if not self._begin(self.session):
                self._set("TRAIN MY VOICE", "No microphone", "I need a working microphone for this.", "",
                          DANGER)
                return
            self._recording = True
        self.session.start_phrases()
        self.primary.configure(text="Next", state=tk.DISABLED)
        self._poll()

    def _poll(self) -> None:
        s = self.session
        if s.mode == "phrases":
            self._set("STEP 1 OF 3", f"Say “{self.phrase}”",
                      "Speak naturally. Vary it a little: louder, softer, faster, slower.",
                      f"{len(s.clips)} / {s.target}    {s.message}")
            self._bar(s.level)
        elif s.mode == "phrases_done":
            s.start_sentences()
        elif s.mode == "sentences":
            idx = min(len(s.others), len(OTHER_SENTENCES) - 1)
            self._set("STEP 2 OF 3", f"Say: “{OTHER_SENTENCES[idx]}”",
                      "These teach me what is NOT the wake phrase in your voice.",
                      f"{len(s.others)} / {len(OTHER_SENTENCES)}    {s.message}")
            self._bar(s.level)
        elif s.mode == "sentences_done":
            s.start_background()
        elif s.mode == "background":
            self._set("STEP 3 OF 3", "Stay quiet for a moment",
                      "Let the room sound as it usually does (fan, TV or music is fine).",
                      f"{int(s.background_progress * s.background_sec)} / {int(s.background_sec)} s")
            self._bar(s.background_progress, SUCCESS)
        elif s.mode == "background_done":
            self._stop_recording()
            self._start_training()
            return
        self._job = self.win.after(80, self._poll)

    def _stop_recording(self) -> None:
        if self._recording:
            self.session.stop()
            self._recording = False
            self._end()

    def _start_training(self) -> None:
        self.primary.configure(state=tk.DISABLED)
        self.secondary.configure(state=tk.DISABLED)
        self._set("ADAPTING", "Learning your voice…", "This takes about half a minute. "
                  "You can keep using the computer.")

        def work() -> None:
            try:
                self._report = self._train(self.session, lambda p, m: setattr(self, "_progress", (p, m)))
            except Exception as exc:
                logging.exception("Personalization failed")
                self._error = str(exc)

        threading.Thread(target=work, name="ks-personalize", daemon=True).start()
        self._watch_training()

    def _watch_training(self) -> None:
        if self._report is None and self._error is None:
            fraction, message = self._progress
            self.status.configure(text=message, fg=ACCENT)
            self._bar(fraction)
            self._job = self.win.after(150, self._watch_training)
            return
        self.secondary.configure(state=tk.NORMAL)
        if self._error:
            self._set("ADAPTING", "Something went wrong", self._error, "", DANGER)
            self.primary.configure(text="Close", command=self.close, state=tk.NORMAL)
            return
        self._show_result(self._report)

    def _show_result(self, r: dict) -> None:
        self._bar(1.0, SUCCESS)
        lines = []
        if r.get("used_personal_model"):
            lines.append(f"Your held-out recordings: {r['detected_before']} → "
                         f"{r['detected_after']} of {r['held_out_clips']} detected.")
            lines.append(f"False alarms in the checks: {r['false_before']} → {r['false_after']} "
                         f"(threshold {r['threshold']:.2f}).")
        else:
            lines.append("The adapted model was not better than the original, so I kept the original.")
        if r.get("aliases"):
            lines.append("The recognizer hears you as: " + ", ".join(r["aliases"]) + ". I'll accept those too.")
        lines.append(f"Took {r.get('seconds', 0):.0f} s.")
        title = "Ready" if (r.get("used_personal_model") or r.get("aliases")) else "Nothing to change"
        self._set("DONE", title, "\n".join(lines), "Press “Use it” to switch now.", SUCCESS)
        self.primary.configure(text="Use it", command=lambda: self._finish(True), state=tk.NORMAL)
        self.secondary.configure(text="Keep the original", command=lambda: self._finish(False))

    def _finish(self, use: bool) -> None:
        try:
            self._apply(use)
        finally:
            self.close()

    def close(self) -> None:
        if self._job is not None:
            try:
                self.win.after_cancel(self._job)
            except tk.TclError:
                pass
        self._stop_recording()
        try:
            self.win.destroy()
        except tk.TclError:
            pass

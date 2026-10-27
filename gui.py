"""
KernalSounder — Tkinter operator console.

The Tk main thread only paints. Speech, NLP, subprocesses, and psutil sampling
run on worker threads and publish events through a queue so the UI cannot freeze
while the microphone is open.
"""

from __future__ import annotations

import queue
import sys
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from config import APP_NAME, APP_VERSION, GUI_QUEUE_POLL_MS, MAX_LOG_WIDGET_LINES, WORKSPACE
from security import ValidationState

BG = "#0d1117"
PANEL = "#161b22"
PANEL_ALT = "#1c2129"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
SUCCESS = "#3fb950"
WARN = "#d29922"
DANGER = "#f85149"
INPUT_BG = "#0b1016"

STATE_COLORS = {
    ValidationState.SAFE.value: SUCCESS,
    ValidationState.NEEDS_CONFIRM.value: WARN,
    ValidationState.BLOCKED.value: DANGER,
    ValidationState.ERROR.value: DANGER,
    "IDLE": MUTED,
    "LISTENING": ACCENT,
    "SPEAKING": WARN,
    "PROCESSING": ACCENT,
}


def _ui_font() -> str:
    return "Segoe UI" if sys.platform.startswith("win") else "Ubuntu"


class KernelSounderGUI:
    def __init__(self, on_toggle_listen: Callable[[], None],
                 on_confirm: Callable[[], None],
                 on_cancel: Callable[[], None],
                 on_typed: Callable[[str], None]) -> None:
        self.on_toggle_listen = on_toggle_listen
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.on_typed = on_typed

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME}  ·  Voice OS Layer")
        self.root.geometry("1040x720")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG)

        self._events: queue.Queue = queue.Queue()
        self._listening = False
        self._log_lines = 0

        self._build_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(GUI_QUEUE_POLL_MS, self._pump)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def emit(self, kind: str, **payload) -> None:
        self._events.put((kind, payload))

    def _on_close(self) -> None:
        self._events.put(("shutdown", {}))
        self.root.after(120, self.close)

    def _pump(self) -> None:
        for _ in range(48):
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            handler = self._handlers.get(kind)
            if handler:
                handler(**payload)
        try:
            self.root.after(GUI_QUEUE_POLL_MS, self._pump)
        except tk.TclError:
            pass

    def _build_style(self) -> None:
        font = _ui_font()
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, font=(font, 10))
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=(font, 18, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=(font, 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=(font, 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(font, 9))
        style.configure("Title.TLabel", background=PANEL, foreground=ACCENT, font=(font, 9, "bold"))
        style.configure("Meter.Horizontal.TProgressbar",
                        troughcolor=INPUT_BG, background=ACCENT, bordercolor=BORDER,
                        lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("Ram.Horizontal.TProgressbar",
                        troughcolor=INPUT_BG, background=WARN, bordercolor=BORDER,
                        lightcolor=WARN, darkcolor=WARN)
        style.configure("Disk.Horizontal.TProgressbar",
                        troughcolor=INPUT_BG, background=SUCCESS, bordercolor=BORDER,
                        lightcolor=SUCCESS, darkcolor=SUCCESS)

    def _card(self, parent: tk.Widget) -> tk.Frame:
        outer = tk.Frame(parent, bg=BORDER, bd=0, highlightthickness=0)
        inner = tk.Frame(outer, bg=PANEL, bd=0)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        return outer, inner

    def _build_layout(self) -> None:
        font = _ui_font()
        pad = tk.Frame(self.root, bg=BG)
        pad.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        header = tk.Frame(pad, bg=BG)
        header.pack(fill=tk.X, pady=(0, 10))

        titles = tk.Frame(header, bg=BG)
        titles.pack(side=tk.LEFT)
        tk.Label(titles, text=APP_NAME.upper(), bg=BG, fg=TEXT,
                 font=(font, 20, "bold")).pack(anchor="w")
        tk.Label(titles, text=f"Voice interaction layer  ·  v{APP_VERSION}  ·  Ubuntu sandbox",
                 bg=BG, fg=MUTED, font=(font, 9)).pack(anchor="w")

        self.mic_badge = tk.Label(
            header, text="  MIC IDLE  ", bg=PANEL_ALT, fg=MUTED,
            font=(font, 9, "bold"), padx=10, pady=6,
        )
        self.mic_badge.pack(side=tk.RIGHT, padx=(8, 0))

        self.os_badge = tk.Label(
            header, text="  LINUX  " if sys.platform.startswith("linux") else "  NOT LINUX  ",
            bg=PANEL_ALT, fg=SUCCESS if sys.platform.startswith("linux") else WARN,
            font=(font, 9, "bold"), padx=10, pady=6,
        )
        self.os_badge.pack(side=tk.RIGHT)

        meters_outer, meters = self._card(pad)
        meters_outer.pack(fill=tk.X, pady=(0, 10))
        meters.columnconfigure(1, weight=1)
        meters.columnconfigure(3, weight=1)
        meters.columnconfigure(5, weight=1)

        self.cpu_var = tk.DoubleVar(value=0)
        self.ram_var = tk.DoubleVar(value=0)
        self.disk_var = tk.DoubleVar(value=0)
        self.cpu_lbl = tk.StringVar(value="CPU  —")
        self.ram_lbl = tk.StringVar(value="RAM  —")
        self.disk_lbl = tk.StringVar(value="DISK  —")

        self._meter_row(meters, 0, self.cpu_lbl, self.cpu_var, "Meter.Horizontal.TProgressbar")
        self._meter_row(meters, 2, self.ram_lbl, self.ram_var, "Ram.Horizontal.TProgressbar")
        self._meter_row(meters, 4, self.disk_lbl, self.disk_var, "Disk.Horizontal.TProgressbar")

        body = tk.Frame(pad, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_outer, left = self._card(body)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right_outer, right = self._card(body)
        right_outer.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._section_title(left, "RECOGNIZED SPEECH")
        self.heard_var = tk.StringVar(value="Waiting for microphone or typed input…")
        tk.Label(left, textvariable=self.heard_var, bg=PANEL, fg=TEXT, wraplength=440,
                 justify="left", font=(font, 13, "bold"), anchor="w").pack(
            fill=tk.X, padx=14, pady=(4, 10)
        )

        self._section_title(left, "INTENT  ·  CONFIDENCE")
        self.intent_var = tk.StringVar(value="UNKNOWN")
        self.conf_var = tk.StringVar(value="0%")
        row = tk.Frame(left, bg=PANEL)
        row.pack(fill=tk.X, padx=14, pady=(0, 12))
        tk.Label(row, textvariable=self.intent_var, bg=PANEL, fg=ACCENT,
                 font=(font, 12, "bold")).pack(side=tk.LEFT)
        tk.Label(row, textvariable=self.conf_var, bg=PANEL, fg=MUTED,
                 font=(font, 10)).pack(side=tk.RIGHT)

        self._section_title(left, "PIPELINE STATUS")
        self.status_var = tk.StringVar(value="Idle. Press Listen or type a command.")
        tk.Label(left, textvariable=self.status_var, bg=PANEL, fg=TEXT, wraplength=440,
                 justify="left", anchor="w").pack(fill=tk.X, padx=14, pady=(0, 14))

        self._section_title(right, "VALIDATION STATE")
        self.valid_var = tk.StringVar(value="IDLE")
        self.valid_badge = tk.Label(
            right, textvariable=self.valid_var, bg=PANEL_ALT, fg=MUTED,
            font=(font, 12, "bold"), padx=12, pady=8,
        )
        self.valid_badge.pack(anchor="w", padx=14, pady=(4, 8))

        self.reason_var = tk.StringVar(value="No command has been validated yet.")
        tk.Label(right, textvariable=self.reason_var, bg=PANEL, fg=TEXT, wraplength=440,
                 justify="left", anchor="w").pack(fill=tk.X, padx=14, pady=(0, 10))

        self._section_title(right, "VALIDATED ARGV  (never raw voice)")
        self.argv_box = tk.Text(
            right, height=4, bg=INPUT_BG, fg=SUCCESS, insertbackground=TEXT,
            relief=tk.FLAT, wrap=tk.WORD, font=("Ubuntu Mono" if not sys.platform.startswith("win") else "Consolas", 10),
        )
        self.argv_box.pack(fill=tk.X, padx=14, pady=(0, 8))
        self.argv_box.insert("1.0", "(no argv)")
        self.argv_box.configure(state=tk.DISABLED)

        self.ws_var = tk.StringVar(value=f"Workspace: {WORKSPACE}")
        tk.Label(right, textvariable=self.ws_var, bg=PANEL, fg=MUTED, wraplength=440,
                 justify="left", anchor="w", font=(font, 8)).pack(
            fill=tk.X, padx=14, pady=(0, 12)
        )

        log_outer, log = self._card(pad)
        log_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self._section_title(log, "ACTIVITY  ·  EXECUTION RESULTS")
        self.log = tk.Text(
            log, height=10, bg=INPUT_BG, fg=TEXT, insertbackground=TEXT,
            relief=tk.FLAT, wrap=tk.WORD,
            font=("Ubuntu Mono" if not sys.platform.startswith("win") else "Consolas", 9),
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.log.configure(state=tk.DISABLED)
        self.log.tag_configure("ok", foreground=SUCCESS)
        self.log.tag_configure("bad", foreground=DANGER)
        self.log.tag_configure("warn", foreground=WARN)
        self.log.tag_configure("info", foreground=MUTED)

        controls = tk.Frame(pad, bg=BG)
        controls.pack(fill=tk.X)

        self.listen_btn = self._button(controls, "Listen", self.on_toggle_listen, ACCENT)
        self.listen_btn.pack(side=tk.LEFT)
        self.confirm_btn = self._button(controls, "Confirm Delete", self.on_confirm, SUCCESS)
        self.confirm_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.cancel_btn = self._button(controls, "Cancel", self.on_cancel, DANGER)
        self.cancel_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._set_confirm_enabled(False)

        self.entry = tk.Entry(
            controls, bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT,
            font=(font, 11), highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=8)
        self.entry.bind("<Return>", self._submit_typed)
        self.entry.insert(0, "Type a command if the VM has no microphone…")
        self.entry.bind("<FocusIn>", self._clear_placeholder)

        go = self._button(controls, "Execute", lambda: self._submit_typed(None), ACCENT)
        go.pack(side=tk.RIGHT)

        self._handlers: Dict[str, Callable[..., None]] = {
            "metrics": self._on_metrics,
            "listen": self._on_listen,
            "partial": self._on_partial,
            "heard": self._on_heard,
            "intent": self._on_intent,
            "validation": self._on_validation,
            "status": self._on_status,
            "log": self._on_log,
            "confirm_ready": self._on_confirm_ready,
            "error": self._on_error,
        }

    def _meter_row(self, parent: tk.Frame, col: int, label: tk.StringVar,
                   value: tk.DoubleVar, style_name: str) -> None:
        tk.Label(parent, textvariable=label, bg=PANEL, fg=MUTED, width=18,
                 anchor="w").grid(row=0, column=col, padx=(12, 6), pady=12, sticky="w")
        bar = ttk.Progressbar(parent, variable=value, maximum=100, style=style_name)
        bar.grid(row=0, column=col + 1, padx=(0, 12), pady=12, sticky="ew")

    def _section_title(self, parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text, bg=PANEL, fg=MUTED,
                 font=(_ui_font(), 8, "bold")).pack(anchor="w", padx=14, pady=(12, 2))

    def _button(self, parent: tk.Widget, text: str, command: Callable[[], None],
                color: str) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, bg=PANEL_ALT, fg=color,
            activebackground=BORDER, activeforeground=color, relief=tk.FLAT,
            font=(_ui_font(), 10, "bold"), padx=14, pady=8, cursor="hand2",
            highlightthickness=0, bd=0,
        )

    def _clear_placeholder(self, _event: Optional[tk.Event] = None) -> None:
        if self.entry.get().startswith("Type a command"):
            self.entry.delete(0, tk.END)

    def _submit_typed(self, _event: Optional[tk.Event]) -> None:
        text = self.entry.get().strip()
        if not text or text.startswith("Type a command"):
            return
        self.entry.delete(0, tk.END)
        self.on_typed(text)

    def _set_confirm_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.confirm_btn.configure(state=state)
        self.cancel_btn.configure(state=state)

    def _on_metrics(self, cpu: float, ram: float, ram_text: str, disk: float) -> None:
        self.cpu_var.set(cpu)
        self.ram_var.set(ram)
        self.disk_var.set(disk)
        self.cpu_lbl.set(f"CPU  {cpu:.0f}%")
        self.ram_lbl.set(f"RAM  {ram_text}")
        self.disk_lbl.set(f"DISK {disk:.0f}%")

    def _on_listen(self, active: bool) -> None:
        self._listening = active
        if active:
            self.mic_badge.configure(text="  MIC LISTENING  ", fg=ACCENT)
            self.listen_btn.configure(text="Stop")
        else:
            self.mic_badge.configure(text="  MIC IDLE  ", fg=MUTED)
            self.listen_btn.configure(text="Listen")

    def _on_partial(self, text: str) -> None:
        if text:
            self.heard_var.set(text)

    def _on_heard(self, text: str) -> None:
        self.heard_var.set(text or "(empty)")

    def _on_intent(self, intent: str, confidence: float, slots: str) -> None:
        self.intent_var.set(intent)
        extra = f"  ·  {slots}" if slots else ""
        self.conf_var.set(f"{confidence * 100:.0f}%{extra}")

    def _on_validation(self, state: str, reason: str, argv: str) -> None:
        self.valid_var.set(state)
        color = STATE_COLORS.get(state, MUTED)
        self.valid_badge.configure(fg=color)
        self.reason_var.set(reason)
        self.argv_box.configure(state=tk.NORMAL)
        self.argv_box.delete("1.0", tk.END)
        self.argv_box.insert("1.0", argv or "(no argv — raw voice is never executed)")
        self.argv_box.configure(state=tk.DISABLED)
        self._set_confirm_enabled(state == ValidationState.NEEDS_CONFIRM.value)

    def _on_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_log(self, text: str, tag: str = "info") -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text.rstrip() + "\n", tag)
        self._log_lines += 1
        if self._log_lines > MAX_LOG_WIDGET_LINES:
            self.log.delete("1.0", "40.0")
            self._log_lines -= 39
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_confirm_ready(self, ready: bool) -> None:
        self._set_confirm_enabled(ready)

    def _on_error(self, text: str) -> None:
        self.status_var.set(text)
        self._on_log(text, "bad")

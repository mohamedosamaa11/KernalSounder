"""
KernalSounder — floating voice overlay.

A small always-on-top pill that stays out of the way while idle and expands
into a live waveform bar when the wake word fires, similar to a phone
assistant. Pure Tk canvas drawing: no images, no extra packages, and the
animation timer only runs while the bar is open, so idle CPU stays at ~0.
"""

from __future__ import annotations

import math
import random
import time
import tkinter as tk
from typing import Callable, Dict, List, Optional

from config import APP_NAME, IS_WINDOWS, RESULT_LINGER_SEC

KEY = "#010203"
PANEL = "#2f3239"
BORDER = "#464a54"
TEXT = "#eceef3"
MUTED = "#9aa0ad"
DOT = "#3b445a"
BLUE = (47, 123, 255)
VIOLET = (146, 140, 255)
DOT_RGB = (59, 68, 90)
BTN = "#484c56"
BTN_HOVER = "#585d69"
SEND = "#3a73f0"
SEND_HOVER = "#5287ff"
DANGER = "#e5484d"
DANGER_HOVER = "#f2666a"
TONES = {"info": TEXT, "muted": MUTED, "ok": "#56d364", "bad": "#ff7b72", "warn": "#e3b341"}

COMPACT_W, COMPACT_H = 236, 52
WIDE_W, WIDE_H = 620, 170
SLOTS = 21
FRAME_MS = 33


def _font(size: int, weight: str = "normal"):
    return ("Segoe UI" if IS_WINDOWS else "Ubuntu", size, weight)


def _mix(a, b, t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _bar_color(amp: float) -> str:
    if amp < 0.12:
        return _mix(DOT_RGB, BLUE, amp / 0.12)
    return _mix(BLUE, VIOLET, (amp - 0.12) / 0.88)


class VoiceOverlay:
    def __init__(self, root: tk.Tk, *, hint: str, show_idle_pill: bool,
                 on_talk: Callable[[], None], on_stop: Callable[[], None],
                 on_send: Callable[[], None], on_confirm: Callable[[], None],
                 on_cancel: Callable[[], None], on_dashboard: Callable[[], None],
                 on_quit: Callable[[], None], on_train: Optional[Callable[[], None]] = None) -> None:
        self._cb = dict(talk=on_talk, stop=on_stop, send=on_send, confirm=on_confirm,
                        cancel=on_cancel, dashboard=on_dashboard, quit=on_quit, train=on_train)
        self._hint = hint
        self._show_idle_pill = show_idle_pill
        self._phase = "idle"
        self._text = ""
        self._tone = "info"
        self._level = 0.0
        self._amps = [0.0] * SLOTS
        self._jitter = [random.random() for _ in range(SLOTS)]
        self._bars: List[int] = []
        self._anim_job: Optional[str] = None
        self._linger_job: Optional[str] = None
        self._drag: Optional[Dict[str, int]] = None
        self._size = (COMPACT_W, COMPACT_H)

        self.win = tk.Toplevel(root)
        self.win.withdraw()
        try:
            self._k = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except tk.TclError:
            self._k = 1.0
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        bg = PANEL
        if IS_WINDOWS:
            bg = KEY
            self.win.configure(bg=KEY)
            try:
                self.win.attributes("-transparentcolor", KEY)
            except tk.TclError:
                bg = PANEL
        try:
            self.win.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(self.win, bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self._anchor = [sw // 2, sh - int((64 if IS_WINDOWS else 48) * self._k)]

        self._menu = tk.Menu(self.win, tearoff=0)
        self._menu.add_command(label="Talk now", command=lambda: self._fire("talk"))
        self._menu.add_command(label="Open dashboard", command=lambda: self._fire("dashboard"))
        if on_train is not None:
            self._menu.add_command(label="Train my voice\u2026", command=lambda: self._fire("train"))
        self._menu.add_separator()
        self._menu.add_command(label=f"Quit {APP_NAME}", command=lambda: self._fire("quit"))

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._popup)
        self._render()

    def _fire(self, name: str) -> None:
        callback = self._cb.get(name)
        if callback:
            callback()

    def _lw(self, value: float) -> int:
        return max(1, round(value * self._k))

    def _place(self, w: int, h: int) -> None:
        self._size = (w, h)
        pw, ph = int(w * self._k), int(h * self._k)
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = min(max(0, self._anchor[0] - pw // 2), max(0, sw - pw))
        y = min(max(0, self._anchor[1] - ph), max(0, sh - ph))
        self.win.geometry(f"{pw}x{ph}+{x}+{y}")

    def _round_rect(self, x1, y1, x2, y2, r, **kw) -> int:
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, splinesteps=24, **kw)

    def set_hint(self, text: str) -> None:
        self._hint = text
        if self._phase == "idle":
            self._render()

    def set_level(self, value: float) -> None:
        self._level = max(self._level * 0.6, float(value))

    def set_text(self, text: str, tone: str = "info") -> None:
        self._text, self._tone = text, tone
        if self._phase != "idle" and self.canvas.find_withtag("label"):
            self.canvas.itemconfigure("label", text=text, fill=TONES.get(tone, TEXT))

    def set_phase(self, phase: str, text: str = "", tone: str = "info") -> None:
        if self._linger_job:
            self.win.after_cancel(self._linger_job)
            self._linger_job = None
        changed_shape = (phase == "idle") != (self._phase == "idle") or phase != self._phase
        self._phase = phase
        self._text, self._tone = text, tone
        if changed_shape:
            self._render()
        else:
            self.set_text(text, tone)
        if phase == "result":
            self._linger_job = self.win.after(int(RESULT_LINGER_SEC * 1000), lambda: self.set_phase("idle"))
        if phase in ("listening", "thinking", "confirm", "result"):
            self._start_anim()

    def _render(self) -> None:
        c = self.canvas
        c.delete("all")
        self._bars = []
        if self._phase == "idle":
            self._stop_anim()
            if not self._show_idle_pill:
                self.win.withdraw()
                return
            self._render_compact()
        else:
            self._render_wide()
        if self._k != 1.0:
            self.canvas.scale("all", 0, 0, self._k, self._k)
        self.win.deiconify()
        self.win.lift()
        self.win.attributes("-topmost", True)

    def _render_compact(self) -> None:
        w, h = COMPACT_W, COMPACT_H
        self._place(w, h)
        c = self.canvas
        self._round_rect(1, 1, w - 2, h - 2, h // 2 - 1, fill=PANEL, outline=BORDER, tags=("bg", "talk"))
        cx, cy = 27, h // 2
        c.create_oval(cx - 17, cy - 17, cx + 17, cy + 17, fill=SEND, outline="", tags=("talk", "mic"))
        c.create_line(cx, cy - 7, cx, cy + 1, width=self._lw(7), capstyle=tk.ROUND, fill="white", tags="talk")
        c.create_arc(cx - 8, cy - 9, cx + 8, cy + 6, start=200, extent=140, style=tk.ARC,
                     outline="white", width=self._lw(2), tags="talk")
        c.create_line(cx, cy + 6, cx, cy + 10, width=self._lw(2), fill="white", tags="talk")
        c.create_text(54, cy, text=self._hint, anchor="w", fill=TEXT, font=_font(10), tags="talk")
        mx = w - 24
        for dy in (-5, 0, 5):
            c.create_line(mx - 6, cy + dy, mx + 6, cy + dy, fill=MUTED, width=self._lw(2), capstyle=tk.ROUND,
                          tags="menu")
        c.create_rectangle(mx - 12, cy - 12, mx + 12, cy + 12, outline="", fill="", tags="menu")
        c.tag_bind("menu", "<ButtonRelease-1>", self._menu_click)
        c.tag_bind("mic", "<Enter>", lambda _e: c.itemconfigure("mic", fill=SEND_HOVER))
        c.tag_bind("mic", "<Leave>", lambda _e: c.itemconfigure("mic", fill=SEND))

    def _render_wide(self) -> None:
        w, h = WIDE_W, WIDE_H
        self._place(w, h)
        c = self.canvas
        self._round_rect(1, 1, w - 2, h - 2, 30, fill=PANEL, outline=BORDER, tags="bg")
        c.create_text(30, 30, text=self._text, anchor="w", width=int((w - 60) * self._k),
                      fill=TONES.get(self._tone, TEXT), font=_font(12), tags="label")
        c.create_text(30, h - 30, text=APP_NAME, anchor="w", fill=MUTED, font=_font(8, "bold"))

        cy = 80
        left, right = 40, w - 40
        step = (right - left) / (SLOTS - 1)
        for i in range(SLOTS):
            x = left + i * step
            self._bars.append(c.create_line(x, cy, x, cy + 0.01, width=self._lw(9), capstyle=tk.ROUND, fill=DOT))
        self._amps = [0.0] * SLOTS

        if self._phase == "confirm":
            self._button(w - 176, h - 52, "cancel", BTN, BTN_HOVER, self._glyph_cross)
            self._button(w - 96, h - 52, "confirm", DANGER, DANGER_HOVER, self._glyph_check)
        elif self._phase == "listening":
            self._button(w - 176, h - 52, "stop", BTN, BTN_HOVER, self._glyph_square)
            self._button(w - 96, h - 52, "send", SEND, SEND_HOVER, self._glyph_arrow)
        else:
            self._button(w - 96, h - 52, "stop", BTN, BTN_HOVER, self._glyph_cross)

    def _button(self, x: int, y: int, name: str, color: str, hover: str, glyph) -> None:
        c = self.canvas
        tag = f"btn_{name}"
        body = f"{tag}_body"
        self._round_rect(x, y, x + 72, y + 40, 20, fill=color, outline="", tags=(tag, body))
        glyph(x + 36, y + 20, tag)
        c.tag_bind(tag, "<Enter>", lambda _e: (c.itemconfigure(body, fill=hover), c.configure(cursor="hand2")))
        c.tag_bind(tag, "<Leave>", lambda _e: (c.itemconfigure(body, fill=color), c.configure(cursor="")))
        c.tag_bind(tag, "<ButtonRelease-1>", lambda _e, n=name: self._button_click(n))

    def _glyph_square(self, x, y, tag) -> None:
        self.canvas.create_rectangle(x - 8, y - 8, x + 8, y + 8, fill="#e6e8f0", outline="", tags=tag)

    def _glyph_arrow(self, x, y, tag) -> None:
        for coords in ((x, y + 10, x, y - 10), (x - 9, y - 1, x, y - 10), (x + 9, y - 1, x, y - 10)):
            self.canvas.create_line(*coords, fill="white", width=self._lw(3), capstyle=tk.ROUND, tags=tag)

    def _glyph_cross(self, x, y, tag) -> None:
        for coords in ((x - 7, y - 7, x + 7, y + 7), (x - 7, y + 7, x + 7, y - 7)):
            self.canvas.create_line(*coords, fill="#e6e8f0", width=self._lw(3), capstyle=tk.ROUND, tags=tag)

    def _glyph_check(self, x, y, tag) -> None:
        self.canvas.create_line(x - 9, y, x - 3, y + 7, x + 9, y - 7, fill="white", width=self._lw(3),
                                capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tag)

    def _button_click(self, name: str) -> None:
        self._drag = None
        self._fire(name)

    def _menu_click(self, event) -> None:
        self._drag = None
        self._popup(event)

    def _popup(self, event) -> None:
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def _press(self, event) -> None:
        current = self.canvas.find_withtag("current")
        tags = self.canvas.gettags(current[0]) if current else ()
        if any(t.startswith("btn_") or t == "menu" for t in tags):
            self._drag = None
            return
        self._drag = {"x": event.x_root, "y": event.y_root, "ax": self._anchor[0],
                      "ay": self._anchor[1], "moved": 0, "talk": int("talk" in tags)}

    def _motion(self, event) -> None:
        if not self._drag:
            return
        dx, dy = event.x_root - self._drag["x"], event.y_root - self._drag["y"]
        if abs(dx) + abs(dy) > 4:
            self._drag["moved"] = 1
        self._anchor = [self._drag["ax"] + dx, self._drag["ay"] + dy]
        self._place(*self._size)

    def _release(self, _event) -> None:
        drag, self._drag = self._drag, None
        if drag and not drag["moved"] and drag["talk"] and self._phase == "idle":
            self._fire("talk")

    def _start_anim(self) -> None:
        if self._anim_job is None:
            self._anim_job = self.win.after(FRAME_MS, self._tick)

    def _stop_anim(self) -> None:
        if self._anim_job is not None:
            try:
                self.win.after_cancel(self._anim_job)
            except tk.TclError:
                pass
            self._anim_job = None

    def _tick(self) -> None:
        self._anim_job = None
        if self._phase == "idle" or not self._bars:
            return
        now = time.monotonic()
        center = (SLOTS - 1) / 2
        targets = []
        for i in range(SLOTS):
            env = math.exp(-((i - center) / (SLOTS * 0.2)) ** 2)
            if self._phase in ("listening", "confirm"):
                self._jitter[i] += (random.random() - self._jitter[i]) * 0.25
                targets.append(self._level * env * (0.45 + 0.75 * self._jitter[i]))
            elif self._phase == "thinking":
                wave = 0.5 + 0.5 * math.sin(now * 6.0 - i * 0.55)
                targets.append(0.08 + 0.32 * wave * env)
            else:
                targets.append(0.0)
        settled = True
        for i, target in enumerate(targets):
            rate = 0.45 if target > self._amps[i] else 0.2
            self._amps[i] += (target - self._amps[i]) * rate
            if abs(self._amps[i] - target) > 0.004:
                settled = False
        self._level *= 0.88
        self._update_bars()
        if self._phase == "result" and settled:
            return
        self._anim_job = self.win.after(FRAME_MS, self._tick)

    def _update_bars(self) -> None:
        c = self.canvas
        k = self._k
        cy = 80 * k
        left, right = 40, self._size[0] - 40
        step = (right - left) / (SLOTS - 1)
        for i, item in enumerate(self._bars):
            amp = min(1.0, self._amps[i])
            x = (left + i * step) * k
            half = amp * 32 * k
            width = (9 + 9 * min(1.0, amp * 4)) * k
            c.coords(item, x, cy - half, x, cy + half + 0.01)
            c.itemconfigure(item, width=width, fill=_bar_color(amp))

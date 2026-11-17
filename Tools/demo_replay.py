"""Replay a recording through the real-time system with the same look as main.py.

Main view:   camera on the left, thresholded-PIR and fused decisions on the right.
Sensor view: camera (top left), PIR signal (bottom left), thermopile stages (right, large).

The sensors are fed to the real-time system causally, exactly as in the live demo.
Seeking backwards resets the system and reprocesses the recording up to the new time,
so every decision shown still only depends on data received before it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tkinter as tk

import cv2
import numpy as np
import pandas as pd

# Import package from project root even when this script is run by filepath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Fusion.realtimeModel import RealtimeModel
from PIR.processing import PIRProcessor, threshold as PIR_THRESHOLD
from Thermopile.processing import ThermopilePipeline
from Tools.synchronised_playback import (
    THERMAL_STAGES, _diverging, _find_video, _nearest_index, _pir_display_range, _thermal_display_range,
    _thermal_stage_view,
)

WINDOW_TITLE = "Outdoor Intrusion Detection — Recorded Demo"
MODEL_PATH = Path("Processed/realtime/fusion_gb.json")
CONFIDENCE_HISTORY_S = 20.0
PIR_DISPLAY_S = 15.0
SKIP_S = 5.0
TICK_MS = 30
VIDEO_SEEK_FRAMES = 90      # forward jumps larger than this re-position the decoder instead of grabbing
WINDOW_SIZE = "1280x820"
STAGE_TABS = ("Raw", "Background", "Difference", "Z score", "Filtered Z", "Filtered Z avg", "Ratio", "Mask")

# Same look as main.py (copied: main.py imports the sensor drivers, so it cannot be imported on the PC)
THEME = {
    "bg": "#ffffff", "card": "#f6f8fb", "raised": "#e9eef4", "border": "#cfd7e2",
    "text": "#0f172a", "muted": "#475569", "accent": "#1d4ed8", "accent_hover": "#2563eb",
    "green": "#16a34a", "yellow": "#facc15", "red": "#dc2626", "off": "#dde3ea",
}
STATE_COLOURS = {"NO INTRUDER": THEME["green"], "MAYBE INTRUDER": THEME["yellow"], "INTRUDER": THEME["red"]}
STATE_TEXT = {"NO INTRUDER": "#15803d", "MAYBE INTRUDER": "#a16207", "INTRUDER": "#b91c1c"}
HUMAN_SHADE = "#fecaca"


class Replay:
    """Feeds the recording to the real-time system in time order."""

    def __init__(self, recording, model_path):
        self.recording = recording
        self.model_path = model_path
        with (recording / "metadata.json").open(encoding="utf-8") as f:
            self.meta = json.load(f)
        pir = pd.read_csv(recording / "pir.csv").sort_values("time_s")
        thermal = pd.read_csv(recording / "thermopile.csv").sort_values("time_s")
        self.pir_t = pir["time_s"].to_numpy(dtype=float)
        self.pir_v = pir["voltage"].to_numpy(dtype=float)
        pixel_cols = [f"pixel_{i}" for i in range(768)]
        self.thermal_t = thermal["time_s"].to_numpy(dtype=float)
        self.thermal_pixels = thermal[pixel_cols].to_numpy(dtype=np.float64).reshape(-1, 24, 32)
        self.raw_range = _thermal_display_range(thermal)
        self.pir_range = _pir_display_range(self.pir_v)
        self.labels = None
        if (recording / "labels.csv").is_file():
            self.labels = pd.read_csv(recording / "labels.csv")
        self.reset()

    def reset(self):
        self.model = RealtimeModel(self.model_path)
        self.window = self.model.window
        self.pir = PIRProcessor(initialBaseline=self.meta["pir_baseline_v"], windowTime=self.window)
        self.thermal = ThermopilePipeline(window_seconds=self.window)
        self.thermal.load_calibration(self.meta["thermal_background"], self.meta["thermal_sigma"])
        self.next_pir = 0
        self.next_thermal = 0
        self.result = None
        self.status = f"Filling the {self.window:g} s PIR window..."
        self.features = None
        self.background_before = self.thermal.bg_model.bg.copy()
        self.confidence_history = []
        self.process_ms = 0.0

    # Process Every PIR Sample And Thermal Frame Up To Time t (causal, like the live system)
    # PIR samples and thermal frames are interleaved in time order, so a large jump (seek or --start)
    # gives exactly the same decisions as playing through: each frame only sees PIR samples up to its own time.
    def _feed_pir(self, t):
        while self.next_pir < len(self.pir_t) and self.pir_t[self.next_pir] <= t:
            self.pir.processSample(self.pir_t[self.next_pir], self.pir_v[self.next_pir])
            self.next_pir += 1

    def advance(self, t):
        while self.next_thermal < len(self.thermal_t) and self.thermal_t[self.next_thermal] <= t:
            self._feed_pir(self.thermal_t[self.next_thermal])
            started = time.perf_counter()
            timestamp = float(self.thermal_t[self.next_thermal])
            self.background_before = self.thermal.bg_model.bg.copy()
            self.features = self.thermal.process(self.thermal_pixels[self.next_thermal], timestamp)
            samples = self.pir.getWindowSamples(self.window, endTime=timestamp)
            result, problem = self.model.fuse(self.features, samples, timestamp)
            if result is not None:
                self.result = result
                self.status = "Live decision"
                self.confidence_history.append((timestamp, result["fusion_confidence"]))
            else:
                self.status = problem
            self.process_ms = (time.perf_counter() - started) * 1000.0
            self.next_thermal += 1
        self._feed_pir(t)

    def human_at(self, t):
        if self.labels is None or self.labels.empty:
            return None
        active = (self.labels["start_s"] <= t) & (t < self.labels["end_s"]) & (self.labels["human"] == 1)
        return bool(active.any())

    def frame_stages(self):
        p = self.thermal
        shift = 0.0 if not p.last_frame_valid else p.last_shift
        raw = self.thermal_pixels[max(self.next_thermal - 1, 0)]
        return {
            "raw": raw, "background": self.background_before,
            "difference": raw - (self.background_before + shift),
            "z": p.last_raw_z, "smoothed_z": p.last_smoothed_z, "smoothed_z_avg": p.last_smoothed_z_avg,
            "ratio": p.last_ratio, "mask": p.last_mask,
        }

    def blobs(self):
        p = self.thermal
        out = []
        for d in p.last_detections:
            track = p._tracks.get(d.features.track_id)
            out.append((d.blob.bbox, d.features.track_id, d.features.track_id == p.primary_track_id,
                        track.traversed_px if track is not None else 0.0))
        return out


# ---------------------------------------------------------------- Widgets (main.py style)

def pick_font(root):
    from tkinter import font as tkfont
    families = set(tkfont.families(root))
    for name in ("Inter", "Segoe UI", "DejaVu Sans", "Helvetica"):
        if name in families:
            return name
    return "TkDefaultFont"


def state_text(state):
    return state if state in STATE_COLOURS else "--"


def to_photo(rgb):
    height, width = rgb.shape[:2]
    data = f"P6 {width} {height} 255 ".encode("ascii") + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def fit(width, height, aspect):
    """Largest width x height box with the given aspect (w / h) inside the space."""
    if width / max(height, 1) > aspect:
        return max(int(height * aspect), 1), max(int(height), 1)
    return max(int(width), 1), max(int(width / aspect), 1)


def format_time(seconds):
    seconds = max(seconds, 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


class Card(tk.Frame):
    def __init__(self, parent, fonts, title, subtitle=None):
        super().__init__(parent, bg=THEME["card"], highlightthickness=1,
                         highlightbackground=THEME["border"], padx=16, pady=10)
        self.header = tk.Frame(self, bg=THEME["card"])
        self.header.pack(fill="x")
        tk.Label(self.header, text=title, bg=THEME["card"], fg=THEME["text"],
                 font=fonts["card"]).pack(side="left", anchor="w")
        self.subtitle = None
        if subtitle:
            self.subtitle = tk.Label(self, text=subtitle, bg=THEME["card"], fg=THEME["muted"], font=fonts["small"],
                                     anchor="w", justify="left")
            self.subtitle.pack(fill="x", pady=(2, 8))


class Lamps(tk.Canvas):
    def __init__(self, parent, fonts, states, width=380):
        super().__init__(parent, width=width, height=80, bg=THEME["card"], highlightthickness=0)
        self.states = states
        self.items = []
        step = width / len(states)
        for i, (label, colour) in enumerate(states):
            x = step * (i + 0.5)
            glow = self.create_oval(x - 27, 3, x + 27, 57, outline="", fill="")
            lamp = self.create_oval(x - 20, 10, x + 20, 50, fill=THEME["off"], outline=THEME["border"], width=2)
            text = self.create_text(x, 70, text=label, fill=THEME["muted"], font=fonts["lamp"])
            self.items.append((glow, lamp, text, colour))

    def set_state(self, state):
        for (label, _), (glow, lamp, text, colour) in zip(self.states, self.items):
            lit = label == state
            self.itemconfigure(lamp, fill=colour if lit else THEME["off"],
                               outline=colour if lit else THEME["border"])
            self.itemconfigure(glow, outline=colour if lit else "", width=3)
            self.itemconfigure(text, fill=THEME["text"] if lit else THEME["muted"])


def make_button(parent, fonts, text, command, padx=18):
    button = tk.Button(parent, text=text, command=command, takefocus=0, font=fonts["button"],
                       bg=THEME["accent"], fg="white", activebackground=THEME["accent_hover"],
                       activeforeground="white", relief="flat", bd=0, padx=padx, pady=8, cursor="hand2")
    button.bind("<Enter>", lambda _e: button.configure(bg=THEME["accent_hover"]))
    button.bind("<Leave>", lambda _e: button.configure(bg=THEME["accent"]))
    return button


class VideoPanel(tk.Canvas):
    """Camera frame scaled to the panel, with the ground-truth label on top."""

    def __init__(self, parent, fonts):
        super().__init__(parent, bg=THEME["raised"], highlightthickness=0, width=320, height=180)
        self.fonts = fonts
        self.photo = None
        self.drawn = None

    def show(self, frame, frame_id, human):
        width, height = self.winfo_width(), self.winfo_height()
        key = (frame_id, width, height, human)
        if key == self.drawn or width < 20 or height < 20:
            return
        self.drawn = key
        w, h = fit(width, height, frame.shape[1] / frame.shape[0])
        image = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.photo = to_photo(image)
        self.delete("all")
        self.create_image(width // 2, height // 2, image=self.photo)
        if human is not None:
            x0, y0 = (width - w) // 2 + 10, (height - h) // 2 + 10
            text = "Ground truth: PERSON IN ZONE" if human else "Ground truth: zone empty"
            label = self.create_text(x0 + 10, y0 + 6, anchor="nw", text=text, font=self.fonts["body_bold"],
                                     fill=STATE_TEXT["INTRUDER"] if human else THEME["muted"])
            box = self.bbox(label)
            back = self.create_rectangle(box[0] - 8, box[1] - 4, box[2] + 8, box[3] + 4, fill="white",
                                         outline=THEME["border"])
            self.tag_lower(back, label)


class Timeline(tk.Canvas):
    """Clickable / draggable time bar. Labelled human intervals are shaded red."""

    def __init__(self, parent, start, end, intervals, on_seek):
        super().__init__(parent, height=34, bg=THEME["card"], highlightthickness=0, cursor="hand2")
        self.start, self.end = start, end
        self.intervals = intervals
        self.on_seek = on_seek
        self.time = start
        self.dragging = False
        self.bind("<Configure>", lambda _e: self.draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def _x(self, t):
        left, right = 10, max(self.winfo_width() - 10, 11)
        return left + (t - self.start) / max(self.end - self.start, 1e-9) * (right - left)

    def _t(self, x):
        left, right = 10, max(self.winfo_width() - 10, 11)
        return self.start + min(max((x - left) / (right - left), 0.0), 1.0) * (self.end - self.start)

    def _press(self, event):
        self.dragging = True
        self.time = self._t(event.x)
        self.draw()

    def _drag(self, event):
        self.time = self._t(event.x)
        self.draw()

    def _release(self, event):
        self.dragging = False
        self.on_seek(self._t(event.x))

    def set_time(self, t):
        if not self.dragging and abs(t - self.time) > 1e-6:
            self.time = t
            self.draw()

    def draw(self):
        self.delete("all")
        mid = self.winfo_height() // 2
        self.create_rectangle(self._x(self.start), mid - 5, self._x(self.end), mid + 5, fill=THEME["raised"],
                              outline=THEME["border"])
        for start, end in self.intervals:
            self.create_rectangle(self._x(max(start, self.start)), mid - 5, self._x(min(end, self.end)), mid + 5,
                                  fill=HUMAN_SHADE, outline="")
        x = self._x(self.time)
        self.create_rectangle(self._x(self.start), mid - 2, x, mid + 2, fill=THEME["accent"], outline="")
        self.create_oval(x - 8, mid - 8, x + 8, mid + 8, fill=THEME["accent"], outline="white", width=2)


def draw_confidence(canvas, fonts, history, now, thresholds):
    canvas.delete("all")
    width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 60)
    left, right, top, bottom = 36, width - 12, 10, height - 22

    def y(value):
        return bottom - value * (bottom - top)

    for level in (0.0, 0.5, 1.0):
        canvas.create_line(left, y(level), right, y(level), fill=THEME["border"])
        canvas.create_text(left - 6, y(level), text=f"{level:.1f}", anchor="e", fill=THEME["muted"], font=fonts["small"])
    for key, name, colour, x, anchor in (("maybe_enter", "maybe", STATE_TEXT["MAYBE INTRUDER"], left + 4, "sw"),
                                         ("intruder_enter", "intruder", STATE_TEXT["INTRUDER"], right - 4, "se")):
        level = float(thresholds[key])
        canvas.create_line(left, y(level), right, y(level), fill=colour, dash=(4, 4))
        canvas.create_text(x, y(level) - 2, text=f"{name} {level:.2f}", anchor=anchor, fill=colour,
                           font=fonts["small"])
    canvas.create_text(left, bottom + 12, text=f"-{CONFIDENCE_HISTORY_S:g} s", anchor="w", fill=THEME["muted"],
                       font=fonts["small"])
    canvas.create_text(right, bottom + 12, text="now", anchor="e", fill=THEME["muted"], font=fonts["small"])
    start = now - CONFIDENCE_HISTORY_S
    points = []
    for t, c in history:
        if start <= t <= now:
            points += [left + (t - start) / CONFIDENCE_HISTORY_S * (right - left), y(c)]
    if len(points) >= 4:
        canvas.create_line(*points, fill=THEME["accent"], width=2)
    if points:
        canvas.create_oval(points[-2] - 4, points[-1] - 4, points[-2] + 4, points[-1] + 4, fill=THEME["accent"],
                           outline="")


# ---------------------------------------------------------------- Views

class MainView(tk.Frame):
    """Camera on the left, thresholded PIR and fused decisions on the right."""

    def __init__(self, parent, fonts):
        super().__init__(parent, bg=THEME["bg"])
        self.fonts = fonts
        self.columnconfigure(0, weight=3, uniform="main")
        self.columnconfigure(1, weight=2, uniform="main")
        self.rowconfigure(0, weight=1)

        camera = Card(self, fonts, "Camera")
        camera.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.video = VideoPanel(camera, fonts)
        self.video.pack(fill="both", expand=True, pady=(8, 0))

        right = tk.Frame(self, bg=THEME["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.pir_card = Card(right, fonts, "Thresholded PIR", " ")
        self.pir_card.pack(fill="x")
        self.pir_lamps = Lamps(self.pir_card, fonts, [("NO INTRUDER", THEME["green"]), ("INTRUDER", THEME["red"])])
        self.pir_lamps.pack()
        self.pir_state = tk.Label(self.pir_card, text="--", bg=THEME["card"], fg=THEME["muted"], font=fonts["state"])
        self.pir_state.pack(pady=(2, 0))
        self.pir_detail = tk.Label(self.pir_card, text="", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"])
        self.pir_detail.pack()

        fused = Card(right, fonts, "Fused System",
                     "PIR and thermopile features combined by the Gradient Boosting classifier")
        fused.pack(fill="x", pady=(10, 0))
        self.fused_lamps = Lamps(fused, fonts, [("NO INTRUDER", THEME["green"]), ("MAYBE INTRUDER", THEME["yellow"]),
                                                ("INTRUDER", THEME["red"])])
        self.fused_lamps.pack()
        self.fused_state = tk.Label(fused, text="--", bg=THEME["card"], fg=THEME["muted"], font=fonts["state"])
        self.fused_state.pack(pady=(2, 2))
        probability = tk.Frame(fused, bg=THEME["card"])
        probability.pack(fill="x")
        tk.Label(probability, text="Human probability", bg=THEME["card"], fg=THEME["muted"],
                 font=fonts["body"]).pack(side="left")
        self.probability_text = tk.Label(probability, text="--", bg=THEME["card"], fg=THEME["text"],
                                         font=fonts["body_bold"])
        self.probability_text.pack(side="right")
        self.probability_bar = tk.Canvas(fused, height=8, bg=THEME["raised"], highlightthickness=0)
        self.probability_bar.pack(fill="x", pady=(4, 0))
        self.probability_fill = self.probability_bar.create_rectangle(0, 0, 0, 8, fill=THEME["accent"], width=0)

        chart_card = Card(right, fonts, "Fused probability")
        chart_card.pack(fill="both", expand=True, pady=(10, 0))
        tk.Label(chart_card.header, text=f"last {CONFIDENCE_HISTORY_S:g} s vs alarm thresholds", bg=THEME["card"],
                 fg=THEME["muted"], font=fonts["small"]).pack(side="right")
        self.chart = tk.Canvas(chart_card, bg="white", highlightthickness=1, highlightbackground=THEME["border"],
                               height=120)
        self.chart.pack(fill="both", expand=True, pady=(6, 0))

    def update_view(self, replay, now, frame, frame_id):
        self.video.show(frame, frame_id, replay.human_at(now))
        threshold = replay.model.pir_threshold
        self.pir_card.subtitle.configure(text=f"Intruder when PIR is more than ±{threshold:.2f} V from its baseline")
        result = replay.result or {}
        pir = result.get("pir_state", "")
        self.pir_lamps.set_state(pir)
        self.pir_state.configure(text=state_text(pir), fg=STATE_TEXT.get(pir, THEME["muted"]))
        voltage = result.get("pir_relative_voltage")
        self.pir_detail.configure(text="PIR signal  --" if voltage is None
                                  else f"PIR signal  {voltage:+.3f} V from baseline")
        fused = result.get("fusion_state", "")
        self.fused_lamps.set_state(fused)
        self.fused_state.configure(text=state_text(fused), fg=STATE_TEXT.get(fused, THEME["muted"]))
        score = result.get("fusion_confidence")
        self.probability_text.configure(text="--" if score is None else f"{score:.1%}")
        width = self.probability_bar.winfo_width()
        self.probability_bar.coords(self.probability_fill, 0, 0, 0 if score is None else width * score, 8)
        self.probability_bar.itemconfigure(self.probability_fill, fill=STATE_COLOURS.get(fused, THEME["accent"]))
        draw_confidence(self.chart, self.fonts, replay.confidence_history, now, replay.model.fusion.thresholds)


class SensorView(tk.Frame):
    """Camera top left, PIR bottom left, thermopile stages on the right (largest panel)."""

    def __init__(self, parent, fonts, on_stage):
        super().__init__(parent, bg=THEME["bg"])
        self.fonts = fonts
        self.stage = len(THERMAL_STAGES) - 1
        self.thermal_photo = None
        self.legend_photo = None
        self.thermal_drawn = None

        # Decision strip (as in main.py)
        strip = tk.Frame(self, bg=THEME["card"], highlightthickness=1, highlightbackground=THEME["border"],
                         padx=14, pady=8)
        strip.pack(fill="x")
        tk.Label(strip, text="Fused System", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(side="left")
        self.fused_pill = tk.Label(strip, text="--", font=fonts["pill_large"], padx=14, pady=3, bg=THEME["off"],
                                   fg=THEME["text"])
        self.fused_pill.pack(side="left", padx=(10, 28))
        tk.Label(strip, text="Human probability", bg=THEME["card"], fg=THEME["muted"],
                 font=fonts["body"]).pack(side="left")
        self.probability = tk.Label(strip, text="--", bg=THEME["card"], fg=THEME["text"], font=fonts["body_bold"])
        self.probability.pack(side="left", padx=(8, 28))
        tk.Label(strip, text="Thresholded PIR", bg=THEME["card"], fg=THEME["muted"],
                 font=fonts["body"]).pack(side="left")
        self.pir_pill = tk.Label(strip, text="--", font=fonts["pill"], padx=12, pady=3, bg=THEME["off"],
                                 fg=THEME["text"])
        self.pir_pill.pack(side="left", padx=(10, 0))

        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True, pady=(12, 0))
        body.columnconfigure(0, weight=2, uniform="sensor")
        body.columnconfigure(1, weight=3, uniform="sensor")
        body.rowconfigure(0, weight=1, uniform="rows")
        body.rowconfigure(1, weight=1, uniform="rows")

        camera = Card(body, fonts, "Camera")
        camera.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 6))
        self.video = VideoPanel(camera, fonts)
        self.video.pack(fill="both", expand=True, pady=(8, 0))

        pir_card = Card(body, fonts, "PIR Signal",
                        f"Raw PIR voltage over the last {PIR_DISPLAY_S:g} s, baseline ± threshold shaded")
        pir_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(6, 0))
        self.pir_value = tk.Label(pir_card.header, text="--", bg=THEME["card"], fg=THEME["accent"],
                                  font=fonts["metric"])
        self.pir_value.pack(side="right")
        self.pir_canvas = tk.Canvas(pir_card, bg="white", highlightthickness=1, highlightbackground=THEME["border"],
                                    width=300, height=150)
        self.pir_canvas.pack(fill="both", expand=True)

        thermal_card = Card(body, fonts, "Thermopile")
        thermal_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        self.thermal_time = tk.Label(thermal_card.header, text="", bg=THEME["card"], fg=THEME["muted"],
                                     font=fonts["body"])
        self.thermal_time.pack(side="right")
        self.stage_title = tk.Label(thermal_card.header, text="", bg=THEME["card"], fg=THEME["muted"],
                                    font=fonts["body_bold"])
        self.stage_title.pack(side="left", padx=(10, 0), pady=(4, 0))
        tabs = tk.Frame(thermal_card, bg=THEME["card"])
        tabs.pack(fill="x", pady=(8, 0))
        self.tabs = []
        for index, short in enumerate(STAGE_TABS[:len(THERMAL_STAGES)]):
            tab = tk.Label(tabs, text=f"{index + 1} {short}", font=fonts["small_bold"], padx=6, pady=3,
                           cursor="hand2")
            tab.pack(side="left", padx=(0, 4))
            tab.bind("<Button-1>", lambda _e, i=index: on_stage(i))
            self.tabs.append(tab)
        bottom = tk.Frame(thermal_card, bg=THEME["card"])
        bottom.pack(side="bottom", fill="x", pady=(6, 0))
        self.legend = tk.Canvas(bottom, width=330, height=28, bg=THEME["card"], highlightthickness=0)
        self.legend.pack(side="right")
        self.thermal_text = tk.Label(bottom, text="", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"],
                                     anchor="w")
        self.thermal_text.pack(side="left", fill="x", expand=True)
        self.thermal_canvas = tk.Canvas(thermal_card, bg=THEME["card"], highlightthickness=0, width=400, height=300)
        self.thermal_canvas.pack(fill="both", expand=True, pady=(8, 0))
        self.set_stage(self.stage)

    def set_stage(self, index):
        self.stage = index
        for i, tab in enumerate(self.tabs):
            active = i == index
            tab.configure(bg=THEME["accent"] if active else THEME["raised"], fg="white" if active else THEME["muted"])
        self.stage_title.configure(text=f"·  {THERMAL_STAGES[index][0]}")
        self.thermal_drawn = None

    def update_view(self, replay, now, frame, frame_id):
        result = replay.result or {}
        fused = result.get("fusion_state", "")
        self.fused_pill.configure(text=state_text(fused), bg=STATE_COLOURS.get(fused, THEME["off"]),
                                  fg=THEME["text"] if fused in ("MAYBE INTRUDER", "") else "white")
        score = result.get("fusion_confidence")
        self.probability.configure(text="--" if score is None else f"{score:.1%}")
        pir = result.get("pir_state", "")
        self.pir_pill.configure(text=state_text(pir), bg=STATE_COLOURS.get(pir, THEME["off"]),
                                fg="white" if pir else THEME["text"])
        self.video.show(frame, frame_id, replay.human_at(now))
        self.draw_pir(replay, now)
        self.draw_thermal(replay)

    def draw_pir(self, replay, now):
        canvas = self.pir_canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 80)
        left, right, top, bottom = 40, width - 12, 10, height - 22
        low, high = 0.0, 3.3

        def y(v):
            return bottom - (min(max(float(v), low), high) - low) / (high - low) * (bottom - top)

        for level in (0.0, 1.0, 2.0, 3.0):
            canvas.create_line(left, y(level), right, y(level), fill=THEME["border"])
            canvas.create_text(left - 6, y(level), text=f"{level:.1f}", anchor="e", fill=THEME["muted"],
                               font=self.fonts["small"])
        canvas.create_text(left, bottom + 12, text=f"-{PIR_DISPLAY_S:g} s", anchor="w", fill=THEME["muted"],
                           font=self.fonts["small"])
        canvas.create_text(right, bottom + 12, text="now", anchor="e", fill=THEME["muted"], font=self.fonts["small"])

        seen = replay.next_pir
        first = int(np.searchsorted(replay.pir_t[:seen], now - PIR_DISPLAY_S))
        times, volts = replay.pir_t[first:seen], replay.pir_v[first:seen]
        if len(times) == 0:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No PIR samples yet",
                               fill=THEME["text"], font=self.fonts["body_bold"])
            self.pir_value.configure(text="--")
            return
        if replay.pir.initialised:
            baseline, threshold = replay.pir.getBaseline(), replay.model.pir_threshold
            canvas.create_rectangle(left, y(baseline + threshold), right, y(baseline - threshold), fill="#dbeafe",
                                    outline="")
            canvas.create_line(left, y(baseline), right, y(baseline), fill=THEME["muted"], dash=(4, 4))
            canvas.create_text(left + 6, y(baseline + threshold) - 3, anchor="sw",
                               text=f"baseline ± {threshold:.2f} V", fill=THEME["muted"], font=self.fonts["small"])
        xs = right - (now - times) / PIR_DISPLAY_S * (right - left)
        points = np.column_stack([xs, [y(v) for v in volts]]).ravel().tolist()
        if len(points) >= 4:
            canvas.create_line(*points, fill=THEME["accent"], width=2)
        canvas.create_oval(xs[-1] - 4, y(volts[-1]) - 4, xs[-1] + 4, y(volts[-1]) + 4, fill=THEME["accent"],
                           outline="")
        self.pir_value.configure(text=f"{volts[-1]:.3f} V")

    def draw_thermal(self, replay):
        canvas = self.thermal_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        key = (replay.next_thermal, self.stage, width, height)
        if key == self.thermal_drawn or width < 20 or height < 20:
            return
        self.thermal_drawn = key
        self.thermal_time.configure(text=f"{replay.process_ms:.1f} ms/frame" if replay.features is not None else "")
        canvas.delete("all")
        if replay.features is None:
            canvas.create_text(width / 2, height / 2, text="No thermopile frames yet", fill=THEME["text"],
                               font=self.fonts["body_bold"])
            self.thermal_text.configure(text="")
            self.legend.delete("all")
            return
        w, h = fit(width, height, 32 / 24)
        image, description, bar = _thermal_stage_view(
            replay.frame_stages(), not replay.thermal.last_frame_valid, replay.blobs(), replay.features,
            self.stage, replay.raw_range, w, h)
        self.thermal_photo = to_photo(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        canvas.create_image(width // 2, height // 2, image=self.thermal_photo)
        self.thermal_text.configure(text=description)
        self.draw_legend(bar)

    def draw_legend(self, bar):
        legend = self.legend
        legend.delete("all")
        width = int(legend["width"])
        if bar is None:
            for x, kind, text in ((4, "fill", "foreground pixel"), (122, "primary", "primary track"),
                                  (236, "other", "other track")):
                if kind == "fill":
                    legend.create_rectangle(x, 7, x + 14, 21, fill="#ffdc00", outline="")
                else:
                    legend.create_rectangle(x, 7, x + 14, 21, outline="#a5dc6e" if kind == "primary" else "#00c8c8",
                                            width=3 if kind == "primary" else 1)
                legend.create_text(x + 20, 14, text=text, anchor="w", fill=THEME["muted"], font=self.fonts["small"])
            return
        colormap, high_label, low_label = bar
        bar_left, bar_right = 50, width - 50
        n = max(bar_right - bar_left, 10)
        gradient = np.linspace(0, 255, n, dtype=np.float32).reshape(1, -1)
        if colormap == "diverging":
            strip = _diverging(gradient / 127.5 - 1.0, 1.0)
        else:
            strip = cv2.applyColorMap(gradient.astype(np.uint8), colormap)
        strip = cv2.cvtColor(np.repeat(strip, 14, axis=0), cv2.COLOR_BGR2RGB)
        self.legend_photo = to_photo(strip)
        legend.create_image(bar_left, 7, image=self.legend_photo, anchor="nw")
        legend.create_text(bar_left - 6, 14, text=low_label, anchor="e", fill=THEME["muted"], font=self.fonts["small"])
        legend.create_text(bar_right + 6, 14, text=high_label, anchor="w", fill=THEME["muted"],
                           font=self.fonts["small"])


# ---------------------------------------------------------------- Application

class ReplayApp:
    def __init__(self, root, args):
        self.root = root
        self.replay = Replay(args.recording, args.model)
        frames = pd.read_csv(args.recording / "video_frames.csv")
        self.video_t = frames["time_s"].to_numpy(dtype=float)
        self.capture = cv2.VideoCapture(str(_find_video(args.recording)))
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the recording video")
        self.usable = min(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or len(self.video_t), len(self.video_t))
        self.duration = float(self.video_t[self.usable - 1])
        self.origin = float(min(self.video_t[0], self.replay.thermal_t[0]))
        self.start_time = float(min(max(args.start, self.origin), self.duration))
        self.video_index = -1
        self.video_frame = np.zeros((360, 640, 3), np.uint8)
        self.play_time = self.start_time
        self.processed_to = -np.inf
        self.paused = False
        self.showing_sensors = False
        self.fullscreen = False

        root.title(WINDOW_TITLE)
        root.geometry(WINDOW_SIZE)
        root.minsize(1100, 720)
        root.configure(bg=THEME["bg"])
        root.protocol("WM_DELETE_WINDOW", self.close)
        family = pick_font(root)
        fonts = {
            "title": (family, 22, "bold"), "card": (family, 15, "bold"), "state": (family, 20, "bold"),
            "metric": (family, 18, "bold"), "body": (family, 11), "body_bold": (family, 11, "bold"),
            "small": (family, 9), "small_bold": (family, 9, "bold"), "lamp": (family, 9, "bold"),
            "pill": (family, 10, "bold"), "pill_large": (family, 12, "bold"), "button": (family, 11, "bold"),
            "mono": ("Consolas", 11),
        }

        page = tk.Frame(root, bg=THEME["bg"], padx=22, pady=16)
        page.pack(fill="both", expand=True)

        # Header
        header = tk.Frame(page, bg=THEME["bg"])
        header.pack(fill="x")
        titles = tk.Frame(header, bg=THEME["bg"])
        titles.pack(side="left")
        self.title = tk.Label(titles, text="Outdoor Intrusion Detection", bg=THEME["bg"], fg=THEME["text"],
                              font=fonts["title"])
        self.title.pack(anchor="w")
        tk.Label(titles, text=f"Recorded demo: {args.recording.name}  |  replayed through the real-time system  |  "
                              f"{self.replay.window:g} s fusion window",
                 bg=THEME["bg"], fg=THEME["muted"], font=fonts["body"]).pack(anchor="w")
        self.view_button = make_button(header, fonts, "Sensor View", self.toggle_view)
        self.view_button.pack(side="right", anchor="n", pady=4)
        self.status_pill = tk.Label(header, text="", font=fonts["pill"], padx=12, pady=4)
        self.status_pill.pack(side="right", anchor="n", pady=10, padx=12)

        # Transport bar (shared by both views)
        transport = tk.Frame(page, bg=THEME["card"], highlightthickness=1, highlightbackground=THEME["border"],
                             padx=12, pady=8)
        transport.pack(side="bottom", fill="x", pady=(12, 0))
        make_button(transport, fonts, "⟲", self.restart, padx=10).pack(side="left")
        make_button(transport, fonts, f"−{SKIP_S:g} s", lambda: self.skip(-SKIP_S), padx=10).pack(side="left", padx=(6, 0))
        self.play_button = make_button(transport, fonts, "❚❚  Pause", self.toggle_pause, padx=12)
        self.play_button.configure(width=9)
        self.play_button.pack(side="left", padx=6)
        make_button(transport, fonts, f"+{SKIP_S:g} s", lambda: self.skip(SKIP_S), padx=10).pack(side="left")
        self.time_text = tk.Label(transport, text="", bg=THEME["card"], fg=THEME["text"], font=fonts["mono"],
                                  width=17)
        self.time_text.pack(side="left", padx=(14, 6))
        intervals = []
        labels = self.replay.labels
        if labels is not None and not labels.empty:
            human = labels[labels["human"] == 1]
            intervals = list(zip(human["start_s"].astype(float), human["end_s"].astype(float)))
        self.timeline = Timeline(transport, self.origin, self.duration, intervals, self.seek)
        self.timeline.pack(side="left", fill="x", expand=True)

        footer = tk.Frame(page, bg=THEME["bg"])
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Label(footer, text="SPACE Play/Pause    ← / → ∓5 s    Click/drag bar to seek    R Restart    "
                              "S Sensor view    1-8 Stage    F Full screen    Q Quit",
                 bg=THEME["bg"], fg=THEME["muted"], font=fonts["small_bold"]).pack(side="right")
        self.status = tk.Label(footer, text="", bg=THEME["bg"], fg=THEME["muted"], font=fonts["small"], anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

        # Views
        content = tk.Frame(page, bg=THEME["bg"])
        content.pack(fill="both", expand=True, pady=(14, 0))
        self.main_view = MainView(content, fonts)
        self.sensor_view = SensorView(content, fonts, self.set_stage)
        self.main_view.pack(fill="both", expand=True)

        for key, action in (("<space>", self.toggle_pause), ("<Left>", lambda: self.skip(-SKIP_S)),
                            ("<Right>", lambda: self.skip(SKIP_S)), ("r", self.restart), ("s", self.toggle_view),
                            ("f", lambda: self.set_fullscreen(not self.fullscreen)), ("q", self.close)):
            root.bind_all(key, lambda _e, a=action: a())
            if len(key) == 1:
                root.bind_all(key.upper(), lambda _e, a=action: a())
        root.bind_all("<Escape>", lambda _e: self.set_fullscreen(False) if self.fullscreen else self.close())
        for index in range(len(THERMAL_STAGES)):
            root.bind_all(str(index + 1), lambda _e, i=index: self.set_stage(i))

        self.seek(self.start_time)
        self.last_wall = time.perf_counter()
        self.tick()

    # ------------------------------------------------ playback control
    def toggle_pause(self):
        if self.paused and self.play_time >= self.duration:
            self.seek(self.origin)
        self.paused = not self.paused
        self.play_button.configure(text="▶  Play" if self.paused else "❚❚  Pause")

    def skip(self, seconds):
        self.seek(self.play_time + seconds)

    def restart(self):
        self.seek(self.start_time)

    def seek(self, t):
        self.play_time = float(min(max(t, self.origin), self.duration))
        self.process_to(self.play_time)
        self.render()

    def set_stage(self, index):
        self.sensor_view.set_stage(index)

    def toggle_view(self):
        self.showing_sensors = not self.showing_sensors
        if self.showing_sensors:
            self.main_view.pack_forget()
            self.sensor_view.pack(fill="both", expand=True)
        else:
            self.sensor_view.pack_forget()
            self.main_view.pack(fill="both", expand=True)
        self.view_button.configure(text="Main View" if self.showing_sensors else "Sensor View")
        self.render()

    def set_fullscreen(self, enabled):
        self.fullscreen = enabled
        self.root.attributes("-fullscreen", enabled)

    # ------------------------------------------------ processing
    def process_to(self, t):
        # Going backwards: restart the real-time system and feed it again up to t (keeps decisions causal)
        if t < self.processed_to:
            self.replay.reset()
        self.replay.advance(t)
        self.processed_to = t
        self.sync_video(t)

    def sync_video(self, t):
        target = _nearest_index(self.video_t[:self.usable], t)
        if target == self.video_index:
            return
        if target < self.video_index or target - self.video_index > VIDEO_SEEK_FRAMES:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, image = self.capture.read()
            if ok:
                self.video_frame = image
            self.video_index = target
            return
        while self.video_index < target:
            if self.video_index < target - 1:
                self.capture.grab()            # skip frames without decoding them
            else:
                ok, image = self.capture.read()
                if ok:
                    self.video_frame = image
            self.video_index += 1

    def tick(self):
        now = time.perf_counter()
        if not self.paused and not self.timeline.dragging:
            self.play_time = min(self.duration, self.play_time + (now - self.last_wall))
            self.process_to(self.play_time)
            if self.play_time >= self.duration:
                self.paused = True
                self.play_button.configure(text="▶  Play")
            self.render()
        self.last_wall = now
        self.root.after(TICK_MS, self.tick)

    def render(self):
        replay = self.replay
        if self.paused:
            self.status_pill.configure(text="❚❚  PAUSED", bg=THEME["raised"], fg=THEME["muted"])
        elif replay.result is None:
            self.status_pill.configure(text="FILLING WINDOW", bg="#fef9c3", fg=STATE_TEXT["MAYBE INTRUDER"])
        else:
            self.status_pill.configure(text="●  REPLAYING", bg="#dcfce7", fg=STATE_TEXT["NO INTRUDER"])
        self.time_text.configure(text=f"{format_time(self.play_time - self.origin)} / "
                                      f"{format_time(self.duration - self.origin)}")
        self.timeline.set_time(self.play_time)
        self.status.configure(text=f"{replay.status}  |  last thermal frame processed in {replay.process_ms:.1f} ms")
        view = self.sensor_view if self.showing_sensors else self.main_view
        view.update_view(replay, self.play_time, self.video_frame, self.video_index)

    def close(self):
        self.capture.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Replay a recording through the real-time system")
    parser.add_argument("--recording", required=True, type=Path, help="Path to one recording folder")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Exported first-principles model")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start the replay this many seconds in (earlier data is still processed)")
    args = parser.parse_args()
    root = tk.Tk()
    ReplayApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

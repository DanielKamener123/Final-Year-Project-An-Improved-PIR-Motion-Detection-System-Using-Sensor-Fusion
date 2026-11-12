"""Start the complete live sensor-fusion system on the ROCK 4C+."""
import argparse
import csv
import json
import os
import queue
import threading
import time
from collections import deque
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import PIRProcessor, warmupTime, threshold as PIR_THRESHOLD
from Thermopile.interface import getFrame, REFRESH_RATE_HZ
from Thermopile.processing import ThermopilePipeline, FRAME_SHAPE
from Fusion.realtimeModel import RealtimeModel, MODE_FUSED, MODE_PIR_BACKUP, NO_INTRUDER

# Live calibration matches the recordings (Collection/rock_agent.py): the first
# N_WARMUP_FRAMES frames after start-up are thrown away, then the background and
# sigma come from CALIBRATION_SECONDS * THERMAL_RATE_HZ = 10 * 4 = 40 frames.
# Same number of frames -> same sigma estimate -> same z values as the training data.
CALIBRATION_DISCARD_FRAMES = 20
CALIBRATION_FRAMES = 40

# Live log: one row per fused decision (both systems from the same acquisition,
# plus the processing time used for the response-time test)
LOG_FOLDER = Path("Processed/live_logs")
LOG_COLUMNS = ["timestamp", "pir_state", "fusion_state", "fusion_confidence",
               "pir_relative_voltage", "processing_ms", "pir_threshold_v"]

# Sensor Failure Handling
# No thermal frame for this long -> PIR-only backup classifier (same 1 s gap at
# which Fusion/dataset.py stops fusing). Back to the fused system once valid
# thermal frames have arrived again for one fusion window.
THERMAL_TIMEOUT_S = 1.0
# No thermal frame at all this long after start-up -> no working thermopile
STARTUP_THERMAL_TIMEOUT_S = 3.0
# No PIR sample for this long -> PIR reported missing (decisions pause, thermal keeps running)
PIR_TIMEOUT_S = 1.0
# Decision rate while no thermal frames arrive (the thermal frame rate)
DECISION_PERIOD_S = 1.0 / REFRESH_RATE_HZ
# Most recent start-up background, reused when the thermopile is not working at start-up
BACKGROUND_FILE = "thermal_background.json"

# Thresholded PIR Comparison Level (Up / Down arrow keys or the - / + buttons change it live).
# Only the thresholded-PIR comparison uses it; the fused system's PIR features keep PIR/processing.py's threshold.
PIR_THRESHOLD_STEP_V = 0.05
PIR_THRESHOLD_MIN_V = 0.05
PIR_THRESHOLD_MAX_V = 1.60

# Sensor View Settings
PIR_DISPLAY_S = 15.0
THERMAL_SCALE = 12
# Thermopile processing stages shown in the sensor view (keys 1-5):
# (tab label, title, plain explanation for the examiners, stage key)
THERMAL_STAGES = [
    ("Raw", "Raw temperature",
     "What the 32 x 24 thermopile measures, in degrees Celsius. A person shows up as a warmer patch.", "raw"),
    ("Background", "Background model",
     "The learned empty scene. It slowly follows the weather but is never updated where a person is detected.",
     "background"),
    ("Z score", "Z score",
     "How unusual each pixel is: (temperature - background) / that pixel's normal noise. Bright = unusual.", "z"),
    ("Filtered Z", "Filtered Z score",
     "The Z score after 3 x 3 Gaussian smoothing, which removes single-pixel noise.", "smoothed_z"),
    ("Detections", "Foreground mask, blobs and tracks",
     "Pixels above their threshold are grouped into blobs and followed from frame to frame. "
     "The tracks feed the thermal features of the fused system.", "mask"),
]
Z_DISPLAY_MAX = 10.0

# ---------------------------------------------------------------- GUI (display only)
# Nothing below does any processing: the GUI only reads the latest results, so
# switching views never pauses or recalibrates the system.

# Light, high-contrast theme: the demo is outdoors in daylight
THEME = {
    "bg": "#ffffff", "card": "#f6f8fb", "raised": "#e9eef4", "border": "#cfd7e2",
    "text": "#0f172a", "muted": "#475569", "accent": "#1d4ed8", "accent_hover": "#2563eb",
    "green": "#16a34a", "yellow": "#facc15", "red": "#dc2626", "off": "#dde3ea",
}
# Lamp / fill colours, and darker versions of the same colours for text on white
STATE_COLOURS = {"NO INTRUDER": THEME["green"], "MAYBE INTRUDER": THEME["yellow"], "INTRUDER": THEME["red"]}
STATE_TEXT = {"NO INTRUDER": "#15803d", "MAYBE INTRUDER": "#a16207", "INTRUDER": "#b91c1c"}
# Requirement 4: alarm within 10 s of the person entering
RESPONSE_LIMIT_S = 10.0
WINDOW_SIZE = "1120x740"


# First Installed Font From A Preference List (ROCK: DejaVu Sans)
def pickFont(root):
    from tkinter import font as tkfont
    families = set(tkfont.families(root))
    for name in ("Inter", "Segoe UI", "DejaVu Sans", "Helvetica"):
        if name in families:
            return name
    return "TkDefaultFont"


# Colour Maps For The Thermal Stage Images (numpy look-up tables, display only)
def colourTable(anchors):
    positions = np.array([a[0] for a in anchors])
    colours = np.array([a[1] for a in anchors], dtype=np.float64)
    x = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(x, positions, colours[:, c]) for c in range(3)], axis=1).astype(np.uint8)

INFERNO = colourTable([(0.0, (0, 0, 4)), (0.13, (31, 12, 72)), (0.25, (85, 15, 109)), (0.38, (136, 34, 106)),
                       (0.5, (186, 54, 85)), (0.63, (227, 89, 51)), (0.75, (249, 140, 10)),
                       (0.88, (249, 201, 50)), (1.0, (252, 255, 164))])
VIRIDIS = colourTable([(0.0, (68, 1, 84)), (0.13, (71, 44, 122)), (0.25, (59, 81, 139)), (0.38, (44, 113, 142)),
                       (0.5, (33, 144, 141)), (0.63, (39, 173, 129)), (0.75, (92, 200, 99)),
                       (0.88, (170, 220, 50)), (1.0, (253, 231, 37))])

def applyColourMap(values, low, high, table):
    scaled = np.clip((np.asarray(values, dtype=np.float64) - low) / max(high - low, 1e-9), 0.0, 1.0)
    return table[(scaled * 255).astype(np.uint8)]

# 24x32x3 uint8 image -> Tk PPM data
def toPpm(rgb):
    height, width = rgb.shape[:2]
    return f"P6 {width} {height} 255 ".encode("ascii") + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()

def hexColour(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


class Card(tk.Frame):
    """Dark rounded-look panel with a title and a one-line explanation."""

    def __init__(self, parent, fonts, title, subtitle=None):
        super().__init__(parent, bg=THEME["card"], highlightthickness=1,
                         highlightbackground=THEME["border"], padx=16, pady=12)
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
    """A row of indicator lights; only the current state is lit."""

    def __init__(self, parent, fonts, states, width=420):
        super().__init__(parent, width=width, height=96, bg=THEME["card"], highlightthickness=0)
        self.states = states
        self.items = []
        step = width / len(states)
        for i, (label, colour) in enumerate(states):
            x = step * (i + 0.5)
            glow = self.create_oval(x - 30, 6, x + 30, 66, outline="", fill="")
            lamp = self.create_oval(x - 22, 14, x + 22, 58, fill=THEME["off"], outline=THEME["border"], width=2)
            text = self.create_text(x, 82, text=label, fill=THEME["muted"], font=fonts["lamp"])
            self.items.append((glow, lamp, text, colour))

    def set_state(self, state):
        for (label, _), (glow, lamp, text, colour) in zip(self.states, self.items):
            lit = label == state
            self.itemconfigure(lamp, fill=colour if lit else THEME["off"],
                               outline=colour if lit else THEME["border"])
            self.itemconfigure(glow, outline=colour if lit else "", width=3)
            self.itemconfigure(text, fill=THEME["text"] if lit else THEME["muted"])


def makeButton(parent, fonts, text, command):
    button = tk.Button(parent, text=text, command=command, takefocus=0, font=fonts["button"],
                       bg=THEME["accent"], fg="white", activebackground=THEME["accent_hover"],
                       activeforeground="white", relief="flat", bd=0, padx=18, pady=8, cursor="hand2")
    button.bind("<Enter>", lambda _e: button.configure(bg=THEME["accent_hover"]))
    button.bind("<Leave>", lambda _e: button.configure(bg=THEME["accent"]))
    return button


def makeSmallButton(parent, fonts, text, command):
    button = tk.Button(parent, text=text, command=command, takefocus=0, font=fonts["button"], width=2,
                       bg=THEME["accent"], fg="white", activebackground=THEME["accent_hover"],
                       activeforeground="white", relief="flat", bd=0, padx=4, pady=0, cursor="hand2")
    button.bind("<Enter>", lambda _e: button.configure(bg=THEME["accent_hover"]))
    button.bind("<Leave>", lambda _e: button.configure(bg=THEME["accent"]))
    return button


def stateText(state):
    return state if state in STATE_COLOURS else "UNAVAILABLE"


class MainView(tk.Frame):
    """Fused system vs thresholded PIR, processing time and the fused detection log."""

    def __init__(self, parent, fonts, on_toggle, on_threshold):
        super().__init__(parent, bg=THEME["bg"], padx=22, pady=18)
        self.fonts = fonts

        # Header
        header = tk.Frame(self, bg=THEME["bg"])
        header.pack(fill="x")
        titles = tk.Frame(header, bg=THEME["bg"])
        titles.pack(side="left")
        tk.Label(titles, text="Outdoor Intrusion Detection", bg=THEME["bg"], fg=THEME["text"],
                 font=fonts["title"]).pack(anchor="w")
        tk.Label(titles, text="Thresholded PIR vs Fused system", bg=THEME["bg"], fg=THEME["muted"],
                 font=fonts["body"]).pack(anchor="w")
        self.status_pill = tk.Label(header, text="STARTING", font=fonts["pill"], padx=12, pady=4,
                                    bg=THEME["raised"], fg=THEME["muted"])
        self.status_pill.pack(side="right", anchor="n", pady=6)

        # Decision cards
        cards = tk.Frame(self, bg=THEME["bg"])
        cards.pack(fill="x", pady=(16, 0))
        cards.columnconfigure(0, weight=1, uniform="cards")
        cards.columnconfigure(1, weight=1, uniform="cards")

        left = Card(cards, fonts, "Thresholded PIR",
                    f"Intruder when PIR is more than ±{PIR_THRESHOLD:g} V from its baseline")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.pir_card = left
        # Live threshold control (also Up / Down arrow keys)
        control = tk.Frame(left.header, bg=THEME["card"])
        control.pack(side="right")
        tk.Label(control, text="Threshold", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(side="left")
        makeSmallButton(control, fonts, "−", lambda: on_threshold(-1)).pack(side="left", padx=(8, 4))
        self.pir_threshold_text = tk.Label(control, text=f"{PIR_THRESHOLD:.2f} V", width=7, bg=THEME["raised"],
                                           fg=THEME["text"], font=fonts["body_bold"], pady=2)
        self.pir_threshold_text.pack(side="left")
        makeSmallButton(control, fonts, "+", lambda: on_threshold(+1)).pack(side="left", padx=(4, 0))
        self.pir_lamps = Lamps(left, fonts, [("NO INTRUDER", THEME["green"]), ("INTRUDER", THEME["red"])])
        self.pir_lamps.pack(pady=(4, 0))
        self.pir_state = tk.Label(left, text="--", bg=THEME["card"], fg=THEME["muted"], font=fonts["state"])
        self.pir_state.pack(pady=(4, 2))
        self.pir_detail = tk.Label(left, text="", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"])
        self.pir_detail.pack()

        right = Card(cards, fonts, "Fused System",
                     "PIR and thermopile features combined by the Gradient Boosting classifier")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.fused_lamps = Lamps(right, fonts, [("NO INTRUDER", THEME["green"]), ("MAYBE INTRUDER", THEME["yellow"]),
                                                ("INTRUDER", THEME["red"])])
        self.fused_lamps.pack(pady=(4, 0))
        self.fused_state = tk.Label(right, text="--", bg=THEME["card"], fg=THEME["muted"], font=fonts["state"])
        self.fused_state.pack(pady=(4, 2))
        probability = tk.Frame(right, bg=THEME["card"])
        probability.pack(fill="x")
        tk.Label(probability, text="Human probability", bg=THEME["card"], fg=THEME["muted"],
                 font=fonts["body"]).pack(side="left")
        self.probability_text = tk.Label(probability, text="--", bg=THEME["card"], fg=THEME["text"],
                                         font=fonts["body_bold"])
        self.probability_text.pack(side="right")
        self.probability_bar = tk.Canvas(right, height=8, bg=THEME["raised"], highlightthickness=0)
        self.probability_bar.pack(fill="x", pady=(4, 0))
        self.probability_fill = self.probability_bar.create_rectangle(0, 0, 0, 8, fill=THEME["accent"], width=0)

        # Processing time and detection log
        lower = tk.Frame(self, bg=THEME["bg"])
        lower.pack(fill="both", expand=True, pady=(16, 0))
        lower.columnconfigure(0, weight=2, uniform="lower")
        lower.columnconfigure(1, weight=3, uniform="lower")
        lower.columnconfigure(2, weight=5, uniform="lower")
        lower.rowconfigure(0, weight=1)

        timing = Card(lower, fonts, "Processing time", "Time to make each decision")
        timing.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.processing = tk.Label(timing, text="--", bg=THEME["card"], fg=THEME["text"], font=fonts["metric"])
        self.processing.pack(anchor="w", pady=(6, 0))
        self.processing_note = tk.Label(timing, text="", bg=THEME["card"], fg=THEME["muted"], font=fonts["small"],
                                        justify="left", anchor="w")
        self.processing_note.pack(anchor="w", fill="x")

        # Requirement 4: press E when the person enters; each timer stops at that system's first INTRUDER
        response = Card(lower, fonts, "Response time", "Press E when the person enters the zone")
        response.grid(row=0, column=1, sticky="nsew", padx=8)
        self.response_rows = {}
        for system in ("Fused System", "Thresholded PIR"):
            row = tk.Frame(response, bg=THEME["card"])
            row.pack(fill="x", pady=(6, 0))
            tk.Label(row, text=system, bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(anchor="w")
            value = tk.Label(row, text="--", bg=THEME["card"], fg=THEME["text"], font=fonts["metric"], anchor="w")
            value.pack(anchor="w")
            self.response_rows[system] = value
        tk.Label(response, text=f"Requirement: alarm within {RESPONSE_LIMIT_S:g} s", bg=THEME["card"],
                 fg=THEME["muted"], font=fonts["small"]).pack(anchor="w", pady=(6, 0))

        log_card = Card(lower, fonts, "Fused system log",
                        "Every MAYBE INTRUDER / INTRUDER raised, and every timed entry (newest first)")
        log_card.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.log_count = tk.Label(log_card.header, text="0", bg=THEME["raised"], fg=THEME["text"],
                                  font=fonts["pill"], padx=10, pady=2)
        self.log_count.pack(side="right")
        self.log_list = tk.Listbox(log_card, bg="white", fg=THEME["text"], font=fonts["mono"],
                                   relief="flat", highlightthickness=1, highlightbackground=THEME["border"],
                                   activestyle="none", takefocus=0,
                                   selectbackground="white", selectforeground=THEME["text"], height=6)
        self.log_list.pack(fill="both", expand=True)
        self.detections = 0

        # Footer
        footer = tk.Frame(self, bg=THEME["bg"])
        footer.pack(fill="x", pady=(14, 0))
        makeButton(footer, fonts, "Sensor View", on_toggle).pack(side="right")
        tk.Label(footer, text="E = Entry    R = Reset    C = Recalibrate    F = Full Screen",
                 bg=THEME["bg"],
                 fg=THEME["muted"], font=fonts["small_bold"]).pack(side="right", padx=16)
        self.status = tk.Label(footer, text="", bg=THEME["bg"], fg=THEME["muted"], font=fonts["small"], anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

    def add_event(self, event):
        clock = time.strftime("%H:%M:%S", time.localtime(event["time"]))
        if event["kind"] == "detection":
            self.detections += 1
            self.log_count.configure(text=str(self.detections))
            line = (f"{clock}  {event['state']:<15} {event['confidence']:6.1%}   "
                    f"PIR: {event['pir_state']}")
            colour = STATE_TEXT.get(event["state"], THEME["text"])
        elif event["kind"] == "entry":
            line = f"{clock}  ENTRY marked (E) - timing both systems"
            colour = THEME["accent"]
        elif event["kind"] == "threshold":
            line = f"{clock}  Thresholded PIR threshold set to ±{event['volts']:.2f} V"
            colour = THEME["muted"]
        else:
            seconds = event["seconds"]
            verdict = "within" if seconds <= RESPONSE_LIMIT_S else "OVER"
            line = f"{clock}  {event['system']} INTRUDER after {seconds:.2f} s ({verdict} {RESPONSE_LIMIT_S:g} s)"
            colour = STATE_TEXT["NO INTRUDER"] if seconds <= RESPONSE_LIMIT_S else STATE_TEXT["INTRUDER"]
        self.log_list.insert(0, line)
        self.log_list.itemconfigure(0, fg=colour)
        if self.log_list.size() > 200:
            self.log_list.delete(200, "end")

    # Thresholded PIR Level On The Card (button value and the explanation line)
    def show_threshold(self, volts):
        self.pir_threshold_text.configure(text=f"{volts:.2f} V")
        self.pir_card.subtitle.configure(
            text=f"Alarm when the PIR voltage moves more than ±{volts:.2f} V from its baseline")

    # Clears The Log, Count And Timers For The Next Demo Trial (processing is not touched)
    def reset(self):
        self.detections = 0
        self.log_count.configure(text="0")
        self.log_list.delete(0, "end")
        for value in self.response_rows.values():
            value.configure(text="--", fg=THEME["text"])

    # Running Or Stopped Response Timers
    def update_response(self, entry):
        if entry is None:
            return
        now = time.time()
        for system, key in (("Fused System", "fused_s"), ("Thresholded PIR", "pir_s")):
            seconds = entry[key]
            value = self.response_rows[system]
            if seconds is None:
                running = now - entry["time"]
                value.configure(text=f"{running:4.1f} s ...",
                                fg=THEME["muted"] if running <= RESPONSE_LIMIT_S else STATE_TEXT["INTRUDER"])
            else:
                value.configure(text=f"{seconds:.2f} s", fg=STATE_TEXT["NO INTRUDER"] if seconds <= RESPONSE_LIMIT_S
                                else STATE_TEXT["INTRUDER"])

    def update_view(self, item):
        warmup = item.get("warmup", "")
        fused = item.get("fusion_state", "")
        if warmup == "READY" and fused in STATE_COLOURS:
            self.status_pill.configure(text="●  LIVE", bg="#dcfce7", fg=STATE_TEXT["NO INTRUDER"])
        elif "WARM" in warmup or "calibrat" in warmup.lower() or warmup.startswith("Preparing"):
            self.status_pill.configure(text="WARMING UP", bg="#fef9c3", fg=STATE_TEXT["MAYBE INTRUDER"])
        else:
            self.status_pill.configure(text="SENSOR ISSUE", bg="#fee2e2", fg=STATE_TEXT["INTRUDER"])

        pir = item.get("pir_state", "")
        self.pir_lamps.set_state(pir)
        self.pir_state.configure(text=stateText(pir), fg=STATE_TEXT.get(pir, THEME["muted"]))
        threshold = item.get("pir_threshold")
        if threshold is not None:
            self.show_threshold(threshold)
        voltage = item.get("pir_relative_voltage")
        self.pir_detail.configure(text="PIR signal  --" if voltage is None
                                  else f"PIR signal  {voltage:+.3f} V from baseline")

        self.fused_lamps.set_state(fused)
        self.fused_state.configure(text=stateText(fused), fg=STATE_TEXT.get(fused, THEME["muted"]))
        score = item.get("fusion_confidence")
        self.probability_text.configure(text="--" if score is None else f"{score:.1%}")
        width = self.probability_bar.winfo_width()
        self.probability_bar.coords(self.probability_fill, 0, 0, 0 if score is None else width * score, 8)
        self.probability_bar.itemconfigure(self.probability_fill, fill=STATE_COLOURS.get(fused, THEME["accent"]))

        processing = item.get("processing_ms")
        self.processing.configure(text="--" if processing is None else f"{processing:.1f} ms")
        worst = item.get("processing_max_ms")
        self.processing_note.configure(text="" if worst is None else f"Slowest this run: {worst:.1f} ms")
        self.status.configure(text=item.get("status", ""))


class SensorView(tk.Frame):
    """Live PIR signal and the thermopile processing stages (keys 1-5). Display only."""

    def __init__(self, parent, fonts, on_toggle):
        super().__init__(parent, bg=THEME["bg"], padx=22, pady=18)
        self.fonts = fonts
        self.stage = 0
        self.last_drawn = None
        self.photo = None

        # Header
        header = tk.Frame(self, bg=THEME["bg"])
        header.pack(fill="x")
        titles = tk.Frame(header, bg=THEME["bg"])
        titles.pack(side="left")
        tk.Label(titles, text="Sensor View", bg=THEME["bg"], fg=THEME["text"], font=fonts["title"]).pack(anchor="w")
        tk.Label(titles, text="Live Sensor Data",
                 bg=THEME["bg"], fg=THEME["muted"], font=fonts["body"]).pack(anchor="w")
        makeButton(header, fonts, "Main view", on_toggle).pack(side="right", anchor="n", pady=4)

        # Decision strip
        strip = tk.Frame(self, bg=THEME["card"], highlightthickness=1, highlightbackground=THEME["border"],
                         padx=14, pady=10)
        strip.pack(fill="x", pady=(14, 0))
        tk.Label(strip, text="Fused System", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(side="left")
        self.fused_pill = tk.Label(strip, text="--", font=fonts["pill_large"], padx=14, pady=3,
                                   bg=THEME["off"], fg=THEME["text"])
        self.fused_pill.pack(side="left", padx=(10, 28))
        tk.Label(strip, text="Human probability", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(side="left")
        self.probability = tk.Label(strip, text="--", bg=THEME["card"], fg=THEME["text"], font=fonts["body_bold"])
        self.probability.pack(side="left", padx=(8, 28))
        tk.Label(strip, text="Thresholded PIR", bg=THEME["card"], fg=THEME["muted"], font=fonts["body"]).pack(side="left")
        self.pir_pill = tk.Label(strip, text="--", font=fonts["pill"], padx=12, pady=3, bg=THEME["off"], fg=THEME["text"])
        self.pir_pill.pack(side="left", padx=(10, 0))

        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True, pady=(14, 0))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # PIR card
        pir_card = Card(body, fonts, "PIR Signal", f"Raw PIR voltage in volts over the last {PIR_DISPLAY_S:g} s (0 V to 3.3 V)")
        pir_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.pir_value = tk.Label(pir_card.header, text="--", bg=THEME["card"], fg=THEME["accent"], font=fonts["metric"])
        self.pir_value.pack(side="right")
        self.pir_canvas = tk.Canvas(pir_card, width=470, height=400, bg="white", highlightthickness=1,
                                    highlightbackground=THEME["border"])
        self.pir_canvas.pack(fill="both", expand=True)

        # Thermopile card
        thermal_card = Card(body, fonts, "Thermopile",
                            "32 × 24 thermal image. Press 1 to 5 to step through the processing")
        thermal_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tabs = tk.Frame(thermal_card, bg=THEME["card"])
        tabs.pack(fill="x")
        self.tabs = []
        for index, (short, _, _, _) in enumerate(THERMAL_STAGES):
            tab = tk.Label(tabs, text=f"{index + 1}  {short}", font=fonts["small_bold"], padx=8, pady=3,
                           cursor="hand2")
            tab.pack(side="left", padx=(0, 4))
            tab.bind("<Button-1>", lambda _e, i=index: self.set_stage(i))
            self.tabs.append(tab)
        self.stage_title = tk.Label(thermal_card, text="", bg=THEME["card"], fg=THEME["text"], font=fonts["body_bold"],
                                    anchor="w")
        self.stage_title.pack(fill="x", pady=(10, 0))
        self.stage_text = tk.Label(thermal_card, text="", bg=THEME["card"], fg=THEME["muted"], font=fonts["small"],
                                   anchor="w", justify="left", wraplength=FRAME_SHAPE[1] * THERMAL_SCALE)
        self.stage_text.pack(fill="x", pady=(2, 6))
        self.thermal_canvas = tk.Canvas(thermal_card, width=FRAME_SHAPE[1] * THERMAL_SCALE,
                                        height=FRAME_SHAPE[0] * THERMAL_SCALE, bg=THEME["raised"],
                                        highlightthickness=0)
        self.thermal_canvas.pack()
        self.legend = tk.Canvas(thermal_card, width=FRAME_SHAPE[1] * THERMAL_SCALE, height=30, bg=THEME["card"],
                                highlightthickness=0)
        self.legend.pack(pady=(6, 0))
        self.set_stage(0)

    def set_stage(self, index):
        self.stage = index
        short, title, explanation, _ = THERMAL_STAGES[index]
        for i, tab in enumerate(self.tabs):
            active = i == index
            tab.configure(bg=THEME["accent"] if active else THEME["raised"],
                          fg="white" if active else THEME["muted"])
        self.stage_title.configure(text=f"{index + 1}. {title}")
        self.stage_text.configure(text=explanation)
        self.last_drawn = None

    def draw_decision(self, item):
        fused = item.get("fusion_state", "")
        self.fused_pill.configure(text=stateText(fused), bg=STATE_COLOURS.get(fused, THEME["off"]),
                                  fg=THEME["text"] if fused == "MAYBE INTRUDER" else "white")
        score = item.get("fusion_confidence")
        self.probability.configure(text="--" if score is None else f"{score:.1%}")
        pir = item.get("pir_state", "")
        self.pir_pill.configure(text=stateText(pir), bg=STATE_COLOURS.get(pir, THEME["off"]), fg="white")

    def draw_pir(self, samples, baseline, initialised, threshold=PIR_THRESHOLD):
        canvas = self.pir_canvas
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 200)
        height = max(int(canvas.winfo_height()), 200)
        left, right, top, bottom = 46, width - 14, 14, height - 28
        low, high = 0.0, 3.3

        def y(v):
            return bottom - (min(max(float(v), low), high) - low) / (high - low) * (bottom - top)

        for level in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3):
            canvas.create_line(left, y(level), right, y(level), fill=THEME["border"])
            canvas.create_text(left - 6, y(level), text=f"{level:.1f}", anchor="e", fill=THEME["muted"],
                               font=self.fonts["small"])
        canvas.create_text(left, bottom + 14, text=f"-{PIR_DISPLAY_S:g} s", anchor="w", fill=THEME["muted"],
                           font=self.fonts["small"])
        canvas.create_text(right, bottom + 14, text="now", anchor="e", fill=THEME["muted"], font=self.fonts["small"])

        if not samples:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No PIR samples received",
                               fill=THEME["text"], font=self.fonts["body_bold"])
            self.pir_value.configure(text="--")
            return
        if initialised:
            # Baseline and the thresholded-PIR trigger levels
            canvas.create_rectangle(left, y(baseline + threshold), right, y(baseline - threshold),
                                    fill="#dbeafe", outline="")
            canvas.create_line(left, y(baseline), right, y(baseline), fill=THEME["muted"], dash=(4, 4))
            canvas.create_text(left + 6, y(baseline + threshold) - 3, anchor="sw",
                               text=f"baseline ± {threshold:.2f} V threshold", fill=THEME["muted"], font=self.fonts["small"])
        times = np.array([s[0] for s in samples])
        raw = np.array([s[1] for s in samples])
        xs = right - (times[-1] - times) / PIR_DISPLAY_S * (right - left)
        keep = xs >= left
        points = np.column_stack([xs[keep], [y(v) for v in raw[keep]]]).ravel().tolist()
        if len(points) >= 4:
            canvas.create_line(*points, fill=THEME["accent"], width=2)
        canvas.create_oval(xs[-1] - 4, y(raw[-1]) - 4, xs[-1] + 4, y(raw[-1]) + 4, fill=THEME["accent"], outline="")
        self.pir_value.configure(text=f"{raw[-1]:.3f} V")

    def draw_legend(self, low, high, table, low_text, high_text):
        legend = self.legend
        legend.delete("all")
        width = int(legend["width"])
        bar_left, bar_right = 70, width - 70
        steps = 64
        for i in range(steps):
            colour = hexColour(table[int(i / (steps - 1) * 255)])
            x0 = bar_left + (bar_right - bar_left) * i / steps
            legend.create_rectangle(x0, 6, x0 + (bar_right - bar_left) / steps + 1, 20, fill=colour, outline="")
        legend.create_text(bar_left - 6, 13, text=low_text, anchor="e", fill=THEME["muted"], font=self.fonts["small"])
        legend.create_text(bar_right + 6, 13, text=high_text, anchor="w", fill=THEME["muted"], font=self.fonts["small"])

    def draw_thermal(self, stages):
        canvas = self.thermal_canvas
        key = THERMAL_STAGES[self.stage][3]
        if stages is None:
            if self.last_drawn != "none":
                canvas.delete("all")
                canvas.create_text(FRAME_SHAPE[1] * THERMAL_SCALE / 2, FRAME_SHAPE[0] * THERMAL_SCALE / 2,
                                   text="No thermopile frames received", fill=THEME["text"],
                                   font=self.fonts["body_bold"])
                self.legend.delete("all")
                self.last_drawn = "none"
            return
        if self.last_drawn == (stages["frame_id"], self.stage):
            return
        self.last_drawn = (stages["frame_id"], self.stage)

        raw = stages["raw"]
        low = float(np.percentile(raw, 1))
        high = max(float(np.percentile(raw, 99)), low + 4.0)
        if key in ("raw", "background"):
            image = applyColourMap(stages[key], low, high, INFERNO)
            self.draw_legend(low, high, INFERNO, f"{low:.1f} °C", f"{high:.1f} °C")
        elif key in ("z", "smoothed_z"):
            image = applyColourMap(stages[key], 0.0, Z_DISPLAY_MAX, VIRIDIS)
            self.draw_legend(0, Z_DISPLAY_MAX, VIRIDIS, "Z = 0", f"Z ≥ {Z_DISPLAY_MAX:g}")
        else:
            grey = np.clip((raw - low) / (high - low) * 120 + 30, 0, 255).astype(np.uint8)
            image = np.stack([grey, grey, grey], axis=2)
            image[stages["mask"]] = (255, 200, 40)
            self.legend.delete("all")
            self.legend.create_rectangle(20, 6, 34, 20, fill="#ffc828", outline="")
            self.legend.create_text(40, 13, text="foreground pixel", anchor="w", fill=THEME["muted"],
                                    font=self.fonts["small"])
            self.legend.create_rectangle(170, 6, 184, 20, outline="#00e5ff", width=2)
            self.legend.create_text(190, 13, text="tracked blob (P = primary)", anchor="w", fill=THEME["muted"],
                                    font=self.fonts["small"])

        self.photo = tk.PhotoImage(data=toPpm(image), format="PPM").zoom(THERMAL_SCALE, THERMAL_SCALE)
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.photo, anchor="nw")
        if key == "mask":
            for (min_row, min_col, max_row, max_col), track_id, primary in stages["detections"]:
                colour = "#00e5ff" if primary else "#ffffff"
                canvas.create_rectangle(min_col * THERMAL_SCALE, min_row * THERMAL_SCALE,
                                        (max_col + 1) * THERMAL_SCALE, (max_row + 1) * THERMAL_SCALE,
                                        outline=colour, width=2)
                canvas.create_text(min_col * THERMAL_SCALE + 3, min_row * THERMAL_SCALE + 2, anchor="nw",
                                   text=f"{'P ' if primary else ''}ID {track_id}", fill=colour,
                                   font=self.fonts["small_bold"])
        if stages["glitch"]:
            canvas.create_text(8, 8, anchor="nw", text="Glitch frame - previous features held",
                               fill=THEME["red"], font=self.fonts["small_bold"])


class LiveGUI:
    def __init__(self, root, system):
        self.root = root
        self.system = system
        self.snapshot = system.snapshot
        self.shutdown = system.stop
        root.title("Outdoor Intrusion Detection — Thresholded PIR vs Fused System")
        root.geometry(WINDOW_SIZE)
        root.minsize(1060, 700)
        root.configure(bg=THEME["bg"])
        root.protocol("WM_DELETE_WINDOW", self.close)

        family = pickFont(root)
        fonts = {
            "title": (family, 22, "bold"), "card": (family, 15, "bold"), "state": (family, 20, "bold"),
            "metric": (family, 20, "bold"), "body": (family, 11), "body_bold": (family, 11, "bold"),
            "small": (family, 9), "small_bold": (family, 9, "bold"), "lamp": (family, 9, "bold"),
            "pill": (family, 10, "bold"), "pill_large": (family, 12, "bold"), "button": (family, 11, "bold"),
            "mono": ("DejaVu Sans Mono", 10),
        }
        self.main_view = MainView(root, fonts, self.toggle_view, self.change_threshold)
        self.sensor_view = SensorView(root, fonts, self.toggle_view)
        self.main_view.pack(fill="both", expand=True)
        self.showing_sensors = False
        self.events_seen = 0
        # Live PIR trace for the sensor view (filled even while the main view is shown)
        self.pir_trace = deque(maxlen=int(PIR_DISPLAY_S * sampleRateHz * 1.5) + 10)

        root.bind_all("<space>", lambda _event: self.toggle_view())
        root.bind_all("<KeyPress-e>", lambda _event: self.system.mark_entry())
        root.bind_all("<KeyPress-E>", lambda _event: self.system.mark_entry())
        root.bind_all("<KeyPress-r>", lambda _event: self.reset())
        root.bind_all("<KeyPress-R>", lambda _event: self.reset())
        root.bind_all("<KeyPress-c>", lambda _event: self.system.request_recalibration())
        root.bind_all("<KeyPress-C>", lambda _event: self.system.request_recalibration())
        root.bind_all("<KeyPress-f>", lambda _event: self.set_fullscreen(not self.fullscreen))
        root.bind_all("<KeyPress-F>", lambda _event: self.set_fullscreen(not self.fullscreen))
        root.bind_all("<Up>", lambda _event: self.change_threshold(+1))
        root.bind_all("<Down>", lambda _event: self.change_threshold(-1))
        root.bind_all("<Escape>", lambda _event: self.set_fullscreen(False))
        self.fullscreen = False
        for index in range(len(THERMAL_STAGES)):
            root.bind_all(str(index + 1), lambda _event, i=index: self.sensor_view.set_stage(i))
        self.refresh()

    def toggle_view(self):
        self.showing_sensors = not self.showing_sensors
        if self.showing_sensors:
            self.main_view.pack_forget()
            self.sensor_view.pack(fill="both", expand=True)
        else:
            self.sensor_view.pack_forget()
            self.main_view.pack(fill="both", expand=True)

    def drain_pir(self):
        source = self.system.pirProcessor.outputQueue
        while True:
            try:
                self.pir_trace.append(source.get_nowait())
            except queue.Empty:
                break
        if self.pir_trace:
            cutoff = self.pir_trace[-1][0] - PIR_DISPLAY_S
            while self.pir_trace and self.pir_trace[0][0] < cutoff:
                self.pir_trace.popleft()

    def refresh(self):
        item = self.snapshot()
        self.drain_pir()
        # New log events from the processing thread (none are missed between refreshes)
        events = self.system.detection_events()
        for event in events[self.events_seen:]:
            self.main_view.add_event(event)
        self.events_seen = len(events)
        self.main_view.update_response(self.system.entry_snapshot())
        self.main_view.update_view(item)
        if self.showing_sensors:
            processor = self.system.pirProcessor
            # Samples older than PIR_TIMEOUT_S are shown as missing
            fresh = bool(self.pir_trace) and time.time() - self.pir_trace[-1][0] <= PIR_TIMEOUT_S
            self.sensor_view.draw_decision(item)
            self.sensor_view.draw_pir(list(self.pir_trace) if fresh else [], processor.getBaseline(),
                                      processor.initialised, self.system.model.pir_threshold)
            self.sensor_view.draw_thermal(self.system.stage_snapshot())
        self.root.after(250, self.refresh)

    def reset(self):
        self.system.clear_entry()
        self.main_view.reset()

    # Up / Down Or - / +: changes the thresholded-PIR comparison level by one step
    def change_threshold(self, steps):
        volts = self.system.set_pir_threshold(self.system.model.pir_threshold + steps * PIR_THRESHOLD_STEP_V)
        self.main_view.show_threshold(volts)

    def set_fullscreen(self, enabled):
        self.fullscreen = enabled
        self.root.attributes("-fullscreen", enabled)

    def close(self):
        self.shutdown()
        self.root.destroy()


class RealTimeSystem:
    def __init__(self, folder):
        self.model = RealtimeModel(folder / "fusion_gb.json", folder / "pir_backup_gb.json")
        self.window = self.model.window
        self.backgroundPath = folder / BACKGROUND_FILE
        self.acquisitionQueue = queue.Queue(maxsize=4096)
        self.pirAcquisition = PIRAcquisition(self.acquisitionQueue)
        self.pirProcessor = PIRProcessor(self.acquisitionQueue, windowTime=self.window)
        self.thermal = ThermopilePipeline(window_seconds=self.window)
        self.stopEvent = threading.Event()
        # C key: recalibrate the thermal background on the next frames (zone must be empty)
        self.recalibrateEvent = threading.Event()
        self.lock = threading.Lock()
        self.latest = {"warmup": "Preparing sensors...", "status": "Starting"}
        # Thermal frames are read on their own thread so a failed or hung read never
        # stops the decisions; the decision thread waits on this queue.
        self.thermalQueue = queue.Queue(maxsize=8)
        self.thermalReader = threading.Thread(target=self.run_thermal_acquisition, daemon=True)
        self.worker = threading.Thread(target=self.run_thermal, daemon=True)
        # Latest thermal processing stages for the sensor view
        self.stages = None
        self.frameCount = 0
        # Console notices are printed only when a condition changes
        self.notices = {}
        # GUI log events (dicts): fused detections, entries marked with E, and response times
        self.events = []
        # Current timed entry (E key): wall-clock time and each system's response in seconds
        self.entry = None
        self.lastFusionState = None
        self.lastPirState = None
        self.processingMaxMs = 0.0
        # Decision log (opened once the system is ready)
        self.logFile = None
        self.logWriter = None
        self.logPath = None
        if not self.model.has_backup:
            print(f"WARNING: {self.model.backup_problem}. Without thermal frames there will be no decisions.",
                  flush=True)

    def publish(self, **updates):
        with self.lock:
            self.latest.update(updates)

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def detection_events(self):
        with self.lock:
            return list(self.events)

    # E Key: the person is entering now; both systems are timed to their first INTRUDER
    def mark_entry(self):
        now = time.time()
        with self.lock:
            self.entry = {"time": now, "fused_s": None, "pir_s": None}
            self.events.append({"kind": "entry", "time": now})
        print(time.strftime("[%H:%M:%S] ") + "Entry marked - timing fused system and thresholded PIR", flush=True)

    # Sets The Thresholded-PIR Comparison Level (the fused system is not affected)
    def set_pir_threshold(self, volts):
        volts = round(min(max(float(volts), PIR_THRESHOLD_MIN_V), PIR_THRESHOLD_MAX_V), 2)
        with self.lock:
            if volts != self.model.pir_threshold:
                self.model.pir_threshold = volts
                self.events.append({"kind": "threshold", "time": time.time(), "volts": volts})
                print(time.strftime("[%H:%M:%S] ") + f"Thresholded PIR threshold set to {volts:.2f} V", flush=True)
        return volts

    # C Key: take a new thermal background (setup only - nobody may be in view)
    def request_recalibration(self):
        self.recalibrateEvent.set()
        print(time.strftime("[%H:%M:%S] ") + "Recalibrating thermopile - keep the detection zone EMPTY "
              f"for the next {CALIBRATION_FRAMES} frames", flush=True)

    def clear_entry(self):
        with self.lock:
            self.entry = None

    def entry_snapshot(self):
        with self.lock:
            return None if self.entry is None else dict(self.entry)

    def stage_snapshot(self):
        with self.lock:
            return self.stages

    # Prints A Console Message Only When The Named Condition Changes
    def notice(self, name, active, message_on, message_off=None):
        if self.notices.get(name, False) == active:
            return
        self.notices[name] = active
        message = message_on if active else message_off
        if message:
            print(time.strftime("[%H:%M:%S] ") + message, flush=True)

    def start(self):
        self.pirProcessor.start()
        self.pirAcquisition.start()
        self.thermalReader.start()
        self.worker.start()

    def stop(self):
        self.stopEvent.set()
        self.pirAcquisition.stop()
        self.pirProcessor.stop()
        self.closeLog()

    # Opens A New CSV Log For This Run (never overwrites an earlier run)
    def openLog(self):
        LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        path = LOG_FOLDER / time.strftime("live_%Y%m%d_%H%M%S.csv")
        self.logFile = open(path, "w", newline="", encoding="utf-8")
        self.logWriter = csv.writer(self.logFile)
        self.logWriter.writerow(LOG_COLUMNS)
        self.logFile.flush()
        return path

    # Writes One Decision; flushed every row so a crash keeps everything logged so far
    def logDecision(self, timestamp, result, processingMs):
        if self.logWriter is None:
            return
        self.logWriter.writerow([f"{timestamp:.3f}", result["pir_state"], result["fusion_state"],
                                 f"{result['fusion_confidence']:.4f}", f"{result['pir_relative_voltage']:.4f}",
                                 f"{processingMs:.2f}", f"{result['pir_threshold']:.2f}"])
        self.logFile.flush()

    def closeLog(self):
        if self.logFile is not None:
            self.logFile.close()
            self.logFile = None
            self.logWriter = None

    # Saves The Start-Up Background So A Later Run Without A Working Thermopile At Start-Up Can Reuse It
    def saveBackground(self):
        data = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_shape": list(self.thermal.frame_shape),
                "background": self.thermal.bg_model.bg.reshape(-1).astype(float).tolist(),
                # Uncorrected calibration sigma (same as the recordings); loading applies the denoise scale
                "sigma": self.thermal.bg_model.calibration_sigma.reshape(-1).astype(float).tolist()}
        self.backgroundPath.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.backgroundPath.with_suffix(".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        os.replace(temporary, self.backgroundPath)

    # Loads The Most Recent Saved Background; returns its save time or None
    def loadBackground(self):
        if not self.backgroundPath.is_file():
            return None
        try:
            data = json.loads(self.backgroundPath.read_text(encoding="utf-8"))
            self.thermal.load_calibration(data["background"], data["sigma"])
            return data.get("saved", "unknown time")
        except (OSError, ValueError, KeyError) as exc:
            self.notice("bad_background", True, f"Could not load saved thermopile background "
                        f"{self.backgroundPath}: {exc}")
            return None

    # Thermal Acquisition Thread: reads frames and hands them to the decision thread
    def run_thermal_acquisition(self):
        failing = False
        while not self.stopEvent.is_set():
            try:
                frame = getFrame()
                timestamp = time.time()  # Timestamp immediately after a complete sensor read.
            except Exception as exc:
                if not failing:
                    print(time.strftime("[%H:%M:%S] ") + f"Thermopile read failed ({type(exc).__name__}: {exc}); "
                          "retrying", flush=True)
                failing = True
                self.stopEvent.wait(0.5)
                continue
            if failing:
                print(time.strftime("[%H:%M:%S] ") + "Thermopile reading again", flush=True)
                failing = False
            frame = np.asarray(frame, dtype=np.float64).reshape(FRAME_SHAPE)
            try:
                self.thermalQueue.put_nowait((timestamp, frame))
            except queue.Full:
                # Keep the newest frames if the decision thread falls behind
                try:
                    self.thermalQueue.get_nowait()
                except queue.Empty:
                    pass
                self.thermalQueue.put_nowait((timestamp, frame))

    def run_thermal(self):
        try:
            self._run_thermal()
        except Exception as exc:
            self.publish(status=f"ERROR: {type(exc).__name__}: {exc}",
                         pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE", warmup="System error")
            print(f"System error: {type(exc).__name__}: {exc}", flush=True)
            self.stopEvent.set()

    # True While PIR Samples Are Arriving (console message when that changes)
    def pirReceiving(self, now):
        lastTime = self.pirProcessor.lastTime
        receiving = lastTime is not None and now - lastTime <= PIR_TIMEOUT_S
        if lastTime is not None or now - self.startTime > STARTUP_THERMAL_TIMEOUT_S:
            self.notice("pir_missing", not receiving,
                        "No PIR data incoming - PIR decisions paused, thermopile processing continues",
                        "PIR data received again")
        return receiving

    # Runs One Frame Through The Thermal Pipeline And Keeps Its Stages For The Sensor View
    def processThermal(self, frame, timestamp):
        background_before = self.thermal.bg_model.bg.copy()
        features = self.thermal.process(frame, timestamp)
        glitch = not self.thermal.last_frame_valid
        detections = [(tuple(int(v) for v in d.blob.bbox), int(d.features.track_id),
                       d.features.track_id == self.thermal.primary_track_id)
                      for d in self.thermal.last_detections]
        self.frameCount += 1
        stages = {
            "frame_id": self.frameCount, "timestamp": timestamp, "glitch": glitch,
            "raw": frame.copy(), "background": background_before,
            "z": np.array(self.thermal.last_raw_z), "smoothed_z": np.array(self.thermal.last_smoothed_z),
            "mask": np.array(self.thermal.last_mask, dtype=bool),
            "detections": detections, "scene_temp": float(self.thermal.last_scene_temp),
        }
        with self.lock:
            self.stages = stages
        return features

    # One Decision: fused when thermalFeatures is given, PIR-only backup when None
    def decide(self, timestamp, thermalFeatures, started, thermalReceiving):
        now = time.time()
        pirReceiving = self.pirReceiving(now)
        self.notice("no_sensors", not pirReceiving and not thermalReceiving,
                    "No data from either sensor - no decisions until a sensor returns",
                    "Sensor data received again")
        if not pirReceiving:
            self.lastFusionState = None
            self.lastPirState = None
            self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                         fusion_confidence=None, pir_relative_voltage=None,
                         warmup="PIR not receiving" + ("" if thermalReceiving else " · thermopile not receiving"),
                         status="No PIR samples - waiting for the PIR")
            return
        if not self.pirProcessor.initialised:
            self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                         fusion_confidence=None, pir_relative_voltage=None,
                         warmup="WARM-UP: PIR baseline - keep detection zone empty",
                         status=f"PIR warming up ({warmupTime} s)")
            return
        samples = self.pirProcessor.getWindowSamples(self.window, endTime=timestamp)
        result, problem = self.model.fuse(thermalFeatures, samples, timestamp)
        if result is None:
            self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                         fusion_confidence=None, pir_relative_voltage=None, status=problem)
            return
        self.notice("pir_backup", result["mode"] == MODE_PIR_BACKUP,
                    "No thermopile data incoming - using the PIR-only backup classifier",
                    "Thermopile data back - fused PIR + thermopile system resumed")
        if self.logPath is None:
            self.logPath = self.openLog()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.logDecision(timestamp, result, elapsed_ms)
        self.processingMaxMs = max(self.processingMaxMs, elapsed_ms)
        # A detection is every change of the fused state into MAYBE INTRUDER or INTRUDER
        state = result["fusion_state"]
        alarmTime = time.time()  # Alarm output time (after processing)
        with self.lock:
            if state != self.lastFusionState and state in ("MAYBE INTRUDER", "INTRUDER"):
                self.events.append({"kind": "detection", "time": alarmTime, "state": state,
                                    "confidence": result["fusion_confidence"], "pir_state": result["pir_state"]})
            # Response time of each system to a marked entry: its first fresh change INTO INTRUDER
            # after E was pressed (an alarm still on from the previous walker does not count)
            if self.entry is not None:
                for key, system, systemState, previous in (
                        ("fused_s", "Fused System", state, self.lastFusionState),
                        ("pir_s", "Thresholded PIR", result["pir_state"], self.lastPirState)):
                    if self.entry[key] is None and systemState == "INTRUDER" and previous != "INTRUDER":
                        seconds = max(0.0, alarmTime - self.entry["time"])
                        self.entry[key] = seconds
                        self.events.append({"kind": "response", "time": alarmTime, "system": system,
                                            "seconds": seconds})
                        print(time.strftime("[%H:%M:%S] ") + f"{system} INTRUDER {seconds:.2f} s after entry",
                              flush=True)
        self.lastFusionState = state
        self.lastPirState = result["pir_state"]
        self.publish(warmup="READY", **result, processing_ms=elapsed_ms, processing_max_ms=self.processingMaxMs,
                     status=f"PIR = 32 Hz | Thermal = 2 Hz")

    def _run_thermal(self):
        # Calibrate the PIR baseline and thermal background with an empty detection zone.
        # Collect thermal calibration frames concurrently with the PIR warm-up.
        # Start-up frames are discarded first, then CALIBRATION_FRAMES frames are kept
        # (same as the recordings). The PIR warm-up runs at the same time.
        # If the thermopile is not working at start-up, the PIR-only backup is used and
        # the most recent saved background is loaded once thermal frames arrive.
        self.startTime = time.time()
        calibrations = []
        discarded = 0
        calibrated = False
        startupFailed = False
        lastFrameTime = None
        validSince = None
        fusedActive = False

        while not self.stopEvent.is_set():
            try:
                timestamp, frame = self.thermalQueue.get(timeout=DECISION_PERIOD_S)
            except queue.Empty:
                timestamp = frame = None
            now = time.time()
            self.pirReceiving(now)

            # ---------------- A thermal frame arrived ----------------
            if frame is not None:
                gap = lastFrameTime is not None and timestamp - lastFrameTime > THERMAL_TIMEOUT_S
                lastFrameTime = timestamp

                # C key pressed: start a fresh calibration (no start-up frames to discard)
                if self.recalibrateEvent.is_set():
                    self.recalibrateEvent.clear()
                    calibrations = []
                    discarded = CALIBRATION_DISCARD_FRAMES
                    calibrated = False
                    startupFailed = False
                    fusedActive = False
                    validSince = None
                    # Old alarm is not carried over; the fused card shows warm-up until it is ready again
                    for machine in (self.model.fusion_state, self.model.backup_state):
                        if machine is not None:
                            machine.state = NO_INTRUDER
                            machine.below_since = None
                    self.publish(fusion_state="", fusion_confidence=None)

                if not calibrated and not startupFailed and not gap:
                    # Normal start-up calibration
                    if discarded < CALIBRATION_DISCARD_FRAMES:
                        discarded += 1
                        self.publish(warmup="WARM-UP: sensor settling — keep detection zone empty",
                                     status=f"Discarding start-up frames · {discarded}/{CALIBRATION_DISCARD_FRAMES}")
                        continue
                    calibrations.append(frame)
                    self.publish(warmup="WARM-UP: calibrating — keep detection zone empty",
                                 status=f"Calibrating thermopile · {len(calibrations)}/{CALIBRATION_FRAMES} frames"
                                        + ("" if self.pirProcessor.initialised else " · PIR warming up"))
                    if len(calibrations) < CALIBRATION_FRAMES:
                        continue
                    self.thermal.calibrate(calibrations)
                    calibrated = True
                    validSince = timestamp
                    try:
                        self.saveBackground()
                        print(time.strftime("[%H:%M:%S] ") + f"Thermopile calibrated; background saved to "
                              f"{self.backgroundPath}", flush=True)
                    except OSError as exc:
                        print(f"Could not save thermopile background: {exc}", flush=True)
                    self.publish(warmup="Thermopile and PIR calibrated; filling PIR history...",
                                 status=f"Waiting for complete {self.window:g}-second PIR window")
                    continue

                if not calibrated:
                    # Thermopile was not working at start-up: no time to recalibrate, so
                    # use the most recent saved background
                    saved = self.loadBackground()
                    if saved is None:
                        self.notice("no_background", True, "Thermopile frames arriving but no saved background "
                                    f"({self.backgroundPath}) - restart main.py with the zone empty to calibrate")
                        self.decide(timestamp, None, time.monotonic(), True)
                        continue
                    calibrated = True
                    validSince = None
                    print(time.strftime("[%H:%M:%S] ") + f"Thermopile frames arriving - using saved background "
                          f"from {saved}", flush=True)
                elif gap:
                    # Frames are back after an outage: keep the background, clear tracks
                    self.thermal.reset()
                    validSince = None

                started = time.monotonic()
                features = self.processThermal(frame, timestamp)
                if self.thermal.last_frame_valid:
                    if validSince is None:
                        validSince = timestamp
                elif not fusedActive:
                    validSince = None
                # Back to (or start) the fused system after one window of valid frames
                if not fusedActive and validSince is not None and timestamp - validSince >= self.window:
                    fusedActive = True
                if fusedActive:
                    self.decide(timestamp, features, started, True)
                elif startupFailed or self.model.mode == MODE_PIR_BACKUP:
                    # Thermal still re-settling after an outage: stay on the PIR backup
                    self.decide(timestamp, None, started, True)
                continue

            # ---------------- No thermal frame this period ----------------
            thermalSilent = lastFrameTime is None or now - lastFrameTime > THERMAL_TIMEOUT_S
            if not calibrated and not startupFailed:
                if (lastFrameTime is None and now - self.startTime > STARTUP_THERMAL_TIMEOUT_S) or \
                        (lastFrameTime is not None and thermalSilent):
                    startupFailed = True
                    calibrations = []
                    self.notice("pir_backup", True,
                                "No thermopile data incoming at start-up - using PIR data only")
                    self.publish(warmup="Thermopile not receiving", status="Using PIR data only")
                else:
                    continue
            if not thermalSilent:
                continue  # Between thermal frames
            fusedActive = False
            validSince = None
            self.decide(now, None, time.monotonic(), False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("Processed/realtime"))
    parser.add_argument("--pir-threshold", type=float, default=PIR_THRESHOLD,
                        help="Starting thresholded-PIR comparison level in volts (change live with Up / Down)")
    args = parser.parse_args()
    system = RealTimeSystem(args.model_dir)
    system.set_pir_threshold(args.pir_threshold)
    root = tk.Tk()
    LiveGUI(root, system)
    system.start()
    try:
        root.mainloop()
    finally:
        system.stop()

if __name__ == "__main__":
    main()

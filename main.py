"""Start the complete live sensor-fusion system on the ROCK 4C+."""
import argparse
import queue
import threading
import time
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIR.interface import PIRAcquisition, sampleRateHz
from PIR.processing import (PIRProcessor, getHistoryFeatureNames,
                            getHistoryFeatureVector, warmupTime)
from Thermopile.interface import getFrame, REFRESH_RATE_HZ
from Thermopile.processing import LiveThermalProcessor, getThermalFeatureNames
from Fusion.realtimeModel import RealtimeModel

class StateLights(ttk.LabelFrame):
    """Three explicit lights; only the currently reported state is illuminated."""
    COLORS = ("#2ecc71", "#f1c40f", "#e74c3c")
    LABELS = ("CLEAR", "POSSIBLE", "INTRUDER")

    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding=12)
        self.canvas = tk.Canvas(self, height=100, bg="#202630", highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)
        self.lamps = []
        for i, (color, label) in enumerate(zip(self.COLORS, self.LABELS)):
            x = 50 + 110 * i
            lamp = self.canvas.create_oval(x - 17, 12, x + 17, 46,
                                           fill="#3b414b", outline="#87909c", width=2)
            self.canvas.create_text(x, 67, text=label, fill="#f2f3f5",
                                    font=("Arial", 9, "bold"))
            self.lamps.append(lamp)
        self.status = ttk.Label(self, text="Waiting for sensors", anchor="center")
        self.status.pack(fill="x", pady=(6, 0))

    def set_state(self, state):
        if state == "NO INTRUDER":
            selected = 0
        elif state == "MAYBE INTRUDER":
            selected = 1
        elif state == "INTRUDER":
            selected = 2
        else:
            selected = None
        for i, lamp in enumerate(self.lamps):
            self.canvas.itemconfigure(lamp, fill=self.COLORS[i] if i == selected else "#3b414b")
        self.status.configure(text="Sensor unavailable / warming up" if selected is None
                              else "Live decision")


class LiveGUI:
    def __init__(self, root, snapshot, shutdown):
        self.root = root
        self.snapshot = snapshot
        self.shutdown = shutdown
        root.title("Outdoor Intrusion Detection — PIR and Fusion")
        root.geometry("780x345")
        root.minsize(735, 320)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.warm = tk.StringVar(value="Starting sensors...")
        self.confidence = tk.StringVar(value="--")
        self.pir_voltage = tk.StringVar(value="--")
        self.health = tk.StringVar(value="Initialising")

        panel = ttk.Frame(root, padding=18)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="OUTDOOR INTRUSION DETECTION", font=("Arial", 17, "bold")).pack(anchor="w")
        ttk.Label(panel, textvariable=self.warm).pack(anchor="w", pady=(5, 10))

        cards = ttk.Frame(panel)
        cards.pack(fill="x", expand=True)
        self.pir_lights = StateLights(cards, "PIR ONLY — voltage threshold")
        self.pir_lights.pack(side="left", fill="both", expand=True, padx=(0, 7))
        self.fused_lights = StateLights(cards, "FUSED — PIR + THERMOPILE")
        self.fused_lights.pack(side="left", fill="both", expand=True, padx=(7, 0))

        metrics = ttk.Frame(panel)
        metrics.pack(fill="x", pady=(13, 7))
        ttk.Label(metrics, text="PIR relative voltage:").pack(side="left")
        ttk.Label(metrics, textvariable=self.pir_voltage, font=("Arial", 11, "bold")).pack(side="left", padx=(6, 28))
        ttk.Label(metrics, text="Fusion human probability:").pack(side="left")
        ttk.Label(metrics, textvariable=self.confidence, font=("Arial", 11, "bold")).pack(side="left", padx=(6, 0))
        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, textvariable=self.health).pack(anchor="w")
        self.refresh()

    def refresh(self):
        item = self.snapshot()
        self.warm.set(item.get("warmup", ""))
        self.pir_lights.set_state(item.get("pir_state", "WARMING UP"))
        self.fused_lights.set_state(item.get("fusion_state", "WARMING UP"))
        score = item.get("fusion_confidence")
        self.confidence.set("--" if score is None else f"{score:.1%}")
        voltage = item.get("pir_relative_voltage")
        self.pir_voltage.set("--" if voltage is None else f"{voltage:+.3f} V")
        self.health.set(item.get("status", ""))
        self.root.after(250, self.refresh)

    def close(self):
        self.shutdown()
        self.root.destroy()


class RealTimeSystem:
    def __init__(self, folder):
        self.model = RealtimeModel(folder / "fusion_rf.json")
        self.window = self.model.fusion.window
        self.acquisitionQueue = queue.Queue(maxsize=4096)
        self.pirAcquisition = PIRAcquisition(self.acquisitionQueue)
        self.pirProcessor = PIRProcessor(self.acquisitionQueue, windowTime=self.window)
        self.thermal = LiveThermalProcessor(window_seconds=self.window)
        self.stopEvent = threading.Event()
        self.lock = threading.Lock()
        self.latest = {"warmup": "Preparing sensors...", "status": "Starting"}
        self.worker = threading.Thread(target=self.run_thermal, daemon=True)

    def publish(self, **updates):
        with self.lock:
            self.latest.update(updates)

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def start(self):
        self.pirProcessor.start()
        self.pirAcquisition.start()
        self.worker.start()

    def stop(self):
        self.stopEvent.set()
        self.pirAcquisition.stop()
        self.pirProcessor.stop()

    def run_thermal(self):
        try:
            self._run_thermal()
        except Exception as exc:
            self.publish(status=f"ERROR: {type(exc).__name__}: {exc}",
                         pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE", warmup="System error")
            self.stopEvent.set()

    def _run_thermal(self):
        # Calibrate the PIR baseline and thermal background with an empty detection zone.
        # Collect thermal calibration frames concurrently with the PIR warm-up.
        calibrations = []
        start = time.monotonic()
        warm_seconds = max(float(warmupTime) + 1, 20.0 / REFRESH_RATE_HZ)
        while not self.stopEvent.is_set():
            frame = np.asarray(getFrame(), dtype=np.float64).reshape(24, 32)
            calibrations.append(frame)
            remaining = max(0, int(round(warm_seconds - (time.monotonic() - start))))
            self.publish(warmup=f"WARM-UP: {remaining}s remaining — keep detection zone empty",
                         status=f"Calibrating PIR and thermopile · {len(calibrations)} thermal frames")
            if time.monotonic() - start >= warm_seconds and self.pirProcessor.initialised and len(calibrations) >= 20:
                break
        if self.stopEvent.is_set():
            return
        self.thermal.calibrate(calibrations)
        # Require a fresh full PIR window after both calibrations are finished.
        ready_at = time.time() + self.window
        self.publish(warmup="Thermopile and PIR calibrated; filling PIR history...",
                     status=f"Waiting for complete {self.window:g}-second PIR window")
        while not self.stopEvent.is_set():
            frame = getFrame()
            timestamp = time.time()  # Timestamp immediately after a complete sensor read.
            if timestamp < ready_at:
                self.thermal.process(timestamp, frame)
                continue
            # Keep thermal tracking and background adaptation running even if PIR is unavailable.
            started = time.monotonic()
            thermal_features = self.thermal.process(timestamp, frame)
            if not self.pirProcessor.initialised or self.pirProcessor.lastError is not None:
                self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                             status=f"PIR error: {self.pirProcessor.lastError}")
                continue
            samples = self.pirProcessor.getWindowSamples(self.window, endTime=timestamp)
            if len(samples) < 2 or samples[0][0] > timestamp - self.window + 0.15:
                self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                             status="Insufficient PIR history; waiting for samples")
                continue
            newest = samples[-1][0]
            if timestamp - newest > 0.3:
                self.publish(pir_state="UNAVAILABLE", fusion_state="UNAVAILABLE",
                             status="Stale PIR samples")
                continue
            history = getHistoryFeatureVector(samples)
            row = {}
            row.update({f"pir{name.capitalize()}": float(v) for name, v in zip(getHistoryFeatureNames(), history)})
            row.update({f"thermal{name.capitalize()}": float(thermal_features[name]) for name in getThermalFeatureNames()})
            result = self.model.predict(row, timestamp, pirVoltage=samples[-1][1])
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self.publish(warmup="READY — both sensors calibrated", **result,
                         status=f"PIR {sampleRateHz} Hz target · Thermal {REFRESH_RATE_HZ:g} Hz target · Processing {elapsed_ms:.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("Processed/realtime"))
    args = parser.parse_args()
    system = RealTimeSystem(args.model_dir)
    root = tk.Tk()
    LiveGUI(root, system.snapshot, system.stop)
    system.start()
    try:
        root.mainloop()
    finally:
        system.stop()

if __name__ == "__main__":
    main()

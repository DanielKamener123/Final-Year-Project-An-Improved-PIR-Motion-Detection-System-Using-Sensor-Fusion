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
from PIR.processing import (PIRProcessor, getCurrentFeatureNames, getHistoryFeatureNames,
                            getCurrentFeatureVector, getHistoryFeatureVector, warmupTime)
from Thermopile.interface import getFrame, REFRESH_RATE_HZ
from Thermopile.processing import LiveThermalProcessor, getThermalFeatureNames
from Fusion.realtimeModel import RealtimeModel

class LiveGUI:
    def __init__(self, root, snapshot, shutdown):
        self.root = root
        self.snapshot = snapshot
        self.shutdown = shutdown
        root.title("PIR + Thermopile Intrusion Detection")
        root.geometry("580x310")
        root.minsize(520, 280)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.warm = tk.StringVar(value="Starting sensors...")
        self.pir = tk.StringVar(value="WARMING UP")
        self.fusion = tk.StringVar(value="WARMING UP")
        self.confidence = tk.StringVar(value="--")
        self.health = tk.StringVar(value="Initialising")
        panel = ttk.Frame(root, padding=18)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="INTRUSION DETECTION SYSTEM", font=("Arial", 16, "bold")).pack(anchor="w")
        ttk.Label(panel, textvariable=self.warm, font=("Arial", 12)).pack(anchor="w", pady=(8, 12))
        cards = ttk.Frame(panel)
        cards.pack(fill="x")
        for title, var in (("PIR THRESHOLD", self.pir), ("FUSED STATE", self.fusion)):
            card = ttk.LabelFrame(cards, text=title, padding=14)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(card, textvariable=var, font=("Arial", 14, "bold")).pack()
        ttk.Label(panel, text="Fusion human probability", font=("Arial", 10)).pack(anchor="w", pady=(14, 0))
        ttk.Label(panel, textvariable=self.confidence, font=("Arial", 14, "bold")).pack(anchor="w")
        ttk.Separator(panel).pack(fill="x", pady=10)
        ttk.Label(panel, textvariable=self.health).pack(anchor="w")
        self.refresh()

    def refresh(self):
        item = self.snapshot()
        self.warm.set(item.get("warmup", ""))
        self.pir.set(item.get("pir_state", "WARMING UP"))
        self.fusion.set(item.get("fusion_state", "WARMING UP"))
        score = item.get("fusion_confidence")
        self.confidence.set("--" if score is None else f"{score:.1%}")
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
                     status="Waiting for complete 1.5-second PIR window")
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
            current = getCurrentFeatureVector(samples[-1][1])
            history = getHistoryFeatureVector(samples)
            row = {}
            row.update({f"pir{name.capitalize()}": float(v) for name, v in zip(getCurrentFeatureNames(), current)})
            row.update({f"pir{name.capitalize()}": float(v) for name, v in zip(getHistoryFeatureNames(), history)})
            row.update({f"thermal{name.capitalize()}": float(thermal_features[name]) for name in getThermalFeatureNames()})
            result = self.model.predict(row)
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

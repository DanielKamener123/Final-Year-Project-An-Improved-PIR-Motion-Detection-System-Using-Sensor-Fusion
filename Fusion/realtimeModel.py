"""NumPy-only Random Forest inference from exported scikit-learn tree parameters."""
import json
import time
from pathlib import Path
import numpy as np
from PIR.processing import threshold as PIR_THRESHOLD

STATES = ("NO INTRUDER", "MAYBE INTRUDER", "INTRUDER")

class RandomForest:
    def __init__(self, filename):
        self.data = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.features = list(self.data["features"])
        self.fill = np.asarray(self.data["medians"], dtype=float)
        self.trees = self.data["trees"]
        self.window = float(self.data["window"])
        self.thresholds = self.data["thresholds"]

    def confidence(self, features):
        if isinstance(features, dict):
            values = np.asarray([features[n] for n in self.features], dtype=float)
        else:
            values = np.asarray(features, dtype=float)
        if values.shape != self.fill.shape:
            raise ValueError(f"Expected {len(self.features)} features; got {values.shape}")
        values = np.where(np.isnan(values), self.fill, values).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature vector")
        probs = []
        for tree in self.trees:
            node = 0
            while tree["left"][node] != -1:
                node = tree["left"][node] if float(values[tree["feature"][node]]) <= tree["threshold"][node] else tree["right"][node]
            probs.append(tree["probability"][node])
        return float(np.mean(probs))

class StateMachine:
    def __init__(self, thresholds):
        self.maybe_enter = float(thresholds["maybe_enter"])
        self.maybe_exit = float(thresholds["maybe_exit"])
        self.intruder_enter = float(thresholds["intruder_enter"])
        self.intruder_exit = float(thresholds["intruder_exit"])
        # Off-delay: INTRUDER only drops after the score has stayed below
        # intruder_exit for hold_s seconds (0 = original behaviour).
        self.hold_s = float(thresholds.get("hold_s", 0.0))
        if not 0 <= self.maybe_exit < self.maybe_enter < self.intruder_exit < self.intruder_enter <= 1:
            raise ValueError("Invalid three-state hysteresis thresholds")
        if self.hold_s < 0:
            raise ValueError("hold_s must be >= 0")
        self.state = 0
        self.below_since = None

    # Same logic as Fusion/alarm_tuning.run_state_machine (used offline)
    def update(self, score, timestamp=None):
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.state == 0:
            if score >= self.intruder_enter:
                self.state = 2
            elif score >= self.maybe_enter:
                self.state = 1
        elif self.state == 1:
            if score >= self.intruder_enter:
                self.state = 2
            elif score < self.maybe_exit:
                self.state = 0
        else:
            if score >= self.intruder_exit:
                self.below_since = None
            else:
                if self.below_since is None:
                    self.below_since = now
                if now - self.below_since >= self.hold_s:
                    self.state = 1 if score >= self.maybe_exit else 0
                    self.below_since = None
        return STATES[self.state]

class RealtimeModel:
    def __init__(self, fusion_path, pir_path=None):
        # PIR-only is a fixed signal threshold, NOT a trained model.
        self.fusion = RandomForest(fusion_path)
        self.fusion_state = StateMachine(self.fusion.thresholds)

    def predict(self, row, timestamp=None, pirVoltage=None):
        # PIR-only baseline uses the newest relative PIR sample (not a model feature)
        voltage = float(pirVoltage) if pirVoltage is not None else float("nan")
        if not np.isfinite(voltage):
            raise ValueError("Invalid PIR relative voltage")
        pir_active = abs(voltage) >= float(PIR_THRESHOLD)
        fusion_score = self.fusion.confidence(row)
        return {
            "pir_state": "INTRUDER" if pir_active else "NO INTRUDER",
            "fusion_state": self.fusion_state.update(fusion_score, timestamp),
            "pir_confidence": None,
            "pir_relative_voltage": voltage,
            "pir_threshold": float(PIR_THRESHOLD),
            "fusion_confidence": fusion_score,
        }

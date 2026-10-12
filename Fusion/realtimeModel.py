"""NumPy-only Random Forest inference from exported scikit-learn tree parameters."""
import json
from pathlib import Path
import numpy as np

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
        if not 0 <= self.maybe_exit < self.maybe_enter < self.intruder_exit < self.intruder_enter <= 1:
            raise ValueError("Invalid three-state hysteresis thresholds")
        self.state = 0

    def update(self, score):
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
        elif score < self.intruder_exit:
            self.state = 1 if score >= self.maybe_exit else 0
        return STATES[self.state]

class RealtimeModel:
    def __init__(self, fusion_path, pir_path):
        self.fusion = RandomForest(fusion_path)
        self.pir = RandomForest(pir_path)
        if self.fusion.window != self.pir.window:
            raise ValueError("PIR and fusion models must use the same history window")
        self.fusion_state = StateMachine(self.fusion.thresholds)
        self.pir_state = StateMachine(self.pir.thresholds)

    def predict(self, row):
        pir_score = self.pir.confidence(row)
        fusion_score = self.fusion.confidence(row)
        return {
            "pir_state": self.pir_state.update(pir_score),
            "fusion_state": self.fusion_state.update(fusion_score),
            "pir_confidence": pir_score,
            "fusion_confidence": fusion_score,
        }

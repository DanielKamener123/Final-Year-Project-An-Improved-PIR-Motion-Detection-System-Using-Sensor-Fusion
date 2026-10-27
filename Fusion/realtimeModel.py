"""Real-time model: first-principles Gradient Boosting + three-state alarm machine."""
import time
import numpy as np
from PIR.processing import threshold as PIR_THRESHOLD, getHistoryFeatureNames, getHistoryFeatureVector
from Thermopile.processing import getThermalFeatureNames
from Fusion.gradientBoosting import GradientBoosting

STATES = ("NO INTRUDER", "MAYBE INTRUDER", "INTRUDER")
NO_INTRUDER, MAYBE_INTRUDER, INTRUDER = 0, 1, 2

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
        self.state = NO_INTRUDER
        self.below_since = None

    # The only state machine: used live and by the offline evaluation
    def update(self, score, timestamp=None):
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.state == NO_INTRUDER:
            if score >= self.intruder_enter:
                self.state = INTRUDER
            elif score >= self.maybe_enter:
                self.state = MAYBE_INTRUDER
        elif self.state == MAYBE_INTRUDER:
            if score >= self.intruder_enter:
                self.state = INTRUDER
            elif score < self.maybe_exit:
                self.state = NO_INTRUDER
        else:
            if score >= self.intruder_exit:
                self.below_since = None
            else:
                if self.below_since is None:
                    self.below_since = now
                if now - self.below_since >= self.hold_s:
                    self.state = MAYBE_INTRUDER if score >= self.maybe_exit else NO_INTRUDER
                    self.below_since = None
        return STATES[self.state]

class RealtimeModel:
    def __init__(self, fusion_path):
        # PIR-only is a fixed signal threshold, NOT a trained model.
        self.fusion = GradientBoosting.load(fusion_path)
        self.fusion_state = StateMachine(self.fusion.thresholds)
        self.window = self.fusion.window

    # Fuses One Thermal Frame With The PIR Window Ending At The Same Time
    # pirSamples: [(timestamp, relative voltage), ...] covering the window.
    # Returns (result, None), or (None, reason) when the PIR window is not usable.
    def fuse(self, thermalFeatures, pirSamples, timestamp):
        if len(pirSamples) < 2 or pirSamples[0][0] > timestamp - self.window + 0.15:
            return None, "Insufficient PIR history; waiting for samples"
        if timestamp - pirSamples[-1][0] > 0.3:
            return None, "Stale PIR samples"
        row = dict(zip(getHistoryFeatureNames(), (float(v) for v in getHistoryFeatureVector(pirSamples))))
        row.update({name: float(thermalFeatures[name]) for name in getThermalFeatureNames()})
        return self.predict(row, timestamp, pirVoltage=pirSamples[-1][1]), None

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
            "pir_relative_voltage": voltage,
            "pir_threshold": float(PIR_THRESHOLD),
            "fusion_confidence": fusion_score,
        }

"""Real-time model: first-principles Gradient Boosting + three-state alarm machine."""
import time
from pathlib import Path
import numpy as np
from PIR.processing import threshold as PIR_THRESHOLD, getHistoryFeatureNames, getHistoryFeatureVector
from Thermopile.processing import getThermalFeatureNames
from Fusion.gradientBoosting import GradientBoosting

STATES = ("NO INTRUDER", "MAYBE INTRUDER", "INTRUDER")
NO_INTRUDER, MAYBE_INTRUDER, INTRUDER = 0, 1, 2

# PIR Strong-Signal Rule (11 Nov): a PIR peak-to-peak this large in the fusion window is always a
# person moving, so the confidence is set to 1 (INTRUDER). The largest value in the empty exam-venue
# recordings is 1.33 V (99.9th percentile 0.76 V); fast passes have a median of 2.24 V.
# 2.0 V keeps a 50% margin above anything seen in an empty scene.
USE_PIR_STRONG_RULE = True
PIR_STRONG_V = 2.0


# Applies The PIR Strong-Signal Rule To Confidences (single value or array)
def applyPirStrongRule(scores, pirPeakToPeak):
    if not USE_PIR_STRONG_RULE:
        return scores
    return np.where(np.asarray(pirPeakToPeak, dtype=np.float64) > PIR_STRONG_V, 1.0, scores)

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

# Modes Of The Real-Time Decision (the fused system is always preferred)
MODE_FUSED = "fused"
MODE_PIR_BACKUP = "pir_backup"

class RealtimeModel:
    def __init__(self, fusion_path, backup_path=None):
        # PIR-only is a fixed signal threshold, NOT a trained model.
        self.fusion = GradientBoosting.load(fusion_path)
        self.fusion_state = StateMachine(self.fusion.thresholds)
        self.window = self.fusion.window

        # Backup: Gradient Boosting trained on the PIR features only. It is used
        # ONLY while no thermal frames are received; it is not the thresholded PIR.
        self.backup = None
        self.backup_state = None
        self.backup_problem = None
        if backup_path is None or not Path(backup_path).is_file():
            self.backup_problem = f"No PIR backup model at {backup_path}; run offlineRun.py --train"
        else:
            backup = GradientBoosting.load(backup_path)
            if abs(float(backup.window) - float(self.window)) > 1e-9:
                self.backup_problem = (f"PIR backup window {backup.window:g} s does not match the fused "
                                       f"window {self.window:g} s; retrain both with offlineRun.py --train")
            elif list(backup.features) != getHistoryFeatureNames():
                self.backup_problem = "PIR backup model features are not the PIR features; retrain it"
            else:
                self.backup = backup
                self.backup_state = StateMachine(backup.thresholds)
        self.mode = MODE_FUSED

    @property
    def has_backup(self):
        return self.backup is not None

    # Fuses One Thermal Frame With The PIR Window Ending At The Same Time
    # pirSamples: [(timestamp, relative voltage), ...] covering the window.
    # thermalFeatures=None means no thermal frame is being received: the PIR-only
    # backup classifier is used instead (if one was trained).
    # Returns (result, None), or (None, reason) when the PIR window is not usable.
    def fuse(self, thermalFeatures, pirSamples, timestamp):
        if len(pirSamples) < 2 or pirSamples[0][0] > timestamp - self.window + 0.15:
            return None, "Insufficient PIR history; waiting for samples"
        if timestamp - pirSamples[-1][0] > 0.3:
            return None, "Stale PIR samples"
        if thermalFeatures is None and self.backup is None:
            return None, f"No thermal frames and no PIR backup model ({self.backup_problem})"
        row = dict(zip(getHistoryFeatureNames(), (float(v) for v in getHistoryFeatureVector(pirSamples))))
        if thermalFeatures is not None:
            row.update({name: float(thermalFeatures[name]) for name in getThermalFeatureNames()})
        mode = MODE_FUSED if thermalFeatures is not None else MODE_PIR_BACKUP
        return self.predict(row, timestamp, pirVoltage=pirSamples[-1][1], mode=mode), None

    # Carries The Alarm State Across A Mode Switch So A Switch Never Drops An Alarm
    def _switch_mode(self, mode):
        if mode == self.mode:
            return
        previous = self.fusion_state if self.mode == MODE_FUSED else self.backup_state
        current = self.fusion_state if mode == MODE_FUSED else self.backup_state
        current.state = previous.state
        current.below_since = None
        self.mode = mode

    def predict(self, row, timestamp=None, pirVoltage=None, mode=MODE_FUSED):
        # PIR-only baseline uses the newest relative PIR sample (not a model feature)
        voltage = float(pirVoltage) if pirVoltage is not None else float("nan")
        if not np.isfinite(voltage):
            raise ValueError("Invalid PIR relative voltage")
        pir_active = abs(voltage) >= float(PIR_THRESHOLD)
        if mode == MODE_PIR_BACKUP and self.backup is None:
            raise ValueError("No PIR backup model loaded")
        self._switch_mode(mode)
        if mode == MODE_FUSED:
            fusion_score = self.fusion.confidence(row)
        else:
            fusion_score = self.backup.confidence(row)
        # PIR strong-signal rule (same rule as the demo training)
        fusion_score = float(applyPirStrongRule(fusion_score, row["pirPeakToPeak"]))
        if mode == MODE_FUSED:
            fusion_state = self.fusion_state.update(fusion_score, timestamp)
        else:
            fusion_state = self.backup_state.update(fusion_score, timestamp)
        return {
            "pir_state": "INTRUDER" if pir_active else "NO INTRUDER",
            "fusion_state": fusion_state,
            "pir_relative_voltage": voltage,
            "pir_threshold": float(PIR_THRESHOLD),
            "fusion_confidence": fusion_score,
            "mode": mode,
        }

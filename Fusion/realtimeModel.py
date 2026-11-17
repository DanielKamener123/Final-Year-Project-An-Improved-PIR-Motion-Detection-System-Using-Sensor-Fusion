import time
from pathlib import Path
import numpy as np
from PIR.processing import threshold as PIR_THRESHOLD, getHistoryFeatureNames, getHistoryFeatureVector
from Thermopile.processing import getThermalFeatureNames
from Fusion.gradientBoosting import GradientBoosting

# Names and numeric values of the three alarm states
STATES = ("NO INTRUDER", "MAYBE INTRUDER", "INTRUDER")
NO_INTRUDER, MAYBE_INTRUDER, INTRUDER = 0, 1, 2

# Convert classifier confidence into the three alarm states
class StateMachine:
    def __init__(self, thresholds):
        # Separate entry and exit limits prevent rapid state switching
        self.maybe_enter = float(thresholds["maybe_enter"])
        self.maybe_exit = float(thresholds["maybe_exit"])
        self.intruder_enter = float(thresholds["intruder_enter"])
        self.intruder_exit = float(thresholds["intruder_exit"])

        # Time the score must stay low before leaving INTRUDER
        self.hold_s = float(thresholds.get("hold_s", 0.0))

        # Start with no alarm and no active hold timer
        self.state = NO_INTRUDER
        self.below_since = None

    # Update the alarm using the latest confidence score
    # Same logic for live operation and offline evaluation
    def update(self, score, timestamp=None):
        now = time.monotonic() if timestamp is None else float(timestamp)

        # From NO INTRUDER, enter MAYBE or INTRUDER if confidence rises
        if self.state == NO_INTRUDER:
            if score >= self.intruder_enter:
                self.state = INTRUDER
            elif score >= self.maybe_enter:
                self.state = MAYBE_INTRUDER

        # From MAYBE, increase or clear the warning
        elif self.state == MAYBE_INTRUDER:
            if score >= self.intruder_enter:
                self.state = INTRUDER
            elif score < self.maybe_exit:
                self.state = NO_INTRUDER

        # From INTRUDER, wait for confidence to remain low
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

# Fused model is preferred, PIR model is the backup
MODE_FUSED = "fused"
MODE_PIR_BACKUP = "pir_backup"

# Load models and make live predictions
class RealtimeModel:
    def __init__(self, fusion_path, backup_path=None):
        # Load the trained fused model and its alarm thresholds
        self.fusion = GradientBoosting.load(fusion_path)
        self.fusion_state = StateMachine(self.fusion.thresholds)
        self.window = self.fusion.window

        # Prepare the optional model trained on PIR features only
        # This is different from the basic thresholded PIR comparison
        self.backup = None
        self.backup_state = None
        self.backup_problem = None

        # Check that the backup model exists and matches the fused model
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

        # Begin in fused mode
        self.mode = MODE_FUSED

        # Separate PIR voltage threshold used for comparison on the display
        self.pir_threshold = float(PIR_THRESHOLD)

    # Whether a valid PIR backup model was loaded
    # @property lets you read this function like a variable
    @property
    def has_backup(self):
        return self.backup is not None

    # Combine the latest thermal features with a matching PIR history window
    def fuse(self, thermalFeatures, pirSamples, timestamp):
        # Reject PIR windows with too little history
        if len(pirSamples) < 2 or pirSamples[0][0] > timestamp - self.window + 0.15:
            return None, "Insufficient PIR history; waiting for samples"

        # Reject outdated PIR data
        if timestamp - pirSamples[-1][0] > 0.3:
            return None, "Stale PIR samples"

        # Require a backup model if thermal frames are missing
        if thermalFeatures is None and self.backup is None:
            return None, f"No thermal frames and no PIR backup model ({self.backup_problem})"

        # Extract the PIR history features
        row = dict(zip(getHistoryFeatureNames(), (float(v) for v in getHistoryFeatureVector(pirSamples))))

        # Add thermal features when a frame is available
        if thermalFeatures is not None:
            row.update({name: float(thermalFeatures[name]) for name in getThermalFeatureNames()})

        # Select fused or backup prediction based on thermal availability
        mode = MODE_FUSED if thermalFeatures is not None else MODE_PIR_BACKUP
        return self.predict(row, timestamp, pirVoltage=pirSamples[-1][1], mode=mode), None

    # Preserve the current alarm when switching between the two models
    def _switch_mode(self, mode):
        if mode == self.mode:
            return

        # Copy the previous alarm state into the newly selected model
        previous = self.fusion_state if self.mode == MODE_FUSED else self.backup_state
        current = self.fusion_state if mode == MODE_FUSED else self.backup_state
        current.state = previous.state
        current.below_since = None
        self.mode = mode

    # Produce the PIR baseline and trained model outputs
    def predict(self, row, timestamp=None, pirVoltage=None, mode=MODE_FUSED):
        # Compare the latest PIR voltage against the simple PIR threshold
        voltage = float(pirVoltage) if pirVoltage is not None else float("nan")
        if not np.isfinite(voltage):
            raise ValueError("Invalid PIR relative voltage")
        pir_active = abs(voltage) >= self.pir_threshold

        # Check backup availability before changing modes
        if mode == MODE_PIR_BACKUP and self.backup is None:
            raise ValueError("No PIR backup model loaded")
        self._switch_mode(mode)

        # Calculate human confidence with the selected Gradient Boosting model
        if mode == MODE_FUSED:
            fusion_score = self.fusion.confidence(row)
        else:
            fusion_score = self.backup.confidence(row)

        # Override confidence for a strong PIR signal
        fusion_score = float(applyPirStrongRule(fusion_score, row["pirPeakToPeak"]))

        # Update the correct alarm state machine
        if mode == MODE_FUSED:
            fusion_state = self.fusion_state.update(fusion_score, timestamp)
        else:
            fusion_state = self.backup_state.update(fusion_score, timestamp)

        # Return both outputs and values needed by the live display
        return {
            "pir_state": "INTRUDER" if pir_active else "NO INTRUDER",
            "fusion_state": fusion_state,
            "pir_relative_voltage": voltage,
            "pir_threshold": self.pir_threshold,
            "fusion_confidence": fusion_score,
            "mode": mode,
        }

# Works with one score (live) or an array of scores (offline evaluation)
def applyPirStrongRule(scores, pirPeakToPeak):
    return np.where(np.asarray(pirPeakToPeak, dtype=np.float64) > 2, 1.0, scores)
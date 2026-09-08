import threading
import queue
from dataclasses import dataclass
from enum import Enum

import numpy as np

from PIR.interface import PIRInterfaceConfig, PIRAcquisition


@dataclass
class PIRProcessingConfig:
    # Warm Up Time (seconds)
    warmupTime = 10.0

    # How Quickly System Adapts To Background Changes While System Is Idle 
    alpha = 0.01

    # Threshold Relative To Background
    threshold = 0.2

    # Checks A Certain Time To See If Still Movement - Occurs When Inside Threshold
    settleDownTime = 1.0


# Determines Initial Signal Direction
class Direction(Enum):
    UNKNOWN = 0
    POSITIVE_FIRST = 1
    NEGATIVE_FIRST = 2

# Movement State
class MovementState(Enum):
    IDLE = 0
    MOVING = 1
    SETTLING = 2


class MovementSegmenter:
    def __init__(self, config):
        self.config = config

        # Baseline Voltage
        self.baselineVoltage = 0.0

        # Warm Up
        self.initialised = False
        self.warmupStartTime = None
        self.warmupCount = 0

        # Movement State
        self.state = MovementState.IDLE
        self.direction = Direction.UNKNOWN

        # Movement Variables
        self.settleStartTime = None
        self.movementSamples = []
        self.lastActiveTime = None
        self.movementBaseline = 0.0
        self.lastTime = None
        self.lastMovement = None

    # Learns Background Temperature As Temperature Changes
    def updateBackground(self, v):
        if not self.initialised:
            self.baselineVoltage = v
            self.initialised = True
        else:
            a = self.config.alpha
            self.baselineVoltage = (1 - a) * self.baselineVoltage + a * v

    # Warm Up
    def warmup(self, ts, v):
        # Check If Warm Up Has Started
        if self.warmupStartTime is None:
            self.warmupStartTime = ts
        self.warmupCount += 1

        # Add New Reading To Mean
        self.baselineVoltage += (v - self.baselineVoltage) / self.warmupCount

        # Check If Warm Up Is Done
        if ts - self.warmupStartTime >= self.config.warmupTime:
            self.initialised = True

    def processSample(self, ts, v):
        # Validation - TODO: Can Remove
        ts, v = float(ts), float(v)
        if not np.isfinite(ts) or not np.isfinite(v):
            raise ValueError("PIR timestamp and voltage must be finite")
        if self.lastTime is not None and ts <= self.lastTime:
            raise ValueError("PIR timestamps must be strictly increasing")
        self.lastTime = ts

        # Make Sure System Is Warmed Up
        if not self.initialised:
            self.warmup(ts, v)
            return None

        # Calculate Threshold Values
        thresholdHigh = self.baselineVoltage + self.config.threshold
        thresholdLow = self.baselineVoltage - self.config.threshold

        # Check If Current Movement State Is Idle
        if self.state == MovementState.IDLE:
            # Check If Sample Is Out Of Threshold Bounds
            if v >= thresholdHigh or v <= thresholdLow:
                # Determine Initial Direction
                self.direction = (Direction.POSITIVE_FIRST if v >= thresholdHigh else Direction.NEGATIVE_FIRST)

                # Update Movement State
                self.state = MovementState.MOVING

                # Set Movement Baseline
                self.movementBaseline = self.baselineVoltage

                # Get Timestamp And Difference Between Baseline Voltage And Samples Voltage
                self.movementSamples = [(ts, v - self.movementBaseline)]
                self.lastActiveTime = ts
                self.settleStartTime = None
            else:
                self.updateBackground(v)
            return None

        # Keep The Movement Baseline Thresholds Fixed Throughout Movement Event
        thresholdHigh = self.movementBaseline + self.config.threshold
        thresholdLow = self.movementBaseline - self.config.threshold

        # Add New Sample To Movement Samples
        self.movementSamples.append((ts, v - self.movementBaseline))

        # Check If Current Sample Is In Threshold Bounds
        inside = thresholdLow < v < thresholdHigh
        if not inside:
            # Store New Sample And Continue
            self.lastActiveTime = ts
            self.settleStartTime = None
            self.state = MovementState.MOVING
            return None

        # If Object Is In Movement State, Sample Is Withing Bounds So Must Start Settling Time
        if self.state == MovementState.MOVING:
            self.state = MovementState.SETTLING
            self.settleStartTime = ts
            return None

        # Checks If Settling Down Timer Is Done
        if ts - self.settleStartTime >= self.config.settleDownTime:
            # Get The Completed Movement Samples - Not Including Settling Down Samples
            completed = [(t, x) for t, x in self.movementSamples if t <= self.lastActiveTime]
            direction = self.direction
            self.lastMovement = {
                "start_ts": completed[0][0],
                "end_ts": self.lastActiveTime,
                "completion_ts": ts,
                "baseline_v": self.movementBaseline,
                "direction": direction,
            }

            # Reset Movement State And Variables
            self.state = MovementState.IDLE
            self.direction = Direction.UNKNOWN
            self.movementSamples = []
            self.settleStartTime = None
            self.lastActiveTime = None

            # Return Movement Samples
            return completed, direction, thresholdHigh, thresholdLow
        return None


#  Find Local Extreme - TODO: Understand
def _find_local_extrema(samples):
    """Return (index, ts, v, is_max) for local extrema.

    A numerical peak detector used to implement the peak-selection stage.
    Its exact comparison rules are implementation details, not a claimed
    reproduction of the paper's unpublished sample-level algorithm.
    """
    extrema = []
    vals = [v for _, v in samples]
    for i in range(1, len(vals) - 1):
        if vals[i - 1] < vals[i] >= vals[i + 1]:
            extrema.append((i, samples[i][0], samples[i][1], True))
        elif vals[i - 1] > vals[i] <= vals[i + 1]:
            extrema.append((i, samples[i][0], samples[i][1], False))
    return extrema

# Returns Positive and Negative Global Voltage Peak
def globalPeaks(samples):
    if not samples:
        return []
    high = max(samples, key=lambda sample: sample[1])
    low = min(samples, key=lambda sample: sample[1])
    return sorted([high, low], key=lambda sample: sample[0])

def getFeatureVector(samples, direction: Direction = Direction.UNKNOWN, config: PIRProcessingConfig = None):
    # Error Checking
    if not samples:
        raise ValueError("Cannot extract features from an empty passage")
    if config is None:
        config = PIRProcessingConfig()
    if not config.enable_peak_selection:
        raise ValueError("Peak selection is required for the core feature vector")

    # Gets Feature Class
    peaks = globalPeaks(samples, direction)
    Vpp = max(v for _, v in peaks) - min(v for _, v in peaks)
    D = samples[-1][0] - samples[0][0]
    return np.array([Vpp, D], dtype=np.float64)

# Processing Pipeline
class PIRProcessor:
    def __init__(self, config: PIRProcessingConfig, sampleQueue: queue.Queue):
        self.config = config

        self.sampleQueue = sampleQueue
        self.featureQueue = queue.Queue()
        self.movementQueue = queue.Queue()

        self.segmenter = MovementSegmenter(config)

        self.workerThread = threading.Thread(target=self.run, daemon=True)
        self.stopEvent = threading.Event()

        self.lastError = None

    def start(self):
        self.workerThread.start()

    def stop(self):
        self.stopEvent.set()

    def run(self):
        while not self.stopEvent.is_set():
            try:
                ts, v = self.sampleQueue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.segmenter.processSample(ts, v)
                if result is not None:
                    samples, direction, Ap, An = result
                    vec = getFeatureVector(samples, direction, self.config)
                    metadata = dict(self.segmenter.lastMovement)
                    metadata.update({"features": vec.copy(),
                                     "samples": list(samples),
                                     "thresholds_v": (An, Ap)})
                    self.movementQueue.put(metadata)
                    self.featureQueue.put(vec)
            except (ValueError, TypeError) as exc:
                self.lastError = exc
                print(f"PIR processing error: {exc}")
            finally:
                self.sampleQueue.task_done()


# Debug
if __name__ == "__main__":
    sample_queue = queue.Queue(maxsize=2048)

    interface_cfg = PIRInterfaceConfig()
    acquisition = PIRAcquisition(interface_cfg, sample_queue)

    processing_cfg = PIRProcessingConfig()
    processor = PIRProcessor(processing_cfg, sample_queue)

    print("PIR pipeline: [Vpp, D]")
    print("Keep the scene empty during background calibration.")
    print("Threshold and settle-time defaults are not calibrated for this PCB.")
    acquisition.start()
    processor.start()

    try:
        while True:
            vec = processor.featureQueue.get()
            np.set_printoptions(precision=4, suppress=True)
            print(vec)
    except KeyboardInterrupt:
        processor.stop()
        acquisition.stop()
        print("\nStopped.")

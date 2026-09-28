import threading
import queue
from collections import deque

import numpy as np

# Warm Up Time (seconds)
warmupTime = 10.0

# How Quickly System Adapts To Background Changes While System Is Idle 
alpha = 0.01

# Threshold Relative To Background
threshold = 0.2

# Default Rolling Window Time Used For Live Feature Extraction
defaultWindowTime = 1.0

# PIR History Is Kept For Twice The Active Feature Window
historyWindowMultiplier = 2.0


def getFeatureVector(samples):
    """Return the compact PIR temporal feature vector for one rolling window.

    Features:
      1. peak_to_peak_v: max(signal) - min(signal)
      2. activity_duration_s: time from the first to last sample whose
         magnitude exceeds the activity threshold
      3. peak_separation_s: absolute time between the dominant positive
         and negative peaks, only when both lobes exceed the threshold
      4. mean_signal_energy: mean squared relative PIR signal in the window

    A window with no meaningful PIR activity returns zero for the original
    activity features, while mean signal energy still describes the signal
    present in the selected history window.
    """
    if not samples:
        return np.zeros(4, dtype=np.float64)

    timestamps = np.asarray([ts for ts, _ in samples], dtype=np.float64)
    values = np.asarray([value for _, value in samples], dtype=np.float64)

    mean_signal_energy = float(np.mean(values ** 2))

    active = np.abs(values) >= float(threshold)
    if not np.any(active):
        return np.array([0.0, 0.0, 0.0, mean_signal_energy], dtype=np.float64)

    peak_to_peak = float(np.max(values) - np.min(values))

    active_indices = np.flatnonzero(active)
    activity_duration = float(timestamps[active_indices[-1]] - timestamps[active_indices[0]])

    positive_index = int(np.argmax(values))
    negative_index = int(np.argmin(values))
    if values[positive_index] >= threshold and values[negative_index] <= -threshold:
        peak_separation = float(abs(timestamps[positive_index] - timestamps[negative_index]))
    else:
        peak_separation = 0.0

    return np.array(
        [peak_to_peak, activity_duration, peak_separation, mean_signal_energy],
        dtype=np.float64,
    )


def getFeatureNames():
    return [
        "peak_to_peak_v",
        "activity_duration_s",
        "peak_separation_s",
        "mean_signal_energy",
    ]


def getCurrentFeatureVector(relativeVoltage):
    if relativeVoltage is None or not np.isfinite(relativeVoltage):
        return np.zeros(2, dtype=np.float64)

    relativeVoltage = float(relativeVoltage)
    pir_active = int(abs(relativeVoltage) >= float(threshold))

    if not pir_active:
        return np.zeros(2, dtype=np.float64)

    return np.array(
        [pir_active, relativeVoltage],
        dtype=np.float64,
    )


def getCurrentFeatureNames():
    return [
        "active",
        "current_relative_voltage",
    ]


class PIRProcessor:
    def __init__(self, sampleQueue: queue.Queue | None = None, initialBaseline=None, windowTime=defaultWindowTime):
        if windowTime <= 0:
            raise ValueError("windowTime must be greater than zero")

        self.sampleQueue = sampleQueue
        self.outputQueue = queue.Queue(maxsize=4096)
        self.windowTime = float(windowTime)
        self.maxHistoryTime = historyWindowMultiplier * self.windowTime

        # Baseline Voltage
        self.baselineVoltage = 0.0 if initialBaseline is None else float(initialBaseline)

        # Warm Up
        self.initialised = initialBaseline is not None
        self.warmupStartTime = None
        self.warmupCount = 0

        self.lastTime = None
        self.lastError = None

        self.history = deque()
        self.historyLock = threading.Lock()
        self.stateLock = threading.Lock()

        self.workerThread = None
        self.stopEvent = threading.Event()

        if self.sampleQueue is not None:
            self.workerThread = threading.Thread(target=self.run, daemon=True)

    # Learns Background Temperature As Temperature Changes
    def updateBackground(self, v):
        with self.stateLock:
            self.baselineVoltage = (1 - alpha) * self.baselineVoltage + alpha * v

    # Warm Up
    def warmup(self, ts, v):
        # Check If Warm Up Has Started
        if self.warmupStartTime is None:
            self.warmupStartTime = ts
        self.warmupCount += 1

        # Add New Reading To Mean
        self.baselineVoltage += (v - self.baselineVoltage) / self.warmupCount

        # Check If Warm Up Is Done
        if ts - self.warmupStartTime >= warmupTime:
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

        with self.stateLock:
            baseline = self.baselineVoltage

        relative = v - baseline

        with self.historyLock:
            self.history.append((ts, relative))
            cutoff = ts - self.maxHistoryTime
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()

        if abs(relative) < threshold:
            self.updateBackground(v)

        return relative

    def start(self):
        if self.workerThread is None:
            raise RuntimeError("PIRProcessor needs a sampleQueue before start() can be used")
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
                relative = self.processSample(ts, v)
                if relative is None:
                    relative = float(v) - self.getBaseline()
                try:
                    self.outputQueue.put_nowait((ts, float(v), float(relative)))
                except queue.Full:
                    try:
                        self.outputQueue.get_nowait()
                    except queue.Empty:
                        pass
                    self.outputQueue.put_nowait((ts, float(v), float(relative)))
            except (ValueError, TypeError) as exc:
                self.lastError = exc
                print(f"PIR processing error: {exc}")
            finally:
                self.sampleQueue.task_done()

    def getWindowSamples(self, windowTime=defaultWindowTime, endTime=None):
        if windowTime <= 0:
            raise ValueError("windowTime must be greater than zero")

        with self.historyLock:
            samples = list(self.history)

        if not samples:
            return []

        if endTime is None:
            endTime = samples[-1][0]
        startTime = endTime - float(windowTime)
        return [(ts, value) for ts, value in samples if startTime < ts <= endTime]

    def snapshot(self, windowTime=defaultWindowTime, endTime=None):
        samples = self.getWindowSamples(windowTime=windowTime, endTime=endTime)
        return getFeatureVector(samples).tolist()

    def getBaseline(self):
        with self.stateLock:
            return float(self.baselineVoltage)


# Debug
if __name__ == "__main__":
    from PIR.interface import PIRAcquisition, sampleRateHz

    sample_queue = queue.Queue(maxsize=2048)

    acquisition = PIRAcquisition(sample_queue)

    processor = PIRProcessor(sample_queue)

    print("PIR pipeline: rolling buffer with peak-to-peak amplitude, activity duration, peak separation, and mean signal energy")
    print("Keep the scene empty during background calibration.")
    print("Threshold default is not calibrated for this PCB.")
    acquisition.start()
    processor.start()

    try:
        while True:
            ts, v, relative = processor.outputQueue.get()
            if processor.initialised and int(ts * 2) != int((ts - 1.0 / sampleRateHz) * 2):
                np.set_printoptions(precision=4, suppress=True)
                print(processor.snapshot(defaultWindowTime))
    except KeyboardInterrupt:
        processor.stop()
        acquisition.stop()
        print("\nStopped.")

import threading
import queue
from collections import deque

import numpy as np

# Warm Up Time
warmupTime = 10

threshold = 0.2
rollingWindow = 1.5
historyBuffer = 10
rawHistoryBuffer = 20

# Get The Context Features From The Last PIR Samples In History Buffer
def getHistoryFeatureVector(samples):
    # Check For No Samples
    if not samples:
        return np.zeros(7)

    # Gets Timestamps and Values from Samples In Seperate Arrays
    timestamps = np.asarray([ts for ts, _ in samples])
    values = np.asarray([value for _, value in samples])

    # Calculate Peak To Peak (Feature 1)
    positivePeak = float(np.max(values))
    negativePeak = float(np.min(values))
    peakToPeak = float(positivePeak - negativePeak)

    # Calculate RMS
    rms = float(np.sqrt(np.mean(values ** 2)))

    # Total time active in the history window (rather than first-to-last
    # threshold crossing). Estimate seconds per sample from PIR timestamps.
    # This works with the same live and offline feature extraction function.
    active = np.abs(values) >= float(threshold)
    intervals = np.diff(timestamps)
    valid_intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    sample_period = float(np.median(valid_intervals)) if valid_intervals.size else 0.0
    activityDuration = float(np.count_nonzero(active) * sample_period)

    # Calculate Peak Separation (Feature 3)
    maxPeakIndex = int(np.argmax(values))
    minPeakIndex = int(np.argmin(values))
    peak_separation = float(abs(timestamps[maxPeakIndex] - timestamps[minPeakIndex]))

    return np.array([
        positivePeak,
        negativePeak,
        peakToPeak,
        rms,
        activityDuration,
        peak_separation,
    ])


def getHistoryFeatureNames():
    return [
        "positivePeak",
        "negativePeak",
        "peakToPeak",
        "rms",
        "activityDuration",
        "peakSeparation",
    ]

# Get Current Features From Current Sample
def getCurrentFeatureVector(relativeVoltage):
    if relativeVoltage is None or not np.isfinite(relativeVoltage):
        return np.zeros(2, dtype=np.float64)

    relativeVoltage = float(relativeVoltage)
    pir_active = int(abs(relativeVoltage) >= float(threshold))

    return np.array(
        [pir_active, relativeVoltage],
        dtype=np.float64,
    )


def getCurrentFeatureNames():
    return [
        "active",
        "currentRelativeVoltage",
    ]


class PIRProcessor:
    def __init__(self, sampleQueue: queue.Queue | None = None, initialBaseline=None, windowTime=rollingWindow):
        if windowTime <= 0:
            raise ValueError("windowTime must be greater than zero")

        self.sampleQueue = sampleQueue
        self.outputQueue = queue.Queue(maxsize=4096)
        self.windowTime = float(windowTime)
        self.maxHistoryTime = historyBuffer

        # Baseline Voltage
        self.baselineVoltage = 0.0 if initialBaseline is None else float(initialBaseline)

        # Warm Up
        self.initialised = initialBaseline is not None
        self.warmupStartTime = None
        self.warmupCount = 0

        self.lastTime = None
        self.lastError = None

        self.history = deque()
        self.rawHistory = deque()
        self.historyLock = threading.Lock()
        self.stateLock = threading.Lock()

        self.workerThread = None
        self.stopEvent = threading.Event()

        if self.sampleQueue is not None:
            self.workerThread = threading.Thread(target=self.run, daemon=True)

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

    def calibrate(self, windowTime=warmupTime):
        with self.historyLock:
            rawSamples = list(self.rawHistory)

        if not rawSamples:
            raise RuntimeError("No PIR samples are available for calibration")

        endTime = rawSamples[-1][0]
        startTime = endTime - float(windowTime)
        calibrationValues = [value for ts, value in rawSamples if ts >= startTime]

        if not calibrationValues:
            raise RuntimeError("No PIR samples are available in the calibration window")

        with self.stateLock:
            self.baselineVoltage = float(np.median(calibrationValues))
            self.initialised = True

        with self.historyLock:
            self.history.clear()

    def processSample(self, ts, v):
        # Validation - TODO: Can Remove
        ts, v = float(ts), float(v)
        if not np.isfinite(ts) or not np.isfinite(v):
            raise ValueError("PIR timestamp and voltage must be finite")
        if self.lastTime is not None and ts <= self.lastTime:
            raise ValueError("PIR timestamps must be strictly increasing")
        self.lastTime = ts

        with self.historyLock:
            self.rawHistory.append((ts, v))
            rawCutoff = ts - rawHistoryBuffer
            while self.rawHistory and self.rawHistory[0][0] < rawCutoff:
                self.rawHistory.popleft()

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

    def getWindowSamples(self, windowTime=rollingWindow, endTime=None):
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

    def snapshot(self, windowTime=rollingWindow, endTime=None):
        samples = self.getWindowSamples(windowTime=windowTime, endTime=endTime)
        return getHistoryFeatureVector(samples).tolist()

    def getBaseline(self):
        with self.stateLock:
            return float(self.baselineVoltage)


# Debug
if __name__ == "__main__":
    from PIR.interface import PIRAcquisition, sampleRateHz

    sample_queue = queue.Queue(maxsize=2048)

    acquisition = PIRAcquisition(sample_queue)

    processor = PIRProcessor(sample_queue)

    print("Keep the scene empty during background calibration.")
    print("Threshold default is not calibrated for this PCB.")
    acquisition.start()
    processor.start()

    try:
        while True:
            ts, v, relative = processor.outputQueue.get()
            if processor.initialised and int(ts * 2) != int((ts - 1.0 / sampleRateHz) * 2):
                np.set_printoptions(precision=4, suppress=True)
                print(processor.snapshot(rollingWindow))
    except KeyboardInterrupt:
        processor.stop()
        acquisition.stop()
        print("\nStopped.")


# Offline PIR signal preprocessing; uses the same PIRProcessor as live sampling
def preprocessPIR(pirData, initialBaseline, windowSeconds):
    processor = PIRProcessor(initialBaseline=initialBaseline, windowTime=windowSeconds)
    timestamps = pirData["time_s"].to_numpy(dtype=np.float64)
    voltages = pirData["voltage"].to_numpy(dtype=np.float64)
    relative = np.empty_like(voltages)

    for index, (timestamp, voltage) in enumerate(zip(timestamps, voltages)):
        relative[index] = processor.processSample(timestamp, voltage)

    return timestamps, relative



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

# Maximum PIR History Kept In Memory
maxHistoryTime = 30.0

# Haar-transform based PIR feature extraction (8 frequency-band
# energies), following the approach used for PIR-based clutter
# rejection against wind-blown vegetation: a fixed-length block of the
# signal is passed through a Haar transform and the transformed
# coefficients are grouped into a small number of frequency bins, whose
# energies form the feature vector.
#
# That paper analyses a 128-sample block and reports 8 frequency bins,
# but does not publish the exact coefficient-to-bin grouping. This
# implementation groups coefficients by dyadic scale using a 3-level
# full Haar wavelet-packet decomposition (2**3 = 8 equal-width leaves),
# which is the natural way to get 8 bins out of a Haar transform.
# Applying a fixed 128-sample block to our variable-length completed
# movement passages (rather than the paper's continuously-updating
# sliding window) is an engineering adaptation, not a reproduction of
# the paper -- flag this in any report text.
# The current fusion path passes rolling-window samples into this same
# feature extractor so the feature set can be changed later in one place.

BLOCK_LENGTH = 128
N_LEVELS = 3
N_BINS = 2 ** N_LEVELS  # 8


def _haarStep(signal):
    """One level of the Haar transform: pairwise sum/difference.

    Splits `signal` (even length) into an approximation half (local
    averages) and a detail half (local differences), each half the
    length of the input.
    """
    evens = signal[0::2]
    odds = signal[1::2]
    approx = (evens + odds) / np.sqrt(2.0)
    detail = (evens - odds) / np.sqrt(2.0)
    return approx, detail


def _fullTreeEnergies(signal, levels):
    """Recursively splits `signal` into 2**levels leaves via repeated
    Haar steps (a full wavelet-packet tree, not just the approximation
    branch), returning the energy (sum of squares) of each leaf.
    """
    if levels == 0:
        return [float(np.sum(signal ** 2))]
    approx, detail = _haarStep(signal)
    return _fullTreeEnergies(approx, levels - 1) + _fullTreeEnergies(detail, levels - 1)


def _prepareBlock(samples, blockLength=BLOCK_LENGTH):
    """Extracts the voltage values from a completed passage and pads or
    truncates them to a fixed `blockLength` so the wavelet-packet tree
    always splits evenly into N_BINS leaves.
    """
    voltages = np.array([v for _, v in samples], dtype=np.float64)
    if voltages.size >= blockLength:
        return voltages[-blockLength:]
    padded = np.zeros(blockLength, dtype=np.float64)
    padded[: voltages.size] = voltages
    return padded


def getHaarFeatureVector(samples, blockLength=BLOCK_LENGTH, levels=N_LEVELS):
    """Returns the 8-element Haar frequency-bin energy vector for a
    completed PIR movement passage.
    """
    block = _prepareBlock(samples, blockLength)
    energies = _fullTreeEnergies(block, levels)
    return np.array(energies, dtype=np.float64)


# Haar-transform feature vector: the passage is padded/truncated to a
# fixed-length block and decomposed into 8 frequency-band energies.
# Replaces the previous single peak-to-peak/duration pair. The transform
# itself is implemented below in this processing file.
def getFeatureVector(samples):
    # Error Checking
    if not samples:
        return np.zeros(N_BINS, dtype=np.float64)

    return getHaarFeatureVector(samples)


def getFeatureNames():
    return [f"haar_energy_{i}" for i in range(N_BINS)]


class PIRProcessor:
    def __init__(self, sampleQueue: queue.Queue | None = None, initialBaseline=None):
        self.sampleQueue = sampleQueue
        self.outputQueue = queue.Queue(maxsize=4096)

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
            cutoff = ts - maxHistoryTime
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

    print("PIR pipeline: rolling buffer with Haar transform, 8 frequency-band energies")
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

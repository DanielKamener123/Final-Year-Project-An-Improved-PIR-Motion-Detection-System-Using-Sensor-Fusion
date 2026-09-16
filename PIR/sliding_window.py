import queue
import threading
from collections import deque

import numpy as np

from haar_features import getHaarFeatureVector, BLOCK_LENGTH, N_BINS
from processing import MovementState


# Continuously-updated PIR feature source.
#
# Unlike PIRProcessor's featureQueue (which only produces a vector once
# a movement passage finishes), this keeps a rolling buffer of the last
# BLOCK_LENGTH raw samples and recomputes the Haar feature vector fresh
# on every snapshot() call -- so the vector sent at each thermopile tick
# always reflects the PIR signal *as it currently is*, not something
# held over from an earlier, possibly long-finished, passage.
#
# During a quiet period the buffer is just baseline noise, which the
# Haar transform in PIR/haar_features.py naturally reports as near-zero
# energy across all bins -- no special "idle" case needed here.
#
# BLOCK_LENGTH (128 samples, ~5-6 s at the PIR's ~20-25 Hz rate) is
# reused as-is from PIR/haar_features.py rather than picking a new
# number, since that value is already the one with literature backing.
# It is a tunable knob, not a fixed constant, if a shorter/longer window
# turns out to work better once there is real labelled data to test
# against.
class PIRSlidingWindow:
    def __init__(self, sample_queue: queue.Queue, segmenter):
        self._sample_queue = sample_queue
        self._segmenter = segmenter
        self._buffer = deque(maxlen=BLOCK_LENGTH)
        self._lock = threading.Lock()
        self._stopEvent = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopEvent.set()

    def _currentBaseline(self):
        # Mirrors MovementSegmenter.processSample: while a movement is
        # in progress the threshold reference is frozen at
        # movementBaseline; otherwise baselineVoltage is the live,
        # continuously-updated background estimate.
        if self._segmenter.state == MovementState.IDLE:
            return self._segmenter.baselineVoltage
        return self._segmenter.movementBaseline

    def _run(self):
        while not self._stopEvent.is_set():
            try:
                ts, v = self._sample_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            relative = v - self._currentBaseline()
            with self._lock:
                self._buffer.append((ts, relative))

    def snapshot(self):
        with self._lock:
            samples = list(self._buffer)
        if not samples:
            return np.zeros(N_BINS, dtype=np.float64).tolist()
        return getHaarFeatureVector(samples).tolist()

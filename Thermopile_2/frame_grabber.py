"""
Runs interface.getFrame() on its own dedicated thread, continuously,
independent of whatever the rest of the program is doing (pipeline
processing, matplotlib rendering, etc). The main thread just reads
whatever the latest cached frame is -- it never blocks on the sensor.

This does NOT fix a raw USB/host latency problem. It only helps if
retries were being caused by the read loop being delayed by other
CPU-bound work running on the same thread.
"""

from __future__ import annotations

import threading
import time

from interface import getFrame as getRawFrame


class FrameGrabber:
    def __init__(self):
        self._lock = threading.Lock()
        self._latestFrame = None
        self._latestFrameTime = None
        self._running = False
        self._thread = None
        self._readErrors = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._readLoop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _readLoop(self):
        while self._running:
            try:
                frame = getRawFrame()
            except RuntimeError:
                # Sensor missed its read window -- count it, skip this
                # cycle, and keep going rather than crashing the thread.
                self._readErrors += 1
                continue

            with self._lock:
                self._latestFrame = frame
                self._latestFrameTime = time.time()

    def getFrame(self, maxAgeS: float = 1.0):
        """
        Return the most recently captured frame. Raises RuntimeError if
        no frame has ever arrived, or if the cached frame is older than
        maxAgeS (a sign the grabber thread has stalled).
        """
        with self._lock:
            frame = self._latestFrame
            frameTime = self._latestFrameTime

        if frame is None:
            raise RuntimeError("No frame captured yet -- grabber thread "
                                "hasn't produced one.")
        age = time.time() - frameTime
        if age > maxAgeS:
            raise RuntimeError(f"Latest frame is {age:.2f}s old -- "
                                f"grabber thread may have stalled.")
        return frame

    @property
    def readErrors(self):
        return self._readErrors
import numpy as np

from interface import getFrame as getRawFrame
from thermopile_pipeline import DetectionResult, PipelineConfig, ThermopilePipeline

WIDTH = 32
HEIGHT = 24
FRAME_SHAPE = (HEIGHT, WIDTH)

N_WARMUP_FRAMES = 20

Config = PipelineConfig()
Pipeline = ThermopilePipeline(frame_shape=FRAME_SHAPE, config=Config)

# Cached from the most recent getFrame() call, so getDetections()
# doesn't have to reprocess the frame or block on the sensor again.
LatestDetections: list[DetectionResult] = []


# Convert a 1D array into a 2D array.
def convertFlatArray(flatFrame):
    return np.array(flatFrame, dtype=np.float64).reshape(FRAME_SHAPE)


# Captures frames to use as initialised background model
def warmUp():
    print(f"Warm up Starting - Make Sure No One Is In Frame")
    calibrationFrames = np.empty((N_WARMUP_FRAMES, *FRAME_SHAPE), dtype=np.float64)
    for i in range(N_WARMUP_FRAMES):
        calibrationFrames[i] = convertFlatArray(getRawFrame())
    Pipeline.warm_start(calibrationFrames)
    print("Warm Up Complete")


warmUp()


def getFrame():
    """
    Return the current raw frame as a flat list of 768 temperatures,
    unchanged -- same return contract as before, so display.py works
    without modification.

    Also runs that same frame through the detection pipeline and
    caches the result for getDetections(). Call this once per loop
    iteration; getDetections() reads the cache from that call.
    """
    global _latestDetections

    rawFrame = getRawFrame()
    frameArray = convertFlatArray(rawFrame)
    LatestDetections = Pipeline.process_frame(frameArray)

    return rawFrame


def getDetections():
    """
    Return the list of DetectionResult objects (blob, features,
    confidence) from the most recently processed frame. Reflects
    whatever frame getFrame() last returned -- call getFrame() first
    each loop iteration.
    """
    return LatestDetections
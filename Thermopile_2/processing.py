import numpy as np

from interface import getFrame as getRawFrame
from thermopile_pipeline import ThermopilePipeline

from frame_grabber import FrameGrabber

_grabber = FrameGrabber()
_grabber.start()

WIDTH = 32
HEIGHT = 24
FRAME_SHAPE = (HEIGHT, WIDTH)

N_WARMUP_FRAMES = 20

Pipeline = ThermopilePipeline(frame_shape=FRAME_SHAPE)

LatestDetections = []

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

# Process Frame
def getFrame():
    global LatestDetections

    rawFrame = _grabber.getFrame()
    frameArray = convertFlatArray(rawFrame)
    LatestDetections = Pipeline.process_frame(frameArray)

    return rawFrame

# Return the list of DetectionResult from the most recently frame
def getDetections():
    """
    Return the list of DetectionResult objects (blob, features,
    confidence) from the most recently processed frame. Reflects
    whatever frame getFrame() last returned -- call getFrame() first
    each loop iteration.
    """
    return LatestDetections
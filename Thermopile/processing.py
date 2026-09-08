import numpy as np

from Thermopile.interface import getFrame as getRawFrame
from Thermopile.thermopile_pipeline import ThermopilePipeline

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

    rawFrame = getRawFrame()
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

# Debugging

# Current background array
def getBackground():
    return Pipeline.bg_model.bg

# Current per pixel sigma
def getSigma():
    return Pipeline.bg_model.sigma

# Pre Filtering Z Value
def getRawZ():
    return Pipeline.last_raw_z

# Filtered Z Value
def getSmoothedZ():
    return Pipeline.last_smoothed_z

# Binary Mask
def getMask():
    return Pipeline._prev_mask

# Threshold Value Used On The Most Recent Frame
def getThreshold():
    return Pipeline.last_threshold

# Mean Background Temperature
def getSceneTemp():
    return Pipeline.last_scene_temp
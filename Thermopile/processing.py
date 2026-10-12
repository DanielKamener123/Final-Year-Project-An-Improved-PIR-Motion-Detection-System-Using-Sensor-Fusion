from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Config
ALPHA = 0.990                 
SIGMA_MIN = 0.05 # Minimum Sigma

# Spatial filter
gaussianKernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]) / 16.0
filterMode = "reflect"

# Threshold
threshold = 2.5
thresholdBackgroundProtection = 1.5

# Blob extraction
connectivity_8 = True
min_blob_pixels = 1 # Blobs smaller than this are discarded

# Feature scoring
max_match_distance = 9
max_missed_frames = 1
min_primary_track_frames = 2
min_displacement_step_px = 1  # Ignore small centroid jitter

# Background Model Class
class BackgroundModel:
    def __init__(self, shape):
        self.bg = np.zeros(shape)
        self.sigma = np.full(shape, SIGMA_MIN)
        self._initialised = False

    # Calibrate Background
    def warm_start(self, calibration_frames):
        self.bg[:] = np.median(calibration_frames, axis=0)
        measured_sigma = calibration_frames.std(axis=0)
        self.sigma = np.maximum(measured_sigma, SIGMA_MIN)
        self._initialised = True

    # Update Background
    def update(self, frame, foreground_mask):
        #Inverts Foreground Mask To Get Background Mask
        update_mask = ~foreground_mask

        # Updates Background Where No Humans
        new_bg = ALPHA * self.bg + (1 - ALPHA) * frame
        self.bg = np.where(update_mask, new_bg, self.bg)

        # Update Sigma Where No Human
        new_sigma_sq = ALPHA * (self.sigma ** 2) + (1 - ALPHA) * (self.bg - frame) ** 2
        new_sigma = np.sqrt(new_sigma_sq)
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)
        self.sigma = np.where(update_mask, new_sigma, self.sigma)

    # Calculate Z value - How many multiples of this pixels normal variation is present
    def z_score(self, frame):
        return np.abs(self.bg - frame) / self.sigma

    # Calculate Mean Background Temp For Adaptive Thresholding
    def scene_temperature(self):
        return float(np.mean(self.bg))

# Spatial filtering
def gaussian_filter(z_grid):
    return ndimage.convolve(z_grid, gaussianKernel, mode=filterMode)

# Apply Threshold To Get Mask
def apply_threshold(smoothed_z, threshold_value=threshold):
    return smoothed_z > threshold_value

# Blob extraction
@dataclass
class Blob:
    id: int
    pixel_count: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]   # (min_row, min_col, max_row, max_col)
    mask: np.ndarray                  

# Extract Blobs
def extract_blobs(binary_mask):
    if connectivity_8:
        structure = np.ones((3, 3))
    else:
        structure = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

    labelled_grid, num_blobs = ndimage.label(binary_mask, structure=structure)

    blobs: list[Blob] = []
    for blob_id in range(1, num_blobs + 1):
        blob_mask = labelled_grid == blob_id
        pixel_count = int(blob_mask.sum())
        if pixel_count < min_blob_pixels:
            continue

        centroid = ndimage.center_of_mass(binary_mask, labelled_grid, blob_id)
        rows, cols = np.where(blob_mask)
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

        blobs.append(Blob(
            id=blob_id,
            pixel_count=pixel_count,
            centroid=centroid,
            bbox=bbox,
            mask=blob_mask,
        ))

    return blobs

@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    age_frames: int
    missed_frames: int = 0
    cumulative_displacement_px: float = 0.0

# Blob Feature Scoring
@dataclass
class BlobFeatures:
    track_id: int
    centroid: tuple[float, float]

    pixel_count: int
    track_age_frames: int

    mean_temperature_difference: float
    max_temperature_difference: float

# Checks Blobs Displacement From Previous Frame
def score_blob(blob, track, temperature_difference):
    # If There Is No Previous Blob, Set Track Age To 1
    if track is None:
        track_age_frames = 1
        track_id = -1
    else:
        track_age_frames = track.age_frames + 1
        track_id = track.id

    # Get Temperature Difference For Blob
    blob_temperature_difference = temperature_difference[blob.mask]

    # Get Mean and Max Temperature Difference For Blob
    mean_temperature_difference = float(np.mean(blob_temperature_difference))
    max_temperature_difference = float(np.max(blob_temperature_difference))

    return BlobFeatures(
        track_id=track_id,
        centroid=blob.centroid,

        pixel_count=blob.pixel_count,
        track_age_frames=track_age_frames,

        mean_temperature_difference=mean_temperature_difference,
        max_temperature_difference=max_temperature_difference,
    )

# Finds Closest Blobs From Previous Frame
def match_previous_blob(blob, tracks, used_track_ids, max_match_distance):
    if not tracks:
        return None

    best_dist = None
    best_track = None

    for track in tracks.values():
        if track.id in used_track_ids:
            continue

        dr = blob.centroid[0] - track.centroid[0]
        dc = blob.centroid[1] - track.centroid[1]
        dist = float(np.hypot(dr, dc))

        if dist <= max_match_distance and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_track = track

    return best_track

# Pipeline
@dataclass
class DetectionResult:
    blob: Blob
    features: BlobFeatures

class ThermopilePipeline:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.bg_model = BackgroundModel(frame_shape)
        self._prev_mask = np.zeros(frame_shape, dtype=bool)
        self._prev_blobs: list[Blob] = []

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self.primary_track_id = None

        # Debug state
        self.last_raw_z = np.zeros(frame_shape)
        self.last_smoothed_z = np.zeros(frame_shape)
        self.last_threshold = threshold
        self.last_scene_temp = 0.0

    # Get Background Without Human
    def warm_start(self, calibration_frames):
        self.bg_model.warm_start(calibration_frames)
        self._prev_mask = np.zeros(self.frame_shape, dtype=bool)
        self._prev_blobs = []
        self._tracks = {}
        self._next_track_id = 1
        self.primary_track_id = None

    def process_frame(self, frame):
        # Step 2: Z Score
        z = self.bg_model.z_score(frame)

        # Step 3: Spatial Filtering
        smoothed_z = gaussian_filter(z)

        # Step 4: Threshold
        scene_temp = self.bg_model.scene_temperature()
        mask = apply_threshold(smoothed_z)
        protection_mask = apply_threshold(smoothed_z, thresholdBackgroundProtection)

        temperature_difference = np.abs(frame - self.bg_model.bg)

        # Steps 1: Update Background + Sigma
        self.bg_model.update(frame, protection_mask)

        # Step 5: Blob extraction
        blobs = extract_blobs(mask)

        # Step 6: Feature Extraction variables
        results = []
        used_track_ids = set()
        updated_tracks = {}

        # Step 6: Feature Extraction For Each Blob
        for blob in blobs:
            # Track Blob To Blob From Previous Frame
            track = match_previous_blob(
                blob,
                self._tracks,
                used_track_ids,
                max_match_distance
            )

            # If There Is No Previous Blob, Create New Track
            if track is None:
                track = Track(
                    id=self._next_track_id,
                    centroid=blob.centroid,
                    age_frames=0,
                )

                self._next_track_id += 1

            # Extract The Features Of The Blob
            features = score_blob(blob, track, temperature_difference)

            features.track_id = track.id

            # Accumulate meaningful centroid movement over the lifetime of this track.
            # A newly created track has age 0; it must not inherit movement.
            displacement = float(np.hypot(
                blob.centroid[0] - track.centroid[0],
                blob.centroid[1] - track.centroid[1],
            )) if track.age_frames > 0 else 0.0
            cumulative_displacement = track.cumulative_displacement_px
            if displacement > min_displacement_step_px:
                cumulative_displacement += displacement

            # Update Data From Tracked Blob
            updated_tracks[track.id] = Track(
                id=track.id,
                centroid=blob.centroid,
                age_frames=features.track_age_frames,
                missed_frames=0,
                cumulative_displacement_px=cumulative_displacement,
            )

            used_track_ids.add(track.id)

            # Store The Result For This Blob
            results.append(DetectionResult(blob=blob,features=features))

        # Remove Tracks That Do Not Appear For More Than max_missed_frames
        for track_id, track in self._tracks.items():
            if track_id in used_track_ids:
                continue

            track.missed_frames += 1

            if track.missed_frames <= max_missed_frames:
                updated_tracks[track_id] = track

        self._tracks = updated_tracks

        # Select the currently visible eligible track with the greatest
        # noise-filtered cumulative travel, not the largest blob.
        # Reconsider each frame so an old, static primary does not lock selection.
        eligible_results = [
            result for result in results
            if result.features.track_age_frames >= min_primary_track_frames
        ]
        if eligible_results:
            primary_result = max(
                eligible_results,
                key=lambda result: (
                    updated_tracks[result.features.track_id].cumulative_displacement_px,
                    result.features.track_id == self.primary_track_id,
                    result.features.pixel_count,
                    result.features.mean_temperature_difference,
                ),
            )
            self.primary_track_id = primary_result.features.track_id
        else:
            self.primary_track_id = None

        # Store Previous Info
        self._prev_mask = mask
        self._prev_blobs = blobs

        # Store Latest Intermediates For Display Access
        self.last_raw_z = z
        self.last_smoothed_z = smoothed_z
        self.last_threshold = threshold
        self.last_scene_temp = scene_temp

        return results



def getThermalFeatureNames():
    return [
    "number_of_blobs",
    "total_foreground_pixels",
    "max_z",
    "scene_temperature",
    "pixel_count",
    "centroid_row",
    "centroid_col",
    "track_age_frames",
    "centroid_speed_px_s",
    "max_temperature_difference",
    # Multi-blob summaries: information from every currently visible track.
    "moving_blob_count",
    "max_blob_displacement_px",
    "max_blob_speed_px_s",
    "max_blob_age_frames",
    "max_blob_pixel_count",
    "max_blob_temperature_difference",
    ]


def emptyThermalFeatures():
    return {name: 0.0 for name in getThermalFeatureNames()}

def flattenDetection(detection, centroidSpeed):
    features = detection.features
    return {
        "track_id": int(features.track_id),
        "pixel_count": float(features.pixel_count),
        "centroid_row": float(features.centroid[0]),
        "centroid_col": float(features.centroid[1]),
        "track_age_frames": float(features.track_age_frames),
        "centroid_speed_px_s": float(centroidSpeed),
        "max_temperature_difference": float(features.max_temperature_difference),
    }


def processThermopile(thermalData, metadata):
    frameShape = tuple(int(value) for value in metadata["thermal_frame_shape"])
    frameLength = int(np.prod(frameShape))
    pixelColumns = [f"pixel_{i}" for i in range(frameLength)]

    missing = [column for column in pixelColumns if column not in thermalData.columns]
    if missing:
        raise ValueError(f"Thermopile CSV is missing {len(missing)} pixel columns")

    pipeline = ThermopilePipeline(frame_shape=frameShape)
    background = np.asarray(metadata["thermal_background"], dtype=np.float64).reshape(frameShape)
    sigma = np.asarray(metadata["thermal_sigma"], dtype=np.float64).reshape(frameShape)
    pipeline.bg_model.bg[:] = background
    pipeline.bg_model.sigma[:] = sigma
    pipeline.bg_model._initialised = True

    timestamps = thermalData["time_s"].to_numpy(dtype=np.float64)
    frameRows = []
    previousTrackPositions = {}

    for frameIndex, (_, row) in enumerate(thermalData.iterrows()):
        frame = row[pixelColumns].to_numpy(dtype=np.float64).reshape(frameShape)
        detections = pipeline.process_frame(frame)
        timestamp = float(timestamps[frameIndex])
        flattened = []
        updatedTrackPositions = dict(previousTrackPositions)

        for detection in detections:
            trackId = int(detection.features.track_id)
            centroidRow = float(detection.features.centroid[0])
            centroidCol = float(detection.features.centroid[1])
            centroidSpeed = 0.0

            if trackId in previousTrackPositions:
                previousTime, previousRow, previousCol = previousTrackPositions[trackId]
                dt = timestamp - previousTime
                if dt > 0:
                    centroidSpeed = float(
                        np.hypot(
                            centroidRow - previousRow,
                            centroidCol - previousCol,
                        ) / dt
                    )

            blobFeatures = flattenDetection(detection, centroidSpeed)
            blobFeatures["cumulative_displacement_px"] = float(
                pipeline._tracks[trackId].cumulative_displacement_px
            )
            flattened.append(blobFeatures)
            updatedTrackPositions[trackId] = (timestamp, centroidRow, centroidCol)

        previousTrackPositions = updatedTrackPositions

        smoothedZ = pipeline.last_smoothed_z
        flatZ = smoothedZ.reshape(-1)

        frameRows.append({
            "detections": flattened,
            "number_of_blobs": len(flattened),
            "total_foreground_pixels": float(sum(detection["pixel_count"] for detection in flattened)),
            "primary_track_id": pipeline.primary_track_id,
            "max_z": float(np.max(flatZ)),
            "scene_temperature": float(pipeline.last_scene_temp),
        })

    return timestamps, frameRows


def getThermalFeatures(thermalRows, thermalIndex):
    features = emptyThermalFeatures()
    currentFrame = thermalRows[thermalIndex]
    selectedTrackId = currentFrame["primary_track_id"]

    features["number_of_blobs"] = float(currentFrame["number_of_blobs"])
    features["total_foreground_pixels"] = float(currentFrame["total_foreground_pixels"])
    features["max_z"] = float(currentFrame["max_z"])
    features["scene_temperature"] = float(currentFrame["scene_temperature"])

    # Scene-wide summaries don't depend on picking the correct primary blob.
    # Values are computed from the current frame only, with track histories
    # accumulated causally from previous frames.
    detections = currentFrame["detections"]
    if detections:
        features["moving_blob_count"] = float(sum(
            detection["centroid_speed_px_s"] > 0.5
            for detection in detections
        ))
        features["max_blob_displacement_px"] = float(max(
            detection["cumulative_displacement_px"] for detection in detections
        ))
        features["max_blob_speed_px_s"] = float(max(
            detection["centroid_speed_px_s"] for detection in detections
        ))
        features["max_blob_age_frames"] = float(max(
            detection["track_age_frames"] for detection in detections
        ))
        features["max_blob_pixel_count"] = float(max(
            detection["pixel_count"] for detection in detections
        ))
        features["max_blob_temperature_difference"] = float(max(
            detection["max_temperature_difference"] for detection in detections
        ))

    if selectedTrackId is None:
        return features

    matching = [
        detection
        for detection in currentFrame["detections"]
        if int(detection["track_id"]) == int(selectedTrackId)
    ]
    if not matching:
        return features

    currentPrimary = matching[0]
    features["pixel_count"] = float(currentPrimary["pixel_count"])
    features["centroid_row"] = float(currentPrimary["centroid_row"])
    features["centroid_col"] = float(currentPrimary["centroid_col"])
    features["track_age_frames"] = float(currentPrimary["track_age_frames"])
    features["centroid_speed_px_s"] = float(currentPrimary["centroid_speed_px_s"])
    features["max_temperature_difference"] = float(currentPrimary["max_temperature_difference"])

    return features




# Streaming adapter: the same features and state updates as processThermopile.
class LiveThermalProcessor:
    def __init__(self, frame_shape=(24, 32)):
        self.pipeline = ThermopilePipeline(frame_shape)
        self.previousTrackPositions = {}

    def calibrate(self, frames):
        self.pipeline.warm_start(np.asarray(frames, dtype=np.float64))
        self.previousTrackPositions.clear()

    def process(self, timestamp, frame):
        frame = np.asarray(frame, dtype=np.float64).reshape(self.pipeline.frame_shape)
        detections = self.pipeline.process_frame(frame)
        flattened = []
        updatedTrackPositions = dict(self.previousTrackPositions)
        for detection in detections:
            trackId = int(detection.features.track_id)
            centroidRow = float(detection.features.centroid[0])
            centroidCol = float(detection.features.centroid[1])
            centroidSpeed = 0.0
            if trackId in self.previousTrackPositions:
                previousTime, previousRow, previousCol = self.previousTrackPositions[trackId]
                dt = float(timestamp) - previousTime
                if dt > 0:
                    centroidSpeed = float(np.hypot(centroidRow - previousRow, centroidCol - previousCol) / dt)
            blobFeatures = flattenDetection(detection, centroidSpeed)
            blobFeatures["cumulative_displacement_px"] = float(
                self.pipeline._tracks[trackId].cumulative_displacement_px
            )
            flattened.append(blobFeatures)
            updatedTrackPositions[trackId] = (float(timestamp), centroidRow, centroidCol)
        self.previousTrackPositions = updatedTrackPositions
        frameRow = {
            "detections": flattened,
            "number_of_blobs": len(flattened),
            "total_foreground_pixels": float(sum(d["pixel_count"] for d in flattened)),
            "primary_track_id": self.pipeline.primary_track_id,
            "max_z": float(np.max(self.pipeline.last_smoothed_z)),
            "scene_temperature": float(self.pipeline.last_scene_temp),
        }
        return getThermalFeatures([frameRow], 0)


# Live acquisition and debug-display compatibility functions (formerly live_processing.py)
# Import the hardware interface only when a live frame is requested so that
# offline dataset creation and model export also work on a PC without I2C hardware.
REFRESH_RATE_HZ = 4.0  # Used for legacy debug calibration; main.py reads the configured interface rate.
def getRawFrame():
    from Thermopile.interface import getFrame
    return getFrame()

WIDTH = 32
HEIGHT = 24
FRAME_SHAPE = (HEIGHT, WIDTH)

N_WARMUP_FRAMES = 20
CALIBRATION_SECONDS = 10

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

def calibrate(seconds=CALIBRATION_SECONDS):
    calibrationFrameCount = max(1, int(round(float(seconds) * REFRESH_RATE_HZ)))
    calibrationFrames = np.empty((calibrationFrameCount, *FRAME_SHAPE), dtype=np.float64)
    for i in range(calibrationFrameCount):
        calibrationFrames[i] = convertFlatArray(getRawFrame())
    Pipeline.warm_start(calibrationFrames)

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


# Current Primary Track ID
def getPrimaryTrackId():
    return Pipeline.primary_track_id

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

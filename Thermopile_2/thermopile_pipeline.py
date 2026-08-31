from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Config
ALPHA = 0.995                 
SIGMA_MIN = 0.05 # Minimum Sigma

# Spatial filter
gaussianKernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]) / 16.0
filterMode = "reflect"

# Threshold
fixed_threshold = 3.0 # When not using adaptive thresholding
use_adaptive_threshold = False
adaptive_coeffs = None # [a, b, c] TODO: Replace with proper coefficients
adaptive_min_threshold = 1.5  # lower bound
adaptive_max_threshold = 8.0  # upper bound

# Blob extraction
connectivity_8 = True
min_blob_pixels = 1 # Blobs smaller than this are discarded

# Feature scoring
# TODO: Replace with proper motion parameters
max_expected_motion_px = 6.0  # per frame centroid displacement ceiling
max_match_distance = max_expected_motion_px * 1.5 

# Background Model Class
class BackgroundModel:
    def __init__(self, shape):
        self.bg = np.zeros(shape)
        self.sigma = np.full(shape, SIGMA_MIN)
        self._initialised = False

    # Calibrate Background
    def warm_start(self, calibration_frames):
        self.bg[:] = calibration_frames.mean(axis=0)
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

# Thresholding 

#TODO: Understand
def fit_adaptive_threshold_curve(scene_temps, optimal_thresholds):
    """
    Fit a quadratic curve: threshold = a*T^2 + b*T + c

    Call this once, offline, on calibration data collected during
    characterisation testing (scene_temps and optimal_thresholds are
    paired arrays of equal length). Store the returned coefficients in
    PipelineConfig.adaptive_coeffs.
    """
    coeffs = np.polyfit(scene_temps, optimal_thresholds, deg=2)  # [a, b, c]
    return coeffs

# Calculate Threshold Using Either Adaptive or Fixed Thresholding
def compute_threshold(scene_temp):
    if not use_adaptive_threshold or adaptive_coeffs is None:
        return fixed_threshold

    a, b, c = adaptive_coeffs
    threshold = a * scene_temp**2 + b * scene_temp + c
    threshold = np.clip(threshold, adaptive_min_threshold, adaptive_max_threshold)
    return float(threshold)

# Apply Threshold To Get Mask
def apply_threshold(smoothed_z, threshold):
    return smoothed_z > threshold

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

# Blob Feature Scoring
@dataclass
class BlobFeatures:
    motion_score: float

# Checks Blobs Displacement From Previous Frame
def score_blob(blob, prev_centroid):
    if prev_centroid is None:
        motion_score = 0.5  # neutral: no prior frame to compare against
    else:
        # Gets Change in X and Y Of Previous Blob Centroid
        dr = blob.centroid[0] - prev_centroid[0]
        dc = blob.centroid[1] - prev_centroid[1]
        displacement = float(np.hypot(dr, dc))

        # Checks Motion By How Much A Person Is Expected To Have Moved In One Frame
        normalised = displacement / max_expected_motion_px

        # Computes Motion Score Based On How Close To Expected Motion The Blob Is
        motion_score = max(1.0 - abs(normalised - 0.3) / 0.3, 0.0)

    return BlobFeatures(
        motion_score=motion_score,
    )

# Finds Closest Blobs From Previous Frame
def match_previous_blob(blob, prev_blobs, max_match_distance):
    if not prev_blobs:
        return None

    best_dist = None
    best_centroid = None
    for pb in prev_blobs:
        dr = blob.centroid[0] - pb.centroid[0]
        dc = blob.centroid[1] - pb.centroid[1]
        dist = float(np.hypot(dr, dc))
        if dist <= max_match_distance and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_centroid = pb.centroid

    return best_centroid

# Confidence score
def compute_confidence(features):
    return float(np.clip(features.motion_score, 0.0, 1.0))

# Pipeline
@dataclass
class DetectionResult:
    blob: Blob
    features: BlobFeatures
    confidence: float

class ThermopilePipeline:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.bg_model = BackgroundModel(frame_shape)
        self._prev_mask = np.zeros(frame_shape, dtype=bool)
        self._prev_blobs: list[Blob] = []

        # Debug state
        self.last_raw_z = np.zeros(frame_shape)
        self.last_smoothed_z = np.zeros(frame_shape)
        self.last_threshold = fixed_threshold
        self.last_scene_temp = 0.0

    # Get Background Without Human
    def warm_start(self, calibration_frames):
        self.bg_model.warm_start(calibration_frames)

    def process_frame(self, frame):
        # Step 2: Z Score
        z = self.bg_model.z_score(frame)

        # Step 3: Spatial Filtering
        smoothed_z = gaussian_filter(z)

        # Step 4: Threshold
        scene_temp = self.bg_model.scene_temperature()
        threshold = compute_threshold(scene_temp)
        mask = apply_threshold(smoothed_z, threshold)

        # Steps 1: Update Background + Sigma
        self.bg_model.update(frame, mask)

        # Step 5: Blob extraction
        blobs = extract_blobs(mask)

        # Steps 6-7: Feature scoring + Confidence Per Blob
        results = []
        for blob in blobs:
            prev_centroid = match_previous_blob(blob, self._prev_blobs, max_match_distance)
            features = score_blob(blob, prev_centroid)
            confidence = compute_confidence(features)
            results.append(DetectionResult(blob=blob, features=features, confidence=confidence))

        # Store Previous Info
        self._prev_mask = mask
        self._prev_blobs = blobs

        # Store latest intermediates for external debug/display access
        self.last_raw_z = z
        self.last_smoothed_z = smoothed_z
        self.last_threshold = threshold
        self.last_scene_temp = scene_temp

        return results
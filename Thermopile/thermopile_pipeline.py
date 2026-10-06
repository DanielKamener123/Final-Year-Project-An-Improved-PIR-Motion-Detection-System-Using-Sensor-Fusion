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
background_protection_threshold = 2.0
use_adaptive_threshold = False
adaptive_coeffs = None # [a, b, c] TODO: Replace with proper coefficients
adaptive_min_threshold = 1.5  # lower bound
adaptive_max_threshold = 8.0  # upper bound

# Blob extraction
connectivity_8 = True
min_blob_pixels = 1 # Blobs smaller than this are discarded

# Feature scoring
max_match_distance = 6 * 1.5
max_missed_frames = 2
min_primary_track_frames = 2

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

@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    age_frames: int
    missed_frames: int = 0

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
        self.last_threshold = fixed_threshold
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
        threshold = compute_threshold(scene_temp)
        mask = apply_threshold(smoothed_z, threshold)
        protection_mask = apply_threshold(smoothed_z, background_protection_threshold)

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

            # Update Data From Tracked Blob
            updated_tracks[track.id] = Track(
                id=track.id,
                centroid=blob.centroid,
                age_frames=features.track_age_frames,
                missed_frames=0,
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

        # Keep Current Primary Track Until It Is Genuinely Lost
        if self.primary_track_id is not None and self.primary_track_id not in self._tracks:
            self.primary_track_id = None

        # Select A New Primary Track Only When There Is No Current Primary
        if self.primary_track_id is None:
            eligible_results = [
                result
                for result in results
                if result.features.track_age_frames >= min_primary_track_frames
            ]

            if eligible_results:
                primary_result = max(
                    eligible_results,
                    key=lambda result: (
                        result.features.pixel_count,
                        result.features.mean_temperature_difference,
                    ),
                )
                self.primary_track_id = primary_result.features.track_id

        # Store Previous Info
        self._prev_mask = mask
        self._prev_blobs = blobs

        # Store Latest Intermediates For Display Access
        self.last_raw_z = z
        self.last_smoothed_z = smoothed_z
        self.last_threshold = threshold
        self.last_scene_temp = scene_temp

        return results

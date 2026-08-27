from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

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
use_adaptive_threshold: bool = False
adaptive_coeffs = None # [a, b, c] TODO: Replace with proper coefficients
adaptive_min_threshold: float = 1.5  # lower bound
adaptive_max_threshold: float = 8.0  # upper bound

# Blob extraction
connectivity_8 = True
min_blob_pixels: int = 1 # Blobs smaller than this are discarded

# Feature scoring
expected_max_z = 10.0         # contrast normalisation ceiling
# TODO: Replace with proper range to pixel count
size_lookup = np.array([
    [1.0, 30.0],
    [2.0, 14.0],
    [3.0, 7.0],
    [4.0, 4.0],
    [5.0, 2.0],
])
aspect_ratio_min = 0.5
aspect_ratio_max = 2.5

# TODO: Replace with proper motion parameters
max_expected_motion_px = 6.0  # per frame centroid displacement ceiling

# Feature weights - must sum to 1.0 TODO: Replace with proper weights
w_contrast: float = 0.35
w_size: float = 0.25
w_aspect: float = 0.20
w_motion: float = 0.20

# --- New: morphological mask cleanup ----------------------------------
# Erosion then dilation on the binary foreground mask before it is used
# for background update / blob extraction. Removes isolated single-pixel
# noise and bridges small gaps within a real target's blob. See Cerutti
# et al. outdoor thermal pipeline discussion of morphological refinement.
use_morphological_cleanup = True
morph_structure = np.ones((3, 3), dtype=bool)
morph_erosion_iterations = 1
morph_dilation_iterations = 1

# --- New: geometric (pinhole/FOV) range estimation ---------------------
# Replaces the naive linear row->range placeholder with a proper
# trigonometric model based on sensor mounting geometry. Falls back to
# the original placeholder if use_geometric_range is False.
# TODO: replace FOV/mounting values with calibrated figures for the
# final Rock 4C+ mount (datasheet vertical FOV, measured height/tilt).
sensor_vertical_fov_deg = 110.0   # MLX90640 narrow-lens vertical FOV
sensor_mount_height_m = 2.5      # sensor height above floor
sensor_tilt_deg = 0.0            # 0 = pointed straight down; +ve tilts forward

# --- New: SORT-style blob tracking (Kalman + Hungarian assignment) -----
# Alternative to the original greedy nearest-centroid matching. Reduces
# identity switches when two blobs pass close to each other.
# tracking_mode: "kalman" (new) or "centroid" (original match_previous_blob)
tracking_mode = "kalman"
kalman_process_noise = 0.5
kalman_measurement_noise = 1.0
max_track_age = 5  # frames a track survives with no matching blob

# --- New: pluggable confidence combiner --------------------------------
# "weighted_sum" (original) or "logistic" (feature-level fusion stretch
# goal). Falls back to weighted_sum automatically if logistic_coeffs
# hasn't been trained/set yet.
confidence_mode = "weighted_sum"
logistic_coeffs = None  # TODO: (intercept, [w_contrast, w_size, w_aspect, w_motion]) via sklearn.linear_model.LogisticRegression once labelled data exists

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

# Cleanup The Foreground Binary Mask
def morphological_cleanup(binary_mask):
    eroded = ndimage.binary_erosion(binary_mask, structure=morph_structure, iterations=morph_erosion_iterations)
    cleaned = ndimage.binary_dilation(eroded, structure=morph_structure, iterations=morph_dilation_iterations)
    return cleaned

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
    contrast: float
    size_score: float
    aspect_score: float
    motion_score: float

# New: pinhole/FOV geometric range model, replacing the linear placeholder.
# Converts a pixel row into an elevation angle from the sensor's optical
# axis (using the vertical FOV), then into a ground-plane range via simple
# trigonometry against the mounting height/tilt. See Naser et al. (2021)
# and Zhang & Wu / TADAR (2024) for the underlying range-estimation idea.
def _estimate_range_geometric(centroid_row, frame_rows):
    angle_per_pixel = sensor_vertical_fov_deg / frame_rows
    pixel_offset = centroid_row - (frame_rows - 1) / 2.0
    pixel_angle = pixel_offset * angle_per_pixel
    angle_from_nadir = np.clip(sensor_tilt_deg + pixel_angle, -89.0, 89.0)
    range_m = sensor_mount_height_m * np.tan(np.radians(angle_from_nadir))
    return float(abs(range_m))

def _estimate_expected_pixels_geometric(centroid_row, frame_rows):
    range_m = _estimate_range_geometric(centroid_row, frame_rows)
    expected_pixels = np.interp(range_m, size_lookup[:, 0], size_lookup[:, 1])
    return float(expected_pixels)

# New: dispatcher -- keeps the original function name/signature so
# score_blob below needs no changes. Selects geometric or legacy mapping
# based on the use_geometric_range config flag.
def _estimate_expected_pixels(centroid_row, frame_rows):
    return _estimate_expected_pixels_geometric(centroid_row, frame_rows)

def score_blob(blob, smoothed_z, frame_shape, prev_centroid):
    # Feature 1: contrast
    peak_z = float(np.max(smoothed_z[blob.mask]))
    contrast_score = min(peak_z / expected_max_z, 1.0)

    # Feature 2: Size at Range
    expected_pixels = _estimate_expected_pixels(blob.centroid[0], frame_shape[0])
    if expected_pixels > 0:
        size_score = 1.0 - abs(blob.pixel_count - expected_pixels) / expected_pixels
        size_score = max(size_score, 0.0)
    else:
        size_score = 0.0

    # Feature 3: Aspect Ratio
    min_row, min_col, max_row, max_col = blob.bbox
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    aspect_ratio = height / width if width > 0 else 0.0
    if aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
        aspect_score = 1.0
    else:
        # linear fall-off outside the plausible band
        if aspect_ratio < aspect_ratio_min:
            dist = aspect_ratio_min - aspect_ratio
        else:
            dist = aspect_ratio - aspect_ratio_max
        aspect_score = max(1.0 - dist, 0.0)

    # Feature 4: centroid motion vs previous frame's matched blob
    if prev_centroid is None:
        motion_score = 0.5  # neutral: no prior frame to compare against
    else:
        dr = blob.centroid[0] - prev_centroid[0]
        dc = blob.centroid[1] - prev_centroid[1]
        displacement = float(np.hypot(dr, dc))
        # simple bell-shaped preference: some motion good, too much or
        # none is penalised -- tune against real walking-speed data
        normalised = displacement / max_expected_motion_px
        motion_score = max(1.0 - abs(normalised - 0.3), 0.0)

    return BlobFeatures(
        contrast=contrast_score,
        size_score=size_score,
        aspect_score=aspect_score,
        motion_score=motion_score,
    )

def match_previous_blob(blob, prev_blobs, max_match_distance):
    """
    Nearest-centroid matching against the previous frame's blobs, used to
    feed prev_centroid into score_blob for the motion feature.
    """
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

# New: SORT-style tracking (constant-velocity Kalman filter per blob +
# Hungarian/linear-sum-assignment matching) as an alternative to the
# nearest-centroid matching above. Reduces identity switches when two
# blobs' centroids come close together. Ref: Bewley et al. (2016) SORT;
# Kuhn (1955) Hungarian algorithm.
class _KalmanTrack:
    def __init__(self, track_id, centroid):
        self.id = track_id
        # State: [row, col, v_row, v_col]
        self.state = np.array([centroid[0], centroid[1], 0.0, 0.0])
        self.covariance = np.eye(4) * 10.0
        self.missed_frames = 0

    def predict(self):
        transition = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        process_noise = np.eye(4) * kalman_process_noise
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        return self.state[:2]

    def update(self, centroid):
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        measurement_noise = np.eye(2) * kalman_measurement_noise
        z = np.array(centroid)
        innovation = z - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + measurement_noise
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (np.eye(4) - kalman_gain @ observation) @ self.covariance
        self.missed_frames = 0

    @property
    def centroid(self):
        return (self.state[0], self.state[1])

class BlobTracker:
    """
    Maintains a Kalman track per blob across frames and assigns this
    frame's blobs to tracks via the Hungarian algorithm (minimising
    total centroid distance), gated by max_match_distance. Returns, per
    blob (in the same order as the input list), the matched track's
    pre-update (predicted) centroid -- or None for a newly created track
    -- for use as score_blob's prev_centroid motion-feature input.
    """
    def __init__(self):
        self._tracks = {}
        self._next_id = 0

    def step(self, blobs, max_match_distance):
        predicted = {tid: track.predict() for tid, track in self._tracks.items()}
        track_ids = list(predicted.keys())

        unmatched_cost = max_match_distance * 10.0
        cost_matrix = np.full((len(track_ids), len(blobs)), unmatched_cost)
        for i, tid in enumerate(track_ids):
            for j, blob in enumerate(blobs):
                dist = float(np.hypot(
                    blob.centroid[0] - predicted[tid][0],
                    blob.centroid[1] - predicted[tid][1],
                ))
                if dist <= max_match_distance:
                    cost_matrix[i, j] = dist

        matches = {}  # blob index -> track id
        if track_ids and blobs:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= max_match_distance:
                    matches[c] = track_ids[r]

        matched_track_ids = set(matches.values())
        prev_centroids = [None] * len(blobs)
        for j, blob in enumerate(blobs):
            if j in matches:
                tid = matches[j]
                prev_centroids[j] = self._tracks[tid].centroid
                self._tracks[tid].update(blob.centroid)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _KalmanTrack(tid, blob.centroid)
                prev_centroids[j] = None

        for tid in list(self._tracks.keys()):
            if tid not in matched_track_ids:
                self._tracks[tid].missed_frames += 1
                if self._tracks[tid].missed_frames > max_track_age:
                    del self._tracks[tid]

        return prev_centroids

# Confidence score
def compute_confidence(features):
    confidence = (
        w_contrast * features.contrast
        + w_size * features.size_score
        + w_aspect * features.aspect_score
        + w_motion * features.motion_score
    )
    return float(np.clip(confidence, 0.0, 1.0))

# New: logistic-regression confidence combiner (feature-level fusion
# stretch goal). Falls back to the weighted-sum confidence above until
# logistic_coeffs has been fitted on labelled data.
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_confidence_logistic(features):
    if logistic_coeffs is None:
        return compute_confidence(features)

    intercept, weights = logistic_coeffs
    w_contrast_l, w_size_l, w_aspect_l, w_motion_l = weights
    z = (
        intercept
        + w_contrast_l * features.contrast
        + w_size_l * features.size_score
        + w_aspect_l * features.aspect_score
        + w_motion_l * features.motion_score
    )
    return float(_sigmoid(z))

# New: dispatcher -- selects weighted-sum or logistic combiner based on
# the confidence_mode config flag.
def compute_confidence_dispatch(features):
    if confidence_mode == "logistic":
        return compute_confidence_logistic(features)
    return compute_confidence(features)

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

        # New: SORT-style tracker, used when tracking_mode == "kalman"
        self._tracker = BlobTracker()

        # Debug/introspection state -- last-frame intermediate arrays, kept
        # around so external tools (e.g. a live display) can inspect the
        # pipeline without recomputing anything.
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

        # New Step 4b: morphological mask cleanup
        if use_morphological_cleanup:
            mask = morphological_cleanup(mask)

        # Steps 1: Update Background + Sigma
        self.bg_model.update(frame, mask)

        # Step 5: Blob extraction
        blobs = extract_blobs(mask)

        # Steps 6-7: Feature scoring + Confidence Per Blob
        results = []
        max_match_distance = max_expected_motion_px * 1.5

        # New: dispatch to Kalman/Hungarian tracking or the original
        # nearest-centroid matching, based on tracking_mode.
        if tracking_mode == "kalman":
            prev_centroids = self._tracker.step(blobs, max_match_distance)
        else:
            prev_centroids = [
                match_previous_blob(blob, self._prev_blobs, max_match_distance)
                for blob in blobs
            ]

        for blob, prev_centroid in zip(blobs, prev_centroids):
            features = score_blob(blob, smoothed_z, self.frame_shape, prev_centroid)
            confidence = compute_confidence_dispatch(features)
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
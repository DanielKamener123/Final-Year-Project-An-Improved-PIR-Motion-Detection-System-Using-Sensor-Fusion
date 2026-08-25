from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

# Config

@dataclass
class PipelineConfig:
    alpha: float = 0.995                 
    sigmaMin: float = 0.05 # minimum sigma

    # Spatial filter
    gaussian_kernel: np.ndarray = field(default_factory=lambda: np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float64) / 16.0)
    filter_mode: str = "reflect"

    # Threshold
    fixed_threshold: float = 3.0 # When not using adaptive thresholding
    use_adaptive_threshold: bool = False
    adaptive_coeffs: Optional[np.ndarray] = None # [a, b, c]
    adaptive_min_threshold: float = 1.5  # lower bound
    adaptive_max_threshold: float = 8.0  # upper bound

    # Blob extraction
    connectivity_8: bool = True
    min_blob_pixels: int = 1 # Blobs smaller than this are discarded

    # Feature scoring
    expected_max_z: float = 10.0         # contrast normalisation ceiling
    # pixel_count-vs-range lookup: list of (range_m, expected_pixels) pairs,
    # sorted by range. Fill in from your own MLX90640 characterisation.
    size_lookup: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 30.0],
        [2.0, 14.0],
        [3.0, 7.0],
        [4.0, 4.0],
        [5.0, 2.0],
    ]))
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 2.5
    max_expected_motion_px: float = 6.0  # per-frame centroid displacement ceiling

    # Feature weights - must sum to 1.0
    w_contrast: float = 0.35
    w_size: float = 0.25
    w_aspect: float = 0.20
    w_motion: float = 0.20

# Background model
class BackgroundModel:

    def __init__(self, shape: tuple[int, int], config: PipelineConfig):
        self.config = config
        self.bg = np.zeros(shape, dtype=np.float64)
        self.sigma = np.full(shape, config.sigmaMin, dtype=np.float64)
        self._initialised = False

    # Calibrate Background
    def warm_start(self, calibration_frames: np.ndarray) -> None:
        self.bg[:] = calibration_frames.mean(axis=0)
        measured_sigma = calibration_frames.std(axis=0)
        self.sigma = np.maximum(measured_sigma, self.config.sigma_floor)
        self._initialised = True

    # Update Background
    def update(self, frame: np.ndarray, foreground_mask: np.ndarray) -> None:
        alpha = self.config.alpha
        update_mask = ~foreground_mask

        # Updates Background Where No Humans
        new_bg = alpha * self.bg + (1 - alpha) * frame
        self.bg = np.where(update_mask, new_bg, self.bg)

        # Update Sigma Where No Human
        new_sigma_sq = alpha * (self.sigma ** 2) + (1 - alpha) * (self.bg - frame) ** 2
        new_sigma = np.sqrt(new_sigma_sq)
        new_sigma = np.maximum(new_sigma, self.config.sigma_floor)
        self.sigma = np.where(update_mask, new_sigma, self.sigma)

    # Calculate Z value - How many multiples of this pixels normal variation is present
    def z_score(self, frame: np.ndarray) -> np.ndarray:
        return np.abs(self.bg - frame) / self.sigma

    # Calculate Mean Background Temp For Adaptive Thresholding
    def scene_temperature(self) -> float:
        return float(np.mean(self.bg))


# Spatial filtering
def gaussian_filter_3x3(z_grid: np.ndarray, config: PipelineConfig) -> np.ndarray:
    return ndimage.convolve(z_grid, config.gaussian_kernel, mode=config.filter_mode)


# Thresholding 


def fit_adaptive_threshold_curve(scene_temps: np.ndarray,
                                  optimal_thresholds: np.ndarray) -> np.ndarray:
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
def compute_threshold(scene_temp: float, config: PipelineConfig) -> float:
    if not config.use_adaptive_threshold or config.adaptive_coeffs is None:
        return config.fixed_threshold

    a, b, c = config.adaptive_coeffs
    threshold = a * scene_temp**2 + b * scene_temp + c
    threshold = np.clip(threshold, config.adaptive_min_threshold, config.adaptive_max_threshold)
    return float(threshold)

# Apply Threshold To Get Mask
def apply_threshold(smoothed_z: np.ndarray, threshold: float) -> np.ndarray:
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
def extract_blobs(binary_mask: np.ndarray, config: PipelineConfig) -> list[Blob]:
    if config.connectivity_8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

    labelled_grid, num_blobs = ndimage.label(binary_mask, structure=structure)

    blobs: list[Blob] = []
    for blob_id in range(1, num_blobs + 1):
        blob_mask = labelled_grid == blob_id
        pixel_count = int(blob_mask.sum())
        if pixel_count < config.min_blob_pixels:
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


def _estimate_expected_pixels(centroid_row: float, frame_rows: int,
                               config: PipelineConfig) -> float:
    """
    Rough range estimate from vertical position in frame, mapped through
    the size_lookup table. Replace this mapping with your actual mounting
    geometry once characterised -- this is a placeholder linear mapping
    from "row position" to "range" before the pixel-count lookup.
    """
    # naive placeholder: assume top row = farthest, bottom row = nearest
    frac = centroid_row / max(frame_rows - 1, 1)
    ranges = config.size_lookup[:, 0]
    range_m = ranges[0] + frac * (ranges[-1] - ranges[0])
    expected_pixels = np.interp(range_m, config.size_lookup[:, 0], config.size_lookup[:, 1])
    return float(expected_pixels)


def score_blob(blob: Blob, smoothed_z: np.ndarray, frame_shape: tuple[int, int],
               prev_centroid: Optional[tuple[float, float]],
               config: PipelineConfig) -> BlobFeatures:
    """Step 6: compute the four features for a single blob."""

    # Feature 1: contrast (peak Z-score within the blob)
    peak_z = float(np.max(smoothed_z[blob.mask]))
    contrast_score = min(peak_z / config.expected_max_z, 1.0)

    # Feature 2: size-at-range
    expected_pixels = _estimate_expected_pixels(blob.centroid[0], frame_shape[0], config)
    if expected_pixels > 0:
        size_score = 1.0 - abs(blob.pixel_count - expected_pixels) / expected_pixels
        size_score = max(size_score, 0.0)
    else:
        size_score = 0.0

    # Feature 3: aspect ratio
    min_row, min_col, max_row, max_col = blob.bbox
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    aspect_ratio = height / width if width > 0 else 0.0
    if config.aspect_ratio_min <= aspect_ratio <= config.aspect_ratio_max:
        aspect_score = 1.0
    else:
        # linear fall-off outside the plausible band
        if aspect_ratio < config.aspect_ratio_min:
            dist = config.aspect_ratio_min - aspect_ratio
        else:
            dist = aspect_ratio - config.aspect_ratio_max
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
        normalised = displacement / config.max_expected_motion_px
        motion_score = max(1.0 - abs(normalised - 0.3), 0.0)

    return BlobFeatures(
        contrast=contrast_score,
        size_score=size_score,
        aspect_score=aspect_score,
        motion_score=motion_score,
    )


def match_previous_blob(blob: Blob, prev_blobs: list[Blob],
                         max_match_distance: float) -> Optional[tuple[float, float]]:
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


# Confidence score
def compute_confidence(features: BlobFeatures, config: PipelineConfig) -> float:
    confidence = (
        config.w_contrast * features.contrast
        + config.w_size * features.size_score
        + config.w_aspect * features.aspect_score
        + config.w_motion * features.motion_score
    )
    return float(np.clip(confidence, 0.0, 1.0))

# Pipeline
@dataclass
class DetectionResult:
    blob: Blob
    features: BlobFeatures
    confidence: float

class ThermopilePipeline:
    def __init__(self, frame_shape: tuple[int, int] = (24, 32),
                 config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.frame_shape = frame_shape
        self.bg_model = BackgroundModel(frame_shape, self.config)
        self._prev_mask = np.zeros(frame_shape, dtype=bool)
        self._prev_blobs: list[Blob] = []

    # Get Background Without Human
    def warm_start(self, calibration_frames: np.ndarray) -> None:
        self.bg_model.warm_start(calibration_frames)

    def process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        # Steps 1: Update Background + Sigma
        self.bg_model.update(frame, self._prev_mask)

        # Step 2: Z Score
        z = self.bg_model.z_score(frame)

        # Step 3: Spatial Filtering
        smoothed_z = gaussian_filter_3x3(z, self.config)

        # Step 4: Threshold
        scene_temp = self.bg_model.scene_temperature()
        threshold = compute_threshold(scene_temp, self.config)
        mask = apply_threshold(smoothed_z, threshold)

        # Step 5: Blob extraction
        blobs = extract_blobs(mask, self.config)

        # Steps 6-7: Feature scoring + Confidence Per Blob
        results: list[DetectionResult] = []
        max_match_distance = self.config.max_expected_motion_px * 1.5
        for blob in blobs:
            prev_centroid = match_previous_blob(blob, self._prev_blobs, max_match_distance)
            features = score_blob(blob, smoothed_z, self.frame_shape, prev_centroid, self.config)
            confidence = compute_confidence(features, self.config)
            results.append(DetectionResult(blob=blob, features=features, confidence=confidence))

        # Store Previous Info
        self._prev_mask = mask
        self._prev_blobs = blobs

        return results


# --------------------------------------------------------------------------
# Minimal usage example
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace this with real frames from the MLX90640 (via the FT232H
    # bridge during bring-up, or native I2C on the Rock 4C+ at deployment).
    frame_shape = (24, 32)
    pipeline = ThermopilePipeline(frame_shape=frame_shape)

    rng = np.random.default_rng(seed=0)
    ambient = 22.0

    # Warm-up: 20 empty-scene frames so sigma reflects real noise levels
    # before live processing starts (see BackgroundModel.warm_start).
    calibration_frames = ambient + rng.normal(0, 0.3, size=(20, *frame_shape))
    pipeline.warm_start(calibration_frames)

    for frame_idx in range(5):
        frame = ambient + rng.normal(0, 0.3, size=frame_shape)
        if frame_idx >= 2:
            frame[10:13, 14:17] += 8.0  # simulate a person-shaped warm blob

        detections = pipeline.process_frame(frame)

        print(f"Frame {frame_idx}: {len(detections)} blob(s)")
        for d in detections:
            print(f"  blob id={d.blob.id} pixels={d.blob.pixel_count} "
                  f"centroid={d.blob.centroid} confidence={d.confidence:.3f}")

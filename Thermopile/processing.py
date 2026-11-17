from dataclasses import dataclass
from collections import deque

import numpy as np

# Thermal sensor dimensions
WIDTH = 32
HEIGHT = 24
FRAME_SHAPE = (HEIGHT, WIDTH)

# Background adaptation and minimum noise level
ALPHA = 0.995
SIGMA_MIN = 0.15
THRESHOLD = 3.0

# Gaussian smoothing weights
centerWeight = 4.0
gaussianKernel = np.array([
    [1, 2, 1],
    [2, centerWeight, 2],
    [1, 2, 1],
])
gaussianKernel /= gaussianKernel.sum()

# Reject sudden temperature jumps and briefly reuse valid features
GLITCH_SHIFT_C = 3.0
GLITCH_MAX_S = 30.0
FEATURE_HOLD_S = 1.0

_rows, _cols = np.indices((HEIGHT, WIDTH))
PIXEL_GROUP = ((_rows + _cols) % 2) * 4 + (_cols % 4)
PIXEL_GROUPS = [PIXEL_GROUP == g for g in range(8)]

# Combine recent frames to reveal weak thermal targets
AVG_WINDOW_S = 0.8

# Blob confirmation and immediate strong detection limits
CONFIRM_M = 2
CONFIRM_WINDOW_S = 0.8
STRONG_RATIO = 1.5
CONFIRM_MAX_STEP_PX = 9.0

# Blob tracking and primary track settings
max_match_distance = 9
max_missed_frames = 1
min_primary_track_frames = 2

# Time window used to measure blob movement
TRAVERSE_WINDOW_S = 2.0

# Minimum centroid movement speed
MOVING_SPEED_PX_S = 0.5

# Timing and noise limits for frame differences
FD_LAG_SECONDS = 0.5
FD_NOISE_THRESHOLD_C = 0.25
FD_MAX_GAP_SECONDS = 1.0

# Frame difference feature names
FD_FEATURE_NAMES = [
    "fdMeanAbs", "fdMaxAbs", "fdEnergy", "fdActivePixels",
    "fdWindowMeanEnergy", "fdWindowMaxEnergy",
    "fdWindowActiveFraction", "fdValid",
]

# Gradually absorb persistent warm pixels into the background
ABSORB_STUCK_PIXELS = True
ABSORB_S = 6.0
ALPHA_ABSORB = 0.9
ABSORB_TAIL_S = 5.0

# Exclude tracks that remain almost stationary
STATIC_GATE = True
STATIC_WINDOW_S = 4.0
STATIC_PX = 1.5
STATIC_PIXEL_SHARE = 0.5

# Calculate Median
def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    middle = n // 2
    if n % 2 == 1:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))

# Pad the array by repeating border pixels
def pad_edges(grid):
    rows, cols = grid.shape
    row_index = [0] + list(range(rows)) + [rows - 1]
    col_index = [0] + list(range(cols)) + [cols - 1]
    return grid[np.ix_(row_index, col_index)]

# Expand the mask to include neighbouring pixels
def grow_mask(mask):
    rows, cols = mask.shape
    padded = pad_edges(mask)
    grown = np.zeros((rows, cols), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            grown |= padded[dr:dr + rows, dc:dc + cols]
    return grown

class BackgroundModel:
    # Initialise Background
    def __init__(self, shape):
        self.bg = np.zeros(shape)
        self.sigma = np.full(shape, SIGMA_MIN)
        self.calibration_sigma = np.full(shape, SIGMA_MIN)
        self._initialised = False

    # Set the working noise estimate from calibration
    def _set_sigma(self, calibration_sigma):
        self.calibration_sigma = np.asarray(calibration_sigma, dtype=np.float64).reshape(self.bg.shape).copy()
        sigma = self.calibration_sigma * 0.4
        self.sigma = np.maximum(sigma, SIGMA_MIN)

    # Learn the background from empty frames
    def calibration(self, calibration_frames):
        # Update Array in memory instead of refrencing new array
        self.bg[:] = np.median(calibration_frames, axis=0)
        measured_sigma = calibration_frames.std(axis=0)
        self._set_sigma(measured_sigma)
        self._initialised = True

    # Restore earlier sensor calibration
    def load(self, background, sigma):
        self.bg[:] = np.asarray(background, dtype=np.float64).reshape(self.bg.shape)
        self._set_sigma(sigma)
        self._initialised = True

    # Update only background pixels and absorb persistent objects
    def update(self, frame, foreground_mask, absorb_mask=None):
        # Invert Foreground Mask
        update_mask = ~foreground_mask

        # Slowly learn persistent foreground into the background
        if absorb_mask is not None and absorb_mask.any():
            absorbed_bg = ALPHA_ABSORB * self.bg + (1 - ALPHA_ABSORB) * frame
            self.bg = np.where(absorb_mask, absorbed_bg, self.bg)
            update_mask = update_mask & ~absorb_mask

        # Exponential moving average for background temperature
        new_bg = ALPHA * self.bg + (1 - ALPHA) * frame

        new_sigma_sq = ALPHA * (self.sigma ** 2) + (1 - ALPHA) * (self.bg - frame) ** 2

        self.bg = np.where(update_mask, new_bg, self.bg)
        new_sigma = np.sqrt(new_sigma_sq)
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)
        self.sigma = np.where(update_mask, new_sigma, self.sigma)

    # Estimate the shared temperature offset of each pixel group - Used For scene shifting
    def group_shift(self, frame, previous_mask):
        difference = frame - self.bg

        # Avoid previously detected objects when estimating offsets
        quiet = ~grow_mask(previous_mask)
        shift = np.zeros_like(difference)
        for group in PIXEL_GROUPS:
            values = difference[group & quiet]
            if values.size < 0.25 * np.count_nonzero(group):
                values = difference[group]
            shift[group] = np.median(values)
        return shift

    # Estimate whole scene temperature movement from quiet pixels - used for glitch detection
    def scene_shift(self, frame, previous_mask):
        difference = frame - self.bg

        # Avoid previously detected objects when estimating offsets
        quiet = ~grow_mask(previous_mask)
        if np.count_nonzero(quiet) >= 0.25 * difference.size:
            values = difference[quiet]
        else:
            values = difference.reshape(-1)
        return median(values.tolist())

    # Scene shift
    def apply_shift(self, shift):
        self.bg = self.bg + shift

    # Determine z Value
    def z_score(self, frame):
        return np.abs(self.bg - frame) / self.sigma

    # Average background temperature
    def scene_temperature(self):
        return float(np.mean(self.bg))

# Gaussian Filter
def gaussian_filter(z_grid):
    rows, cols = z_grid.shape
    padded = pad_edges(z_grid)
    smoothed = np.zeros((rows, cols))
    for dr in range(3):
        for dc in range(3):
            smoothed += gaussianKernel[dr, dc] * padded[dr:dr + rows, dc:dc + cols]
    return smoothed

# Calculate the Z score across recent frames
def averaged_z(difference_history, sigma):
    n = len(difference_history)
    total = np.zeros(sigma.shape)
    for _, difference in difference_history:
        total = total + difference
    mean = total / n
    root_n = n ** 0.5
    return np.abs(mean) * root_n / sigma

# Combine the current frame and averaged detectors
def threshold_ratio(smoothed_raw, threshold_raw, smoothed_avg, threshold_avg):
    raw_ratio = smoothed_raw / threshold_raw
    avg_ratio = smoothed_avg / threshold_avg
    return np.where(raw_ratio > avg_ratio, raw_ratio, avg_ratio)


# Store the pixels, centre and bounds of one blob
@dataclass
class Blob:
    id: int
    pixel_count: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    mask: np.ndarray

# Find the blob centre with stronger pixels weighted more
def weighted_centroid(weights, blob_mask):
    sum_w = 0.0
    sum_wr = 0.0
    sum_wc = 0.0
    sum_r = 0.0
    sum_c = 0.0
    count = 0
    rows, cols = np.nonzero(blob_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        w = float(weights[r, c])
        sum_w += w
        sum_wr += w * r
        sum_wc += w * c
        sum_r += r
        sum_c += c
        count += 1
    if count == 0:
        return (0.0, 0.0)
    if sum_w <= 0.0:

        return (sum_r / count, sum_c / count)
    return (sum_wr / sum_w, sum_wc / sum_w)

def centroid(blob_mask):
    rows, cols = np.nonzero(blob_mask)               
    if rows.size == 0:                              
        return (0.0, 0.0)
    return (float(rows.mean()), float(cols.mean()))

# Find maximum displacement over the movement window
def pixels_traversed(current_centre, history):
    largest = 0.0
    for _, centre in history:
        dr = current_centre[0] - centre[0]
        dc = current_centre[1] - centre[1]
        distance = (dr * dr + dc * dc) ** 0.5
        if distance > largest:
            largest = distance
    return largest

# Group touching foreground pixels using flood fill
def label_blobs(binary_mask):
    rows, cols = binary_mask.shape
    labelled_grid = np.zeros((rows, cols), dtype=int)
    num_blobs = 0

    for r in range(rows):
        for c in range(cols):

            # Skip if background or already belongs to a blob
            if not binary_mask[r, c] or labelled_grid[r, c]:
                continue

            num_blobs += 1
            labelled_grid[r, c] = num_blobs
            to_visit = [(r, c)]

            while to_visit:
                # Remove first item in to visit list
                pr, pc = to_visit.pop()

                # Look through every pixel in the 3x3 radius from pixel
                for nr in range(pr - 1, pr + 2):
                    for nc in range(pc - 1, pc + 2):
                        # If it exists, is in the binary mask and is not labelled aready, add to blob
                        if (0 <= nr < rows and 0 <= nc < cols and binary_mask[nr, nc] and not labelled_grid[nr, nc]):
                            labelled_grid[nr, nc] = num_blobs
                            to_visit.append((nr, nc))
    return labelled_grid, num_blobs

# Turn connected components into blob objects
def extract_blobs(binary_mask, weights=None):
    # First Get All Blobs
    labelled_grid, num_blobs = label_blobs(binary_mask)

    blobs: list[Blob] = []
    for blob_id in range(1, num_blobs + 1):
        blob_mask = labelled_grid == blob_id
        pixel_count = int(blob_mask.sum())

        if weights is None:
            centroid = weighted_centroid(blob_mask, blob_mask)
        else:
            centroid = weighted_centroid(weights, blob_mask)
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

# Remember a blob position and history between frames
@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    age_frames: int
    missed_frames: int = 0

    traversed_px: float = 0.0
    history: list = None
    seen_times: list = None
    first_seen: float = None
    long_history: list = None

# Store measurements extracted from a blob
@dataclass
class BlobFeatures:
    track_id: int
    centroid: tuple[float, float]

    pixel_count: int
    track_age_frames: int

    mean_temperature_difference: float
    max_temperature_difference: float

# Extract size, age and temperature features
def score_blob(blob, track, temperature_difference):
    # Create a new track if no previous blob matches
    if track is None:
        track_age_frames = 1
        track_id = -1
    else:
        track_age_frames = track.age_frames + 1
        track_id = track.id

    blob_temperature_difference = temperature_difference[blob.mask]

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

# Find the closest available track within the allowed distance
def match_previous_blob(blob, tracks, used_track_ids, max_match_distance):
    if not tracks:
        return None

    best_dist = None
    best_track = None

    # Check each unused track for the nearest valid match
    for track in tracks.values():
        if track.id in used_track_ids:
            continue

        dr = blob.centroid[0] - track.centroid[0]
        dc = blob.centroid[1] - track.centroid[1]
        dist = float(np.hypot(dr, dc))

        # Maximum distance this blob may have moved since the track's last position - Small/far blobs wont move more than 9 pixels
        # Large/close blobs move further across the image per frame, so the limit grows
        # with blob width: 2.5 x sqrt(pixel count)
        gate = 2.5 * (blob.pixel_count ** 0.5)
        if gate < max_match_distance:
            gate = max_match_distance

        # Find closest track
        if dist <= gate and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_track = track

    return best_track

# Bundle a confirmed blob with its extracted features
@dataclass
class DetectionResult:
    blob: Blob
    features: BlobFeatures

# Return the exact thermal feature order used by the model
def getThermalFeatureNames():
    return [
    "thermalNumberOfBlobs",
    "thermalTotalForegroundPixels",
    "thermalMaxZ",

    "thermalPixelCount",
    "thermalCentroidRow",
    "thermalCentroidCol",
    "thermalTrackAgeFrames",
    "thermalCentroidSpeed",
    "thermalMaxTemperatureDifference",

    "thermalMovingBlobCount",
    "thermalMaxBlobTraversed",
    "thermalMaxBlobSpeed",
    "thermalMaxBlobAgeFrames",
    "thermalMaxBlobPixelCount",
    "thermalMaxBlobTemperatureDifference",
    *["thermal" + name[0].upper() + name[1:] for name in FD_FEATURE_NAMES],
    ]

# Build a feature row with zero values
def emptyThermalFeatures():
    return {name: 0.0 for name in getThermalFeatureNames()}

# Process thermal frames into features for fusion
class ThermopilePipeline:
    # Create background, tracking and frame history state
    def __init__(self, frame_shape=FRAME_SHAPE, window_seconds=2.0):
        self.frame_shape = frame_shape
        self.bg_model = BackgroundModel(frame_shape)
        self._prev_mask = np.zeros(frame_shape, dtype=bool)

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self.primary_track_id = None

        # Keep intermediate arrays for the live display
        self.last_raw_z = np.zeros(frame_shape)
        self.last_smoothed_z = np.zeros(frame_shape)
        self.last_mask = np.zeros(frame_shape, dtype=bool)
        self.last_scene_temp = 0.0
        self.last_shift = 0.0
        self.last_ratio = np.zeros(frame_shape)
        self.last_smoothed_z_avg = np.zeros(frame_shape)
        self.last_detections = []

        # Track sensor glitches and last valid output
        self.last_frame_valid = True
        self.glitch_start = None
        self.glitch_frames = 0

        # Store recent frame changes and motion activity
        self.difference_history = deque(maxlen=16)
        self.fd_frames = deque(maxlen=128)
        self.fd_activity = deque(maxlen=128)
        self.fd_window_seconds = float(window_seconds)
        self.fd_latest = {name: 0.0 for name in FD_FEATURE_NAMES}

        # Remember previous positions for speed calculations
        self._track_positions = {}
        self._last_good_features = None

        # Track persistent warm pixels for background absorption
        self.fg_time = np.zeros(frame_shape)
        self._last_frame_time = None
        self.absorb_until = np.full(frame_shape, -np.inf)

    # Clear tracking history while keeping calibration
    def reset(self):
        self._prev_mask = np.zeros(self.frame_shape, dtype=bool)
        self._tracks = {}
        self._next_track_id = 1
        self.primary_track_id = None
        self.fd_frames.clear()
        self.fd_activity.clear()
        self.fd_latest = {name: 0.0 for name in FD_FEATURE_NAMES}

        # Track sensor glitches and last valid output
        self.last_frame_valid = True
        self.glitch_start = None
        self.glitch_frames = 0
        self.difference_history.clear()

        # Remember previous positions for speed calculations
        self._track_positions = {}
        self._last_good_features = None
        self.fg_time = np.zeros(self.frame_shape)
        self._last_frame_time = None
        self.absorb_until = np.full(self.frame_shape, -np.inf)

    # Calibrate from a fresh set of thermal frames
    def calibrate(self, calibration_frames):
        frames = np.asarray(calibration_frames, dtype=np.float64).reshape(-1, *self.frame_shape)
        self.bg_model.calibration(frames)
        self.reset()

    # Load recorded calibration for offline processing
    def load_calibration(self, background, sigma):
        self.bg_model.load(background, sigma)
        self.reset()

    # Calculate frame difference features without crossing data gaps
    def update_fd(self, frame, timestamp):
        timestamp = float(timestamp)

        # Reset history if timestamps are out of order
        if self.fd_frames and timestamp <= self.fd_frames[-1][0]:
            self.fd_frames.clear()
            self.fd_activity.clear()

        # Reset history after missing thermal frames
        if self.fd_frames and timestamp - self.fd_frames[-1][0] > FD_MAX_GAP_SECONDS:
            self.fd_frames.clear()
            self.fd_activity.clear()

        filtered = np.asarray(frame, dtype=np.float64)
        
        # Find the frame closest to the required time offset
        target = timestamp - FD_LAG_SECONDS
        previous = min(self.fd_frames, key=lambda item: abs(item[0] - target), default=None)
        data = {name: 0.0 for name in FD_FEATURE_NAMES}

        # Only compare frames with an acceptable time separation
        if previous is not None and previous[0] < timestamp and abs(previous[0] - target) <= 0.20:

            # Remove shared readout changes before smoothing
            change = filtered - previous[1]
            for group in PIXEL_GROUPS:
                change[group] = change[group] - np.median(change[group])
            absdiff = np.abs(gaussian_filter(change))
            noise_threshold = 0.6

            # Measure frame difference strength and active pixel count
            data.update(fdMeanAbs=float(absdiff.mean()), fdMaxAbs=float(absdiff.max()),
                        fdEnergy=float(np.mean(absdiff ** 2)),
                        fdActivePixels=float(np.count_nonzero(absdiff > noise_threshold)),
                        fdValid=1.0)
            self.fd_activity.append((timestamp, data["fdEnergy"], data["fdActivePixels"]))

        # Keep only recent frame history
        self.fd_frames.append((timestamp, filtered))
        while self.fd_frames and timestamp - self.fd_frames[0][0] > max(self.fd_window_seconds + FD_LAG_SECONDS, 3.0):
            self.fd_frames.popleft()
        while self.fd_activity and timestamp - self.fd_activity[0][0] > self.fd_window_seconds:
            self.fd_activity.popleft()

        # Summarise movement over the full feature window
        if self.fd_activity:
            energies = [v[1] for v in self.fd_activity]
            data["fdWindowMeanEnergy"] = float(np.mean(energies))
            data["fdWindowMaxEnergy"] = float(max(energies))
            data["fdWindowActiveFraction"] = float(np.mean([v[2] > 0 for v in self.fd_activity]))
        self.fd_latest = data

    # Run one thermal frame through the complete detection pipeline
    def process_frame(self, frame, timestamp):
        now = float(timestamp)
        frame = np.asarray(frame, dtype=np.float64).reshape(self.frame_shape)

        # Estimate the temperature shift since the background
        shift = self.bg_model.scene_shift(frame, self._prev_mask)
        self.last_shift = shift

        # Detect sudden global jumps and wait for stable frames
        if self.glitch_start is None:
            if abs(shift) > GLITCH_SHIFT_C:
                self.glitch_start = now
        else:
            if abs(shift) <= GLITCH_SHIFT_C:
                self.glitch_start = None
            elif now - self.glitch_start > GLITCH_MAX_S:
                self.glitch_start = None
                self.difference_history.clear()
                self.fd_frames.clear()
                self.fd_activity.clear()

        # Discard invalid frames without updating the background
        if self.glitch_start is not None:
            self.last_frame_valid = False
            self.glitch_frames += 1
            return []

        # Track sensor glitches and last valid output
        self.last_frame_valid = True

        # Update motion features from valid frame differences
        self.update_fd(frame, now)

        # Correct background using the shared offset of pixel groups
        self.bg_model.apply_shift(self.bg_model.group_shift(frame, self._prev_mask))

        # Calculate pixel deviation from the background
        z = self.bg_model.z_score(frame)

        # Reduce isolated pixel noise with Gaussian smoothing
        smoothed_z = gaussian_filter(z)

        # Average corrected changes to detect weak targets
        difference = frame - self.bg_model.bg
        self.difference_history.append((now, difference))
        while self.difference_history and now - self.difference_history[0][0] > AVG_WINDOW_S:
            self.difference_history.popleft()
        smoothed_z_avg = gaussian_filter(averaged_z(self.difference_history, self.bg_model.sigma))

        # Select fixed or temperature based detection limits
        scene_temp = self.bg_model.scene_temperature()

        threshold_raw = np.full(self.frame_shape, THRESHOLD)
        threshold_avg = np.full(self.frame_shape, 3.45)

        # Mark a pixel when either detector passes its threshold
        ratio = threshold_ratio(smoothed_z, threshold_raw, smoothed_z_avg, threshold_avg)
        mask = ratio > 1.0

        temperature_difference = np.abs(difference)

        # Measure how long foreground pixels remain active
        dt = 0.0
        if self._last_frame_time is not None:
            dt = min(max(now - self._last_frame_time, 0.0), 1.0)
        self._last_frame_time = now
        self.fg_time = np.where(mask, self.fg_time + dt, 0.0)
        self.absorb_until = np.where(self.fg_time > ABSORB_S, now + ABSORB_TAIL_S, self.absorb_until)
        stuck = now <= self.absorb_until

        # Update the background while absorbing persistent hot pixels
        if ABSORB_STUCK_PIXELS:
            self.bg_model.update(frame, mask, stuck)
        else:
            self.bg_model.update(frame, mask)

        # Extract blobs and calculate their weighted centres
        blobs = extract_blobs(mask, weights=ratio)

        # Prepare confirmed detections and updated tracks
        results = []
        used_track_ids = set()
        updated_tracks = {}

        # Process and match each detected blob
        for blob in blobs:

            # Match the blob to a track from an earlier frame
            track = match_previous_blob(
                blob,
                self._tracks,
                used_track_ids,
                max_match_distance
            )

            # Create a new track if no previous blob matches
            if track is None:
                track = Track(
                    id=self._next_track_id,
                    centroid=blob.centroid,
                    age_frames=0,
                )

                self._next_track_id += 1

            # Calculate size, age and temperature measurements
            features = score_blob(blob, track, temperature_difference)

            features.track_id = track.id

            # Update movement history and distance travelled
            history = list(track.history) if track.history is not None else []
            history.append((now, blob.centroid))
            history = [item for item in history if now - item[0] <= TRAVERSE_WINDOW_S]
            traversed = pixels_traversed(blob.centroid, history)

            # Keep recent sightings for blob confirmation
            seen_times = list(track.seen_times) if track.seen_times is not None else []
            step_dr = blob.centroid[0] - track.centroid[0]
            step_dc = blob.centroid[1] - track.centroid[1]

            # Allow fast blobs a larger confirmation distance
            confirm_step = CONFIRM_MAX_STEP_PX
            confirm_step = max(CONFIRM_MAX_STEP_PX, 2.5 * (blob.pixel_count ** 0.5))
            if (step_dr * step_dr + step_dc * step_dc) ** 0.5 > confirm_step:
                seen_times = []
            seen_times.append(now)
            seen_times = [t for t in seen_times if now - t <= CONFIRM_WINDOW_S]

            # Check whether the blob has remained stationary
            first_seen = track.first_seen if track.first_seen is not None else now
            long_history = list(track.long_history) if track.long_history is not None else []
            long_history.append((now, blob.centroid))
            long_history = [item for item in long_history if now - item[0] <= STATIC_WINDOW_S]
            is_static = (now - first_seen >= STATIC_WINDOW_S and
                         pixels_traversed(blob.centroid, long_history) < STATIC_PX)
            if not is_static:
                stuck_share = np.count_nonzero(blob.mask & stuck) / blob.pixel_count
                is_static = stuck_share >= STATIC_PIXEL_SHARE

            # Save the latest position and history of this track
            updated_tracks[track.id] = Track(
                id=track.id,
                centroid=blob.centroid,
                age_frames=features.track_age_frames,
                missed_frames=0,
                traversed_px=traversed,
                history=history,
                seen_times=seen_times,
                first_seen=first_seen,
                long_history=long_history,
            )

            used_track_ids.add(track.id)

            # Confirm strong blobs immediately or faint blobs after repeated sightings
            peak_ratio = float(np.max(ratio[blob.mask]))
            confirmed = peak_ratio >= STRONG_RATIO or len(seen_times) >= CONFIRM_M

            # Suppress stationary blobs from the reported detections
            if STATIC_GATE and is_static:
                confirmed = False

            # Include only confirmed moving detections
            if confirmed:
                results.append(DetectionResult(blob=blob,features=features))

        # Retain recently missed tracks and remove older ones
        for track_id, track in self._tracks.items():
            if track_id in used_track_ids:
                continue

            track.missed_frames += 1

            if track.missed_frames <= max_missed_frames:
                updated_tracks[track_id] = track

        self._tracks = updated_tracks

        # Choose a primary blob from sufficiently established tracks
        eligible_results = [
            result for result in results
            if result.features.track_age_frames >= min_primary_track_frames
        ]

        # Prefer the track with the greatest measured movement
        if eligible_results:
            primary_result = max(
                eligible_results,
                key=lambda result: (
                    updated_tracks[result.features.track_id].traversed_px,
                    result.features.track_id == self.primary_track_id,
                    result.features.pixel_count,
                    result.features.mean_temperature_difference,
                ),
            )
            self.primary_track_id = primary_result.features.track_id
        else:
            self.primary_track_id = None

        # Save the current mask for the next frame
        self._prev_mask = mask

        # Expose intermediate processing stages for the display
        self.last_raw_z = z
        self.last_smoothed_z = smoothed_z
        self.last_smoothed_z_avg = smoothed_z_avg
        self.last_mask = mask
        self.last_threshold_map = threshold_raw
        self.last_threshold = float(np.mean(threshold_raw))
        self.last_ratio = ratio
        self.last_scene_temp = scene_temp

        return results

    # Return features from one complete thermal frame
    def process(self, frame, timestamp):
        timestamp = float(timestamp)

        # Run detection and store the current blobs
        detections = self.process_frame(frame, timestamp)
        self.last_detections = detections

        # Reuse recent valid features during brief sensor glitches
        if not self.last_frame_valid:
            last_good = self._last_good_features
            if last_good is not None and timestamp - last_good[0] <= FEATURE_HOLD_S:
                return dict(last_good[1])
            features = emptyThermalFeatures()
            features["thermalSceneTemperature"] = float(self.last_scene_temp)
            return features

        # Calculate current features and save them for recovery
        features = self.frame_features(detections, timestamp)
        self._last_good_features = (timestamp, features)
        return dict(features)

    # Convert confirmed blobs into the classifier feature vector
    def frame_features(self, detections, timestamp):
        features = emptyThermalFeatures()
        features["thermalNumberOfBlobs"] = float(len(detections))
        features["thermalTotalForegroundPixels"] = float(sum(d.features.pixel_count for d in detections))

        # Only consider moving blobs when measuring maximum Z
        if STATIC_GATE:

            features["thermalMaxZ"] = float(max((np.max(self.last_smoothed_z[d.blob.mask]) for d in detections),
                                                default=0.0))
        else:
            features["thermalMaxZ"] = float(np.max(self.last_smoothed_z))
        features["thermalSceneTemperature"] = float(self.last_scene_temp)

        # Add recent frame difference features
        for name, value in self.fd_latest.items():
            features["thermal" + name[0].upper() + name[1:]] = value

        # Calculate speed from each track last reported position
        speeds = {}

        # Fill the primary blob features when its track is found
        for detection in detections:
            track_id = int(detection.features.track_id)
            centroid_row, centroid_col = detection.features.centroid
            speed = 0.0
            if track_id in self._track_positions:
                previous_time, previous_row, previous_col = self._track_positions[track_id]
                dt = timestamp - previous_time
                if dt > 0:
                    speed = float(np.hypot(centroid_row - previous_row, centroid_col - previous_col) / dt)
            speeds[track_id] = speed
            self._track_positions[track_id] = (timestamp, float(centroid_row), float(centroid_col))

        # Summarise movement, size and age across all blobs
        if detections:
            ids = [int(d.features.track_id) for d in detections]
            features["thermalMovingBlobCount"] = float(sum(speeds[i] > MOVING_SPEED_PX_S for i in ids))
            features["thermalMaxBlobTraversed"] = float(max(self._tracks[i].traversed_px for i in ids))
            features["thermalMaxBlobSpeed"] = float(max(speeds[i] for i in ids))
            features["thermalMaxBlobAgeFrames"] = float(max(d.features.track_age_frames for d in detections))
            features["thermalMaxBlobPixelCount"] = float(max(d.features.pixel_count for d in detections))
            features["thermalMaxBlobTemperatureDifference"] = float(max(
                d.features.max_temperature_difference for d in detections
            ))

        # Fill the primary blob features when its track is found
        for detection in detections:
            if detection.features.track_id == self.primary_track_id:
                primary = detection.features
                features["thermalPixelCount"] = float(primary.pixel_count)
                features["thermalCentroidRow"] = float(primary.centroid[0])
                features["thermalCentroidCol"] = float(primary.centroid[1])
                features["thermalTrackAgeFrames"] = float(primary.track_age_frames)
                features["thermalCentroidSpeed"] = speeds[int(primary.track_id)]
                features["thermalMaxTemperatureDifference"] = float(primary.max_temperature_difference)
                break

        return features

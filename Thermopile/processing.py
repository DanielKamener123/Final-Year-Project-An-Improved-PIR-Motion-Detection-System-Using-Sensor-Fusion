from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy import ndimage

# Config
ALPHA = 0.995                 
# Fix 2: realistic noise floor. Empty-scene noise in the recordings is
# 0.2-0.9 C; one dead pixel per recording otherwise sits at a tiny sigma and
# produces huge Z values.
SIGMA_MIN = 0.15 # Minimum Sigma (C)

# Fix 2 - Whole-scene shift removal (Trofimova et al., 2017)
# Pixels within 1 pixel of last frame's foreground are not used to measure
# the shift. If too few pixels remain, all pixels are used instead.
MIN_QUIET_FRACTION = 0.25

# Spatial filter
centerWeight = 8.0

gaussianKernel = np.array([
    [1, 2, 1],
    [2, centerWeight, 2],
    [1, 2, 1],
])

gaussianKernel /= gaussianKernel.sum()
filterMode = "reflect"

# Threshold (legacy single value; kept for the debug display and as fallback)
threshold = 2

# Fix 3 - Glitch guard
# Real scenes never shifted more than 2.6 C between frames in the recordings;
# the CefimTraining2 sensor glitch jumped 6.8 C in one frame.
GLITCH_SHIFT_C = 3.0      # Whole-scene jump that marks a glitch frame
GLITCH_MAX_S = 30.0       # Longer than this: accept the change as real
FEATURE_HOLD_S = 1.0      # Repeat last good thermal features for at most this long

# Fix 4 - Adaptive threshold per pixel from that pixel's background temperature
# (after Cerutti et al., who make the threshold a function of background
# temperature). Values were measured on the TRAINING recordings' empty frames:
# for each 2 C band of background temperature, the threshold that empty pixels
# cross 2.88% of the time (the rate the old fixed threshold of 2 gave
# overall). A quadratic (Cerutti) fitted the hot end poorly, so a lookup table
# with straight lines between bands is used. Outside the table: nearest end.
# Cold backgrounds (e.g. sky) are very noisy -> high thresholds; typical
# 23-33 C backgrounds get thresholds slightly below the old 2.
THRESHOLD_FLOOR = 2.0     # Thresholds are only ever RAISED above the old value
THRESHOLD_TABLE_RAW = [
    (13.0, 15.49),
    (15.0, 9.11),
    (17.0, 6.85),
    (19.0, 4.29),
    (21.0, 2.14),
    (23.0, 1.87),
    (25.0, 1.70),
    (27.0, 1.76),
    (29.0, 1.79),
    (31.0, 1.86),
    (33.0, 1.69),
]
THRESHOLD_TABLE_AVG = [
    (13.0, 21.90),
    (15.0, 12.86),
    (17.0, 9.67),
    (19.0, 5.98),
    (21.0, 2.72),
    (23.0, 2.11),
    (25.0, 1.78),
    (27.0, 1.83),
    (29.0, 1.91),
    (31.0, 2.28),
    (33.0, 1.97),
]

# Fix 5 - Second detector on the average of recent frames (Trofimova et al.
# filter each pixel over time). Noise shrinks by sqrt(n); a slow, faint person
# does not. A pixel is foreground if EITHER detector fires.
AVG_WINDOW_S = 0.6        # 3 frames at 4 Hz, 2 frames at 2 Hz

# Fix 6 - Confirmation (M-of-N): a blob is only reported if its track was seen
# in at least CONFIRM_M frames within CONFIRM_WINDOW_S, unless it is strong.
CONFIRM_M = 2
CONFIRM_WINDOW_S = 0.6
STRONG_RATIO = 1.5        # Peak score >= 1.5 x its threshold -> report immediately
CONFIRM_MAX_STEP_PX = 3.0 # A faint blob only counts as "seen again" if it is
                          # within this distance of its track's last centre

# Fix 7 - Tracker gate grows with blob size (near people are big and fast)
GATE_PER_SQRT_PX = 2.5    # gate = max(max_match_distance, 2.5 * sqrt(pixels))

# Blob extraction - TODO: Remove
connectivity_8 = True

# Feature scoring
max_match_distance = 9
max_missed_frames = 1
min_primary_track_frames = 2

# Fix 1 - Motion = pixels traversed (Narayana et al., LOCI) in a sliding window.
# For each track: the largest distance between its current centre and any
# centre it had in the last TRAVERSE_WINDOW_S seconds. Centroid jitter stays
# bounded, real walking grows, and a blob that stops decays to ~0.
# Measured in seconds (not frames) because recordings use 2 Hz and 4 Hz.
TRAVERSE_WINDOW_S = 2.0

# Difference compared against the most recent frame about 0.5 seconds earlier.
FD_LAG_SECONDS = 0.5
FD_NOISE_THRESHOLD_C = 0.25  # Initial value; calibrate using empty-scene recordings.
FD_MAX_GAP_SECONDS = 1.0     # Do not difference across sensor outages.
FD_WINDOW_SECONDS = None     # Supplied by dataset/live caller; defaults to 2 s.
FD_FEATURE_NAMES = [
    "fd_mean_abs", "fd_max_abs", "fd_energy", "fd_active_pixels",
    "fd_window_mean_energy", "fd_window_max_energy",
    "fd_window_active_fraction", "fd_valid",
]

# Fix 2 - Median (first principles): sort and take the middle value
def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    middle = n // 2
    if n % 2 == 1:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))


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

        # Update Sigma Where No Human
        new_sigma_sq = ALPHA * (self.sigma ** 2) + (1 - ALPHA) * (self.bg - frame) ** 2

        self.bg = np.where(update_mask, new_bg, self.bg)
        new_sigma = np.sqrt(new_sigma_sq)
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)
        self.sigma = np.where(update_mask, new_sigma, self.sigma)

    # Fix 2 - Measure The Whole-Scene Shift (first principles)
    # shift = median of (frame - background) over "quiet" pixels: pixels that
    # were not foreground last frame and have no foreground 8-neighbour.
    def scene_shift(self, frame, previous_mask):
        rows, cols = frame.shape
        quiet_values = []
        all_values = []
        for r in range(rows):
            for c in range(cols):
                difference = float(frame[r, c] - self.bg[r, c])
                all_values.append(difference)
                near_foreground = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < rows and 0 <= cc < cols and previous_mask[rr, cc]:
                            near_foreground = True
                if not near_foreground:
                    quiet_values.append(difference)
        if len(quiet_values) >= MIN_QUIET_FRACTION * rows * cols:
            values = quiet_values
        else:
            values = all_values
        return median(values)

    # Fix 2 - Move The Whole Background With The Scene (every pixel,
    # including pixels currently under a person)
    def apply_shift(self, shift):
        self.bg = self.bg + shift

    # Calculate Z value - How much variation is present
    def z_score(self, frame):
        return np.abs(self.bg - frame) / self.sigma

    # Calculate Mean Background Temp
    def scene_temperature(self):
        return float(np.mean(self.bg))

# Spatial filtering
def gaussian_filter(z_grid):
    #return z_grid
    return ndimage.convolve(z_grid, gaussianKernel, mode=filterMode)

def adaptive_threshold_map(background):
    centre_temp = 28.0
    min_threshold = 1.5
    max_threshold = 3.0
    curvature = 0.01

    threshold_map = (
        min_threshold
        + curvature * (background - centre_temp)**2
    )

    return np.clip(
        threshold_map,
        min_threshold,
        max_threshold
    )

# Apply Threshold To Get Mask
def apply_threshold(smoothed_z, threshold_value=threshold):
    return smoothed_z > threshold_value


# Fix 4 - Threshold Map From Each Pixel's Background Temperature (first principles)
# Linear interpolation in the (background temperature, threshold) table.
def threshold_from_table(background, table):
    rows, cols = background.shape
    threshold_map = np.empty((rows, cols))
    for r in range(rows):
        for col in range(cols):
            bg = float(background[r, col])
            if bg <= table[0][0]:
                value = table[0][1]
            elif bg >= table[-1][0]:
                value = table[-1][1]
            else:
                value = table[-1][1]
                for k in range(len(table) - 1):
                    bg0, t0 = table[k]
                    bg1, t1 = table[k + 1]
                    if bg0 <= bg <= bg1:
                        value = t0 + (t1 - t0) * (bg - bg0) / (bg1 - bg0)
                        break
            # Never go below the old fixed threshold: noisy sites (e.g. the
            # 4 Hz Cefim recordings) crossed the measured mid-range values
            # 4-5% of the time, so lowering it is not justified by the data.
            if value < THRESHOLD_FLOOR:
                value = THRESHOLD_FLOOR
            threshold_map[r, col] = value
    return threshold_map


# Fix 5 - Z Score Of The Average Of The Recent Corrected Differences
# The mean of n frames has noise sigma / sqrt(n), so the Z is scaled by sqrt(n).
def averaged_z(difference_history, sigma):
    n = len(difference_history)
    total = np.zeros(sigma.shape)
    for _, difference in difference_history:
        total = total + difference
    mean = total / n
    rows, cols = sigma.shape
    z = np.empty((rows, cols))
    root_n = n ** 0.5
    for r in range(rows):
        for c in range(cols):
            z[r, c] = abs(mean[r, c]) * root_n / sigma[r, c]
    return z


# Fix 4/5 - Combined Score: how many times over its own threshold each pixel is
# (>1 means foreground). Used for the mask, blob strength and blob centres.
def threshold_ratio(smoothed_raw, threshold_raw, smoothed_avg, threshold_avg):
    rows, cols = smoothed_raw.shape
    ratio = np.empty((rows, cols))
    for r in range(rows):
        for c in range(cols):
            raw_ratio = smoothed_raw[r, c] / threshold_raw[r, c]
            avg_ratio = smoothed_avg[r, c] / threshold_avg[r, c]
            ratio[r, c] = raw_ratio if raw_ratio > avg_ratio else avg_ratio
    return ratio

# Blob extraction
@dataclass
class Blob:
    id: int
    pixel_count: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]   # (min_row, min_col, max_row, max_col)
    mask: np.ndarray                  

# Fix 1 - Weighted Centre Of A Blob (first principles)
# centre = (sum(w*row) / sum(w), sum(w*col) / sum(w)), with w = smoothed Z.
# Strong pixels pull the centre towards them, so the centre moves in smooth
# sub-pixel steps instead of snapping between pixel centres.
def weighted_centroid(weights, blob_mask):
    rows, cols = blob_mask.shape
    sum_w = 0.0
    sum_wr = 0.0
    sum_wc = 0.0
    sum_r = 0.0
    sum_c = 0.0
    count = 0
    for r in range(rows):
        for c in range(cols):
            if blob_mask[r, c]:
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
        # No usable weights: fall back to the plain average position
        return (sum_r / count, sum_c / count)
    return (sum_wr / sum_w, sum_wc / sum_w)


# Fix 1 - Pixels Traversed Within The Window (first principles)
# history: list of (timestamp, centre). Returns the largest distance between
# the current centre and any centre in the history.
def pixels_traversed(current_centre, history):
    largest = 0.0
    for _, centre in history:
        dr = current_centre[0] - centre[0]
        dc = current_centre[1] - centre[1]
        distance = (dr * dr + dc * dc) ** 0.5
        if distance > largest:
            largest = distance
    return largest


# Extract Blobs
# weights: per-pixel smoothed Z used for the weighted centre (Fix 1)
def extract_blobs(binary_mask, weights=None):
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

@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    age_frames: int
    missed_frames: int = 0
    # Fix 1: now holds PIXELS TRAVERSED in the last TRAVERSE_WINDOW_S seconds
    # (name kept so the feature names and exported model format are unchanged)
    cumulative_displacement_px: float = 0.0
    history: list = None  # (timestamp, centre) pairs inside the window
    seen_times: list = None  # Fix 6: timestamps this track was detected

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

        # Fix 7: bigger (nearer) blobs may jump further between frames
        gate = GATE_PER_SQRT_PX * (blob.pixel_count ** 0.5)
        if gate < max_match_distance:
            gate = max_match_distance

        if dist <= gate and (best_dist is None or dist < best_dist):
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
        self.last_shift = 0.0
        self.last_threshold_map = np.full(frame_shape, float(threshold))
        self.last_ratio = np.zeros(frame_shape)
        self.last_smoothed_z_avg = np.zeros(frame_shape)
        # Fix 3: glitch state
        self.last_frame_valid = True
        self.glitch_start = None
        self.glitch_frames = 0
        # Fix 5: recent shift-corrected differences (timestamp, frame - bg)
        self.difference_history = deque(maxlen=16)
        self.fd_frames = deque(maxlen=128)
        self.fd_activity = deque(maxlen=128)
        self.fd_window_seconds = 2.0
        self.fd_latest = {name: 0.0 for name in FD_FEATURE_NAMES}

    # Get Background Without Human
    def warm_start(self, calibration_frames):
        self.bg_model.warm_start(calibration_frames)
        self._prev_mask = np.zeros(self.frame_shape, dtype=bool)
        self._prev_blobs = []
        self._tracks = {}
        self._next_track_id = 1
        self.primary_track_id = None
        self.fd_frames.clear()
        self.fd_activity.clear()
        self.fd_latest = {name: 0.0 for name in FD_FEATURE_NAMES}
        self.last_frame_valid = True
        self.glitch_start = None
        self.glitch_frames = 0
        self.difference_history.clear()

    def update_fd(self, frame, timestamp):
        """Causal FD, never bridges an acquisition outage."""
        timestamp = float(timestamp)
        if self.fd_frames and timestamp <= self.fd_frames[-1][0]:
            self.fd_frames.clear()
            self.fd_activity.clear()
        if self.fd_frames and timestamp - self.fd_frames[-1][0] > FD_MAX_GAP_SECONDS:
            self.fd_frames.clear()
            self.fd_activity.clear()
        filtered = gaussian_filter(np.asarray(frame, dtype=np.float64))
        target = timestamp - FD_LAG_SECONDS
        previous = min(self.fd_frames, key=lambda item: abs(item[0] - target), default=None)
        data = {name: 0.0 for name in FD_FEATURE_NAMES}
        if previous is not None and previous[0] < timestamp and abs(previous[0] - target) <= 0.20:
            absdiff = np.abs(filtered - previous[1])
            data.update(fd_mean_abs=float(absdiff.mean()), fd_max_abs=float(absdiff.max()),
                        fd_energy=float(np.mean(absdiff ** 2)),
                        fd_active_pixels=float(np.count_nonzero(absdiff > FD_NOISE_THRESHOLD_C)),
                        fd_valid=1.0)
            self.fd_activity.append((timestamp, data["fd_energy"], data["fd_active_pixels"]))
        self.fd_frames.append((timestamp, filtered))
        while self.fd_frames and timestamp - self.fd_frames[0][0] > max(self.fd_window_seconds + FD_LAG_SECONDS, 3.0):
            self.fd_frames.popleft()
        while self.fd_activity and timestamp - self.fd_activity[0][0] > self.fd_window_seconds:
            self.fd_activity.popleft()
        if self.fd_activity:
            energies = [v[1] for v in self.fd_activity]
            data["fd_window_mean_energy"] = float(np.mean(energies))
            data["fd_window_max_energy"] = float(max(energies))
            data["fd_window_active_fraction"] = float(np.mean([v[2] > 0 for v in self.fd_activity]))
        self.fd_latest = data

    def process_frame(self, frame, timestamp=None):
        if timestamp is not None:
            now = float(timestamp)
        else:
            # Debug path without timestamps: count frames at the nominal rate
            self._frame_count = getattr(self, "_frame_count", 0) + 1
            now = self._frame_count / REFRESH_RATE_HZ

        # Fix 2: measure the whole-scene shift against the current background
        shift = self.bg_model.scene_shift(frame, self._prev_mask)
        self.last_shift = shift

        # Fix 3: glitch guard. During a glitch the background is NOT moved and
        # nothing is learned; the frame produces no detections.
        if self.glitch_start is None:
            if abs(shift) > GLITCH_SHIFT_C:
                self.glitch_start = now
        else:
            if abs(shift) <= GLITCH_SHIFT_C:
                self.glitch_start = None                 # back to normal
            elif now - self.glitch_start > GLITCH_MAX_S:
                self.glitch_start = None                 # accept as a real change
                self.difference_history.clear()
                self.fd_frames.clear()
                self.fd_activity.clear()
        if self.glitch_start is not None:
            self.last_frame_valid = False
            self.glitch_frames += 1
            return []
        self.last_frame_valid = True

        # Frame-difference features use valid frames only
        if timestamp is not None:
            self.update_fd(frame, timestamp)

        # Fix 2: move the background with the scene. The Z score, temperature
        # difference and sigma update below all use the corrected difference.
        self.bg_model.apply_shift(shift)

        # Step 2: Z Score (detector A: current frame)
        z = self.bg_model.z_score(frame)

        # Step 3: Spatial Filtering
        smoothed_z = gaussian_filter(z)

        # Fix 5: detector B on the average of the recent corrected differences
        difference = frame - self.bg_model.bg
        self.difference_history.append((now, difference))
        while self.difference_history and now - self.difference_history[0][0] > AVG_WINDOW_S:
            self.difference_history.popleft()
        smoothed_z_avg = gaussian_filter(averaged_z(self.difference_history, self.bg_model.sigma))

        # Step 4 / Fix 4: each pixel's threshold from its own background temperature
        scene_temp = self.bg_model.scene_temperature()
        threshold_raw = threshold_from_table(self.bg_model.bg, THRESHOLD_TABLE_RAW)
        threshold_avg = threshold_from_table(self.bg_model.bg, THRESHOLD_TABLE_AVG)
        ratio = threshold_ratio(smoothed_z, threshold_raw, smoothed_z_avg, threshold_avg)
        mask = ratio > 1.0                     # foreground if EITHER detector fires

        temperature_difference = np.abs(difference)

        # Steps 1: Update Background + Sigma
        self.bg_model.update(frame, mask)

        # Step 5: Blob extraction (centre weighted by how far over threshold)
        blobs = extract_blobs(mask, weights=ratio)

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

            # Fix 1: add this centre to the track history, drop entries older
            # than the window, then measure pixels traversed. A new track has
            # an empty history, so it cannot inherit movement.
            history = list(track.history) if track.history is not None else []
            history.append((now, blob.centroid))
            history = [item for item in history if now - item[0] <= TRAVERSE_WINDOW_S]
            traversed = pixels_traversed(blob.centroid, history)

            # Fix 6: remember when this track was seen
            seen_times = list(track.seen_times) if track.seen_times is not None else []
            step_dr = blob.centroid[0] - track.centroid[0]
            step_dc = blob.centroid[1] - track.centroid[1]
            if (step_dr * step_dr + step_dc * step_dc) ** 0.5 > CONFIRM_MAX_STEP_PX:
                seen_times = []           # jumped too far: not the same faint object
            seen_times.append(now)
            seen_times = [t for t in seen_times if now - t <= CONFIRM_WINDOW_S]

            # Update Data From Tracked Blob
            updated_tracks[track.id] = Track(
                id=track.id,
                centroid=blob.centroid,
                age_frames=features.track_age_frames,
                missed_frames=0,
                cumulative_displacement_px=traversed,
                history=history,
                seen_times=seen_times,
            )

            used_track_ids.add(track.id)

            # Fix 6: report strong blobs immediately; faint blobs only once
            # their track has been seen CONFIRM_M times in CONFIRM_WINDOW_S.
            peak_ratio = 0.0
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if blob.mask[r, c] and ratio[r, c] > peak_ratio:
                        peak_ratio = ratio[r, c]
            confirmed = peak_ratio >= STRONG_RATIO or len(seen_times) >= CONFIRM_M

            # Store The Result For This Blob
            if confirmed:
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
        self.last_smoothed_z_avg = smoothed_z_avg
        self.last_threshold_map = threshold_raw
        self.last_threshold = float(np.mean(threshold_raw))
        self.last_ratio = ratio
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
    *FD_FEATURE_NAMES,
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


# Fix 3 - Feature Row For A Glitch Frame
# Repeat the last good thermal feature row for up to FEATURE_HOLD_S seconds,
# then report an empty row (no blobs). last_good = (timestamp, row) or None.
def glitchFrameRow(timestamp, last_good, scene_temperature):
    if last_good is not None and float(timestamp) - last_good[0] <= FEATURE_HOLD_S:
        return dict(last_good[1])
    return {
        "detections": [],
        "number_of_blobs": 0,
        "total_foreground_pixels": 0.0,
        "primary_track_id": None,
        "max_z": 0.0,
        "scene_temperature": float(scene_temperature),
        "fd_features": {name: 0.0 for name in FD_FEATURE_NAMES},
    }


def processThermopile(thermalData, metadata, windowSeconds=2.0):
    frameShape = tuple(int(value) for value in metadata["thermal_frame_shape"])
    frameLength = int(np.prod(frameShape))
    pixelColumns = [f"pixel_{i}" for i in range(frameLength)]

    missing = [column for column in pixelColumns if column not in thermalData.columns]
    if missing:
        raise ValueError(f"Thermopile CSV is missing {len(missing)} pixel columns")

    pipeline = ThermopilePipeline(frame_shape=frameShape)
    pipeline.fd_window_seconds = float(windowSeconds)
    background = np.asarray(metadata["thermal_background"], dtype=np.float64).reshape(frameShape)
    sigma = np.asarray(metadata["thermal_sigma"], dtype=np.float64).reshape(frameShape)
    pipeline.bg_model.bg[:] = background
    # Fix 2: apply the same sigma floor the ROCK applies during calibration
    pipeline.bg_model.sigma[:] = np.maximum(sigma, SIGMA_MIN)
    pipeline.bg_model._initialised = True

    timestamps = thermalData["time_s"].to_numpy(dtype=np.float64)
    frameRows = []
    previousTrackPositions = {}
    lastGoodRow = None  # Fix 3

    for frameIndex, (_, row) in enumerate(thermalData.iterrows()):
        frame = row[pixelColumns].to_numpy(dtype=np.float64).reshape(frameShape)
        timestamp = float(timestamps[frameIndex])
        detections = pipeline.process_frame(frame, timestamp)
        # Fix 3: glitch frame -> repeat last good features (max 1 s) or empty
        if not pipeline.last_frame_valid:
            frameRows.append(glitchFrameRow(timestamp, lastGoodRow, pipeline.last_scene_temp))
            continue
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

        frameRow = {
            "detections": flattened,
            "number_of_blobs": len(flattened),
            "total_foreground_pixels": float(sum(detection["pixel_count"] for detection in flattened)),
            "primary_track_id": pipeline.primary_track_id,
            "max_z": float(np.max(flatZ)),
            "scene_temperature": float(pipeline.last_scene_temp),
            "fd_features": dict(pipeline.fd_latest),
        }
        frameRows.append(frameRow)
        lastGoodRow = (timestamp, frameRow)

    return timestamps, frameRows


def getThermalFeatures(thermalRows, thermalIndex):
    features = emptyThermalFeatures()
    currentFrame = thermalRows[thermalIndex]
    selectedTrackId = currentFrame["primary_track_id"]

    features["number_of_blobs"] = float(currentFrame["number_of_blobs"])
    features["total_foreground_pixels"] = float(currentFrame["total_foreground_pixels"])
    features["max_z"] = float(currentFrame["max_z"])
    features["scene_temperature"] = float(currentFrame["scene_temperature"])
    features.update(currentFrame.get("fd_features", {}))

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
    def __init__(self, frame_shape=(24, 32), window_seconds=2.0):
        self.pipeline = ThermopilePipeline(frame_shape)
        self.pipeline.fd_window_seconds = float(window_seconds)
        self.previousTrackPositions = {}
        self.lastGoodRow = None  # Fix 3

    def calibrate(self, frames):
        self.pipeline.warm_start(np.asarray(frames, dtype=np.float64))
        self.previousTrackPositions.clear()
        self.lastGoodRow = None

    def process(self, timestamp, frame):
        frame = np.asarray(frame, dtype=np.float64).reshape(self.pipeline.frame_shape)
        detections = self.pipeline.process_frame(frame, timestamp)
        # Fix 3: glitch frame -> repeat last good features (max 1 s) or empty
        if not self.pipeline.last_frame_valid:
            frameRow = glitchFrameRow(timestamp, self.lastGoodRow, self.pipeline.last_scene_temp)
            return getThermalFeatures([frameRow], 0)
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
            "fd_features": dict(self.pipeline.fd_latest),
        }
        self.lastGoodRow = (float(timestamp), frameRow)
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

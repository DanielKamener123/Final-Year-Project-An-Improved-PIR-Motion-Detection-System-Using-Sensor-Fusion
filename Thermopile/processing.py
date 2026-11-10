from dataclasses import dataclass
from collections import deque

import numpy as np

# Config
WIDTH = 32
HEIGHT = 24
FRAME_SHAPE = (HEIGHT, WIDTH)

ALPHA = 0.995
SIGMA_MIN = 0.15

# Fix 2 - Whole-scene shift removal (Trofimova et al., 2017)
# Pixels within 1 pixel of last frame's foreground are not used to measure
# the shift. If too few pixels remain, all pixels are used instead.
MIN_QUIET_FRACTION = 0.25

# Gaussian filter
centerWeight = 4.0
gaussianKernel = np.array([
    [1, 2, 1],
    [2, centerWeight, 2],
    [1, 2, 1],
])
gaussianKernel /= gaussianKernel.sum()

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
THRESHOLD_FLOOR = 1.2     # Thresholds are only ever RAISED above the old value

# Demo Setting: One Fixed Z Threshold For Every Pixel Instead Of The Temperature Table.
# The table raised the threshold to ~4.5-7 in cold scenes (17 C) and the thermopile
# missed 81% of the walks there; at 27 C it was too low and flagged most empty frames.
# These are the effective thresholds (the table path scales its values by 0.8).
# 10 Nov: raised from 1.6 / 1.84. At the exam venue Z 1.6 put a blob on the empty
# ground in 68% of frames; Z 2.2 drops that to ~3% while the fused system (PIR
# carries most detections) keeps 180/183 held-out venue walks and cuts INTRUDER
# time in the empty venue minutes from 57% to 21%.
# 10 Nov (evening): raised to 3.0 / 3.45 together with GROUP_DENOISE below. With the
# read-out pattern removed the noise is ~0.2 C instead of ~0.55 C, so Z 3.0 sees more
# walkers than Z 2.2 did and the empty ground stays quiet (Z 2.8 and below: many blobs).
USE_FIXED_THRESHOLD = True
FIXED_Z_RAW = 3.0         # current-frame detector
FIXED_Z_AVG = 3.45        # recent-frame-average detector (same 1.15 ratio as before)

# Read-out pattern correction: the MLX90640 reads pixels in an interleaved chess pattern and
# pixels in the same (chess parity, column mod 4) group share a common per-frame offset.
# Removing each group's offset every frame (median over quiet pixels, like the whole-scene
# shift) cuts per-pixel noise from ~0.55 C to ~0.2 C in empty scenes.
GROUP_DENOISE = True
# Saved/sent calibration sigma is always the plain (uncorrected) per-pixel std, so old
# recordings, new recordings and the live system all use sigma = saved sigma x this scale.
DENOISE_SIGMA_SCALE = 0.4
_rows, _cols = np.indices((HEIGHT, WIDTH))
PIXEL_GROUP = ((_rows + _cols) % 2) * 4 + (_cols % 4)
PIXEL_GROUPS = [PIXEL_GROUP == g for g in range(8)]
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
AVG_WINDOW_S = 0.8        # 2 frames at ~2 Hz (0.8 not 0.6: still works if live frames slow past 0.6 s)

# Fix 6 - Confirmation (M-of-N): a blob is only reported if its track was seen
# in at least CONFIRM_M frames within CONFIRM_WINDOW_S, unless it is strong.
CONFIRM_M = 2
CONFIRM_WINDOW_S = 0.8     # was 0.6, same reason as AVG_WINDOW_S
STRONG_RATIO = 1.5        # Peak score >= 1.5 x its threshold -> report immediately
CONFIRM_MAX_STEP_PX = 9.0 # A faint blob only counts as "seen again" if it is
                          # within this distance of its track's last centre

# CONFIRM_MAX_STEP_PX now equals max_match_distance (9 px): "seen again" means the
# same as "matched by the tracker". At 2 frames/s a person running at 4 m moves
# about 10 px per frame, so the old 3 px limit reset their confirmation every
# frame and fast walkers never became a reported blob.

# Fix 7 - Tracker gate grows with blob size (near people are big and fast)
GATE_PER_SQRT_PX = 2.5    # gate = max(max_match_distance, 2.5 * sqrt(pixels))

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

# A blob counts as moving when its centre moves faster than this (pixels/s)
MOVING_SPEED_PX_S = 0.5

# Difference compared against the most recent frame about 0.5 seconds earlier.
FD_LAG_SECONDS = 0.5
FD_NOISE_THRESHOLD_C = 0.25  # Initial value; calibrate using empty-scene recordings.
FD_MAX_GAP_SECONDS = 1.0     # Do not difference across sensor outages.
FD_FEATURE_NAMES = [
    "fdMeanAbs", "fdMaxAbs", "fdEnergy", "fdActivePixels",
    "fdWindowMeanEnergy", "fdWindowMaxEnergy",
    "fdWindowActiveFraction", "fdValid",
]

def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    middle = n // 2
    if n % 2 == 1:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))


# Pads A Grid By Repeating Its Edge Pixels (1 pixel on every side)
# Same edge handling as the "reflect" mode used before for a 3x3 kernel.
def pad_edges(grid):
    rows, cols = grid.shape
    row_index = [0] + list(range(rows)) + [rows - 1]
    col_index = [0] + list(range(cols)) + [cols - 1]
    return grid[np.ix_(row_index, col_index)]


# Marks Every Pixel That Is In Or Touching (8-neighbour) A True Pixel
def grow_mask(mask):
    rows, cols = mask.shape
    padded = pad_edges(mask)
    grown = np.zeros((rows, cols), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            grown |= padded[dr:dr + rows, dc:dc + cols]
    return grown


# Background Model Class
class BackgroundModel:
    def __init__(self, shape):
        self.bg = np.zeros(shape)
        self.sigma = np.full(shape, SIGMA_MIN)
        self.calibration_sigma = np.full(shape, SIGMA_MIN)   # uncorrected sigma, the one that is saved/sent
        self._initialised = False

    # Working Sigma From The Saved (uncorrected) Calibration Sigma
    def _set_sigma(self, calibration_sigma):
        self.calibration_sigma = np.asarray(calibration_sigma, dtype=np.float64).reshape(self.bg.shape).copy()
        sigma = self.calibration_sigma * DENOISE_SIGMA_SCALE if GROUP_DENOISE else self.calibration_sigma
        self.sigma = np.maximum(sigma, SIGMA_MIN)

    # Calibrate Background
    def calibration(self, calibration_frames):
        self.bg[:] = np.median(calibration_frames, axis=0)
        measured_sigma = calibration_frames.std(axis=0)
        self._set_sigma(measured_sigma)
        self._initialised = True

    # Load A Saved Background
    def load(self, background, sigma):
        self.bg[:] = np.asarray(background, dtype=np.float64).reshape(self.bg.shape)
        self._set_sigma(sigma)
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

    # Read-Out Pattern - Each Pixel Group's Offset (median over quiet pixels of that group)
    def group_shift(self, frame, previous_mask):
        difference = frame - self.bg
        quiet = ~grow_mask(previous_mask)
        shift = np.zeros_like(difference)
        for group in PIXEL_GROUPS:
            values = difference[group & quiet]
            if values.size < MIN_QUIET_FRACTION * np.count_nonzero(group):
                values = difference[group]
            shift[group] = np.median(values)
        return shift

    # Fix 2 - Measure The Whole-Scene Shift (first principles)
    # shift = median of (frame - background) over "quiet" pixels: pixels that
    # were not foreground last frame and have no foreground 8-neighbour.
    def scene_shift(self, frame, previous_mask):
        difference = frame - self.bg
        quiet = ~grow_mask(previous_mask)
        if np.count_nonzero(quiet) >= MIN_QUIET_FRACTION * difference.size:
            values = difference[quiet]
        else:
            values = difference.reshape(-1)
        return median(values.tolist())

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

# Spatial filtering (first principles 3x3 convolution)
# Each output pixel is the kernel-weighted sum of itself and its 8 neighbours.
def gaussian_filter(z_grid):
    rows, cols = z_grid.shape
    padded = pad_edges(z_grid)
    smoothed = np.zeros((rows, cols))
    for dr in range(3):
        for dc in range(3):
            smoothed += gaussianKernel[dr, dc] * padded[dr:dr + rows, dc:dc + cols]
    return smoothed


# Fix 4 - Threshold Map From Each Pixel's Background Temperature (first principles)
# Linear interpolation in the (background temperature, threshold) table.
def threshold_from_table(background, table):
    # Outside the table: nearest end
    threshold_map = np.full(background.shape, table[-1][1])
    done = background >= table[-1][0]
    below = background <= table[0][0]
    threshold_map[below] = table[0][1]
    done |= below
    for k in range(len(table) - 1):
        bg0, t0 = table[k]
        bg1, t1 = table[k + 1]
        inside = ~done & (background >= bg0) & (background <= bg1)
        threshold_map[inside] = t0 + (t1 - t0) * (background[inside] - bg0) / (bg1 - bg0)
        done |= inside
    # Never go below the old fixed threshold: noisy sites (e.g. the
    # 4 Hz Cefim recordings) crossed the measured mid-range values
    # 4-5% of the time, so lowering it is not justified by the data.
    return np.maximum(threshold_map * 0.8, THRESHOLD_FLOOR)


# Fix 5 - Z Score Of The Average Of The Recent Corrected Differences
# The mean of n frames has noise sigma / sqrt(n), so the Z is scaled by sqrt(n).
def averaged_z(difference_history, sigma):
    n = len(difference_history)
    total = np.zeros(sigma.shape)
    for _, difference in difference_history:
        total = total + difference
    mean = total / n
    root_n = n ** 0.5
    return np.abs(mean) * root_n / sigma


# Fix 4/5 - Combined Score: how many times over its own threshold each pixel is
# (>1 means foreground). Used for the mask, blob strength and blob centres.
def threshold_ratio(smoothed_raw, threshold_raw, smoothed_avg, threshold_avg):
    raw_ratio = smoothed_raw / threshold_raw
    avg_ratio = smoothed_avg / threshold_avg
    return np.where(raw_ratio > avg_ratio, raw_ratio, avg_ratio)

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


# Connected Component Labelling (first principles flood fill, 8-connectivity)
# Pixels are scanned row by row, so blobs are numbered in the order their
# first pixel is found. Returns (labelled grid, number of blobs).
def label_blobs(binary_mask):
    rows, cols = binary_mask.shape
    labelled_grid = np.zeros((rows, cols), dtype=int)
    num_blobs = 0
    for r in range(rows):
        for c in range(cols):
            if not binary_mask[r, c] or labelled_grid[r, c]:
                continue
            # New blob: flood out to every touching foreground pixel
            num_blobs += 1
            labelled_grid[r, c] = num_blobs
            to_visit = [(r, c)]
            while to_visit:
                pr, pc = to_visit.pop()
                for nr in range(pr - 1, pr + 2):
                    for nc in range(pc - 1, pc + 2):
                        if (0 <= nr < rows and 0 <= nc < cols
                                and binary_mask[nr, nc] and not labelled_grid[nr, nc]):
                            labelled_grid[nr, nc] = num_blobs
                            to_visit.append((nr, nc))
    return labelled_grid, num_blobs


# Extract Blobs
# weights: per-pixel smoothed Z used for the weighted centre (Fix 1)
def extract_blobs(binary_mask, weights=None):
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

@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    age_frames: int
    missed_frames: int = 0
    # Fix 1: PIXELS TRAVERSED in the last TRAVERSE_WINDOW_S seconds
    traversed_px: float = 0.0
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


# Thermal Feature Names (in the order the fusion models use them)
def getThermalFeatureNames():
    return [
    "thermalNumberOfBlobs",
    "thermalTotalForegroundPixels",
    "thermalMaxZ",
    # thermalSceneTemperature is no longer a model input: it changes with the day
    # and time of recording, so the classifier learned recording conditions
    # instead of human presence. It is still computed for the display.
    # Primary (most-travelled) blob
    "thermalPixelCount",
    "thermalCentroidRow",
    "thermalCentroidCol",
    "thermalTrackAgeFrames",
    "thermalCentroidSpeed",
    "thermalMaxTemperatureDifference",
    # Multi-blob summaries: information from every currently visible track.
    "thermalMovingBlobCount",
    "thermalMaxBlobTraversed",
    "thermalMaxBlobSpeed",
    "thermalMaxBlobAgeFrames",
    "thermalMaxBlobPixelCount",
    "thermalMaxBlobTemperatureDifference",
    *["thermal" + name[0].upper() + name[1:] for name in FD_FEATURE_NAMES],
    ]


def emptyThermalFeatures():
    return {name: 0.0 for name in getThermalFeatureNames()}


class ThermopilePipeline:
    def __init__(self, frame_shape=FRAME_SHAPE, window_seconds=2.0):
        self.frame_shape = frame_shape
        self.bg_model = BackgroundModel(frame_shape)
        self._prev_mask = np.zeros(frame_shape, dtype=bool)

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self.primary_track_id = None

        # Debug state
        self.last_raw_z = np.zeros(frame_shape)
        self.last_smoothed_z = np.zeros(frame_shape)
        self.last_mask = np.zeros(frame_shape, dtype=bool)
        self.last_threshold = THRESHOLD_FLOOR
        self.last_scene_temp = 0.0
        self.last_shift = 0.0
        self.last_threshold_map = np.full(frame_shape, THRESHOLD_FLOOR)
        self.last_ratio = np.zeros(frame_shape)
        self.last_smoothed_z_avg = np.zeros(frame_shape)
        self.last_detections = []
        # Fix 3: glitch state
        self.last_frame_valid = True
        self.glitch_start = None
        self.glitch_frames = 0
        # Fix 5: recent shift-corrected differences (timestamp, frame - bg)
        self.difference_history = deque(maxlen=16)
        self.fd_frames = deque(maxlen=128)
        self.fd_activity = deque(maxlen=128)
        self.fd_window_seconds = float(window_seconds)
        self.fd_latest = {name: 0.0 for name in FD_FEATURE_NAMES}
        # Feature state: last position of each reported track (for speed) and
        # Fix 3: the last good feature row (timestamp, features)
        self._track_positions = {}
        self._last_good_features = None

    # Clears Everything Learned From Previous Frames (background is kept)
    def reset(self):
        self._prev_mask = np.zeros(self.frame_shape, dtype=bool)
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
        self._track_positions = {}
        self._last_good_features = None

    # Get Background Without Human (live: frames from an empty scene)
    def calibrate(self, calibration_frames):
        frames = np.asarray(calibration_frames, dtype=np.float64).reshape(-1, *self.frame_shape)
        self.bg_model.calibration(frames)
        self.reset()

    # Use A Saved Background (offline: background and sigma from metadata.json)
    def load_calibration(self, background, sigma):
        self.bg_model.load(background, sigma)
        self.reset()

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
            data.update(fdMeanAbs=float(absdiff.mean()), fdMaxAbs=float(absdiff.max()),
                        fdEnergy=float(np.mean(absdiff ** 2)),
                        fdActivePixels=float(np.count_nonzero(absdiff > FD_NOISE_THRESHOLD_C)),
                        fdValid=1.0)
            self.fd_activity.append((timestamp, data["fdEnergy"], data["fdActivePixels"]))
        self.fd_frames.append((timestamp, filtered))
        while self.fd_frames and timestamp - self.fd_frames[0][0] > max(self.fd_window_seconds + FD_LAG_SECONDS, 3.0):
            self.fd_frames.popleft()
        while self.fd_activity and timestamp - self.fd_activity[0][0] > self.fd_window_seconds:
            self.fd_activity.popleft()
        if self.fd_activity:
            energies = [v[1] for v in self.fd_activity]
            data["fdWindowMeanEnergy"] = float(np.mean(energies))
            data["fdWindowMaxEnergy"] = float(max(energies))
            data["fdWindowActiveFraction"] = float(np.mean([v[2] > 0 for v in self.fd_activity]))
        self.fd_latest = data

    # Runs One Frame Through Every Stage And Returns The Confirmed Detections
    def process_frame(self, frame, timestamp):
        now = float(timestamp)
        frame = np.asarray(frame, dtype=np.float64).reshape(self.frame_shape)

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
        self.update_fd(frame, now)

        # Fix 2: move the background with the scene. The Z score, temperature
        # difference and sigma update below all use the corrected difference.
        # Read-out pattern: each pixel group gets its own shift instead (includes the scene shift)
        if GROUP_DENOISE:
            self.bg_model.apply_shift(self.bg_model.group_shift(frame, self._prev_mask))
        else:
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
        if USE_FIXED_THRESHOLD:
            threshold_raw = np.full(self.frame_shape, FIXED_Z_RAW)
            threshold_avg = np.full(self.frame_shape, FIXED_Z_AVG)
        else:
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
                traversed_px=traversed,
                history=history,
                seen_times=seen_times,
            )

            used_track_ids.add(track.id)

            # Fix 6: report strong blobs immediately; faint blobs only once
            # their track has been seen CONFIRM_M times in CONFIRM_WINDOW_S.
            peak_ratio = float(np.max(ratio[blob.mask]))
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
        # noise-filtered pixels traversed, not the largest blob.
        # Reconsider each frame so an old, static primary does not lock selection.
        eligible_results = [
            result for result in results
            if result.features.track_age_frames >= min_primary_track_frames
        ]
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

        # Store Previous Info
        self._prev_mask = mask

        # Store Latest Intermediates For Display Access
        self.last_raw_z = z
        self.last_smoothed_z = smoothed_z
        self.last_smoothed_z_avg = smoothed_z_avg
        self.last_mask = mask
        self.last_threshold_map = threshold_raw
        self.last_threshold = float(np.mean(threshold_raw))
        self.last_ratio = ratio
        self.last_scene_temp = scene_temp

        return results

    # Runs One Frame And Returns Its Thermal Feature Vector (as a dict)
    # This is the function offline processing and the live system both use.
    def process(self, frame, timestamp):
        timestamp = float(timestamp)
        detections = self.process_frame(frame, timestamp)
        self.last_detections = detections

        # Fix 3: glitch frame -> repeat last good features (max 1 s) or empty
        if not self.last_frame_valid:
            last_good = self._last_good_features
            if last_good is not None and timestamp - last_good[0] <= FEATURE_HOLD_S:
                return dict(last_good[1])
            features = emptyThermalFeatures()
            features["thermalSceneTemperature"] = float(self.last_scene_temp)
            return features

        features = self.frame_features(detections, timestamp)
        self._last_good_features = (timestamp, features)
        return dict(features)

    # Turns This Frame's Detections Into The Thermal Feature Vector
    def frame_features(self, detections, timestamp):
        features = emptyThermalFeatures()
        features["thermalNumberOfBlobs"] = float(len(detections))
        features["thermalTotalForegroundPixels"] = float(sum(d.features.pixel_count for d in detections))
        features["thermalMaxZ"] = float(np.max(self.last_smoothed_z))
        features["thermalSceneTemperature"] = float(self.last_scene_temp)
        for name, value in self.fd_latest.items():
            features["thermal" + name[0].upper() + name[1:]] = value

        # Centre speed of each detection from where its track was last reported
        speeds = {}
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

        # Scene-wide summaries don't depend on picking the correct primary blob.
        # Values are computed from the current frame only, with track histories
        # accumulated causally from previous frames.
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

        # Primary blob features
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

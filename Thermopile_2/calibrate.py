"""
calibrate.py

Interactive live calibration tool for the thermopile pipeline. Run this
directly wherever interface.py can reach the MLX90640 (Rock 4C+ or the
FT232H bring-up rig) to generate real, measured numbers for:

  - adaptive_coeffs        (adaptive threshold quadratic fit)
  - size_lookup             (pixel-count-vs-range table)
  - max_expected_motion_px  (motion score scaling)
  - expected_max_z          (contrast score ceiling)

Nothing here is applied automatically -- each mode prints the numbers it
found so you can review and paste them into thermopile_pipeline.py
yourself.

Methodology notes:
  - Adaptive threshold sweep uses an F-beta score with beta=1.2, weighting
    recall over precision, following the calibration procedure described
    in the Outdoor People Detection (Cerutti et al.) paper.
  - Background is frozen (not updated) during the threshold sweep so the
    sweep isn't confounded by a moving background estimate mid-session.
  - The field session (size/motion/contrast) runs with the background
    updating masked, same as live deployment, since those features are
    measured under real operating conditions.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from interface import getFrame as getRawFrame
from thermopile_pipeline import (
    BackgroundModel,
    gaussian_filter_3x3,
    extract_blobs,
    apply_threshold,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FRAME_SHAPE = (24, 32)
N_WARMUP_FRAMES = 20

DATA_DIR = "calibration_data"
THRESHOLD_CSV = os.path.join(DATA_DIR, "threshold_points.csv")

SEGMENT_DURATION_S = 5.0   # length of each labelled capture segment
N_SEGMENTS = 6              # number of alternating pos/neg segments
FBETA_BETA = 1.2            # weight recall over precision, per literature

WALK_SESSION_DURATION_S = 20.0
SIZE_SAMPLE_DURATION_S = 4.0
FIXED_THRESHOLD_FOR_CALIBRATION = 3.0  # only used to find a blob during calibration


# ---------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------

def _flatToGrid(flatFrame):
    return np.array(flatFrame, dtype=np.float64).reshape(FRAME_SHAPE)


def _captureFrames(durationS):
    """Capture frames continuously for durationS seconds, return list of 2D arrays."""
    frames = []
    startTime = time.time()
    while time.time() - startTime < durationS:
        frames.append(_flatToGrid(getRawFrame()))
    return frames


def _warmUpBackground():
    print(f"\nWarming up background model on {N_WARMUP_FRAMES} frames.")
    print("Make sure NO ONE is in frame right now.")
    input("Press Enter when the scene is clear...")
    calibFrames = np.empty((N_WARMUP_FRAMES, *FRAME_SHAPE), dtype=np.float64)
    for i in range(N_WARMUP_FRAMES):
        calibFrames[i] = _flatToGrid(getRawFrame())
    bgModel = BackgroundModel(FRAME_SHAPE)
    bgModel.warm_start(calibFrames)
    print(f"Warm-up complete. Scene temperature = {bgModel.scene_temperature():.2f}C")
    return bgModel


# ---------------------------------------------------------------------
# Pure math: F-beta sweep (kept separate from I/O so it's testable)
# ---------------------------------------------------------------------

def fbeta(precision, recall, beta):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def sweepThreshold(peakZs, labels, thresholds):
    """Find the threshold that maximises F-beta over labelled (peakZ, label) pairs."""
    peakZs = np.asarray(peakZs)
    labels = np.asarray(labels, dtype=bool)
    best = None
    for t in thresholds:
        predicted = peakZs > t
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = fbeta(precision, recall, FBETA_BETA)
        if best is None or f > best["f"]:
            best = {"threshold": float(t), "f": f, "precision": precision, "recall": recall}
    return best


def fitQuadratic(sceneTemps, optimalThresholds):
    coeffs = np.polyfit(sceneTemps, optimalThresholds, deg=2)
    residuals = np.array(optimalThresholds) - np.polyval(coeffs, sceneTemps)
    return coeffs, residuals


# ---------------------------------------------------------------------
# Mode 1: adaptive threshold data collection
# ---------------------------------------------------------------------

def runAdaptiveThresholdSession():
    print("\n=== Adaptive threshold calibration ===")
    print(f"This will run {N_SEGMENTS} segments of {SEGMENT_DURATION_S:.0f}s each,")
    print("alternating between 'no one in frame' and 'person in frame'.")
    print("Background is warmed up once and frozen for the whole session so")
    print("threshold sweeping isn't confounded by a moving background estimate.\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    bgModel = _warmUpBackground()
    sceneTemp = bgModel.scene_temperature()

    peakZs = []
    labels = []

    for i in range(N_SEGMENTS):
        label = (i % 2 == 1)  # alternate: empty, person, empty, person...
        tag = "PERSON IN FRAME" if label else "EMPTY SCENE"
        input(f"\nSegment {i + 1}/{N_SEGMENTS}: get ready for [{tag}], "
              f"then press Enter to start {SEGMENT_DURATION_S:.0f}s capture...")
        print("Capturing...")
        frames = _captureFrames(SEGMENT_DURATION_S)
        for frame in frames:
            z = bgModel.z_score(frame)
            smoothedZ = gaussian_filter_3x3(z)
            peakZs.append(float(np.max(smoothedZ)))
            labels.append(label)
        print(f"  Captured {len(frames)} frames.")

    thresholds = np.linspace(0.5, 12.0, 231)
    best = sweepThreshold(peakZs, labels, thresholds)

    print("\n--- Result ---")
    print(f"Scene temperature: {sceneTemp:.2f}C")
    print(f"Optimal threshold: {best['threshold']:.2f}  "
          f"(F-{FBETA_BETA}={best['f']:.3f}, "
          f"precision={best['precision']:.3f}, recall={best['recall']:.3f})")

    fileExists = os.path.isfile(THRESHOLD_CSV)
    with open(THRESHOLD_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["timestamp", "scene_temp", "optimal_threshold",
                              "fbeta", "precision", "recall", "n_frames"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{sceneTemp:.3f}",
                          f"{best['threshold']:.3f}", f"{best['f']:.3f}",
                          f"{best['precision']:.3f}", f"{best['recall']:.3f}",
                          len(peakZs)])
    print(f"Logged to {THRESHOLD_CSV}.")
    print("Run this session again at DIFFERENT room temperatures / times of day")
    print("to build up enough points for a reliable quadratic fit.")


def fitAdaptiveThresholdCurve():
    if not os.path.isfile(THRESHOLD_CSV):
        print(f"No data yet at {THRESHOLD_CSV}. Run the adaptive threshold session first.")
        return

    sceneTemps = []
    optimalThresholds = []
    with open(THRESHOLD_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sceneTemps.append(float(row["scene_temp"]))
            optimalThresholds.append(float(row["optimal_threshold"]))

    print(f"\nLoaded {len(sceneTemps)} calibration points from {THRESHOLD_CSV}.")
    if len(sceneTemps) < 3:
        print("Need at least 3 points to fit a quadratic. Keep collecting sessions.")
        return
    if len(sceneTemps) < 6:
        print("Fewer than 6 points -- the fit will run but is unlikely to be")
        print("reliable yet. Aim for sessions spread across a wide range of")
        print("background temperatures before trusting this.")

    coeffs, residuals = fitQuadratic(sceneTemps, optimalThresholds)
    a, b, c = coeffs
    print("\n--- Quadratic fit ---")
    print(f"threshold = {a:.6f} * T^2 + {b:.6f} * T + {c:.6f}")
    print("\nPaste this into thermopile_pipeline.py:")
    print(f"adaptive_coeffs = np.array([{a:.6f}, {b:.6f}, {c:.6f}])")
    print("use_adaptive_threshold = True")

    print(f"\nFit residuals: mean={residuals.mean():.3f}, "
          f"std={residuals.std():.3f}, max abs={np.abs(residuals).max():.3f}")
    print("Large residuals mean the quadratic isn't capturing your data well --")
    print("collect more points spread across a wider temperature range.")


# ---------------------------------------------------------------------
# Mode 2: size-at-range, motion, and contrast (walk-and-stand session)
# ---------------------------------------------------------------------

def _biggestBlob(frame, bgModel):
    """Run one frame through the live-style pipeline (masked bg update) and
    return the biggest blob found, or None."""
    bgModel.update(frame, np.zeros(FRAME_SHAPE, dtype=bool))
    z = bgModel.z_score(frame)
    smoothedZ = gaussian_filter_3x3(z)
    mask = apply_threshold(smoothedZ, FIXED_THRESHOLD_FOR_CALIBRATION)
    blobs = extract_blobs(mask)
    if not blobs:
        return None, smoothedZ
    return max(blobs, key=lambda b: b.pixel_count), smoothedZ


def runFieldSession():
    print("\n=== Size / motion / contrast calibration ===")
    print("This session runs the pipeline the same way it runs live: the")
    print("background continues to update (masked) while you move around.\n")

    bgModel = _warmUpBackground()

    print("\nPart A: size-at-range")
    print("You'll be prompted to stand still at a series of known distances.")
    sizeSamples = []  # (distance_m, avg_pixel_count, avg_centroid_row)

    while True:
        distanceStr = input(
            "\nEnter a distance in metres to test (or blank to move to Part B): ").strip()
        if distanceStr == "":
            break
        try:
            distance = float(distanceStr)
        except ValueError:
            print("Not a number, try again.")
            continue

        input(f"Stand still at {distance:.1f}m, facing the sensor, then press Enter...")
        frames = _captureFrames(SIZE_SAMPLE_DURATION_S)
        pixelCounts = []
        centroidRows = []
        for frame in frames:
            blob, _ = _biggestBlob(frame, bgModel)
            if blob is None:
                continue
            pixelCounts.append(blob.pixel_count)
            centroidRows.append(blob.centroid[0])

        if not pixelCounts:
            print("  No blob detected at this distance -- skipped. Check the "
                  "person is actually crossing threshold at this range.")
            continue

        avgPixels = float(np.mean(pixelCounts))
        avgRow = float(np.mean(centroidRows))
        sizeSamples.append((distance, avgPixels, avgRow))
        print(f"  avg pixel_count={avgPixels:.1f}, avg centroid_row={avgRow:.1f} "
              f"over {len(pixelCounts)} frames.")

    if sizeSamples:
        sizeSamples.sort(key=lambda s: s[0])
        print("\n--- size_lookup (paste into thermopile_pipeline.py) ---")
        print("size_lookup = np.array([")
        for distance, pixels, _ in sizeSamples:
            print(f"    [{distance:.1f}, {pixels:.1f}],")
        print("])")
        print("\nMeasured row-vs-range (for replacing the placeholder linear")
        print("mapping in _estimate_expected_pixels with your real geometry):")
        for distance, _, row in sizeSamples:
            print(f"  distance={distance:.1f}m -> centroid_row={row:.1f}")

    print("\nPart B: motion and contrast")
    print(f"Walk naturally across the sensor's field of view for "
          f"{WALK_SESSION_DURATION_S:.0f}s.")
    input("Press Enter to start...")

    displacements = []
    peakZsWalking = []
    prevCentroid = None
    startTime = time.time()
    while time.time() - startTime < WALK_SESSION_DURATION_S:
        frame = _flatToGrid(getRawFrame())
        blob, smoothedZ = _biggestBlob(frame, bgModel)
        if blob is None:
            prevCentroid = None
            continue
        peakZsWalking.append(float(np.max(smoothedZ[blob.mask])))
        if prevCentroid is not None:
            dr = blob.centroid[0] - prevCentroid[0]
            dc = blob.centroid[1] - prevCentroid[1]
            displacements.append(float(np.hypot(dr, dc)))
        prevCentroid = blob.centroid

    if displacements:
        d = np.array(displacements)
        print(f"\nCaptured {len(d)} frame-to-frame displacements.")
        print(f"  median={np.median(d):.2f}px  "
              f"p75={np.percentile(d, 75):.2f}px  "
              f"p90={np.percentile(d, 90):.2f}px  "
              f"max={d.max():.2f}px")
        suggested = np.percentile(d, 90)
        print(f"\nSuggested max_expected_motion_px = {suggested:.1f}  "
              f"(90th percentile of observed displacement)")
    else:
        print("\nNo motion data captured -- check the person stayed detected "
              "throughout the walk.")

    if peakZsWalking:
        pz = np.array(peakZsWalking)
        print(f"\nPeak Z-score while walking: "
              f"median={np.median(pz):.2f}  p75={np.percentile(pz, 75):.2f}  "
              f"p90={np.percentile(pz, 90):.2f}  max={pz.max():.2f}")
        suggestedMaxZ = np.percentile(pz, 90)
        print(f"Suggested expected_max_z = {suggestedMaxZ:.1f}  "
              f"(90th percentile of observed peak Z-score)")


# ---------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------

def main():
    while True:
        print("\n" + "=" * 60)
        print("Thermopile pipeline calibration")
        print("=" * 60)
        print("1) Adaptive threshold: record one labelled session")
        print("2) Adaptive threshold: fit quadratic curve from all sessions")
        print("3) Size / motion / contrast field session")
        print("q) Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice == "1":
            runAdaptiveThresholdSession()
        elif choice == "2":
            fitAdaptiveThresholdCurve()
        elif choice == "3":
            runFieldSession()
        elif choice == "q":
            break
        else:
            print("Not a valid option.")


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIR.processing import (
    PIRProcessor,
    getCurrentFeatureNames,
    getCurrentFeatureVector,
    getHistoryFeatureNames,
    getHistoryFeatureVector,
)
from Thermopile.thermopile_pipeline import ThermopilePipeline

# Current Thermopile Features Come From The Latest Valid Frame
THERMAL_CURRENT_FEATURE_NAMES = [
    "number_of_blobs",
    "total_foreground_pixels",
    "max_z",
    "top_10_mean_z",
    "pixels_z_above_2",
    "pixels_z_above_3",
    "scene_temperature",
    "pixel_count",
    "centroid_row",
    "centroid_col",
    "track_age_frames",
    "centroid_speed_px_s",
    "mean_temperature_difference",
    "max_temperature_difference",
]


def thermalFeatureNames():
    return THERMAL_CURRENT_FEATURE_NAMES


# If No Blob Is Detected
def emptyThermalFeatures():
    return {name: 0.0 for name in thermalFeatureNames()}


# Gets Human/Not Human Labels From Video
def loadLabels(recordingDir):
    labelsPath = recordingDir / "labels.csv"
    if not labelsPath.exists():
        return None

    labels = pd.read_csv(labelsPath)
    required = {"start_s", "end_s", "human"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{labelsPath} must contain columns {sorted(required)}")
    return labels


# Check If Time Is Within A Labelled Human Interval
def labelAtTime(labels, relativeVideoTime):
    if labels.empty:
        return 0
    active = labels[
        (labels["start_s"] <= relativeVideoTime)
        & (relativeVideoTime < labels["end_s"])
        & (labels["human"] == 1)
    ]
    return int(not active.empty)


def preprocessPIR(pirData, initialBaseline, windowSeconds):
    processor = PIRProcessor(initialBaseline=initialBaseline, windowTime=windowSeconds)
    timestamps = pirData["time_s"].to_numpy(dtype=np.float64)
    voltages = pirData["voltage"].to_numpy(dtype=np.float64)
    relative = np.empty_like(voltages)

    for index, (timestamp, voltage) in enumerate(zip(timestamps, voltages)):
        relative[index] = processor.processSample(timestamp, voltage)

    return timestamps, relative


def flattenDetection(detection, centroidSpeed):
    features = detection.features
    return {
        "track_id": int(features.track_id),
        "pixel_count": float(features.pixel_count),
        "centroid_row": float(features.centroid[0]),
        "centroid_col": float(features.centroid[1]),
        "track_age_frames": float(features.track_age_frames),
        "centroid_speed_px_s": float(centroidSpeed),
        "mean_temperature_difference": float(features.mean_temperature_difference),
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

            flattened.append(flattenDetection(detection, centroidSpeed))
            updatedTrackPositions[trackId] = (timestamp, centroidRow, centroidCol)

        previousTrackPositions = updatedTrackPositions

        smoothedZ = pipeline.last_smoothed_z
        flatZ = smoothedZ.reshape(-1)
        strongestCount = min(10, flatZ.size)
        strongestValues = np.partition(flatZ, flatZ.size - strongestCount)[-strongestCount:]

        frameRows.append({
            "detections": flattened,
            "number_of_blobs": len(flattened),
            "total_foreground_pixels": float(sum(detection["pixel_count"] for detection in flattened)),
            "primary_track_id": pipeline.primary_track_id,
            "max_z": float(np.max(flatZ)),
            "top_10_mean_z": float(np.mean(strongestValues)),
            "pixels_z_above_2": float(np.count_nonzero(smoothedZ > 2.0)),
            "pixels_z_above_3": float(np.count_nonzero(smoothedZ > 3.0)),
            "scene_temperature": float(pipeline.last_scene_temp),
        })

    return timestamps, frameRows


def thermalFeaturesAtIndex(thermalRows, thermalIndex):
    features = emptyThermalFeatures()
    currentFrame = thermalRows[thermalIndex]
    selectedTrackId = currentFrame["primary_track_id"]

    features["number_of_blobs"] = float(currentFrame["number_of_blobs"])
    features["total_foreground_pixels"] = float(currentFrame["total_foreground_pixels"])
    features["max_z"] = float(currentFrame["max_z"])
    features["top_10_mean_z"] = float(currentFrame["top_10_mean_z"])
    features["pixels_z_above_2"] = float(currentFrame["pixels_z_above_2"])
    features["pixels_z_above_3"] = float(currentFrame["pixels_z_above_3"])
    features["scene_temperature"] = float(currentFrame["scene_temperature"])

    if selectedTrackId is None:
        return False, features

    matching = [
        detection
        for detection in currentFrame["detections"]
        if int(detection["track_id"]) == int(selectedTrackId)
    ]
    if not matching:
        return False, features

    currentPrimary = matching[0]
    features["pixel_count"] = float(currentPrimary["pixel_count"])
    features["centroid_row"] = float(currentPrimary["centroid_row"])
    features["centroid_col"] = float(currentPrimary["centroid_col"])
    features["track_age_frames"] = float(currentPrimary["track_age_frames"])
    features["centroid_speed_px_s"] = float(currentPrimary["centroid_speed_px_s"])
    features["mean_temperature_difference"] = float(currentPrimary["mean_temperature_difference"])
    features["max_temperature_difference"] = float(currentPrimary["max_temperature_difference"])

    return True, features


def recordingFolders(datasetDir):
    datasetDir = Path(datasetDir)
    if not datasetDir.exists():
        return []
    return sorted(path.parent for path in datasetDir.rglob("metadata.json"))


def buildRecordingDataset(recordingDir, windowSeconds, maxPirAgePeriods=2.0):
    recordingDir = Path(recordingDir)
    labels = loadLabels(recordingDir)
    if labels is None:
        print(f"Skipping {recordingDir}: labels.csv not found")
        return pd.DataFrame()

    with open(recordingDir / "metadata.json", "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)

    pirData = pd.read_csv(recordingDir / metadata.get("pir_file", "pir.csv"))
    thermalData = pd.read_csv(recordingDir / metadata.get("thermopile_file", "thermopile.csv"))

    if pirData.empty or thermalData.empty:
        print(f"Skipping {recordingDir}: one of the sensor CSV files is empty")
        return pd.DataFrame()

    pirTimestamps, pirRelative = preprocessPIR(
        pirData,
        metadata["pir_baseline_v"],
        windowSeconds,
    )
    thermalTimestamps, thermalRows = processThermopile(thermalData, metadata)

    # No Fusion Decision Is Made Until The PIR History Buffer Is Full
    startTime = pirTimestamps[0] + windowSeconds
    endTime = pirTimestamps[-1]
    if endTime <= startTime:
        print(f"Skipping {recordingDir}: recording is shorter than the selected window")
        return pd.DataFrame()

    pirRate = float(metadata.get("pir_sample_rate_hz", 0.0))
    maxPirAge = maxPirAgePeriods / pirRate if pirRate > 0 else np.inf

    pirHistoryNames = getHistoryFeatureNames()
    pirCurrentNames = getCurrentFeatureNames()
    thermalNames = thermalFeatureNames()
    rows = []

    validThermalIndices = np.flatnonzero(
        (thermalTimestamps >= startTime)
        & (thermalTimestamps <= endTime)
    )

    for thermalIndex in validThermalIndices:
        fusionTime = float(thermalTimestamps[thermalIndex])
        pirStart = np.searchsorted(pirTimestamps, fusionTime - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTime, side="right")
        if pirEnd <= pirStart:
            continue

        pirSamples = list(zip(pirTimestamps[pirStart:pirEnd], pirRelative[pirStart:pirEnd]))
        pirHistoryVector = np.asarray(getHistoryFeatureVector(pirSamples), dtype=np.float64)
        if len(pirHistoryVector) != len(pirHistoryNames):
            raise ValueError(
                "PIR getFeatureVector() and getFeatureNames() return different lengths. "
                "Update both together in PIR/processing.py."
            )

        latestPirIndex = pirEnd - 1
        pirAge = float(fusionTime - pirTimestamps[latestPirIndex])
        pirValid = int(pirAge <= maxPirAge)
        pirCurrentVector = (
            np.asarray(getCurrentFeatureVector(pirRelative[latestPirIndex]), dtype=np.float64)
            if pirValid
            else np.zeros(len(pirCurrentNames), dtype=np.float64)
        )

        thermalDetected, thermalFeatures = thermalFeaturesAtIndex(
            thermalRows,
            int(thermalIndex),
        )

        relativeVideoTime = float(fusionTime)

        row = {
            "recording": str(recordingDir),
            "fusion_time_s": float(fusionTime),
            "video_time_s": relativeVideoTime,
            "window_seconds": float(windowSeconds),
            "pir_valid": pirValid,
            "thermal_detected": int(thermalDetected),
            "human": labelAtTime(labels, relativeVideoTime),
        }
        row.update({f"pir_{name}": float(value) for name, value in zip(pirCurrentNames, pirCurrentVector)})
        row.update({f"pir_{name}": float(value) for name, value in zip(pirHistoryNames, pirHistoryVector)})
        row.update({f"thermal_{name}": float(thermalFeatures[name]) for name in thermalNames})
        rows.append(row)

    return pd.DataFrame(rows)


def buildDataset(datasetDir, windowSeconds, maxPirAgePeriods=2.0):
    allRows = []
    folders = recordingFolders(datasetDir)
    if not folders:
        raise FileNotFoundError(f"No recording folders found under {datasetDir}")

    for recordingDir in folders:
        print(f"Processing {recordingDir} ...")
        data = buildRecordingDataset(
            recordingDir,
            windowSeconds,
            maxPirAgePeriods=maxPirAgePeriods,
        )
        if not data.empty:
            allRows.append(data)

    if not allRows:
        raise ValueError(f"No labelled recordings could be processed under {datasetDir}")

    return pd.concat(allRows, ignore_index=True)

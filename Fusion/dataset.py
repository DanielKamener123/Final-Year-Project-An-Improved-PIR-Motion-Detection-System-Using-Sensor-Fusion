import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIR.processing import (
    PIRProcessor,
    getCurrentFeatureNames,
    getCurrentFeatureVector,
    getFeatureNames,
    getFeatureVector,
)
from Thermopile.thermopile_pipeline import ThermopilePipeline

# Current Thermopile Features Come From The Latest Valid Frame
THERMAL_CURRENT_FEATURE_NAMES = [
    "pixel_count",
    "width",
    "height",
    "centroid_row",
    "centroid_col",
    "mean_temperature_difference",
    "max_temperature_difference",
    "virtual_position_x",
    "horizontal_pixels_covered",
    "vertical_pixels_covered",
    "number_of_blobs",
    "total_foreground_pixels",
]

# Historical Thermopile Features Are Calculated Over The Shared History Window
THERMAL_HISTORY_FEATURE_NAMES = [
    "row_displacement",
    "col_displacement",
    "displacement_px",
    "normalised_displacement",
    "cumulative_displacement_px",
    "horizontal_pixels_traversed",
    "vertical_pixels_traversed",
    "horizontal_direction",
    "vertical_direction",
    "presence_ratio",
]


def thermalFeatureNames():
    return THERMAL_CURRENT_FEATURE_NAMES + THERMAL_HISTORY_FEATURE_NAMES


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


def flattenDetection(detection):
    features = detection.features
    return {
        "track_id": int(features.track_id),
        "pixel_count": float(features.pixel_count),
        "width": float(features.width),
        "height": float(features.height),
        "centroid_row": float(features.centroid[0]),
        "centroid_col": float(features.centroid[1]),
        "mean_temperature_difference": float(features.mean_temperature_difference),
        "max_temperature_difference": float(features.max_temperature_difference),
        "virtual_position_x": float(features.virtual_position_x),
        "horizontal_pixels_covered": float(features.horizontal_pixels_covered),
        "vertical_pixels_covered": float(features.vertical_pixels_covered),
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

    for _, row in thermalData.iterrows():
        frame = row[pixelColumns].to_numpy(dtype=np.float64).reshape(frameShape)
        detections = pipeline.process_frame(frame)
        flattened = [flattenDetection(detection) for detection in detections]
        frameRows.append({
            "detections": flattened,
            "number_of_blobs": len(flattened),
            "total_foreground_pixels": float(sum(detection["pixel_count"] for detection in flattened)),
        })

    return timestamps, frameRows


def largestDetection(frameRow):
    detections = frameRow["detections"]
    if not detections:
        return None
    return max(detections, key=lambda detection: detection["pixel_count"])


def thermalFeaturesAtTime(thermalTimestamps, thermalRows, fusionTime, windowSeconds, thermalValid):
    features = emptyThermalFeatures()

    thermalIndex = np.searchsorted(thermalTimestamps, fusionTime, side="right") - 1
    if thermalIndex < 0:
        return False, features

    currentFrame = thermalRows[thermalIndex]
    currentPrimary = largestDetection(currentFrame) if thermalValid else None

    # Current Frame Features
    if currentPrimary is not None:
        for name in THERMAL_CURRENT_FEATURE_NAMES:
            if name == "number_of_blobs":
                features[name] = float(currentFrame["number_of_blobs"])
            elif name == "total_foreground_pixels":
                features[name] = float(currentFrame["total_foreground_pixels"])
            else:
                features[name] = float(currentPrimary[name])
    elif thermalValid:
        features["number_of_blobs"] = float(currentFrame["number_of_blobs"])
        features["total_foreground_pixels"] = float(currentFrame["total_foreground_pixels"])

    # Use The Current Primary Track For History. If There Is No Current Blob,
    # Use The Most Recent Primary Track Still Present In The History Window.
    historyStart = np.searchsorted(thermalTimestamps, fusionTime - windowSeconds, side="right")
    historyEnd = thermalIndex + 1
    historyIndices = list(range(historyStart, historyEnd))

    selectedTrackId = None
    if currentPrimary is not None:
        selectedTrackId = int(currentPrimary["track_id"])
    else:
        for index in reversed(historyIndices):
            primary = largestDetection(thermalRows[index])
            if primary is not None:
                selectedTrackId = int(primary["track_id"])
                break

    if selectedTrackId is None or not historyIndices:
        return currentPrimary is not None, features

    trackHistory = []
    detectedFrameCount = 0

    for index in historyIndices:
        matching = [
            detection
            for detection in thermalRows[index]["detections"]
            if int(detection["track_id"]) == selectedTrackId
        ]
        if not matching:
            continue

        detectedFrameCount += 1
        trackHistory.append((float(thermalTimestamps[index]), matching[0]))

    features["presence_ratio"] = float(detectedFrameCount / len(historyIndices))

    if len(trackHistory) < 2:
        return currentPrimary is not None, features

    first = trackHistory[0][1]
    last = trackHistory[-1][1]

    rowDisplacement = float(last["centroid_row"] - first["centroid_row"])
    colDisplacement = float(last["centroid_col"] - first["centroid_col"])
    displacement = float(np.hypot(rowDisplacement, colDisplacement))

    cumulativeDisplacement = 0.0
    horizontalTraversed = 0.0
    verticalTraversed = 0.0

    for (_, previous), (_, current) in zip(trackHistory[:-1], trackHistory[1:]):
        dr = float(current["centroid_row"] - previous["centroid_row"])
        dc = float(current["centroid_col"] - previous["centroid_col"])
        cumulativeDisplacement += float(np.hypot(dr, dc))
        horizontalTraversed += abs(dc)
        verticalTraversed += abs(dr)

    normalisingHeight = max(float(last["height"]), 1.0)

    features["row_displacement"] = rowDisplacement
    features["col_displacement"] = colDisplacement
    features["displacement_px"] = displacement
    features["normalised_displacement"] = float(displacement / normalisingHeight)
    features["cumulative_displacement_px"] = float(cumulativeDisplacement)
    features["horizontal_pixels_traversed"] = float(horizontalTraversed)
    features["vertical_pixels_traversed"] = float(verticalTraversed)
    features["horizontal_direction"] = float(np.sign(colDisplacement))
    features["vertical_direction"] = float(np.sign(rowDisplacement))

    return currentPrimary is not None, features


def recordingFolders(datasetDir):
    datasetDir = Path(datasetDir)
    if not datasetDir.exists():
        return []
    return sorted(path.parent for path in datasetDir.rglob("metadata.json"))


def buildRecordingDataset(recordingDir, windowSeconds, stepSeconds, maxSensorAgePeriods=2.0):
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

    # No Fusion Decision Is Made Until Both History Buffers Are Full
    startTime = max(pirTimestamps[0], thermalTimestamps[0]) + windowSeconds
    endTime = min(pirTimestamps[-1], thermalTimestamps[-1])
    if endTime <= startTime:
        print(f"Skipping {recordingDir}: recording is shorter than the selected window")
        return pd.DataFrame()

    pirRate = float(metadata.get("pir_sample_rate_hz", 0.0))
    thermalRate = float(metadata.get("thermal_rate_hz", 0.0))
    maxPirAge = maxSensorAgePeriods / pirRate if pirRate > 0 else np.inf
    maxThermalAge = maxSensorAgePeriods / thermalRate if thermalRate > 0 else np.inf

    pirHistoryNames = getFeatureNames()
    pirCurrentNames = getCurrentFeatureNames()
    thermalNames = thermalFeatureNames()
    rows = []

    fusionTimes = np.arange(startTime, endTime + 1e-9, stepSeconds)
    for fusionTime in fusionTimes:
        pirStart = np.searchsorted(pirTimestamps, fusionTime - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTime, side="right")
        if pirEnd <= pirStart:
            continue

        pirSamples = list(zip(pirTimestamps[pirStart:pirEnd], pirRelative[pirStart:pirEnd]))
        pirHistoryVector = np.asarray(getFeatureVector(pirSamples), dtype=np.float64)
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

        thermalIndex = np.searchsorted(thermalTimestamps, fusionTime, side="right") - 1
        if thermalIndex < 0:
            continue
        thermalAge = float(fusionTime - thermalTimestamps[thermalIndex])
        thermalValid = int(thermalAge <= maxThermalAge)
        thermalDetected, thermalFeatures = thermalFeaturesAtTime(
            thermalTimestamps,
            thermalRows,
            fusionTime,
            windowSeconds,
            bool(thermalValid),
        )

        relativeVideoTime = float(fusionTime)

        row = {
            "recording": str(recordingDir),
            "fusion_time_s": float(fusionTime),
            "video_time_s": relativeVideoTime,
            "window_seconds": float(windowSeconds),
            "pir_valid": pirValid,
            "thermal_valid": thermalValid,
            "thermal_detected": int(thermalDetected),
            "human": labelAtTime(labels, relativeVideoTime),
        }
        row.update({f"pir_{name}": float(value) for name, value in zip(pirCurrentNames, pirCurrentVector)})
        row.update({f"pir_{name}": float(value) for name, value in zip(pirHistoryNames, pirHistoryVector)})
        row.update({f"thermal_{name}": float(thermalFeatures[name]) for name in thermalNames})
        rows.append(row)

    return pd.DataFrame(rows)


def buildDataset(datasetDir, windowSeconds, stepSeconds, maxSensorAgePeriods=2.0):
    allRows = []
    folders = recordingFolders(datasetDir)
    if not folders:
        raise FileNotFoundError(f"No recording folders found under {datasetDir}")

    for recordingDir in folders:
        print(f"Processing {recordingDir} ...")
        data = buildRecordingDataset(
            recordingDir,
            windowSeconds,
            stepSeconds,
            maxSensorAgePeriods=maxSensorAgePeriods,
        )
        if not data.empty:
            allRows.append(data)

    if not allRows:
        raise ValueError(f"No labelled recordings could be processed under {datasetDir}")

    return pd.concat(allRows, ignore_index=True)
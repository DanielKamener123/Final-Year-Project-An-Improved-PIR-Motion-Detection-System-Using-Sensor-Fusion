import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from PIR.processing import PIRProcessor, getFeatureNames, getFeatureVector
from Thermopile.thermopile_pipeline import BlobFeatures, ThermopilePipeline


def thermalFeatureNames():
    names = []
    for field in fields(BlobFeatures):
        if field.name == "centroid":
            names.extend(["centroid_row", "centroid_col"])
        else:
            names.append(field.name)
    return names


def flattenThermalFeatures(features):
    values = {}
    for field in fields(BlobFeatures):
        value = getattr(features, field.name)
        if field.name == "centroid":
            values["centroid_row"] = float(value[0])
            values["centroid_col"] = float(value[1])
        else:
            values[field.name] = float(value)
    return values


def emptyThermalFeatures():
    return {name: 0.0 for name in thermalFeatureNames()}


def loadLabels(recordingDir):
    labelsPath = recordingDir / "labels.csv"
    if not labelsPath.exists():
        return None

    labels = pd.read_csv(labelsPath)
    required = {"start_s", "end_s", "human"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{labelsPath} must contain columns {sorted(required)}")
    return labels


def labelAtTime(labels, relativeVideoTime):
    if labels.empty:
        return 0
    active = labels[
        (labels["start_s"] <= relativeVideoTime)
        & (relativeVideoTime < labels["end_s"])
        & (labels["human"] == 1)
    ]
    return int(not active.empty)


def preprocessPIR(pirData, initialBaseline):
    processor = PIRProcessor(initialBaseline=initialBaseline)
    timestamps = pirData["pc_timestamp"].to_numpy(dtype=np.float64)
    voltages = pirData["voltage"].to_numpy(dtype=np.float64)
    relative = np.empty_like(voltages)

    for index, (timestamp, voltage) in enumerate(zip(timestamps, voltages)):
        relative[index] = processor.processSample(timestamp, voltage)

    return timestamps, relative


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

    timestamps = thermalData["pc_timestamp"].to_numpy(dtype=np.float64)
    featureRows = []

    for _, row in thermalData.iterrows():
        frame = row[pixelColumns].to_numpy(dtype=np.float64).reshape(frameShape)
        detections = pipeline.process_frame(frame)
        if detections:
            best = max(detections, key=lambda detection: detection.blob.pixel_count)
            featureRows.append((True, flattenThermalFeatures(best.features)))
        else:
            featureRows.append((False, emptyThermalFeatures()))

    return timestamps, featureRows


def recordingFolders(datasetDir):
    datasetDir = Path(datasetDir)
    if not datasetDir.exists():
        return []
    return sorted(path.parent for path in datasetDir.rglob("metadata.json"))


def buildRecordingDataset(recordingDir, windowSeconds, stepSeconds):
    recordingDir = Path(recordingDir)
    labels = loadLabels(recordingDir)
    if labels is None:
        print(f"Skipping {recordingDir}: labels.csv not found")
        return pd.DataFrame()

    with open(recordingDir / "metadata.json", "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)

    videoStart = metadata.get("video_start_pc_time")
    if videoStart is None:
        raise ValueError(f"{recordingDir}: metadata does not contain video_start_pc_time")

    pirData = pd.read_csv(recordingDir / metadata.get("pir_file", "pir.csv"))
    thermalData = pd.read_csv(recordingDir / metadata.get("thermopile_file", "thermopile.csv"))

    if pirData.empty or thermalData.empty:
        print(f"Skipping {recordingDir}: one of the sensor CSV files is empty")
        return pd.DataFrame()

    pirTimestamps, pirRelative = preprocessPIR(pirData, metadata["pir_baseline_v"])
    thermalTimestamps, thermalRows = processThermopile(thermalData, metadata)

    startTime = max(pirTimestamps[0] + windowSeconds, thermalTimestamps[0])
    endTime = min(pirTimestamps[-1], thermalTimestamps[-1])
    if endTime <= startTime:
        print(f"Skipping {recordingDir}: recording is shorter than the selected window")
        return pd.DataFrame()

    pirNames = getFeatureNames()
    thermalNames = thermalFeatureNames()
    rows = []

    fusionTimes = np.arange(startTime, endTime + 1e-9, stepSeconds)
    for fusionTime in fusionTimes:
        pirStart = np.searchsorted(pirTimestamps, fusionTime - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTime, side="right")
        if pirEnd <= pirStart:
            continue

        pirSamples = list(zip(pirTimestamps[pirStart:pirEnd], pirRelative[pirStart:pirEnd]))
        pirVector = np.asarray(getFeatureVector(pirSamples), dtype=np.float64)
        if len(pirVector) != len(pirNames):
            raise ValueError(
                "PIR getFeatureVector() and getFeatureNames() return different lengths. "
                "Update both together in PIR/processing.py."
            )

        thermalIndex = np.searchsorted(thermalTimestamps, fusionTime, side="right") - 1
        if thermalIndex < 0:
            continue
        detected, thermalFeatures = thermalRows[thermalIndex]

        relativeVideoTime = fusionTime - float(videoStart)
        if relativeVideoTime < 0:
            continue

        row = {
            "recording": str(recordingDir),
            "fusion_pc_timestamp": float(fusionTime),
            "video_time_s": float(relativeVideoTime),
            "window_seconds": float(windowSeconds),
            "thermal_detected": int(detected),
            "human": labelAtTime(labels, relativeVideoTime),
        }
        row.update({f"pir_{name}": float(value) for name, value in zip(pirNames, pirVector)})
        row.update({f"thermal_{name}": float(thermalFeatures[name]) for name in thermalNames})
        rows.append(row)

    return pd.DataFrame(rows)


def buildDataset(datasetDir, windowSeconds, stepSeconds):
    allRows = []
    folders = recordingFolders(datasetDir)
    if not folders:
        raise FileNotFoundError(f"No recording folders found under {datasetDir}")

    for recordingDir in folders:
        print(f"Processing {recordingDir} ...")
        data = buildRecordingDataset(recordingDir, windowSeconds, stepSeconds)
        if not data.empty:
            allRows.append(data)

    if not allRows:
        raise ValueError(f"No labelled recordings could be processed under {datasetDir}")

    return pd.concat(allRows, ignore_index=True)

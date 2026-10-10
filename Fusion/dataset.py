import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIR.processing import (
    preprocessPIR,
    getCurrentFeatureNames,
    getCurrentFeatureVector,
    getHistoryFeatureNames,
    getHistoryFeatureVector,
)
from Thermopile.thermopile_pipeline import (
    processThermopile, thermalFeaturesAtIndex, thermalFeatureNames,
)

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

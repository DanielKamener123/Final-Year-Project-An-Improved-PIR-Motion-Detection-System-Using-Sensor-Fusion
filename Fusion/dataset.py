import json
from pathlib import Path
import numpy as np
import pandas as pd

from PIR.processing import threshold as PIR_THRESHOLD, preprocessPIR, getCurrentFeatureNames, getHistoryFeatureNames, getCurrentFeatureVector, getHistoryFeatureVector
from Thermopile.processing import processThermopile, getThermalFeatures, getThermalFeatureNames

# Gets Labels CSV File
def loadLabels(labelPath):
    labelsPath = labelPath / "labels.csv"
    if not labelsPath.exists():
        return None

    labels = pd.read_csv(labelsPath)
    return labels

# Check If Time Is Within A Labelled Human Interval
def labelAtTime(labels, videoTime):
    if labels.empty:
        return 0

    started = labels["start_s"] <= videoTime
    notEnded = videoTime < labels["end_s"]
    isHuman = labels["human"] == 1
    active = labels[started & notEnded & isHuman]   

    if active.empty:
        return 0
    else: 
        return 1

# Loads All Recroding Folders
def recordingFolders(datasetPath):
    datasetPath = Path(datasetPath)
    if not datasetPath.exists():
        return []
    
    # Returns A List With All Folders Containing Metadata
    return [path.parent for path in datasetPath.rglob("metadata.json")]


def buildRecordingDataset(recordingPath, windowSeconds):
    recordingPath = Path(recordingPath)
    labels = loadLabels(recordingPath)

    # Makes Sure There Is A Label File
    if labels is None:
        print(f"Skipping {recordingPath}: labels not found")
        return pd.DataFrame()

    # Opens Metadata File
    with open(recordingPath / "metadata.json", "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)

    pirData = pd.read_csv(recordingPath / "pir.csv")
    thermalData = pd.read_csv(recordingPath / "thermopile.csv")

    # Makes Sure Both PIR And Thermopile Data Is Populated
    if pirData.empty or thermalData.empty:
        print(f"Skipping {recordingPath}: one of the sensor CSV files is empty")
        return pd.DataFrame()

    # Partial recordings can still be used after a dropout, but do not fuse
    # across gaps: stale PIR/thermal values would contaminate training.
    pirData = pirData.sort_values("time_s")
    thermalData = thermalData.sort_values("time_s")
    # Gets Timestamps With Relevant Data Associated With That Timestamp
    pirTimestamps, pirRelative = preprocessPIR(pirData, metadata["pir_baseline_v"], windowSeconds)
    thermalTimestamps, thermalRows = processThermopile(thermalData, metadata, windowSeconds)

    # No Fusion Decision Is Made Until The PIR History Buffer Is Full
    startTime = pirTimestamps[0] + windowSeconds
    endTime = pirTimestamps[-1]

    pirHistoryFeatureNames = getHistoryFeatureNames()
    pirCurrentFeatureNames = getCurrentFeatureNames()
    thermalFeatureNames = getThermalFeatureNames()
    rows = []

    # Get Valid Thermopile Timestamps
    validThermalTimestamp = np.flatnonzero((thermalTimestamps >= startTime)& (thermalTimestamps <= endTime))

    previousFusionTimestamp = None
    maximumThermalGap = 1.0
    for thermalIndex in validThermalTimestamp:
        # Get Master Fusion Timestamp
        fusionTimestamp = float(thermalTimestamps[thermalIndex])
        if thermalIndex and fusionTimestamp - thermalTimestamps[thermalIndex - 1] > maximumThermalGap:
            previousFusionTimestamp = None
            continue

         # Skip Corrupted Section Of HomeGardenTam Recording
        if recordingPath.name == "HomeGardenTam":
            if 264.0 <= fusionTimestamp <= 342.0:
                continue

        # The simple PIR detector runs at the PIR sampling rate, not merely at
        # the most recent sample before a thermal frame. Preserve whether ANY
        # sample exceeded threshold since the preceding fusion decision.
        # This is an interval-wise OR, not a history-window persistence model.
        interval_start = (previousFusionTimestamp if previousFusionTimestamp is not None
                          else max(float(pirTimestamps[0]), fusionTimestamp - 1.0 / 2.0))
        interval_begin = np.searchsorted(pirTimestamps, interval_start, side="right")
        interval_end = np.searchsorted(pirTimestamps, fusionTimestamp, side="right")
        interval_values = pirRelative[interval_begin:interval_end]
        interval_peak = float(np.max(np.abs(interval_values))) if len(interval_values) else 0.0
        interval_active = int(interval_peak >= float(PIR_THRESHOLD))
        previousFusionTimestamp = fusionTimestamp

        # Get PIR Window Timestamps  
        pirStart = np.searchsorted(pirTimestamps, fusionTimestamp - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTimestamp, side="right")
        
        # If No PIR Samples Exist, The Thermopile Frame Is Skipped
        if pirEnd <= pirStart:
            continue
        selectedPIRTimes = pirTimestamps[pirStart:pirEnd]
        # Require coverage throughout the window, with no excessive gap.
        nominalPirPeriod = 1.0 / max(float(metadata.get("pir_sample_rate_hz") or 20.0), 1.0)
        if (selectedPIRTimes[0] > fusionTimestamp - windowSeconds + 2 * nominalPirPeriod or
            selectedPIRTimes[-1] < fusionTimestamp - 2 * nominalPirPeriod or
            (len(selectedPIRTimes) > 1 and np.max(np.diff(selectedPIRTimes)) > max(0.3, 3 * nominalPirPeriod))):
            continue

        # Gets Timestamp And Sample fOR Window Period
        pirSamples = list(zip(pirTimestamps[pirStart:pirEnd], pirRelative[pirStart:pirEnd]))

        # Get PIR History and Current Features
        pirHistoryVector = np.asarray(getHistoryFeatureVector(pirSamples), dtype=np.float64)
        pirCurrentVector = (np.asarray(getCurrentFeatureVector(pirRelative[pirEnd - 1]), dtype=np.float64))

        # Get Thermal Features
        thermalFeatures = getThermalFeatures(thermalRows,int(thermalIndex))

        relativeVideoTime = float(fusionTimestamp)

        row = {
            "recording": str(recordingPath),
            "fusionTime": float(fusionTimestamp),
            "videoTime": relativeVideoTime,
            "windowSeconds": float(windowSeconds),
            "human": labelAtTime(labels, relativeVideoTime),
            "pirThresholdIntervalActive": interval_active,
            "pirThresholdIntervalPeak": interval_peak,
        }
        row.update({f"pir{name.capitalize()}": float(value) for name, value in zip(pirCurrentFeatureNames, pirCurrentVector)})
        row.update({f"pir{name.capitalize()}": float(value) for name, value in zip(pirHistoryFeatureNames, pirHistoryVector)})
        row.update({f"thermal{name.capitalize()}": float(thermalFeatures[name]) for name in thermalFeatureNames})
        rows.append(row)

    return pd.DataFrame(rows)


def buildDataset(datasetPath, windowSeconds):
    allRows = []
    folders = recordingFolders(datasetPath)

    for recordingPath in folders:
        data = buildRecordingDataset(recordingPath,windowSeconds)
        if not data.empty:
            allRows.append(data)

    return pd.concat(allRows, ignore_index=True)
import json
from pathlib import Path
import numpy as np
import pandas as pd

from PIR.processing import threshold as PIR_THRESHOLD, preprocessPIR, getHistoryFeatureNames, getHistoryFeatureVector
from Thermopile.processing import ThermopilePipeline, getThermalFeatureNames

# Model Input Columns: PIR history features followed by the thermal features
def getFusedFeatureNames():
    return getHistoryFeatureNames() + getThermalFeatureNames()

# Gets Labels CSV File
def loadLabels(labelPath):
    labelsPath = labelPath / "labels.csv"
    if not labelsPath.exists():
        return None

    labels = pd.read_csv(labelsPath)
    return labels

# Gets Optional Excluded Intervals (start_s,end_s) For Corrupted Sections Of A Recording
def loadExcludedIntervals(recordingPath):
    excludedPath = Path(recordingPath) / "excluded.csv"
    if not excludedPath.exists():
        return []
    excluded = pd.read_csv(excludedPath)
    return [(float(start), float(end)) for start, end in excluded[["start_s", "end_s"]].itertuples(index=False, name=None)]

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

# Runs A Recording's Thermopile Frames Through The Pipeline
# Returns the frame timestamps and one thermal feature dict per frame.
def processThermopile(thermalData, metadata, windowSeconds=2.0):
    frameShape = tuple(int(value) for value in metadata["thermal_frame_shape"])
    frameLength = int(np.prod(frameShape))
    pixelColumns = [f"pixel_{i}" for i in range(frameLength)]

    missing = [column for column in pixelColumns if column not in thermalData.columns]
    if missing:
        raise ValueError(f"Thermopile CSV is missing {len(missing)} pixel columns")

    # Start from the background saved on the ROCK during calibration
    pipeline = ThermopilePipeline(frame_shape=frameShape, window_seconds=windowSeconds)
    pipeline.load_calibration(metadata["thermal_background"], metadata["thermal_sigma"])

    timestamps = thermalData["time_s"].to_numpy(dtype=np.float64)
    frames = thermalData[pixelColumns].to_numpy(dtype=np.float64).reshape(-1, *frameShape)
    thermalRows = [pipeline.process(frame, timestamp) for frame, timestamp in zip(frames, timestamps)]

    return timestamps, thermalRows

# Loads All Recroding Folders
def recordingFolders(datasetPath):
    datasetPath = Path(datasetPath)
    if not datasetPath.exists():
        return []
    
    # Returns A List With All Folders Containing Metadata (sorted so every run uses the same order)
    return sorted(path.parent for path in datasetPath.rglob("metadata.json"))


# requireLabels=False also builds unlabelled recordings (every "human" value is 0);
# used to make predictions on recordings that are not labelled yet.
def buildRecordingDataset(recordingPath, windowSeconds, requireLabels=True):
    recordingPath = Path(recordingPath)
    labels = loadLabels(recordingPath)

    # Makes Sure There Is A Label File
    if labels is None:
        if requireLabels:
            print(f"Skipping {recordingPath}: labels not found")
            return pd.DataFrame()
        labels = pd.DataFrame(columns=["start_s", "end_s", "human"])

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
    thermalFeatureNames = getThermalFeatureNames()
    rows = []

    # Get Valid Thermopile Timestamps
    validThermalTimestamp = np.flatnonzero((thermalTimestamps >= startTime)& (thermalTimestamps <= endTime))

    excludedIntervals = loadExcludedIntervals(recordingPath)
    previousFusionTimestamp = None
    maximumThermalGap = 1.0
    for thermalIndex in validThermalTimestamp:
        # Get Master Fusion Timestamp
        fusionTimestamp = float(thermalTimestamps[thermalIndex])
        if thermalIndex and fusionTimestamp - thermalTimestamps[thermalIndex - 1] > maximumThermalGap:
            previousFusionTimestamp = None
            continue

        # Skip Sections Listed In The Recording's excluded.csv (e.g. sensor dropouts)
        if any(start <= fusionTimestamp <= end for start, end in excludedIntervals):
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

        # Get Thermal Features
        thermalFeatures = thermalRows[int(thermalIndex)]

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
        row.update({name: float(value) for name, value in zip(pirHistoryFeatureNames, pirHistoryVector)})
        row.update({name: float(thermalFeatures[name]) for name in thermalFeatureNames})
        rows.append(row)

    return pd.DataFrame(rows)


def buildDataset(datasetPath, windowSeconds):
    allRows = []
    folders = recordingFolders(datasetPath)

    for recordingPath in folders:
        data = buildRecordingDataset(recordingPath,windowSeconds)
        if not data.empty:
            allRows.append(data)

    if not allRows:
        raise ValueError(f"No labelled recordings found in {datasetPath}. Every recording needs a labels.csv "
                         "(and labels(outbound).csv for evaluation); a header-only file means nobody was present.")
    return pd.concat(allRows, ignore_index=True)
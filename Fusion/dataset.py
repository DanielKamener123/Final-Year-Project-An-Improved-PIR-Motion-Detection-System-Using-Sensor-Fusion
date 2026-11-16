import json
from pathlib import Path
import numpy as np
import pandas as pd

from PIR.processing import threshold as PIR_THRESHOLD, preprocessPIR, getHistoryFeatureNames, getHistoryFeatureVector
from Thermopile.processing import ThermopilePipeline, getThermalFeatureNames

# PIR features followed by thermal features
def getFusedFeatureNames():
    return getHistoryFeatureNames() + getThermalFeatureNames()

# Load human labels, if available
def loadLabels(labelPath):
    labelsPath = labelPath / "labels.csv"
    if not labelsPath.exists():
        return None

    labels = pd.read_csv(labelsPath)
    return labels

# Load sections to skip
def loadExcludedIntervals(recordingPath):
    excludedPath = Path(recordingPath) / "excluded.csv"
    if not excludedPath.exists():
        return []
    excluded = pd.read_csv(excludedPath)
    return [(float(start), float(end)) for start, end in excluded[["start_s", "end_s"]].itertuples(index=False, name=None)]

# Human present at this timestamp: 1 or 0
def labelAtTime(labels, videoTime):
    if labels.empty:
        return 0

    # Find active human intervals
    started = labels["start_s"] <= videoTime
    notEnded = videoTime < labels["end_s"]
    isHuman = labels["human"] == 1
    active = labels[started & notEnded & isHuman]   

    if active.empty:
        return 0
    else: 
        return 1

# Process all thermal frames into features
def processThermopile(thermalData, metadata, windowSeconds=2.0):
    # Get frame size and pixel columns
    frameShape = tuple(int(value) for value in metadata["thermal_frame_shape"])
    frameLength = int(np.prod(frameShape))
    pixelColumns = [f"pixel_{i}" for i in range(frameLength)]

    # Check for missing pixels
    missing = [column for column in pixelColumns if column not in thermalData.columns]
    if missing:
        raise ValueError(f"Thermopile CSV is missing {len(missing)} pixel columns")

    # Restore recorded background calibration
    pipeline = ThermopilePipeline(frame_shape=frameShape, window_seconds=windowSeconds)
    pipeline.load_calibration(metadata["thermal_background"], metadata["thermal_sigma"])

    # Process frames in timestamp order
    timestamps = thermalData["time_s"].to_numpy(dtype=np.float64)
    frames = thermalData[pixelColumns].to_numpy(dtype=np.float64).reshape(-1, *frameShape)
    thermalRows = [pipeline.process(frame, timestamp) for frame, timestamp in zip(frames, timestamps)]

    return timestamps, thermalRows

# Find recording folders in a fixed order
def recordingFolders(datasetPath):
    datasetPath = Path(datasetPath)
    if not datasetPath.exists():
        return []
    
    return sorted(path.parent for path in datasetPath.rglob("metadata.json"))


# Build fused data for one recording
# Unlabelled mode assigns human = 0
def buildRecordingDataset(recordingPath, windowSeconds, requireLabels=True):
    recordingPath = Path(recordingPath)
    labels = loadLabels(recordingPath)

    # Skip missing labels when required
    if labels is None:
        if requireLabels:
            print(f"Skipping {recordingPath}: labels not found")
            return pd.DataFrame()
        labels = pd.DataFrame(columns=["start_s", "end_s", "human"])

    # Load calibration and recording settings
    with open(recordingPath / "metadata.json", "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)

    # Load recorded sensor data
    pirData = pd.read_csv(recordingPath / "pir.csv")
    thermalData = pd.read_csv(recordingPath / "thermopile.csv")

    # Skip empty sensor files
    if pirData.empty or thermalData.empty:
        print(f"Skipping {recordingPath}: one of the sensor CSV files is empty")
        return pd.DataFrame()

    # Sort samples before processing
    pirData = pirData.sort_values("time_s")
    thermalData = thermalData.sort_values("time_s")
    # Prepare PIR samples and thermal features
    pirTimestamps, pirRelative = preprocessPIR(pirData, metadata["pir_baseline_v"], windowSeconds)
    thermalTimestamps, thermalRows = processThermopile(thermalData, metadata, windowSeconds)

    # Wait for a full PIR history window
    startTime = pirTimestamps[0] + windowSeconds
    endTime = pirTimestamps[-1]

    # Set up fused feature columns
    pirHistoryFeatureNames = getHistoryFeatureNames()
    thermalFeatureNames = getThermalFeatureNames()
    rows = []

    # Thermal frames within the PIR recording
    validThermalTimestamp = np.flatnonzero((thermalTimestamps >= startTime)& (thermalTimestamps <= endTime))

    # Dropout checks and fusion timing
    excludedIntervals = loadExcludedIntervals(recordingPath)
    previousFusionTimestamp = None
    maximumThermalGap = 1.0
    for thermalIndex in validThermalTimestamp:
        # Use thermal frame as fusion timestamp
        fusionTimestamp = float(thermalTimestamps[thermalIndex])
        # Skip frames after large thermal gaps
        if thermalIndex and fusionTimestamp - thermalTimestamps[thermalIndex - 1] > maximumThermalGap:
            previousFusionTimestamp = None
            continue

        # Skip manually excluded intervals
        if any(start <= fusionTimestamp <= end for start, end in excludedIntervals):
            continue

        # PIR baseline: any threshold crossing since last fusion
        interval_start = (previousFusionTimestamp if previousFusionTimestamp is not None
                          else max(float(pirTimestamps[0]), fusionTimestamp - 1.0 / 2.0))
        interval_begin = np.searchsorted(pirTimestamps, interval_start, side="right")
        interval_end = np.searchsorted(pirTimestamps, fusionTimestamp, side="right")
        interval_values = pirRelative[interval_begin:interval_end]
        # Largest absolute PIR value in the interval
        interval_peak = float(np.max(np.abs(interval_values))) if len(interval_values) else 0.0
        interval_active = int(interval_peak >= float(PIR_THRESHOLD))
        previousFusionTimestamp = fusionTimestamp

        # Select PIR history ending at this frame
        pirStart = np.searchsorted(pirTimestamps, fusionTimestamp - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTimestamp, side="right")
        
        # Skip missing PIR history
        if pirEnd <= pirStart:
            continue
        selectedPIRTimes = pirTimestamps[pirStart:pirEnd]
        # Reject incomplete windows and large PIR gaps
        nominalPirPeriod = 1.0 / max(float(metadata.get("pir_sample_rate_hz") or 20.0), 1.0)
        if (selectedPIRTimes[0] > fusionTimestamp - windowSeconds + 2 * nominalPirPeriod or
            selectedPIRTimes[-1] < fusionTimestamp - 2 * nominalPirPeriod or
            (len(selectedPIRTimes) > 1 and np.max(np.diff(selectedPIRTimes)) > max(0.3, 3 * nominalPirPeriod))):
            continue

        # Pair PIR timestamps with samples
        pirSamples = list(zip(pirTimestamps[pirStart:pirEnd], pirRelative[pirStart:pirEnd]))

        # Extract PIR history features
        pirHistoryVector = np.asarray(getHistoryFeatureVector(pirSamples), dtype=np.float64)

        # Match thermal features to fusion frame
        thermalFeatures = thermalRows[int(thermalIndex)]

        # Match timestamp to human labels
        relativeVideoTime = float(fusionTimestamp)

        # Combine labels, baseline and sensor features
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

    # One row per valid fusion timestamp
    return pd.DataFrame(rows)


# Combine all labelled recordings
def buildDataset(datasetPath, windowSeconds):
    allRows = []
    folders = recordingFolders(datasetPath)

    # Build each recording separately
    for recordingPath in folders:
        data = buildRecordingDataset(recordingPath,windowSeconds)
        if not data.empty:
            allRows.append(data)

    # Require at least one usable recording
    if not allRows:
        raise ValueError(f"No labelled recordings found in {datasetPath}. Every recording needs a labels.csv "
                         "(and labels(outbound).csv for evaluation); a header-only file means nobody was present.")
    return pd.concat(allRows, ignore_index=True)
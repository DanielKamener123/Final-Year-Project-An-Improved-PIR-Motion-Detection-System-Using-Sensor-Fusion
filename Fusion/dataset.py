import json
from pathlib import Path
import numpy as np
import pandas as pd

from PIR.processing import preprocessPIR, getCurrentFeatureNames, getHistoryFeatureNames, getCurrentFeatureVector, getHistoryFeatureVector
from Thermopile.thermopile_pipeline import processThermopile, getThermalFeatures, getThermalFeatureNames

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

    # Gets Timestamps With Relevant Data Associated With That Timestamp
    pirTimestamps, pirRelative = preprocessPIR(pirData, metadata["pir_baseline_v"], windowSeconds)
    thermalTimestamps, thermalRows = processThermopile(thermalData, metadata)

    # No Fusion Decision Is Made Until The PIR History Buffer Is Full
    startTime = pirTimestamps[0] + windowSeconds
    endTime = pirTimestamps[-1]

    pirHistoryFeatureNames = getHistoryFeatureNames()
    pirCurrentFeatureNames = getCurrentFeatureNames()
    thermalFeatureNames = getThermalFeatureNames()
    rows = []

    # Get Valid Thermopile Timestamps
    validThermalTimestamp = np.flatnonzero((thermalTimestamps >= startTime)& (thermalTimestamps <= endTime))

    for thermalIndex in validThermalTimestamp:
        # Get Master Fusion Timestamp
        fusionTimestamp = float(thermalTimestamps[thermalIndex])

        # Get PIR Window Timestamps  
        pirStart = np.searchsorted(pirTimestamps, fusionTimestamp - windowSeconds, side="right")
        pirEnd = np.searchsorted(pirTimestamps, fusionTimestamp, side="right")
        
        # If No PIR Samples Exist, The Thermopile Frame Is Skipped
        if pirEnd <= pirStart:
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
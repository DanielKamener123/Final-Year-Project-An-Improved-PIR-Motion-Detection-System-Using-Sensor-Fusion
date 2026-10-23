"""Offline PIR / thermal feature-fusion runs.

  python offlineRun.py --compare                 every scikit-learn classifier at every
                                                 window vs thresholded PIR (validation)
  python offlineRun.py --train --window 1.5      train the first-principles Random Forest,
                                                 tune its alarm thresholds on validation and
                                                 export it for the real-time system
  python offlineRun.py --test                    final verification of the exported model
                                                 on the testing recordings

Each run writes report.txt and PNG figures to Processed/<name>/.
"""
from pathlib import Path
import argparse
import re

import numpy as np

from Fusion.dataset import buildDataset, getFusedFeatureNames
from Fusion.evaluation import (LABEL_COLUMN, addOutboundLabels, pirBaselineStates,
                               runStateMachine, tuneThresholds)
from Fusion.randomForest import RandomForest
from Fusion.report import writeComparisonReport, writeEvaluationReport


# Data Path
trainingDataset = Path("Datasets/training")
validationDataset = Path("Datasets/validation")
testingDataset = Path("Datasets/testing")
outputPath = Path("Processed")
modelPath = outputPath / "realtime" / "fusion_rf.json"


# Output Folder Name For This Run (never overwrites an earlier run)
def outputFolder(runName):
    if not runName:
        runName = input("Name this results folder: ").strip()
    safeName = re.sub(r"[^A-Za-z0-9_.-]+", "_", (runName or "").strip()).strip("._")
    if not safeName:
        raise ValueError("Folder name must contain letters or numbers")
    folder = outputPath / safeName
    if folder.exists():
        raise FileExistsError(f"Results folder already exists: {folder}. Choose another name.")
    return safeName, folder


# Builds A Labelled Dataset; Evaluation Data Also Gets The Outbound Labels
def loadData(path, window, evaluation=False):
    print(f"Building features from {path} ({window:g} s window)...")
    data = buildDataset(path, windowSeconds=window)
    print(f"  {len(data)} fusion intervals from {data['recording'].nunique()} recordings")
    return addOutboundLabels(data) if evaluation else data


def runCompare(runName, folder):
    from Fusion.models import compareModels, bestConfiguration
    results = compareModels(trainingDataset, validationDataset)
    best = bestConfiguration(results)
    bestForest = bestConfiguration(results, "Random Forest")
    text = writeComparisonReport(runName, results, best, bestForest, folder, trainingDataset, validationDataset)
    print("\n" + text)


def runTrain(runName, folder, window):
    from Fusion.models import buildModels, confidenceScores, featureMatrix
    names = getFusedFeatureNames()
    trainData = loadData(trainingDataset, window)
    evalData = loadData(validationDataset, window, evaluation=True)

    print("Training the first-principles Random Forest...")
    forest = RandomForest().fit(featureMatrix(trainData).to_numpy(), trainData[LABEL_COLUMN].to_numpy(),
                                names, progress=print)
    scores = forest.predictProba(featureMatrix(evalData).to_numpy())
    print("Tuning alarm thresholds on validation...")
    tuned = tuneThresholds(evalData, scores)
    path = forest.save(modelPath, window, tuned["thresholds"])
    print(f"Saved real-time model to {path}")
    # Keep the playback review files in step with the newest model
    from Tools.export_review_predictions import exportAll
    exportAll(modelPath)

    # Same settings in scikit-learn, as a check on the first-principles implementation
    print("Training the scikit-learn Random Forest for comparison...")
    library = buildModels()["Random Forest"].fit(featureMatrix(trainData), trainData[LABEL_COLUMN])
    libraryScores = confidenceScores(library, featureMatrix(evalData))
    libraryThresholds = tuneThresholds(evalData, libraryScores)["thresholds"]

    systems = {
        "Thresholded PIR": {"states": pirBaselineStates(evalData)},
        "First-principles RF": {"scores": scores, "thresholds": tuned["thresholds"],
                                "states": runStateMachine(evalData, scores, tuned["thresholds"])},
        "scikit-learn RF": {"scores": libraryScores, "thresholds": libraryThresholds,
                            "states": runStateMachine(evalData, libraryScores, libraryThresholds)},
    }
    agreement = np.corrcoef(scores, libraryScores)[0, 1]
    text, _ = writeEvaluationReport(
        runName, evalData, systems, folder,
        f"Trained on {trainingDataset}, evaluated on {validationDataset} ({window:g} s window). "
        "Alarm thresholds tuned on this data, so these numbers are optimistic.",
        f"{window:g} s window, validation",
        featureNames=names, importance=forest.importance,
        extraLines=[f"First-principles vs scikit-learn RF confidence correlation: {agreement:.3f}",
                    f"Exported model: {modelPath}"])
    print("\n" + text)


def runTest(runName, folder):
    if not modelPath.is_file():
        raise FileNotFoundError(f"No trained model at {modelPath}; run offlineRun.py --train first")
    forest = RandomForest.load(modelPath)
    evalData = loadData(testingDataset, forest.window, evaluation=True)
    scores = forest.predictProba(evalData[forest.features].to_numpy(dtype=float))
    systems = {
        "Thresholded PIR": {"states": pirBaselineStates(evalData)},
        "First-principles RF": {"scores": scores, "thresholds": forest.thresholds,
                                "states": runStateMachine(evalData, scores, forest.thresholds)},
    }
    text, _ = writeEvaluationReport(
        runName, evalData, systems, folder,
        f"FINAL VERIFICATION: exported model {modelPath} with frozen alarm thresholds, "
        f"evaluated on {testingDataset} ({forest.window:g} s window).",
        f"{forest.window:g} s window, testing",
        featureNames=forest.features, importance=forest.importance)
    print("\n" + text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline PIR / thermal feature-fusion runs")
    parser.add_argument("--name", help="Output folder name, e.g. Compare_22Oct")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compare", action="store_true",
                       help="Compare every classifier at every window on the validation recordings")
    group.add_argument("--train", action="store_true",
                       help="Train, tune and export the first-principles Random Forest")
    group.add_argument("--test", action="store_true",
                       help=f"Final verification of {modelPath} on the testing recordings")
    parser.add_argument("--window", type=float, default=1.5, help="Fusion window for --train (seconds)")
    args = parser.parse_args()

    runName, folder = outputFolder(args.name)
    if args.compare:
        runCompare(runName, folder)
    elif args.train:
        runTrain(runName, folder, args.window)
    else:
        runTest(runName, folder)
    print(f"\nSaved report and figures to: {folder}")

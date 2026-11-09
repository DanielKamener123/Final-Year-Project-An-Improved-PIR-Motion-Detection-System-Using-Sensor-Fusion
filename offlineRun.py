"""Offline PIR / thermal feature-fusion runs.

  python offlineRun.py --compare                 every scikit-learn classifier at every
                                                 window vs thresholded PIR (validation)
  python offlineRun.py --train --window 1.0      train the first-principles Gradient Boosting,
                                                 tune its alarm thresholds on validation and
                                                 export it for the real-time system; also trains
                                                 the PIR-only backup (same data, PIR features only)
  python offlineRun.py --test                    final verification of the exported model
                                                 on the testing recordings
  python offlineRun.py --demo --name Demo_Tue    DEMO: train on every recording (no split), choose the
                                                 alarm settings leave-one-session-out, export the fused
                                                 model and the PIR-only backup (Fusion/demoTraining.py)
  python offlineRun.py --summary                 recordings, durations and labelled minutes
                                                 of every split (Section 3 data paragraph)

Each run writes report.txt and PNG figures to Processed/<name>/.
--test also writes the Section 4 qualification tests: qt*.png figures and
LaTeX tables (tables/*.tex). Testing folders must be named
<Site>_<Test>_<Day|Night>, see Fusion/qualification.py.
"""
from pathlib import Path
import argparse
import re

import numpy as np

from Fusion.dataset import buildDataset, getFusedFeatureNames
from PIR.processing import getHistoryFeatureNames
from Fusion.evaluation import (LABEL_COLUMN, addOutboundLabels, pirBaselineStates,
                               runStateMachine, tuneThresholds)
from Fusion.gradientBoosting import GradientBoosting
from Fusion.report import writeComparisonReport, writeEvaluationReport, figImplementationCheck
from Fusion.qualification import writeQualificationReport, datasetSummary


# Data Path
trainingDataset = Path("Datasets/training")
validationDataset = Path("Datasets/validation")
testingDataset = Path("Datasets/testing")
# Recordings from other sites: not used for training, validation or testing of the
# site-trained model, only listed in the dataset summary (development data)
otherSitesDataset = Path("Datasets/other_sites")
outputPath = Path("Processed")
modelPath = outputPath / "realtime" / "fusion_gb.json"
# PIR-only backup: used live ONLY while no thermal frames are received
backupModelPath = outputPath / "realtime" / "pir_backup_gb.json"


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
    bestBoosting = bestConfiguration(results, "Gradient Boosting")
    text = writeComparisonReport(runName, results, best, bestBoosting, folder, trainingDataset, validationDataset)
    print("\n" + text)


def runTrain(runName, folder, window):
    from Fusion.models import buildModels, confidenceScores, featureMatrix
    names = getFusedFeatureNames()
    trainData = loadData(trainingDataset, window)
    evalData = loadData(validationDataset, window, evaluation=True)

    print("Training the first-principles Gradient Boosting...")
    model = GradientBoosting().fit(featureMatrix(trainData).to_numpy(), trainData[LABEL_COLUMN].to_numpy(),
                                   names, progress=print)
    scores = model.predictProba(featureMatrix(evalData).to_numpy())
    print("Tuning alarm thresholds on validation...")
    tuned = tuneThresholds(evalData, scores)
    path = model.save(modelPath, window, tuned["thresholds"])
    print(f"Saved real-time model to {path}")

    # PIR-only backup: same training rows, same settings, same threshold tuning,
    # but only the PIR feature columns (the live system uses it only without thermal frames)
    backupNames = getHistoryFeatureNames()
    print("Training the first-principles PIR-only backup Gradient Boosting...")
    backup = GradientBoosting().fit(trainData[backupNames].to_numpy(dtype=float),
                                    trainData[LABEL_COLUMN].to_numpy(), backupNames, progress=print)
    backupScores = backup.predictProba(evalData[backupNames].to_numpy(dtype=float))
    print("Tuning PIR-only backup alarm thresholds on validation...")
    backupTuned = tuneThresholds(evalData, backupScores)
    backupPath = backup.save(backupModelPath, window, backupTuned["thresholds"])
    print(f"Saved PIR-only backup model to {backupPath}")
    # Keep the playback review files in step with the newest model
    from Tools.export_review_predictions import exportAll
    exportAll(modelPath)

    # Same settings in scikit-learn, as a check on the first-principles implementation
    print("Training the scikit-learn Gradient Boosting for comparison...")
    library = buildModels()["Gradient Boosting"].fit(featureMatrix(trainData), trainData[LABEL_COLUMN])
    libraryScores = confidenceScores(library, featureMatrix(evalData))
    libraryThresholds = tuneThresholds(evalData, libraryScores)["thresholds"]

    systems = {
        "Thresholded PIR": {"states": pirBaselineStates(evalData)},
        "First-principles GB": {"scores": scores, "thresholds": tuned["thresholds"],
                                "states": runStateMachine(evalData, scores, tuned["thresholds"])},
        "scikit-learn GB": {"scores": libraryScores, "thresholds": libraryThresholds,
                            "states": runStateMachine(evalData, libraryScores, libraryThresholds)},
    }
    agreement = np.corrcoef(scores, libraryScores)[0, 1]
    difference = np.abs(scores - libraryScores)
    text, metrics = writeEvaluationReport(
        runName, evalData, systems, folder,
        f"Trained on {trainingDataset}, evaluated on {validationDataset} ({window:g} s window). "
        "Alarm thresholds tuned on this data, so these numbers are optimistic.",
        f"{window:g} s window, validation",
        featureNames=names, importance=model.importance,
        extraLines=[f"First-principles vs scikit-learn GB confidence correlation: {agreement:.5f}",
                    f"First-principles vs scikit-learn GB confidence difference: mean {difference.mean():.2e}, "
                    f"max {difference.max():.2e}",
                    f"Exported model: {modelPath}"])
    # Report figure for the real-time implementation check (Section 3)
    figImplementationCheck(scores, libraryScores, metrics, "First-principles GB", "scikit-learn GB", folder)
    print("\n" + text)


def runTest(runName, folder):
    if not modelPath.is_file():
        raise FileNotFoundError(f"No trained model at {modelPath}; run offlineRun.py --train first")
    model = GradientBoosting.load(modelPath)
    evalData = loadData(testingDataset, model.window, evaluation=True)
    scores = model.predictProba(evalData[model.features].to_numpy(dtype=float))
    # One run of the frozen model, split into the report's qualification tests
    text = writeQualificationReport(
        runName, evalData, scores, model, folder,
        f"FINAL VERIFICATION: exported model {modelPath} with frozen alarm thresholds, "
        f"evaluated on {testingDataset} ({model.window:g} s window).",
        {"Training": trainingDataset, "Validation": validationDataset, "Testing": testingDataset})
    print("\n" + text)


def runSummary(runName, folder):
    folder.mkdir(parents=True, exist_ok=False)
    lines, table, tex = datasetSummary(
        {"Training": trainingDataset, "Validation": validationDataset, "Testing": testingDataset,
         "Other sites": otherSitesDataset})
    text = "\n".join([f"DATASET SUMMARY: {runName}", *lines]) + "\n"
    (folder / "report.txt").write_text(text, encoding="utf-8")
    table.to_csv(folder / "recordings.csv", index=False)
    if tex:
        (folder / "data_split.tex").write_text(tex, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline PIR / thermal feature-fusion runs")
    parser.add_argument("--name", help="Output folder name, e.g. Compare_22Oct")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compare", action="store_true",
                       help="Compare every classifier at every window on the validation recordings")
    group.add_argument("--train", action="store_true",
                       help="Train, tune and export the first-principles Gradient Boosting")
    group.add_argument("--test", action="store_true",
                       help=f"Final verification of {modelPath} on the testing recordings")
    group.add_argument("--demo", action="store_true",
                       help="Train on every recording for the demo and export both real-time models")
    group.add_argument("--summary", action="store_true",
                       help="Recordings, durations and labelled minutes of every dataset split")
    parser.add_argument("--window", type=float, default=None,
                        help="Fusion window in seconds (--train default 1.0, --demo default 2.0)")
    args = parser.parse_args()

    runName, folder = outputFolder(args.name)
    if args.compare:
        runCompare(runName, folder)
    elif args.train:
        runTrain(runName, folder, args.window if args.window is not None else 1.0)
    elif args.demo:
        from Fusion.demoTraining import runDemoTraining, DEMO_WINDOW_S
        print("\n" + runDemoTraining(runName, folder, args.window if args.window is not None else DEMO_WINDOW_S,
                                     modelPath, backupModelPath))
    elif args.summary:
        runSummary(runName, folder)
    else:
        runTest(runName, folder)
    print(f"\nSaved report and figures to: {folder}")

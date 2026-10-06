from pathlib import Path

import pandas as pd

from Fusion.dataset import buildDataset
from Fusion.models import runModels


# Offline Fusion Configuration
HISTORY_WINDOWS_SECONDS = [1.0, 1.5, 2.0, 2.5, 3.0]

# A Sensor Is Marked Invalid If Its Latest Reading Is Older Than This Many Expected Sample Periods
MAX_PIR_AGE_PERIODS = 2.0

TRAIN_DATASET = Path("Datasets/training")
VALIDATION_DATASET = Path("Datasets/validation")

OUTPUT_ROOT = Path("Processed")


def main():
    allResults = []

    for windowSeconds in HISTORY_WINDOWS_SECONDS:
        outputDir = OUTPUT_ROOT / f"window_{windowSeconds:.2f}s"
        outputDir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("Fusion configuration")
        print(f"  PIR window length : {windowSeconds:.3f} s")
        print(f"  Training data     : {TRAIN_DATASET}")
        print(f"  Validation data   : {VALIDATION_DATASET}")
        print("=" * 70)
        print()

        print("Building training feature dataset from raw recordings...")
        trainData = buildDataset(
            TRAIN_DATASET,
            windowSeconds=windowSeconds,
            maxPirAgePeriods=MAX_PIR_AGE_PERIODS,
        )
        trainPath = outputDir / "train_features.csv"
        trainData.to_csv(trainPath, index=False)
        print(f"Saved {len(trainData)} training rows to {trainPath}")

        print("\nBuilding validation feature dataset from raw recordings...")
        validationData = buildDataset(
            VALIDATION_DATASET,
            windowSeconds=windowSeconds,
            maxPirAgePeriods=MAX_PIR_AGE_PERIODS,
        )
        validationPath = outputDir / "validation_features.csv"
        validationData.to_csv(validationPath, index=False)
        print(f"Saved {len(validationData)} validation rows to {validationPath}")

        print("\nRunning machine-learning algorithms...")
        resultsPath = outputDir / "model_comparison_results.csv"
        results = runModels(trainData, validationData, outputPath=resultsPath)
        results.insert(0, "History Window (s)", float(windowSeconds))
        results.to_csv(resultsPath, index=False)
        allResults.append(results)
        print(f"\nResults saved to {resultsPath}")

    if allResults:
        comparison = pd.concat(allResults, ignore_index=True)
        comparisonPath = OUTPUT_ROOT / "history_window_comparison.csv"
        comparison.to_csv(comparisonPath, index=False)
        print(f"\nAll history-window results saved to {comparisonPath}")


if __name__ == "__main__":
    main()

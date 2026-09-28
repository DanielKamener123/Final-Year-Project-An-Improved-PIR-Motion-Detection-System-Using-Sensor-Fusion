from pathlib import Path

import pandas as pd

from Fusion.dataset import buildDataset
from Fusion.models import runModels


# Offline Fusion Configuration
HISTORY_WINDOWS_SECONDS = [step / 2 for step in range(1, 21)]
WINDOW_STEP_SECONDS = 0.25

# A Sensor Is Marked Invalid If Its Latest Reading Is Older Than This Many Expected Sample Periods
MAX_SENSOR_AGE_PERIODS = 2.0

TRAIN_DATASET = Path("Datasets/training")
TEST_DATASET = Path("Datasets/testing")

OUTPUT_ROOT = Path("Processed")


def main():
    allResults = []

    for windowSeconds in HISTORY_WINDOWS_SECONDS:
        outputDir = OUTPUT_ROOT / f"window_{windowSeconds:.2f}s_step_{WINDOW_STEP_SECONDS:.2f}s"
        outputDir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("Fusion configuration")
        print(f"  Window length : {windowSeconds:.3f} s")
        print(f"  Window step   : {WINDOW_STEP_SECONDS:.3f} s")
        print(f"  Training data : {TRAIN_DATASET}")
        print(f"  Testing data  : {TEST_DATASET}")
        print("=" * 70)
        print()

        print("Building training feature dataset from raw recordings...")
        trainData = buildDataset(
            TRAIN_DATASET,
            windowSeconds=windowSeconds,
            stepSeconds=WINDOW_STEP_SECONDS,
            maxSensorAgePeriods=MAX_SENSOR_AGE_PERIODS,
        )
        trainPath = outputDir / "train_features.csv"
        trainData.to_csv(trainPath, index=False)
        print(f"Saved {len(trainData)} training rows to {trainPath}")

        print("\nBuilding testing feature dataset from raw recordings...")
        testData = buildDataset(
            TEST_DATASET,
            windowSeconds=windowSeconds,
            stepSeconds=WINDOW_STEP_SECONDS,
            maxSensorAgePeriods=MAX_SENSOR_AGE_PERIODS,
        )
        testPath = outputDir / "test_features.csv"
        testData.to_csv(testPath, index=False)
        print(f"Saved {len(testData)} testing rows to {testPath}")

        print("\nRunning machine-learning algorithms...")
        resultsPath = outputDir / "model_comparison_results.csv"
        results = runModels(trainData, testData, outputPath=resultsPath)
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

from pathlib import Path

from Fusion.dataset import buildDataset
from Fusion.models import runModels


# Offline Fusion Configuration
WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = 0.25

TRAIN_DATASET = Path("Datasets/training")
TEST_DATASET = Path("Datasets/testing")

OUTPUT_ROOT = Path("Processed")


def main():
    outputDir = OUTPUT_ROOT / f"window_{WINDOW_SECONDS:.2f}s_step_{WINDOW_STEP_SECONDS:.2f}s"
    outputDir.mkdir(parents=True, exist_ok=True)

    print("Fusion configuration")
    print(f"  Window length : {WINDOW_SECONDS:.3f} s")
    print(f"  Window step   : {WINDOW_STEP_SECONDS:.3f} s")
    print(f"  Training data : {TRAIN_DATASET}")
    print(f"  Testing data  : {TEST_DATASET}")
    print()

    print("Building training feature dataset from raw recordings...")
    trainData = buildDataset(
        TRAIN_DATASET,
        windowSeconds=WINDOW_SECONDS,
        stepSeconds=WINDOW_STEP_SECONDS,
    )
    trainPath = outputDir / "train_features.csv"
    trainData.to_csv(trainPath, index=False)
    print(f"Saved {len(trainData)} training rows to {trainPath}")

    print("\nBuilding testing feature dataset from raw recordings...")
    testData = buildDataset(
        TEST_DATASET,
        windowSeconds=WINDOW_SECONDS,
        stepSeconds=WINDOW_STEP_SECONDS,
    )
    testPath = outputDir / "test_features.csv"
    testData.to_csv(testPath, index=False)
    print(f"Saved {len(testData)} testing rows to {testPath}")

    print("\nRunning machine-learning algorithms...")
    resultsPath = outputDir / "model_comparison_results.csv"
    runModels(trainData, testData, outputPath=resultsPath)
    print(f"\nResults saved to {resultsPath}")


if __name__ == "__main__":
    main()

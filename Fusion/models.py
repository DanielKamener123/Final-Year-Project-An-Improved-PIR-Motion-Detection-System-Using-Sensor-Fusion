import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
)


# Config
RANDOM_STATE = 42
LABEL_COLUMN = "human"


# Preprocessing
scaled_preprocessing = [
    SimpleImputer(strategy="median"),
    StandardScaler(),
]

tree_preprocessing = [
    SimpleImputer(strategy="median"),
]


# A fresh model set is created every time this function is called.
# This keeps the feature-fusion and PIR-only systems independent.
def buildModels():
    return {
        "Logistic Regression": make_pipeline(
            *scaled_preprocessing,
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Linear SVM": make_pipeline(
            *scaled_preprocessing,
            SVC(kernel="linear", random_state=RANDOM_STATE),
        ),
        "RBF SVM": make_pipeline(
            *scaled_preprocessing,
            SVC(kernel="rbf", random_state=RANDOM_STATE),
        ),
        "Random Forest": make_pipeline(
            *tree_preprocessing,
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "k-Nearest Neighbours": make_pipeline(
            *scaled_preprocessing,
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Decision Tree": make_pipeline(
            *tree_preprocessing,
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "Gradient Boosting": make_pipeline(
            *tree_preprocessing,
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
        "Gaussian Naive Bayes": make_pipeline(
            *scaled_preprocessing,
            GaussianNB(),
        ),
        "Neural Network (MLP)": make_pipeline(
            *scaled_preprocessing,
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
        "PCA + RBF SVM": make_pipeline(
            *scaled_preprocessing,
            PCA(n_components=0.95),
            SVC(kernel="rbf", random_state=RANDOM_STATE),
        ),
    }


def calculateMetrics(y_true, predictions):
    """Return all project performance metrics as percentages."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    positive_total = tp + fn
    negative_total = fp + tn

    probability_detection = (
        100.0 * tp / positive_total
        if positive_total > 0
        else np.nan
    )

    false_alarm_probability = (
        100.0 * fp / negative_total
        if negative_total > 0
        else np.nan
    )

    accuracy = 100.0 * accuracy_score(y_true, predictions)
    precision = 100.0 * precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    f1 = 100.0 * f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return {
        "Probability of Detection (%)": probability_detection,
        "False Alarm Probability (%)": false_alarm_probability,
        "Accuracy (%)": accuracy,
        "Precision (%)": precision,
        "F1 Score (%)": f1,
    }


def percentageImprovement(new_value, baseline_value, reduction=False):
    """
    Direct difference between two percentage results.

    Positive values always mean the fused system improved:
      Detection improvement = fused Pd - PIR-only Pd
      False-alarm improvement = PIR-only Pfa - fused Pfa
    """
    if pd.isna(new_value) or pd.isna(baseline_value):
        return np.nan

    if reduction:
        return baseline_value - new_value

    return new_value - baseline_value


def prepareFeatures(train_data, test_data, featureColumns):
    missingTrain = [
        column for column in featureColumns
        if column not in train_data.columns
    ]
    missingTest = [
        column for column in featureColumns
        if column not in test_data.columns
    ]

    if missingTrain or missingTest:
        raise ValueError(
            "Training and test datasets must contain the same feature columns. "
            f"Missing from training data: {missingTrain}; "
            f"missing from test data: {missingTest}"
        )

    X_train = train_data[featureColumns].copy()
    X_test = test_data[featureColumns].copy()

    # Ensure all features are numerical.
    X_train = X_train.apply(pd.to_numeric, errors="raise")
    X_test = X_test.apply(pd.to_numeric, errors="raise")

    return X_train, X_test


def runDetectionSystem(train_data, test_data, featureColumns, systemName):
    """Train and evaluate every ML algorithm for one sensor configuration."""
    X_train, X_test = prepareFeatures(
        train_data,
        test_data,
        featureColumns,
    )
    y_train = train_data[LABEL_COLUMN]
    y_test = test_data[LABEL_COLUMN]

    models = buildModels()
    results = {}

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"{systemName} - {name}")
        print("=" * 60)

        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = calculateMetrics(y_test, predictions)
            results[name] = metrics

            print(
                f"Probability of Detection : "
                f"{metrics['Probability of Detection (%)']:.2f}%"
            )
            print(
                f"False Alarm Probability : "
                f"{metrics['False Alarm Probability (%)']:.2f}%"
            )
            print(f"Accuracy                : {metrics['Accuracy (%)']:.2f}%")
            print(f"Precision               : {metrics['Precision (%)']:.2f}%")
            print(f"F1 Score                : {metrics['F1 Score (%)']:.2f}%")

        except Exception as error:
            print(f"Model failed: {error}")
            results[name] = {
                "Probability of Detection (%)": np.nan,
                "False Alarm Probability (%)": np.nan,
                "Accuracy (%)": np.nan,
                "Precision (%)": np.nan,
                "F1 Score (%)": np.nan,
            }

    return results


def runModels(train_data, test_data, outputPath="model_comparison_results.csv"):
    """
    Compare feature fusion against PIR-only detection.

    Both systems use the same rows, labels and ML algorithms. The only
    difference is that the fused classifier receives PIR + thermopile
    features, while the PIR-only classifier receives only PIR features.
    """
    y_train = train_data[LABEL_COLUMN]
    y_test = test_data[LABEL_COLUMN]

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test labels must be 0 or 1.")

    pirFeatures = [
        column for column in train_data.columns
        if column.startswith("pir_")
    ]

    thermalFeatures = [
        column for column in train_data.columns
        if column.startswith("thermal_")
        and column != "thermal_track_id"
    ]

    fusedFeatures = pirFeatures + thermalFeatures

    if not pirFeatures:
        raise ValueError("No PIR feature columns were found in the dataset.")

    if not thermalFeatures:
        raise ValueError("No thermopile feature columns were found in the dataset.")

    print("\n" + "#" * 70)
    print("FEATURE-FUSION SYSTEM")
    print(
        f"Features used: {len(fusedFeatures)} "
        f"({len(pirFeatures)} PIR + {len(thermalFeatures)} thermopile)"
    )
    print("#" * 70)
    fusedResults = runDetectionSystem(
        train_data,
        test_data,
        fusedFeatures,
        "Feature Fusion",
    )

    print("\n" + "#" * 70)
    print("PIR-ONLY SYSTEM")
    print(f"Features used: {len(pirFeatures)} PIR")
    print("#" * 70)
    pirResults = runDetectionSystem(
        train_data,
        test_data,
        pirFeatures,
        "PIR Only",
    )

    comparisonRows = []

    for algorithm in buildModels().keys():
        fused = fusedResults[algorithm]
        pir = pirResults[algorithm]

        fusedPd = fused["Probability of Detection (%)"]
        fusedFa = fused["False Alarm Probability (%)"]
        pirPd = pir["Probability of Detection (%)"]
        pirFa = pir["False Alarm Probability (%)"]

        comparisonRows.append({
            "Algorithm": algorithm,
            "Fused Probability of Detection (%)": fusedPd,
            "Fused False Alarm Probability (%)": fusedFa,
            "PIR-Only Probability of Detection (%)": pirPd,
            "PIR-Only False Alarm Probability (%)": pirFa,
            "Probability of Detection Improvement (%)": percentageImprovement(
                fusedPd,
                pirPd,
                reduction=False,
            ),
            "False Alarm Probability Improvement (%)": percentageImprovement(
                fusedFa,
                pirFa,
                reduction=True,
            ),
                        "Fused Accuracy (%)": fused["Accuracy (%)"],
            "Fused Precision (%)": fused["Precision (%)"],
            "Fused F1 Score (%)": fused["F1 Score (%)"],
        })

    results_df = pd.DataFrame(comparisonRows)

    print("\n" + "=" * 140)
    print("FUSION VS PIR-ONLY COMPARISON")
    print("=" * 140)

    display_df = results_df.copy()
    percentageColumns = [
        column for column in display_df.columns
        if column.endswith("(%)")
    ]
    for column in percentageColumns:
        display_df[column] = display_df[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.2f}%"
        )

    print(display_df.to_string(index=False))

    # Keep numeric percentage values in the CSV for later analysis.
    results_df.to_csv(outputPath, index=False, float_format="%.4f")

    return results_df

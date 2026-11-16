import numpy as np
import pandas as pd

from Fusion.dataset import buildDataset, getFusedFeatureNames
from Fusion.evaluation import (
    LABEL_COLUMN, Recordings, addOutboundLabels, compareToPir, evaluateStates,
    pirBaselineStates, runStateMachine, tuneThresholds,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)

# Keep model results repeatable
RANDOM_STATE = 42
WINDOWS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
PIR_SYSTEM = "Thresholded PIR"

# Fill missing values and standardise features for linear models
scaledFeatureVector = [
    SimpleImputer(strategy="median"),
    StandardScaler(),
]

# Fill missing values for tree models
treePreprocessing = [
    SimpleImputer(strategy="median"),
]

# Create six classifiers with their matching preprocessing
def buildModels():
    return {
        "Logistic Regression": make_pipeline(
            *scaledFeatureVector,
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Linear SVM": make_pipeline(
            *scaledFeatureVector,
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "RBF SVM": make_pipeline(
            *scaledFeatureVector,
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": make_pipeline(
            *treePreprocessing,
            RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   class_weight="balanced_subsample", random_state=RANDOM_STATE,
                                   n_jobs=-1),
        ),
        "Extra Trees": make_pipeline(
            *treePreprocessing,
            ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                 class_weight="balanced", random_state=RANDOM_STATE,
                                 n_jobs=-1),
        ),
        "Gradient Boosting": make_pipeline(
            *treePreprocessing,
            GradientBoostingClassifier(n_estimators=100, min_samples_leaf=20, random_state=RANDOM_STATE),
        ),
    }

# Select the fused sensor features in a fixed order
def featureMatrix(data):
    return data[getFusedFeatureNames()].apply(pd.to_numeric, errors="raise")

# Extract the probability of the human class
def confidenceScores(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Classifier was not trained with positive class 1")
    return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=float)

# Train a model, tune its alarms and measure performance
def evaluateClassifier(model, trainData, evalData, recordings, pirMetrics):

    # Fit only on the training recordings
    model.fit(featureMatrix(trainData), trainData[LABEL_COLUMN])

    # Get human probabilities on validation data
    scores = confidenceScores(model, featureMatrix(evalData))

    # Find alarm thresholds using validation scores
    tuned = tuneThresholds(evalData, scores)

    # Convert probabilities into the three alarm states
    states = runStateMachine(evalData, scores, tuned["thresholds"])
    metrics = evaluateStates(evalData, states, recordings)
    metrics.update(compareToPir(metrics, pirMetrics))
    metrics["Thresholds"] = tuned["thresholds"]
    return metrics

# Test every classifier with each PIR history window
def compareModels(trainingPath, validationPath, windows=WINDOWS, progress=print):
    rows = []
    for window in windows:
        progress(f"\nFusion window {window:g} s: building datasets...")
        trainData = buildDataset(trainingPath, windowSeconds=window)
        evalData = addOutboundLabels(buildDataset(validationPath, windowSeconds=window))
        recordings = Recordings(evalData)
        progress(f"  training rows {len(trainData)}, validation rows {len(evalData)}")

        # Measure the PIR baseline on the same validation data
        pirMetrics = evaluateStates(evalData, pirBaselineStates(evalData), recordings)
        rows.append({"Window (s)": window, "System": PIR_SYSTEM, **pirMetrics})

        # Evaluate all six models independently
        for name, model in buildModels().items():
            progress(f"  {name}...")
            metrics = evaluateClassifier(model, trainData, evalData, recordings, pirMetrics)
            rows.append({"Window (s)": window, "System": name, **metrics})
            progress(f"    Event Pd {metrics['Event Pd (%)']:.1f}%  Pfa {metrics['Pfa (%)']:.1f}%  "
                     f"accuracy gain {metrics['Accuracy Gain (%)']:.1f}%  "
                     f"Pfa reduction {metrics['Pfa Reduction (%)']:.1f}%")
    return pd.DataFrame(rows)

SELECTION_MAX_DELAY_S = 9.0

# Rank models by requirements, detections and false alarms
def bestConfiguration(results, system=None):

    # Remove the PIR baseline from model ranking
    fused = results[results["System"] != PIR_SYSTEM].copy()
    if system is not None:
        fused = fused[fused["System"] == system]
    fused["Delay Margin"] = fused["Max Delay (s)"] <= SELECTION_MAX_DELAY_S
    
    # Prioritise requirement checks and detected events
    order = fused.sort_values(["Requirements Met", "Detected Events", "Delay Margin", "Pfa Reduction (%)",
                               "Accuracy Gain (%)"],
                              ascending=[False, False, False, False, False], kind="stable")
    return order.iloc[0]

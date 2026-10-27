"""Model and window selection with scikit-learn classifiers.

For every fusion window, every classifier is trained on the training
recordings and evaluated on the validation recordings against the thresholded
PIR. Each classifier gets its own alarm thresholds, tuned on validation, so the
comparison is fair. The best configuration is then trained from first
principles (Fusion/gradientBoosting.py) by offlineRun.py --train.
"""
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


# Config
RANDOM_STATE = 42
WINDOWS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
PIR_SYSTEM = "Thresholded PIR"

# Preprocessing
scaledFeatureVector = [
    SimpleImputer(strategy="median"),
    StandardScaler(),
]

treePreprocessing = [
    SimpleImputer(strategy="median"),
]


# A fresh model set is created every time this function is called.
def buildModels():
    """Small, interpretable baselines plus nonlinear feature classifiers.

    All models support predict_proba for the three-state alarm.
    Preprocessing is fitted only on the training recordings.
    """
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


# Fused Feature Matrix (all numeric, same column order everywhere)
def featureMatrix(data):
    return data[getFusedFeatureNames()].apply(pd.to_numeric, errors="raise")


# Continuous Class-1 Probability From A Fitted Pipeline
def confidenceScores(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Classifier was not trained with positive class 1")
    return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=float)


# Train, Tune And Evaluate One Classifier On The Validation Data
def evaluateClassifier(model, trainData, evalData, recordings, pirMetrics):
    model.fit(featureMatrix(trainData), trainData[LABEL_COLUMN])
    scores = confidenceScores(model, featureMatrix(evalData))
    tuned = tuneThresholds(evalData, scores)
    states = runStateMachine(evalData, scores, tuned["thresholds"])
    metrics = evaluateStates(evalData, states, recordings)
    metrics.update(compareToPir(metrics, pirMetrics))
    metrics["Thresholds"] = tuned["thresholds"]
    return metrics


# Compare Every Classifier At Every Window (validation recordings)
def compareModels(trainingPath, validationPath, windows=WINDOWS, progress=print):
    rows = []
    for window in windows:
        progress(f"\nFusion window {window:g} s: building datasets...")
        trainData = buildDataset(trainingPath, windowSeconds=window)
        evalData = addOutboundLabels(buildDataset(validationPath, windowSeconds=window))
        recordings = Recordings(evalData)
        progress(f"  training rows {len(trainData)}, validation rows {len(evalData)}")

        pirMetrics = evaluateStates(evalData, pirBaselineStates(evalData), recordings)
        rows.append({"Window (s)": window, "System": PIR_SYSTEM, **pirMetrics})

        for name, model in buildModels().items():
            progress(f"  {name}...")
            metrics = evaluateClassifier(model, trainData, evalData, recordings, pirMetrics)
            rows.append({"Window (s)": window, "System": name, **metrics})
            progress(f"    Event Pd {metrics['Event Pd (%)']:.1f}%  Pfa {metrics['Pfa (%)']:.1f}%  "
                     f"accuracy gain {metrics['Accuracy Gain (%)']:.1f}%  "
                     f"Pfa reduction {metrics['Pfa Reduction (%)']:.1f}%")
    return pd.DataFrame(rows)


# Worst-case detection delay allowed when selecting a configuration: a margin on
# the 10 s requirement so that one slow entry does not fail Spec 4
SELECTION_MAX_DELAY_S = 8.0


# Best Fused Configuration: most requirement checks met (Spec 1 accuracy,
# Spec 1 false alarms, Spec 4 response), then worst-case delay within
# SELECTION_MAX_DELAY_S, then highest false alarm reduction (accuracy gain is
# almost the same for every tree model), then highest accuracy gain
def bestConfiguration(results, system=None):
    fused = results[results["System"] != PIR_SYSTEM].copy()
    if system is not None:
        fused = fused[fused["System"] == system]
    fused["Delay Margin"] = fused["Max Delay (s)"] <= SELECTION_MAX_DELAY_S
    order = fused.sort_values(["Requirements Met", "Delay Margin", "Pfa Reduction (%)", "Accuracy Gain (%)"],
                              ascending=[False, False, False, False], kind="stable")
    return order.iloc[0]

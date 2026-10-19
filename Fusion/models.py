from pathlib import Path
import numpy as np
import pandas as pd

from Fusion.event_analysis import analyze_events
from PIR.processing import threshold as PIR_THRESHOLD

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
scaledFeatureVector = [
    SimpleImputer(strategy="median"),
    StandardScaler(),
]

treePreprocessing = [
    SimpleImputer(strategy="median"),
]


# A fresh model set is created every time this function is called.
# This keeps the feature-fusion and PIR-only systems independent.
def buildModels():
    """Small, interpretable baselines plus nonlinear feature classifiers.

    All models support predict_proba for the existing three-state alarm.
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
            GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
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
    """Relative percentage improvement against the PIR-only baseline.

    For increases: 100 * (fused - PIR) / PIR.
    For reductions: 100 * (PIR - fused) / PIR.
    Undefined when the baseline is 0 or either value is missing.
    """
    if pd.isna(new_value) or pd.isna(baseline_value) or baseline_value == 0:
        return np.nan

    if reduction:
        return 100.0 * (baseline_value - new_value) / baseline_value

    return 100.0 * (new_value - baseline_value) / baseline_value


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



def calculateEventMetrics(eval_data, predictions, maxDelaySeconds=10.0, consecutiveFrames=1):
    """Measure labelled 0->1 intrusion events for EACH recording separately.

    An event beginning at the first available frame is excluded: its true onset
    cannot be determined. Detection must begin on/after the labelled entry and
    have consecutiveFrames positive samples within maxDelaySeconds and the event.
    """
    required = {"recording", "fusionTime", LABEL_COLUMN}
    missing = required - set(eval_data.columns)
    if missing:
        raise ValueError(f"Cannot evaluate event Pd; missing columns: {sorted(missing)}")
    if consecutiveFrames < 1 or maxDelaySeconds <= 0:
        raise ValueError("Invalid event evaluation parameters")

    evaluation = eval_data[["recording", "fusionTime", LABEL_COLUMN]].copy()
    evaluation["prediction"] = np.asarray(predictions, dtype=int)
    event_count = detected_count = 0
    delays = []
    for recording, group in evaluation.groupby("recording", sort=False):
        group = group.sort_values("fusionTime", kind="stable")
        times = group["fusionTime"].to_numpy(dtype=float)
        actual = group[LABEL_COLUMN].to_numpy(dtype=int)
        predicted = group["prediction"].to_numpy(dtype=int)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"Non-increasing times in recording {recording}")

        starts = np.flatnonzero((actual == 1) & (np.r_[0, actual[:-1]] == 0))
        ends = np.flatnonzero((actual == 1) & (np.r_[actual[1:], 0] == 0))
        for start, end in zip(starts, ends):
            if start == 0:
                continue  # Left-censored event: actual entry time unknown
            event_count += 1
            deadline = min(times[start] + maxDelaySeconds, times[end])
            for index in range(start, end - consecutiveFrames + 2):
                final_index = index + consecutiveFrames - 1
                if times[final_index] > deadline:
                    break
                if np.all(predicted[index:final_index + 1] == 1):
                    detected_count += 1
                    # Confirmation time, not just the first positive frame
                    delays.append(float(times[final_index] - times[start]))
                    break

    return {
        "Event Pd (%)": 100.0 * detected_count / event_count if event_count else np.nan,
        "Events": event_count,
        "Detected Events": detected_count,
        "Mean Event Delay (s)": float(np.mean(delays)) if delays else np.nan,
    }


# Initial operating points; tune ONLY on the validation recordings.
# Model confidence is the class-1 predict_proba output, not a hard prediction.
MAYBE_ENTER = 0.58  
MAYBE_EXIT = 0.45   
INTRUDER_ENTER = 0.60
INTRUDER_EXIT = 0.59

NO_INTRUDER, MAYBE_INTRUDER, INTRUDER = 0, 1, 2


def applyThreeState(confidences, eval_data, maybe_enter=MAYBE_ENTER,
                    maybe_exit=MAYBE_EXIT, intruder_enter=INTRUDER_ENTER,
                    intruder_exit=INTRUDER_EXIT):
    """Causal state machine; reset independently for each recording.

    The asymmetric enter/exit thresholds provide hysteresis. State 1 and 2
    both constitute an alert, but only state 2 is a confirmed alert.
    """
    if not (0 <= maybe_exit < maybe_enter < intruder_exit < intruder_enter <= 1):
        raise ValueError("Threshold order must be maybe_exit < maybe_enter < "
                         "intruder_exit < intruder_enter, within [0, 1]")
    scores = np.asarray(confidences, dtype=float)
    if len(scores) != len(eval_data) or not np.isfinite(scores).all():
        raise ValueError("Confidence must be finite and aligned to evaluation rows")
    states = np.zeros(len(scores), dtype=int)
    # Grouping with sort=False preserves input row order within each recording.
    for _, positions in pd.Series(np.arange(len(scores))).groupby(eval_data["recording"].to_numpy(), sort=False):
        indices = positions.to_numpy(dtype=int)
        indices = indices[np.argsort(eval_data.iloc[indices]["fusionTime"].to_numpy(), kind="stable")]
        state = NO_INTRUDER
        for index in indices:
            score = scores[index]
            if state == NO_INTRUDER:
                if score >= intruder_enter:
                    state = INTRUDER
                elif score >= maybe_enter:
                    state = MAYBE_INTRUDER
            elif state == MAYBE_INTRUDER:
                if score >= intruder_enter:
                    state = INTRUDER
                elif score < maybe_exit:
                    state = NO_INTRUDER
            else:
                if score < intruder_exit:
                    state = MAYBE_INTRUDER if score >= maybe_exit else NO_INTRUDER
            states[index] = state
    return states


def confidenceScores(model, X):
    """Continuous class-1 probability estimate from the fitted pipeline."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("All classifiers must support predict_proba")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Classifier was not trained with positive class 1")
    return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=float)


def addOutboundEvaluationLabels(eval_data):
    """Read outbound labels at validation timestamps; never used for training."""
    required = {"recording", "fusionTime"}

    evaluation = eval_data.copy()
    outbound = np.zeros(len(evaluation), dtype=np.uint8)
    for recording, positions in evaluation.groupby("recording", sort=False).indices.items():
        label_path = Path(recording) / "labels(outbound).csv"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing outbound evaluation labels: {label_path}")
        intervals = pd.read_csv(label_path)
        if not {"start_s", "end_s", "human"}.issubset(intervals.columns):
            raise ValueError(f"Invalid outbound label columns in {label_path}")
        times = evaluation.iloc[positions]["fusionTime"].to_numpy(dtype=float)
        for start, end, human in intervals[["start_s", "end_s", "human"]].itertuples(index=False, name=None):
            if int(human) == 1:
                outbound[positions] |= ((times >= float(start)) & (times < float(end)))
    evaluation["outboundHuman"] = outbound
    return evaluation


def calculateStateMetrics(eval_data, states):
    """Event Pd, confirmed event Pd, false alerts and confirmed false alerts."""
    values = np.asarray(states, dtype=int)
    any_alert = (values >= MAYBE_INTRUDER).astype(int)
    confirmed = (values == INTRUDER).astype(int)
    any_frame = calculateMetrics(eval_data[LABEL_COLUMN], any_alert)
    # Application-specific Pfa: with inbound=0, MAYBE is acceptable only
    # when the outbound labels indicate a human is visible to a sensor.
    if "outboundHuman" not in eval_data.columns:
        raise ValueError("Outbound evaluation labels were not loaded")
    inbound = eval_data[LABEL_COLUMN].to_numpy(dtype=int)
    outbound = eval_data["outboundHuman"].to_numpy(dtype=int)
    if not np.isin(outbound, [0, 1]).all():
        raise ValueError("Outbound evaluation labels must be binary")
    negative = inbound == 0
    false_alert = negative & ((values == INTRUDER) | ((values == MAYBE_INTRUDER) & (outbound == 0)))
    any_frame["False Alarm Probability (%)"] = (
        100.0 * np.count_nonzero(false_alert) / np.count_nonzero(negative)
        if np.any(negative) else np.nan
    )
    # Count distinct false-alert episodes (rising edges during negative labels).
    # Distinct from Pfa and from the Ward event insertion diagnostic.
    false_episode_count = 0
    for _, positions in eval_data.groupby("recording", sort=False).indices.items():
        indices = np.asarray(positions)
        indices = indices[np.argsort(eval_data.iloc[indices]["fusionTime"].to_numpy())]
        prior_false = False
        for index in indices:
            active_false = bool(false_alert[index])
            if active_false and not prior_false:
                false_episode_count += 1
            prior_false = active_false
    confirmed_frame = calculateMetrics(eval_data[LABEL_COLUMN], confirmed)
    any_events = calculateEventMetrics(eval_data, any_alert)
    confirmed_events = calculateEventMetrics(eval_data, confirmed)
    ward_any = analyze_events(eval_data, any_alert)
    ward_confirmed = analyze_events(eval_data, confirmed)
    return {
        "Probability of Detection (%)": any_frame["Probability of Detection (%)"],
        "False Alarm Probability (%)": any_frame["False Alarm Probability (%)"],
        "Accuracy (%)": any_frame["Accuracy (%)"],
        "Precision (%)": any_frame["Precision (%)"],
        "F1 Score (%)": any_frame["F1 Score (%)"],
        **any_events,
        "Confirmed Event Pd (%)": confirmed_events["Event Pd (%)"],
        "Confirmed False Alarm Probability (%)": confirmed_frame["False Alarm Probability (%)"],
        "Confirmed Mean Event Delay (s)": confirmed_events["Mean Event Delay (s)"],
        **{f"Ward {key}": value for key, value in ward_any.items()},
        **{f"Ward Confirmed {key}": value for key, value in ward_confirmed.items()},
        "Maybe Frames": int(np.count_nonzero(values == MAYBE_INTRUDER)),
        "Intruder Frames": int(np.count_nonzero(values == INTRUDER)),
        "False Alert Episodes": int(false_episode_count),
    }


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
            scores = confidenceScores(model, X_test)
            states = applyThreeState(scores, test_data)
            metrics = calculateStateMetrics(test_data, states)
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
                "Event Pd (%)": np.nan,
                "Events": np.nan,
                "Detected Events": np.nan,
                "Mean Event Delay (s)": np.nan,
                "Confirmed Event Pd (%)": np.nan,
                "Confirmed False Alarm Probability (%)": np.nan,
                "Confirmed Mean Event Delay (s)": np.nan,
                **{f"Ward {key}": np.nan for key in ("C", "D", "F", "M", "FM", "I", "TrueEvents", "PredEvents", "Preexisting", "Eligible", "NewDetected", "Overfill s", "Underfill s", "Fresh Event Pd (%)", "Fresh Event Delay (s)")},
                **{f"Ward Confirmed {key}": np.nan for key in ("C", "D", "F", "M", "FM", "I", "TrueEvents", "PredEvents", "Preexisting", "Eligible", "NewDetected", "Overfill s", "Underfill s", "Fresh Event Pd (%)", "Fresh Event Delay (s)")},
            }

    return results


def runModels(train_data, test_data, outputPath=None):
    """
    Compare feature fusion against PIR-only detection.

    Compare each trained fusion model to the same simple PIR voltage threshold.
    The baseline has no ML training, and is evaluated on the same timestamps.
    """
    y_train = train_data[LABEL_COLUMN]
    y_test = test_data[LABEL_COLUMN]

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test labels must be 0 or 1.")

    # Outbound labels are used for evaluation only, never feature extraction or fitting.
    test_data = addOutboundEvaluationLabels(test_data)

    pirFeatures = [
        column for column in train_data.columns
        if column.startswith("pir")
        and not column.startswith("pirThresholdInterval")
        and column not in ("pirActive", "pirCurrentrelativevoltage")
    ]
    
    thermalFeatures = [
        column for column in train_data.columns
        if column.startswith("thermal") and column != "thermal_track_id"
    ]

    fusedFeatures = pirFeatures + thermalFeatures

    if not pirFeatures:
        raise ValueError("No PIR feature columns were found in the dataset.")

    if not thermalFeatures:
        raise ValueError("No thermopile feature columns were found in the dataset.")

    # Fair ablation: compare identical ML algorithms on identical recordings.
    # Prefixes originate in Fusion/dataset.py (thermal* and pir*).
    fdFeatures = [name for name in thermalFeatures if "fd_" in name.lower() or
                  "framediff" in name.lower() or "frame_diff" in name.lower()]
    bvdFeatures = [name for name in thermalFeatures if name not in fdFeatures]
    featureSets = {
        "PIR ML": pirFeatures,
        "Thermal ML": thermalFeatures,
        "PIR + BVD": pirFeatures + bvdFeatures,
        "PIR + BVD + FD": fusedFeatures,
    }
    # If there are no actual FD columns, do not claim an FD comparison occurred.
    if not fdFeatures:
        featureSets.pop("PIR + BVD")
        print("WARNING: No frame-difference columns found; BVD-vs-FD ablation unavailable.")
    print("\n" + "#" * 70)
    print("FAIR FEATURE ABLATION (SAME ML MODELS AND TEST RECORDINGS)")
    print(f"PIR={len(pirFeatures)}, thermal={len(thermalFeatures)}, "
          f"BVD={len(bvdFeatures)}, FD={len(fdFeatures)}")
    print("#" * 70)
    systemResults = {
        name: runDetectionSystem(train_data, test_data, columns, name)
        for name, columns in featureSets.items()
    }
    fusedResults = systemResults["PIR + BVD + FD"]

    print("\nMODEL ABLATION SUMMARY (MAYBE+INTRUDER; UNCHANGED ALARM THRESHOLDS)")
    for algorithm in buildModels():
        print(f"\n{algorithm}:")
        for systemName, systemMetrics in systemResults.items():
            m = systemMetrics[algorithm]
            print(f"  {systemName:18s} | Acc {m['Accuracy (%)']:6.2f}% "
                  f"| Pd {m['Probability of Detection (%)']:6.2f}% "
                  f"| Pfa {m['False Alarm Probability (%)']:6.2f}% "
                  f"| Event Pd {m['Event Pd (%)']:6.2f}%")

    # True thresholded PIR baseline: mark a fusion-time interval active if
    # ANY raw PIR sample crossed threshold since the preceding thermal frame.
    # This avoids losing short threshold crossings between thermal frames.
    if "pirThresholdIntervalActive" not in test_data.columns:
        raise ValueError("Rebuild the dataset with the updated Fusion/dataset.py: "
                         "pirThresholdIntervalActive is missing")
    interval_active = pd.to_numeric(test_data["pirThresholdIntervalActive"], errors="raise").to_numpy()
    if not np.isin(interval_active, [0, 1]).all():
        raise ValueError("Nonbinary PIR interval flags in validation data")
    pir_states = np.where(interval_active == 1, INTRUDER, NO_INTRUDER)
    pir_metrics = calculateStateMetrics(test_data, pir_states)
    print(f"\nPIR THRESHOLD BASELINE: ANY sample in thermal interval >= {PIR_THRESHOLD:g} V absolute deviation")
    print(f"PIR threshold accuracy: {pir_metrics['Accuracy (%)']:.2f}%")
    print(f"PIR baseline: {np.count_nonzero(interval_active)} active fusion intervals "
          f"of {len(interval_active)}")
    print("NOTE: Pfa is a percentage of evaluation intervals, NOT false alarm events/hour.")
    pirResults = {name: pir_metrics for name in fusedResults}

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
            **{
                f"{systemName} {metricName}": systemResults[systemName][algorithm][metricName]
                for systemName in systemResults
                for metricName in (
                    "Accuracy (%)", "Probability of Detection (%)",
                    "False Alarm Probability (%)", "Event Pd (%)",
                    "Confirmed Event Pd (%)", "False Alert Episodes",
                    "Ward Fresh Event Pd (%)", "Ward Fresh Event Delay (s)",
                )
            },
            "Fused Probability of Detection (%)": fusedPd,
            "Fused False Alarm Probability (%)": fusedFa,
            "PIR-Only Probability of Detection (%)": pirPd,
            "PIR-Only False Alarm Probability (%)": pirFa,
            "Pd Change (percentage points)": fusedPd - pirPd,
            "False Alarm Change (percentage points)": pirFa - fusedFa,
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
            "PIR-Only Accuracy (%)": pir["Accuracy (%)"],
            "Accuracy Improvement (%)": percentageImprovement(
                fused["Accuracy (%)"], pir["Accuracy (%)"]
            ),
            "PIR-Only Precision (%)": pir["Precision (%)"],
            "Fused Precision (%)": fused["Precision (%)"],
            "PIR-Only F1 Score (%)": pir["F1 Score (%)"],
            "Fused F1 Score (%)": fused["F1 Score (%)"],
            "PIR-Only Event Pd (%)": pir["Event Pd (%)"],
            "Fused Event Pd (%)": fused["Event Pd (%)"],
            "PIR-Only Confirmed Event Pd (%)": pir["Confirmed Event Pd (%)"],
            "Fused Confirmed Event Pd (%)": fused["Confirmed Event Pd (%)"],
            "PIR-Only Confirmed Pfa (%)": pir["Confirmed False Alarm Probability (%)"],
            "Fused Confirmed Pfa (%)": fused["Confirmed False Alarm Probability (%)"],
            "PIR-Only False Alert Episodes": pir["False Alert Episodes"],
            "Fused False Alert Episodes": fused["False Alert Episodes"],
            "PIR-Only Detected Events": pir["Detected Events"],
            "Fused Detected Events": fused["Detected Events"],
            "Evaluated Events": fused["Events"],
            "PIR-Only Mean Event Delay (s)": pir["Mean Event Delay (s)"],
            "Fused Mean Event Delay (s)": fused["Mean Event Delay (s)"],
            **{f"Fused Ward {key}": fused[f"Ward {key}"] for key in ("C", "D", "F", "M", "FM", "I", "TrueEvents", "PredEvents", "Preexisting", "Eligible", "NewDetected", "Overfill s", "Underfill s", "Fresh Event Pd (%)", "Fresh Event Delay (s)")},
            **{f"PIR-Only Ward {key}": pir[f"Ward {key}"] for key in ("C", "D", "F", "M", "FM", "I", "TrueEvents", "PredEvents", "Preexisting", "Eligible", "NewDetected", "Overfill s", "Underfill s", "Fresh Event Pd (%)", "Fresh Event Delay (s)")},
            **{f"Fused Ward Confirmed {key}": fused[f"Ward Confirmed {key}"] for key in ("C", "D", "F", "M", "FM", "I", "Fresh Event Pd (%)", "Fresh Event Delay (s)")},
        })

    results_df = pd.DataFrame(comparisonRows)

    print("\n" + "=" * 140)
    print("FUSION VS THRESHOLDED PIR COMPARISON")
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

    compact = ["Algorithm", "PIR-Only Accuracy (%)", "Fused Accuracy (%)",
               "Accuracy Improvement (%)", "PIR-Only False Alarm Probability (%)",
               "Fused False Alarm Probability (%)", "False Alarm Probability Improvement (%)"]
    print(display_df[compact].to_string(index=False))

    # Optional text report when this function is called independently.
    # main.py creates one combined text report for all window lengths.
    if outputPath is not None:
        from pathlib import Path
        path = Path(outputPath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(formatResultsTable(results_df), encoding="utf-8")
        print(f"Results saved to {path}")

    return results_df


def formatResultsTable(results_df):
    """Readable, plain-text comparison; metrics are FRAME-BASED only."""
    headers = [
        ("History Window (s)", "Win"),
        ("Algorithm", "Model"),
        ("PIR-Only Accuracy (%)", "PIR Acc"),
        ("Fused Accuracy (%)", "Fusion Acc"),
        ("Accuracy Improvement (%)", "Acc gain"),
        ("PIR-Only Probability of Detection (%)", "PIR Pd"),
        ("Fused Probability of Detection (%)", "Fusion Pd"),
        ("PIR-Only False Alarm Probability (%)", "PIR Pfa"),
        ("Fused False Alarm Probability (%)", "Fusion Pfa"),
        ("False Alarm Probability Improvement (%)", "Pfa drop"),
        ("PIR-Only Precision (%)", "PIR Prec"),
        ("Fused Precision (%)", "Fusion Prec"),
        ("PIR-Only F1 Score (%)", "PIR F1"),
        ("Fused F1 Score (%)", "Fusion F1"),
        ("PIR-Only Event Pd (%)", "PIR Event Pd"),
        ("Fused Event Pd (%)", "Fusion Event Pd"),
        ("PIR-Only Detected Events", "PIR Events"),
        ("Fused Detected Events", "Fusion Events"),
        ("Evaluated Events", "Total Events"),
        ("PIR-Only Mean Event Delay (s)", "PIR Delay s"),
        ("Fused Mean Event Delay (s)", "Fusion Delay s"),
    ]
    widths = [max(len(label), 7) for _, label in headers]
    widths[1] = 23

    def fmt(value, width, is_model=False):
        if is_model:
            return str(value).ljust(width)
        if pd.isna(value):
            return "N/A".rjust(width)
        return f"{float(value):.2f}".rjust(width)

    def line(fields):
        return " | ".join(fields)

    heading = line([label.ljust(width) for (_, label), width in zip(headers, widths)])
    divider = "-" * len(heading)
    lines = [
        "OFFLINE PIR VS FEATURE-FUSION RESULTS (VALIDATION)",
        "Accuracy, Pd, Pfa, precision and F1 are FRAME-based; Event Pd is event-based.",
        "Event Pd = events detected within 10s / labelled entries (one positive frame required).",
        "Each event is a 0->1 labelled transition per recording; left-censored entries are excluded.",
        "Event delay is measured from entry to the first confirmed detection.",
        "PIR Events and Fusion Events are detected counts; Total Events is the denominator.",
        "Relative Acc gain = 100 * (Fusion Acc - PIR Acc) / PIR Acc.",
        "Relative Pfa drop = 100 * (PIR Pfa - Fusion Pfa) / PIR Pfa.",
        "N/A indicates an undefined metric or a failed model.",
        "",
        heading,
        divider,
    ]
    for _, record in results_df.iterrows():
        values = []
        for key, label in headers:
            if key not in record:
                values.append("-".rjust(widths[len(values)]))
            else:
                values.append(fmt(record[key], widths[len(values)], is_model=key == "Algorithm"))
        lines.append(line(values))
    lines.append(divider)
    lines.append("")
    lines.append("Target checks: accuracy improvement >= 20%; Pfa reduction >= 50%.")
    qualifying = results_df.loc[
        (results_df["Accuracy Improvement (%)"] >= 20)
        & (results_df["False Alarm Probability Improvement (%)"] >= 50)
    ]
    if qualifying.empty:
        lines.append("No configuration meets BOTH relative targets in this validation run.")
    else:
        for _, row in qualifying.iterrows():
            w = row.get("History Window (s)", "-")
            lines.append(f"PASS BOTH: window={w}s, model={row['Algorithm']}")
    lines.append("")
    return "\n".join(lines)

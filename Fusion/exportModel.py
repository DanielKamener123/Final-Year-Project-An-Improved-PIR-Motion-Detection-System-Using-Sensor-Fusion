"""Train and export only the fused Random Forest; PIR baseline is thresholded."""
import argparse
import json
from pathlib import Path
import numpy as np
from Fusion.dataset import buildDataset
from Fusion.models import (buildModels, MAYBE_ENTER, MAYBE_EXIT, INTRUDER_ENTER, INTRUDER_EXIT)
from PIR.processing import getCurrentFeatureNames, getHistoryFeatureNames
from Thermopile.processing import getThermalFeatureNames


def alarm_thresholds(window):
    """Tuned thresholds from offlineRun.py (tune mode) if available, else defaults."""
    from Fusion.alarm_tuning import load_thresholds
    tuned = load_thresholds(window, "Random Forest")
    if tuned is not None:
        print(f"Using tuned alarm thresholds for {window:g} s: {tuned}")
        return tuned
    print("WARNING: no tuned alarm thresholds found; exporting the default thresholds")
    return {"maybe_enter": MAYBE_ENTER, "maybe_exit": MAYBE_EXIT,
            "intruder_enter": INTRUDER_ENTER, "intruder_exit": INTRUDER_EXIT, "hold_s": 0.0}


def export_model(model, names, window, destination):
    imputer = model.steps[0][1]
    forest = model.steps[-1][1]
    medians = np.asarray(imputer.statistics_, dtype=float)
    if medians.size != len(names) or not np.isfinite(medians).all():
        raise ValueError("Cannot export a model with missing/all-NaN training features")
    positive = np.flatnonzero(forest.classes_ == 1)
    if positive.size != 1:
        raise ValueError("Training data must contain both class 0 and class 1")
    pos = int(positive[0])
    trees = []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        counts = tree.value[:, 0, :]
        denom = counts.sum(axis=1)
        prob = np.divide(counts[:, pos], denom, out=np.zeros(len(denom)), where=denom > 0)
        trees.append({
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.astype(float).tolist(),
            "left": tree.children_left.astype(int).tolist(),
            "right": tree.children_right.astype(int).tolist(),
            "probability": prob.tolist(),
        })
    data = {
        "features": list(names), "medians": medians.tolist(), "trees": trees,
        "window": float(window),
        "thresholds": alarm_thresholds(window),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    # Check the exported tree traversal against training-time sklearn probabilities.
    from Fusion.realtimeModel import RandomForest
    check = RandomForest(destination)
    sample = model._export_check_sample
    expected = model.predict_proba(sample)[:, pos]
    actual = np.asarray([check.confidence(row) for row in sample.to_numpy()])
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-10:
        # Locate the exact tree and training example responsible. Never accept a
        # mismatched model, since its live predictions may change alarm states.
        prepared = imputer.transform(sample)
        prepared32 = np.asarray(prepared, dtype=np.float32)
        worst = int(np.argmax(np.abs(expected - actual)))
        for tree_index, estimator in enumerate(forest.estimators_):
            reference_leaf = int(estimator.apply(prepared32[worst:worst + 1])[0])
            exported_tree = check.trees[tree_index]
            node = 0
            route = []
            while exported_tree["left"][node] != -1:
                feature_index = exported_tree["feature"][node]
                feature_value = float(prepared32[worst, feature_index])
                split = float(exported_tree["threshold"][node])
                next_node = (exported_tree["left"][node] if feature_value <= split
                             else exported_tree["right"][node])
                route.append((node, names[feature_index], feature_value, split, next_node))
                node = next_node
            if node != reference_leaf:
                print(f"DISAGREEMENT: sample={worst}, tree={tree_index}, "
                      f"sklearn_leaf={reference_leaf}, exported_leaf={node}")
                print(f"  expected_probability={expected[worst]:.17g}, "
                      f"actual_probability={actual[worst]:.17g}")
                print("  Traversed exported decision path (node, feature, value, threshold, child):")
                for step in route:
                    print("   ", step)
                break
        else:
            print("Every per-tree leaf matches on the worst sample. Check class "
                  "probability conversion or feature preprocessing instead.")
        raise AssertionError(f"Exported tree prediction mismatch: {error}")
    print(f"Verified {destination}: {len(trees)} trees, max probability error {error:.2g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=Path("Datasets/training"))
    parser.add_argument("--output", type=Path, default=Path("Processed/realtime"))
    parser.add_argument("--window", type=float, default=1.5)
    args = parser.parse_args()
    data = buildDataset(args.training, args.window)
    current = ["pir" + n.capitalize() for n in getCurrentFeatureNames()]
    history = ["pir" + n.capitalize() for n in getHistoryFeatureNames()]
    thermal = ["thermal" + n.capitalize() for n in getThermalFeatureNames()]
    for suffix, features in (("fusion", current + history + thermal),):
        missing = [name for name in features if name not in data.columns]
        if missing:
            raise ValueError(f"Missing training features: {missing}")
        X = data[features]
        y = data["human"]
        model = buildModels()["Random Forest"]
        model.fit(X, y)
        model._export_check_sample = X.iloc[:min(100, len(X))]
        export_model(model, features, args.window, args.output / f"{suffix}_rf.json")


if __name__ == "__main__":
    main()

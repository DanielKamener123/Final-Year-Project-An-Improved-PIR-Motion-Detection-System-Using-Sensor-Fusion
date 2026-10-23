"""First-principles Random Forest (Breiman, 2001) for the fused feature vector.

Training: every tree is grown on a bootstrap sample of the training rows. At
each node a random subset of the features is tried, and the split with the
lowest class-weighted Gini impurity is kept. A node becomes a leaf when it is
pure or cannot be split without leaving fewer than MIN_SAMPLES_LEAF rows on a
side. Inference: each tree returns the (weighted) fraction of human rows in the
leaf the feature vector lands in, and the forest averages the trees.

Only numpy is used, so the same file trains offline and runs on the ROCK.
"""
import json
from pathlib import Path

import numpy as np

# Config (same settings as the scikit-learn Random Forest used for model selection)
N_TREES = 200
MIN_SAMPLES_LEAF = 2
RANDOM_SEED = 42

LEAF = -1


# Finds The Best Split For One Node
# rows: training rows in the node. Returns (child impurity, feature, threshold)
# or None if no feature can be split.
def bestSplit(X, y, weights, rows, featureOrder, featuresPerSplit, minSamplesLeaf):
    best = None
    tried = 0
    nodeWeights = weights[rows]
    nodeHuman = nodeWeights * y[rows]
    nodeEmpty = nodeWeights - nodeHuman
    humanTotal = float(nodeHuman.sum())
    emptyTotal = float(nodeEmpty.sum())
    count = len(rows)
    leftCount = np.arange(1, count)

    for feature in featureOrder:
        values = X[rows, feature]
        order = np.argsort(values, kind="stable")
        sortedValues = values[order]

        # A constant feature cannot split the node and does not count as tried
        if sortedValues[0] == sortedValues[-1]:
            continue
        tried += 1

        # Weight of each class to the left of every possible split position
        leftHuman = np.cumsum(nodeHuman[order])[:-1]
        leftEmpty = np.cumsum(nodeEmpty[order])[:-1]
        rightHuman = humanTotal - leftHuman
        rightEmpty = emptyTotal - leftEmpty
        leftTotal = leftHuman + leftEmpty
        rightTotal = rightHuman + rightEmpty

        # Only split between two different values, keeping enough rows on each side
        valid = ((sortedValues[:-1] < sortedValues[1:])
                 & (leftCount >= minSamplesLeaf) & (count - leftCount >= minSamplesLeaf))
        if valid.any():
            # Weighted Gini of the two children: W * (1 - p0^2 - p1^2) = W - (w0^2 + w1^2) / W
            impurity = ((leftTotal - (leftHuman ** 2 + leftEmpty ** 2) / leftTotal)
                        + (rightTotal - (rightHuman ** 2 + rightEmpty ** 2) / rightTotal))
            impurity[~valid] = np.inf
            k = int(np.argmin(impurity))
            if best is None or impurity[k] < best[0]:
                # Threshold half-way between the two neighbouring values
                threshold = 0.5 * (sortedValues[k] + sortedValues[k + 1])
                if threshold >= sortedValues[k + 1]:
                    threshold = sortedValues[k]
                best = (float(impurity[k]), int(feature), float(threshold))

        if tried >= featuresPerSplit:
            break

    return best


# Grows One Decision Tree On A Bootstrap Sample
def growTree(X, y, weights, rows, rng, featuresPerSplit, minSamplesLeaf, importance):
    tree = {"feature": [], "threshold": [], "left": [], "right": [], "probability": []}

    def newNode():
        for key in tree:
            tree[key].append(LEAF if key in ("feature", "left", "right") else 0.0)
        return len(tree["left"]) - 1

    toSplit = [(rows, newNode())]
    while toSplit:
        nodeRows, node = toSplit.pop()
        humanWeight = float(np.sum(weights[nodeRows] * y[nodeRows]))
        totalWeight = float(np.sum(weights[nodeRows]))
        emptyWeight = totalWeight - humanWeight
        tree["probability"][node] = humanWeight / totalWeight

        # Leaf: pure node or too few rows to give both children enough rows
        if humanWeight == 0 or emptyWeight == 0 or len(nodeRows) < 2 * minSamplesLeaf:
            continue

        featureOrder = rng.permutation(X.shape[1])
        split = bestSplit(X, y, weights, nodeRows, featureOrder, featuresPerSplit, minSamplesLeaf)
        if split is None:
            continue

        childImpurity, feature, threshold = split
        # Impurity removed by this split (for feature importance)
        nodeImpurity = totalWeight - (humanWeight ** 2 + emptyWeight ** 2) / totalWeight
        importance[feature] += nodeImpurity - childImpurity

        goesLeft = X[nodeRows, feature] <= threshold
        left = newNode()
        right = newNode()
        tree["feature"][node] = feature
        tree["threshold"][node] = threshold
        tree["left"][node] = left
        tree["right"][node] = right
        toSplit.append((nodeRows[goesLeft], left))
        toSplit.append((nodeRows[~goesLeft], right))

    return tree


class RandomForest:
    def __init__(self, nTrees=N_TREES, minSamplesLeaf=MIN_SAMPLES_LEAF, seed=RANDOM_SEED):
        self.nTrees = int(nTrees)
        self.minSamplesLeaf = int(minSamplesLeaf)
        self.seed = int(seed)
        self.features = []
        self.fill = None
        self.trees = []
        self.importance = None
        # Set when the model is saved for the real-time system
        self.window = None
        self.thresholds = None

    # Replaces Missing Values With The Training Median Of That Feature
    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(X), self.fill, X)

    # Train On The Fused Training Rows (X: rows x features, y: 1 = human)
    def fit(self, X, y, featureNames, progress=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[1] != len(featureNames):
            raise ValueError("Feature names do not match the training columns")
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training labels must contain both 0 and 1")

        self.features = list(featureNames)
        self.fill = np.array([np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.0
                              for column in X.T])
        X = self._prepare(X)

        rng = np.random.default_rng(self.seed)
        count = len(y)
        featuresPerSplit = max(1, int(np.sqrt(X.shape[1])))
        importance = np.zeros(X.shape[1])
        self.trees = []

        for treeIndex in range(self.nTrees):
            # Bootstrap: draw as many rows as there are, with replacement
            rows = rng.integers(0, count, count)

            # Balanced class weights for this sample: each class carries half the total weight
            humanCount = np.count_nonzero(y[rows] == 1)
            emptyCount = count - humanCount
            weights = np.where(y == 1, count / (2.0 * max(humanCount, 1)), count / (2.0 * max(emptyCount, 1)))

            treeImportance = np.zeros(X.shape[1])
            tree = growTree(X, y, weights, rows, rng, featuresPerSplit, self.minSamplesLeaf, treeImportance)
            self.trees.append(tree)
            if treeImportance.sum() > 0:
                importance += treeImportance / treeImportance.sum()
            if progress is not None and (treeIndex + 1) % 25 == 0:
                progress(f"  grown {treeIndex + 1}/{self.nTrees} trees")

        self.importance = importance / self.nTrees
        return self

    # Human Probability For Many Rows At Once (offline evaluation)
    def predictProba(self, X):
        X = self._prepare(X)
        total = np.zeros(len(X))
        for tree in self.trees:
            feature = np.asarray(tree["feature"])
            threshold = np.asarray(tree["threshold"])
            left = np.asarray(tree["left"])
            right = np.asarray(tree["right"])
            node = np.zeros(len(X), dtype=int)
            active = left[node] != LEAF
            while active.any():
                rows = np.flatnonzero(active)
                current = node[rows]
                goesLeft = X[rows, feature[current]] <= threshold[current]
                node[rows] = np.where(goesLeft, left[current], right[current])
                active = left[node] != LEAF
            total += np.asarray(tree["probability"])[node]
        return total / len(self.trees)

    # Human Probability For One Feature Vector (live system)
    def confidence(self, features):
        if isinstance(features, dict):
            values = [features[name] for name in self.features]
        else:
            values = list(features)
        if len(values) != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features; got {len(values)}")
        values = [float(self.fill[i]) if np.isnan(v) else float(v) for i, v in enumerate(values)]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature vector")
        total = 0.0
        for tree in self.trees:
            node = 0
            while tree["left"][node] != LEAF:
                if values[tree["feature"][node]] <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]
            total += tree["probability"][node]
        return total / len(self.trees)

    # Save The Trained Forest (with its window and alarm thresholds) For The Real-Time System
    def save(self, path, window, thresholds):
        self.window = float(window)
        self.thresholds = dict(thresholds)
        data = {
            "features": self.features,
            "medians": [float(v) for v in self.fill],
            "trees": self.trees,
            "window": self.window,
            "thresholds": self.thresholds,
            "importance": [float(v) for v in self.importance] if self.importance is not None else None,
            "settings": {"trees": self.nTrees, "min_samples_leaf": self.minSamplesLeaf, "seed": self.seed},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = data.get("settings", {})
        forest = cls(settings.get("trees", len(data["trees"])), settings.get("min_samples_leaf", MIN_SAMPLES_LEAF),
                     settings.get("seed", RANDOM_SEED))
        forest.features = list(data["features"])
        forest.fill = np.asarray(data["medians"], dtype=np.float64)
        forest.trees = data["trees"]
        forest.window = float(data["window"])
        forest.thresholds = data["thresholds"]
        if data.get("importance") is not None:
            forest.importance = np.asarray(data["importance"])
        return forest

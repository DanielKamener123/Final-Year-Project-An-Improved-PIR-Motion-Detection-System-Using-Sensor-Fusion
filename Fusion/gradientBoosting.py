"""First-principles Gradient Boosting (Friedman, 2001) for the fused feature vector.

Model: the human log-odds F(x) is built up from shallow regression trees,
    F(x) = F0 + learningRate * (tree_1(x) + tree_2(x) + ... + tree_M(x))
and the human probability is the logistic function p(x) = 1 / (1 + exp(-F(x))).

Training (binary log-loss):
    1. F0 = ln(p / (1 - p)), with p the fraction of human rows in the training set.
    2. For every boosting round m = 1..N_TREES:
         p_i = 1 / (1 + exp(-F(x_i)))          current human probability of every row
         r_i = y_i - p_i                        residual (negative gradient of the log-loss)
         A regression tree of depth MAX_DEPTH is fitted to the residuals. Every split
         is the one with the largest squared-error reduction
             nL * nR / (nL + nR) * (mean(r_L) - mean(r_R))^2
         Every leaf value is one Newton step on the log-loss
             gamma = sum(r_i) / sum(p_i * (1 - p_i))
         F(x_i) += learningRate * gamma(leaf of x_i)

Inference: every tree returns the gamma of the leaf the feature vector lands in;
the sum is scaled, added to F0 and passed through the logistic function.

Only numpy is used, so the same file trains offline and runs on the ROCK.
"""
import json
from pathlib import Path

import numpy as np

# Config (same settings as the scikit-learn Gradient Boosting used for model selection)
N_TREES = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 20
# Every leaf must hold at least 20 training rows (about 10 s of data at 2 frames/s),
# so a leaf cannot be fitted to one short stretch of a single recording.

LEAF = -1


# Logistic Function (probability from log-odds), safe for large |F|
def sigmoid(F):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(F, dtype=np.float64)))


# Finds The Split With The Largest Squared-Error Reduction For One Node
# rows: training rows in the node. Returns (reduction, feature, threshold) or None.
def bestSplit(X, residual, rows, minSamplesLeaf):
    best = None
    count = len(rows)
    nodeResidual = residual[rows]
    total = float(nodeResidual.sum())
    leftCount = np.arange(1, count)
    rightCount = count - leftCount

    for feature in range(X.shape[1]):
        values = X[rows, feature]
        order = np.argsort(values, kind="stable")
        sortedValues = values[order]

        # A constant feature cannot split the node
        if sortedValues[0] == sortedValues[-1]:
            continue

        # Residual sum to the left of every possible split position
        leftSum = np.cumsum(nodeResidual[order])[:-1]
        rightSum = total - leftSum

        # Only split between two different values, keeping enough rows on each side
        valid = ((sortedValues[:-1] < sortedValues[1:])
                 & (leftCount >= minSamplesLeaf) & (rightCount >= minSamplesLeaf))
        if not valid.any():
            continue

        # Squared-error reduction: nL * nR / n * (mean left - mean right)^2
        meanDifference = leftSum / leftCount - rightSum / rightCount
        reduction = leftCount * rightCount / count * meanDifference ** 2
        reduction[~valid] = -np.inf
        k = int(np.argmax(reduction))
        if best is None or reduction[k] > best[0]:
            # Threshold half-way between the two neighbouring values
            threshold = 0.5 * (sortedValues[k] + sortedValues[k + 1])
            if threshold >= sortedValues[k + 1]:
                threshold = sortedValues[k]
            best = (float(reduction[k]), int(feature), float(threshold))

    return best


# Grows One Regression Tree On The Residuals And Sets Its Leaves By A Newton Step
def growTree(X, residual, hessian, maxDepth, minSamplesLeaf, importance):
    tree = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

    def newNode():
        for key in tree:
            tree[key].append(LEAF if key in ("feature", "left", "right") else 0.0)
        return len(tree["left"]) - 1

    leaves = []
    toSplit = [(np.arange(X.shape[0]), newNode(), 0)]
    while toSplit:
        nodeRows, node, depth = toSplit.pop()

        # Leaf: maximum depth reached or too few rows to give both children enough rows
        split = None
        if depth < maxDepth and len(nodeRows) >= 2 * minSamplesLeaf:
            split = bestSplit(X, residual, nodeRows, minSamplesLeaf)
        if split is None or split[0] <= 0:
            leaves.append((node, nodeRows))
            continue

        reduction, feature, threshold = split
        importance[feature] += reduction

        goesLeft = X[nodeRows, feature] <= threshold
        left = newNode()
        right = newNode()
        tree["feature"][node] = feature
        tree["threshold"][node] = threshold
        tree["left"][node] = left
        tree["right"][node] = right
        toSplit.append((nodeRows[goesLeft], left, depth + 1))
        toSplit.append((nodeRows[~goesLeft], right, depth + 1))

    # Newton step for every leaf: gamma = sum(r) / sum(p * (1 - p))
    for node, nodeRows in leaves:
        denominator = float(hessian[nodeRows].sum())
        tree["value"][node] = float(residual[nodeRows].sum()) / denominator if denominator > 1e-150 else 0.0

    return tree


# Leaf Index Of Every Row In One Tree
def leafIndex(tree, X):
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
    return node


class GradientBoosting:
    def __init__(self, nTrees=N_TREES, maxDepth=MAX_DEPTH, learningRate=LEARNING_RATE,
                 minSamplesLeaf=MIN_SAMPLES_LEAF):
        self.nTrees = int(nTrees)
        self.maxDepth = int(maxDepth)
        self.learningRate = float(learningRate)
        self.minSamplesLeaf = int(minSamplesLeaf)
        self.features = []
        self.fill = None
        self.initScore = 0.0
        self.trees = []
        self.importance = None
        # Set when the model is saved for the real-time system
        self.window = None
        self.thresholds = None

    # Replaces Missing Values With The Training Median Of That Feature
    # Values are rounded to 32-bit floats, the precision the live feature values need
    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(X), self.fill, X).astype(np.float32).astype(np.float64)

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

        # Starting log-odds from the fraction of human rows
        humanFraction = float(y.mean())
        self.initScore = float(np.log(humanFraction / (1.0 - humanFraction)))
        F = np.full(len(y), self.initScore)

        importance = np.zeros(X.shape[1])
        self.trees = []
        for treeIndex in range(self.nTrees):
            p = sigmoid(F)
            residual = y - p
            hessian = p * (1.0 - p)

            treeImportance = np.zeros(X.shape[1])
            tree = growTree(X, residual, hessian, self.maxDepth, self.minSamplesLeaf, treeImportance)
            self.trees.append(tree)
            F += self.learningRate * np.asarray(tree["value"])[leafIndex(tree, X)]

            if treeImportance.sum() > 0:
                importance += treeImportance / treeImportance.sum()
            if progress is not None and (treeIndex + 1) % 25 == 0:
                progress(f"  boosted {treeIndex + 1}/{self.nTrees} trees")

        self.importance = importance / max(importance.sum(), 1e-300)
        return self

    # Human Log-Odds For Many Rows At Once
    def decisionFunction(self, X):
        X = self._prepare(X)
        F = np.full(len(X), self.initScore)
        for tree in self.trees:
            F += self.learningRate * np.asarray(tree["value"])[leafIndex(tree, X)]
        return F

    # Human Probability For Many Rows At Once (offline evaluation)
    def predictProba(self, X):
        return sigmoid(self.decisionFunction(X))

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
        values = [float(np.float32(v)) for v in values]
        F = self.initScore
        for tree in self.trees:
            node = 0
            while tree["left"][node] != LEAF:
                if values[tree["feature"][node]] <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]
            F += self.learningRate * tree["value"][node]
        return float(sigmoid(F))

    # Save The Trained Model (with its window and alarm thresholds) For The Real-Time System
    def save(self, path, window, thresholds):
        self.window = float(window)
        self.thresholds = dict(thresholds)
        data = {
            "model": "gradient_boosting",
            "features": self.features,
            "medians": [float(v) for v in self.fill],
            "init_score": self.initScore,
            "trees": self.trees,
            "window": self.window,
            "thresholds": self.thresholds,
            "importance": [float(v) for v in self.importance] if self.importance is not None else None,
            "settings": {"trees": self.nTrees, "max_depth": self.maxDepth, "learning_rate": self.learningRate,
                         "min_samples_leaf": self.minSamplesLeaf},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = data.get("settings", {})
        model = cls(settings.get("trees", len(data["trees"])), settings.get("max_depth", MAX_DEPTH),
                    settings.get("learning_rate", LEARNING_RATE), settings.get("min_samples_leaf", MIN_SAMPLES_LEAF))
        model.features = list(data["features"])
        model.fill = np.asarray(data["medians"], dtype=np.float64)
        model.initScore = float(data["init_score"])
        model.trees = data["trees"]
        model.window = float(data["window"])
        model.thresholds = data["thresholds"]
        if data.get("importance") is not None:
            model.importance = np.asarray(data["importance"])
        return model

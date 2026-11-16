import json
from pathlib import Path

import numpy as np

# Training settings
# 100 small trees, each with at most three split levels
# Learning rate controls how much each tree changes the prediction
# Minimum leaf size avoids fitting to very small groups
N_TREES = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 20

# Special value used to mark the end of a tree branch
LEAF = -1

# Converts the model score from log odds to a probability
# The tanh form handles very large scores safely
def sigmoid(F):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(F, dtype=np.float64)))

# Find the best feature and threshold for splitting a tree node
# Choose the split that best separates the remaining prediction errors
def bestSplit(X, residual, rows, minSamplesLeaf):

    # Track the best split found so far
    best = None
    count = len(rows)

    # Get the errors belonging to this node
    nodeResidual = residual[rows]
    total = float(nodeResidual.sum())

    # Number of samples on either side of every possible split
    leftCount = np.arange(1, count)
    rightCount = count - leftCount

    # Test each input feature as a possible split
    for feature in range(X.shape[1]):

        # Sort feature values so possible split points are adjacent
        values = X[rows, feature]
        order = np.argsort(values, kind="stable")
        sortedValues = values[order]

        # Skip features that have the same value for every sample
        if sortedValues[0] == sortedValues[-1]:
            continue

        # Cumulative errors on the left and right of each split
        leftSum = np.cumsum(nodeResidual[order])[:-1]
        rightSum = total - leftSum

        # Only split between different values
        # Both child nodes must contain enough samples
        valid = ((sortedValues[:-1] < sortedValues[1:])
                 & (leftCount >= minSamplesLeaf) & (rightCount >= minSamplesLeaf))
        if not valid.any():
            continue

        # Compare average errors on both sides
        # Larger squared error reduction means a better split
        meanDifference = leftSum / leftCount - rightSum / rightCount
        reduction = leftCount * rightCount / count * meanDifference ** 2
        reduction[~valid] = -np.inf

        # Take the split with the greatest improvement
        k = int(np.argmax(reduction))
        if best is None or reduction[k] > best[0]:

            # Place the threshold between the two neighbouring values
            threshold = 0.5 * (sortedValues[k] + sortedValues[k + 1])
            if threshold >= sortedValues[k + 1]:
                threshold = sortedValues[k]
            best = (float(reduction[k]), int(feature), float(threshold))

    return best

# Build one regression tree to correct the current model errors
# Split on residuals, then calculate each final leaf correction
def growTree(X, residual, hessian, maxDepth, minSamplesLeaf, importance):
    tree = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

    # Add an empty tree node and return its position
    def newNode():
        for key in tree:
            tree[key].append(LEAF if key in ("feature", "left", "right") else 0.0)

        return len(tree["left"]) - 1

    # Start at the root with every training sample
    # Keep a list of nodes still needing a split
    leaves = []
    toSplit = [(np.arange(X.shape[0]), newNode(), 0)]
    while toSplit:

        # Process one node at its current depth
        nodeRows, node, depth = toSplit.pop()

        split = None

        # Stop splitting at the depth limit or with too few samples
        if depth < maxDepth and len(nodeRows) >= 2 * minSamplesLeaf:
            split = bestSplit(X, residual, nodeRows, minSamplesLeaf)

        # Save nodes that cannot be split further as leaves
        if split is None or split[0] <= 0:
            leaves.append((node, nodeRows))
            continue

        reduction, feature, threshold = split

        # Record how useful this feature was for the split
        importance[feature] += reduction

        # Send samples to the left or right child by the threshold
        goesLeft = X[nodeRows, feature] <= threshold
        left = newNode()
        right = newNode()
        tree["feature"][node] = feature
        tree["threshold"][node] = threshold
        tree["left"][node] = left
        tree["right"][node] = right
        toSplit.append((nodeRows[goesLeft], left, depth + 1))
        toSplit.append((nodeRows[~goesLeft], right, depth + 1))

    # Set each leaf value using a Newton correction
    # Residual sum gives the remaining error
    # Hessian sum scales the correction to the probability sensitivity
    for node, nodeRows in leaves:
        denominator = float(hessian[nodeRows].sum())
        tree["value"][node] = float(residual[nodeRows].sum()) / denominator if denominator > 1e-150 else 0.0

    return tree

# Find the final leaf reached by each input row
def leafIndex(tree, X):
    feature = np.asarray(tree["feature"])
    threshold = np.asarray(tree["threshold"])
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])

    # Every row starts at the tree root
    node = np.zeros(len(X), dtype=int)
    active = left[node] != LEAF

    # Follow split thresholds until every row reaches a leaf
    while active.any():
        rows = np.flatnonzero(active)
        current = node[rows]
        goesLeft = X[rows, feature[current]] <= threshold[current]
        node[rows] = np.where(goesLeft, left[current], right[current])
        active = left[node] != LEAF

    return node

# Custom gradient boosting classifier for human detection
class GradientBoosting:

    # Store training settings and initialise model data
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

        # Window duration and alarm thresholds are stored when exporting
        self.window = None
        self.thresholds = None

    # Replace missing feature values using training medians
    # Match the numeric precision used by the live model
    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(X), self.fill, X).astype(np.float32).astype(np.float64)

    # Train the classifier on feature rows and human labels
    # X contains sensor features, y contains zero or one
    def fit(self, X, y, featureNames, progress=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Check that feature names and training labels are valid
        if X.shape[1] != len(featureNames):
            raise ValueError("Feature names do not match the training columns")
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training labels must contain both 0 and 1")

        # Calculate a median replacement value for each feature
        self.features = list(featureNames)
        self.fill = np.array([np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.0
                              for column in X.T])
        X = self._prepare(X)

        # The mean of zero and one labels is the fraction of human rows
        # Convert this starting probability to log odds
        humanFraction = float(y.mean())
        self.initScore = float(np.log(humanFraction / (1.0 - humanFraction)))
        F = np.full(len(y), self.initScore)

        # Add regression trees one at a time
        # Each new tree learns from the errors of the current prediction
        importance = np.zeros(X.shape[1])
        self.trees = []
        for treeIndex in range(self.nTrees):

            # Convert the current model scores into human probabilities
            p = sigmoid(F)

            # Residual is actual label minus predicted probability
            # Hessian measures how sensitive the probability is to score changes
            residual = y - p
            hessian = p * (1.0 - p)

            # Fit the next tree to the residuals and calculate its leaf values
            treeImportance = np.zeros(X.shape[1])
            tree = growTree(X, residual, hessian, self.maxDepth, self.minSamplesLeaf, treeImportance)
            self.trees.append(tree)

            # Add a small correction from the new tree to every row
            F += self.learningRate * np.asarray(tree["value"])[leafIndex(tree, X)]

            # Combine the relative feature importance from all trees
            if treeImportance.sum() > 0:
                importance += treeImportance / treeImportance.sum()
            if progress is not None and (treeIndex + 1) % 25 == 0:
                progress(f"  boosted {treeIndex + 1}/{self.nTrees} trees")

        self.importance = importance / max(importance.sum(), 1e-300)

        return self

    # Sum the starting log odds and corrections from every tree
    def decisionFunction(self, X):
        X = self._prepare(X)
        F = np.full(len(X), self.initScore)
        for tree in self.trees:
            F += self.learningRate * np.asarray(tree["value"])[leafIndex(tree, X)]

        return F

    # Convert final scores to human probabilities for multiple rows
    def predictProba(self, X):

        return sigmoid(self.decisionFunction(X))

    # Predict the human probability for one live sensor feature vector
    def confidence(self, features):

        # Accept either named features or an ordered feature list
        if isinstance(features, dict):
            values = [features[name] for name in self.features]
        else:
            values = list(features)
        if len(values) != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features; got {len(values)}")

        # Fill missing values and reject invalid numeric inputs
        values = [float(self.fill[i]) if np.isnan(v) else float(v) for i, v in enumerate(values)]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature vector")
        values = [float(np.float32(v)) for v in values]

        # Start with the training human fraction expressed as log odds
        F = self.initScore
        for tree in self.trees:
            node = 0

            # Walk down the tree based on the current feature values
            while tree["left"][node] != LEAF:
                if values[tree["feature"][node]] <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]

            # Apply the leaf correction with the learning rate
            F += self.learningRate * tree["value"][node]
        return float(sigmoid(F))

    # Save all model settings, tree structures and alarm thresholds
    # The live system can load this file without retraining
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

    # Load a previously saved model for prediction
    @classmethod
    def load(cls, path):

        # Read saved training settings and tree data
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = data.get("settings", {})
        model = cls(settings.get("trees", len(data["trees"])), settings.get("max_depth", MAX_DEPTH),
                    settings.get("learning_rate", LEARNING_RATE), settings.get("min_samples_leaf", MIN_SAMPLES_LEAF))

        # Restore the trained feature order and model parameters
        model.features = list(data["features"])
        model.fill = np.asarray(data["medians"], dtype=np.float64)
        model.initScore = float(data["init_score"])
        model.trees = data["trees"]
        model.window = float(data["window"])
        model.thresholds = data["thresholds"]
        if data.get("importance") is not None:
            model.importance = np.asarray(data["importance"])
        return model

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Config
RANDOM_STATE = 42
LABEL_COLUMN = "human"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


# Optional CCA feature transformation
class CCAFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        thermal_indices,
        pir_indices,
        n_components=2
    ):
        self.thermal_indices = thermal_indices
        self.pir_indices = pir_indices
        self.n_components = n_components

    def fit(self, X, y=None):

        X_thermal = X[:, self.thermal_indices]
        X_pir = X[:, self.pir_indices]

        n_components = min(
            self.n_components,
            X_thermal.shape[1],
            X_pir.shape[1],
            X.shape[0] - 1
        )

        self.cca_ = CCA(
            n_components=n_components,
            scale=False,
            max_iter=1000
        )

        self.cca_.fit(X_thermal, X_pir)

        return self

    def transform(self, X):

        X_thermal = X[:, self.thermal_indices]
        X_pir = X[:, self.pir_indices]

        thermal_cca, pir_cca = self.cca_.transform(
            X_thermal,
            X_pir
        )

        # Keep original features as well as CCA features
        return np.hstack([
            X,
            thermal_cca,
            pir_cca
        ])


# Load datasets
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)


# Separate features and labels
X_train = train_data.drop(columns=[LABEL_COLUMN])
y_train = train_data[LABEL_COLUMN]

X_test = test_data.drop(columns=[LABEL_COLUMN])
y_test = test_data[LABEL_COLUMN]


# Ensure both datasets have the same features
if set(X_train.columns) != set(X_test.columns):
    raise ValueError(
        "Training and test datasets must contain the same feature columns."
    )

X_test = X_test[X_train.columns]


# Ensure all features are numerical
X_train = X_train.apply(pd.to_numeric, errors="raise")
X_test = X_test.apply(pd.to_numeric, errors="raise")


# Check labels
if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Training labels must be 0 or 1.")

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Test labels must be 0 or 1.")


# Preprocessing
scaled_preprocessing = [
    SimpleImputer(strategy="median"),
    StandardScaler()
]

tree_preprocessing = [
    SimpleImputer(strategy="median")
]


# Define algorithms
models = {

    "Logistic Regression": make_pipeline(
        *scaled_preprocessing,
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    ),

    "Linear SVM": make_pipeline(
        *scaled_preprocessing,
        SVC(
            kernel="linear",
            random_state=RANDOM_STATE
        )
    ),

    "RBF SVM": make_pipeline(
        *scaled_preprocessing,
        SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        )
    ),

    "Random Forest": make_pipeline(
        *tree_preprocessing,
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    ),

    "k-Nearest Neighbours": make_pipeline(
        *scaled_preprocessing,
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "Decision Tree": make_pipeline(
        *tree_preprocessing,
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )
    ),

    "Gradient Boosting": make_pipeline(
        *tree_preprocessing,
        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    ),

    "Gaussian Naive Bayes": make_pipeline(
        *scaled_preprocessing,
        GaussianNB()
    ),

    "Neural Network (MLP)": make_pipeline(
        *scaled_preprocessing,
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    ),

    "PCA + RBF SVM": make_pipeline(
        *scaled_preprocessing,
        PCA(
            n_components=0.95
        ),
        SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        )
    )
}


# Optional CCA + SVM
# Requires feature names such as:
# thermal_area, thermal_motion, pir_energy, pir_amplitude

thermal_indices = [
    i for i, column in enumerate(X_train.columns)
    if column.startswith("thermal_")
]

pir_indices = [
    i for i, column in enumerate(X_train.columns)
    if column.startswith("pir_")
]

if len(thermal_indices) >= 2 and len(pir_indices) >= 2:

    models["CCA + RBF SVM"] = make_pipeline(
        *scaled_preprocessing,
        CCAFeatures(
            thermal_indices=thermal_indices,
            pir_indices=pir_indices,
            n_components=2
        ),
        SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        )
    )


# Train and test every algorithm
results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    try:

        # Train model
        model.fit(X_train, y_train)

        # Test model
        predictions = model.predict(X_test)

        # Calculate performance
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        # Store results
        results.append({
            "Algorithm": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Print detailed results
        print("\nConfusion Matrix:")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ))

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Human", "Human"],
            zero_division=0
        ))

    except Exception as error:

        print(f"Model failed: {error}")


# Create leaderboard
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Score", "Accuracy"],
    ascending=False
).reset_index(drop=True)


# Display results
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# Save results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


# Display best model
if not results_df.empty:

    print("\nBest model based on F1 Score:")
    print(results_df.iloc[0]["Algorithm"])
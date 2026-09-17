import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

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


# Preprocessing
# Fusion combination step: PIR and thermopile feature columns arrive
# already concatenated into one row per sample (feature-level fusion).
# Rescaling both sensors onto a common scale here, before the classifier
# sees them, is the chosen combination method -- it stops whichever
# sensor happens to have larger raw numbers from dominating the model.
scaled_preprocessing = [
    SimpleImputer(strategy="median"),
    StandardScaler()
]

tree_preprocessing = [
    SimpleImputer(strategy="median")
]


# Define algorithms
def buildModels():
    return {

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


def runModels(train_data, test_data, outputPath="model_comparison_results.csv"):
    featureColumns = [
        column for column in train_data.columns
        if column.startswith("pir_") or column.startswith("thermal_")
    ]

    # Track ID is an internal identifier rather than a physical feature.
    featureColumns = [column for column in featureColumns if column != "thermal_track_id"]

    # Separate features and labels
    X_train = train_data[featureColumns].copy()
    y_train = train_data[LABEL_COLUMN]

    X_test = test_data[featureColumns].copy()
    y_test = test_data[LABEL_COLUMN]

    # Ensure both datasets have the same features
    missingTest = [column for column in featureColumns if column not in test_data.columns]
    if missingTest:
        raise ValueError(
            "Training and test datasets must contain the same feature columns. "
            f"Missing from test data: {missingTest}"
        )

    X_test = X_test[featureColumns]

    # Ensure all features are numerical
    X_train = X_train.apply(pd.to_numeric, errors="raise")
    X_test = X_test.apply(pd.to_numeric, errors="raise")

    # Check labels
    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test labels must be 0 or 1.")

    models = buildModels()

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

    if not results_df.empty:
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
        outputPath,
        index=False
    )

    # Display best model
    if not results_df.empty:

        print("\nBest model based on F1 Score:")
        print(results_df.iloc[0]["Algorithm"])

    return results_df

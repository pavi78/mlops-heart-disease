"""
MLOps training pipeline for the Heart Disease classification project.

This script:
1. Loads the raw UCI heart disease dataset
2. Converts the original multiclass target to binary
3. Performs a stratified train/test split
4. Builds preprocessing pipelines
5. Tunes Logistic Regression and Random Forest
6. Evaluates both models on the untouched test set
7. Selects the best model using CV ROC-AUC
8. Logs the experiment to MLflow
9. Saves the final model and training artifacts
"""

from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ===================================================================
# PROJECT PATHS
# ===================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "heart_disease_raw.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# CONFIGURATION
# ===================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Convert Windows path to a SQLite-compatible URI
MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(
    "Heart Disease Classification"
)


# ===================================================================
# LOAD DATA
# ===================================================================

print("=" * 70)
print("HEART DISEASE MLOPS TRAINING PIPELINE")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ===================================================================
# CREATE BINARY TARGET
# ===================================================================

# Original target:
# 0       -> No disease
# 1 to 4  -> Disease

df["target"] = (df["num"] > 0).astype(int)

df = df.drop(columns=["num"])

X = df.drop(columns=["target"])
y = df["target"]

print("\nBinary target distribution:")
print(y.value_counts().sort_index())

print("\nBinary target percentages:")
print(
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ===================================================================
# TRAIN / TEST SPLIT
# ===================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain/Test Split:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")


# ===================================================================
# FEATURE GROUPS
# ===================================================================

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]


# ===================================================================
# PREPROCESSING PIPELINES
# ===================================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ===================================================================
# MODEL PIPELINES
# ===================================================================

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)


# ===================================================================
# CROSS-VALIDATION
# ===================================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ===================================================================
# HYPERPARAMETER GRIDS
# ===================================================================

logistic_param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ],
    "classifier__solver": [
        "liblinear",
        "lbfgs",
    ],
}


rf_param_grid = {
    "classifier__n_estimators": [
        100,
        200,
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
    ],
    "classifier__min_samples_split": [
        2,
        5,
    ],
    "classifier__min_samples_leaf": [
        1,
        2,
    ],
}


# ===================================================================
# EVALUATION FUNCTION
# ===================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a binary classification model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# ===================================================================
# MLflow RUN
# ===================================================================

with mlflow.start_run(
    run_name="Heart Disease Model Training"
):

    print("\n" + "=" * 70)
    print("MLFLOW RUN STARTED")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Log general experiment parameters
    # ---------------------------------------------------------------

    mlflow.log_params(
        {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": 5,
            "dataset_rows": len(df),
            "dataset_features": len(X.columns),
            "numerical_feature_count": len(
                numerical_features
            ),
            "categorical_feature_count": len(
                categorical_features
            ),
        }
    )

    # ---------------------------------------------------------------
    # Logistic Regression tuning
    # ---------------------------------------------------------------

    print("\nTuning Logistic Regression...")

    logistic_grid = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    logistic_grid.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest Logistic Regression parameters:"
    )
    print(logistic_grid.best_params_)

    print(
        "Best Logistic Regression CV ROC-AUC:",
        round(
            logistic_grid.best_score_,
            6,
        ),
    )

    # ---------------------------------------------------------------
    # Random Forest tuning
    # ---------------------------------------------------------------

    print("\nTuning Random Forest...")

    rf_grid = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=rf_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    rf_grid.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest Random Forest parameters:"
    )
    print(rf_grid.best_params_)

    print(
        "Best Random Forest CV ROC-AUC:",
        round(
            rf_grid.best_score_,
            6,
        ),
    )

    # ---------------------------------------------------------------
    # Evaluate tuned models
    # ---------------------------------------------------------------

    logistic_metrics = evaluate_model(
        logistic_grid.best_estimator_,
        X_test,
        y_test,
    )

    rf_metrics = evaluate_model(
        rf_grid.best_estimator_,
        X_test,
        y_test,
    )

    # ---------------------------------------------------------------
    # Create results table
    # ---------------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "Tuned Logistic Regression",
                **logistic_metrics,
            },
            {
                "model": "Tuned Random Forest",
                **rf_metrics,
            },
        ]
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Select final model using CV ROC-AUC
    # ---------------------------------------------------------------

    if (
        logistic_grid.best_score_
        >= rf_grid.best_score_
    ):
        final_model = (
            logistic_grid.best_estimator_
        )

        final_model_name = (
            "Tuned Logistic Regression"
        )

        final_cv_roc_auc = (
            logistic_grid.best_score_
        )

        final_metrics = logistic_metrics

        final_best_params = (
            logistic_grid.best_params_
        )

    else:
        final_model = (
            rf_grid.best_estimator_
        )

        final_model_name = (
            "Tuned Random Forest"
        )

        final_cv_roc_auc = (
            rf_grid.best_score_
        )

        final_metrics = rf_metrics

        final_best_params = (
            rf_grid.best_params_
        )

    print(
        f"\nSelected final model: "
        f"{final_model_name}"
    )

    # ---------------------------------------------------------------
    # Log final model parameters
    # ---------------------------------------------------------------

    mlflow.log_param(
        "selected_model",
        final_model_name,
    )

    mlflow.log_param(
        "selection_metric",
        "CV ROC-AUC",
    )

    mlflow.log_metric(
        "final_cv_roc_auc",
        final_cv_roc_auc,
    )

    # Log best hyperparameters
    for parameter, value in final_best_params.items():

        mlflow.log_param(
            parameter,
            str(value),
        )

    # ---------------------------------------------------------------
    # Log final test metrics
    # ---------------------------------------------------------------

    mlflow.log_metrics(
        {
            "test_accuracy": final_metrics[
                "accuracy"
            ],
            "test_precision": final_metrics[
                "precision"
            ],
            "test_recall": final_metrics[
                "recall"
            ],
            "test_roc_auc": final_metrics[
                "roc_auc"
            ],
            "logistic_test_accuracy": logistic_metrics[
                "accuracy"
            ],
            "logistic_test_precision": logistic_metrics[
                "precision"
            ],
            "logistic_test_recall": logistic_metrics[
                "recall"
            ],
            "logistic_test_roc_auc": logistic_metrics[
                "roc_auc"
            ],
            "random_forest_test_accuracy": rf_metrics[
                "accuracy"
            ],
            "random_forest_test_precision": rf_metrics[
                "precision"
            ],
            "random_forest_test_recall": rf_metrics[
                "recall"
            ],
            "random_forest_test_roc_auc": rf_metrics[
                "roc_auc"
            ],
            "logistic_cv_roc_auc": logistic_grid.best_score_,
            "random_forest_cv_roc_auc": rf_grid.best_score_,
        }
    )

    # ---------------------------------------------------------------
    # Save results CSV
    # ---------------------------------------------------------------

    results_path = (
        ARTIFACTS_DIR
        / "training_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------------

    metadata = {
        "final_model": final_model_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "logistic_best_params":
            logistic_grid.best_params_,
        "logistic_cv_roc_auc":
            logistic_grid.best_score_,
        "random_forest_best_params":
            rf_grid.best_params_,
        "random_forest_cv_roc_auc":
            rf_grid.best_score_,
        "final_test_metrics":
            final_metrics,
    }

    metadata_path = (
        ARTIFACTS_DIR
        / "training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    confusion_matrix_path = (
        FIGURES_DIR
        / "confusion_matrix_final_model.png"
    )

    ConfusionMatrixDisplay.from_estimator(
        final_model,
        X_test,
        y_test,
        display_labels=[
            "No Disease",
            "Disease",
        ],
    )

    plt.title(
        "Confusion Matrix - "
        + final_model_name
    )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------------

    roc_curve_path = (
        FIGURES_DIR
        / "roc_curve_final_model.png"
    )

    RocCurveDisplay.from_estimator(
        final_model,
        X_test,
        y_test,
    )

    plt.title(
        "ROC Curve - "
        + final_model_name
    )

    plt.tight_layout()

    plt.savefig(
        roc_curve_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------------
    # Log artifacts
    # ---------------------------------------------------------------

    mlflow.log_artifact(
        str(results_path),
        artifact_path="results",
    )

    mlflow.log_artifact(
        str(metadata_path),
        artifact_path="results",
    )

    mlflow.log_artifact(
        str(confusion_matrix_path),
        artifact_path="figures",
    )

    mlflow.log_artifact(
        str(roc_curve_path),
        artifact_path="figures",
    )

    # ---------------------------------------------------------------
    # Log final model to MLflow
    # ---------------------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=final_model,
        name="heart_disease_model",
        skops_trusted_types=[
            "numpy.dtype"
        ],
    )

    # ---------------------------------------------------------------
    # Also save local joblib model
    # ---------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "heart_disease_model.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(
        f"\nLocal model saved to: "
        f"{model_path}"
    )

    # ---------------------------------------------------------------
    # Print MLflow information
    # ---------------------------------------------------------------

    current_run = mlflow.active_run()

    print("\n" + "=" * 70)
    print("MLFLOW RUN COMPLETED")
    print("=" * 70)

    print(
        f"Experiment: "
        f"Heart Disease Classification"
    )

    print(
        f"Run ID: "
        f"{current_run.info.run_id}"
    )

    print(
        f"Selected model: "
        f"{final_model_name}"
    )

    print(
        f"Test ROC-AUC: "
        f"{final_metrics['roc_auc']:.6f}"
    )

    print(
        "\nTraining pipeline completed successfully."
    )
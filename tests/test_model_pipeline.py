
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_test_pipeline():
    numeric_features = ["age", "chol"]
    categorical_features = ["sex", "cp"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def sample_data():
    X = pd.DataFrame({
        "age": [45, 52, 38, 61, 49, 57, 42, 65],
        "chol": [220, 250, 190, 280, 230, 260, 200, 290],
        "sex": [1, 0, 1, 0, 1, 0, 1, 0],
        "cp": [0, 1, 2, 3, 1, 2, 0, 3],
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test_pipeline_fits_and_predicts():
    X, y = sample_data()
    model = create_test_pipeline()

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_pipeline_handles_missing_values():
    X, y = sample_data()
    X.loc[0, "age"] = np.nan
    X.loc[1, "sex"] = np.nan

    model = create_test_pipeline()
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_pipeline_handles_unseen_categories():
    X, y = sample_data()
    model = create_test_pipeline()
    model.fit(X, y)

    new_data = pd.DataFrame({
        "age": [50],
        "chol": [240],
        "sex": [1],
        "cp": [99],
    })

    prediction = model.predict(new_data)

    assert prediction[0] in [0, 1]


def test_evaluation_metrics_are_valid():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.8, 0.9]

    assert 0 <= accuracy_score(y_true, y_pred) <= 1
    assert 0 <= precision_score(y_true, y_pred) <= 1
    assert 0 <= recall_score(y_true, y_pred) <= 1
    assert 0 <= roc_auc_score(y_true, y_prob) <= 1

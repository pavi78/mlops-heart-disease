import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------------------------------------
# Load model from MLflow Model Registry
# ---------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "model_artifacts"

if os.getenv("DOCKER_MODE") == "true":
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
else:
    MODEL_URI = "models:/HeartDiseaseModel@champion"
    model = mlflow.sklearn.load_model(MODEL_URI)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps inference API using FastAPI and MLflow Model Registry",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app)

# ---------------------------------------------------------
# Input schema
# ---------------------------------------------------------
class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float | None = None
    thal: float | None = None


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running",
        "model": "HeartDiseaseModel@champion",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "HeartDiseaseModel@champion",
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = pd.DataFrame(
        [
            {
                "age": data.age,
                "sex": data.sex,
                "cp": data.cp,
                "trestbps": data.trestbps,
                "chol": data.chol,
                "fbs": data.fbs,
                "restecg": data.restecg,
                "thalach": data.thalach,
                "exang": data.exang,
                "oldpeak": data.oldpeak,
                "slope": data.slope,
                "ca": data.ca,
                "thal": data.thal,
            }
        ]
    )

    prediction = int(model.predict(input_data)[0])

    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "model": "HeartDiseaseModel@champion",
    }
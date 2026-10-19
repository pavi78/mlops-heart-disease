from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "HeartDiseaseModel@champion"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "HeartDiseaseModel@champion"


def test_prediction():
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "model" in data

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model"] == "HeartDiseaseModel@champion"


def test_invalid_prediction():
    payload = {
        "age": "invalid",
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
# Heart Disease Prediction — MLOps Pipeline

## Project Overview
This project implements an end-to-end machine learning pipeline to predict heart disease using the UCI Heart Disease dataset. It covers data preprocessing, model training, experiment tracking, model registration, API development, Docker containerization, and automated CI/CD.

## Technology Stack
- Python 3.11
- Pandas and scikit-learn
- MLflow for experiment tracking and model registry
- FastAPI for prediction APIs
- Pytest for automated testing
- Docker for containerization
- GitHub Actions for CI/CD
- GitHub Container Registry (GHCR) for image publishing

## Model Development
The dataset contains 303 records. The original target is converted into a binary label, where `0` represents the negative class and values greater than `0` represent the positive class.

The preprocessing pipeline handles missing values, scales numerical features, and one-hot encodes categorical features. Logistic Regression and Random Forest models were evaluated using five-fold cross-validation and hyperparameter tuning.

The tuned Logistic Regression model was selected as the final model.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| ROC-AUC | 0.9621 |

## MLflow
- Experiment: `Heart Disease Classification`
- Registered model: `HeartDiseaseModel`
- Model version: `1`
- Model alias: `champion`

MLflow tracks experiments and stores model artifacts. The local API loads the registered model, while the Docker deployment uses packaged model artifacts.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Generate predictions |

Interactive API documentation is available at `http://127.0.0.1:8000/docs` when the container is running.

## Docker
Build the image:

```bash
docker build -t heart-disease-api:1.0 .
```

Run the container:

```bash
docker run -d --name heart-disease-api-container -p 8000:8000 heart-disease-api:1.0
```

## CI/CD and Container Registry
GitHub Actions automates API testing, Docker image building, and publishing the image to GitHub Container Registry when changes are pushed to `main` and the required jobs succeed.

Published image:

`ghcr.io/pavi78/mlops-heart-disease:main`

Pull the image:

```bash
docker pull ghcr.io/pavi78/mlops-heart-disease:main
```

## Verification
The following have been verified:
- Automated API tests: four tests passed.
- Docker image successfully pulled from GHCR.
- Health endpoint returned a healthy response.
- Prediction endpoint returned HTTP 200.
- GitHub Actions test, build, and publish jobs completed successfully.

## Limitations and Future Improvements
Potential improvements include Kubernetes deployment, application monitoring, model drift detection, and automated retraining, depending on project requirements.

**Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis or treatment.

# Heart Disease Prediction — MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline to predict heart disease using the UCI Heart Disease dataset. It covers data preprocessing, model training and evaluation, hyperparameter tuning, experiment tracking, model registration, API development, Docker containerization, Kubernetes deployment, automated CI/CD, application logging, and Prometheus metrics.

The project demonstrates how a trained machine learning model can be packaged, deployed, tested, and monitored through a reproducible workflow.

## 2. Technology Stack

- **Language:** Python 3.11
- **Data Processing and ML:** Pandas, NumPy, scikit-learn
- **Experiment Tracking and Model Registry:** MLflow
- **API Development:** FastAPI
- **Testing:** Pytest
- **Containerization:** Docker
- **Orchestration:** Kubernetes using Docker Desktop
- **CI/CD:** GitHub Actions
- **Container Registry:** GitHub Container Registry (GHCR)
- **Monitoring:** Prometheus-compatible metrics using `prometheus-fastapi-instrumentator`

## 3. Dataset and Model Development

The dataset contains 303 records. The original target variable is converted into a binary classification label:

- `0`: Negative class
- Values greater than `0`: Positive class

The preprocessing pipeline handles missing values, scales numerical features, and one-hot encodes categorical features.

Logistic Regression and Random Forest models were evaluated using stratified train-test splitting, five-fold cross-validation, and hyperparameter tuning.

The tuned Logistic Regression model was selected as the final model based on its test-set performance.

### Final Model Performance

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| ROC-AUC | 0.9621 |

The final model achieved a ROC-AUC of approximately 0.9621 on the held-out test set.

## 4. MLflow Experiment Tracking and Model Registry

MLflow is used to track experiments, record model performance, and manage the registered model.

- **Experiment:** `Heart Disease Classification`
- **Registered model:** `HeartDiseaseModel`
- **Model version:** `1`
- **Model alias:** `champion`

The local API loads the registered model through MLflow. The containerized deployment uses packaged model artifacts so that inference can run inside the container without depending on the local MLflow tracking server.

## 5. API Development

The prediction service is implemented using FastAPI.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Generate a heart disease prediction |
| GET | `/metrics` | Expose Prometheus-compatible metrics |

### Running Locally with Docker

When the container is running with the port mapping shown below, the interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

The API accepts the required patient feature values and returns a binary prediction, prediction probability, and model identifier.

## 6. Docker Containerization

The FastAPI application and its model artifacts are packaged into a Docker image.

### Build the Docker Image

```bash
docker build -t heart-disease-api:1.1 .
```

### Run the Container

```bash
docker run -d --name heart-disease-api-container -p 8000:8000 heart-disease-api:1.1
```

### Verify the API

Open the following URLs when the container is running:

- Health check: `http://127.0.0.1:8000/health`
- API documentation: `http://127.0.0.1:8000/docs`
- Prometheus metrics: `http://127.0.0.1:8000/metrics`

The image version `1.1` includes Prometheus instrumentation.

## 7. Kubernetes Deployment

The application is deployed to a local Kubernetes cluster managed by Docker Desktop.

The Kubernetes configuration files are located in the `k8s/` directory:

- `deployment.yaml`: Defines the application Deployment and container configuration.
- `service.yaml`: Exposes the application internally through a Kubernetes Service.

### Apply the Kubernetes Configuration

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Verify the Deployment

Check the rollout status:

```powershell
kubectl rollout status deployment/heart-disease-api
```

Check the running pods:

```powershell
kubectl get pods
```

Check the Service:

```powershell
kubectl get services
```

### Access the API Using Port Forwarding

The Kubernetes Service exposes the API on port `8000` inside the cluster. Forward it to local port `8001`:

```powershell
kubectl port-forward service/heart-disease-api 8001:8000
```

Keep the terminal running while accessing the service.

- API documentation: `http://127.0.0.1:8001/docs`
- Health check: `http://127.0.0.1:8001/health`
- Prometheus metrics: `http://127.0.0.1:8001/metrics`

## 8. CI/CD Pipeline

GitHub Actions automates testing, Docker image building, and image publishing to GitHub Container Registry.

The workflow is defined in `.github/workflows/ci.yml`.

### Pipeline Jobs

1. **Test:** Installs dependencies and runs the API test suite.
2. **Docker Build:** Builds the application image after the tests succeed.
3. **Publish:** Builds and publishes the container image to GHCR when changes are pushed to the `main` branch and the required jobs succeed.

### Published Container Image

```text
ghcr.io/pavi78/mlops-heart-disease:main
```

Pull the published image:

```bash
docker pull ghcr.io/pavi78/mlops-heart-disease:main
```

If the package is private, authenticate to GHCR before pulling the image.

### CI/CD Verification

The GitHub Actions workflow completed successfully for the Prometheus instrumentation changes, with all three jobs—test, Docker build, and publish—passing.

## 9. Monitoring and Logging

### Application Logging

The FastAPI application writes HTTP request logs to standard output. Kubernetes captures these logs, allowing health-check and prediction requests to be inspected.

View recent application logs:

```powershell
kubectl logs deployment/heart-disease-api --tail=50
```

View prediction requests from the last five minutes:

```powershell
kubectl logs deployment/heart-disease-api --since=5m | Select-String "/predict"
```

Successful requests return HTTP `200 OK`. These logs help verify that the API is receiving and processing requests.

### Prometheus Metrics

The API uses `prometheus-fastapi-instrumentator` to expose Prometheus-compatible metrics through the `/metrics` endpoint.

The metrics include:

- HTTP request counts grouped by handler, method, and response status.
- HTTP request duration and latency histograms.
- Python garbage-collection statistics.
- Process memory usage and CPU time.

### Accessing the Metrics Endpoint

With Kubernetes port forwarding active, open:

`http://127.0.0.1:8001/metrics`

The endpoint was verified against the running Kubernetes deployment. The observed metrics included successful health-check and prediction requests, along with request-duration measurements.

**Monitoring scope:** The application exposes Prometheus-compatible metrics, but a separate Prometheus server and Grafana dashboard have not been configured as part of the current implementation.

## 10. Testing and Verification

The following project components have been verified:

- Four API tests passed using Pytest.
- The Docker image built successfully.
- The API health endpoint returned a healthy response.
- The prediction endpoint returned HTTP `200 OK`.
- The Kubernetes Deployment rolled out successfully.
- Kubernetes logs captured health-check and prediction requests.
- The `/metrics` endpoint exposed Prometheus-compatible metrics.
- GitHub Actions completed the test, Docker build, and publish jobs successfully.

## 11. Limitations and Future Improvements

Potential extensions to the current implementation include:

- Deploying a dedicated Prometheus server and Grafana dashboard.
- Adding model drift and data quality monitoring.
- Implementing automated model retraining and evaluation.
- Adding automated deployment and rollback strategies.
- Tracking prediction latency and model performance over time.

These are future improvements and are not claimed as implemented in the current project.

## Disclaimer

This project is for educational purposes only and is not intended for medical diagnosis or treatment. Predictions generated by the model should not be used as a substitute for professional medical advice.

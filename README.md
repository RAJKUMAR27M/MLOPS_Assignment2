# Cats vs Dogs MLOps Assignment 2

This repository contains a complete end-to-end MLOps solution for the Cats vs Dogs image classification task. It includes data preprocessing, model training, evaluation, FastAPI deployment, Docker packaging, CI/CD automation, monitoring, and smoke testing.

## Project structure
- [src/data_preprocessing.py](src/data_preprocessing.py): dataset download/preprocessing/splitting logic
- [src/train.py](src/train.py) and [src/evaluate.py](src/evaluate.py): training and evaluation workflow
- [src/model.py](src/model.py): CNN model definition
- [app/main.py](app/main.py) and [app/utils.py](app/utils.py): FastAPI service, prediction logic, and preprocessing utilities
- [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml): containerization and local deployment
- [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml): CI/CD workflow
- [k8s/deployment.yaml](k8s/deployment.yaml) and [k8s/service.yaml](k8s/service.yaml): Kubernetes deployment assets
- [monitoring/prometheus.yml](monitoring/prometheus.yml): monitoring config

## Setup
1. Create and activate a virtual environment
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies
   ```powershell
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```
3. Run tests
   ```powershell
   pytest tests -q
   ```

## Train and evaluate
Run the DVC pipeline if you want to reproduce training and evaluation:
```powershell
dvc repro
```
This generates the trained model artifact in [models/cnn_latest.pt](models/cnn_latest.pt) and evaluation artifacts in [artifacts](artifacts).

## Run the API locally
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API endpoints
- GET / returns a welcome message and docs link
- GET /health returns the service health status
- GET /metrics returns Prometheus-style metrics
- POST /predict accepts an image upload and returns the predicted class, confidence, and class probabilities

## Smoke test
```powershell
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

## Run with Docker
```powershell
docker compose up --build
```
The container exposes the same API on port 8000 and includes a health check for /health.

## Notes
- The repository is configured for local development, Docker-based deployment, and CI/CD-driven deployment workflows.
- The service is resilient to missing or incompatible model artifacts by falling back to an internal lightweight model during startup, though the trained artifact in [models/cnn_latest.pt](models/cnn_latest.pt) should be used for best accuracy.
- This README is intended to be the main submission document.

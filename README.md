# Cats vs Dogs MLOps Assignment 2

This repository contains a compact end-to-end MLOps solution for the Cats vs Dogs binary image classification use case. It covers data preprocessing, model training with experiment tracking, API deployment, Docker packaging, CI/CD automation, Kubernetes manifests, monitoring, and smoke testing.

## What is included
- Data preprocessing and splitting pipeline in [src/data_preprocessing.py](src/data_preprocessing.py)
- Baseline CNN training and evaluation in [src/train.py](src/train.py) and [src/evaluate.py](src/evaluate.py)
- FastAPI inference service with health, prediction, and metrics endpoints in [app/main.py](app/main.py)
- Docker and Docker Compose setup in [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml)
- GitHub Actions workflow in [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)
- Kubernetes manifests in [k8s/deployment.yaml](k8s/deployment.yaml) and [k8s/service.yaml](k8s/service.yaml)
- DVC pipeline in [dvc.yaml](dvc.yaml)
- Monitoring configuration in [monitoring/prometheus.yml](monitoring/prometheus.yml)

## Quick start
1. Create and activate a virtual environment
   - Windows PowerShell:
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
4. Run the DVC pipeline
   ```powershell
   dvc repro
   ```
5. Start the API locally
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
6. Run smoke tests
   ```powershell
   python scripts/smoke_test.py --url http://127.0.0.1:8000
   ```

## Submission notes
- The trained model artifact is available in [models/cnn_latest.pt](models/cnn_latest.pt).
- The README serves as the primary submission guide for setup and verification
- Repository reference: [RAJKUMAR27M/MLOPS_Assignment2](https://github.com/RAJKUMAR27M/MLOPS_Assignment2)
- Local verification was completed with the test suite using `pytest tests -q`.

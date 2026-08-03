# Assignment 2 Final Submission Report

## Student Submission Scope
This report consolidates all required implementation details for the MLOps Assignment 2 (Cats vs Dogs binary image classification), excluding only the explanation video as requested.

## Problem Statement and Dataset

### Use Case
A pet adoption platform needs a binary image classifier to predict whether an uploaded image is a cat or a dog.

### Dataset Requirement
The project is designed for the Kaggle Dogs vs Cats dataset.

### Implemented Dataset Handling
1. Dataset acquisition is implemented in src/data_preprocessing.py.
2. The pipeline attempts Kaggle download first when KAGGLE_USERNAME and KAGGLE_KEY are configured.
3. If credentials are unavailable, a synthetic fallback dataset is generated so the pipeline remains runnable in local/offline/CI environments.
4. All images are converted to RGB and resized to 224x224.
5. Dataset split is implemented as train/validation/test = 80/10/10.
6. Data augmentation is applied to training data using random flip, random rotation, and color jitter.

## M1: Model Development and Experiment Tracking

### M1.1 Data and Code Versioning
Implemented items:
1. Source code and configuration are organized for Git versioning.
2. DVC pipeline is defined in dvc.yaml.
3. DVC stage lock metadata is generated in dvc.lock.
4. Preprocess, train, and evaluate stages are reproducible and dependency-tracked.

Evidence:
1. dvc.yaml
2. dvc.lock
3. .dvc/

### M1.2 Model Building
Implemented items:
1. Baseline CNN model implemented in src/model.py.
2. Additional logistic regression baseline included in src/model.py.
3. Training script in src/train.py serializes model as models/cnn_latest.pt.

Evidence:
1. src/model.py
2. src/train.py
3. models/cnn_latest.pt

### M1.3 Experiment Tracking
Implemented items:
1. MLflow integrated into training and evaluation scripts.
2. Logged parameters include epochs, batch size, model name, and learning rate.
3. Logged metrics include training and validation loss/accuracy and test metrics.
4. Logged artifacts include training loss curves and confusion matrix.

Evidence:
1. src/train.py
2. src/evaluate.py
3. artifacts/loss_curves.png
4. artifacts/confusion_matrix.png

## M2: Model Packaging and Containerization

### M2.1 Inference Service
Implemented items:
1. FastAPI service implemented in app/main.py.
2. Required endpoints:
   - GET /health
   - POST /predict
3. Predict endpoint accepts image file input and returns label with confidence/probabilities.
4. Input validation returns 400 for non-image uploads.

Evidence:
1. app/main.py
2. app/utils.py

### M2.2 Environment Specification
Implemented items:
1. Runtime dependencies pinned in requirements.txt.
2. Development/testing dependencies pinned in requirements-dev.txt.

Evidence:
1. requirements.txt
2. requirements-dev.txt

### M2.3 Containerization
Implemented items:
1. Multi-stage Docker build in Dockerfile.
2. Healthcheck configured in container.
3. Docker Compose service definition with model path and restart policy.

Evidence:
1. Dockerfile
2. docker-compose.yml

## M3: CI Pipeline for Build, Test, and Image Creation

### M3.1 Automated Testing
Implemented items:
1. Unit tests for preprocessing functions in tests/test_preprocessing.py.
2. Unit tests for inference/model utility behavior in tests/test_inference.py.
3. Test suite passes locally.

Evidence:
1. tests/test_preprocessing.py
2. tests/test_inference.py

### M3.2 CI Setup
Implemented items:
1. GitHub Actions workflow on push and pull_request to main.
2. Workflow installs dependencies and runs pytest.
3. Docker image build is triggered after tests pass.

Evidence:
1. .github/workflows/ci-cd.yml

### M3.3 Artifact Publishing
Implemented items:
1. Image is pushed to GitHub Container Registry (GHCR).
2. Image tags include commit SHA and latest.

Evidence:
1. .github/workflows/ci-cd.yml

## M4: CD Pipeline and Deployment

### M4.1 Deployment Target and Manifests
Implemented items:
1. Primary deployment target: Docker Compose.
2. Additional Kubernetes manifests included.

Evidence:
1. docker-compose.yml
2. k8s/deployment.yaml
3. k8s/service.yaml

### M4.2 CD / GitOps Flow
Implemented items:
1. Deploy job configured for main branch changes.
2. Deployment job pulls latest image and updates running service.
3. Deploy job is defined for self-hosted runner execution.

Evidence:
1. .github/workflows/ci-cd.yml

### M4.3 Smoke Tests / Health Check
Implemented items:
1. Post-deployment smoke test script implemented.
2. Smoke test validates health, prediction, and metrics endpoints.
3. CI/CD workflow fails deployment if smoke test fails.

Evidence:
1. scripts/smoke_test.py
2. .github/workflows/ci-cd.yml

## M5: Monitoring, Logs, and Post-Deployment Tracking

### M5.1 Basic Monitoring and Logging
Implemented items:
1. Request logging middleware logs method, path, status code, and latency.
2. Prometheus-compatible /metrics endpoint implemented.
3. Counters and latency histograms are exposed.

Evidence:
1. app/main.py
2. monitoring/prometheus.yml

### M5.2 Post-Deployment Model Performance Tracking
Implemented items:
1. Simulated request script implemented in scripts/simulate_requests.py.
2. Script supports labeled test set usage when available.
3. Script outputs detailed run records and summary for analysis.

Evidence:
1. scripts/simulate_requests.py
2. artifacts/post_deploy_requests.csv (generated when simulation is run)
3. artifacts/post_deploy_summary.json (generated when simulation is run)

## Implementation Output Images

### 1. Training Loss Curve
![Training Loss Curve](artifacts/loss_curves.png)

### 2. Confusion Matrix
![Confusion Matrix](artifacts/confusion_matrix.png)

## End-to-End Run Instructions (From Scratch to Endpoint)

This section provides complete, sequential steps to run the project from a fresh machine checkout until the inference endpoint is working.

### Step 0: Prerequisites

Install the following tools:
1. Python 3.11 or later (3.14 is also supported in this project).
2. Git.
3. DVC.
4. Docker Desktop (optional, for container run path).
5. Kaggle CLI (optional, if using real Kaggle dataset instead of fallback data).

### Step 1: Open Project Folder

```bash
cd MLOPS-2
```

### Step 2: Create and Activate Virtual Environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Step 4: Optional Kaggle Credentials Setup

If you want real Kaggle data, set credentials before running pipeline.

Windows PowerShell:

```bash
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_key"
```

If this is not set, the pipeline automatically uses synthetic fallback data and still runs end-to-end.

### Step 5: Run Full Data + Train + Evaluate Pipeline

```bash
dvc repro
```

Expected outputs:
1. models/cnn_latest.pt
2. artifacts/loss_curves.png
3. artifacts/confusion_matrix.png

### Step 6: Run Unit Tests

```bash
pytest tests -q
```

### Step 7: Start Inference API (Local)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Keep this terminal running.

### Step 8: Verify Endpoints (New Terminal)

Health check:

```bash
curl http://localhost:8000/health
```

Prediction endpoint:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@sample.jpg"
```

Metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

Swagger docs UI:
1. Open http://localhost:8000/docs in browser.

### Step 9: Run Smoke Test

```bash
python scripts/smoke_test.py --url http://localhost:8000
```

### Step 10: Run Post-Deployment Simulation Evidence (Optional but Recommended)

```bash
python scripts/simulate_requests.py --url http://localhost:8000 --num-requests 20
```

Expected optional outputs:
1. artifacts/post_deploy_requests.csv
2. artifacts/post_deploy_summary.json

### Optional Container Path (Instead of Local Uvicorn)

Build and run via Docker:

```bash
docker build -t cats-dogs-api:latest .
docker compose up -d
```

Then test the same endpoints at http://localhost:8000.

## Deliverables Checklist

1. Source code included
2. DVC pipeline files included
3. Trained model artifact included
4. CI/CD configuration included
5. Docker and deployment manifests included
6. Tests included
7. Detailed explanation document included (this file)
8. Output images included
9. Video excluded intentionally (to be added later)

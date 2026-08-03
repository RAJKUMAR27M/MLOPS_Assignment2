# Cats vs Dogs MLOps Assignment 2

This workspace contains a compact end-to-end MLOps solution for the Cats vs Dogs binary image classification use case.

## What is included
- Data preprocessing and splitting pipeline
- Baseline CNN training with MLflow experiment tracking
- FastAPI inference service with health, prediction, and metrics endpoints
- Docker containerization and Docker Compose deployment
- GitHub Actions CI/CD workflow with tests and image build/push
- Kubernetes manifests for deployment and service
- Prometheus/Grafana monitoring setup
- Smoke test scripts for deployment verification

## Quick start
1. Install dependencies
   - `pip install -r requirements-dev.txt`
2. Run tests
   - `pytest tests -q`
3. Start the API locally
   - `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
4. Run smoke tests
   - `python scripts/smoke_test.py --url http://127.0.0.1:8000`

## Submission artifacts
- Final report: [FINAL_SUBMISSION_REPORT.md](FINAL_SUBMISSION_REPORT.md)
- DVC pipeline: [dvc.yaml](dvc.yaml)
- CI/CD workflow: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)

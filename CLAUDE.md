# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOps pipeline for fraud detection in digital payments. Trains XGBoost and Random Forest classifiers with a conservative decision threshold (0.10) to maximize fraud recall.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Commands

```bash
# 1. Start MLflow server (Docker)
docker-compose up mlflow -d

# 2. Run full training pipeline (logs models to MLflow at localhost:5000)
dvc repro
# or: python main.py

# 3. Start the prediction API (Docker)
docker-compose up api

# View pipeline metrics
dvc metrics show

# Clean generated artifacts
make clean
```

API endpoints once running:
- `GET  http://localhost:8000/health`
- `POST http://localhost:8000/predict`  (JSON body with transaction features)

## Architecture

Two-stage DVC pipeline defined in `dvc.yaml`, with hyperparameters centralized in `params.yaml`.

**Stage 1 — Preprocess** (`pipeline/preprocess.py`):
- Loads `data/raw/Digital_Payment_Fraud_Detection_Dataset.csv`
- `DataPreprocessor` cleans data and engineers features (amount ratios, night transaction flag, behavior risk score, new account flag, international high-risk flag)
- Builds a `ColumnTransformer` pipeline, performs 80/20 stratified split
- Outputs: `data/processed/X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy`, `preprocessor.joblib`

**Stage 2 — Train** (`pipeline/train.py`):
- `ModelTrainer` fits XGBoost and Random Forest on the processed sparse matrices
- Both models use class-weight balancing; fraud classification threshold is 0.10
- Logs parameters, metrics (ROC-AUC, PR-AUC, Accuracy), and model artifacts to MLflow
- Outputs: `models/xgboost_model.joblib`, `models/random_forest_model.joblib`, `metrics/scores.json`

**API** (`api/app.py`):
- FastAPI service (port 8000) built from project root context so it can import `src/`
- On startup loads the champion model via `mlflow.sklearn.load_model("models:/fraud_detection_champion@champion")` and downloads `preprocessor.joblib` from MLflow artifacts — no local file paths
- `POST /predict` applies `DataPreprocessor.clean_data()` + `create_features()` then `preprocessor.transform()` before calling the model

**Supporting code:**
- `src/config.py` — loads `.env` variables (`MODEL_NAME`, `ENV`, `DEBUG`, `MLFLOW_TRACKING_URI`)
- `src/preprocessor.py` / `src/model_trainer.py` — core logic consumed by the pipeline scripts
- `notebooks/ProjetoMLops_refeito.ipynb` — exploratory reference notebook

**Docker** (`docker-compose.yml`): two services — `mlflow` (file-based store + `--serve-artifacts`, port 5000) and `api` (FastAPI, port 8000). Pipeline runs locally against the MLflow container.

## Key Design Decisions

- **Threshold 0.10**: intentionally low to prioritize catching fraud (high recall) at the cost of more false positives.
- **Sparse matrix format (NPZ)**: used for processed features to handle high-dimensional one-hot encoded data efficiently.
- **DVC tracks** `metrics/scores.json` and `data/processed/` so pipeline runs are reproducible and comparable across commits.
- All hyperparameters live in `params.yaml`; changing them there and re-running `dvc repro` is the correct workflow.
- **Champion model**: after training, `ModelTrainer` automatically registers the best model (by ROC-AUC) to the MLflow Model Registry as `fraud_detection_champion` with the alias `@champion`. The API loads exclusively from this alias.
- **Preprocessor artifact**: `preprocessor.joblib` is logged as an MLflow artifact inside each training run. The API downloads it from MLflow at startup (never from local disk).

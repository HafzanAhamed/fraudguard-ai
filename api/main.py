from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drift_utils import generate_drift_report
from explainability import explain_transaction
from fraud_utils import RAW_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "model_metrics.json"
DRIFT_BASELINE_PATH = BASE_DIR / "model" / "drift_baseline.json"
DRIFT_REPORT_PATH = BASE_DIR / "model" / "drift_report.json"


class TransactionRequest(BaseModel):
    transaction_amount: float = Field(..., ge=0)
    transaction_type: str
    account_age_days: int = Field(..., ge=0)
    previous_failed_transactions: int = Field(..., ge=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    location_risk_score: float = Field(..., ge=0, le=100)
    device_risk_score: float = Field(..., ge=0, le=100)
    is_international: int = Field(..., ge=0, le=1)
    average_daily_transaction_amount: float = Field(..., ge=0)


class DriftBatchRequest(BaseModel):
    records: list[TransactionRequest]


app = FastAPI(
    title="FraudGuard AI API",
    description="FastAPI service for bank transaction fraud detection, explainability, and drift monitoring.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model bundle not found. Run `python train_model.py` first.")

    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict):
        return loaded
    return {
        "model": loaded,
        "threshold": 0.5,
        "model_name": "Saved Model",
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
    }


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@lru_cache(maxsize=1)
def load_drift_baseline() -> dict:
    if not DRIFT_BASELINE_PATH.exists():
        raise FileNotFoundError("Drift baseline not found. Run `python train_model.py` first.")
    with DRIFT_BASELINE_PATH.open("r", encoding="utf-8") as baseline_file:
        return json.load(baseline_file)


def _predict_from_payload(payload: TransactionRequest) -> dict:
    model_bundle = load_model_bundle()
    pipeline = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))
    raw_frame = pd.DataFrame([payload.model_dump()], columns=RAW_FEATURE_COLUMNS)

    from fraud_utils import engineer_features  # Local import keeps startup light.

    prediction_frame = engineer_features(raw_frame)
    probability = float(pipeline.predict_proba(prediction_frame)[0][1])
    prediction = int(probability >= threshold)
    shap_summary = explain_transaction(model_bundle, raw_frame, top_n=6)

    return {
        "model_name": model_bundle.get("model_name", "Saved Model"),
        "decision_threshold": round(threshold, 4),
        "prediction": prediction,
        "prediction_label": "Fraudulent" if prediction == 1 else "Legitimate",
        "fraud_probability": round(probability, 4),
        "shap_base_value": round(float(shap_summary["base_value"]), 4),
        "top_shap_contributions": [
            {
                "feature": row["feature"],
                "shap_value": round(float(row["shap_value"]), 4),
                "impact": row["impact"],
            }
            for _, row in shap_summary["top_contributions"].iterrows()
        ],
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> dict:
    return load_metrics()


@app.get("/drift/latest")
def get_latest_drift_report() -> dict:
    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No drift report found.")
    with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


@app.post("/drift/evaluate")
def evaluate_drift(batch: DriftBatchRequest) -> dict:
    if not batch.records:
        raise HTTPException(status_code=400, detail="At least one record is required.")
    current_frame = pd.DataFrame([record.model_dump() for record in batch.records])
    return generate_drift_report(current_frame, load_drift_baseline())


@app.post("/predict")
def predict_transaction(payload: TransactionRequest) -> dict:
    try:
        return _predict_from_payload(payload)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

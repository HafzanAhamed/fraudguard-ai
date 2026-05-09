from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from drift_utils import create_drift_baseline, generate_drift_report, save_json
from fraud_utils import (
    CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    RAW_FEATURE_COLUMNS,
    engineer_features,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - graceful fallback if dependency is unavailable.
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = DATA_DIR / "transactions.csv"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
DRIFT_BASELINE_PATH = MODEL_DIR / "drift_baseline.json"
DRIFT_REPORT_PATH = MODEL_DIR / "drift_report.json"

TARGET_COLUMN = "fraud"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(
    output_path: Path,
    rows: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic fraud dataset when no real data is present."""
    rng = np.random.default_rng(random_state)

    transaction_types = np.array(
        ["payment", "transfer", "withdrawal", "purchase", "deposit"]
    )
    transaction_type = rng.choice(
        transaction_types,
        size=rows,
        p=[0.33, 0.22, 0.15, 0.22, 0.08],
    )

    average_daily_transaction_amount = np.clip(
        rng.gamma(shape=5.0, scale=55.0, size=rows),
        20,
        4500,
    )
    account_age_days = np.clip(rng.gamma(shape=4.8, scale=210, size=rows), 5, 3650)
    previous_failed_transactions = np.clip(rng.poisson(0.6, size=rows), 0, 10)
    transaction_hour = rng.integers(0, 24, size=rows)

    location_risk_score = np.clip(
        rng.beta(2.0, 4.0, size=rows) * 100
        + (transaction_hour <= 5) * rng.uniform(5, 18, size=rows),
        0,
        100,
    )
    device_risk_score = np.clip(
        rng.beta(2.3, 3.8, size=rows) * 100
        + previous_failed_transactions * rng.uniform(1.5, 4.0, size=rows),
        0,
        100,
    )
    is_international = rng.binomial(1, 0.18, size=rows)

    type_multiplier_map = {
        "payment": 1.0,
        "transfer": 1.55,
        "withdrawal": 1.25,
        "purchase": 0.95,
        "deposit": 0.75,
    }
    type_multiplier = np.vectorize(type_multiplier_map.get)(transaction_type)
    night_multiplier = np.where(
        (transaction_hour >= 22) | (transaction_hour <= 5),
        rng.uniform(1.5, 2.6, size=rows),
        1.0,
    )
    international_multiplier = np.where(
        is_international == 1,
        rng.uniform(1.3, 2.8, size=rows),
        1.0,
    )
    risk_multiplier = 1 + (location_risk_score + device_risk_score) / 220

    transaction_amount = (
        average_daily_transaction_amount
        * type_multiplier
        * night_multiplier
        * international_multiplier
        * risk_multiplier
        * rng.lognormal(mean=0.1, sigma=0.55, size=rows)
    )
    transaction_amount = np.clip(transaction_amount, 8, 15000).round(2)

    high_amount_flag = transaction_amount > (average_daily_transaction_amount * 2.6)
    late_night_flag = (transaction_hour >= 23) | (transaction_hour <= 4)
    new_account_flag = account_age_days < 120
    high_location_flag = location_risk_score > 70
    high_device_flag = device_risk_score > 65

    linear_score = (
        -6.2
        + 0.00115 * transaction_amount
        + 0.028 * location_risk_score
        + 0.027 * device_risk_score
        + 0.38 * previous_failed_transactions
        + 0.95 * is_international
        + 0.75 * high_amount_flag.astype(int)
        + 0.72 * late_night_flag.astype(int)
        + 0.82 * high_location_flag.astype(int)
        + 0.78 * high_device_flag.astype(int)
        + 0.70 * new_account_flag.astype(int)
        - 0.0018 * account_age_days
        - 0.0010 * average_daily_transaction_amount
    )

    transaction_type_risk = {
        "transfer": 0.78,
        "withdrawal": 0.43,
        "payment": 0.18,
        "purchase": 0.07,
        "deposit": -0.45,
    }
    linear_score += np.vectorize(transaction_type_risk.get)(transaction_type)
    linear_score += rng.normal(0, 0.62, size=rows)

    fraud_probability = 1 / (1 + np.exp(-linear_score))
    fraud = rng.binomial(1, fraud_probability)

    dataset = pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "transaction_type": transaction_type,
            "account_age_days": account_age_days.round(0).astype(int),
            "previous_failed_transactions": previous_failed_transactions.astype(int),
            "transaction_hour": transaction_hour.astype(int),
            "location_risk_score": location_risk_score.round(2),
            "device_risk_score": device_risk_score.round(2),
            "is_international": is_international.astype(int),
            "average_daily_transaction_amount": average_daily_transaction_amount.round(2),
            "fraud": fraud.astype(int),
        }
    )

    dataset.to_csv(output_path, index=False)
    return dataset


def load_dataset(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        return pd.read_csv(data_path)
    return generate_synthetic_dataset(data_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_pipeline, MODEL_NUMERIC_FEATURES),
        ]
    )


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    return candidates


def find_best_threshold(
    probabilities: np.ndarray,
    y_true: pd.Series,
) -> tuple[float, float]:
    """Tune the classification threshold on validation scores using F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-9
    )
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_true, predictions),
    }


def extract_feature_importance(
    trained_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    estimator = trained_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def summarize_cv_results(cv_results: dict[str, np.ndarray]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in cv_results.items():
        metric_name = key.replace("test_", "")
        summary[f"{metric_name}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{metric_name}_std"] = round(float(np.std(values)), 4)
    return summary


def train_and_select_model(dataset: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    x_raw = dataset[RAW_FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    x = engineer_features(x_raw)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    best_model_name = ""
    best_threshold = 0.5
    best_validation_metrics: dict[str, Any] | None = None
    best_estimator: Any | None = None
    comparison_rows: list[dict[str, Any]] = []

    for model_name, estimator in get_model_candidates().items():
        if model_name == "XGBoost":
            fraud_count = max(int((y_train == 1).sum()), 1)
            legitimate_count = max(int((y_train == 0).sum()), 1)
            estimator.set_params(scale_pos_weight=legitimate_count / fraud_count)
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)

        validation_probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold, optimized_f1 = find_best_threshold(validation_probabilities, y_val)
        validation_metrics = evaluate_predictions(y_val, validation_probabilities, threshold)

        comparison_rows.append(
            {
                "model": model_name,
                "threshold": round(threshold, 4),
                "accuracy": round(float(validation_metrics["accuracy"]), 4),
                "precision": round(float(validation_metrics["precision"]), 4),
                "recall": round(float(validation_metrics["recall"]), 4),
                "f1_score": round(float(validation_metrics["f1_score"]), 4),
                "roc_auc": round(float(validation_metrics["roc_auc"]), 4),
                "pr_auc": round(float(validation_metrics["pr_auc"]), 4),
                "optimized_validation_f1": round(optimized_f1, 4),
            }
        )

        if (
            best_validation_metrics is None
            or validation_metrics["f1_score"] > best_validation_metrics["f1_score"]
        ):
            best_model_name = model_name
            best_threshold = threshold
            best_validation_metrics = validation_metrics
            best_estimator = estimator

    assert best_estimator is not None and best_validation_metrics is not None

    final_estimator = clone(best_estimator)
    if best_model_name == "XGBoost":
        fraud_count = max(int((y_train_val == 1).sum()), 1)
        legitimate_count = max(int((y_train_val == 0).sum()), 1)
        final_estimator.set_params(scale_pos_weight=legitimate_count / fraud_count)

    final_pipeline = build_pipeline(final_estimator)
    final_pipeline.fit(x_train_val, y_train_val)
    test_probabilities = final_pipeline.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probabilities, best_threshold)

    cv_results = cross_validate(
        build_pipeline(clone(best_estimator)),
        x_train_val,
        y_train_val,
        cv=5,
        scoring={
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
    )

    model_bundle = {
        "model": final_pipeline,
        "threshold": float(best_threshold),
        "model_name": best_model_name,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "background_sample": x_train_val[RAW_FEATURE_COLUMNS].sample(
            n=min(200, len(x_train_val)),
            random_state=42,
        ),
    }

    metrics_payload = {
        "best_model": best_model_name,
        "decision_threshold": round(float(best_threshold), 4),
        "dataset_rows": int(len(dataset)),
        "fraud_rate": round(float(dataset[TARGET_COLUMN].mean()), 4),
        "split_summary": {
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
        },
        "comparison": comparison_rows,
        "validation_metrics": {
            "accuracy": round(float(best_validation_metrics["accuracy"]), 4),
            "precision": round(float(best_validation_metrics["precision"]), 4),
            "recall": round(float(best_validation_metrics["recall"]), 4),
            "f1_score": round(float(best_validation_metrics["f1_score"]), 4),
            "roc_auc": round(float(best_validation_metrics["roc_auc"]), 4),
            "pr_auc": round(float(best_validation_metrics["pr_auc"]), 4),
        },
        "best_model_metrics": {
            "accuracy": round(float(test_metrics["accuracy"]), 4),
            "precision": round(float(test_metrics["precision"]), 4),
            "recall": round(float(test_metrics["recall"]), 4),
            "f1_score": round(float(test_metrics["f1_score"]), 4),
            "roc_auc": round(float(test_metrics["roc_auc"]), 4),
            "pr_auc": round(float(test_metrics["pr_auc"]), 4),
        },
        "cross_validation": summarize_cv_results(cv_results),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
    }
    return model_bundle, metrics_payload


def main() -> None:
    ensure_directories()

    print("Loading dataset...")
    dataset = load_dataset(DATA_PATH)
    print(f"Dataset ready: {DATA_PATH} ({len(dataset)} rows)")

    print("\nTraining candidate models with feature engineering and threshold tuning...")
    model_bundle, metrics_payload = train_and_select_model(dataset)

    joblib.dump(model_bundle, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    trained_pipeline = model_bundle["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    importance_frame = extract_feature_importance(trained_pipeline, preprocessor)
    if not importance_frame.empty:
        importance_frame.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    drift_baseline = create_drift_baseline(dataset[RAW_FEATURE_COLUMNS])
    drift_report = generate_drift_report(dataset[RAW_FEATURE_COLUMNS], drift_baseline)
    save_json(DRIFT_BASELINE_PATH, drift_baseline)
    save_json(DRIFT_REPORT_PATH, drift_report)

    print(f"\nBest model selected: {metrics_payload['best_model']}")
    print(f"Decision threshold: {metrics_payload['decision_threshold']:.4f}")
    print("Test metrics:")
    for metric_name, metric_value in metrics_payload["best_model_metrics"].items():
        print(f"  {metric_name:>10}: {metric_value:.4f}")

    print("\nClassification report:")
    print(metrics_payload["classification_report"])

    print(f"Saved trained model bundle to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    if FEATURE_IMPORTANCE_PATH.exists():
        print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved drift baseline to: {DRIFT_BASELINE_PATH}")
    print(f"Saved drift report to: {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    main()

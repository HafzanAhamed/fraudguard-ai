from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_utils import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES


EPSILON = 1e-6
WARNING_PSI = 0.1
ALERT_PSI = 0.2


def _safe_proportions(values: np.ndarray) -> list[float]:
    adjusted = np.clip(values.astype(float), EPSILON, None)
    normalized = adjusted / adjusted.sum()
    return normalized.tolist()


def create_drift_baseline(dataset: pd.DataFrame) -> dict[str, Any]:
    baseline: dict[str, Any] = {
        "thresholds": {"warning_psi": WARNING_PSI, "alert_psi": ALERT_PSI},
        "numeric_features": {},
        "categorical_features": {},
    }

    for feature in BASE_NUMERIC_FEATURES:
        values = dataset[feature].astype(float).to_numpy()
        quantiles = np.linspace(0, 1, 11)
        bin_edges = np.unique(np.quantile(values, quantiles))
        if len(bin_edges) < 3:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            bin_edges = np.array([minimum - 1, minimum, maximum + 1], dtype=float)
        counts, _ = np.histogram(values, bins=bin_edges)
        baseline["numeric_features"][feature] = {
            "bin_edges": [float(edge) for edge in bin_edges],
            "expected_proportions": _safe_proportions(counts),
            "mean": float(np.mean(values)),
            "std": float(np.std(values) + EPSILON),
        }

    for feature in CATEGORICAL_FEATURES:
        proportions = dataset[feature].value_counts(normalize=True).sort_index()
        baseline["categorical_features"][feature] = {
            "categories": proportions.index.astype(str).tolist(),
            "expected_proportions": _safe_proportions(proportions.to_numpy()),
        }

    return baseline


def calculate_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_safe = np.clip(expected.astype(float), EPSILON, None)
    actual_safe = np.clip(actual.astype(float), EPSILON, None)
    return float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))


def calculate_numeric_drift(feature: str, current_frame: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, Any]:
    metadata = baseline["numeric_features"][feature]
    bin_edges = np.array(metadata["bin_edges"], dtype=float)
    expected = np.array(metadata["expected_proportions"], dtype=float)
    current_values = current_frame[feature].astype(float).to_numpy()
    counts, _ = np.histogram(current_values, bins=bin_edges)
    actual = np.array(_safe_proportions(counts))
    psi = calculate_psi(expected, actual)
    mean = float(np.mean(current_values))
    std_shift = abs(mean - float(metadata["mean"])) / float(metadata["std"])

    return {
        "feature": feature,
        "psi": round(psi, 4),
        "mean": round(mean, 4),
        "baseline_mean": round(float(metadata["mean"]), 4),
        "std_shift": round(std_shift, 4),
        "status": drift_status(psi),
    }


def calculate_categorical_drift(feature: str, current_frame: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, Any]:
    metadata = baseline["categorical_features"][feature]
    categories = metadata["categories"]
    expected = np.array(metadata["expected_proportions"], dtype=float)
    actual_series = (
        current_frame[feature].astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    )
    actual = np.array(_safe_proportions(actual_series.to_numpy()))
    psi = calculate_psi(expected, actual)

    return {
        "feature": feature,
        "psi": round(psi, 4),
        "status": drift_status(psi),
        "distribution": {category: round(float(value), 4) for category, value in actual_series.items()},
    }


def drift_status(psi: float) -> str:
    if psi >= ALERT_PSI:
        return "alert"
    if psi >= WARNING_PSI:
        return "warning"
    return "stable"


def generate_drift_report(current_frame: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, Any]:
    numeric_results = [
        calculate_numeric_drift(feature, current_frame, baseline)
        for feature in BASE_NUMERIC_FEATURES
    ]
    categorical_results = [
        calculate_categorical_drift(feature, current_frame, baseline)
        for feature in CATEGORICAL_FEATURES
    ]

    all_statuses = [item["status"] for item in numeric_results + categorical_results]
    overall_status = "alert" if "alert" in all_statuses else "warning" if "warning" in all_statuses else "stable"

    return {
        "overall_status": overall_status,
        "thresholds": baseline["thresholds"],
        "numeric_feature_drift": numeric_results,
        "categorical_feature_drift": categorical_results,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


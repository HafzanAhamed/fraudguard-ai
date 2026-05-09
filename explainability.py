from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from fraud_utils import engineer_features


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _select_shap_values(explanation: shap.Explanation) -> tuple[np.ndarray, float]:
    values = explanation.values
    base_values = explanation.base_values

    if values.ndim == 3:
        selected_values = values[0, :, 1]
        selected_base = float(np.asarray(base_values)[0, 1])
    else:
        selected_values = values[0]
        base_array = np.asarray(base_values)
        selected_base = float(base_array[0] if base_array.ndim else base_array)

    return np.asarray(selected_values, dtype=float), selected_base


def build_shap_explainer(model_bundle: dict[str, Any]) -> tuple[Any, list[str]]:
    pipeline = model_bundle["model"]
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    background_raw = model_bundle.get("background_sample")
    if background_raw is None or len(background_raw) == 0:
        raise ValueError("Model bundle does not contain a SHAP background sample.")

    background_engineered = engineer_features(background_raw.copy())
    transformed_background = _to_dense(preprocessor.transform(background_engineered))
    feature_names = preprocessor.get_feature_names_out().tolist()

    if hasattr(estimator, "feature_importances_") or estimator.__class__.__name__.startswith(
        "XGB"
    ):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, transformed_background)
    else:
        explainer = shap.Explainer(estimator, transformed_background)

    return explainer, feature_names


def explain_transaction(
    model_bundle: dict[str, Any],
    transaction_frame: pd.DataFrame,
    top_n: int = 8,
) -> dict[str, Any]:
    pipeline = model_bundle["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    engineered_frame = engineer_features(transaction_frame.copy())
    transformed_input = _to_dense(preprocessor.transform(engineered_frame))

    explainer, feature_names = build_shap_explainer(model_bundle)
    explanation = explainer(transformed_input)
    shap_values, base_value = _select_shap_values(explanation)

    contribution_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values,
        }
    )
    contribution_frame["abs_shap_value"] = contribution_frame["shap_value"].abs()
    contribution_frame["impact"] = contribution_frame["shap_value"].apply(
        lambda value: "Increase risk" if value >= 0 else "Reduce risk"
    )
    contribution_frame = contribution_frame.sort_values(
        "abs_shap_value", ascending=False
    ).reset_index(drop=True)

    return {
        "base_value": base_value,
        "top_contributions": contribution_frame.head(top_n).copy(),
        "all_contributions": contribution_frame,
    }


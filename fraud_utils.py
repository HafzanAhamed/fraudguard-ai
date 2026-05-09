from __future__ import annotations

import pandas as pd


RAW_FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_type",
    "account_age_days",
    "previous_failed_transactions",
    "transaction_hour",
    "location_risk_score",
    "device_risk_score",
    "is_international",
    "average_daily_transaction_amount",
]

CATEGORICAL_FEATURES = ["transaction_type"]
BASE_NUMERIC_FEATURES = [
    "transaction_amount",
    "account_age_days",
    "previous_failed_transactions",
    "transaction_hour",
    "location_risk_score",
    "device_risk_score",
    "is_international",
    "average_daily_transaction_amount",
]
ENGINEERED_NUMERIC_FEATURES = [
    "amount_to_average_ratio",
    "amount_deviation",
    "combined_risk_score",
    "failed_transaction_velocity",
    "is_night_transaction",
    "high_risk_interaction",
]
MODEL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create derived fraud signals from the raw transaction fields."""
    enriched = frame.copy()
    average_amount = enriched["average_daily_transaction_amount"].clip(lower=1)

    enriched["amount_to_average_ratio"] = (
        enriched["transaction_amount"] / average_amount
    ).round(4)
    enriched["amount_deviation"] = (
        enriched["transaction_amount"] - enriched["average_daily_transaction_amount"]
    ).round(2)
    enriched["combined_risk_score"] = (
        0.45 * enriched["location_risk_score"]
        + 0.35 * enriched["device_risk_score"]
        + 20 * enriched["is_international"]
    ).round(2)
    enriched["failed_transaction_velocity"] = (
        enriched["previous_failed_transactions"] / (enriched["account_age_days"] + 30)
        * 365
    ).round(4)
    enriched["is_night_transaction"] = (
        (enriched["transaction_hour"] >= 22)
        | (enriched["transaction_hour"] <= 5)
    ).astype(int)
    enriched["high_risk_interaction"] = (
        (
            (enriched["location_risk_score"] >= 70)
            & (enriched["device_risk_score"] >= 70)
        )
        | (
            (enriched["is_international"] == 1)
            & (enriched["amount_to_average_ratio"] >= 2.5)
        )
    ).astype(int)
    return enriched


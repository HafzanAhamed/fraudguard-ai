from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from explainability import explain_transaction
from fraud_utils import RAW_FEATURE_COLUMNS, engineer_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "model" / "feature_importance.csv"
DRIFT_REPORT_PATH = BASE_DIR / "model" / "drift_report.json"

TRANSACTION_TYPES = ["payment", "transfer", "withdrawal", "purchase", "deposit"]
SUSPICIOUS_SAMPLE = {
    "transaction_amount": 7800.0,
    "transaction_type": "transfer",
    "account_age_days": 18,
    "previous_failed_transactions": 4,
    "transaction_hour": 1,
    "location_risk_score": 89.0,
    "device_risk_score": 84.0,
    "is_international": 1,
    "average_daily_transaction_amount": 320.0,
}


def load_model_bundle() -> tuple[object, float, str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` first."
        )

    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict) and "model" in loaded:
        return (
            loaded["model"],
            float(loaded.get("threshold", 0.5)),
            str(loaded.get("model_name", "Saved Model")),
        )

    return loaded, 0.5, "Saved Model"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python train_model.py` to generate it."
        )
    return pd.read_csv(DATA_PATH)


def load_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def load_drift_report() -> dict:
    if not DRIFT_REPORT_PATH.exists():
        return {}
    with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def risk_level_from_probability(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


def explain_risk_factors(row: dict, probability: float) -> str:
    reasons: list[str] = []
    ratio = row["transaction_amount"] / max(row["average_daily_transaction_amount"], 1)

    if ratio >= 2.5:
        reasons.append("the amount is far above the customer's normal transaction pattern")
    if row["transaction_hour"] >= 23 or row["transaction_hour"] <= 4:
        reasons.append("the transaction happened during high-risk overnight hours")
    if row["location_risk_score"] >= 70:
        reasons.append("the transaction location has a high risk score")
    if row["device_risk_score"] >= 70:
        reasons.append("the device profile appears risky")
    if row["previous_failed_transactions"] >= 3:
        reasons.append("multiple failed transactions were recorded before this attempt")
    if row["account_age_days"] <= 90:
        reasons.append("the account is relatively new")
    if int(row["is_international"]) == 1:
        reasons.append("the transaction is international")
    if row["location_risk_score"] >= 70 and row["device_risk_score"] >= 70:
        reasons.append("both location and device signals are elevated at the same time")

    if not reasons:
        return (
            f"The model assigned a {probability:.1%} fraud probability with only mild risk signals. "
            "This currently looks like a lower-risk transaction."
        )

    prefix = f"The model assigned a {probability:.1%} fraud probability because "
    if len(reasons) == 1:
        return prefix + reasons[0] + "."
    return prefix + ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."


def build_prediction_frame(input_data: dict) -> pd.DataFrame:
    raw_frame = pd.DataFrame([input_data], columns=RAW_FEATURE_COLUMNS)
    return engineer_features(raw_frame)


def add_prediction_columns(dataset: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    frame = engineer_features(dataset[RAW_FEATURE_COLUMNS])
    probabilities = model.predict_proba(frame)[:, 1]

    enriched = dataset.copy()
    enriched["fraud_probability"] = probabilities
    enriched["predicted_fraud"] = (enriched["fraud_probability"] >= threshold).astype(int)
    enriched["risk_level"] = enriched["fraud_probability"].apply(risk_level_from_probability)
    return enriched


def initialise_sidebar_state() -> None:
    defaults = {
        "transaction_amount": 1200.0,
        "transaction_type": "payment",
        "account_age_days": 420,
        "previous_failed_transactions": 0,
        "transaction_hour": 14,
        "location_risk_score": 28.0,
        "device_risk_score": 24.0,
        "is_international": 0,
        "is_international_checkbox": False,
        "average_daily_transaction_amount": 340.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_suspicious_sample() -> None:
    for key, value in SUSPICIOUS_SAMPLE.items():
        st.session_state[key] = value
    st.session_state["is_international_checkbox"] = bool(
        SUSPICIOUS_SAMPLE["is_international"]
    )


def render_probability_gauge(probability: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0b3954"},
                "steps": [
                    {"range": [0, 45], "color": "#cdeee6"},
                    {"range": [45, 80], "color": "#f8d9a7"},
                    {"range": [80, 100], "color": "#f6b0ad"},
                ],
            },
        )
    )


def main() -> None:
    st.set_page_config(page_title="FraudGuard AI", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f8fb 0%, #edf3f8 100%);
        }
        [data-testid="stMetricValue"] {
            color: #0b3954;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, threshold, saved_model_name = load_model_bundle()
        dataset = load_dataset()
        metrics = load_metrics()
        feature_importance = load_feature_importance()
        drift_report = load_drift_report()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Unable to load the app assets: {error}")
        st.stop()

    initialise_sidebar_state()

    st.title("FraudGuard AI")
    st.caption("Advanced bank transaction fraud detection dashboard")

    st.sidebar.header("Transaction Input")
    if st.sidebar.button("Load Suspicious Sample"):
        apply_suspicious_sample()

    transaction_input = {
        "transaction_amount": st.sidebar.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(st.session_state["transaction_amount"]),
            step=50.0,
            key="transaction_amount",
        ),
        "transaction_type": st.sidebar.selectbox(
            "Transaction Type",
            TRANSACTION_TYPES,
            index=TRANSACTION_TYPES.index(st.session_state["transaction_type"]),
            key="transaction_type",
        ),
        "account_age_days": st.sidebar.number_input(
            "Account Age (days)",
            min_value=0,
            value=int(st.session_state["account_age_days"]),
            step=10,
            key="account_age_days",
        ),
        "previous_failed_transactions": st.sidebar.number_input(
            "Previous Failed Transactions",
            min_value=0,
            value=int(st.session_state["previous_failed_transactions"]),
            step=1,
            key="previous_failed_transactions",
        ),
        "transaction_hour": st.sidebar.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=int(st.session_state["transaction_hour"]),
            key="transaction_hour",
        ),
        "location_risk_score": st.sidebar.slider(
            "Location Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["location_risk_score"]),
            key="location_risk_score",
        ),
        "device_risk_score": st.sidebar.slider(
            "Device Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["device_risk_score"]),
            key="device_risk_score",
        ),
        "is_international": 1
        if st.sidebar.checkbox(
            "International Transaction",
            value=bool(st.session_state["is_international_checkbox"]),
            key="is_international_checkbox",
        )
        else 0,
        "average_daily_transaction_amount": st.sidebar.number_input(
            "Average Daily Transaction Amount",
            min_value=0.0,
            value=float(st.session_state["average_daily_transaction_amount"]),
            step=25.0,
            key="average_daily_transaction_amount",
        ),
    }
    st.session_state["is_international"] = transaction_input["is_international"]

    prediction_frame = build_prediction_frame(transaction_input)
    probability = float(model.predict_proba(prediction_frame)[0][1])
    prediction = int(probability >= threshold)
    risk_level = risk_level_from_probability(probability)
    explanation = explain_risk_factors(transaction_input, probability)
    shap_summary = explain_transaction(
        {
            "model": model,
            "threshold": threshold,
            "model_name": saved_model_name,
            "background_sample": dataset[RAW_FEATURE_COLUMNS].sample(
                n=min(200, len(dataset)),
                random_state=42,
            ),
        },
        pd.DataFrame([transaction_input]),
        top_n=8,
    )

    summary_col, gauge_col = st.columns([1.2, 1.0])
    metric_cols = summary_col.columns(4)
    metric_cols[0].metric(
        "Prediction",
        "Fraudulent" if prediction == 1 else "Legitimate",
    )
    metric_cols[1].metric("Fraud Probability", f"{probability:.1%}")
    metric_cols[2].metric("Risk Level", risk_level)
    metric_cols[3].metric("Decision Threshold", f"{threshold:.2f}")
    summary_col.info(explanation)
    gauge_col.plotly_chart(render_probability_gauge(probability), use_container_width=True)

    shap_col_1, shap_col_2 = st.columns([1.15, 0.85])
    contribution_frame = shap_summary["top_contributions"].copy()
    contribution_frame["direction_color"] = contribution_frame["impact"].map(
        {"Increase risk": "#e63946", "Reduce risk": "#2a9d8f"}
    )
    fig_shap = px.bar(
        contribution_frame.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="impact",
        title="Per-Transaction SHAP Explanation",
        color_discrete_map={"Increase risk": "#e63946", "Reduce risk": "#2a9d8f"},
    )
    shap_col_1.plotly_chart(fig_shap, use_container_width=True)
    shap_col_2.markdown("#### SHAP Insight")
    shap_col_2.caption(
        "Positive SHAP values push the prediction toward fraud, while negative values reduce fraud risk."
    )
    shap_col_2.dataframe(
        contribution_frame[["feature", "shap_value", "impact"]],
        use_container_width=True,
    )

    if metrics:
        st.subheader("Model Performance")
        header_metrics = st.columns(6)
        best_metrics = metrics.get("best_model_metrics", {})
        header_metrics[0].metric("Best Model", metrics.get("best_model", saved_model_name))
        header_metrics[1].metric("Accuracy", f"{best_metrics.get('accuracy', 0):.3f}")
        header_metrics[2].metric("Precision", f"{best_metrics.get('precision', 0):.3f}")
        header_metrics[3].metric("Recall", f"{best_metrics.get('recall', 0):.3f}")
        header_metrics[4].metric("F1-score", f"{best_metrics.get('f1_score', 0):.3f}")
        header_metrics[5].metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.3f}")

        secondary_metrics = st.columns(4)
        secondary_metrics[0].metric("PR-AUC", f"{best_metrics.get('pr_auc', 0):.3f}")
        secondary_metrics[1].metric("Fraud Rate", f"{metrics.get('fraud_rate', 0):.1%}")
        secondary_metrics[2].metric(
            "Dataset Size",
            f"{metrics.get('dataset_rows', 0):,}",
        )
        secondary_metrics[3].metric(
            "CV F1 Mean",
            f"{metrics.get('cross_validation', {}).get('f1_mean', 0):.3f}",
        )

    if drift_report:
        st.subheader("Drift Monitoring")
        drift_status = drift_report.get("overall_status", "unknown").upper()
        if drift_report.get("overall_status") == "alert":
            st.error(f"Overall drift status: {drift_status}")
        elif drift_report.get("overall_status") == "warning":
            st.warning(f"Overall drift status: {drift_status}")
        else:
            st.success(f"Overall drift status: {drift_status}")

        drift_frame = pd.DataFrame(drift_report.get("numeric_feature_drift", []))
        if not drift_frame.empty:
            fig_drift = px.bar(
                drift_frame.sort_values("psi", ascending=False),
                x="feature",
                y="psi",
                color="status",
                title="Population Stability Index by Feature",
                color_discrete_map={
                    "stable": "#2a9d8f",
                    "warning": "#f4a261",
                    "alert": "#e63946",
                },
            )
            st.plotly_chart(fig_drift, use_container_width=True)

    chart_frame = add_prediction_columns(dataset, model, threshold)

    st.subheader("Dashboard Analytics")
    chart_col_1, chart_col_2 = st.columns(2)
    chart_col_3, chart_col_4 = st.columns(2)

    fraud_counts = (
        chart_frame["fraud"]
        .map({0: "Legitimate", 1: "Fraud"})
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    fig_counts = px.bar(
        fraud_counts,
        x="label",
        y="count",
        color="label",
        color_discrete_map={"Legitimate": "#2a9d8f", "Fraud": "#e63946"},
        title="Fraud vs Legitimate Transaction Count",
    )
    chart_col_1.plotly_chart(fig_counts, use_container_width=True)

    fraud_by_type = (
        chart_frame.groupby("transaction_type", as_index=False)["fraud"]
        .mean()
        .sort_values("fraud", ascending=False)
    )
    fraud_by_type["fraud_rate"] = fraud_by_type["fraud"] * 100
    fig_type = px.bar(
        fraud_by_type,
        x="transaction_type",
        y="fraud_rate",
        color="transaction_type",
        title="Fraud Rate by Transaction Type",
        labels={"fraud_rate": "Fraud Rate (%)", "transaction_type": "Transaction Type"},
    )
    chart_col_2.plotly_chart(fig_type, use_container_width=True)

    fig_amount = px.histogram(
        chart_frame.assign(
            fraud_label=chart_frame["fraud"].map({0: "Legitimate", 1: "Fraud"})
        ),
        x="transaction_amount",
        color="fraud_label",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        title="Transaction Amount Distribution",
        labels={"fraud_label": "Class", "transaction_amount": "Transaction Amount"},
        color_discrete_map={"Legitimate": "#457b9d", "Fraud": "#e76f51"},
    )
    chart_col_3.plotly_chart(fig_amount, use_container_width=True)

    avg_probability = (
        chart_frame.groupby("risk_level", as_index=False)["fraud_probability"]
        .mean()
        .sort_values("fraud_probability", ascending=True)
    )
    fig_probability = px.bar(
        avg_probability,
        x="risk_level",
        y="fraud_probability",
        color="risk_level",
        title="Average Fraud Probability by Risk Level",
        labels={"fraud_probability": "Average Fraud Probability", "risk_level": "Risk Level"},
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#e63946"},
    )
    chart_col_4.plotly_chart(fig_probability, use_container_width=True)

    detail_col_1, detail_col_2 = st.columns(2)

    confusion = metrics.get("confusion_matrix", []) if metrics else []
    if confusion:
        confusion_frame = pd.DataFrame(
            confusion,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )
        fig_confusion = px.imshow(
            confusion_frame,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        detail_col_1.plotly_chart(fig_confusion, use_container_width=True)

    if not feature_importance.empty:
        top_features = feature_importance.head(10).copy()
        fig_features = px.bar(
            top_features.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importance",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        detail_col_2.plotly_chart(fig_features, use_container_width=True)

    with st.expander("Model Comparison Table"):
        comparison = metrics.get("comparison", []) if metrics else []
        if comparison:
            st.dataframe(pd.DataFrame(comparison), use_container_width=True)

    with st.expander("Input Snapshot"):
        st.dataframe(pd.DataFrame([transaction_input]))


if __name__ == "__main__":
    main()

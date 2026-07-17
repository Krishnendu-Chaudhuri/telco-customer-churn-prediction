"""Streamlit dashboard for churn analytics and retention insights."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.dashboard.logic import (
    build_segment_clv_summary,
    compute_executive_kpis,
    format_champion_panel,
    segmentation_has_data,
)
from src.data.data_loader import load_raw_data
from src.models.predictor import ChurnPredictor
from src.models.registry import ModelRegistry
from src.utils.config import get_config
from src.utils.paths import ProjectPaths, ensure_project_imports

_bootstrap_root = Path(__file__).resolve().parents[2]
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))


PROJECT_ROOT = ensure_project_imports(Path(__file__))

for marker in ("configs/config.yaml", "app/dashboard/logic.py"):
    if not (PROJECT_ROOT / marker).exists():
        raise RuntimeError(
            f"Expected Telco Churn repository marker missing: {PROJECT_ROOT / marker}"
        )

print(f"[Telco Churn] cwd={os.getcwd()}")
print(f"[Telco Churn] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[Telco Churn] streamlit_app={Path(__file__).resolve()}")
print(f"[Telco Churn] PID={os.getpid()}")


st.set_page_config(
    page_title="Telco Churn & Retention Engine",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1f3b73; }
    .kpi-card {
        background: linear-gradient(135deg, #1f3b73 0%, #3d6cb9 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; }
    .kpi-label { font-size: 0.9rem; opacity: 0.9; }
    .risk-critical { color: #d62728; font-weight: 700; }
    .risk-high { color: #ff7f0e; font-weight: 700; }
    .risk-medium { color: #bcbd22; font-weight: 700; }
    .risk-low { color: #2ca02c; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor() -> ChurnPredictor | None:
    predictor = ChurnPredictor()
    if not predictor.is_ready:
        return None
    try:
        return predictor.load()
    except Exception:
        return None


@st.cache_data
def load_data() -> pd.DataFrame:
    paths = ProjectPaths()
    if paths.processed_data.exists():
        return pd.read_parquet(paths.processed_data)
    return load_raw_data()


def render_kpi_card(label: str, value: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def executive_dashboard(df: pd.DataFrame, predictor: ChurnPredictor | None) -> None:
    st.markdown('<p class="main-header">Executive Dashboard</p>', unsafe_allow_html=True)

    kpis = compute_executive_kpis(df, predictor)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_kpi_card("Churn Rate", f"{kpis['churn_rate']:.1%}"), unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            render_kpi_card("Revenue at Risk", f"${kpis['revenue_at_risk']:,.0f}"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_kpi_card("High-Risk Customers", f"{kpis['high_risk_count']:,}"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(render_kpi_card("Avg CLV", f"${kpis['avg_clv']:,.0f}"), unsafe_allow_html=True)

    st.subheader("Churn Distribution")
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    if "churn_probability" in df.columns:
        st.subheader("High-Risk Customers")
        high_risk_df = (
            df[df["churn_probability"] >= 0.7]
            .sort_values("churn_probability", ascending=False)
            .head(20)
        )
        display_cols = [
            col
            for col in ["customerID", "tenure", "MonthlyCharges", "churn_probability", "segment"]
            if col in high_risk_df.columns
        ]
        st.dataframe(high_risk_df[display_cols], use_container_width=True)


def customer_lookup(predictor: ChurnPredictor | None) -> None:
    st.markdown('<p class="main-header">Customer Lookup</p>', unsafe_allow_html=True)
    if predictor is None:
        st.warning("Train the model first: `python src/models/train_model.py`")
        return

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.text_input("Customer ID", "CUST-001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        with col2:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
        with col3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
            total = st.number_input("Total Charges", min_value=0.0, value=840.0)

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        customer = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        result = predictor.predict_single(customer, include_explanation=True)
        risk_class = result["risk_level"].lower()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        c2.markdown(
            f'<p class="risk-{risk_class}">Risk Level: {result["risk_level"]}</p>',
            unsafe_allow_html=True,
        )
        c3.metric("CLV Estimate", f"${result['clv']:,.2f}")
        c4.metric("Retention Cost", f"${result['retention_cost']:,.2f}")

        st.success(f"Recommended Strategy: **{result['retention_strategy']}**")
        st.info(f"Expected Revenue Saved: ${result['expected_revenue_saved']:,.2f}")
        if result.get("explanation"):
            st.subheader("Churn Drivers")
            st.write(result["explanation"])
            if result.get("top_contributors"):
                contrib_df = pd.DataFrame(result["top_contributors"])
                fig = px.bar(
                    contrib_df,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    title="Top SHAP Contributors",
                )
                st.plotly_chart(fig, use_container_width=True)


def segmentation_dashboard(df: pd.DataFrame) -> None:
    st.markdown('<p class="main-header">Segmentation Dashboard</p>', unsafe_allow_html=True)
    if not segmentation_has_data(df):
        st.info("Run training to generate customer segments.")
        return

    segment_counts = df["segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    fig = px.bar(segment_counts, x="Segment", y="Count", color="Segment")
    st.plotly_chart(fig, use_container_width=True)

    if {"tenure", "MonthlyCharges", "segment"}.issubset(df.columns):
        scatter = px.scatter(
            df,
            x="tenure",
            y="MonthlyCharges",
            color="segment",
            opacity=0.6,
            title="Customer Segments",
        )
        st.plotly_chart(scatter, use_container_width=True)

    if "clv_estimate" in df.columns:
        revenue_by_segment = build_segment_clv_summary(df)
        if revenue_by_segment is not None:
            fig2 = px.bar(
                revenue_by_segment,
                x="segment",
                y="clv_estimate",
                title="CLV by Segment",
            )
            st.plotly_chart(fig2, use_container_width=True)


def _render_champion_challenger_panel(paths: ProjectPaths) -> None:
    """Show champion/challenger roles, metrics, and promotion history."""
    st.subheader("Champion vs Challenger")

    if not paths.registry_db.exists():
        st.info("Run training to initialize the champion/challenger registry.")
        return

    registry_obj = ModelRegistry.load(registry_path=paths.registry_db)
    panel = format_champion_panel(registry_obj)

    role_cols = st.columns(2)
    role_cols[0].metric("Champion (serving)", panel["champion_model"] or "—")
    role_cols[1].metric("Challenger (candidate)", panel["challenger_model"] or "—")
    st.caption(
        f"Promotion threshold: challenger must beat champion ROC-AUC by at least "
        f"{panel['promotion_threshold']:.3f} and maintain recall ≥ "
        f"{get_config()['champion_challenger']['min_recall_floor']:.2f}."
    )

    if panel["metric_rows"]:
        st.dataframe(pd.DataFrame(panel["metric_rows"]).set_index("role"), use_container_width=True)

    if panel["history"]:
        st.markdown("**Recent promotion decisions**")
        history_df = pd.DataFrame(panel["history"])
        display_cols = [
            col
            for col in [
                "timestamp",
                "action",
                "previous_champion",
                "new_champion",
                "champion_metric",
                "challenger_metric",
                "delta",
            ]
            if col in history_df.columns
        ]
        st.dataframe(history_df[display_cols], use_container_width=True)


def model_performance_dashboard() -> None:
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    paths = ProjectPaths()

    if paths.model_comparison.exists():
        comparison = pd.read_csv(paths.model_comparison, index_col=0)
        st.subheader("Model Comparison")
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

    _render_champion_challenger_panel(paths)

    if paths.model_metadata.exists():
        with paths.model_metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        st.subheader("Best Model Metrics")
        metrics = metadata.get("metrics", {})
        cols = st.columns(5)
        for idx, (name, value) in enumerate(metrics.items()):
            if idx < 5:
                cols[idx].metric(name.upper(), f"{value:.4f}")

    eval_dir = paths.evaluation_dir
    image_files = {
        "Confusion Matrix": eval_dir / "confusion_matrix.png",
        "ROC Curve": eval_dir / "roc_curve.png",
        "PR Curve": eval_dir / "pr_curve.png",
        "Feature Importance": eval_dir / "feature_importance.png",
    }
    for title, image_path in image_files.items():
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)


def explainability_dashboard(df: pd.DataFrame, predictor: ChurnPredictor | None) -> None:
    st.markdown('<p class="main-header">Explainability Dashboard</p>', unsafe_allow_html=True)
    paths = ProjectPaths()
    shap_path = paths.evaluation_dir / "shap_summary.png"
    if shap_path.exists():
        st.subheader("Global SHAP Summary")
        st.image(str(shap_path), use_container_width=True)

    importance_path = paths.evaluation_dir / "feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(15)
        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

    if predictor is not None and "customerID" in df.columns:
        st.subheader("Local Explanation")
        customer_id = st.selectbox("Select Customer", df["customerID"].head(100))
        customer_row = df[df["customerID"] == customer_id].iloc[0]
        raw_cols = [
            col
            for col in customer_row.index
            if col not in {"churn_probability", "segment", "cluster_id", "Churn"}
        ]
        result = predictor.predict_single(
            customer_row[raw_cols].to_dict(),
            include_explanation=True,
        )
        st.write(result.get("explanation", "No explanation available."))


def main() -> None:
    predictor = load_predictor()
    df = load_data()

    st.sidebar.title("Telco Retention Engine")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Customer Lookup",
            "Segmentation",
            "Model Performance",
            "Explainability",
        ],
    )

    if predictor is None:
        st.sidebar.error("Model not trained yet.")

    if page == "Executive Dashboard":
        executive_dashboard(df, predictor)
    elif page == "Customer Lookup":
        customer_lookup(predictor)
    elif page == "Segmentation":
        segmentation_dashboard(df)
    elif page == "Model Performance":
        model_performance_dashboard()
    elif page == "Explainability":
        explainability_dashboard(df, predictor)


if __name__ == "__main__":
    main()

"""Streamlit application for Customer Churn Prediction.

Provides an interactive web interface for uploading customer CSV data,
scoring churn probabilities, viewing SHAP explanations, retention
strategies, and revenue impact analysis.

Usage:
    streamlit run app.py
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import yaml

from src.explainability import (
    compute_shap_values,
    get_customer_explanation,
    get_global_feature_importance,
)
from src.impact import (
    calculate_projected_savings,
    calculate_revenue_at_risk,
    build_impact_summary,
)
from src.ingestion import validate_schema
from src.persistence import load_artifacts
from src.preprocessing import transform
from src.scoring import (
    assign_risk_segments,
    build_scored_results,
    score_customers,
)
from src.strategy import load_strategy_mapping, recommend_batch

logger = logging.getLogger(__name__)

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# ── Helper functions ─────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _find_latest_artifact_dir(artifacts_root: str) -> str | None:
    """Find the most recent timestamped subdirectory in *artifacts_root*.

    Returns the full path to the latest version directory, or ``None``
    if no valid subdirectories exist.
    """
    if not os.path.isdir(artifacts_root):
        return None

    subdirs = [
        d for d in os.listdir(artifacts_root)
        if os.path.isdir(os.path.join(artifacts_root, d))
        and d != ".gitkeep"
    ]

    if not subdirs:
        return None

    # Timestamped directories sort lexicographically (YYYYMMDD_HHMMSS)
    subdirs.sort(reverse=True)
    return os.path.join(artifacts_root, subdirs[0])


@st.cache_resource
def load_model_and_pipeline(version_dir: str):
    """Load and cache model, pipeline, and metadata from a version directory."""
    model, pipeline, metadata = load_artifacts(version_dir)
    return model, pipeline, metadata


@st.cache_data
def load_config(config_dir: str):
    """Load and cache all configuration files."""
    schema_config = _load_yaml(os.path.join(config_dir, "schema.yaml"))
    thresholds_config = _load_yaml(os.path.join(config_dir, "thresholds.yaml"))
    return schema_config, thresholds_config


def _auto_train(config_dir: str, artifacts_root: str) -> None:
    """Generate sample data (if missing) and run the training pipeline.

    This enables zero-config deployment on Streamlit Community Cloud.
    """
    import subprocess
    import sys

    data_path = os.path.join("data", "customers.csv")
    if not os.path.exists(data_path):
        subprocess.run(
            [sys.executable, "scripts/generate_sample_data.py"],
            check=True,
        )

    subprocess.run(
        [
            sys.executable, "main.py",
            "--input", data_path,
            "--config-dir", config_dir,
            "--output-dir", artifacts_root,
        ],
        check=True,
    )


# ── Main application ─────────────────────────────────────────────────

def main() -> None:
    """Run the Streamlit application."""
    st.title("📊 Customer Churn Prediction")
    st.markdown(
        "Upload a CSV of customer records to score churn probabilities, "
        "view SHAP explanations, and get retention strategy recommendations."
    )

    # ── Sidebar: configuration ───────────────────────────────────────
    st.sidebar.header("Configuration")

    config_dir = st.sidebar.text_input("Config directory", value="config")
    artifacts_root = st.sidebar.text_input("Artifacts directory", value="artifacts")

    # Attempt to find the latest artifact version
    latest_version = _find_latest_artifact_dir(artifacts_root)

    if latest_version is None:
        with st.spinner("No model found — training on sample data. This takes about a minute on first launch..."):
            try:
                _auto_train(config_dir, artifacts_root)
            except Exception as exc:
                st.error(
                    f"Auto-training failed: {exc}\n\n"
                    "Please run the pipeline manually:\n"
                    "```\npython main.py --input data/customers.csv\n```"
                )
                return
        latest_version = _find_latest_artifact_dir(artifacts_root)
        if latest_version is None:
            st.error("Auto-training completed but no artifacts were created.")
            return
        st.rerun()

    st.sidebar.success(f"Model loaded from: `{os.path.basename(latest_version)}`")

    # Load model, pipeline, and config
    try:
        model, pipeline, metadata = load_model_and_pipeline(latest_version)
        schema_config, thresholds_config = load_config(config_dir)
    except Exception as exc:
        st.error(f"Failed to load model or configuration: {exc}")
        return

    feature_names = metadata.feature_list
    required_columns = schema_config["required_columns"]
    revenue_column = schema_config.get("revenue_column", "annual_revenue")
    id_column = schema_config.get("id_column", "customer_id")
    target_column = schema_config.get("target_column", "churn")

    # Parse risk thresholds
    risk_thresholds: dict[str, float] = {}
    for segment, cfg in thresholds_config.get("risk_segments", {}).items():
        risk_thresholds[segment] = cfg["min_probability"]

    retention_cfg = thresholds_config.get("retention", {})
    retention_target_pct = retention_cfg.get("target_percentage", 0.10)
    retention_success_rate = retention_cfg.get("success_rate", 0.50)
    estimated_cost_per_customer = retention_cfg.get(
        "estimated_cost_per_customer", 100.0
    )

    # Strategy mapping
    strategies_path = os.path.join(config_dir, "strategies.yaml")
    try:
        strategy_mapping = load_strategy_mapping(strategies_path)
    except Exception as exc:
        st.error(f"Failed to load strategy mapping: {exc}")
        return

    # Display model metadata in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.markdown(f"**Type:** {metadata.model_type}")
    st.sidebar.markdown(f"**Trained:** {metadata.training_date}")
    st.sidebar.markdown(f"**Training rows:** {metadata.dataset_row_count:,}")
    if metadata.evaluation_metrics:
        auc = metadata.evaluation_metrics.get("auc_roc", "N/A")
        if isinstance(auc, float):
            st.sidebar.markdown(f"**AUC-ROC:** {auc:.4f}")

    # ── Data source selection ────────────────────────────────────────
    st.header("Customer Data")

    data_mode = st.radio(
        "Choose data source:",
        options=["📂 Upload CSV", "🎯 Demo Mode (preloaded example)"],
        horizontal=True,
    )

    df = None

    if data_mode == "🎯 Demo Mode (preloaded example)":
        demo_path = os.path.join("data", "customers.csv")
        if not os.path.exists(demo_path):
            st.error(
                f"Demo dataset not found at `{demo_path}`. "
                "Generate it first by running:\n\n"
                "```\npython scripts/generate_sample_data.py\n```"
            )
            return
        df = pd.read_csv(demo_path)
        st.success(
            f"Demo dataset loaded — {len(df):,} customers, "
            f"{len(df.columns)} columns from `{demo_path}`"
        )
    else:
        st.markdown(
            "**Required CSV columns:** `customer_id`, `churn`, plus feature "
            "columns defined in `config/schema.yaml`:\n"
            "- **Numerical:** `tenure_months`, `monthly_charges`, "
            "`total_charges`, `num_support_tickets`, `annual_revenue`, `age`\n"
            "- **Categorical:** `gender`, `location`, `contract_type`, "
            "`payment_method`"
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV file with customer records",
            type=["csv"],
        )

        if uploaded_file is None:
            st.info("Upload a CSV file to get started, or switch to Demo Mode.")
            return

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
            return

        st.success(
            f"CSV loaded successfully — {len(df):,} customers, "
            f"{len(df.columns)} columns."
        )

    # ── Validate ─────────────────────────────────────────────────────
    try:
        validate_schema(df, required_columns)
    except ValueError as exc:
        st.error(f"Schema validation failed: {exc}")
        return

    # ── Preprocess ───────────────────────────────────────────────────
    try:
        df_features = df.drop(columns=[target_column], errors="ignore")
        # Also drop the ID column before transforming
        if id_column in df_features.columns:
            df_features_for_transform = df_features.drop(columns=[id_column])
        else:
            df_features_for_transform = df_features
        X = transform(pipeline, df_features_for_transform)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    # ── Score ────────────────────────────────────────────────────────
    try:
        probabilities = score_customers(model, X)
        segments = assign_risk_segments(probabilities, risk_thresholds)
        scored_customers = build_scored_results(probabilities, segments)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        return

    # ── SHAP explanations ────────────────────────────────────────────
    with st.spinner("Computing SHAP explanations..."):
        try:
            shap_values, base_value = compute_shap_values(
                model, X, feature_names
            )
            explanations = [
                get_customer_explanation(
                    shap_values, base_value, idx, feature_names
                )
                for idx in range(len(X))
            ]
        except Exception as exc:
            st.error(f"SHAP computation failed: {exc}")
            return

    # ── Retention strategies ─────────────────────────────────────────
    try:
        customer_strategies = recommend_batch(
            scored_customers, explanations, strategy_mapping
        )
    except Exception as exc:
        st.error(f"Strategy recommendation failed: {exc}")
        return

    # ── Revenue impact ───────────────────────────────────────────────
    try:
        revenue_at_risk = calculate_revenue_at_risk(
            df, segments, revenue_column
        )
        projected_saved, targeted_count = calculate_projected_savings(
            df, segments, probabilities,
            retention_target_pct=retention_target_pct,
            retention_success_rate=retention_success_rate,
            revenue_column=revenue_column,
        )
        estimated_retention_cost = targeted_count * estimated_cost_per_customer
        impact_summary = build_impact_summary(
            total_revenue_at_risk=revenue_at_risk,
            targeted_customer_count=targeted_count,
            projected_revenue_saved=projected_saved,
            estimated_retention_cost=estimated_retention_cost,
        )
    except Exception as exc:
        st.error(f"Impact analysis failed: {exc}")
        return

    # ── Display: Summary metrics ─────────────────────────────────────
    st.header("Summary")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(scored_customers)
    high_count = sum(1 for s in segments if s == "High")
    medium_count = sum(1 for s in segments if s == "Medium")
    low_count = sum(1 for s in segments if s == "Low")

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("High Risk", f"{high_count:,}", delta=f"{high_count / total_customers * 100:.1f}%")
    col3.metric("Medium Risk", f"{medium_count:,}", delta=f"{medium_count / total_customers * 100:.1f}%")
    col4.metric("Low Risk", f"{low_count:,}", delta=f"{low_count / total_customers * 100:.1f}%")

    # ── Display: Results table ───────────────────────────────────────
    st.header("Scored Results")

    # Build the results DataFrame
    strategy_lookup = {cs.customer_index: cs for cs in customer_strategies}
    explanation_lookup = {ex.customer_index: ex for ex in explanations}

    rows = []
    for sc in scored_customers:
        idx = sc.customer_index

        # Customer ID
        if id_column in df.columns and idx < len(df):
            cust_id = df.iloc[idx][id_column]
        else:
            cust_id = idx

        # Top 3 drivers
        explanation = explanation_lookup.get(idx)
        top_drivers = []
        if explanation and explanation.ranked_features:
            for feat_name, _ in explanation.ranked_features[:3]:
                top_drivers.append(feat_name)

        # Strategy
        strategy = strategy_lookup.get(idx)
        retention_strategy = strategy.retention_strategy if strategy else ""

        rows.append({
            "Customer ID": cust_id,
            "Churn Probability": round(sc.churn_probability, 4),
            "Risk Segment": sc.risk_segment,
            "Top Driver 1": top_drivers[0] if len(top_drivers) > 0 else "",
            "Top Driver 2": top_drivers[1] if len(top_drivers) > 1 else "",
            "Top Driver 3": top_drivers[2] if len(top_drivers) > 2 else "",
            "Retention Strategy": retention_strategy,
        })

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ── Display: Revenue impact summary ──────────────────────────────
    st.header("Revenue Impact Summary")

    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    impact_col1.metric(
        "Revenue at Risk",
        f"${impact_summary.total_revenue_at_risk:,.2f}",
    )
    impact_col2.metric(
        "Targeted Customers",
        f"{impact_summary.targeted_customer_count:,}",
    )
    impact_col3.metric(
        "Projected Revenue Saved",
        f"${impact_summary.projected_revenue_saved:,.2f}",
    )
    impact_col4.metric(
        "Retention ROI",
        f"{impact_summary.retention_roi:.2f}x",
    )

    # ── Display: Individual SHAP waterfall plot ──────────────────────
    st.header("Individual Customer Explanation")

    # Build customer ID options for the selector
    customer_options = []
    for sc in scored_customers:
        idx = sc.customer_index
        if id_column in df.columns and idx < len(df):
            cust_id = df.iloc[idx][id_column]
        else:
            cust_id = idx
        customer_options.append((cust_id, idx))

    # Create a mapping from display label to index
    option_labels = [str(cid) for cid, _ in customer_options]
    label_to_index = {str(cid): idx for cid, idx in customer_options}

    selected_label = st.selectbox(
        "Select a customer to view SHAP waterfall plot",
        options=option_labels,
    )

    if selected_label is not None:
        selected_index = label_to_index[selected_label]
        explanation = explanation_lookup[selected_index]

        # Build a shap.Explanation for the waterfall plot
        shap_explanation = shap.Explanation(
            values=explanation.shap_values,
            base_values=explanation.base_value,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()

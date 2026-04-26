"""Flask web application for Customer Churn Prediction.

Serves a modern HTML/CSS/JS dashboard with REST API endpoints
for scoring, explanations, and impact analysis.

Usage:
    python webapp.py
    # Then open http://localhost:5000
"""

import gc
import io
import json
import logging
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from flask import Flask, jsonify, render_template, request, send_file

from src.explainability import (
    compute_shap_values,
    get_customer_explanation,
    get_global_feature_importance,
)
from src.impact import (
    build_impact_summary,
    calculate_projected_savings,
    calculate_revenue_at_risk,
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.after_request
def add_cors_headers(response):
    """Add CORS headers to every response for the Vercel frontend."""
    origin = request.headers.get("Origin", "")
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    if request.method == "OPTIONS":
        response.status_code = 204
    return response

# ── Global state ─────────────────────────────────────────────────────
_state = {
    "model": None,
    "pipeline": None,
    "metadata": None,
    "schema_config": None,
    "thresholds_config": None,
    "strategy_mapping": None,
    "version_dir": None,
}


# ── Helpers ──────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _find_latest_artifact_dir(artifacts_root: str = "artifacts") -> str | None:
    if not os.path.isdir(artifacts_root):
        return None
    subdirs = [
        d for d in os.listdir(artifacts_root)
        if os.path.isdir(os.path.join(artifacts_root, d)) and d != ".gitkeep"
    ]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)
    return os.path.join(artifacts_root, subdirs[0])


def _ensure_model_loaded(
    config_dir: str = "config", artifacts_root: str = "artifacts"
) -> bool:
    """Load model and config if not already loaded. Returns True on success."""
    if _state["model"] is not None:
        return True

    version_dir = _find_latest_artifact_dir(artifacts_root)
    if version_dir is None:
        logger.error(
            "No model artifacts found in %s. "
            "Run build.sh or main.py to train a model first.",
            artifacts_root,
        )
        return False

    model, pipeline, metadata = load_artifacts(version_dir)
    _state["model"] = model
    _state["pipeline"] = pipeline
    _state["metadata"] = metadata
    _state["version_dir"] = version_dir
    _state["schema_config"] = _load_yaml(os.path.join(config_dir, "schema.yaml"))
    _state["thresholds_config"] = _load_yaml(
        os.path.join(config_dir, "thresholds.yaml")
    )
    _state["strategy_mapping"] = load_strategy_mapping(
        os.path.join(config_dir, "strategies.yaml")
    )
    return True


def _run_scoring_pipeline(df: pd.DataFrame) -> dict:
    """Run the full scoring pipeline on a DataFrame and return results dict."""
    schema = _state["schema_config"]
    thresholds_cfg = _state["thresholds_config"]
    model = _state["model"]
    pipeline = _state["pipeline"]
    metadata = _state["metadata"]
    strategy_mapping = _state["strategy_mapping"]

    feature_names = metadata.feature_list
    id_column = schema.get("id_column", "customer_id")
    target_column = schema.get("target_column", "churn")
    revenue_column = schema.get("revenue_column", "annual_revenue")

    risk_thresholds = {}
    for segment, cfg in thresholds_cfg.get("risk_segments", {}).items():
        risk_thresholds[segment] = cfg["min_probability"]

    retention_cfg = thresholds_cfg.get("retention", {})
    retention_target_pct = retention_cfg.get("target_percentage", 0.10)
    retention_success_rate = retention_cfg.get("success_rate", 0.50)
    cost_per_customer = retention_cfg.get("estimated_cost_per_customer", 100.0)

    # Validate
    required = schema["required_columns"]
    validate_schema(df, required)

    # Preprocess
    df_features = df.drop(columns=[target_column], errors="ignore")
    if id_column in df_features.columns:
        df_features = df_features.drop(columns=[id_column])
    X = transform(pipeline, df_features)

    # Score
    probabilities = score_customers(model, X)
    segments = assign_risk_segments(probabilities, risk_thresholds)
    scored_customers = build_scored_results(probabilities, segments)

    # SHAP
    shap_values, base_value = compute_shap_values(model, X, feature_names)
    explanations = [
        get_customer_explanation(shap_values, base_value, idx, feature_names)
        for idx in range(len(X))
    ]

    # Global feature importance
    global_importance = get_global_feature_importance(shap_values, feature_names)

    # Strategies
    customer_strategies = recommend_batch(
        scored_customers, explanations, strategy_mapping
    )

    # Impact
    revenue_at_risk = calculate_revenue_at_risk(df, segments, revenue_column)
    projected_saved, targeted_count = calculate_projected_savings(
        df, segments, probabilities,
        retention_target_pct=retention_target_pct,
        retention_success_rate=retention_success_rate,
        revenue_column=revenue_column,
    )
    estimated_cost = targeted_count * cost_per_customer
    impact = build_impact_summary(
        total_revenue_at_risk=revenue_at_risk,
        targeted_customer_count=targeted_count,
        projected_revenue_saved=projected_saved,
        estimated_retention_cost=estimated_cost,
    )

    # Build customer rows
    strategy_lookup = {cs.customer_index: cs for cs in customer_strategies}
    explanation_lookup = {ex.customer_index: ex for ex in explanations}

    customers = []
    for sc in scored_customers:
        idx = sc.customer_index
        cust_id = (
            str(df.iloc[idx][id_column])
            if id_column in df.columns and idx < len(df)
            else str(idx)
        )
        expl = explanation_lookup.get(idx)
        drivers = []
        driver_values = []
        if expl and expl.ranked_features:
            for feat, val in expl.ranked_features[:3]:
                drivers.append(feat)
                driver_values.append(round(float(val), 4))

        strat = strategy_lookup.get(idx)
        rev = (
            float(df.iloc[idx][revenue_column])
            if revenue_column in df.columns and idx < len(df)
            else 0.0
        )

        customers.append({
            "customer_id": cust_id,
            "churn_probability": round(sc.churn_probability, 4),
            "risk_segment": sc.risk_segment,
            "drivers": drivers,
            "driver_values": driver_values,
            "retention_strategy": strat.retention_strategy if strat else "",
            "annual_revenue": round(rev, 2),
        })

    # Segment counts
    high = sum(1 for s in segments if s == "High")
    medium = sum(1 for s in segments if s == "Medium")
    low = sum(1 for s in segments if s == "Low")

    return {
        "summary": {
            "total_customers": len(scored_customers),
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
        },
        "impact": asdict(impact),
        "customers": customers,
        "feature_importance": [
            {"feature": f, "importance": round(float(v), 6)}
            for f, v in global_importance[:15]
        ],
        "model_info": {
            "model_type": metadata.model_type,
            "training_date": metadata.training_date,
            "dataset_row_count": metadata.dataset_row_count,
            "auc_roc": metadata.evaluation_metrics.get("auc_roc"),
            "accuracy": metadata.evaluation_metrics.get("accuracy"),
            "version": os.path.basename(_state["version_dir"]),
        },
        # Store for waterfall generation
        "_shap_values": shap_values,
        "_base_value": base_value,
        "_feature_names": feature_names,
    }


# Keep last scoring result for waterfall requests
_last_result = {}


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info")
def api_model_info():
    if not _ensure_model_loaded():
        return jsonify({"error": "No model available"}), 500
    meta = _state["metadata"]
    return jsonify({
        "model_type": meta.model_type,
        "training_date": meta.training_date,
        "dataset_row_count": meta.dataset_row_count,
        "evaluation_metrics": meta.evaluation_metrics,
        "version": os.path.basename(_state["version_dir"]),
    })


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score customers from uploaded CSV or use demo data."""
    global _last_result

    if not _ensure_model_loaded():
        return jsonify({"error": "No model available"}), 500

    # Check if file uploaded or demo mode
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            return jsonify({"error": f"Failed to read CSV: {exc}"}), 400
    else:
        # Demo mode
        demo_path = os.path.join("data", "customers.csv")
        if not os.path.exists(demo_path):
            return jsonify({"error": "Demo dataset not found"}), 404
        df = pd.read_csv(demo_path)

    try:
        result = _run_scoring_pipeline(df)
        # Store internal data for waterfall, remove from response
        _last_result = {
            "shap_values": result.pop("_shap_values"),
            "base_value": result.pop("_base_value"),
            "feature_names": result.pop("_feature_names"),
        }
        # Free the DataFrame to reduce memory pressure
        del df
        gc.collect()
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Scoring failed")
        return jsonify({"error": f"Scoring failed: {exc}"}), 500


@app.route("/api/waterfall/<int:customer_index>")
def api_waterfall(customer_index: int):
    """Generate a SHAP waterfall plot PNG for a specific customer."""
    if not _last_result:
        return jsonify({"error": "No scoring results available. Score data first."}), 400

    shap_values = _last_result["shap_values"]
    base_value = _last_result["base_value"]
    feature_names = _last_result["feature_names"]

    if customer_index < 0 or customer_index >= len(shap_values):
        return jsonify({"error": "Invalid customer index"}), 400

    explanation = get_customer_explanation(
        shap_values, base_value, customer_index, feature_names
    )

    shap_expl = shap.Explanation(
        values=explanation.shap_values,
        base_values=explanation.base_value,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_expl, show=False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png", download_name="waterfall.png")


@app.route("/api/export/csv")
def api_export_csv():
    """Export the last scoring results as a downloadable CSV."""
    if not _last_result:
        return jsonify({"error": "No scoring results available"}), 400

    # Re-run demo to get the data (or use cached)
    demo_path = os.path.join("data", "customers.csv")
    if not os.path.exists(demo_path):
        return jsonify({"error": "No data available"}), 404

    # Read the latest dashboard CSV if available
    version_dir = _state.get("version_dir")
    if version_dir:
        csv_path = os.path.join(version_dir, "dashboards", "customer_summary.csv")
        if os.path.exists(csv_path):
            return send_file(
                csv_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name="customer_summary.csv",
            )

    return jsonify({"error": "Export not available"}), 404


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    _ensure_model_loaded()
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )

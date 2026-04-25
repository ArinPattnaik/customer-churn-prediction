"""CLI entry point for the Customer Churn Prediction pipeline.

Wires all modules together into an end-to-end pipeline:
load config → ingest CSV → preprocess → train → score → explain →
recommend strategies → analyze impact → persist artifacts.

Usage:
    # Training mode (full pipeline):
    python main.py --input data/customers.csv

    # Inference-only mode (load existing model):
    python main.py --input data/customers.csv --model-version artifacts/20240115_103000

    # Custom directories:
    python main.py --input data/customers.csv --config-dir config --output-dir artifacts
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

import yaml

from src.ingestion import load_csv, validate_schema, log_data_summary
from src.preprocessing import (
    build_preprocessing_pipeline,
    fit_transform,
    transform,
    load_pipeline,
)
from src.training import split_data, train_models, select_best_model
from src.scoring import score_customers, assign_risk_segments, build_scored_results
from src.explainability import (
    compute_shap_values,
    get_customer_explanation,
    get_global_feature_importance,
    save_summary_plot,
)
from src.strategy import load_strategy_mapping, recommend_batch
from src.impact import (
    calculate_revenue_at_risk,
    calculate_projected_savings,
    build_impact_summary,
)
from src.persistence import save_artifacts, load_artifacts, ArtifactMetadata
from src.dashboard_export import (
    export_customer_summary_csv,
    export_feature_importance_csv,
    export_impact_summary_json,
)

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_pipeline(
    input_csv: str,
    config_dir: str,
    output_dir: str,
    model_version: str | None = None,
) -> None:
    """Execute the full churn prediction pipeline.

    Args:
        input_csv: Path to the input CSV file.
        config_dir: Directory containing schema.yaml, thresholds.yaml,
            and strategies.yaml.
        output_dir: Directory where artifacts will be saved.
        model_version: Optional path to a previously saved model version
            directory. When provided, the pipeline skips training and
            loads the existing model for inference only.
    """
    # ── 1. Load configuration ────────────────────────────────────────
    logger.info("Stage 1/9: Loading configuration from %s", config_dir)

    schema_path = os.path.join(config_dir, "schema.yaml")
    thresholds_path = os.path.join(config_dir, "thresholds.yaml")
    strategies_path = os.path.join(config_dir, "strategies.yaml")

    schema_config = _load_yaml(schema_path)
    thresholds_config = _load_yaml(thresholds_path)

    required_columns = schema_config["required_columns"]
    numerical_columns = schema_config["numerical_columns"]
    categorical_columns = schema_config["categorical_columns"]
    target_column = schema_config["target_column"]
    revenue_column = schema_config.get("revenue_column", "annual_revenue")
    encoding_strategy = schema_config.get("encoding_strategy", "onehot")

    # Parse risk segment thresholds into {segment_name: min_probability}
    risk_thresholds: dict[str, float] = {}
    for segment, cfg in thresholds_config.get("risk_segments", {}).items():
        risk_thresholds[segment] = cfg["min_probability"]

    retention_cfg = thresholds_config.get("retention", {})
    retention_target_pct = retention_cfg.get("target_percentage", 0.10)
    retention_success_rate = retention_cfg.get("success_rate", 0.50)
    estimated_cost_per_customer = retention_cfg.get(
        "estimated_cost_per_customer", 100.0
    )

    logger.info("Configuration loaded successfully.")

    # ── 2. Ingest and validate CSV ───────────────────────────────────
    logger.info("Stage 2/9: Ingesting CSV from %s", input_csv)

    df = load_csv(input_csv)
    validate_schema(df, required_columns)
    log_data_summary(df)

    logger.info("CSV ingestion and validation complete.")

    # ── 3. Preprocess ────────────────────────────────────────────────
    inference_only = model_version is not None

    if inference_only:
        logger.info(
            "Stage 3/9: Preprocessing (inference-only mode — loading "
            "pipeline from %s)",
            model_version,
        )
        model, pipeline, metadata = load_artifacts(model_version)
        # Drop target column if present for inference
        df_features = df.drop(columns=[target_column], errors="ignore")
        X = transform(pipeline, df_features)
        feature_names = list(pipeline.get_feature_names_out())
        logger.info("Preprocessing complete (inference mode).")
    else:
        logger.info("Stage 3/9: Preprocessing (training mode)")
        pipeline = build_preprocessing_pipeline(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            encoding_strategy=encoding_strategy,
        )
        X, y, feature_names = fit_transform(pipeline, df, target_column)
        logger.info("Preprocessing complete (training mode).")

    # ── 4. Train or skip ─────────────────────────────────────────────
    if inference_only:
        logger.info(
            "Stage 4/9: Skipping training (inference-only mode, "
            "model loaded from %s)",
            model_version,
        )
    else:
        logger.info("Stage 4/9: Training models")
        X_train, X_test, y_train, y_test = split_data(X, y)
        trained_models = train_models(X_train, y_train, X_test, y_test)
        best = select_best_model(trained_models)
        model = best.model
        logger.info(
            "Training complete. Best model: %s (AUC-ROC: %.4f)",
            best.name,
            best.metrics.auc_roc,
        )

    # ── 5. Score customers ───────────────────────────────────────────
    logger.info("Stage 5/9: Scoring customers")

    probabilities = score_customers(model, X)
    segments = assign_risk_segments(probabilities, risk_thresholds)
    scored_customers = build_scored_results(probabilities, segments)

    logger.info("Scoring complete. %d customers scored.", len(scored_customers))

    # ── 6. Compute SHAP explanations ─────────────────────────────────
    logger.info("Stage 6/9: Computing SHAP explanations")

    shap_values, base_value = compute_shap_values(model, X, feature_names)

    explanations = [
        get_customer_explanation(shap_values, base_value, idx, feature_names)
        for idx in range(len(X))
    ]
    global_importance = get_global_feature_importance(shap_values, feature_names)

    logger.info("SHAP explanations computed for %d customers.", len(explanations))

    # ── 7. Recommend retention strategies ────────────────────────────
    logger.info("Stage 7/9: Recommending retention strategies")

    strategy_mapping = load_strategy_mapping(strategies_path)
    customer_strategies = recommend_batch(
        scored_customers, explanations, strategy_mapping
    )

    logger.info(
        "Strategies recommended for %d customers.", len(customer_strategies)
    )

    # ── 8. Analyze revenue impact ────────────────────────────────────
    logger.info("Stage 8/9: Analyzing revenue impact")

    revenue_at_risk = calculate_revenue_at_risk(df, segments, revenue_column)
    projected_saved, targeted_count = calculate_projected_savings(
        df,
        segments,
        probabilities,
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

    logger.info(
        "Impact analysis complete. Revenue at risk: %.2f, "
        "Projected saved: %.2f, ROI: %.2f",
        impact_summary.total_revenue_at_risk,
        impact_summary.projected_revenue_saved,
        impact_summary.retention_roi,
    )

    # ── 9. Persist artifacts ─────────────────────────────────────────
    logger.info("Stage 9/9: Persisting artifacts to %s", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if not inference_only:
        metadata = ArtifactMetadata(
            model_type=best.name,
            training_date=datetime.now().isoformat(),
            dataset_row_count=len(df),
            feature_list=feature_names,
            evaluation_metrics=asdict(best.metrics),
        )

    version_dir = save_artifacts(
        output_dir=output_dir,
        model=model,
        pipeline=pipeline,
        metadata=metadata,
        feature_config={
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "encoding_strategy": encoding_strategy,
            "target_column": target_column,
            "revenue_column": revenue_column,
        },
    )

    # ── 10. Export dashboard-ready data ─────────────────────────────
    logger.info("Stage 10: Exporting dashboard-ready data")

    dashboard_dir = os.path.join(version_dir, "dashboards")
    os.makedirs(dashboard_dir, exist_ok=True)

    # Save SHAP summary plot for dashboard embedding
    summary_plot_path = os.path.join(dashboard_dir, "shap_summary.png")
    save_summary_plot(shap_values, feature_names, summary_plot_path)

    # Customer summary CSV
    export_customer_summary_csv(
        df=df,
        scored_customers=scored_customers,
        customer_strategies=customer_strategies,
        explanations=explanations,
        revenue_column=revenue_column,
        output_path=os.path.join(dashboard_dir, "customer_summary.csv"),
    )

    # Global feature importance CSV
    export_feature_importance_csv(
        global_importance=global_importance,
        output_path=os.path.join(dashboard_dir, "feature_importance.csv"),
    )

    # Revenue impact summary JSON
    export_impact_summary_json(
        impact_summary=impact_summary,
        output_path=os.path.join(dashboard_dir, "impact_summary.json"),
    )

    logger.info(
        "Dashboard exports saved to %s", dashboard_dir
    )

    logger.info("Pipeline complete. Artifacts saved to %s", version_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with input, config_dir, output_dir, and
        model_version attributes.
    """
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file containing customer data.",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing YAML configuration files "
        "(schema.yaml, thresholds.yaml, strategies.yaml). "
        "Default: config",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where model artifacts will be saved. "
        "Default: artifacts",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Path to a previously saved model version directory. "
        "When provided, the pipeline skips training and uses the "
        "existing model for inference only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging, parse args, and run the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args(argv)

    logger.info(
        "Starting pipeline — input=%s, config_dir=%s, output_dir=%s, "
        "model_version=%s",
        args.input,
        args.config_dir,
        args.output_dir,
        args.model_version,
    )

    run_pipeline(
        input_csv=args.input,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )


if __name__ == "__main__":
    main()

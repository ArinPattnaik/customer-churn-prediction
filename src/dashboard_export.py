"""Dashboard Data Export module.

Exports dashboard-ready data files for Power BI / Tableau consumption:
- Customer summary CSV with churn probabilities, risk segments, top drivers,
  retention strategies, and revenue.
- Global feature importance CSV.
- Revenue impact summary JSON.

The SHAP summary plot PNG is handled by
:func:`src.explainability.save_summary_plot` and should be called
separately to save the beeswarm plot for dashboard embedding.
"""

import json
import logging
import os
from dataclasses import asdict

import pandas as pd

from src.explainability import CustomerExplanation
from src.impact import ImpactSummary
from src.scoring import ScoredCustomer
from src.strategy import CustomerStrategy

logger = logging.getLogger(__name__)


def export_customer_summary_csv(
    df: pd.DataFrame,
    scored_customers: list[ScoredCustomer],
    customer_strategies: list[CustomerStrategy],
    explanations: list[CustomerExplanation],
    revenue_column: str,
    output_path: str,
    id_column: str = "customer_id",
) -> str:
    """Export a dashboard-ready customer summary CSV.

    Creates a CSV with columns: customer_id, churn_probability,
    risk_segment, top_driver_1, top_driver_2, top_driver_3,
    retention_strategy, annual_revenue.

    Args:
        df: Original customer DataFrame (used for customer_id and revenue).
        scored_customers: List of scored customers with probabilities and
            risk segments.
        customer_strategies: List of customer strategy recommendations.
        explanations: List of customer explanations with ranked features.
        revenue_column: Name of the revenue column in *df*.
        output_path: File path where the CSV will be saved.
        id_column: Name of the customer ID column in *df*.
            Defaults to ``"customer_id"``.

    Returns:
        The *output_path* where the CSV was written.
    """
    # Build lookup dicts keyed by customer_index
    strategy_lookup = {cs.customer_index: cs for cs in customer_strategies}
    explanation_lookup = {ex.customer_index: ex for ex in explanations}

    rows: list[dict] = []
    for sc in scored_customers:
        idx = sc.customer_index

        # Customer ID from original DataFrame
        if id_column in df.columns and idx < len(df):
            cust_id = df.iloc[idx][id_column]
        else:
            cust_id = idx

        # Top 3 drivers from SHAP explanation
        explanation = explanation_lookup.get(idx)
        top_drivers = ["", "", ""]
        if explanation and explanation.ranked_features:
            for i, (feat_name, _) in enumerate(explanation.ranked_features[:3]):
                top_drivers[i] = feat_name

        # Retention strategy
        strategy = strategy_lookup.get(idx)
        retention_strategy = strategy.retention_strategy if strategy else ""

        # Annual revenue from original DataFrame
        if revenue_column in df.columns and idx < len(df):
            annual_revenue = df.iloc[idx][revenue_column]
        else:
            annual_revenue = None

        rows.append(
            {
                "customer_id": cust_id,
                "churn_probability": sc.churn_probability,
                "risk_segment": sc.risk_segment,
                "top_driver_1": top_drivers[0],
                "top_driver_2": top_drivers[1],
                "top_driver_3": top_drivers[2],
                "retention_strategy": retention_strategy,
                "annual_revenue": annual_revenue,
            }
        )

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    logger.info(
        "Customer summary CSV exported (%d rows) to %s.",
        len(rows),
        output_path,
    )
    return output_path


def export_feature_importance_csv(
    global_importance: list[tuple[str, float]],
    output_path: str,
) -> str:
    """Export global feature importance as a CSV.

    Creates a CSV with columns: feature_name, mean_abs_shap.

    Args:
        global_importance: List of (feature_name, mean_abs_shap) tuples
            as returned by
            :func:`src.explainability.get_global_feature_importance`.
        output_path: File path where the CSV will be saved.

    Returns:
        The *output_path* where the CSV was written.
    """
    importance_df = pd.DataFrame(
        global_importance, columns=["feature_name", "mean_abs_shap"]
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    logger.info(
        "Feature importance CSV exported (%d features) to %s.",
        len(global_importance),
        output_path,
    )
    return output_path


def export_impact_summary_json(
    impact_summary: ImpactSummary,
    output_path: str,
) -> str:
    """Export the revenue impact summary as a JSON file.

    Serialises all :class:`ImpactSummary` fields to JSON.

    Args:
        impact_summary: The computed impact summary.
        output_path: File path where the JSON will be saved.

    Returns:
        The *output_path* where the JSON was written.
    """
    data = asdict(impact_summary)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    logger.info("Impact summary JSON exported to %s.", output_path)
    return output_path

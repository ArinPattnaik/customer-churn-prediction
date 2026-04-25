"""Revenue Impact Analysis module.

Calculates the projected revenue impact of retention efforts by
analysing revenue at risk from high-risk customers, estimating
projected savings from targeted retention campaigns, and producing
an overall impact summary with ROI metrics.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImpactSummary:
    """Summary of the revenue impact of retention efforts.

    Attributes:
        total_revenue_at_risk: Total annual revenue from High-risk
            customers (excluding records with missing revenue).
        targeted_customer_count: Number of high-risk customers
            targeted by the retention campaign.
        projected_revenue_saved: Estimated revenue saved by the
            retention campaign.
        retention_roi: Return on investment ratio computed as
            ``projected_revenue_saved / estimated_retention_cost``.
    """

    total_revenue_at_risk: float
    targeted_customer_count: int
    projected_revenue_saved: float
    retention_roi: float


def calculate_revenue_at_risk(
    df: pd.DataFrame,
    risk_segments: list[str],
    revenue_column: str = "annual_revenue",
) -> float:
    """Sum revenue for High-risk customers.

    Records with missing (NaN) revenue are excluded from the total and
    a warning is logged for each excluded record.

    Args:
        df: Customer DataFrame containing the revenue column.
        risk_segments: List of risk segment labels, same length as
            ``len(df)``.
        revenue_column: Name of the column holding annual revenue.

    Returns:
        Total revenue at risk (sum of non-NaN revenue for High-risk
        customers).
    """
    if revenue_column not in df.columns:
        raise ValueError(
            f"Revenue column '{revenue_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    total = 0.0
    for idx, (seg, rev) in enumerate(
        zip(risk_segments, df[revenue_column])
    ):
        if seg != "High":
            continue
        if pd.isna(rev):
            logger.warning(
                "Record %d: missing revenue for High-risk customer — "
                "excluded from revenue-at-risk calculation.",
                idx,
            )
            continue
        total += float(rev)

    logger.info("Total revenue at risk from High-risk customers: %.2f", total)
    return total


def calculate_projected_savings(
    df: pd.DataFrame,
    risk_segments: list[str],
    probabilities: list[float],
    retention_target_pct: float = 0.10,
    retention_success_rate: float = 0.50,
    revenue_column: str = "annual_revenue",
) -> tuple[float, int]:
    """Calculate projected revenue saved by targeting top N% of high-risk customers.

    The function identifies High-risk customers, ranks them by churn
    probability in descending order, selects the top
    *retention_target_pct* fraction, and estimates the revenue that
    could be saved assuming *retention_success_rate* of those customers
    are successfully retained.

    Args:
        df: Customer DataFrame containing the revenue column.
        risk_segments: List of risk segment labels.
        probabilities: List of churn probabilities for each customer.
        retention_target_pct: Fraction of high-risk customers to target
            (default ``0.10`` = top 10%).
        retention_success_rate: Assumed success rate for retention
            efforts (default ``0.50`` = 50%).
        revenue_column: Name of the column holding annual revenue.

    Returns:
        Tuple of (projected_revenue_saved, targeted_customer_count).
    """
    if revenue_column not in df.columns:
        raise ValueError(
            f"Revenue column '{revenue_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Collect high-risk customers with their probabilities and revenue
    high_risk_customers: list[tuple[int, float, float]] = []
    for idx, (seg, prob, rev) in enumerate(
        zip(risk_segments, probabilities, df[revenue_column])
    ):
        if seg != "High":
            continue
        if pd.isna(rev):
            logger.warning(
                "Record %d: missing revenue for High-risk customer — "
                "excluded from projected savings calculation.",
                idx,
            )
            continue
        high_risk_customers.append((idx, float(prob), float(rev)))

    if not high_risk_customers:
        logger.info("No High-risk customers with valid revenue found.")
        return 0.0, 0

    # Sort by churn probability descending to target the riskiest first
    high_risk_customers.sort(key=lambda x: x[1], reverse=True)

    # Select top N% (at least 1 if there are any customers)
    target_count = max(1, math.ceil(len(high_risk_customers) * retention_target_pct))
    targeted = high_risk_customers[:target_count]

    # Projected savings = sum of targeted revenue * success rate
    targeted_revenue = sum(rev for _, _, rev in targeted)
    projected_saved = targeted_revenue * retention_success_rate

    logger.info(
        "Targeting %d of %d High-risk customers (%.0f%%). "
        "Projected revenue saved: %.2f (success rate: %.0f%%).",
        target_count,
        len(high_risk_customers),
        retention_target_pct * 100,
        projected_saved,
        retention_success_rate * 100,
    )

    return projected_saved, target_count


def build_impact_summary(
    total_revenue_at_risk: float,
    targeted_customer_count: int,
    projected_revenue_saved: float,
    estimated_retention_cost: float,
) -> ImpactSummary:
    """Build the revenue impact summary.

    Enforces the boundary property: ``projected_revenue_saved`` is
    clamped to be at most ``total_revenue_at_risk``.

    The retention ROI is computed as::

        retention_roi = projected_revenue_saved / estimated_retention_cost

    If *estimated_retention_cost* is zero or negative, ``retention_roi``
    is set to ``0.0`` and a warning is logged.

    Args:
        total_revenue_at_risk: Total revenue from High-risk customers.
        targeted_customer_count: Number of customers targeted.
        projected_revenue_saved: Estimated revenue saved.
        estimated_retention_cost: Total cost of the retention campaign.

    Returns:
        :class:`ImpactSummary` with all fields populated.
    """
    # Enforce boundary: projected savings cannot exceed revenue at risk
    if projected_revenue_saved > total_revenue_at_risk:
        logger.warning(
            "Projected revenue saved (%.2f) exceeds total revenue at risk "
            "(%.2f). Clamping to total revenue at risk.",
            projected_revenue_saved,
            total_revenue_at_risk,
        )
        projected_revenue_saved = total_revenue_at_risk

    # Compute ROI
    if estimated_retention_cost <= 0:
        logger.warning(
            "Estimated retention cost is %.2f (zero or negative). "
            "Setting retention ROI to 0.0.",
            estimated_retention_cost,
        )
        retention_roi = 0.0
    else:
        retention_roi = projected_revenue_saved / estimated_retention_cost

    summary = ImpactSummary(
        total_revenue_at_risk=total_revenue_at_risk,
        targeted_customer_count=targeted_customer_count,
        projected_revenue_saved=projected_revenue_saved,
        retention_roi=retention_roi,
    )

    logger.info(
        "Impact summary — Revenue at risk: %.2f, Targeted: %d, "
        "Projected saved: %.2f, ROI: %.2f.",
        summary.total_revenue_at_risk,
        summary.targeted_customer_count,
        summary.projected_revenue_saved,
        summary.retention_roi,
    )

    return summary

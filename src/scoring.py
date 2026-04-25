"""Churn Probability Scoring module.

Produces churn probability scores for each customer, assigns risk
segments based on configurable thresholds, and combines results
into ScoredCustomer objects.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "High": 0.7,
    "Medium": 0.4,
}


@dataclass
class ScoredCustomer:
    """Container for a scored customer with churn probability and risk segment."""

    customer_index: int
    churn_probability: float
    risk_segment: str  # "High", "Medium", "Low"


def score_customers(model: object, X: np.ndarray) -> list[float]:
    """Produce churn probability scores in [0.0, 1.0] for each record.

    Uses the model's ``predict_proba`` method and returns the probability
    of the positive (churn) class.

    Args:
        model: A fitted estimator with a ``predict_proba`` method.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        List of churn probabilities, one per row in *X*.

    Raises:
        RuntimeError: If the model does not expose ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Model does not support predict_proba. "
            "Ensure the model is fitted and supports probability estimation."
        )

    probabilities = model.predict_proba(X)[:, 1].tolist()
    logger.info("Scored %d customers. Probability range: [%.4f, %.4f]",
                len(probabilities),
                min(probabilities),
                max(probabilities))
    return probabilities


def assign_risk_segments(
    probabilities: list[float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Assign each probability to exactly one Risk_Segment.

    Thresholds define the *minimum* probability for a segment.  Segments
    are evaluated from highest threshold to lowest so that each
    probability falls into the first matching bucket:

    * **High** – probability >= high threshold (default 0.7)
    * **Medium** – probability >= medium threshold (default 0.4)
    * **Low** – everything else (probability < medium threshold)

    Args:
        probabilities: List of churn probabilities in [0.0, 1.0].
        thresholds: Optional mapping of segment name to minimum
            probability.  Defaults to ``{"High": 0.7, "Medium": 0.4}``.

    Returns:
        List of segment labels, same length as *probabilities*.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    high_threshold = thresholds.get("High", 0.7)
    medium_threshold = thresholds.get("Medium", 0.4)

    segments: list[str] = []
    for prob in probabilities:
        if prob >= high_threshold:
            segments.append("High")
        elif prob >= medium_threshold:
            segments.append("Medium")
        else:
            segments.append("Low")

    # Log segment distribution
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for seg in segments:
        counts[seg] += 1
    logger.info(
        "Risk segment distribution — High: %d, Medium: %d, Low: %d",
        counts["High"],
        counts["Medium"],
        counts["Low"],
    )

    return segments


def build_scored_results(
    probabilities: list[float],
    segments: list[str],
) -> list[ScoredCustomer]:
    """Combine probabilities and segments into ScoredCustomer objects.

    Args:
        probabilities: List of churn probabilities.
        segments: List of risk segment labels (same length as
            *probabilities*).

    Returns:
        List of :class:`ScoredCustomer` objects, one per customer.

    Raises:
        ValueError: If *probabilities* and *segments* differ in length.
    """
    if len(probabilities) != len(segments):
        raise ValueError(
            f"Length mismatch: probabilities ({len(probabilities)}) "
            f"vs segments ({len(segments)})."
        )

    results = [
        ScoredCustomer(
            customer_index=idx,
            churn_probability=prob,
            risk_segment=seg,
        )
        for idx, (prob, seg) in enumerate(zip(probabilities, segments))
    ]
    logger.info("Built %d scored customer results.", len(results))
    return results

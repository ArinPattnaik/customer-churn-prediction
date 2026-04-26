"""Retention Strategy Recommendation module.

Maps risk segments and top churn drivers to retention strategies
using a configurable YAML mapping. Provides single-customer and
batch recommendation functions.
"""

import logging
from dataclasses import dataclass

import yaml

from src.explainability import CustomerExplanation
from src.scoring import ScoredCustomer

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "General Outreach"


@dataclass
class CustomerStrategy:
    """Retention strategy recommendation for a single customer.

    Attributes:
        customer_index: Row index of the customer in the dataset.
        risk_segment: The customer's risk segment (High, Medium, Low).
        top_driver: The top SHAP-derived churn driver feature name.
        retention_strategy: The recommended retention strategy.
    """

    customer_index: int
    risk_segment: str
    top_driver: str
    retention_strategy: str


def load_strategy_mapping(config_path: str) -> dict[tuple[str, str], str]:
    """Load (Risk_Segment, top_driver) → Retention_Strategy mapping from YAML.

    The YAML file is expected to have the structure::

        strategies:
          - risk_segment: High
            driver: monthly_charges
            strategy: "Offer discounted plan or loyalty pricing"
          ...
        default_strategy: "General Outreach"

    Only the ``strategies`` list is loaded into the mapping dict.
    The ``default_strategy`` value is handled separately by
    :func:`recommend_strategy`.

    Args:
        config_path: Path to the strategies YAML file.

    Returns:
        Dictionary mapping (risk_segment, driver) tuples to strategy
        strings.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    mapping: dict[tuple[str, str], str] = {}
    for entry in config.get("strategies", []):
        key = (entry["risk_segment"], entry["driver"])
        mapping[key] = entry["strategy"]

    logger.info(
        "Loaded %d strategy mappings from %s.", len(mapping), config_path
    )
    return mapping


def _clean_feature_name(name: str) -> str:
    """Strip sklearn ColumnTransformer prefixes from feature names.

    Handles prefixes like ``num__``, ``cat__``, and encoded suffixes
    like ``cat__contract_type_Month-to-month`` → ``contract_type``.
    """
    import re

    # Remove num__ or cat__ prefix
    cleaned = re.sub(r"^(num__|cat__)", "", name)
    # For one-hot encoded features, take only the base column name
    # e.g. "contract_type_Month-to-month" → "contract_type"
    # But don't strip if the original column name has underscores
    # We match against known patterns: if there's a suffix after the
    # last underscore that contains spaces or hyphens, it's likely encoded
    return cleaned


def recommend_strategy(
    risk_segment: str,
    top_driver: str,
    mapping: dict[tuple[str, str], str],
    default_strategy: str = DEFAULT_STRATEGY,
) -> str:
    """Look up the retention strategy for a given segment and driver.

    Strips sklearn preprocessing prefixes (``num__``, ``cat__``) from
    the driver name before lookup, and also tries matching the base
    column name for one-hot encoded features.

    Falls back to *default_strategy* when the (risk_segment, top_driver)
    pair is not found in the mapping.

    Args:
        risk_segment: The customer's risk segment.
        top_driver: The top churn driver feature name.
        mapping: Strategy mapping from :func:`load_strategy_mapping`.
        default_strategy: Fallback strategy when no match is found.
            Defaults to ``"General Outreach"``.

    Returns:
        The matched retention strategy string, or *default_strategy*.
    """
    # Try exact match first
    if (risk_segment, top_driver) in mapping:
        return mapping[(risk_segment, top_driver)]

    # Try with cleaned feature name (strip num__/cat__ prefix)
    cleaned = _clean_feature_name(top_driver)
    if (risk_segment, cleaned) in mapping:
        return mapping[(risk_segment, cleaned)]

    # For one-hot encoded features like "contract_type_Month-to-month",
    # try matching just the base column name "contract_type"
    # Walk through mapping keys to find partial matches
    for (seg, driver), strategy in mapping.items():
        if seg == risk_segment and cleaned.startswith(driver):
            return strategy

    logger.debug(
        "No strategy mapping for (%s, %s). Using default: '%s'.",
        risk_segment,
        top_driver,
        default_strategy,
    )
    return default_strategy


def recommend_batch(
    scored_customers: list[ScoredCustomer],
    explanations: list[CustomerExplanation],
    mapping: dict[tuple[str, str], str],
) -> list[CustomerStrategy]:
    """Assign a retention strategy to every customer in the batch.

    For each customer, the top churn driver is extracted from the
    corresponding :class:`CustomerExplanation` (the first entry in
    ``ranked_features``). The strategy is then looked up via
    :func:`recommend_strategy`.

    Guarantees: ``len(result) == len(scored_customers)``.

    Args:
        scored_customers: List of scored customers with risk segments.
        explanations: List of customer explanations with ranked features.
            Must be the same length as *scored_customers*.
        mapping: Strategy mapping from :func:`load_strategy_mapping`.

    Returns:
        List of :class:`CustomerStrategy` objects, one per customer.
    """
    # Build a lookup from customer_index to explanation for efficient access
    explanation_lookup: dict[int, CustomerExplanation] = {
        exp.customer_index: exp for exp in explanations
    }

    strategies: list[CustomerStrategy] = []
    for sc in scored_customers:
        explanation = explanation_lookup.get(sc.customer_index)

        # Extract the top driver from ranked features
        if explanation and explanation.ranked_features:
            top_driver = explanation.ranked_features[0][0]
        else:
            top_driver = ""
            logger.warning(
                "No explanation found for customer %d. "
                "Using empty top_driver.",
                sc.customer_index,
            )

        strategy = recommend_strategy(
            risk_segment=sc.risk_segment,
            top_driver=top_driver,
            mapping=mapping,
        )

        strategies.append(
            CustomerStrategy(
                customer_index=sc.customer_index,
                risk_segment=sc.risk_segment,
                top_driver=top_driver,
                retention_strategy=strategy,
            )
        )

    logger.info(
        "Recommended strategies for %d customers.", len(strategies)
    )
    return strategies

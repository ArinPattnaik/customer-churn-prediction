"""SHAP Explainability module.

Computes SHAP values for model predictions, generates per-customer
and global feature importance rankings, and saves summary and
waterfall plots as PNG images.
"""

import logging
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

logger = logging.getLogger(__name__)

# Tree-based model types that support TreeExplainer
_TREE_MODEL_TYPES: tuple[type, ...] = (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

try:
    from xgboost import XGBClassifier

    _TREE_MODEL_TYPES = (*_TREE_MODEL_TYPES, XGBClassifier)
except ImportError:  # pragma: no cover
    pass

try:
    from lightgbm import LGBMClassifier

    _TREE_MODEL_TYPES = (*_TREE_MODEL_TYPES, LGBMClassifier)
except ImportError:  # pragma: no cover
    pass


@dataclass
class CustomerExplanation:
    """Explanation for a single customer's churn prediction.

    Attributes:
        customer_index: Row index of the customer in the dataset.
        shap_values: SHAP values array for this customer (one per feature).
        base_value: The expected (base) value of the model output.
        ranked_features: List of (feature_name, shap_value) tuples sorted
            in descending order of absolute SHAP value.
    """

    customer_index: int
    shap_values: np.ndarray
    base_value: float
    ranked_features: list[tuple[str, float]]


def compute_shap_values(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    """Compute SHAP values for all records in *X*.

    Uses :class:`shap.TreeExplainer` when the model is a recognised
    tree-based estimator (XGBClassifier, RandomForestClassifier,
    GradientBoostingClassifier, LGBMClassifier).  Falls back to
    :class:`shap.KernelExplainer` for all other model types.

    Args:
        model: A fitted estimator with a ``predict_proba`` method.
        X: Feature matrix of shape (n_samples, n_features).
        feature_names: List of feature names matching columns of *X*.

    Returns:
        Tuple of (shap_values_matrix, base_value) where
        *shap_values_matrix* has shape (n_samples, n_features) and
        *base_value* is a scalar float.
    """
    use_kernel = True

    if isinstance(model, _TREE_MODEL_TYPES):
        logger.info("Using TreeExplainer for %s.", type(model).__name__)
        try:
            explainer = shap.TreeExplainer(model)
            shap_result = explainer(X)

            # shap_result.values may be 3-D for multi-output models;
            # take the positive-class slice (index 1) when needed.
            sv = shap_result.values
            bv = shap_result.base_values

            if sv.ndim == 3:
                sv = sv[:, :, 1]
            if isinstance(bv, np.ndarray) and bv.ndim == 2:
                bv = bv[:, 1]

            base_value = float(bv[0]) if isinstance(bv, np.ndarray) else float(bv)
            use_kernel = False
        except (ValueError, TypeError) as exc:
            logger.warning(
                "TreeExplainer failed for %s (%s). "
                "Falling back to KernelExplainer.",
                type(model).__name__,
                exc,
            )
    else:
        logger.warning(
            "Model type %s is not tree-based. Falling back to KernelExplainer.",
            type(model).__name__,
        )

    if use_kernel:
        # KernelExplainer needs a callable that returns probabilities
        # for the positive class.
        def predict_fn(data: np.ndarray) -> np.ndarray:
            return model.predict_proba(data)[:, 1]

        # Use a small background sample to keep computation tractable.
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(X)
        base_value = float(explainer.expected_value)

    shap_values_matrix = np.asarray(sv, dtype=np.float64)
    logger.info(
        "SHAP values computed. Shape: %s, base_value: %.6f",
        shap_values_matrix.shape,
        base_value,
    )
    return shap_values_matrix, base_value


def get_customer_explanation(
    shap_values: np.ndarray,
    base_value: float,
    customer_index: int,
    feature_names: list[str],
) -> CustomerExplanation:
    """Build a ranked explanation for a single customer.

    Features are ranked in descending order of absolute SHAP value so
    that the most influential drivers appear first.

    Args:
        shap_values: SHAP value matrix of shape (n_samples, n_features).
        base_value: The base (expected) value of the model output.
        customer_index: Row index of the customer to explain.
        feature_names: List of feature names matching columns.

    Returns:
        A :class:`CustomerExplanation` with ranked features.
    """
    customer_shap = shap_values[customer_index]

    # Pair each feature with its SHAP value and sort by |value| descending
    paired = list(zip(feature_names, customer_shap.tolist()))
    ranked = sorted(paired, key=lambda item: abs(item[1]), reverse=True)

    return CustomerExplanation(
        customer_index=customer_index,
        shap_values=customer_shap,
        base_value=base_value,
        ranked_features=ranked,
    )


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Compute global feature importance as mean |SHAP| per feature.

    Args:
        shap_values: SHAP value matrix of shape (n_samples, n_features).
        feature_names: List of feature names matching columns.

    Returns:
        List of (feature_name, mean_abs_shap) sorted in descending order
        of importance.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    paired = list(zip(feature_names, mean_abs.tolist()))
    ranked = sorted(paired, key=lambda item: item[1], reverse=True)

    logger.info("Global feature importance (top 5): %s", ranked[:5])
    return ranked


def save_summary_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
    path: str,
) -> None:
    """Generate and save a SHAP beeswarm summary plot as PNG.

    Args:
        shap_values: SHAP value matrix of shape (n_samples, n_features).
        feature_names: List of feature names matching columns.
        path: File path where the PNG image will be saved.
    """
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s.", path)


def save_waterfall_plot(
    explanation: CustomerExplanation,
    path: str,
) -> None:
    """Generate and save a SHAP waterfall plot for one customer as PNG.

    Args:
        explanation: A :class:`CustomerExplanation` for the customer.
        path: File path where the PNG image will be saved.
    """
    # Build a shap.Explanation object that the waterfall plot expects
    feature_names = [name for name, _ in explanation.ranked_features]
    # Re-order SHAP values to match the original feature order stored
    # in explanation.shap_values (waterfall plot handles ordering itself).
    shap_explanation = shap.Explanation(
        values=explanation.shap_values,
        base_values=explanation.base_value,
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP waterfall plot saved to %s.", path)

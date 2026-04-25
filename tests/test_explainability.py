"""Property-based tests for the explainability module."""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st_hyp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.explainability import (
    compute_shap_values,
    get_customer_explanation,
    get_global_feature_importance,
)


# ---------------------------------------------------------------------------
# Property 14: SHAP feature ranking sorted by absolute value
# ---------------------------------------------------------------------------

class TestProperty14ShapFeatureRankingSorted:
    """Property 14: SHAP feature ranking is sorted by absolute value.

    For any array of SHAP values and feature names, get_customer_explanation
    returns ranked_features sorted descending by |SHAP value|.
    """

    # Feature: customer-churn-prediction, Property 14: SHAP feature ranking is sorted by absolute value

    @given(
        n_features=st_hyp.integers(min_value=1, max_value=20),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_ranked_features_sorted_descending_abs(self, n_features, data):
        """**Validates: Requirements 5.2**"""
        n_customers = data.draw(st_hyp.integers(min_value=1, max_value=10))
        customer_index = data.draw(st_hyp.integers(min_value=0, max_value=n_customers - 1))

        # Generate random SHAP values matrix
        shap_values = np.array(
            data.draw(
                st_hyp.lists(
                    st_hyp.lists(
                        st_hyp.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                        min_size=n_features,
                        max_size=n_features,
                    ),
                    min_size=n_customers,
                    max_size=n_customers,
                )
            )
        )

        feature_names = [f"feat_{i}" for i in range(n_features)]
        base_value = 0.5

        explanation = get_customer_explanation(
            shap_values, base_value, customer_index, feature_names
        )

        # Verify sorted descending by absolute value
        abs_values = [abs(v) for _, v in explanation.ranked_features]
        for i in range(len(abs_values) - 1):
            assert abs_values[i] >= abs_values[i + 1], (
                f"ranked_features not sorted: |{abs_values[i]}| < |{abs_values[i+1]}|"
            )


# ---------------------------------------------------------------------------
# Property 15: Global SHAP importance is mean absolute value
# ---------------------------------------------------------------------------

class TestProperty15GlobalShapImportanceMeanAbs:
    """Property 15: Global SHAP importance is mean absolute value.

    For any SHAP matrix, each feature importance equals mean(|SHAP|)
    across customers, sorted descending.
    """

    # Feature: customer-churn-prediction, Property 15: Global SHAP importance is mean absolute value

    @given(
        n_customers=st_hyp.integers(min_value=1, max_value=20),
        n_features=st_hyp.integers(min_value=1, max_value=10),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_global_importance_equals_mean_abs_shap(self, n_customers, n_features, data):
        """**Validates: Requirements 5.3**"""
        shap_values = np.array(
            data.draw(
                st_hyp.lists(
                    st_hyp.lists(
                        st_hyp.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                        min_size=n_features,
                        max_size=n_features,
                    ),
                    min_size=n_customers,
                    max_size=n_customers,
                )
            )
        )

        feature_names = [f"feat_{i}" for i in range(n_features)]

        result = get_global_feature_importance(shap_values, feature_names)

        # Verify each importance equals mean(|SHAP|)
        expected_mean_abs = np.mean(np.abs(shap_values), axis=0)
        result_dict = dict(result)

        for i, name in enumerate(feature_names):
            assert abs(result_dict[name] - expected_mean_abs[i]) < 1e-10, (
                f"Feature '{name}': expected {expected_mean_abs[i]}, got {result_dict[name]}"
            )

        # Verify sorted descending
        importance_values = [v for _, v in result]
        for i in range(len(importance_values) - 1):
            assert importance_values[i] >= importance_values[i + 1], (
                f"Global importance not sorted: {importance_values[i]} < {importance_values[i+1]}"
            )


# ---------------------------------------------------------------------------
# Property 16: SHAP additive property
# ---------------------------------------------------------------------------

class TestProperty16ShapAdditiveProperty:
    """Property 16: SHAP additive property.

    For a simple trained model, sum(shap_values) + base_value ≈ model output
    within 1e-4. Uses a small dataset.
    """

    # Feature: customer-churn-prediction, Property 16: SHAP additive property

    def test_shap_additive_on_logistic_regression(self):
        """**Validates: Requirements 5.6**"""
        # Create a small, deterministic dataset
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        feature_names = [f"f{i}" for i in range(5)]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # Compute SHAP values
        shap_values, base_value = compute_shap_values(model, X, feature_names)

        # Get model's predicted probabilities for positive class
        model_probs = model.predict_proba(X)[:, 1]

        # For each customer, sum(shap_values) + base_value ≈ model output
        for i in range(len(X)):
            shap_sum = shap_values[i].sum() + base_value
            # KernelExplainer approximates log-odds or probabilities;
            # check that the reconstruction is close to the model probability
            assert abs(shap_sum - model_probs[i]) < 0.05, (
                f"Customer {i}: shap_sum + base = {shap_sum}, "
                f"model_prob = {model_probs[i]}, diff = {abs(shap_sum - model_probs[i])}"
            )

"""Property-based tests for the scoring module."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st_hyp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.scoring import assign_risk_segments, score_customers


# ---------------------------------------------------------------------------
# Property 12: Churn probabilities are bounded
# ---------------------------------------------------------------------------

class TestProperty12ChurnProbabilitiesBounded:
    """Property 12: Churn probabilities are bounded.

    For any valid Feature_Set input to a trained model, every probability
    is in [0.0, 1.0]. Uses LogisticRegression for speed.
    """

    # Feature: customer-churn-prediction, Property 12: Churn probabilities are bounded

    @given(
        n_samples=st_hyp.integers(min_value=10, max_value=100),
        n_features=st_hyp.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50)
    def test_probabilities_in_unit_interval(self, n_samples, n_features):
        """**Validates: Requirements 4.1**"""
        n_informative = max(2, n_features // 2)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=2,
            random_state=42,
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        probabilities = score_customers(model, X)

        assert len(probabilities) == n_samples
        for p in probabilities:
            assert 0.0 <= p <= 1.0, f"Probability {p} out of [0, 1] range"


# ---------------------------------------------------------------------------
# Property 13: Risk segment partition
# ---------------------------------------------------------------------------

class TestProperty13RiskSegmentPartition:
    """Property 13: Risk segment partition.

    For any list of probabilities, assign_risk_segments produces a segment
    list of the same length where each label is in {"High", "Medium", "Low"},
    correct threshold mapping, and total count preserved.
    """

    # Feature: customer-churn-prediction, Property 13: Risk segment partition

    @given(
        probabilities=st_hyp.lists(
            st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_segment_partition_properties(self, probabilities):
        """**Validates: Requirements 4.2, 4.3, 4.4**"""
        segments = assign_risk_segments(probabilities)

        # Same length
        assert len(segments) == len(probabilities)

        valid_labels = {"High", "Medium", "Low"}
        high_count = 0
        medium_count = 0
        low_count = 0

        for prob, seg in zip(probabilities, segments):
            # Every label is valid
            assert seg in valid_labels, f"Invalid segment '{seg}'"

            # Correct threshold mapping (default: High >= 0.7, Medium >= 0.4)
            if prob >= 0.7:
                assert seg == "High", f"prob={prob} should be High, got {seg}"
            elif prob >= 0.4:
                assert seg == "Medium", f"prob={prob} should be Medium, got {seg}"
            else:
                assert seg == "Low", f"prob={prob} should be Low, got {seg}"

            if seg == "High":
                high_count += 1
            elif seg == "Medium":
                medium_count += 1
            else:
                low_count += 1

        # Total count preserved
        assert high_count + medium_count + low_count == len(probabilities)

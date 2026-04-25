"""Property-based tests for the persistence module."""

import json
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st_hyp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.persistence import (
    ArtifactMetadata,
    load_metadata,
    save_metadata,
)

import joblib


# ---------------------------------------------------------------------------
# Property 11: Model serialization round-trip
# ---------------------------------------------------------------------------

class TestProperty11ModelSerializationRoundTrip:
    """Property 11: Model serialization round-trip.

    Save a trained LogisticRegression → load → predictions identical
    on same input.
    """

    # Feature: customer-churn-prediction, Property 11: Model serialization round-trip

    @given(
        n_samples=st_hyp.integers(min_value=20, max_value=100),
        n_features=st_hyp.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_model_round_trip_identical_predictions(self, n_samples, n_features):
        """**Validates: Requirements 3.6, 2.6, 10.3**"""
        n_informative = max(2, n_features // 2)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        original_preds = model.predict(X)
        original_proba = model.predict_proba(X)

        # Save and load using tempfile
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model_path = f.name

        try:
            joblib.dump(model, model_path)
            loaded_model = joblib.load(model_path)

            loaded_preds = loaded_model.predict(X)
            loaded_proba = loaded_model.predict_proba(X)

            np.testing.assert_array_equal(original_preds, loaded_preds)
            np.testing.assert_array_almost_equal(original_proba, loaded_proba)
        finally:
            os.unlink(model_path)


# ---------------------------------------------------------------------------
# Property 21: Metadata serialization round-trip
# ---------------------------------------------------------------------------

class TestProperty21MetadataSerializationRoundTrip:
    """Property 21: Metadata serialization round-trip.

    For any valid ArtifactMetadata, serialize → deserialize produces
    a dict with all required keys and equivalent values.
    """

    # Feature: customer-churn-prediction, Property 21: Metadata serialization round-trip

    @given(
        model_type=st_hyp.sampled_from(["LogisticRegression", "XGBClassifier", "RandomForest"]),
        training_date=st_hyp.from_regex(r"2\d{3}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", fullmatch=True),
        dataset_row_count=st_hyp.integers(min_value=1, max_value=100000),
        n_features=st_hyp.integers(min_value=1, max_value=20),
        data=st_hyp.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_metadata_round_trip(
        self, model_type, training_date, dataset_row_count, n_features, data
    ):
        """**Validates: Requirements 10.5, 10.2**"""
        feature_list = [f"feature_{i}" for i in range(n_features)]

        metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        evaluation_metrics = {}
        for name in metric_names:
            val = data.draw(
                st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            evaluation_metrics[name] = val

        metadata = ArtifactMetadata(
            model_type=model_type,
            training_date=training_date,
            dataset_row_count=dataset_row_count,
            feature_list=feature_list,
            evaluation_metrics=evaluation_metrics,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            save_metadata(metadata, path)
            loaded = load_metadata(path)

            # All required keys present and values equivalent
            assert loaded.model_type == metadata.model_type
            assert loaded.training_date == metadata.training_date
            assert loaded.dataset_row_count == metadata.dataset_row_count
            assert loaded.feature_list == metadata.feature_list

            for key in metric_names:
                assert abs(loaded.evaluation_metrics[key] - metadata.evaluation_metrics[key]) < 1e-10
        finally:
            os.unlink(path)

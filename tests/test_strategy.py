"""Unit tests for the strategy module."""

import os
import tempfile

import numpy as np
import pytest
import yaml

from src.explainability import CustomerExplanation
from src.scoring import ScoredCustomer
from src.strategy import (
    CustomerStrategy,
    load_strategy_mapping,
    recommend_batch,
    recommend_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_mapping() -> dict[tuple[str, str], str]:
    """A small strategy mapping for testing."""
    return {
        ("High", "monthly_charges"): "Offer discounted plan",
        ("High", "tenure_months"): "Offer long-term incentive",
        ("Medium", "contract_type"): "Propose contract upgrade",
    }


@pytest.fixture
def strategies_yaml_path(tmp_path):
    """Write a temporary strategies YAML file and return its path."""
    config = {
        "strategies": [
            {"risk_segment": "High", "driver": "monthly_charges", "strategy": "Discount"},
            {"risk_segment": "Medium", "driver": "tenure_months", "strategy": "Incentive"},
        ],
        "default_strategy": "General Outreach",
    }
    path = tmp_path / "strategies.yaml"
    with open(path, "w") as fh:
        yaml.dump(config, fh)
    return str(path)


# ---------------------------------------------------------------------------
# load_strategy_mapping
# ---------------------------------------------------------------------------

class TestLoadStrategyMapping:
    def test_loads_correct_entries(self, strategies_yaml_path):
        mapping = load_strategy_mapping(strategies_yaml_path)
        assert len(mapping) == 2
        assert mapping[("High", "monthly_charges")] == "Discount"
        assert mapping[("Medium", "tenure_months")] == "Incentive"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_strategy_mapping("/nonexistent/path/strategies.yaml")

    def test_empty_strategies_list(self, tmp_path):
        config = {"strategies": [], "default_strategy": "Fallback"}
        path = tmp_path / "empty.yaml"
        with open(path, "w") as fh:
            yaml.dump(config, fh)
        mapping = load_strategy_mapping(str(path))
        assert mapping == {}

    def test_loads_actual_config(self):
        """Verify the real config/strategies.yaml loads correctly."""
        mapping = load_strategy_mapping("config/strategies.yaml")
        assert len(mapping) == 5
        assert ("High", "monthly_charges") in mapping


# ---------------------------------------------------------------------------
# recommend_strategy
# ---------------------------------------------------------------------------

class TestRecommendStrategy:
    def test_returns_mapped_strategy(self, sample_mapping):
        result = recommend_strategy("High", "monthly_charges", sample_mapping)
        assert result == "Offer discounted plan"

    def test_returns_default_when_no_match(self, sample_mapping):
        result = recommend_strategy("Low", "unknown_feature", sample_mapping)
        assert result == "General Outreach"

    def test_custom_default_strategy(self, sample_mapping):
        result = recommend_strategy(
            "Low", "unknown", sample_mapping, default_strategy="Do Nothing"
        )
        assert result == "Do Nothing"

    def test_empty_mapping_returns_default(self):
        result = recommend_strategy("High", "monthly_charges", {})
        assert result == "General Outreach"


# ---------------------------------------------------------------------------
# recommend_batch
# ---------------------------------------------------------------------------

class TestRecommendBatch:
    def _make_scored(self, n: int) -> list[ScoredCustomer]:
        segments = ["High", "Medium", "Low"]
        return [
            ScoredCustomer(
                customer_index=i,
                churn_probability=0.5,
                risk_segment=segments[i % 3],
            )
            for i in range(n)
        ]

    def _make_explanations(self, n: int) -> list[CustomerExplanation]:
        drivers = ["monthly_charges", "tenure_months", "contract_type"]
        return [
            CustomerExplanation(
                customer_index=i,
                shap_values=np.array([0.1, 0.2]),
                base_value=0.5,
                ranked_features=[(drivers[i % 3], 0.3), ("other", 0.1)],
            )
            for i in range(n)
        ]

    def test_batch_length_matches_input(self, sample_mapping):
        scored = self._make_scored(5)
        explanations = self._make_explanations(5)
        results = recommend_batch(scored, explanations, sample_mapping)
        assert len(results) == 5

    def test_every_customer_gets_strategy(self, sample_mapping):
        scored = self._make_scored(3)
        explanations = self._make_explanations(3)
        results = recommend_batch(scored, explanations, sample_mapping)
        for r in results:
            assert isinstance(r, CustomerStrategy)
            assert isinstance(r.retention_strategy, str)
            assert len(r.retention_strategy) > 0

    def test_correct_strategy_assignment(self, sample_mapping):
        scored = [
            ScoredCustomer(customer_index=0, churn_probability=0.9, risk_segment="High"),
        ]
        explanations = [
            CustomerExplanation(
                customer_index=0,
                shap_values=np.array([0.5]),
                base_value=0.4,
                ranked_features=[("monthly_charges", 0.5)],
            ),
        ]
        results = recommend_batch(scored, explanations, sample_mapping)
        assert results[0].retention_strategy == "Offer discounted plan"
        assert results[0].top_driver == "monthly_charges"

    def test_empty_batch(self, sample_mapping):
        results = recommend_batch([], [], sample_mapping)
        assert results == []

    def test_missing_explanation_uses_empty_driver(self, sample_mapping):
        scored = [
            ScoredCustomer(customer_index=99, churn_probability=0.8, risk_segment="High"),
        ]
        # No matching explanation for customer_index=99
        explanations = []
        results = recommend_batch(scored, explanations, sample_mapping)
        assert len(results) == 1
        assert results[0].top_driver == ""
        assert results[0].retention_strategy == "General Outreach"


# ---------------------------------------------------------------------------
# Property-based tests (appended)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st_hyp


# ---------------------------------------------------------------------------
# Property 17: Strategy recommendation with default fallback
# ---------------------------------------------------------------------------

class TestProperty17StrategyRecommendationFallback:
    """Property 17: Strategy recommendation with default fallback.

    For any (segment, driver) pair and mapping, returns mapped strategy
    if exists, "General Outreach" otherwise.
    """

    # Feature: customer-churn-prediction, Property 17: Strategy recommendation with default fallback

    @given(
        segment=st_hyp.sampled_from(["High", "Medium", "Low"]),
        driver=st_hyp.sampled_from([
            "monthly_charges", "tenure_months", "contract_type",
            "num_support_tickets", "total_charges", "unknown_feature",
        ]),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_returns_mapped_or_default(self, segment, driver, data):
        """**Validates: Requirements 6.2, 6.3**"""
        # Build a random mapping with some known entries
        possible_keys = [
            ("High", "monthly_charges"),
            ("High", "tenure_months"),
            ("Medium", "contract_type"),
            ("Low", "total_charges"),
        ]
        mapping = {}
        for key in possible_keys:
            include = data.draw(st_hyp.booleans())
            if include:
                mapping[key] = f"Strategy for {key}"

        result = recommend_strategy(segment, driver, mapping)

        if (segment, driver) in mapping:
            assert result == mapping[(segment, driver)]
        else:
            assert result == "General Outreach"


# ---------------------------------------------------------------------------
# Property 18: Batch strategy completeness
# ---------------------------------------------------------------------------

class TestProperty18BatchStrategyCompleteness:
    """Property 18: Batch strategy completeness.

    For any batch of customers, result length equals input length and
    every customer gets exactly one strategy.
    """

    # Feature: customer-churn-prediction, Property 18: Batch strategy completeness

    @given(
        n_customers=st_hyp.integers(min_value=0, max_value=30),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_batch_length_and_one_strategy_each(self, n_customers, data):
        """**Validates: Requirements 6.4, 6.5**"""
        segments = ["High", "Medium", "Low"]
        drivers = ["monthly_charges", "tenure_months", "contract_type"]

        scored = []
        explanations = []
        for i in range(n_customers):
            seg = data.draw(st_hyp.sampled_from(segments))
            prob = data.draw(st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False))
            drv = data.draw(st_hyp.sampled_from(drivers))

            scored.append(
                ScoredCustomer(customer_index=i, churn_probability=prob, risk_segment=seg)
            )
            explanations.append(
                CustomerExplanation(
                    customer_index=i,
                    shap_values=np.array([0.1]),
                    base_value=0.5,
                    ranked_features=[(drv, 0.3)],
                )
            )

        mapping = {
            ("High", "monthly_charges"): "Discount plan",
            ("Medium", "tenure_months"): "Loyalty offer",
        }

        results = recommend_batch(scored, explanations, mapping)

        assert len(results) == n_customers
        for r in results:
            assert isinstance(r, CustomerStrategy)
            assert isinstance(r.retention_strategy, str)
            assert len(r.retention_strategy) > 0

"""Property-based and unit tests for the impact analysis module."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st_hyp

from src.impact import (
    ImpactSummary,
    build_impact_summary,
    calculate_projected_savings,
    calculate_revenue_at_risk,
)


# ---------------------------------------------------------------------------
# Property 19: Revenue at risk with missing data handling
# ---------------------------------------------------------------------------

class TestProperty19RevenueAtRiskMissingData:
    """Property 19: Revenue at risk calculation with missing data handling.

    For any DataFrame with revenue and NaN patterns, returns sum of
    non-NaN revenue for High-risk customers only.
    """

    # Feature: customer-churn-prediction, Property 19: Revenue at risk calculation with missing data handling

    @given(
        n_customers=st_hyp.integers(min_value=1, max_value=50),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_revenue_at_risk_sums_high_risk_non_nan(self, n_customers, data):
        """**Validates: Requirements 7.1, 7.4**"""
        segments = data.draw(
            st_hyp.lists(
                st_hyp.sampled_from(["High", "Medium", "Low"]),
                min_size=n_customers,
                max_size=n_customers,
            )
        )
        revenues = data.draw(
            st_hyp.lists(
                st_hyp.one_of(
                    st_hyp.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
                    st_hyp.none(),
                ),
                min_size=n_customers,
                max_size=n_customers,
            )
        )

        df = pd.DataFrame({"annual_revenue": revenues})

        result = calculate_revenue_at_risk(df, segments)

        # Manually compute expected
        expected = 0.0
        for seg, rev in zip(segments, revenues):
            if seg == "High" and rev is not None and not pd.isna(rev):
                expected += rev

        assert abs(result - expected) < 1e-6, (
            f"Expected {expected}, got {result}"
        )


# ---------------------------------------------------------------------------
# Property 20: Revenue impact boundary
# ---------------------------------------------------------------------------

class TestProperty20RevenueImpactBoundary:
    """Property 20: Revenue impact boundary.

    For any valid inputs, projected_revenue_saved ≤ total_revenue_at_risk.
    """

    # Feature: customer-churn-prediction, Property 20: Revenue impact boundary

    @given(
        total_revenue_at_risk=st_hyp.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
        projected_revenue_saved=st_hyp.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
        estimated_retention_cost=st_hyp.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_projected_savings_le_revenue_at_risk(
        self, total_revenue_at_risk, projected_revenue_saved, estimated_retention_cost
    ):
        """**Validates: Requirements 7.5**"""
        summary = build_impact_summary(
            total_revenue_at_risk=total_revenue_at_risk,
            targeted_customer_count=10,
            projected_revenue_saved=projected_revenue_saved,
            estimated_retention_cost=estimated_retention_cost,
        )

        assert summary.projected_revenue_saved <= summary.total_revenue_at_risk, (
            f"projected_revenue_saved ({summary.projected_revenue_saved}) > "
            f"total_revenue_at_risk ({summary.total_revenue_at_risk})"
        )


# ---------------------------------------------------------------------------
# Unit tests for impact analysis
# ---------------------------------------------------------------------------

class TestCalculateRevenueAtRiskUnit:
    """Unit tests for calculate_revenue_at_risk."""

    def test_known_revenue_data(self):
        df = pd.DataFrame({"annual_revenue": [10000.0, 20000.0, 5000.0, 15000.0]})
        segments = ["High", "Low", "High", "Medium"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == pytest.approx(15000.0)  # 10000 + 5000

    def test_all_revenue_missing(self):
        df = pd.DataFrame({"annual_revenue": [np.nan, np.nan, np.nan]})
        segments = ["High", "High", "High"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == 0.0

    def test_single_customer_high_risk(self):
        df = pd.DataFrame({"annual_revenue": [50000.0]})
        segments = ["High"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == pytest.approx(50000.0)

    def test_zero_revenue(self):
        df = pd.DataFrame({"annual_revenue": [0.0, 0.0]})
        segments = ["High", "High"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == pytest.approx(0.0)

    def test_no_high_risk_customers(self):
        df = pd.DataFrame({"annual_revenue": [10000.0, 20000.0]})
        segments = ["Low", "Medium"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == 0.0

    def test_mixed_nan_and_valid(self):
        df = pd.DataFrame({"annual_revenue": [10000.0, np.nan, 30000.0]})
        segments = ["High", "High", "High"]
        result = calculate_revenue_at_risk(df, segments)
        assert result == pytest.approx(40000.0)  # 10000 + 30000


class TestBuildImpactSummaryUnit:
    """Unit tests for build_impact_summary and ROI calculation."""

    def test_basic_roi_calculation(self):
        summary = build_impact_summary(
            total_revenue_at_risk=100000.0,
            targeted_customer_count=10,
            projected_revenue_saved=25000.0,
            estimated_retention_cost=10000.0,
        )
        assert summary.retention_roi == pytest.approx(2.5)

    def test_zero_retention_cost_roi(self):
        summary = build_impact_summary(
            total_revenue_at_risk=100000.0,
            targeted_customer_count=5,
            projected_revenue_saved=50000.0,
            estimated_retention_cost=0.0,
        )
        assert summary.retention_roi == 0.0

    def test_negative_retention_cost_roi(self):
        summary = build_impact_summary(
            total_revenue_at_risk=100000.0,
            targeted_customer_count=5,
            projected_revenue_saved=50000.0,
            estimated_retention_cost=-100.0,
        )
        assert summary.retention_roi == 0.0

    def test_clamping_projected_savings(self):
        summary = build_impact_summary(
            total_revenue_at_risk=50000.0,
            targeted_customer_count=5,
            projected_revenue_saved=100000.0,  # exceeds revenue at risk
            estimated_retention_cost=10000.0,
        )
        assert summary.projected_revenue_saved == pytest.approx(50000.0)

    def test_all_zero_values(self):
        summary = build_impact_summary(
            total_revenue_at_risk=0.0,
            targeted_customer_count=0,
            projected_revenue_saved=0.0,
            estimated_retention_cost=0.0,
        )
        assert summary.total_revenue_at_risk == 0.0
        assert summary.projected_revenue_saved == 0.0
        assert summary.retention_roi == 0.0


class TestCalculateProjectedSavingsUnit:
    """Unit tests for calculate_projected_savings."""

    def test_basic_projected_savings(self):
        df = pd.DataFrame({"annual_revenue": [100000.0, 50000.0, 80000.0]})
        segments = ["High", "High", "Low"]
        probabilities = [0.9, 0.8, 0.2]

        saved, count = calculate_projected_savings(
            df, segments, probabilities,
            retention_target_pct=1.0,
            retention_success_rate=0.5,
        )
        # Both High-risk customers targeted: (100000 + 50000) * 0.5 = 75000
        assert saved == pytest.approx(75000.0)
        assert count == 2

    def test_no_high_risk_returns_zero(self):
        df = pd.DataFrame({"annual_revenue": [10000.0]})
        segments = ["Low"]
        probabilities = [0.1]

        saved, count = calculate_projected_savings(df, segments, probabilities)
        assert saved == 0.0
        assert count == 0

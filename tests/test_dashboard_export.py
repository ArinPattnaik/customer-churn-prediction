"""Tests for the dashboard export module."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.dashboard_export import (
    export_customer_summary_csv,
    export_feature_importance_csv,
    export_impact_summary_json,
)
from src.explainability import CustomerExplanation
from src.impact import ImpactSummary
from src.scoring import ScoredCustomer
from src.strategy import CustomerStrategy


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "annual_revenue": [10000.0, 25000.0, 5000.0],
            "tenure_months": [12, 36, 6],
        }
    )


@pytest.fixture
def scored_customers():
    return [
        ScoredCustomer(customer_index=0, churn_probability=0.85, risk_segment="High"),
        ScoredCustomer(customer_index=1, churn_probability=0.55, risk_segment="Medium"),
        ScoredCustomer(customer_index=2, churn_probability=0.20, risk_segment="Low"),
    ]


@pytest.fixture
def explanations():
    return [
        CustomerExplanation(
            customer_index=0,
            shap_values=np.array([0.3, -0.2, 0.1]),
            base_value=0.5,
            ranked_features=[
                ("tenure_months", 0.3),
                ("monthly_charges", -0.2),
                ("total_charges", 0.1),
            ],
        ),
        CustomerExplanation(
            customer_index=1,
            shap_values=np.array([0.1, 0.05, -0.15]),
            base_value=0.5,
            ranked_features=[
                ("total_charges", -0.15),
                ("tenure_months", 0.1),
                ("monthly_charges", 0.05),
            ],
        ),
        CustomerExplanation(
            customer_index=2,
            shap_values=np.array([-0.1, 0.05, 0.02]),
            base_value=0.5,
            ranked_features=[
                ("tenure_months", -0.1),
                ("monthly_charges", 0.05),
                ("total_charges", 0.02),
            ],
        ),
    ]


@pytest.fixture
def customer_strategies():
    return [
        CustomerStrategy(
            customer_index=0,
            risk_segment="High",
            top_driver="tenure_months",
            retention_strategy="Offer long-term contract incentive",
        ),
        CustomerStrategy(
            customer_index=1,
            risk_segment="Medium",
            top_driver="total_charges",
            retention_strategy="Send targeted promotional offer",
        ),
        CustomerStrategy(
            customer_index=2,
            risk_segment="Low",
            top_driver="tenure_months",
            retention_strategy="General Outreach",
        ),
    ]


class TestExportCustomerSummaryCsv:
    """Tests for export_customer_summary_csv."""

    def test_creates_csv_with_correct_columns(
        self, tmp_path, sample_df, scored_customers, customer_strategies, explanations
    ):
        output_path = str(tmp_path / "dashboards" / "customer_summary.csv")
        export_customer_summary_csv(
            df=sample_df,
            scored_customers=scored_customers,
            customer_strategies=customer_strategies,
            explanations=explanations,
            revenue_column="annual_revenue",
            output_path=output_path,
        )

        result = pd.read_csv(output_path)
        expected_columns = [
            "customer_id",
            "churn_probability",
            "risk_segment",
            "top_driver_1",
            "top_driver_2",
            "top_driver_3",
            "retention_strategy",
            "annual_revenue",
        ]
        assert list(result.columns) == expected_columns

    def test_correct_row_count(
        self, tmp_path, sample_df, scored_customers, customer_strategies, explanations
    ):
        output_path = str(tmp_path / "summary.csv")
        export_customer_summary_csv(
            df=sample_df,
            scored_customers=scored_customers,
            customer_strategies=customer_strategies,
            explanations=explanations,
            revenue_column="annual_revenue",
            output_path=output_path,
        )

        result = pd.read_csv(output_path)
        assert len(result) == 3

    def test_correct_values(
        self, tmp_path, sample_df, scored_customers, customer_strategies, explanations
    ):
        output_path = str(tmp_path / "summary.csv")
        export_customer_summary_csv(
            df=sample_df,
            scored_customers=scored_customers,
            customer_strategies=customer_strategies,
            explanations=explanations,
            revenue_column="annual_revenue",
            output_path=output_path,
        )

        result = pd.read_csv(output_path)

        # First row
        assert result.iloc[0]["customer_id"] == "C001"
        assert result.iloc[0]["churn_probability"] == pytest.approx(0.85)
        assert result.iloc[0]["risk_segment"] == "High"
        assert result.iloc[0]["top_driver_1"] == "tenure_months"
        assert result.iloc[0]["top_driver_2"] == "monthly_charges"
        assert result.iloc[0]["top_driver_3"] == "total_charges"
        assert result.iloc[0]["retention_strategy"] == "Offer long-term contract incentive"
        assert result.iloc[0]["annual_revenue"] == pytest.approx(10000.0)

    def test_returns_output_path(
        self, tmp_path, sample_df, scored_customers, customer_strategies, explanations
    ):
        output_path = str(tmp_path / "summary.csv")
        returned = export_customer_summary_csv(
            df=sample_df,
            scored_customers=scored_customers,
            customer_strategies=customer_strategies,
            explanations=explanations,
            revenue_column="annual_revenue",
            output_path=output_path,
        )
        assert returned == output_path

    def test_creates_parent_directories(
        self, tmp_path, sample_df, scored_customers, customer_strategies, explanations
    ):
        output_path = str(tmp_path / "nested" / "dir" / "summary.csv")
        export_customer_summary_csv(
            df=sample_df,
            scored_customers=scored_customers,
            customer_strategies=customer_strategies,
            explanations=explanations,
            revenue_column="annual_revenue",
            output_path=output_path,
        )
        assert os.path.exists(output_path)


class TestExportFeatureImportanceCsv:
    """Tests for export_feature_importance_csv."""

    def test_creates_csv_with_correct_columns(self, tmp_path):
        importance = [
            ("tenure_months", 0.35),
            ("monthly_charges", 0.25),
            ("total_charges", 0.10),
        ]
        output_path = str(tmp_path / "importance.csv")
        export_feature_importance_csv(importance, output_path)

        result = pd.read_csv(output_path)
        assert list(result.columns) == ["feature_name", "mean_abs_shap"]

    def test_correct_values(self, tmp_path):
        importance = [
            ("tenure_months", 0.35),
            ("monthly_charges", 0.25),
        ]
        output_path = str(tmp_path / "importance.csv")
        export_feature_importance_csv(importance, output_path)

        result = pd.read_csv(output_path)
        assert len(result) == 2
        assert result.iloc[0]["feature_name"] == "tenure_months"
        assert result.iloc[0]["mean_abs_shap"] == pytest.approx(0.35)
        assert result.iloc[1]["feature_name"] == "monthly_charges"
        assert result.iloc[1]["mean_abs_shap"] == pytest.approx(0.25)

    def test_empty_importance_list(self, tmp_path):
        output_path = str(tmp_path / "importance.csv")
        export_feature_importance_csv([], output_path)

        result = pd.read_csv(output_path)
        assert len(result) == 0
        assert list(result.columns) == ["feature_name", "mean_abs_shap"]

    def test_returns_output_path(self, tmp_path):
        output_path = str(tmp_path / "importance.csv")
        returned = export_feature_importance_csv(
            [("feat", 0.1)], output_path
        )
        assert returned == output_path


class TestExportImpactSummaryJson:
    """Tests for export_impact_summary_json."""

    def test_creates_json_with_correct_fields(self, tmp_path):
        summary = ImpactSummary(
            total_revenue_at_risk=100000.0,
            targeted_customer_count=10,
            projected_revenue_saved=25000.0,
            retention_roi=2.5,
        )
        output_path = str(tmp_path / "impact.json")
        export_impact_summary_json(summary, output_path)

        with open(output_path, "r") as fh:
            data = json.load(fh)

        assert data["total_revenue_at_risk"] == pytest.approx(100000.0)
        assert data["targeted_customer_count"] == 10
        assert data["projected_revenue_saved"] == pytest.approx(25000.0)
        assert data["retention_roi"] == pytest.approx(2.5)

    def test_json_has_all_keys(self, tmp_path):
        summary = ImpactSummary(
            total_revenue_at_risk=0.0,
            targeted_customer_count=0,
            projected_revenue_saved=0.0,
            retention_roi=0.0,
        )
        output_path = str(tmp_path / "impact.json")
        export_impact_summary_json(summary, output_path)

        with open(output_path, "r") as fh:
            data = json.load(fh)

        expected_keys = {
            "total_revenue_at_risk",
            "targeted_customer_count",
            "projected_revenue_saved",
            "retention_roi",
        }
        assert set(data.keys()) == expected_keys

    def test_returns_output_path(self, tmp_path):
        summary = ImpactSummary(
            total_revenue_at_risk=50000.0,
            targeted_customer_count=5,
            projected_revenue_saved=10000.0,
            retention_roi=1.0,
        )
        output_path = str(tmp_path / "impact.json")
        returned = export_impact_summary_json(summary, output_path)
        assert returned == output_path

    def test_creates_parent_directories(self, tmp_path):
        summary = ImpactSummary(
            total_revenue_at_risk=50000.0,
            targeted_customer_count=5,
            projected_revenue_saved=10000.0,
            retention_roi=1.0,
        )
        output_path = str(tmp_path / "nested" / "dir" / "impact.json")
        export_impact_summary_json(summary, output_path)
        assert os.path.exists(output_path)

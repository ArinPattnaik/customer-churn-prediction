"""Property-based and unit tests for the ingestion module."""

import os
import stat
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st_hyp

from src.ingestion import load_csv, log_data_summary, validate_schema


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for generating a list of unique column names
column_name_st = st_hyp.text(
    alphabet=st_hyp.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=8,
)

unique_columns_st = st_hyp.lists(column_name_st, min_size=1, max_size=10, unique=True)


# ---------------------------------------------------------------------------
# Property 1: Schema validation completeness
# ---------------------------------------------------------------------------

class TestProperty1SchemaValidationCompleteness:
    """Property 1: Schema validation completeness.

    For any DataFrame and any set of required columns, validate_schema
    passes iff every required column is present. When validation fails,
    the error message contains every missing column name.
    """

    # Feature: customer-churn-prediction, Property 1: Schema validation completeness

    @given(
        all_columns=st_hyp.lists(
            st_hyp.text(
                alphabet=st_hyp.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
                min_size=1,
                max_size=6,
            ),
            min_size=2,
            max_size=10,
            unique=True,
        ),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_validate_schema_passes_iff_all_required_present(self, all_columns, data):
        """**Validates: Requirements 1.2, 1.3**"""
        # Pick a random subset of all_columns to be the DataFrame columns
        df_columns = data.draw(
            st_hyp.lists(
                st_hyp.sampled_from(all_columns),
                min_size=0,
                max_size=len(all_columns),
                unique=True,
            )
        )
        # Pick a random subset of all_columns to be the required columns
        required_columns = data.draw(
            st_hyp.lists(
                st_hyp.sampled_from(all_columns),
                min_size=1,
                max_size=len(all_columns),
                unique=True,
            )
        )

        df = pd.DataFrame({col: [1] for col in df_columns})
        missing = [c for c in required_columns if c not in df_columns]

        if not missing:
            # Should pass without error
            validate_schema(df, required_columns)
        else:
            # Should raise ValueError mentioning every missing column
            with pytest.raises(ValueError) as exc_info:
                validate_schema(df, required_columns)
            error_msg = str(exc_info.value)
            for col in missing:
                assert col in error_msg, (
                    f"Missing column '{col}' not mentioned in error: {error_msg}"
                )


# ---------------------------------------------------------------------------
# Property 2: Data summary accuracy
# ---------------------------------------------------------------------------

class TestProperty2DataSummaryAccuracy:
    """Property 2: Data summary accuracy.

    For any valid DataFrame, log_data_summary returns correct row_count,
    column_count, and missing_values_per_column.
    """

    # Feature: customer-churn-prediction, Property 2: Data summary accuracy

    @given(
        n_rows=st_hyp.integers(min_value=0, max_value=50),
        n_cols=st_hyp.integers(min_value=1, max_value=10),
        data=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_summary_matches_dataframe(self, n_rows, n_cols, data):
        """**Validates: Requirements 1.5**"""
        col_names = [f"col_{i}" for i in range(n_cols)]

        # Build a DataFrame with random NaN patterns
        df_data = {}
        for col in col_names:
            values = data.draw(
                st_hyp.lists(
                    st_hyp.one_of(
                        st_hyp.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
                        st_hyp.none(),
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
            df_data[col] = values

        df = pd.DataFrame(df_data)
        summary = log_data_summary(df)

        assert summary["row_count"] == len(df)
        assert summary["column_count"] == len(df.columns)

        for col in col_names:
            expected_missing = df[col].isnull().sum()
            assert summary["missing_values_per_column"][col] == expected_missing


# ---------------------------------------------------------------------------
# Unit tests for ingestion module
# ---------------------------------------------------------------------------

class TestLoadCsv:
    """Unit tests for load_csv."""

    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "valid.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_path, index=False)

        result = load_csv(str(csv_path))
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_csv("/nonexistent/path/data.csv")

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows does not support removing read permission via chmod",
    )
    def test_load_unreadable_file_raises(self, tmp_path):
        csv_path = tmp_path / "locked.csv"
        csv_path.write_text("a,b\n1,2\n")
        # Remove read permission
        os.chmod(str(csv_path), 0o000)
        try:
            with pytest.raises(PermissionError, match="Permission denied"):
                load_csv(str(csv_path))
        finally:
            # Restore permissions for cleanup
            os.chmod(str(csv_path), stat.S_IRUSR | stat.S_IWUSR)


class TestValidateSchema:
    """Unit tests for validate_schema."""

    def test_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        validate_schema(df, ["a", "b", "c"])

    def test_some_columns_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="b"):
            validate_schema(df, ["a", "b", "c"])

    def test_all_columns_missing(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df, ["a", "b"])
        msg = str(exc_info.value)
        assert "a" in msg
        assert "b" in msg


class TestLogDataSummary:
    """Unit tests for log_data_summary."""

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
        summary = log_data_summary(df)
        assert summary["row_count"] == 0
        assert summary["column_count"] == 2
        assert summary["missing_values_per_column"]["a"] == 0

    def test_no_missing_values(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        summary = log_data_summary(df)
        assert summary["row_count"] == 3
        assert summary["missing_values_per_column"]["x"] == 0
        assert summary["missing_values_per_column"]["y"] == 0

    def test_mixed_missing_values(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        summary = log_data_summary(df)
        assert summary["row_count"] == 3
        assert summary["missing_values_per_column"]["a"] == 1
        assert summary["missing_values_per_column"]["b"] == 2

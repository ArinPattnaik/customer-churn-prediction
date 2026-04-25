"""Property-based tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st_hyp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.preprocessing import build_preprocessing_pipeline, fit_transform


# ---------------------------------------------------------------------------
# Property 3: Standardization produces zero mean and unit variance
# ---------------------------------------------------------------------------

class TestProperty3StandardizationZeroMeanUnitVariance:
    """Property 3: Standardization produces zero mean and unit variance.

    For any numerical array with at least two distinct values, after
    StandardScaler, mean is within 1e-6 of 0.0 and std within 1e-6 of 1.0.
    """

    # Feature: customer-churn-prediction, Property 3: Standardization produces zero mean and unit variance

    @given(
        values=st_hyp.lists(
            st_hyp.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_standard_scaler_zero_mean_unit_var(self, values):
        """**Validates: Requirements 2.2**"""
        arr = np.array(values).reshape(-1, 1)
        # Need at least 2 distinct values with meaningful difference
        assume(len(set(values)) >= 2)
        assume(np.std(values) > 1e-12)

        scaler = StandardScaler()
        transformed = scaler.fit_transform(arr)

        assert abs(transformed.mean()) < 1e-6, f"Mean {transformed.mean()} not near 0"
        assert abs(transformed.std() - 1.0) < 1e-6, f"Std {transformed.std()} not near 1"


# ---------------------------------------------------------------------------
# Property 4: Numerical imputation uses median
# ---------------------------------------------------------------------------

class TestProperty4NumericalImputationMedian:
    """Property 4: Numerical imputation uses median.

    For any numerical column with NaN, after SimpleImputer(strategy='median'),
    NaN positions contain the median of original non-NaN values.
    """

    # Feature: customer-churn-prediction, Property 4: Numerical imputation uses median

    @given(
        non_nan_values=st_hyp.lists(
            st_hyp.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
        nan_positions=st_hyp.data(),
    )
    @settings(max_examples=100)
    def test_imputed_nans_contain_median(self, non_nan_values, nan_positions):
        """**Validates: Requirements 2.3**"""
        n_total = len(non_nan_values) + 1  # at least one NaN
        # Build array with non-NaN values and at least one NaN
        values = list(non_nan_values) + [np.nan]
        # Optionally add more NaNs
        extra_nans = nan_positions.draw(st_hyp.integers(min_value=0, max_value=5))
        values.extend([np.nan] * extra_nans)

        arr = np.array(values, dtype=float).reshape(-1, 1)
        nan_mask = np.isnan(arr.ravel())

        expected_median = np.median(non_nan_values)

        imputer = SimpleImputer(strategy="median")
        result = imputer.fit_transform(arr)

        for i in range(len(values)):
            if nan_mask[i]:
                assert abs(result[i, 0] - expected_median) < 1e-10, (
                    f"Position {i}: expected {expected_median}, got {result[i, 0]}"
                )


# ---------------------------------------------------------------------------
# Property 5: Categorical imputation uses mode
# ---------------------------------------------------------------------------

class TestProperty5CategoricalImputationMode:
    """Property 5: Categorical imputation uses mode.

    For any categorical column with NaN, after SimpleImputer(strategy='most_frequent'),
    NaN positions contain the mode of original non-NaN values.
    """

    # Feature: customer-churn-prediction, Property 5: Categorical imputation uses mode

    @given(
        non_nan_values=st_hyp.lists(
            st_hyp.sampled_from(["A", "B", "C", "D", "E"]),
            min_size=1,
            max_size=50,
        ),
        extra_nans=st_hyp.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_imputed_nans_contain_mode(self, non_nan_values, extra_nans):
        """**Validates: Requirements 2.4**"""
        values = list(non_nan_values) + [np.nan] * extra_nans
        arr = np.array(values, dtype=object).reshape(-1, 1)

        # Compute expected mode: most frequent, ties broken by smallest value
        from collections import Counter
        counts = Counter(non_nan_values)
        max_count = max(counts.values())
        # Among values with max count, pick the smallest (matches sklearn behavior)
        expected_mode = min(v for v, c in counts.items() if c == max_count)

        imputer = SimpleImputer(strategy="most_frequent")
        result = imputer.fit_transform(arr)

        for i in range(len(non_nan_values), len(values)):
            assert result[i, 0] == expected_mode, (
                    f"Position {i}: expected '{expected_mode}', got '{result[i, 0]}'"
                )


# ---------------------------------------------------------------------------
# Property 6: Preprocessing eliminates all missing values
# ---------------------------------------------------------------------------

class TestProperty6PreprocessingEliminatesMissing:
    """Property 6: Preprocessing eliminates all missing values.

    For any DataFrame with NaN patterns, after the full pipeline,
    the Feature_Set contains zero NaN values.
    """

    # Feature: customer-churn-prediction, Property 6: Preprocessing eliminates all missing values

    @given(
        n_rows=st_hyp.integers(min_value=5, max_value=50),
        data=st_hyp.data(),
    )
    @settings(max_examples=50)
    def test_no_nans_after_pipeline(self, n_rows, data):
        """**Validates: Requirements 2.5**"""
        num_cols = ["num_a", "num_b"]
        cat_cols = ["cat_a"]

        # Generate numerical data with possible NaNs
        num_a = data.draw(
            st_hyp.lists(
                st_hyp.one_of(
                    st_hyp.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    st_hyp.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        num_b = data.draw(
            st_hyp.lists(
                st_hyp.one_of(
                    st_hyp.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    st_hyp.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cat_a = data.draw(
            st_hyp.lists(
                st_hyp.one_of(
                    st_hyp.sampled_from(["X", "Y", "Z"]),
                    st_hyp.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

        # Ensure at least one non-NaN value per column for imputation
        assume(any(v is not None for v in num_a))
        assume(any(v is not None for v in num_b))
        assume(any(v is not None for v in cat_a))

        target = data.draw(
            st_hyp.lists(
                st_hyp.integers(min_value=0, max_value=1),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

        df = pd.DataFrame({
            "num_a": num_a,
            "num_b": num_b,
            "cat_a": cat_a,
            "churn": target,
        })

        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
        X, y, feature_names = fit_transform(pipeline, df, "churn")

        assert not np.isnan(X).any(), "Feature set still contains NaN after preprocessing"


# ---------------------------------------------------------------------------
# Property 7: Numerical preprocessing round-trip
# ---------------------------------------------------------------------------

class TestProperty7NumericalRoundTrip:
    """Property 7: Numerical preprocessing round-trip.

    For any numerical values, fit → transform → inverse_transform
    produces values within 1e-6 of originals. Uses StandardScaler directly.
    """

    # Feature: customer-churn-prediction, Property 7: Numerical preprocessing round-trip

    @given(
        values=st_hyp.lists(
            st_hyp.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        ),
    )
    @settings(max_examples=100)
    def test_scaler_round_trip(self, values):
        """**Validates: Requirements 2.7**"""
        assume(len(set(values)) >= 2)

        arr = np.array(values).reshape(-1, 1)
        scaler = StandardScaler()
        transformed = scaler.fit_transform(arr)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_allclose(recovered, arr, atol=1e-6)

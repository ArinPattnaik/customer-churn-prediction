"""Data Preprocessing and Feature Engineering module.

Builds and applies a Scikit-learn preprocessing pipeline using
ColumnTransformer with imputation, encoding, and scaling.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    numerical_columns: list[str],
    categorical_columns: list[str],
    encoding_strategy: str = "onehot",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with imputation, encoding, and scaling.

    Numerical path: MedianImputer -> StandardScaler
    Categorical path: ModeImputer -> OneHotEncoder or OrdinalEncoder

    Args:
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.
        encoding_strategy: "onehot" for OneHotEncoder or "label" for OrdinalEncoder.

    Returns:
        Unfitted ColumnTransformer.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if encoding_strategy == "label":
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    logger.info(
        "Built preprocessing pipeline with %d numerical and %d categorical columns "
        "(encoding: %s).",
        len(numerical_columns),
        len(categorical_columns),
        encoding_strategy,
    )

    return preprocessor


def fit_transform(
    pipeline: ColumnTransformer,
    df: pd.DataFrame,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fit the pipeline on df and transform features.

    Args:
        pipeline: An unfitted ColumnTransformer.
        df: DataFrame containing features and the target column.
        target_column: Name of the target column to separate from features.

    Returns:
        (X_transformed, y, feature_names)
    """
    y = df[target_column].to_numpy()
    X = df.drop(columns=[target_column])

    logger.info("Fitting preprocessing pipeline on %d rows.", len(X))
    X_transformed = pipeline.fit_transform(X)

    feature_names = _get_feature_names(pipeline)

    logger.info(
        "Preprocessing complete. Output shape: %s, features: %d.",
        X_transformed.shape,
        len(feature_names),
    )

    return X_transformed, y, feature_names


def transform(pipeline: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    """
    Transform new data using a fitted pipeline.

    Args:
        pipeline: A fitted ColumnTransformer.
        df: DataFrame containing features (no target column).

    Returns:
        Transformed numpy array.
    """
    logger.info("Transforming %d rows using fitted pipeline.", len(df))
    X_transformed = pipeline.transform(df)
    return X_transformed


def save_pipeline(pipeline: ColumnTransformer, path: str) -> None:
    """
    Persist fitted pipeline to disk using joblib.

    Args:
        pipeline: A fitted ColumnTransformer.
        path: File path to save the pipeline.
    """
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s.", path)


def load_pipeline(path: str) -> ColumnTransformer:
    """
    Load a fitted pipeline from disk.

    Args:
        path: File path to load the pipeline from.

    Returns:
        Loaded ColumnTransformer.
    """
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s.", path)
    return pipeline


def _get_feature_names(pipeline: ColumnTransformer) -> list[str]:
    """
    Extract feature names from a fitted ColumnTransformer.

    Args:
        pipeline: A fitted ColumnTransformer.

    Returns:
        List of output feature names.
    """
    return list(pipeline.get_feature_names_out())

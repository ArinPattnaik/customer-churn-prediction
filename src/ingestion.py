"""Data Ingestion and Validation module.

Responsible for loading CSV files, validating schema compliance,
and logging data summaries.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame with raw customer data.

    Raises:
        FileNotFoundError: If file_path does not exist.
        PermissionError: If file_path is not readable.
        pd.errors.ParserError: If file is not valid CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: '{file_path}'. Please verify the file path exists."
        )

    if not os.access(file_path, os.R_OK):
        raise PermissionError(
            f"Cannot read CSV file: '{file_path}'. Permission denied."
        )

    logger.info("Loading CSV file: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Successfully loaded CSV with %d rows and %d columns.", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: The loaded DataFrame.
        required_columns: List of column names that must be present.

    Raises:
        ValueError: With a message listing all missing column names.
    """
    existing_columns = set(df.columns)
    missing_columns = [col for col in required_columns if col not in existing_columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"The DataFrame is missing {len(missing_columns)} required column(s)."
        )

    logger.info("Schema validation passed. All %d required columns are present.", len(required_columns))


def log_data_summary(df: pd.DataFrame) -> dict:
    """
    Log and return a summary of the loaded data.

    Args:
        df: The loaded DataFrame.

    Returns:
        dict with keys: row_count, column_count, missing_values_per_column.
    """
    missing_per_column = df.isnull().sum().to_dict()

    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values_per_column": missing_per_column,
    }

    logger.info("Data summary — rows: %d, columns: %d", summary["row_count"], summary["column_count"])

    for col, count in missing_per_column.items():
        if count > 0:
            logger.info("  Column '%s': %d missing values", col, count)

    return summary

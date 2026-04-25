"""Artifact Persistence module.

Handles versioned saving and loading of trained models, preprocessing
pipelines, metadata, and feature configuration using joblib and JSON.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import joblib

logger = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = {
    "model_type",
    "training_date",
    "dataset_row_count",
    "feature_list",
    "evaluation_metrics",
}


@dataclass
class ArtifactMetadata:
    """Metadata describing a trained model artifact."""

    model_type: str
    training_date: str
    dataset_row_count: int
    feature_list: list[str]
    evaluation_metrics: dict[str, float]


def save_artifacts(
    output_dir: str,
    model: object,
    pipeline: object,
    metadata: ArtifactMetadata,
    feature_config: dict,
) -> str:
    """
    Save all artifacts to a timestamped subdirectory of output_dir.

    Creates: model.joblib, pipeline.joblib, metadata.json, feature_config.json

    Args:
        output_dir: Parent directory for versioned artifact directories.
        model: Fitted model estimator.
        pipeline: Fitted preprocessing pipeline.
        metadata: ArtifactMetadata describing the training run.
        feature_config: Dictionary of feature configuration to persist.

    Returns:
        Path to the created version directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(output_dir, timestamp)
    os.makedirs(version_dir, exist_ok=True)

    model_path = os.path.join(version_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info("Model saved to %s.", model_path)

    pipeline_path = os.path.join(version_dir, "pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    logger.info("Pipeline saved to %s.", pipeline_path)

    metadata_path = os.path.join(version_dir, "metadata.json")
    save_metadata(metadata, metadata_path)

    config_path = os.path.join(version_dir, "feature_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)
    logger.info("Feature config saved to %s.", config_path)

    logger.info("All artifacts saved to %s.", version_dir)
    return version_dir


def load_artifacts(version_dir: str) -> tuple[object, object, ArtifactMetadata]:
    """
    Load model, pipeline, and metadata from a version directory.

    Args:
        version_dir: Path to a versioned artifact directory.

    Returns:
        (model, pipeline, metadata)

    Raises:
        FileNotFoundError: If version_dir does not exist.
    """
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(
            f"Version directory does not exist: {version_dir}"
        )

    model_path = os.path.join(version_dir, "model.joblib")
    model = joblib.load(model_path)
    logger.info("Model loaded from %s.", model_path)

    pipeline_path = os.path.join(version_dir, "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    logger.info("Pipeline loaded from %s.", pipeline_path)

    metadata_path = os.path.join(version_dir, "metadata.json")
    metadata = load_metadata(metadata_path)

    logger.info("All artifacts loaded from %s.", version_dir)
    return model, pipeline, metadata


def save_metadata(metadata: ArtifactMetadata, path: str) -> None:
    """
    Serialize metadata to JSON.

    Args:
        metadata: ArtifactMetadata to persist.
        path: File path for the JSON output.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
    logger.info("Metadata saved to %s.", path)


def load_metadata(path: str) -> ArtifactMetadata:
    """
    Deserialize metadata from JSON.

    Validates that all required keys are present.

    Args:
        path: File path to the metadata JSON.

    Returns:
        ArtifactMetadata populated from the JSON file.

    Raises:
        ValueError: If required keys are missing from the JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_keys = REQUIRED_METADATA_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Metadata JSON is missing required keys: {sorted(missing_keys)}"
        )

    metadata = ArtifactMetadata(
        model_type=data["model_type"],
        training_date=data["training_date"],
        dataset_row_count=data["dataset_row_count"],
        feature_list=data["feature_list"],
        evaluation_metrics=data["evaluation_metrics"],
    )
    logger.info("Metadata loaded from %s.", path)
    return metadata

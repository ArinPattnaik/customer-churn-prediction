"""Model Training and Evaluation module.

Trains multiple classifiers (LogisticRegression, XGBClassifier),
evaluates them on standard metrics, and selects the best model
by AUC-ROC.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float


@dataclass
class TrainedModel:
    """Container for a trained model and its evaluation metrics."""

    name: str
    model: object  # fitted sklearn/xgboost estimator
    metrics: EvaluationMetrics


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/test split preserving class distribution.

    Args:
        X: Feature matrix.
        y: Target labels (binary).
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Data split complete. Train: %d samples, Test: %d samples.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def _evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationMetrics:
    """
    Evaluate a fitted classifier on the test set.

    Args:
        model: A fitted estimator with predict and predict_proba methods.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        EvaluationMetrics with accuracy, precision, recall, F1, and AUC-ROC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = EvaluationMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0.0),
        recall=recall_score(y_test, y_pred, zero_division=0.0),
        f1_score=f1_score(y_test, y_pred, zero_division=0.0),
        auc_roc=roc_auc_score(y_test, y_proba),
    )
    return metrics


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[TrainedModel]:
    """
    Train baseline (LogisticRegression) and XGBoost classifiers.
    Evaluate each on the test set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        List of TrainedModel with metrics populated.
    """
    models: list[TrainedModel] = []

    # Baseline: Logistic Regression
    logger.info("Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_metrics = _evaluate_model(lr, X_test, y_test)
    models.append(TrainedModel(name="LogisticRegression", model=lr, metrics=lr_metrics))
    logger.info(
        "LogisticRegression — Accuracy: %.4f, Precision: %.4f, Recall: %.4f, "
        "F1: %.4f, AUC-ROC: %.4f",
        lr_metrics.accuracy,
        lr_metrics.precision,
        lr_metrics.recall,
        lr_metrics.f1_score,
        lr_metrics.auc_roc,
    )

    # XGBoost
    logger.info("Training XGBClassifier...")
    xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    xgb.fit(X_train, y_train)
    xgb_metrics = _evaluate_model(xgb, X_test, y_test)
    models.append(TrainedModel(name="XGBClassifier", model=xgb, metrics=xgb_metrics))
    logger.info(
        "XGBClassifier — Accuracy: %.4f, Precision: %.4f, Recall: %.4f, "
        "F1: %.4f, AUC-ROC: %.4f",
        xgb_metrics.accuracy,
        xgb_metrics.precision,
        xgb_metrics.recall,
        xgb_metrics.f1_score,
        xgb_metrics.auc_roc,
    )

    return models


def select_best_model(models: list[TrainedModel]) -> TrainedModel:
    """
    Select the model with the highest AUC-ROC.

    Args:
        models: List of TrainedModel objects to compare.

    Returns:
        The TrainedModel with the highest AUC-ROC score.

    Raises:
        ValueError: If the models list is empty.
    """
    if not models:
        raise ValueError("Cannot select best model from an empty list.")

    best = max(models, key=lambda m: m.metrics.auc_roc)
    logger.info(
        "Best model: %s with AUC-ROC: %.4f",
        best.name,
        best.metrics.auc_roc,
    )
    return best

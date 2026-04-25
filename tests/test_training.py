"""Unit tests for the training module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.training import (
    EvaluationMetrics,
    TrainedModel,
    select_best_model,
    split_data,
    train_models,
)


@pytest.fixture
def binary_dataset():
    """Create a small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


class TestEvaluationMetrics:
    def test_dataclass_fields(self):
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            auc_roc=0.90,
        )
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.80
        assert metrics.recall == 0.75
        assert metrics.f1_score == 0.77
        assert metrics.auc_roc == 0.90


class TestTrainedModel:
    def test_dataclass_fields(self):
        metrics = EvaluationMetrics(0.8, 0.7, 0.6, 0.65, 0.85)
        model = TrainedModel(name="TestModel", model=None, metrics=metrics)
        assert model.name == "TestModel"
        assert model.model is None
        assert model.metrics.auc_roc == 0.85


class TestSplitData:
    def test_split_sizes(self, binary_dataset):
        X, y = binary_dataset
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == 160
        assert len(X_test) == 40
        assert len(y_train) == 160
        assert len(y_test) == 40

    def test_stratification_preserves_class_ratio(self, binary_dataset):
        X, y = binary_dataset
        original_ratio = np.mean(y)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05

    def test_reproducibility(self, binary_dataset):
        X, y = binary_dataset
        split1 = split_data(X, y, random_state=123)
        split2 = split_data(X, y, random_state=123)
        np.testing.assert_array_equal(split1[0], split2[0])
        np.testing.assert_array_equal(split1[1], split2[1])


class TestTrainModels:
    def test_returns_two_models(self, binary_dataset):
        X, y = binary_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)
        models = train_models(X_train, y_train, X_test, y_test)
        assert len(models) == 2

    def test_model_names(self, binary_dataset):
        X, y = binary_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)
        models = train_models(X_train, y_train, X_test, y_test)
        names = {m.name for m in models}
        assert names == {"LogisticRegression", "XGBClassifier"}

    def test_metrics_in_valid_range(self, binary_dataset):
        X, y = binary_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)
        models = train_models(X_train, y_train, X_test, y_test)
        for m in models:
            assert 0.0 <= m.metrics.accuracy <= 1.0
            assert 0.0 <= m.metrics.precision <= 1.0
            assert 0.0 <= m.metrics.recall <= 1.0
            assert 0.0 <= m.metrics.f1_score <= 1.0
            assert 0.0 <= m.metrics.auc_roc <= 1.0

    def test_models_are_fitted(self, binary_dataset):
        X, y = binary_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)
        models = train_models(X_train, y_train, X_test, y_test)
        for m in models:
            preds = m.model.predict(X_test)
            assert len(preds) == len(y_test)


class TestSelectBestModel:
    def test_selects_highest_auc_roc(self):
        models = [
            TrainedModel("A", None, EvaluationMetrics(0.8, 0.7, 0.6, 0.65, 0.75)),
            TrainedModel("B", None, EvaluationMetrics(0.7, 0.6, 0.8, 0.69, 0.90)),
            TrainedModel("C", None, EvaluationMetrics(0.9, 0.9, 0.9, 0.9, 0.80)),
        ]
        best = select_best_model(models)
        assert best.name == "B"
        assert best.metrics.auc_roc == 0.90

    def test_single_model(self):
        model = TrainedModel("Only", None, EvaluationMetrics(0.5, 0.5, 0.5, 0.5, 0.5))
        best = select_best_model([model])
        assert best.name == "Only"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_best_model([])

"""
tests/test_pipeline.py
=======================
Tests for the ML engineering pipeline components.
Pure Python — no numpy/sklearn required.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Ensure this module's 'pipeline' package takes priority
_ml_root = str(Path(__file__).parent.parent)
if _ml_root not in sys.path:
    sys.path.insert(0, _ml_root)
# Remove stale 'pipeline' entries that may have been loaded from 08_data_engineering
for _key in list(sys.modules.keys()):
    if _key == "pipeline" or _key.startswith("pipeline."):
        del sys.modules[_key]

from pipeline.feature_engineering import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    VarianceThresholdSelector,
    to_feature_matrix,
)
from pipeline.model_trainer import (
    KNNClassifier,
    LogisticRegression,
    cross_validate,
    train_test_split,
)
from pipeline.model_evaluator import (
    accuracy,
    accuracy as eval_accuracy,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_f1,
    r2_score,
    roc_auc,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_binary_dataset(
    n: int = 100,
    seed: int = 42,
) -> tuple[list[list[float]], list[float]]:
    """Two separable Gaussian clusters."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    for i in range(n):
        label = float(i % 2)
        offset = 3.0 if label == 1.0 else -3.0
        X.append([rng.gauss(offset, 0.5), rng.gauss(offset, 0.5)])
        y.append(label)
    return X, y


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestMinMaxScaler:
    def test_output_range(self) -> None:
        data = [{"x": float(i)} for i in range(10)]
        scaler = MinMaxScaler(["x"])
        result = scaler.fit_transform(data)
        values = [r["x"] for r in result]
        assert min(values) == pytest.approx(0.0)
        assert max(values) == pytest.approx(1.0)

    def test_constant_column(self) -> None:
        data = [{"x": 5.0} for _ in range(5)]
        scaler = MinMaxScaler(["x"])
        result = scaler.fit_transform(data)
        assert all(r["x"] == 0.0 for r in result)


class TestStandardScaler:
    def test_zero_mean(self) -> None:
        data = [{"x": float(i)} for i in range(10)]
        scaler = StandardScaler(["x"])
        result = scaler.fit_transform(data)
        mean = sum(r["x"] for r in result) / len(result)
        assert mean == pytest.approx(0.0, abs=1e-9)


class TestOneHotEncoder:
    def test_basic(self) -> None:
        data = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
        enc = OneHotEncoder(["color"])
        result = enc.fit_transform(data)
        assert "color_red" in result[0]
        assert "color_blue" in result[0]
        assert result[0]["color_red"] == 1.0
        assert result[0]["color_blue"] == 0.0

    def test_removes_original_column(self) -> None:
        data = [{"color": "red"}]
        enc = OneHotEncoder(["color"])
        result = enc.fit_transform(data)
        assert "color" not in result[0]


class TestLabelEncoder:
    def test_encode_decode(self) -> None:
        enc = LabelEncoder()
        labels = ["cat", "dog", "cat", "bird"]
        enc.fit(labels)
        encoded = enc.transform(labels)
        decoded = enc.inverse_transform(encoded)
        assert decoded == labels

    def test_classes(self) -> None:
        enc = LabelEncoder()
        enc.fit(["b", "a", "c"])
        assert enc.classes == ["a", "b", "c"]


class TestVarianceThreshold:
    def test_removes_constant(self) -> None:
        data = [{"a": 1.0, "b": float(i)} for i in range(10)]
        sel = VarianceThresholdSelector(["a", "b"], threshold=0.01)
        result = sel.fit_transform(data)
        assert "a" not in result[0]
        assert "b" in result[0]


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_sizes(self) -> None:
        X, y = make_binary_dataset(100)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        assert len(X_te) == 20
        assert len(X_tr) == 80
        assert len(y_te) == 20

    def test_no_overlap(self) -> None:
        X, y = make_binary_dataset(50)
        X_tr, X_te, _, _ = train_test_split(X, y)
        train_set = set(map(tuple, X_tr))
        test_set  = set(map(tuple, X_te))
        assert not train_set.intersection(test_set)


class TestLogisticRegression:
    def test_learns_separable(self) -> None:
        X, y = make_binary_dataset(200)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
        model = LogisticRegression(learning_rate=0.5, max_iter=100)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        acc = sum(a == b for a, b in zip(y_te, preds)) / len(y_te)
        assert acc > 0.85, f"Expected accuracy > 0.85, got {acc:.3f}"

    def test_predict_proba_sums_to_1(self) -> None:
        X, y = make_binary_dataset(50)
        model = LogisticRegression(max_iter=50)
        model.fit(X, y)
        probas = model.predict_proba(X[:5])
        for p in probas:
            assert sum(p) == pytest.approx(1.0)

    def test_loss_decreases(self) -> None:
        X, y = make_binary_dataset(100)
        model = LogisticRegression(max_iter=50)
        model.fit(X, y)
        assert model.losses[-1] < model.losses[0]


class TestKNN:
    def test_perfect_separation(self) -> None:
        X_train = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
        y_train = [0.0, 0.0, 0.0]
        X_test  = [[10.0, 10.0]]
        X_train += [[10.0, 10.0], [9.9, 10.0]]
        y_train += [1.0, 1.0]
        model = KNNClassifier(k=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds[0] == 1.0

    def test_cross_validate(self) -> None:
        X, y = make_binary_dataset(60)
        cv = cross_validate(KNNClassifier(k=3), X, y, k=3, random_state=0)
        assert cv.mean > 0.7


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self) -> None:
        assert eval_accuracy([1, 0, 1], [1, 0, 1]) == 1.0

    def test_half(self) -> None:
        assert eval_accuracy([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5


class TestPrecisionRecallF1:
    def test_perfect_binary(self) -> None:
        y = [1.0, 0.0, 1.0, 0.0]
        p, r, f = precision_recall_f1(y, y)
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_no_positive_predictions(self) -> None:
        y_true = [1.0, 1.0]
        y_pred = [0.0, 0.0]
        p, r, f = precision_recall_f1(y_true, y_pred)
        assert p == 0.0 and r == 0.0 and f == 0.0


class TestConfusionMatrix:
    def test_binary(self) -> None:
        y = [1.0, 0.0, 1.0, 0.0]
        p = [1.0, 1.0, 0.0, 0.0]
        cm = confusion_matrix(y, p)
        assert cm[(1.0, 1.0)] == 1   # TP
        assert cm[(0.0, 1.0)] == 1   # FP
        assert cm[(1.0, 0.0)] == 1   # FN
        assert cm[(0.0, 0.0)] == 1   # TN


class TestRocAuc:
    def test_perfect(self) -> None:
        y     = [1.0, 1.0, 0.0, 0.0]
        score = [0.9, 0.8, 0.2, 0.1]
        assert roc_auc(y, score) == pytest.approx(1.0)

    def test_random(self) -> None:
        y     = [1.0, 0.0, 1.0, 0.0]
        score = [0.5, 0.5, 0.5, 0.5]
        assert 0.0 <= roc_auc(y, score) <= 1.0


class TestRegressionMetrics:
    def test_mae_perfect(self) -> None:
        y = [1.0, 2.0, 3.0]
        assert mean_absolute_error(y, y) == 0.0

    def test_rmse(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 3.0]
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            math.sqrt(1 / 3)
        )

    def test_r2_perfect(self) -> None:
        y = [1.0, 2.0, 3.0]
        assert r2_score(y, y) == pytest.approx(1.0)

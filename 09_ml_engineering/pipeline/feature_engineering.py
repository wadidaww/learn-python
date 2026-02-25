"""
pipeline/feature_engineering.py
=================================
Feature engineering utilities: scaling, encoding, and selection.
All implemented in pure Python (no numpy/sklearn dependency).
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any


Vector = list[float]
Matrix = list[Vector]
Dataset = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Base transformer
# ---------------------------------------------------------------------------

class FeatureTransformer(ABC):
    """Abstract feature transformer that fits on training data and transforms."""

    @abstractmethod
    def fit(self, data: Dataset) -> FeatureTransformer:
        """Fit parameters from *data*; return self."""

    @abstractmethod
    def transform(self, data: Dataset) -> Dataset:
        """Apply transformation to *data*; return transformed dataset."""

    def fit_transform(self, data: Dataset) -> Dataset:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


# ---------------------------------------------------------------------------
# Min-Max Scaler
# ---------------------------------------------------------------------------

class MinMaxScaler(FeatureTransformer):
    """
    Scale numeric features to [0, 1].

    Example::

        scaler = MinMaxScaler(columns=["age", "salary"])
        train = scaler.fit_transform(train_data)
        test  = scaler.transform(test_data)
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def fit(self, data: Dataset) -> MinMaxScaler:
        for col in self.columns:
            values = [float(r[col]) for r in data if r.get(col) is not None]
            if not values:
                self._min[col], self._max[col] = 0.0, 1.0
            else:
                self._min[col] = min(values)
                self._max[col] = max(values)
        return self

    def transform(self, data: Dataset) -> Dataset:
        result: Dataset = []
        for row in data:
            out = dict(row)
            for col in self.columns:
                v = float(out.get(col, 0.0))
                mn, mx = self._min[col], self._max[col]
                out[col] = (v - mn) / (mx - mn) if mx != mn else 0.0
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Standard Scaler (Z-score)
# ---------------------------------------------------------------------------

class StandardScaler(FeatureTransformer):
    """
    Standardise numeric features to zero mean and unit variance.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    def fit(self, data: Dataset) -> StandardScaler:
        for col in self.columns:
            values = [float(r[col]) for r in data if r.get(col) is not None]
            if not values:
                self._mean[col], self._std[col] = 0.0, 1.0
            else:
                self._mean[col] = statistics.mean(values)
                self._std[col]  = statistics.stdev(values) if len(values) > 1 else 1.0
                if self._std[col] == 0:
                    self._std[col] = 1.0
        return self

    def transform(self, data: Dataset) -> Dataset:
        result: Dataset = []
        for row in data:
            out = dict(row)
            for col in self.columns:
                v = float(out.get(col, 0.0))
                out[col] = (v - self._mean[col]) / self._std[col]
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# One-Hot Encoder
# ---------------------------------------------------------------------------

class OneHotEncoder(FeatureTransformer):
    """
    Encode categorical columns as one-hot binary features.

    Original column is removed; new columns named "<col>_<value>" are added.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._categories: dict[str, list[str]] = {}

    def fit(self, data: Dataset) -> OneHotEncoder:
        for col in self.columns:
            cats = sorted({str(r[col]) for r in data if r.get(col) is not None})
            self._categories[col] = cats
        return self

    def transform(self, data: Dataset) -> Dataset:
        result: Dataset = []
        for row in data:
            out = {k: v for k, v in row.items() if k not in self.columns}
            for col in self.columns:
                cats = self._categories.get(col, [])
                val = str(row.get(col, ""))
                for cat in cats:
                    out[f"{col}_{cat}"] = 1.0 if val == cat else 0.0
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Label Encoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Encode a target column as integer class labels."""

    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def fit(self, labels: list[Any]) -> LabelEncoder:
        unique = sorted({str(l) for l in labels})
        self._mapping = {v: i for i, v in enumerate(unique)}
        self._inverse = {i: v for v, i in self._mapping.items()}
        return self

    def transform(self, labels: list[Any]) -> list[int]:
        return [self._mapping[str(l)] for l in labels]

    def inverse_transform(self, indices: list[int]) -> list[str]:
        return [self._inverse[i] for i in indices]

    @property
    def classes(self) -> list[str]:
        return [self._inverse[i] for i in sorted(self._inverse)]


# ---------------------------------------------------------------------------
# Feature selector (variance threshold)
# ---------------------------------------------------------------------------

class VarianceThresholdSelector(FeatureTransformer):
    """
    Remove features with variance below *threshold*.
    Useful for removing near-constant features.
    """

    def __init__(self, numeric_cols: list[str], threshold: float = 0.01) -> None:
        self.numeric_cols = numeric_cols
        self.threshold = threshold
        self._selected: list[str] = []

    def fit(self, data: Dataset) -> VarianceThresholdSelector:
        self._selected = []
        for col in self.numeric_cols:
            values = [float(r[col]) for r in data if r.get(col) is not None]
            if len(values) < 2:
                continue
            variance = statistics.variance(values)
            if variance >= self.threshold:
                self._selected.append(col)
        return self

    def transform(self, data: Dataset) -> Dataset:
        all_cols_to_keep = self._selected
        result: Dataset = []
        for row in data:
            result.append({k: v for k, v in row.items()
                           if k not in self.numeric_cols or k in all_cols_to_keep})
        return result

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected)


# ---------------------------------------------------------------------------
# Feature extraction to matrix
# ---------------------------------------------------------------------------

def to_feature_matrix(
    data: Dataset,
    feature_cols: list[str],
) -> tuple[Matrix, list[str]]:
    """
    Convert dataset rows to a numeric feature matrix.

    Returns:
        (matrix, column_names) where matrix is list[list[float]].
    """
    matrix: Matrix = []
    for row in data:
        vector = [float(row.get(col, 0.0)) for col in feature_cols]
        matrix.append(vector)
    return matrix, feature_cols

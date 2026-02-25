"""
pipeline/model_trainer.py
==========================
Generic model training pipeline with cross-validation.
Pure Python implementation — no numpy/sklearn required.

Includes:
  - Logistic Regression from scratch (gradient descent)
  - K-Fold cross-validation
  - Train/test split
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

Matrix = list[list[float]]
Vector = list[float]


# ---------------------------------------------------------------------------
# Abstract model interface
# ---------------------------------------------------------------------------

class Model(ABC):
    """Abstract ML model."""

    @abstractmethod
    def fit(self, X: Matrix, y: Vector) -> Model:
        """Train on feature matrix *X* and target vector *y*."""

    @abstractmethod
    def predict(self, X: Matrix) -> Vector:
        """Return predictions for *X*."""

    def predict_proba(self, X: Matrix) -> list[Vector]:
        """Return class probabilities (optional, default raises NotImplementedError)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def dot(a: Vector, b: Vector) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))


def train_test_split(
    X: Matrix,
    y: Vector,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[Matrix, Matrix, Vector, Vector]:
    """
    Split (X, y) into training and test sets.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    rng = random.Random(random_state)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    n_test = max(1, int(len(X) * test_size))
    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def k_fold_split(
    X: Matrix,
    y: Vector,
    k: int = 5,
    random_state: int | None = None,
) -> list[tuple[Matrix, Matrix, Vector, Vector]]:
    """
    K-fold cross-validation split.

    Returns list of (X_train, X_val, y_train, y_val) tuples.
    """
    rng = random.Random(random_state)
    indices = list(range(len(X)))
    rng.shuffle(indices)

    fold_size = len(X) // k
    folds: list[tuple[Matrix, Matrix, Vector, Vector]] = []

    for i in range(k):
        val_start = i * fold_size
        val_end   = val_start + fold_size if i < k - 1 else len(X)
        val_idx   = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]

        folds.append((
            [X[j] for j in train_idx],
            [X[j] for j in val_idx],
            [y[j] for j in train_idx],
            [y[j] for j in val_idx],
        ))

    return folds


# ---------------------------------------------------------------------------
# Logistic Regression (binary classifier)
# ---------------------------------------------------------------------------

@dataclass
class LogisticRegression(Model):
    """
    Binary logistic regression trained with mini-batch gradient descent.

    Attributes:
        learning_rate: Step size for gradient descent.
        max_iter:      Maximum number of training epochs.
        tol:           Convergence tolerance for log-loss.
        batch_size:    Mini-batch size (0 = full batch).
        random_state:  Seed for reproducibility.
    """

    learning_rate: float = 0.01
    max_iter:      int   = 200
    tol:           float = 1e-4
    batch_size:    int   = 32
    random_state:  int   = 42
    verbose:       bool  = False

    weights: Vector       = field(default_factory=list, init=False, repr=False)
    bias:    float        = field(default=0.0, init=False, repr=False)
    losses:  list[float]  = field(default_factory=list, init=False, repr=False)

    def _log_loss(self, X: Matrix, y: Vector) -> float:
        total = 0.0
        eps = 1e-12
        for xi, yi in zip(X, y):
            p = sigmoid(dot(self.weights, xi) + self.bias)
            total += -(yi * math.log(p + eps) + (1 - yi) * math.log(1 - p + eps))
        return total / len(X)

    def fit(self, X: Matrix, y: Vector) -> LogisticRegression:
        n_features = len(X[0])
        rng = random.Random(self.random_state)

        self.weights = [rng.gauss(0, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        self.losses = []

        indices = list(range(len(X)))

        for epoch in range(self.max_iter):
            rng.shuffle(indices)
            bs = self.batch_size if self.batch_size > 0 else len(X)

            for start in range(0, len(indices), bs):
                batch = indices[start : start + bs]
                grad_w = [0.0] * n_features
                grad_b = 0.0

                for i in batch:
                    xi, yi = X[i], y[i]
                    p = sigmoid(dot(self.weights, xi) + self.bias)
                    err = p - yi
                    for j in range(n_features):
                        grad_w[j] += err * xi[j]
                    grad_b += err

                m = len(batch)
                self.weights = [
                    w - self.learning_rate * g / m
                    for w, g in zip(self.weights, grad_w)
                ]
                self.bias -= self.learning_rate * grad_b / m

            loss = self._log_loss(X, y)
            self.losses.append(loss)

            if self.verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:>4}: loss={loss:.6f}")

            if epoch > 0 and abs(self.losses[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"  Converged at epoch {epoch}")
                break

        return self

    def predict_proba(self, X: Matrix) -> list[Vector]:
        probs: list[Vector] = []
        for xi in X:
            p1 = sigmoid(dot(self.weights, xi) + self.bias)
            probs.append([1 - p1, p1])
        return probs

    def predict(self, X: Matrix) -> Vector:
        return [1.0 if p[1] >= 0.5 else 0.0 for p in self.predict_proba(X)]


# ---------------------------------------------------------------------------
# K-Nearest Neighbours
# ---------------------------------------------------------------------------

@dataclass
class KNNClassifier(Model):
    """
    K-Nearest Neighbours classifier.

    Uses Euclidean distance. Ties broken by returning the most common class.
    """

    k: int = 3

    _X_train: Matrix = field(default_factory=list, init=False, repr=False)
    _y_train: Vector = field(default_factory=list, init=False, repr=False)

    @staticmethod
    def _distance(a: Vector, b: Vector) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit(self, X: Matrix, y: Vector) -> KNNClassifier:
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: Matrix) -> Vector:
        preds: Vector = []
        for xi in X:
            distances = [
                (self._distance(xi, xj), yj)
                for xj, yj in zip(self._X_train, self._y_train)
            ]
            distances.sort(key=lambda t: t[0])
            k_labels = [label for _, label in distances[: self.k]]
            # majority vote
            counts: dict[float, int] = {}
            for lbl in k_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            preds.append(max(counts, key=lambda x: counts[x]))
        return preds


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

@dataclass
class CrossValidationResult:
    """Result of k-fold cross-validation."""

    scores: list[float]

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores)

    @property
    def std(self) -> float:
        import statistics
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0


def cross_validate(
    model: Model,
    X: Matrix,
    y: Vector,
    k: int = 5,
    scorer: Any = None,
    random_state: int = 42,
) -> CrossValidationResult:
    """
    K-fold cross-validation.

    Args:
        model:        Model instance (will be re-trained each fold).
        X, y:         Feature matrix and target vector.
        k:            Number of folds.
        scorer:       Callable(y_true, y_pred) → float. Defaults to accuracy.
        random_state: Random seed.

    Returns:
        CrossValidationResult with per-fold scores.
    """
    if scorer is None:
        def scorer(y_true: Vector, y_pred: Vector) -> float:
            return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

    folds = k_fold_split(X, y, k=k, random_state=random_state)
    scores: list[float] = []

    for X_train, X_val, y_train, y_val in folds:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(scorer(y_val, y_pred))

    return CrossValidationResult(scores=scores)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Train a logistic regression on a synthetic dataset."""
    import math

    rng = random.Random(0)

    def make_dataset(n: int) -> tuple[Matrix, Vector]:
        X: Matrix = []
        y: Vector = []
        for _ in range(n):
            # Two Gaussian clusters
            label = rng.choice([0.0, 1.0])
            offset = 2.0 if label == 1.0 else -2.0
            x1 = rng.gauss(offset, 1.0)
            x2 = rng.gauss(offset, 1.0)
            X.append([x1, x2])
            y.append(label)
        return X, y

    X, y = make_dataset(300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(learning_rate=0.1, max_iter=200, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = sum(a == b for a, b in zip(y_test, y_pred)) / len(y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")

    cv = cross_validate(LogisticRegression(learning_rate=0.1, max_iter=200), X, y, k=5)
    print(f"5-fold CV: {cv.mean:.4f} ± {cv.std:.4f}")


if __name__ == "__main__":
    main()

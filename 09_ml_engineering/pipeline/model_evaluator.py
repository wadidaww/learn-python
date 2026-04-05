"""
pipeline/model_evaluator.py
============================
Model evaluation metrics implemented in pure Python.

Metrics:
  - Accuracy
  - Precision, Recall, F1 (binary and macro/micro)
  - Confusion matrix
  - ROC AUC (trapezoidal rule)
  - Mean Absolute Error, RMSE (regression)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


Vector = list[float]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: Vector, y_pred: Vector) -> float:
    """Fraction of correct predictions."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(
    y_true: Vector,
    y_pred: Vector,
    labels: list[Any] | None = None,
) -> dict[tuple[Any, Any], int]:
    """
    Compute a confusion matrix.

    Returns:
        dict mapping (true_label, predicted_label) → count
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix: dict[tuple[Any, Any], int] = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        key = (t, p)
        if key in matrix:
            matrix[key] += 1
    return matrix


def precision_recall_f1(
    y_true: Vector,
    y_pred: Vector,
    positive_class: Any = 1.0,
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for binary classification.

    Returns:
        (precision, recall, f1)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return prec, rec, f1


def classification_report(
    y_true: Vector,
    y_pred: Vector,
) -> dict[str, dict[str, float]]:
    """
    Per-class precision, recall, F1 plus macro and weighted averages.
    """
    labels = sorted(set(y_true) | set(y_pred))
    report: dict[str, dict[str, float]] = {}
    support: dict[Any, int] = {lbl: y_true.count(lbl) for lbl in labels}
    total = len(y_true)

    for lbl in labels:
        p, r, f = precision_recall_f1(y_true, y_pred, positive_class=lbl)
        report[str(lbl)] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "support":   support[lbl],
        }

    # Macro average
    macro_p = sum(report[str(l)]["precision"] for l in labels) / len(labels)
    macro_r = sum(report[str(l)]["recall"]    for l in labels) / len(labels)
    macro_f = sum(report[str(l)]["f1"]        for l in labels) / len(labels)
    report["macro avg"] = {
        "precision": round(macro_p, 4),
        "recall":    round(macro_r, 4),
        "f1":        round(macro_f, 4),
        "support":   total,
    }

    # Weighted average
    w_p = sum(report[str(l)]["precision"] * support[l] for l in labels) / total
    w_r = sum(report[str(l)]["recall"]    * support[l] for l in labels) / total
    w_f = sum(report[str(l)]["f1"]        * support[l] for l in labels) / total
    report["weighted avg"] = {
        "precision": round(w_p, 4),
        "recall":    round(w_r, 4),
        "f1":        round(w_f, 4),
        "support":   total,
    }

    return report


# ---------------------------------------------------------------------------
# ROC AUC
# ---------------------------------------------------------------------------

def roc_auc(y_true: Vector, y_score: Vector) -> float:
    """
    Compute ROC AUC using the trapezoidal rule.

    Args:
        y_true:  Binary ground truth labels (0.0 or 1.0).
        y_score: Predicted probabilities for the positive class.

    Returns:
        AUC score in [0, 1].
    """
    # Sort by score descending
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp, fp = 0, 0
    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0

    for _, label in pairs:
        if label == 1.0:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return auc


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: Vector, y_pred: Vector) -> float:
    """Mean Absolute Error."""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_squared_error(y_true: Vector, y_pred: Vector) -> float:
    """Mean Squared Error."""
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Vector, y_pred: Vector) -> float:
    """Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true: Vector, y_pred: Vector) -> float:
    """
    Coefficient of determination R².

    Returns 1.0 for perfect predictions; can be negative for very bad models.
    """
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_classification_report(report: dict[str, dict[str, float]]) -> None:
    """Print a formatted classification report."""
    header = f"{'':>15}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>8}"
    print(header)
    print("-" * len(header))
    for label, metrics in report.items():
        print(
            f"  {label:>13}  {metrics['precision']:>10.4f}  "
            f"{metrics['recall']:>8.4f}  {metrics['f1']:>8.4f}  "
            f"{int(metrics['support']):>8}"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate evaluation metrics."""
    y_true: Vector = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred: Vector = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    y_prob: Vector = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3, 0.85, 0.1]

    print(f"Accuracy:  {accuracy(y_true, y_pred):.4f}")
    p, r, f = precision_recall_f1(y_true, y_pred)
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f:.4f}")
    print(f"ROC AUC:   {roc_auc(y_true, y_prob):.4f}")

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    print_classification_report(report)

    print("\nRegression Metrics:")
    y_reg_true: Vector = [3.0, -0.5, 2.0, 7.0]
    y_reg_pred: Vector = [2.5, 0.0,  2.0, 8.0]
    print(f"  MAE:  {mean_absolute_error(y_reg_true, y_reg_pred):.4f}")
    print(f"  RMSE: {root_mean_squared_error(y_reg_true, y_reg_pred):.4f}")
    print(f"  R²:   {r2_score(y_reg_true, y_reg_pred):.4f}")


if __name__ == "__main__":
    main()

"""
competitive_programming/tips_and_tricks.py
==========================================
Small, reusable patterns that show up often in programming contests.
"""

from __future__ import annotations

from collections import Counter


def prefix_sums(values: list[int]) -> list[int]:
    """
    Return prefix sums with a leading zero for O(1) range-sum queries.

    Example:
        values = [3, 1, 4]
        result = [0, 3, 4, 8]
    """
    result = [0]
    running_total = 0
    for value in values:
        running_total += value
        result.append(running_total)
    return result


def sliding_window_sums(values: list[int], window_size: int) -> list[int]:
    """
    Return the sum of each fixed-size window.

    Raises:
        ValueError: If *window_size* is not positive.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > len(values):
        return []

    window_total = sum(values[:window_size])
    result = [window_total]
    for index in range(window_size, len(values)):
        window_total += values[index] - values[index - window_size]
        result.append(window_total)
    return result


def two_sum_sorted(values: list[int], target: int) -> tuple[int, int] | None:
    """
    Return indices of two values in sorted input whose sum equals *target*.

    Uses the classic two-pointer pattern in O(n).
    """
    left = 0
    right = len(values) - 1

    while left < right:
        current = values[left] + values[right]
        if current == target:
            return left, right
        if current < target:
            left += 1
        else:
            right -= 1

    return None


def top_k_frequent(values: list[int], limit: int) -> list[int]:
    """
    Return the *limit* most frequent values.

    Ties are broken by the smaller value first for stable contest-friendly output.
    """
    if limit <= 0:
        return []

    counts = Counter(values)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [value for value, _count in ranked[:limit]]


def transpose_grid(grid: list[list[int]]) -> list[list[int]]:
    """Return the transposed grid."""
    if not grid:
        return []
    return [list(column) for column in zip(*grid, strict=True)]

"""
Tests for competitive programming examples.
"""

from __future__ import annotations

import pytest
from competitive_programming.syntax_patterns import (
    CARDINAL_DIRECTIONS,
    build_adjacency_list,
    pairwise_differences,
    sort_scoreboard,
    unique_preserving_order,
)
from competitive_programming.tips_and_tricks import (
    prefix_sums,
    sliding_window_sums,
    top_k_frequent,
    transpose_grid,
    two_sum_sorted,
)


def test_prefix_sums_has_leading_zero() -> None:
    assert prefix_sums([3, 1, 4, 1, 5]) == [0, 3, 4, 8, 9, 14]


def test_sliding_window_sums() -> None:
    assert sliding_window_sums([1, 2, 3, 4, 5], window_size=3) == [6, 9, 12]


def test_sliding_window_invalid_size() -> None:
    with pytest.raises(ValueError):
        sliding_window_sums([1, 2, 3], window_size=0)


def test_two_sum_sorted_finds_pair() -> None:
    assert two_sum_sorted([1, 2, 4, 7, 11], target=9) == (1, 3)


def test_two_sum_sorted_returns_none_when_missing() -> None:
    assert two_sum_sorted([1, 3, 5], target=100) is None


def test_top_k_frequent_breaks_ties_by_value() -> None:
    assert top_k_frequent([4, 1, 2, 2, 4, 3, 4, 2], limit=2) == [2, 4]


def test_transpose_grid() -> None:
    assert transpose_grid([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_cardinal_directions_order() -> None:
    assert CARDINAL_DIRECTIONS == ((-1, 0), (0, 1), (1, 0), (0, -1))


def test_pairwise_differences() -> None:
    assert pairwise_differences([3, 8, 10, 15]) == [5, 2, 5]


def test_unique_preserving_order() -> None:
    assert unique_preserving_order([5, 1, 5, 2, 1, 3]) == [5, 1, 2, 3]


def test_sort_scoreboard() -> None:
    standings = [
        ("bob", 4, 120),
        ("alice", 5, 200),
        ("carol", 5, 180),
        ("dave", 5, 180),
    ]
    assert sort_scoreboard(standings) == [
        ("carol", 5, 180),
        ("dave", 5, 180),
        ("alice", 5, 200),
        ("bob", 4, 120),
    ]


def test_build_adjacency_list() -> None:
    assert build_adjacency_list(4, [(1, 2), (2, 4), (1, 3)]) == {
        1: [2, 3],
        2: [1, 4],
        3: [1],
        4: [2],
    }

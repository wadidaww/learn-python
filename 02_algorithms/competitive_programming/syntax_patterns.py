"""
competitive_programming/syntax_patterns.py
==========================================
Compact Python syntax patterns that are useful during contests.
"""

from __future__ import annotations

from collections import defaultdict

CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)


def pairwise_differences(values: list[int]) -> list[int]:
    """Return consecutive differences using zip on shifted slices."""
    return [current - previous for previous, current in zip(values, values[1:], strict=False)]


def unique_preserving_order(values: list[int]) -> list[int]:
    """Return unique values while preserving their first appearance order."""
    return list(dict.fromkeys(values))


def sort_scoreboard(entries: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """
    Sort contest standings by score descending, penalty ascending, then name.

    Each tuple is (name, solved, penalty).
    """
    return sorted(entries, key=lambda entry: (-entry[1], entry[2], entry[0]))


def build_adjacency_list(
    node_count: int,
    edges: list[tuple[int, int]],
) -> dict[int, list[int]]:
    """
    Build a 1-indexed adjacency list for an undirected graph.
    """
    graph: defaultdict[int, list[int]] = defaultdict(list)
    for start, end in edges:
        graph[start].append(end)
        graph[end].append(start)

    return {node: graph[node] for node in range(1, node_count + 1)}

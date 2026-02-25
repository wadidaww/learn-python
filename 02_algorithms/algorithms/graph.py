"""
algorithms/graph.py
====================
Graph algorithms: BFS, DFS, Dijkstra, and topological sort.

Graphs are represented as adjacency lists:
    dict[str, list[tuple[str, int]]]  – (neighbor, weight) pairs
or
    dict[str, list[str]]              – unweighted
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any


Graph = dict[str, list[tuple[str, int]]]    # weighted
UGraph = dict[str, list[str]]               # unweighted


# ---------------------------------------------------------------------------
# Breadth-First Search
# ---------------------------------------------------------------------------

def bfs(graph: UGraph, start: str) -> list[str]:
    """
    Return nodes visited in BFS order starting from *start*.

    Time:  O(V + E)
    Space: O(V)
    """
    visited: set[str] = set()
    order: list[str] = []
    queue: deque[str] = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def bfs_shortest_path(graph: UGraph, start: str, end: str) -> list[str] | None:
    """
    Return the shortest path (fewest hops) from *start* to *end*, or None.

    Time:  O(V + E)
    """
    if start == end:
        return [start]

    visited: set[str] = {start}
    queue: deque[list[str]] = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)

    return None


# ---------------------------------------------------------------------------
# Depth-First Search
# ---------------------------------------------------------------------------

def dfs_iterative(graph: UGraph, start: str) -> list[str]:
    """
    Return nodes visited in DFS order (iterative, using an explicit stack).

    Time:  O(V + E)
    Space: O(V)
    """
    visited: set[str] = set()
    order: list[str] = []
    stack: list[str] = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Add neighbors in reverse order to maintain left-to-right traversal
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def dfs_recursive(
    graph: UGraph,
    start: str,
    visited: set[str] | None = None,
) -> list[str]:
    """Return nodes visited in DFS order (recursive)."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))
    return order


# ---------------------------------------------------------------------------
# Dijkstra's shortest path
# ---------------------------------------------------------------------------

def dijkstra(graph: Graph, start: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """
    Compute shortest paths from *start* to all reachable nodes.

    Returns:
        distances: mapping node → minimum distance from start
        predecessors: mapping node → previous node on shortest path

    Time:  O((V + E) log V) with binary heap
    Space: O(V)
    """
    distances: dict[str, float] = {node: math.inf for node in graph}
    distances[start] = 0.0
    predecessors: dict[str, str | None] = {node: None for node in graph}

    # (distance, node)
    heap: list[tuple[float, str]] = [(0.0, start)]

    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue  # stale entry

        for neighbor, weight in graph.get(node, []):
            new_dist = distances[node] + weight
            if new_dist < distances.get(neighbor, math.inf):
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    return distances, predecessors


def reconstruct_path(
    predecessors: dict[str, str | None],
    start: str,
    end: str,
) -> list[str] | None:
    """Reconstruct the shortest path from *predecessors* table."""
    path: list[str] = []
    current: str | None = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current)
        if current == start:
            path.append(start)
            break
    else:
        return None

    path.reverse()
    return path if path[0] == start else None


# ---------------------------------------------------------------------------
# Topological Sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def topological_sort(graph: UGraph) -> list[str] | None:
    """
    Return a topological ordering of nodes in a DAG, or None if a cycle exists.

    Uses Kahn's algorithm (BFS-based).
    Time:  O(V + E)
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    queue: deque[str] = deque(
        node for node, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(graph):
        return None  # cycle detected

    return order


# ---------------------------------------------------------------------------
# Strongly Connected Components (Kosaraju's algorithm)
# ---------------------------------------------------------------------------

def strongly_connected_components(graph: UGraph) -> list[list[str]]:
    """
    Return all SCCs of a directed graph using Kosaraju's algorithm.

    Time:  O(V + E)
    """
    visited: set[str] = set()
    finish_order: list[str] = []

    def dfs1(node: str) -> None:
        visited.add(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                dfs1(nb)
        finish_order.append(node)

    for node in list(graph):
        if node not in visited:
            dfs1(node)

    # Build reversed graph
    rev: UGraph = {n: [] for n in graph}
    for node in graph:
        for nb in graph[node]:
            rev.setdefault(nb, []).append(node)

    visited.clear()
    sccs: list[list[str]] = []

    def dfs2(node: str, component: list[str]) -> None:
        visited.add(node)
        component.append(node)
        for nb in rev.get(node, []):
            if nb not in visited:
                dfs2(nb, component)

    for node in reversed(finish_order):
        if node not in visited:
            comp: list[str] = []
            dfs2(node, comp)
            sccs.append(comp)

    return sccs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate graph algorithms."""
    # Unweighted graph
    ug: UGraph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }
    print("BFS from A:", bfs(ug, "A"))
    print("DFS from A:", dfs_iterative(ug, "A"))
    print("Shortest A→F:", bfs_shortest_path(ug, "A", "F"))

    # Weighted graph for Dijkstra
    wg: Graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("A", 1), ("C", 2), ("D", 5)],
        "C": [("A", 4), ("B", 2), ("D", 1)],
        "D": [("B", 5), ("C", 1)],
    }
    distances, predecessors = dijkstra(wg, "A")
    print("\nDijkstra from A:", distances)
    print("Path A→D:", reconstruct_path(predecessors, "A", "D"))

    # Topological sort
    dag: UGraph = {
        "A": ["C"],
        "B": ["C", "D"],
        "C": ["E"],
        "D": ["F"],
        "E": ["F"],
        "F": [],
    }
    print("\nTopological sort:", topological_sort(dag))


if __name__ == "__main__":
    main()

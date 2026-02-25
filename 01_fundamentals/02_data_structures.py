"""
Module 01 – Data Structures
============================
Practical demonstrations of Python's core built-in data structures:
list, tuple, dict, set, and the collections module.

Run directly:  python 02_data_structures.py
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque, namedtuple
from typing import Any


# ---------------------------------------------------------------------------
# 1. Lists
# ---------------------------------------------------------------------------

def demonstrate_lists() -> None:
    """Common list operations, slicing, and sorting."""
    nums: list[int] = [5, 3, 8, 1, 9, 2, 7, 4, 6]

    print("Original:      ", nums)
    print("Sorted (copy): ", sorted(nums))
    print("Reversed:      ", list(reversed(nums)))
    print("Slice [2:5]:   ", nums[2:5])
    print("Every 2nd:     ", nums[::2])

    nums.sort()
    print("In-place sort: ", nums)

    # Stack operations
    stack: list[int] = []
    for v in [1, 2, 3]:
        stack.append(v)
    print("Stack pop:", stack.pop())  # 3

    # Flattening nested list
    nested = [[1, 2], [3, 4], [5, 6]]
    flat = [x for row in nested for x in row]
    print("Flattened:", flat)


# ---------------------------------------------------------------------------
# 2. Tuples
# ---------------------------------------------------------------------------

Point = namedtuple("Point", ["x", "y"])


def demonstrate_tuples() -> None:
    """Tuples as immutable records and namedtuples."""
    coord: tuple[int, int, int] = (10, 20, 30)
    x, y, z = coord  # unpacking
    print(f"Unpacked: x={x}, y={y}, z={z}")

    p = Point(3, 4)
    distance = (p.x ** 2 + p.y ** 2) ** 0.5
    print(f"Point {p}, distance from origin: {distance:.2f}")

    # Tuple as dict key (hashable)
    grid: dict[tuple[int, int], str] = {(0, 0): "origin", (1, 0): "east"}
    print("Grid:", grid)


# ---------------------------------------------------------------------------
# 3. Dictionaries
# ---------------------------------------------------------------------------

def word_frequency(text: str) -> dict[str, int]:
    """Count word frequencies in *text* (case-insensitive)."""
    freq: dict[str, int] = {}
    for word in text.lower().split():
        word = word.strip(".,!?;:")
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))


def demonstrate_dicts() -> None:
    """Dict operations, comprehensions, and defaultdict."""
    sample = "to be or not to be that is the question to be"
    freq = word_frequency(sample)
    print("Word frequencies:", freq)

    # defaultdict
    graph: defaultdict[str, list[str]] = defaultdict(list)
    edges = [("A", "B"), ("A", "C"), ("B", "D")]
    for src, dst in edges:
        graph[src].append(dst)
    print("Adjacency list:", dict(graph))

    # Dict merging (Python 3.9+)
    base = {"a": 1, "b": 2}
    extra = {"b": 99, "c": 3}
    merged = base | extra
    print("Merged:", merged)

    # Invert a dict
    original: dict[str, int] = {"one": 1, "two": 2, "three": 3}
    inverted = {v: k for k, v in original.items()}
    print("Inverted:", inverted)


# ---------------------------------------------------------------------------
# 4. Sets
# ---------------------------------------------------------------------------

def demonstrate_sets() -> None:
    """Set operations: union, intersection, difference."""
    python_devs = {"Alice", "Bob", "Carol", "Dave"}
    js_devs     = {"Bob", "Eve", "Frank", "Alice"}

    print("Both languages:    ", python_devs & js_devs)
    print("Either language:   ", python_devs | js_devs)
    print("Python only:       ", python_devs - js_devs)
    print("Not both:          ", python_devs ^ js_devs)

    # Fast membership testing
    allowed = frozenset(["admin", "editor", "viewer"])
    role = "editor"
    print(f"Role '{role}' allowed:", role in allowed)

    # Remove duplicates preserving order
    items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    unique = list(dict.fromkeys(items))
    print("Unique (ordered):", unique)


# ---------------------------------------------------------------------------
# 5. collections module
# ---------------------------------------------------------------------------

def demonstrate_collections() -> None:
    """Counter, deque, and heapq from the standard library."""
    # Counter
    votes = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
    tally = Counter(votes)
    print("Vote tally:", tally)
    print("Top 2:", tally.most_common(2))

    # deque (double-ended queue)
    dq: deque[int] = deque([1, 2, 3], maxlen=5)
    dq.appendleft(0)
    dq.append(4)
    dq.append(5)   # triggers maxlen eviction
    print("Deque:", dq)

    # heapq – priority queue
    heap: list[tuple[int, str]] = []
    tasks = [(3, "low"), (1, "critical"), (2, "medium")]
    for priority, task in tasks:
        heapq.heappush(heap, (priority, task))
    while heap:
        p, t = heapq.heappop(heap)
        print(f"  Processing [{p}] {t}")


# ---------------------------------------------------------------------------
# 6. Practical: inventory management
# ---------------------------------------------------------------------------

Inventory = dict[str, dict[str, Any]]


def build_inventory(items: list[tuple[str, int, float]]) -> Inventory:
    """
    Build an inventory mapping from a list of (name, qty, price) tuples.

    Returns:
        dict mapping item name → {"qty": int, "price": float, "total": float}
    """
    return {
        name: {"qty": qty, "price": price, "total": round(qty * price, 2)}
        for name, qty, price in items
    }


def low_stock(inventory: Inventory, threshold: int = 5) -> list[str]:
    """Return item names with quantity below *threshold*."""
    return [name for name, data in inventory.items() if data["qty"] < threshold]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all data structure demonstrations."""
    print("=== Lists ===")
    demonstrate_lists()

    print("\n=== Tuples ===")
    demonstrate_tuples()

    print("\n=== Dicts ===")
    demonstrate_dicts()

    print("\n=== Sets ===")
    demonstrate_sets()

    print("\n=== Collections ===")
    demonstrate_collections()

    print("\n=== Inventory ===")
    raw = [("apple", 50, 0.99), ("mango", 3, 2.49), ("banana", 2, 0.59)]
    inv = build_inventory(raw)
    for name, data in inv.items():
        print(f"  {name}: {data}")
    print("Low stock:", low_stock(inv))


if __name__ == "__main__":
    main()

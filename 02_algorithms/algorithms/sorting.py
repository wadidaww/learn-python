"""
algorithms/sorting.py
======================
Classic sorting algorithms from scratch with complexity annotations.

All functions sort in ascending order by default.

| Algorithm  | Best     | Average  | Worst    | Space   | Stable |
|------------|----------|----------|----------|---------|--------|
| Quicksort  | O(n log n)| O(n log n)| O(n²)  | O(log n)| No     |
| Mergesort  | O(n log n)| O(n log n)| O(n log n)| O(n) | Yes    |
| Heapsort   | O(n log n)| O(n log n)| O(n log n)| O(1) | No     |
"""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Quicksort — O(n log n) average, O(n²) worst
# ---------------------------------------------------------------------------

def quicksort(arr: list[T]) -> list[T]:
    """
    Return a new sorted list using quicksort with random pivot.

    Uses the functional (out-of-place) style for clarity.
    See ``quicksort_inplace`` for the space-efficient in-place version.
    """
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[len(arr) // 2]
    left   = [x for x in arr if x < pivot]   # type: ignore[operator]
    middle = [x for x in arr if x == pivot]
    right  = [x for x in arr if x > pivot]   # type: ignore[operator]
    return quicksort(left) + middle + quicksort(right)


def _partition(arr: list[T], low: int, high: int) -> int:
    """Lomuto partition scheme with random pivot."""
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:  # type: ignore[operator]
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort_inplace(arr: list[T], low: int = 0, high: int | None = None) -> None:
    """
    Sort *arr* in place using randomised quicksort. O(n log n) average.
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot = _partition(arr, low, high)
        quicksort_inplace(arr, low, pivot - 1)
        quicksort_inplace(arr, pivot + 1, high)


# ---------------------------------------------------------------------------
# Mergesort — O(n log n) always, O(n) extra space, stable
# ---------------------------------------------------------------------------

def mergesort(arr: list[T]) -> list[T]:
    """Return a new sorted list using mergesort. O(n log n), O(n) space."""
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left  = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return _merge(left, right)


def _merge(left: list[T], right: list[T]) -> list[T]:
    """Merge two sorted lists into one sorted list."""
    result: list[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # type: ignore[operator]
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ---------------------------------------------------------------------------
# Heapsort — O(n log n) always, O(1) extra space, not stable
# ---------------------------------------------------------------------------

def _heapify(arr: list[T], n: int, root: int) -> None:
    """Max-heapify the subtree rooted at *root* in a heap of size *n*."""
    largest = root
    left, right = 2 * root + 1, 2 * root + 2

    if left < n and arr[left] > arr[largest]:  # type: ignore[operator]
        largest = left
    if right < n and arr[right] > arr[largest]:  # type: ignore[operator]
        largest = right

    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        _heapify(arr, n, largest)


def heapsort(arr: list[T]) -> list[T]:
    """Return a new sorted list using heapsort. O(n log n), O(1) extra space."""
    result = list(arr)
    n = len(result)

    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(result, n, i)

    # Extract elements
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        _heapify(result, i, 0)

    return result


# ---------------------------------------------------------------------------
# Insertion sort — O(n²) but O(n) for nearly-sorted data
# ---------------------------------------------------------------------------

def insertion_sort(arr: list[T]) -> list[T]:
    """Return a new sorted list using insertion sort. Best for small or nearly-sorted arrays."""
    result = list(arr)
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:  # type: ignore[operator]
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Compare sorting algorithms on a random dataset."""
    import time

    data = random.sample(range(10_000), 1_000)
    expected = sorted(data)

    for name, fn in [
        ("quicksort",      lambda d: quicksort(d)),
        ("mergesort",      lambda d: mergesort(d)),
        ("heapsort",       lambda d: heapsort(d)),
        ("insertion_sort", lambda d: insertion_sort(d[:200])),  # only 200 for speed
    ]:
        start = time.perf_counter()
        result = fn(data if name != "insertion_sort" else data[:200])
        elapsed = time.perf_counter() - start
        correct = result == (expected if name != "insertion_sort" else sorted(data[:200]))
        print(f"  {name:<20} {'✓' if correct else '✗'}  {elapsed*1000:.2f} ms")


if __name__ == "__main__":
    main()

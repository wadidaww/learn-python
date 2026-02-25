"""
data_structures/heap.py
========================
Min-heap and max-heap implementations from scratch using a dynamic array.

Time complexity:
    - push:   O(log n)
    - pop:    O(log n)
    - peek:   O(1)
    - heapify from list: O(n)
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class MinHeap(Generic[T]):
    """
    A binary min-heap.

    The smallest element is always at the root and can be accessed in O(1).

    Example::

        h: MinHeap[int] = MinHeap()
        h.push(5); h.push(1); h.push(3)
        assert h.pop() == 1
    """

    def __init__(self) -> None:
        self._data: list[T] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = self._parent(i)
            if self._data[i] < self._data[parent]:  # type: ignore[operator]
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._data)
        while True:
            smallest = i
            left, right = self._left(i), self._right(i)
            if left < n and self._data[left] < self._data[smallest]:  # type: ignore[operator]
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:  # type: ignore[operator]
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, value: T) -> None:
        """Insert *value* into the heap."""
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> T:
        """Remove and return the minimum element. Raises IndexError if empty."""
        if not self._data:
            raise IndexError("pop from empty heap")
        self._swap(0, len(self._data) - 1)
        minimum = self._data.pop()
        if self._data:
            self._sift_down(0)
        return minimum

    def peek(self) -> T:
        """Return the minimum without removing it. Raises IndexError if empty."""
        if not self._data:
            raise IndexError("peek at empty heap")
        return self._data[0]

    @classmethod
    def heapify(cls, data: list[T]) -> MinHeap[T]:
        """Build a heap from *data* in O(n) time."""
        h: MinHeap[T] = cls()
        h._data = list(data)
        # Sift down all non-leaf nodes
        for i in range(len(h._data) // 2 - 1, -1, -1):
            h._sift_down(i)
        return h

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __repr__(self) -> str:
        return f"MinHeap({self._data!r})"


class MaxHeap(Generic[T]):
    """
    A binary max-heap built on top of MinHeap by negating values.

    Restricted to numeric types (int / float).
    For custom comparators use MinHeap with tuples (-priority, value).

    Example::

        h: MaxHeap[int] = MaxHeap()
        h.push(5); h.push(1); h.push(3)
        assert h.pop() == 5
    """

    def __init__(self) -> None:
        self._inner: MinHeap[T] = MinHeap()

    def push(self, value: T) -> None:
        """Insert *value*."""
        self._inner.push(-value)  # type: ignore[operator]

    def pop(self) -> T:
        """Remove and return the maximum element."""
        return -self._inner.pop()  # type: ignore[operator]

    def peek(self) -> T:
        """Return the maximum without removing it."""
        return -self._inner.peek()  # type: ignore[operator]

    @classmethod
    def heapify(cls, data: list[T]) -> MaxHeap[T]:
        """Build a max-heap from *data* in O(n) time."""
        h: MaxHeap[T] = cls()
        h._inner = MinHeap.heapify([-v for v in data])  # type: ignore[misc]
        return h

    def __len__(self) -> int:
        return len(self._inner)

    def __bool__(self) -> bool:
        return bool(self._inner)

    def __repr__(self) -> str:
        return f"MaxHeap({[-v for v in self._inner._data]!r})"


def nlargest(n: int, data: list[T]) -> list[T]:
    """Return the *n* largest values from *data* using a min-heap. O(k log n)."""
    if n <= 0:
        return []
    h: MinHeap[T] = MinHeap()
    for item in data:
        h.push(item)
        if len(h) > n:
            h.pop()
    result = []
    while h:
        result.append(h.pop())
    return list(reversed(result))


def nsmallest(n: int, data: list[T]) -> list[T]:
    """Return the *n* smallest values from *data* using a max-heap. O(k log n)."""
    if n <= 0:
        return []
    h: MaxHeap[T] = MaxHeap()
    for item in data:  # type: ignore[assignment]
        h.push(item)  # type: ignore[arg-type]
        if len(h) > n:
            h.pop()
    result = []
    while h:
        result.append(h.pop())
    return list(reversed(result))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate heap usage."""
    import random
    data = random.sample(range(100), 20)
    print("Data:     ", data)

    h = MinHeap.heapify(data)
    sorted_data = []
    while h:
        sorted_data.append(h.pop())
    print("Heap-sort:", sorted_data)

    mh = MaxHeap.heapify(data)
    print("Top 5 (nlargest):", nlargest(5, data))
    print("Bot 5 (nsmallest):", nsmallest(5, data))


if __name__ == "__main__":
    main()

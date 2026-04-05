"""
data_structures/hash_table.py
==============================
Hash table implementation with separate chaining for collision resolution.

Time complexity:
    - Insert / get / delete: O(1) average, O(n) worst case
    - Space: O(n)
"""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_DELETED = object()  # sentinel for deleted slots (not used in chaining, but kept for clarity)


class _Node(Generic[K, V]):
    """A node in a hash table chain."""

    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V, next_node: _Node[K, V] | None = None) -> None:
        self.key = key
        self.value = value
        self.next = next_node


class HashTable(Generic[K, V]):
    """
    Hash table using separate chaining.

    Supports arbitrary hashable keys and any value type.
    Automatically resizes when load factor exceeds 0.75.

    Example::

        ht: HashTable[str, int] = HashTable()
        ht["apple"] = 5
        ht["banana"] = 3
        assert ht["apple"] == 5
        del ht["banana"]
    """

    DEFAULT_CAPACITY = 16
    LOAD_FACTOR_THRESHOLD = 0.75
    RESIZE_FACTOR = 2

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be >= 1")
        self._capacity = capacity
        self._size = 0
        self._buckets: list[_Node[K, V] | None] = [None] * capacity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_index(self, key: K) -> int:
        """Return the bucket index for *key*."""
        return hash(key) % self._capacity

    def _resize(self) -> None:
        """Double the capacity and rehash all existing entries."""
        old_buckets = self._buckets
        self._capacity *= self.RESIZE_FACTOR
        self._buckets = [None] * self._capacity
        self._size = 0
        for head in old_buckets:
            node = head
            while node is not None:
                self[node.key] = node.value
                node = node.next

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __setitem__(self, key: K, value: V) -> None:
        """Insert or update *key* → *value*."""
        if self._size / self._capacity >= self.LOAD_FACTOR_THRESHOLD:
            self._resize()

        idx = self._bucket_index(key)
        node = self._buckets[idx]
        while node is not None:
            if node.key == key:
                node.value = value
                return
            node = node.next

        # Prepend new node (O(1))
        self._buckets[idx] = _Node(key, value, self._buckets[idx])
        self._size += 1

    def __getitem__(self, key: K) -> V:
        """Return value for *key*; raise KeyError if absent."""
        idx = self._bucket_index(key)
        node = self._buckets[idx]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        raise KeyError(key)

    def __delitem__(self, key: K) -> None:
        """Remove *key*; raise KeyError if absent."""
        idx = self._bucket_index(key)
        prev: _Node[K, V] | None = None
        node = self._buckets[idx]
        while node is not None:
            if node.key == key:
                if prev is None:
                    self._buckets[idx] = node.next
                else:
                    prev.next = node.next
                self._size -= 1
                return
            prev = node
            node = node.next
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Return True if *key* is present."""
        try:
            self[key]  # type: ignore[index]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return self._size

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return value for *key* or *default* if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Iterator[K]:
        """Iterate over all keys."""
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.key
                node = node.next

    def values(self) -> Iterator[V]:
        """Iterate over all values."""
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.value
                node = node.next

    def items(self) -> Iterator[tuple[K, V]]:
        """Iterate over all (key, value) pairs."""
        for head in self._buckets:
            node = head
            while node is not None:
                yield node.key, node.value
                node = node.next

    @property
    def load_factor(self) -> float:
        """Current load factor (size / capacity)."""
        return self._size / self._capacity

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"HashTable({{{pairs}}})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate HashTable usage."""
    ht: HashTable[str, int] = HashTable(capacity=4)

    words = "the quick brown fox jumps over the lazy dog".split()
    for word in words:
        ht[word] = ht.get(word, 0) + 1  # type: ignore[assignment]

    print("Word counts:")
    for k, v in sorted(ht.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:<10} {v}")

    print(f"\nSize: {len(ht)}, Load factor: {ht.load_factor:.2f}")

    del ht["the"]
    print(f"After deleting 'the': {len(ht)} entries")


if __name__ == "__main__":
    main()

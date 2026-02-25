"""
data_structures/lru_cache.py
=============================
LRU (Least-Recently-Used) Cache with O(1) get and put.

Two implementations:
1. ``LRUCacheOrderedDict`` – uses collections.OrderedDict (simple, idiomatic).
2. ``LRUCacheLinkedList`` – doubly-linked list + hash map (interview-style).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Implementation 1: OrderedDict
# ---------------------------------------------------------------------------

class LRUCacheOrderedDict(Generic[K, V]):
    """
    LRU cache backed by ``collections.OrderedDict``.

    Example::

        cache: LRUCacheOrderedDict[str, int] = LRUCacheOrderedDict(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")   # touch "a" → recent
        cache.put("c", 3)
        cache.put("d", 4)  # evicts "b" (least recently used)
        assert cache.get("b") is None
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be >= 1")
        self._capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        """Return the value for *key*, or None if not present. Marks as recent."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* → *value*. Evicts LRU entry if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)  # evict oldest

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def __repr__(self) -> str:
        items = list(self._cache.items())
        return f"LRUCache(capacity={self._capacity}, items={items!r})"


# ---------------------------------------------------------------------------
# Implementation 2: Doubly-linked list + hashmap
# ---------------------------------------------------------------------------

class _DLLNode(Generic[K, V]):
    """Node in a doubly-linked list."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.prev: _DLLNode[K, V] | None = None
        self.next: _DLLNode[K, V] | None = None


class LRUCacheLinkedList(Generic[K, V]):
    """
    LRU cache using a doubly-linked list plus a hash map for O(1) operations.

    The list is ordered most-recent → least-recent.
    ``_head.next`` is the most-recently used, ``_tail.prev`` is the LRU.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be >= 1")
        self._capacity = capacity
        self._map: dict[K, _DLLNode[K, V]] = {}

        # Sentinel nodes avoid edge-case handling
        self._head: _DLLNode[K, V] = _DLLNode(None, None)  # type: ignore[arg-type]
        self._tail: _DLLNode[K, V] = _DLLNode(None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove(self, node: _DLLNode[K, V]) -> None:
        """Unlink *node* from the list."""
        prev, nxt = node.prev, node.next
        assert prev is not None and nxt is not None
        prev.next = nxt
        nxt.prev = prev

    def _insert_after_head(self, node: _DLLNode[K, V]) -> None:
        """Insert *node* right after the head sentinel (marks as most recent)."""
        node.next = self._head.next
        node.prev = self._head
        assert self._head.next is not None
        self._head.next.prev = node
        self._head.next = node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: K) -> V | None:
        """Return the value for *key*, or None if absent. Marks as recent."""
        node = self._map.get(key)
        if node is None:
            return None
        self._remove(node)
        self._insert_after_head(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* → *value*. Evicts LRU if at capacity."""
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._insert_after_head(node)
            return

        new_node: _DLLNode[K, V] = _DLLNode(key, value)
        self._map[key] = new_node
        self._insert_after_head(new_node)

        if len(self._map) > self._capacity:
            # Evict the LRU (just before tail)
            lru = self._tail.prev
            assert lru is not None and lru is not self._head
            self._remove(lru)
            del self._map[lru.key]

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: object) -> bool:
        return key in self._map

    def __repr__(self) -> str:
        items: list[tuple[K, V]] = []
        node = self._head.next
        while node is not self._tail:
            assert node is not None
            items.append((node.key, node.value))
            node = node.next
        return f"LRUCacheLinkedList(capacity={self._capacity}, items={items!r})"


# Expose a single alias
LRUCache = LRUCacheOrderedDict


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate both LRU cache implementations."""
    for CacheClass in [LRUCacheOrderedDict, LRUCacheLinkedList]:
        print(f"\n=== {CacheClass.__name__} ===")
        cache: LRUCacheOrderedDict[str, int] = CacheClass(capacity=3)  # type: ignore[assignment]
        for k, v in [("a", 1), ("b", 2), ("c", 3)]:
            cache.put(k, v)
        print("After adding a,b,c:", cache)
        cache.get("a")  # touch a → a is now MRU
        cache.put("d", 4)  # evict LRU (b)
        print("After accessing 'a' and adding 'd':", cache)
        print(f"  get('b') = {cache.get('b')!r}  (evicted)")
        print(f"  get('a') = {cache.get('a')!r}")


if __name__ == "__main__":
    main()

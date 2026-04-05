"""
tests/test_data_structures.py
==============================
pytest tests for all data structure implementations in 02_algorithms.
"""

from __future__ import annotations

import pytest

from data_structures.hash_table import HashTable
from data_structures.heap import MaxHeap, MinHeap, nlargest, nsmallest
from data_structures.lru_cache import LRUCacheLinkedList, LRUCacheOrderedDict
from data_structures.trie import Trie
from algorithms.sorting import heapsort, mergesort, quicksort
from algorithms.graph import bfs, bfs_shortest_path, dijkstra, topological_sort
from algorithms.dynamic_programming import (
    coin_change,
    fibonacci_tabulation,
    knapsack_01,
    lcs,
    lis_length,
)


# ---------------------------------------------------------------------------
# HashTable
# ---------------------------------------------------------------------------

class TestHashTable:
    def test_set_and_get(self) -> None:
        ht: HashTable[str, int] = HashTable()
        ht["a"] = 1
        assert ht["a"] == 1

    def test_update(self) -> None:
        ht: HashTable[str, int] = HashTable()
        ht["a"] = 1
        ht["a"] = 99
        assert ht["a"] == 99
        assert len(ht) == 1

    def test_delete(self) -> None:
        ht: HashTable[str, int] = HashTable()
        ht["a"] = 1
        del ht["a"]
        assert "a" not in ht
        assert len(ht) == 0

    def test_key_error(self) -> None:
        ht: HashTable[str, int] = HashTable()
        with pytest.raises(KeyError):
            _ = ht["missing"]

    def test_resize(self) -> None:
        ht: HashTable[str, int] = HashTable(capacity=2)
        for i in range(20):
            ht[str(i)] = i
        assert len(ht) == 20
        for i in range(20):
            assert ht[str(i)] == i

    def test_iteration(self) -> None:
        ht: HashTable[str, int] = HashTable()
        for i in range(5):
            ht[str(i)] = i
        assert sorted(ht.keys()) == [str(i) for i in range(5)]
        assert sorted(ht.values()) == list(range(5))


# ---------------------------------------------------------------------------
# MinHeap / MaxHeap
# ---------------------------------------------------------------------------

class TestMinHeap:
    def test_push_pop(self) -> None:
        h: MinHeap[int] = MinHeap()
        for v in [5, 3, 8, 1]:
            h.push(v)
        assert h.pop() == 1
        assert h.pop() == 3

    def test_heapify(self) -> None:
        data = [5, 3, 8, 1, 9, 2]
        h = MinHeap.heapify(data)
        assert h.pop() == 1

    def test_empty_pop(self) -> None:
        with pytest.raises(IndexError):
            MinHeap[int]().pop()

    def test_heap_sort(self) -> None:
        import random
        data = random.sample(range(100), 20)
        h = MinHeap.heapify(data)
        result = []
        while h:
            result.append(h.pop())
        assert result == sorted(data)


class TestMaxHeap:
    def test_push_pop(self) -> None:
        h: MaxHeap[int] = MaxHeap()
        for v in [5, 3, 8, 1]:
            h.push(v)
        assert h.pop() == 8
        assert h.pop() == 5

    def test_nlargest(self) -> None:
        data = list(range(10))
        assert sorted(nlargest(3, data)) == [7, 8, 9]

    def test_nsmallest(self) -> None:
        data = list(range(10))
        assert sorted(nsmallest(3, data)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class TestTrie:
    def test_insert_search(self) -> None:
        t = Trie()
        t.insert("apple")
        assert t.search("apple")
        assert not t.search("app")

    def test_prefix(self) -> None:
        t = Trie()
        t.insert("apple")
        assert t.starts_with("app")
        assert not t.starts_with("xyz")

    def test_words_with_prefix(self) -> None:
        t = Trie()
        for w in ["apple", "app", "apt", "bat"]:
            t.insert(w)
        assert sorted(t.words_with_prefix("ap")) == ["app", "apple", "apt"]

    def test_delete(self) -> None:
        t = Trie()
        t.insert("apple")
        t.insert("app")
        assert t.delete("apple")
        assert not t.search("apple")
        assert t.search("app")

    def test_len(self) -> None:
        t = Trie()
        words = ["a", "b", "c", "ab"]
        for w in words:
            t.insert(w)
        assert len(t) == 4
        t.delete("a")
        assert len(t) == 3


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CacheClass", [LRUCacheOrderedDict, LRUCacheLinkedList])
class TestLRUCache:
    def test_basic_get_put(self, CacheClass: type) -> None:
        cache = CacheClass(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_eviction(self, CacheClass: type) -> None:
        cache = CacheClass(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)   # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_touch_prevents_eviction(self, CacheClass: type) -> None:
        cache = CacheClass(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")       # touch a → a is now MRU
        cache.put("c", 3)    # evicts b (LRU)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_update_existing(self, CacheClass: type) -> None:
        cache = CacheClass(capacity=2)
        cache.put("a", 1)
        cache.put("a", 99)
        assert cache.get("a") == 99
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    @pytest.mark.parametrize("fn", [quicksort, mergesort, heapsort])
    def test_sorts_correctly(self, fn: object) -> None:
        import random
        data = random.sample(range(1000), 50)
        assert fn(data) == sorted(data)  # type: ignore[operator]

    def test_empty(self) -> None:
        assert quicksort([]) == []
        assert mergesort([]) == []
        assert heapsort([]) == []

    def test_single(self) -> None:
        assert quicksort([42]) == [42]

    def test_duplicates(self) -> None:
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        assert quicksort(data) == sorted(data)


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

class TestGraph:
    UGRAPH = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A"],
        "D": ["B"],
    }

    def test_bfs_visits_all(self) -> None:
        visited = bfs(self.UGRAPH, "A")
        assert sorted(visited) == ["A", "B", "C", "D"]

    def test_bfs_shortest_path(self) -> None:
        path = bfs_shortest_path(self.UGRAPH, "A", "D")
        assert path == ["A", "B", "D"]

    def test_bfs_no_path(self) -> None:
        disconnected = {"A": ["B"], "B": [], "C": []}
        assert bfs_shortest_path(disconnected, "A", "C") is None

    def test_dijkstra(self) -> None:
        wg = {
            "A": [("B", 1), ("C", 4)],
            "B": [("C", 2), ("D", 5)],
            "C": [("D", 1)],
            "D": [],
        }
        dist, _ = dijkstra(wg, "A")
        assert dist["D"] == 4   # A→B→C→D = 1+2+1

    def test_topological_sort(self) -> None:
        dag = {"A": ["C"], "B": ["C"], "C": []}
        order = topological_sort(dag)
        assert order is not None
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("C")

    def test_topological_sort_cycle(self) -> None:
        cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
        assert topological_sort(cyclic) is None


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------

class TestDynamicProgramming:
    def test_fibonacci(self) -> None:
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, val in enumerate(expected):
            assert fibonacci_tabulation(i) == val

    def test_knapsack(self) -> None:
        max_val, items = knapsack_01(8, [2, 3, 4, 5], [3, 4, 5, 6])
        assert max_val == 10

    def test_lcs(self) -> None:
        assert lcs("ABCBDAB", "BDCAB") in {"BDAB", "BCAB", "BCBA", "BDAB"}
        assert len(lcs("ABCBDAB", "BDCAB")) == 4

    def test_coin_change(self) -> None:
        assert coin_change([1, 5, 6, 9], 11) == 2  # 5+6
        assert coin_change([2], 3) == -1

    def test_lis_length(self) -> None:
        assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
        assert lis_length([0, 1, 0, 3, 2, 3]) == 4

# Module 02: Algorithms & Data Structures

Implementations from scratch — understand the internals before using built-ins.

## Topics Covered

### Data Structures
| File | What it implements |
|------|--------------------|
| `data_structures/hash_table.py` | Hash table with separate chaining |
| `data_structures/heap.py` | Min-heap and max-heap |
| `data_structures/trie.py` | Prefix trie |
| `data_structures/lru_cache.py` | LRU cache (OrderedDict + raw doubly-linked list) |

### Algorithms
| File | What it implements |
|------|--------------------|
| `algorithms/sorting.py` | Quicksort, mergesort, heapsort |
| `algorithms/graph.py` | BFS, DFS, Dijkstra, topological sort |
| `algorithms/dynamic_programming.py` | Fibonacci, knapsack, LCS, coin change |

## Running

```bash
python data_structures/hash_table.py
pytest tests/ -v
```

## Complexity Cheat-Sheet

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Hash table | O(1) avg | O(1) avg | O(1) avg | O(1) avg |
| Min-heap | O(1) min | O(n) | O(log n) | O(log n) |
| Trie | O(m) | O(m) | O(m) | O(m) |
| LRU cache | O(1) | O(1) | O(1) | O(1) |

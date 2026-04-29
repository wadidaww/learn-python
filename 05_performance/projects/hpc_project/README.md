# HPC Project – High-Performance Computing in Python

A self-contained mini-project demonstrating **High-Performance Computing (HPC)
patterns** using only the Python standard library (no NumPy, no external C extensions).

## Project Layout

```
hpc_project/
├── __init__.py           Package marker & module docstring
├── benchmark.py          Reusable benchmarking harness (timeit + tracemalloc)
├── vectorized_ops.py     Typed arrays, memoryview, struct-packed buffers
├── memory_efficient.py   __slots__, generators, array vs list, ctypes structs
├── caching.py            lru_cache, manual memo, TimedCache, LFUCache
├── parallel_matrix.py    Matrix multiply: sequential → Pool → shared_memory
└── main.py               Entry-point – runs all demos
```

## Concepts Covered

| File | HPC Concept |
|---|---|
| `benchmark.py` | Micro-benchmarking, peak-memory measurement (`tracemalloc`) |
| `vectorized_ops.py` | SIMD-friendly loops via `array.array` + `memoryview`; `map`/`operator` over Python loops; sliding-window sum; binary encoding |
| `memory_efficient.py` | `__slots__` vs `__dict__` (~50 % memory saving); lazy generator pipelines; `array.array` vs `list`; `struct`/`ctypes.Structure` packed records |
| `caching.py` | `lru_cache` / `cache` decorator; manual `MemoDict` with hit-ratio tracking; TTL-based `TimedCache`; O(1) `LFUCache` |
| `parallel_matrix.py` | Row-partitioned parallel multiply with `multiprocessing.Pool`; zero-copy version with `multiprocessing.shared_memory` |

## Running

```bash
# Run the full demo suite
python main.py

# Run individual modules
python benchmark.py
python vectorized_ops.py
python memory_efficient.py
python caching.py
python parallel_matrix.py
```

## Key Takeaways

1. **Avoid per-element Python overhead** – `array.array` + `memoryview` keep
   data in C-level buffers; `map(operator.add, a, b)` avoids repeated
   `__getitem__` / `__add__` dispatch on Python objects.

2. **Memory layout matters** – `__slots__` eliminates the `__dict__` hash
   table from every instance; `struct` / `ctypes.Structure` give exact C-compatible
   layouts suitable for shared-memory IPC.

3. **Generator pipelines are O(1) memory** – multi-stage processing chains stay
   at constant heap regardless of input size.

4. **Cache the right thing** – `lru_cache` is ideal for pure functions with a
   bounded domain; `TimedCache` handles stale-by-age data; `LFUCache` suits
   workloads with a power-law access distribution.

5. **Parallelism has overheads** – for small matrices `Pool` spawn and pickle
   cost exceeds compute; `shared_memory` eliminates serialisation but adds
   setup cost; both shine only above a matrix size threshold.

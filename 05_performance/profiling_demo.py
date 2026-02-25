"""
profiling_demo.py
==================
Demonstrates Python profiling and benchmarking tools:
  - cProfile / pstats
  - timeit
  - Manual timing with time.perf_counter
  - Memory estimation patterns (no external deps)

Run:  python profiling_demo.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from functools import lru_cache
from typing import Any


# ---------------------------------------------------------------------------
# Subject code: slow vs fast implementations
# ---------------------------------------------------------------------------

def fibonacci_naive(n: int) -> int:
    """Exponential-time Fibonacci (intentionally slow)."""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """O(n) memoised Fibonacci with lru_cache."""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


def fibonacci_iterative(n: int) -> int:
    """O(n) time, O(1) space Fibonacci."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def slow_string_concat(parts: list[str]) -> str:
    """O(n²) string concatenation via += (creates new string each time)."""
    result = ""
    for part in parts:
        result += part
    return result


def fast_string_concat(parts: list[str]) -> str:
    """O(n) string join via str.join."""
    return "".join(parts)


def bubble_sort(arr: list[int]) -> list[int]:
    """O(n²) bubble sort (slow)."""
    a = list(arr)
    n = len(a)
    for i in range(n):
        for j in range(n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context manager and decorator for measuring elapsed time."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> bool:
        self.elapsed = time.perf_counter() - self._start
        return False


def benchmark(func: Any, *args: Any, repeat: int = 5) -> dict[str, float]:
    """
    Run *func(*args)* *repeat* times and return timing statistics.

    Returns dict with 'min', 'max', 'mean' in milliseconds.
    """
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)

    return {
        "min":  min(times) * 1000,
        "max":  max(times) * 1000,
        "mean": sum(times) / len(times) * 1000,
    }


def profile_func(func: Any, *args: Any) -> str:
    """Run *func(*args)* under cProfile and return a formatted stats string."""
    pr = cProfile.Profile()
    pr.enable()
    func(*args)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(10)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Memory estimation (no external deps)
# ---------------------------------------------------------------------------

def estimate_size(obj: Any) -> int:
    """Recursively estimate memory usage of a Python object in bytes."""
    seen: set[int] = set()

    def _size(o: Any) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        s = sys.getsizeof(o)
        if isinstance(o, dict):
            s += sum(_size(k) + _size(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            s += sum(_size(item) for item in o)
        return s

    return _size(obj)


# ---------------------------------------------------------------------------
# Profiling demonstration
# ---------------------------------------------------------------------------

def demo_fibonacci_comparison() -> None:
    """Compare naive vs cached vs iterative Fibonacci."""
    n = 30
    print("=== Fibonacci Benchmark (n=30) ===")

    with Timer("naive") as t:
        r = fibonacci_naive(n)
    print(f"  naive:     {r}  in {t.elapsed * 1000:.2f} ms")

    fibonacci_cached.cache_clear()
    with Timer("cached") as t:
        r = fibonacci_cached(n)
    print(f"  cached:    {r}  in {t.elapsed * 1000:.4f} ms")

    with Timer("iterative") as t:
        r = fibonacci_iterative(n)
    print(f"  iterative: {r}  in {t.elapsed * 1000:.4f} ms")


def demo_string_concat() -> None:
    """Compare slow += concat vs fast join."""
    parts = [f"part{i}" for i in range(5_000)]
    print("\n=== String Concatenation (5000 parts) ===")
    for label, fn in [("slow (+=)", slow_string_concat), ("fast (join)", fast_string_concat)]:
        stats = benchmark(fn, parts, repeat=10)
        print(f"  {label:<20} mean={stats['mean']:.3f} ms")


def demo_cprofile() -> None:
    """Run cProfile on Fibonacci and display top functions."""
    print("\n=== cProfile Output (fibonacci_naive n=25) ===")
    output = profile_func(fibonacci_naive, 25)
    # Show first 8 lines
    lines = output.strip().splitlines()
    for line in lines[:8]:
        print(" ", line)


def demo_memory_estimation() -> None:
    """Estimate the memory of various data structures."""
    print("\n=== Memory Estimation ===")
    objects: list[tuple[str, Any]] = [
        ("list of 1000 ints", list(range(1000))),
        ("dict 100 str→int",  {str(i): i for i in range(100)}),
        ("set of 500 ints",   set(range(500))),
        ("string 10k chars",  "x" * 10_000),
    ]
    for label, obj in objects:
        size = estimate_size(obj)
        print(f"  {label:<25} ~{size:>8,} bytes")


def main() -> None:
    """Run all profiling demonstrations."""
    demo_fibonacci_comparison()
    demo_string_concat()
    demo_cprofile()
    demo_memory_estimation()


if __name__ == "__main__":
    main()

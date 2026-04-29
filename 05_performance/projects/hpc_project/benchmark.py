"""
benchmark.py
============
Reusable benchmarking harness using only the Python standard library.

Features
--------
- ``timer`` context-manager / decorator – wall-clock elapsed time
- ``memory_snapshot`` context-manager – peak heap growth (tracemalloc)
- ``compare`` – run several callables, print a ranked comparison table
- ``BenchResult`` dataclass – structured result returned by every helper

Run standalone::

    python benchmark.py
"""

from __future__ import annotations

import gc
import statistics
import timeit
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Result produced by a single benchmark run."""

    name: str
    mean_s: float
    stdev_s: float
    min_s: float
    max_s: float
    peak_bytes: int
    iterations: int

    @property
    def mean_ms(self) -> float:
        """Mean elapsed time in milliseconds."""
        return self.mean_s * 1_000

    def __str__(self) -> str:
        return (
            f"{self.name:<35} "
            f"mean={self.mean_ms:8.3f} ms  "
            f"stdev={self.stdev_s * 1_000:7.3f} ms  "
            f"peak={self.peak_bytes / 1024:8.1f} KB  "
            f"n={self.iterations}"
        )


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def memory_snapshot() -> Any:
    """
    Context manager that records peak memory growth of the enclosed block.

    Usage::

        with memory_snapshot() as mem:
            do_work()
        print(mem.peak_bytes)
    """

    class _Mem:
        peak_bytes: int = 0

    result = _Mem()
    gc.collect()
    tracemalloc.start()
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        result.peak_bytes = peak
        tracemalloc.stop()


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def measure(
    fn: Callable[[], Any],
    *,
    name: str = "",
    iterations: int = 10,
    warmup: int = 2,
) -> BenchResult:
    """
    Measure *fn* over *iterations* runs and return a :class:`BenchResult`.

    Args:
        fn:         Zero-argument callable to benchmark.
        name:       Label used in output tables (defaults to ``fn.__name__``).
        iterations: Number of timed runs to average.
        warmup:     Number of un-timed warm-up runs before measurement.

    Returns:
        A populated :class:`BenchResult`.
    """
    label = name or getattr(fn, "__name__", repr(fn))

    # Warm-up: populate caches, JIT stubs, etc.
    for _ in range(warmup):
        fn()

    # Measure peak memory on a single representative run
    with memory_snapshot() as mem:
        fn()

    # Timed runs
    samples: list[float] = []
    for _ in range(iterations):
        gc.collect()
        t = timeit.timeit(fn, number=1)
        samples.append(t)

    return BenchResult(
        name=label,
        mean_s=statistics.mean(samples),
        stdev_s=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min_s=min(samples),
        max_s=max(samples),
        peak_bytes=mem.peak_bytes,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare(
    *fns: Callable[[], Any],
    names: list[str] | None = None,
    iterations: int = 10,
    warmup: int = 2,
) -> list[BenchResult]:
    """
    Benchmark several callables and print a ranked comparison table.

    Args:
        fns:        Zero-argument callables to compare.
        names:      Optional labels aligned with *fns*.
        iterations: Timed runs per callable.
        warmup:     Warm-up runs per callable.

    Returns:
        List of :class:`BenchResult` sorted fastest-first.
    """
    labels = (
        list(names)
        if names
        else [getattr(f, "__name__", f"fn_{i}") for i, f in enumerate(fns)]
    )
    results: list[BenchResult] = [
        measure(fn, name=label, iterations=iterations, warmup=warmup)
        for fn, label in zip(fns, labels, strict=True)
    ]
    results.sort(key=lambda r: r.mean_s)

    # Pretty-print table
    fastest = results[0].mean_s
    print(f"\n{'Benchmark':<35} {'mean':>10} {'stdev':>10} {'peak':>10}  speedup")
    print("-" * 80)
    for r in results:
        ratio = r.mean_s / fastest
        print(
            f"{r.name:<35} {r.mean_ms:9.3f}ms {r.stdev_s * 1_000:9.3f}ms "
            f"{r.peak_bytes / 1024:8.1f}KB  {ratio:.2f}x"
        )

    return results


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Self-test: compare list sum vs generator sum."""
    n = 100_000

    def list_sum() -> int:
        return sum([i * i for i in range(n)])

    def gen_sum() -> int:
        return sum(i * i for i in range(n))

    print("=== Benchmark harness self-test ===")
    compare(list_sum, gen_sum, iterations=5)


if __name__ == "__main__":
    _demo()

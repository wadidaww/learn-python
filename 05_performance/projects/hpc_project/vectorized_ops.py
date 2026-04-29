"""
vectorized_ops.py
=================
Demonstrates how to write fast, SIMD-friendly numeric code in Python
*without* external libraries such as NumPy.

Techniques shown
----------------
1. ``array.array``       – C-level typed arrays (no per-element boxing overhead)
2. ``memoryview``        – zero-copy buffer slicing / bulk mutation
3. ``map`` + built-ins  – avoid Python-level loops where possible
4. ``struct.pack_into``  – direct binary encoding into pre-allocated buffers

Each technique is compared against a naive Python list baseline using the
bundled :mod:`benchmark` module.

Run::

    python vectorized_ops.py
"""

from __future__ import annotations

import array
import math
import operator
import struct
from typing import Union

from benchmark import BenchResult, compare, measure

# Type alias for the numeric containers we handle
NumBuffer = Union[list[float], array.array, memoryview]  # noqa: UP007

# ---------------------------------------------------------------------------
# 1. Element-wise arithmetic
# ---------------------------------------------------------------------------

N = 50_000  # default problem size


def _make_list(n: int = N) -> list[float]:
    return [float(i) for i in range(n)]


def _make_array(n: int = N) -> array.array:  # type: ignore[type-arg]
    return array.array("d", range(n))


# --- addition (list) ---


def add_lists(a: list[float], b: list[float]) -> list[float]:
    """Element-wise addition using a list-comprehension."""
    return [x + y for x, y in zip(a, b)]


# --- addition (array + map) ---


def add_arrays(a: array.array, b: array.array) -> array.array:  # type: ignore[type-arg]
    """Element-wise addition using ``map`` and ``operator.add``."""
    result: array.array = array.array("d", map(operator.add, a, b))  # type: ignore[type-arg]
    return result


# --- in-place addition via memoryview ---


def add_inplace_memoryview(buf: array.array, delta: float) -> None:  # type: ignore[type-arg]
    """
    Add *delta* to every element of *buf* in-place using a memoryview.

    Because memoryview exposes the raw buffer, we avoid creating an
    intermediate list, keeping peak memory low.
    """
    mv = memoryview(buf)  # 'd' format – no cast needed
    for i in range(len(mv)):
        mv[i] += delta


# ---------------------------------------------------------------------------
# 2. Dot product
# ---------------------------------------------------------------------------


def dot_list(a: list[float], b: list[float]) -> float:
    """Naive dot product using a generator expression."""
    return sum(x * y for x, y in zip(a, b))


def dot_map(a: array.array, b: array.array) -> float:  # type: ignore[type-arg]
    """Dot product via ``map(operator.mul, …)`` – avoids generator overhead."""
    return sum(map(operator.mul, a, b))


def dot_array_direct(a: array.array, b: array.array) -> float:  # type: ignore[type-arg]
    """
    Dot product using ``math.fsum`` for improved floating-point precision,
    fed by a ``map`` iterator.
    """
    return math.fsum(map(operator.mul, a, b))


# ---------------------------------------------------------------------------
# 3. Norm (Euclidean length)
# ---------------------------------------------------------------------------


def norm_list(v: list[float]) -> float:
    """L2 norm via list-comprehension."""
    return math.sqrt(sum(x * x for x in v))


def norm_map(v: array.array) -> float:  # type: ignore[type-arg]
    """L2 norm via ``math.fsum(map(…))`` – accurate and fast."""
    return math.sqrt(math.fsum(map(operator.mul, v, v)))


# ---------------------------------------------------------------------------
# 4. Struct-packed binary buffer encoding
# ---------------------------------------------------------------------------


def encode_floats_struct(values: list[float]) -> bytes:
    """
    Pack a list of floats into a compact binary buffer using
    ``struct.pack``.  ~3× smaller than ``repr`` and much faster to
    decode than CSV.
    """
    fmt = f"{len(values)}d"
    return struct.pack(fmt, *values)


def encode_floats_array(values: list[float]) -> bytes:
    """Use ``array.tobytes`` – zero extra allocation vs struct for doubles."""
    buf: array.array = array.array("d", values)  # type: ignore[type-arg]
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 5. Chunked memoryview processing (sliding-window sum)
# ---------------------------------------------------------------------------


def sliding_sum_list(data: list[float], window: int) -> list[float]:
    """Sliding-window sum using plain Python lists."""
    result: list[float] = []
    total = sum(data[:window])
    result.append(total)
    for i in range(window, len(data)):
        total += data[i] - data[i - window]
        result.append(total)
    return result


def sliding_sum_array(data: array.array, window: int) -> array.array:  # type: ignore[type-arg]
    """
    Sliding-window sum operating on ``array.array`` with a running total.

    The key difference from the list version: the result buffer is
    pre-allocated with ``array.array``, avoiding repeated list appends.
    """
    n = len(data)
    result: array.array = array.array("d", [0.0] * (n - window + 1))  # type: ignore[type-arg]
    mv_in = memoryview(data)   # already 'd' (float64) format
    mv_out = memoryview(result)

    total = math.fsum(mv_in[:window])
    mv_out[0] = total
    for i in range(window, n):
        total += mv_in[i] - mv_in[i - window]
        mv_out[i - window + 1] = total
    return result


# ---------------------------------------------------------------------------
# Demo / benchmarks
# ---------------------------------------------------------------------------


def demo_addition() -> None:
    """Compare list vs array element-wise addition."""
    print("\n=== Element-wise addition ===")
    a_list, b_list = _make_list(), _make_list()
    a_arr, b_arr = _make_array(), _make_array()

    compare(
        lambda: add_lists(a_list, b_list),
        lambda: add_arrays(a_arr, b_arr),
        names=["add_lists (list comprehension)", "add_arrays (map + operator)"],
        iterations=8,
    )


def demo_dot_product() -> None:
    """Compare three dot-product implementations."""
    print("\n=== Dot product ===")
    a_list, b_list = _make_list(), _make_list()
    a_arr, b_arr = _make_array(), _make_array()

    compare(
        lambda: dot_list(a_list, b_list),
        lambda: dot_map(a_arr, b_arr),
        lambda: dot_array_direct(a_arr, b_arr),
        names=["dot_list (generator)", "dot_map (map+operator)", "dot_array (fsum)"],
        iterations=8,
    )


def demo_encoding() -> None:
    """Compare struct vs array binary encoding."""
    print("\n=== Binary encoding ===")
    values = _make_list()

    compare(
        lambda: encode_floats_struct(values),
        lambda: encode_floats_array(values),
        names=["encode struct.pack", "encode array.tobytes"],
        iterations=8,
    )


def demo_sliding_sum() -> None:
    """Compare list vs array sliding-window sum."""
    print("\n=== Sliding-window sum (window=100) ===")
    data_list = _make_list()
    data_arr = _make_array()
    W = 100

    compare(
        lambda: sliding_sum_list(data_list, W),
        lambda: sliding_sum_array(data_arr, W),
        names=["sliding_sum_list", "sliding_sum_array (memoryview)"],
        iterations=8,
    )


def main() -> None:
    """Run all vectorized-ops demonstrations."""
    print("=" * 60)
    print("  Vectorized Operations Demo")
    print("=" * 60)
    demo_addition()
    demo_dot_product()
    demo_encoding()
    demo_sliding_sum()


if __name__ == "__main__":
    main()

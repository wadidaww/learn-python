"""
main.py
=======
Entry-point for the HPC project.

Runs all demonstrations in sequence:
  1. Vectorized operations  (array, memoryview)
  2. Memory-efficient patterns (__slots__, generators, struct)
  3. Caching strategies  (lru_cache, TimedCache, LFUCache)
  4. Parallel matrix multiplication  (Pool, shared_memory)

Run::

    python main.py
"""

from __future__ import annotations

import sys
import time


def _section(title: str) -> None:
    print(f"\n{'#' * 60}")
    print(f"#  {title}")
    print(f"{'#' * 60}")


def main() -> None:
    """Run every HPC sub-module demo in sequence."""
    total_start = time.perf_counter()

    # -------------------------------------------------------------------
    # 1. Vectorized operations
    # -------------------------------------------------------------------
    _section("1 / 4  –  Vectorized Operations")
    try:
        from vectorized_ops import main as vec_main  # type: ignore[import]

        vec_main()
    except Exception as exc:
        print(f"  [ERROR] vectorized_ops: {exc}", file=sys.stderr)

    # -------------------------------------------------------------------
    # 2. Memory-efficient patterns
    # -------------------------------------------------------------------
    _section("2 / 4  –  Memory-Efficient Patterns")
    try:
        from memory_efficient import main as mem_main  # type: ignore[import]

        mem_main()
    except Exception as exc:
        print(f"  [ERROR] memory_efficient: {exc}", file=sys.stderr)

    # -------------------------------------------------------------------
    # 3. Caching strategies
    # -------------------------------------------------------------------
    _section("3 / 4  –  Caching Strategies")
    try:
        from caching import main as cache_main  # type: ignore[import]

        cache_main()
    except Exception as exc:
        print(f"  [ERROR] caching: {exc}", file=sys.stderr)

    # -------------------------------------------------------------------
    # 4. Parallel matrix multiplication
    # -------------------------------------------------------------------
    _section("4 / 4  –  Parallel Matrix Multiplication")
    try:
        from parallel_matrix import main as mat_main  # type: ignore[import]

        mat_main()
    except Exception as exc:
        print(f"  [ERROR] parallel_matrix: {exc}", file=sys.stderr)

    # -------------------------------------------------------------------
    elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(f"  All demos completed in {elapsed:.2f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

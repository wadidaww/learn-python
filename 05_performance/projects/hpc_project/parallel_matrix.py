"""
parallel_matrix.py
==================
Parallel dense matrix multiplication using the Python standard library.

Techniques
----------
- Naive O(n³) sequential multiplication (baseline)
- Row-partitioned parallel multiplication with ``multiprocessing.Pool``
- Shared-memory parallel multiplication with ``multiprocessing.shared_memory``
  (avoids pickling large matrices between processes)

All matrices are stored as flat ``array.array('d', …)`` buffers (row-major)
to minimise per-element overhead and to remain compatible with the shared
memory API.

Run::

    python parallel_matrix.py
"""

from __future__ import annotations

import array
import math
import multiprocessing as mp
import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def make_matrix(rows: int, cols: int, *, fill: float = 0.0) -> array.array:  # type: ignore[type-arg]
    """Return a row-major matrix as a flat ``array.array('d')``."""
    return array.array("d", [fill] * (rows * cols))


def identity(n: int) -> array.array:  # type: ignore[type-arg]
    """Return an n×n identity matrix."""
    m = make_matrix(n, n)
    for i in range(n):
        m[i * n + i] = 1.0
    return m


def mat_get(m: array.array, row: int, col: int, cols: int) -> float:  # type: ignore[type-arg]
    """Read element (row, col) from a flat row-major matrix."""
    return m[row * cols + col]


def mat_set(m: array.array, row: int, col: int, cols: int, val: float) -> None:  # type: ignore[type-arg]
    """Write element (row, col) into a flat row-major matrix."""
    m[row * cols + col] = val


def random_matrix(rows: int, cols: int, seed: int = 42) -> array.array:  # type: ignore[type-arg]
    """Generate a reproducible pseudo-random matrix using a simple LCG."""
    # LCG parameters (same as glibc)
    a, c, mod = 1_103_515_245, 12_345, 2**31
    state = seed
    out = make_matrix(rows, cols)
    for i in range(rows * cols):
        state = (a * state + c) % mod
        out[i] = (state % 1000) / 1000.0
    return out


# ---------------------------------------------------------------------------
# 1. Sequential matrix multiply
# ---------------------------------------------------------------------------


def matmul_sequential(
    a: array.array,  # type: ignore[type-arg]
    b: array.array,  # type: ignore[type-arg]
    n: int,
    m: int,
    k: int,
) -> array.array:  # type: ignore[type-arg]
    """
    Compute C = A × B where A is n×m and B is m×k.

    Returns a flat n×k ``array.array('d')``.
    This is the naive O(n·m·k) algorithm used as a baseline.
    """
    c = make_matrix(n, k)
    for i in range(n):
        for p in range(m):
            a_ip = a[i * m + p]
            if a_ip == 0.0:
                continue
            for j in range(k):
                c[i * k + j] += a_ip * b[p * k + j]
    return c


# ---------------------------------------------------------------------------
# 2. Parallel matrix multiply – row partitioning (Pool.map)
# ---------------------------------------------------------------------------


class RowChunk(NamedTuple):
    """Describes a row slice to be processed by a worker."""

    row_start: int
    row_end: int
    a_data: list[float]  # serialised rows of A
    b_data: list[float]  # full B
    n: int
    m: int
    k: int


def _compute_rows(chunk: RowChunk) -> tuple[int, int, list[float]]:
    """
    Worker function: multiply rows [row_start, row_end) of A by B.

    Returns ``(row_start, row_end, result_data)`` so the parent can
    reassemble the output matrix in the correct order.
    """
    row_start, row_end, a_data, b_data, _, m, k = chunk
    n_rows = row_end - row_start
    result: list[float] = [0.0] * (n_rows * k)

    for local_i in range(n_rows):
        for p in range(m):
            a_ip = a_data[local_i * m + p]
            if a_ip == 0.0:
                continue
            base_b = p * k
            base_r = local_i * k
            for j in range(k):
                result[base_r + j] += a_ip * b_data[base_b + j]

    return row_start, row_end, result


def matmul_parallel(
    a: array.array,  # type: ignore[type-arg]
    b: array.array,  # type: ignore[type-arg]
    n: int,
    m: int,
    k: int,
    num_workers: int | None = None,
) -> array.array:  # type: ignore[type-arg]
    """
    Parallel matrix multiply via ``multiprocessing.Pool``.

    Strategy: split *A* into row chunks and send each chunk to a worker.
    Workers receive serialised copies of their slice of A and all of B.
    Results are gathered and reassembled in order.

    Args:
        a:           Flat n×m matrix.
        b:           Flat m×k matrix.
        n, m, k:     Dimensions.
        num_workers: Pool size (defaults to CPU count).

    Returns:
        Flat n×k result matrix.
    """
    workers = num_workers or os.cpu_count() or 2
    chunk_size = max(1, math.ceil(n / workers))

    b_list = list(b)
    chunks: list[RowChunk] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a_slice = list(a[start * m : end * m])
        chunks.append(RowChunk(start, end, a_slice, b_list, n, m, k))

    c = make_matrix(n, k)
    with mp.Pool(processes=workers) as pool:
        for row_start, row_end, result_data in pool.map(_compute_rows, chunks):
            for local_i, global_i in enumerate(range(row_start, row_end)):
                base_r = local_i * k
                base_c = global_i * k
                for j in range(k):
                    c[base_c + j] = result_data[base_r + j]

    return c


# ---------------------------------------------------------------------------
# 3. Shared-memory matrix multiply
# ---------------------------------------------------------------------------


class _ShmChunkArgs(NamedTuple):
    """Arguments passed to shared-memory worker (all picklable scalars)."""

    shm_a_name: str
    shm_b_name: str
    shm_c_name: str
    row_start: int
    row_end: int
    n: int
    m: int
    k: int


def _shm_worker(args: _ShmChunkArgs) -> None:
    """
    Worker that reads A and B from shared memory, writes rows to C.

    No large data is pickled – only names and scalar dimensions are sent.
    """
    shm_a = SharedMemory(name=args.shm_a_name)
    shm_b = SharedMemory(name=args.shm_b_name)
    shm_c = SharedMemory(name=args.shm_c_name)

    try:
        a = array.array("d")
        b = array.array("d")
        c = array.array("d")
        a.frombytes(shm_a.buf[: args.n * args.m * 8])
        b.frombytes(shm_b.buf[: args.m * args.k * 8])
        c.frombytes(shm_c.buf[: args.n * args.k * 8])

        for i in range(args.row_start, args.row_end):
            for p in range(args.m):
                a_ip = a[i * args.m + p]
                if a_ip == 0.0:
                    continue
                for j in range(args.k):
                    c[i * args.k + j] += a_ip * b[p * args.k + j]

        # Write computed rows back to the shared buffer
        row_offset = args.row_start * args.k * 8
        row_bytes = (args.row_end - args.row_start) * args.k * 8
        result_slice = c[args.row_start * args.k : args.row_end * args.k]
        shm_c.buf[row_offset : row_offset + row_bytes] = result_slice.tobytes()
    finally:
        shm_a.close()
        shm_b.close()
        shm_c.close()


def matmul_shared_memory(
    a: array.array,  # type: ignore[type-arg]
    b: array.array,  # type: ignore[type-arg]
    n: int,
    m: int,
    k: int,
    num_workers: int | None = None,
) -> array.array:  # type: ignore[type-arg]
    """
    Parallel matrix multiply using ``multiprocessing.shared_memory``.

    Unlike :func:`matmul_parallel`, this version places *A*, *B*, and the
    output *C* in OS-level shared memory regions, so workers access them
    directly without any serialisation overhead.

    Args:
        a, b:        Input matrices (flat row-major).
        n, m, k:     Dimensions of A (n×m) and B (m×k).
        num_workers: Pool size (defaults to CPU count).

    Returns:
        Flat n×k result matrix.
    """
    workers = num_workers or os.cpu_count() or 2
    chunk_size = max(1, math.ceil(n / workers))

    shm_a = SharedMemory(create=True, size=a.itemsize * n * m)
    shm_b = SharedMemory(create=True, size=b.itemsize * m * k)
    shm_c = SharedMemory(create=True, size=a.itemsize * n * k)

    try:
        shm_a.buf[: len(a.tobytes())] = a.tobytes()
        shm_b.buf[: len(b.tobytes())] = b.tobytes()
        # Zero-init C
        shm_c.buf[: a.itemsize * n * k] = bytes(a.itemsize * n * k)

        chunk_args = [
            _ShmChunkArgs(
                shm_a_name=shm_a.name,
                shm_b_name=shm_b.name,
                shm_c_name=shm_c.name,
                row_start=start,
                row_end=min(start + chunk_size, n),
                n=n,
                m=m,
                k=k,
            )
            for start in range(0, n, chunk_size)
        ]

        with mp.Pool(processes=workers) as pool:
            pool.map(_shm_worker, chunk_args)

        c: array.array = array.array("d")  # type: ignore[type-arg]
        c.frombytes(bytes(shm_c.buf[: a.itemsize * n * k]))
        return c

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_c.close()
        shm_c.unlink()


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def matrices_close(
    a: array.array,  # type: ignore[type-arg]
    b: array.array,  # type: ignore[type-arg]
    tol: float = 1e-9,
) -> bool:
    """Return True if every element of *a* and *b* differs by less than *tol*."""
    return all(abs(x - y) < tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Benchmark sequential, Pool, and shared-memory matrix multiplication."""
    print("=" * 60)
    print("  Parallel Matrix Multiplication Demo")
    print("=" * 60)

    N, M, K = 64, 64, 64  # keep modest so the demo runs in a few seconds
    print(f"\n  Matrix dimensions: A={N}×{M}, B={M}×{K}")

    A = random_matrix(N, M)
    B = random_matrix(M, K)

    # --- sequential baseline ---
    t0 = time.perf_counter()
    C_seq = matmul_sequential(A, B, N, M, K)
    t_seq = time.perf_counter() - t0
    print(f"\n  Sequential:    {t_seq * 1000:.1f} ms")

    # --- parallel (Pool) ---
    t0 = time.perf_counter()
    C_par = matmul_parallel(A, B, N, M, K)
    t_par = time.perf_counter() - t0
    ok_par = matrices_close(C_seq, C_par)
    print(f"  Parallel Pool: {t_par * 1000:.1f} ms  correct={ok_par}")

    # --- shared memory ---
    t0 = time.perf_counter()
    C_shm = matmul_shared_memory(A, B, N, M, K)
    t_shm = time.perf_counter() - t0
    ok_shm = matrices_close(C_seq, C_shm)
    print(f"  Shared memory: {t_shm * 1000:.1f} ms  correct={ok_shm}")

    print(f"\n  Speedup (Pool vs seq):    {t_seq / t_par:.2f}x")
    print(f"  Speedup (Shm  vs seq):    {t_seq / t_shm:.2f}x")
    print(
        "\n  Note: for small matrices spawn/IPC overhead dominates."
        "\n  Increase N, M, K to see near-linear speedup on multi-core hardware."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

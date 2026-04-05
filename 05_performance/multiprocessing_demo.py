"""
multiprocessing_demo.py
========================
Demonstrates Python multiprocessing patterns:
  - Pool.map / Pool.starmap (CPU-bound parallelism)
  - Pool.imap_unordered (streaming results)
  - multiprocessing.Queue (inter-process communication)
  - shared Value / Array

Run:  python multiprocessing_demo.py
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from collections.abc import Iterator
from typing import Any


# ---------------------------------------------------------------------------
# CPU-bound work
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Deterministic primality test (trial division)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def count_primes_in_range(start: int, end: int) -> int:
    """Count primes in [start, end)."""
    return sum(1 for n in range(start, end) if is_prime(n))


def worker_task(chunk: tuple[int, int]) -> tuple[int, int, int]:
    """Process a (start, end) chunk and return (start, end, count)."""
    start, end = chunk
    return start, end, count_primes_in_range(start, end)


# ---------------------------------------------------------------------------
# Demo: Pool.map
# ---------------------------------------------------------------------------

def demo_pool_map(limit: int = 50_000) -> None:
    """Count primes below *limit* using a process pool."""
    chunk_size = limit // mp.cpu_count()
    chunks = [
        (i * chunk_size, (i + 1) * chunk_size)
        for i in range(mp.cpu_count())
    ]

    # Sequential baseline
    start = time.perf_counter()
    seq_count = count_primes_in_range(0, limit)
    seq_time = time.perf_counter() - start

    # Parallel with Pool
    start = time.perf_counter()
    with mp.Pool() as pool:
        results = pool.map(worker_task, chunks)
    par_time = time.perf_counter() - start
    par_count = sum(r[2] for r in results)

    print("=== Pool.map: Prime counting ===")
    print(f"  CPUs:       {mp.cpu_count()}")
    print(f"  Sequential: {seq_count} primes in {seq_time:.3f}s")
    print(f"  Parallel:   {par_count} primes in {par_time:.3f}s")
    print(f"  Speedup:    {seq_time / par_time:.2f}x")
    assert seq_count == par_count


# ---------------------------------------------------------------------------
# Demo: Pool.imap_unordered (streaming)
# ---------------------------------------------------------------------------

def demo_imap(numbers: list[int]) -> None:
    """Use imap_unordered to check primality of a batch of numbers."""
    print("\n=== Pool.imap_unordered ===")
    with mp.Pool() as pool:
        primes = [n for n, p in zip(numbers, pool.imap(is_prime, numbers)) if p]
    print(f"  Primes in {numbers[:5]}…: {primes[:10]}")


# ---------------------------------------------------------------------------
# Demo: multiprocessing.Queue
# ---------------------------------------------------------------------------

def queue_producer(q: "mp.Queue[int]", count: int) -> None:
    """Put *count* items into *q*."""
    for i in range(count):
        q.put(i)
    q.put(-1)  # sentinel


def queue_consumer(q: "mp.Queue[int]", result_list: "mp.Queue[list[int]]") -> None:
    """Drain *q* until sentinel and put collected items into *result_list*."""
    collected: list[int] = []
    while True:
        item = q.get()
        if item == -1:
            break
        collected.append(item)
    result_list.put(collected)


def demo_queue() -> None:
    """Demonstrate inter-process communication with Queue."""
    print("\n=== multiprocessing.Queue ===")
    q: mp.Queue[int] = mp.Queue()
    result_q: mp.Queue[list[int]] = mp.Queue()

    prod = mp.Process(target=queue_producer, args=(q, 10))
    cons = mp.Process(target=queue_consumer, args=(q, result_q))

    prod.start()
    cons.start()
    prod.join()
    cons.join()

    items = result_q.get()
    print(f"  Consumer received: {items}")


# ---------------------------------------------------------------------------
# Demo: shared memory Value
# ---------------------------------------------------------------------------

def increment_counter(counter: "mp.Value[int]", lock: mp.Lock, n: int) -> None:  # type: ignore[type-arg]
    """Increment a shared counter *n* times (process-safe)."""
    for _ in range(n):
        with lock:
            counter.value += 1


def demo_shared_value() -> None:
    """Demonstrate shared memory with multiprocessing.Value."""
    print("\n=== Shared Value ===")
    counter = mp.Value("i", 0)
    lock = mp.Lock()
    n_each = 1000
    procs = [
        mp.Process(target=increment_counter, args=(counter, lock, n_each))
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    expected = 4 * n_each
    print(f"  Expected: {expected}, Got: {counter.value}, Correct: {counter.value == expected}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run multiprocessing demonstrations."""
    demo_pool_map(limit=20_000)
    demo_imap(list(range(2, 100)))
    demo_queue()
    demo_shared_value()


if __name__ == "__main__":
    mp.freeze_support()  # needed on Windows
    main()

"""
async_demo.py
=============
Demonstrates Python asyncio patterns:
  - Coroutines and tasks
  - asyncio.gather (parallel execution)
  - asyncio.Queue (producer-consumer)
  - Semaphores (rate limiting)
  - Timeouts

Run:  python async_demo.py
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import Any


# ---------------------------------------------------------------------------
# 1. Basic coroutines and tasks
# ---------------------------------------------------------------------------

async def fetch_data(url: str, delay: float = 0.1) -> dict[str, Any]:
    """Simulate an async HTTP GET request."""
    await asyncio.sleep(delay)
    return {"url": url, "status": 200, "data": f"content from {url}"}


async def demo_tasks() -> None:
    """Run multiple coroutines concurrently with asyncio.gather."""
    urls = [f"https://api.example.com/item/{i}" for i in range(5)]

    print("=== Sequential fetch ===")
    start = time.perf_counter()
    for url in urls:
        result = await fetch_data(url)
    elapsed = time.perf_counter() - start
    print(f"  Sequential: {elapsed:.3f}s")

    print("=== Concurrent fetch (gather) ===")
    start = time.perf_counter()
    results = await asyncio.gather(*[fetch_data(url) for url in urls])
    elapsed = time.perf_counter() - start
    print(f"  Concurrent: {elapsed:.3f}s  ({len(results)} results)")


# ---------------------------------------------------------------------------
# 2. asyncio.Queue – producer / consumer
# ---------------------------------------------------------------------------

async def producer(queue: asyncio.Queue[int], count: int) -> None:
    """Put *count* items into *queue* with simulated work."""
    for i in range(count):
        await asyncio.sleep(random.uniform(0.01, 0.03))
        await queue.put(i)
        print(f"  [producer] put {i}")
    await queue.put(-1)  # sentinel


async def consumer(queue: asyncio.Queue[int], name: str) -> list[int]:
    """Consume items from *queue* until sentinel (-1) is received."""
    consumed: list[int] = []
    while True:
        item = await queue.get()
        if item == -1:
            await queue.put(-1)  # re-enqueue for other consumers
            break
        await asyncio.sleep(random.uniform(0.01, 0.02))
        consumed.append(item)
        queue.task_done()
    return consumed


async def demo_producer_consumer() -> None:
    """Show a producer/consumer pattern with asyncio.Queue."""
    print("\n=== Producer / Consumer ===")
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=5)

    prod = asyncio.create_task(producer(queue, 8))
    cons1 = asyncio.create_task(consumer(queue, "consumer-1"))
    cons2 = asyncio.create_task(consumer(queue, "consumer-2"))

    await prod
    r1, r2 = await asyncio.gather(cons1, cons2)
    all_items = sorted(r1 + r2)
    print(f"  Consumed: {all_items}")


# ---------------------------------------------------------------------------
# 3. Semaphore – rate limiting
# ---------------------------------------------------------------------------

async def rate_limited_fetch(
    semaphore: asyncio.Semaphore,
    url: str,
    idx: int,
) -> str:
    """Acquire semaphore before fetching (limits concurrent requests)."""
    async with semaphore:
        await asyncio.sleep(0.05)  # simulate I/O
        return f"done-{idx}"


async def demo_semaphore() -> None:
    """Show concurrency limiting with asyncio.Semaphore."""
    print("\n=== Semaphore (max 3 concurrent) ===")
    sem = asyncio.Semaphore(3)
    urls = [f"https://api.example.com/{i}" for i in range(10)]

    start = time.perf_counter()
    results = await asyncio.gather(
        *[rate_limited_fetch(sem, url, i) for i, url in enumerate(urls)]
    )
    elapsed = time.perf_counter() - start
    print(f"  10 requests, max 3 concurrent: {elapsed:.3f}s")
    print(f"  Results: {results[:5]}…")


# ---------------------------------------------------------------------------
# 4. Async generators and streaming
# ---------------------------------------------------------------------------

async def stream_integers(n: int) -> AsyncGenerator[int, None]:
    """Async generator that yields integers 0..n-1 with simulated delays."""
    for i in range(n):
        await asyncio.sleep(0.005)
        yield i


async def demo_async_generator() -> None:
    """Consume an async generator."""
    print("\n=== Async Generator ===")
    total = 0
    async for value in stream_integers(10):
        total += value
    print(f"  Sum 0..9 = {total}")


# ---------------------------------------------------------------------------
# 5. Timeout handling
# ---------------------------------------------------------------------------

async def slow_operation(delay: float = 2.0) -> str:
    """A simulated slow operation."""
    await asyncio.sleep(delay)
    return "completed"


async def demo_timeouts() -> None:
    """Show timeout handling with asyncio.wait_for."""
    print("\n=== Timeouts ===")
    try:
        result = await asyncio.wait_for(slow_operation(0.05), timeout=1.0)
        print(f"  Fast op completed: {result}")
    except asyncio.TimeoutError:
        print("  Fast op timed out (unexpected)")

    try:
        result = await asyncio.wait_for(slow_operation(2.0), timeout=0.1)
    except asyncio.TimeoutError:
        print("  Slow op timed out as expected")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all asyncio demonstrations."""
    await demo_tasks()
    await demo_producer_consumer()
    await demo_semaphore()
    await demo_async_generator()
    await demo_timeouts()


if __name__ == "__main__":
    asyncio.run(main())

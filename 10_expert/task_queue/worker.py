"""
task_queue/worker.py
=====================
Worker pool for the async task queue.

Each worker runs in an asyncio task and pulls jobs from the shared queue.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from task_queue.queue import TaskQueue, TaskState


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class WorkerStats:
    """Runtime statistics for a single worker."""

    worker_id:    int
    tasks_done:   int = 0
    tasks_failed: int = 0
    total_time:   float = 0.0

    @property
    def avg_time(self) -> float:
        total = self.tasks_done + self.tasks_failed
        return self.total_time / total if total else 0.0


class Worker:
    """
    Async worker that pulls tasks from a queue and executes them.
    """

    def __init__(self, worker_id: int, queue: TaskQueue) -> None:
        self.worker_id = worker_id
        self._queue    = queue
        self._running  = False
        self._task:    asyncio.Task | None = None  # type: ignore[type-arg]
        self.stats     = WorkerStats(worker_id=worker_id)

    async def _run(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.dequeue(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            job.state   = TaskState.RUNNING
            job.attempt += 1
            start = time.perf_counter()

            try:
                job.result       = await job.func(*job.args, **job.kwargs)
                job.state        = TaskState.DONE
                job.completed_at = time.time()
                self.stats.tasks_done += 1
            except Exception as exc:
                if job.attempt <= job.max_retries:
                    job.state = TaskState.RETRYING
                    await asyncio.sleep(job.retry_delay * (2 ** (job.attempt - 1)))
                    await self._queue._queue.put(job)
                else:
                    job.state = TaskState.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.completed_at = time.time()
                    self.stats.tasks_failed += 1
            finally:
                self.stats.total_time += time.perf_counter() - start
                self._queue.task_done()

    def start(self) -> None:
        """Start the worker coroutine as an asyncio task."""
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"worker-{self.worker_id}")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """
    Manages a pool of async workers draining a shared task queue.

    Example::

        async def my_task(x: int) -> int:
            return x * 2

        pool = WorkerPool(workers=4)
        await pool.start()

        tid = await pool.submit(my_task, 21)
        result = await pool.queue.wait_for_result(tid)
        print(result)  # 42

        await pool.stop()
    """

    def __init__(self, workers: int = 4, queue: TaskQueue | None = None) -> None:
        self.queue   = queue or TaskQueue()
        self._workers: list[Worker] = [
            Worker(i, self.queue) for i in range(workers)
        ]

    async def start(self) -> None:
        """Start all workers."""
        for w in self._workers:
            w.start()

    async def stop(self) -> None:
        """Stop all workers."""
        for w in self._workers:
            await w.stop()

    async def submit(self, func: Any, *args: Any, **kwargs: Any) -> str:
        """
        Enqueue a task and return its task_id.
        """
        return await self.queue.enqueue(func, *args, **kwargs)

    @property
    def stats(self) -> list[WorkerStats]:
        return [w.stats for w in self._workers]

    def summary(self) -> str:
        lines = ["Worker Pool Stats:"]
        for s in self.stats:
            lines.append(
                f"  worker-{s.worker_id}: done={s.tasks_done} "
                f"failed={s.tasks_failed} avg={s.avg_time * 1000:.1f}ms"
            )
        lines.append(f"  Queue stats: {self.queue.stats}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo() -> None:
    """Demonstrate the worker pool."""
    import random

    async def slow_add(a: int, b: int) -> int:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return a + b

    async def flaky_task(n: int) -> int:
        await asyncio.sleep(0.01)
        if n % 3 == 0:
            raise ValueError(f"flaky failure for n={n}")
        return n * 10

    pool = WorkerPool(workers=3)
    await pool.start()

    # Submit tasks
    task_ids: list[str] = []
    for i in range(10):
        tid = await pool.submit(slow_add, i, i + 1)
        task_ids.append(tid)

    # Also submit some flaky tasks
    flaky_ids = [await pool.submit(flaky_task, i) for i in range(5)]

    # Collect results
    results: list[Any] = []
    for tid in task_ids:
        try:
            r = await pool.queue.wait_for_result(tid, timeout=5.0)
            results.append(r)
        except Exception as e:
            results.append(f"ERROR: {e}")

    await pool.stop()

    print("=== Worker Pool Demo ===")
    print(f"  Results: {results}")
    print(pool.summary())


def main() -> None:
    asyncio.run(demo())


if __name__ == "__main__":
    main()

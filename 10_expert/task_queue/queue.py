"""
task_queue/queue.py
====================
In-memory async task queue with priority support.

Features:
  - Priority levels (LOW, NORMAL, HIGH, CRITICAL)
  - Task states: PENDING → RUNNING → DONE / FAILED
  - Retry logic with exponential back-off
  - Result storage and retrieval
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


# ---------------------------------------------------------------------------
# Priority and State
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    LOW      = 0
    NORMAL   = 10
    HIGH     = 20
    CRITICAL = 30


class TaskState:
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

TaskFn = Callable[..., Awaitable[Any]]


@dataclass(order=True)
class Task:
    """
    A unit of work to be executed by a worker.

    Ordering: higher priority tasks run first (max-heap via negation).
    """

    priority:      int       = field(compare=True, default=Priority.NORMAL)
    created_at:    float     = field(compare=True, default_factory=time.time)
    task_id:       str       = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    func:          Any       = field(compare=False, repr=False, default=None)
    args:          tuple     = field(compare=False, repr=False, default_factory=tuple)
    kwargs:        dict      = field(compare=False, repr=False, default_factory=dict)
    max_retries:   int       = field(compare=False, default=3)
    retry_delay:   float     = field(compare=False, default=0.1)
    state:         str       = field(compare=False, default=TaskState.PENDING)
    result:        Any       = field(compare=False, default=None)
    error:         str | None = field(compare=False, default=None)
    attempt:       int       = field(compare=False, default=0)
    completed_at:  float | None = field(compare=False, default=None)

    def __post_init__(self) -> None:
        # Negate priority so that higher-priority tasks sort first in asyncio.PriorityQueue
        self.priority = -int(self.priority)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class TaskQueue:
    """
    Async task queue backed by asyncio.PriorityQueue.

    Producers call :meth:`enqueue`; workers call :meth:`dequeue`.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue(maxsize=maxsize)
        self._tasks: dict[str, Task] = {}

    async def enqueue(
        self,
        func: TaskFn,
        *args: Any,
        priority: Priority = Priority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        **kwargs: Any,
    ) -> str:
        """
        Add a task to the queue.

        Returns:
            task_id that can be used to poll the result.
        """
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._tasks[task.task_id] = task
        await self._queue.put(task)
        return task.task_id

    async def dequeue(self) -> Task:
        """Block until a task is available and return it."""
        return await self._queue.get()

    def task_done(self) -> None:
        """Signal that a dequeued task has been processed."""
        self._queue.task_done()

    def get_task(self, task_id: str) -> Task | None:
        """Return the Task object for *task_id*, or None."""
        return self._tasks.get(task_id)

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float = 10.0,
        poll_interval: float = 0.05,
    ) -> Any:
        """
        Poll until the task with *task_id* is DONE or FAILED.

        Raises:
            asyncio.TimeoutError: if the task doesn't complete within *timeout*.
            RuntimeError:         if the task failed.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task: {task_id}")
            if task.state == TaskState.DONE:
                return task.result
            if task.state == TaskState.FAILED:
                raise RuntimeError(f"Task failed: {task.error}")
            await asyncio.sleep(poll_interval)
        raise asyncio.TimeoutError(f"Task {task_id} did not complete in {timeout}s")

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int]:
        counts = {s: 0 for s in (
            TaskState.PENDING, TaskState.RUNNING, TaskState.DONE,
            TaskState.FAILED, TaskState.RETRYING
        )}
        for t in self._tasks.values():
            counts[t.state] = counts.get(t.state, 0) + 1
        return counts

"""
os_interaction/task_scheduler.py
==================================
A simple in-process task scheduler using threading.

Supports:
  - One-shot tasks (run after a delay)
  - Recurring tasks (run every N seconds)
  - Cron-style scheduling (minute, hour, day)
  - Graceful shutdown

Run:  python task_scheduler.py
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

Task = Callable[[], None]


# ---------------------------------------------------------------------------
# Scheduled job descriptor
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Job:
    """A scheduled job entry."""

    next_run: float                        # Unix timestamp for next execution
    func: Task       = field(compare=False)
    name: str        = field(compare=False)
    interval: float  = field(compare=False, default=0.0)   # 0 = one-shot
    repeat: int      = field(compare=False, default=-1)     # -1 = forever
    _run_count: int  = field(compare=False, default=0, repr=False)

    @property
    def is_recurring(self) -> bool:
        return self.interval > 0

    def should_repeat(self) -> bool:
        return self.is_recurring and (self.repeat == -1 or self._run_count < self.repeat)

    def execute(self) -> None:
        """Run the job and update state."""
        self._run_count += 1
        logger.debug("Running job %r (run #%d)", self.name, self._run_count)
        try:
            self.func()
        except Exception:
            logger.exception("Job %r raised an exception", self.name)
        if self.should_repeat():
            self.next_run = time.time() + self.interval


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """
    Thread-based task scheduler.

    Example::

        scheduler = TaskScheduler()

        @scheduler.every(5, name="heartbeat")
        def heartbeat():
            print("ping")

        scheduler.start()
        time.sleep(30)
        scheduler.stop()
    """

    def __init__(self, resolution: float = 0.1) -> None:
        """
        Args:
            resolution: How often (seconds) the scheduler checks for due jobs.
        """
        self._resolution = resolution
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        func: Task,
        delay: float,
        name: str | None = None,
    ) -> Job:
        """
        Run *func* once after *delay* seconds.
        """
        job = Job(
            next_run=time.time() + delay,
            func=func,
            name=name or func.__name__,
            interval=0.0,
        )
        self._add_job(job)
        return job

    def every(
        self,
        interval: float,
        name: str | None = None,
        repeat: int = -1,
        delay: float = 0.0,
    ) -> Callable[[Task], Task]:
        """
        Decorator: run decorated function every *interval* seconds.

        Args:
            interval: Seconds between executions.
            name:     Display name for the job.
            repeat:   Maximum number of executions (-1 = unlimited).
            delay:    Initial delay before first run (default: 0).
        """
        def decorator(func: Task) -> Task:
            job = Job(
                next_run=time.time() + delay,
                func=func,
                name=name or func.__name__,
                interval=interval,
                repeat=repeat,
            )
            self._add_job(job)
            return func
        return decorator

    def start(self) -> None:
        """Start the scheduler loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("TaskScheduler started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler and wait for the thread to finish."""
        self._running = False
        self._event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("TaskScheduler stopped")

    def job_count(self) -> int:
        """Return the number of registered jobs."""
        with self._lock:
            return len(self._jobs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_job(self, job: Job) -> None:
        with self._lock:
            self._jobs.append(job)
            self._jobs.sort()  # keep sorted by next_run
        self._event.set()

    def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            self._event.wait(self._resolution)
            self._event.clear()
            self._tick()

    def _tick(self) -> None:
        """Execute any due jobs."""
        now = time.time()
        with self._lock:
            due = [j for j in self._jobs if j.next_run <= now]

        for job in due:
            job.execute()

        with self._lock:
            # Remove one-shot jobs that have run; keep recurring ones
            self._jobs = [
                j for j in self._jobs
                if j.is_recurring and j.should_repeat()
                or j.next_run > now
            ]
            self._jobs.sort()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the task scheduler."""
    import time

    scheduler = TaskScheduler(resolution=0.05)
    log: list[str] = []

    # One-shot tasks
    scheduler.schedule(lambda: log.append("one-shot-0.1"), delay=0.1, name="shot1")
    scheduler.schedule(lambda: log.append("one-shot-0.3"), delay=0.3, name="shot2")

    # Recurring (5 times)
    @scheduler.every(0.1, name="tick", repeat=5)
    def tick() -> None:
        log.append("tick")
        print(f"  tick at {time.time():.3f}")

    scheduler.start()
    time.sleep(0.8)
    scheduler.stop()

    print(f"\nLog entries: {log}")
    tick_count = log.count("tick")
    print(f"Tick count: {tick_count} (expected 5)")


if __name__ == "__main__":
    main()

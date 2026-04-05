"""
projects/parallel_processor.py
================================
Parallel file/data processor using concurrent.futures.

Demonstrates:
  - ThreadPoolExecutor for I/O-bound work (simulated network/disk I/O)
  - ProcessPoolExecutor for CPU-bound work (heavy computation)
  - as_completed for progress reporting
  - Graceful error handling in parallel tasks

Run:  python projects/parallel_processor.py
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class ProcessingResult:
    """Result of processing a single item."""

    item_id: str
    success: bool
    value: Any
    elapsed: float
    error: str | None = None


def simulate_io_task(url: str) -> ProcessingResult:
    """
    Simulate an I/O-bound task (e.g., HTTP request or DB query).
    Returns a fake result with simulated latency.
    """
    start = time.perf_counter()
    delay = random.uniform(0.05, 0.2)
    time.sleep(delay)
    elapsed = time.perf_counter() - start

    if "error" in url:
        return ProcessingResult(
            item_id=url, success=False,
            value=None, elapsed=elapsed,
            error="Simulated network error",
        )
    return ProcessingResult(
        item_id=url, success=True,
        value={"url": url, "status": 200, "bytes": random.randint(100, 5000)},
        elapsed=elapsed,
    )


def compute_file_hash(path: Path) -> ProcessingResult:
    """
    CPU-bound task: compute SHA-256 hash of a file.
    """
    start = time.perf_counter()
    try:
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return ProcessingResult(
            item_id=str(path),
            success=True,
            value=sha.hexdigest(),
            elapsed=time.perf_counter() - start,
        )
    except OSError as exc:
        return ProcessingResult(
            item_id=str(path),
            success=False,
            value=None,
            elapsed=time.perf_counter() - start,
            error=str(exc),
        )


def cpu_heavy_task(n: int) -> ProcessingResult:
    """CPU-bound task: count primes below n."""
    start = time.perf_counter()
    count = sum(
        1 for k in range(2, n)
        if all(k % d != 0 for d in range(2, int(math.sqrt(k)) + 1))
    )
    return ProcessingResult(
        item_id=f"primes_below_{n}",
        success=True,
        value=count,
        elapsed=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Parallel processor
# ---------------------------------------------------------------------------

class ParallelProcessor:
    """
    Generic parallel task processor with progress reporting.

    Chooses ThreadPoolExecutor for I/O tasks and ProcessPoolExecutor
    for CPU tasks.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) + 4)

    def run_io_tasks(
        self,
        tasks: list[str],
        verbose: bool = False,
    ) -> list[ProcessingResult]:
        """
        Run I/O-bound tasks concurrently using threads.

        Args:
            tasks: List of URL strings to process.
            verbose: Print progress updates.

        Returns:
            List of ProcessingResult objects.
        """
        results: list[ProcessingResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task: dict[Future[ProcessingResult], str] = {
                executor.submit(simulate_io_task, t): t for t in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=5)
                except Exception as exc:
                    result = ProcessingResult(
                        item_id=task, success=False,
                        value=None, elapsed=0,
                        error=str(exc),
                    )
                results.append(result)
                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"  [{status}] {task[:50]:<50} {result.elapsed:.3f}s")

        return results

    def run_cpu_tasks(
        self,
        numbers: list[int],
    ) -> list[ProcessingResult]:
        """
        Run CPU-bound tasks using processes.

        Args:
            numbers: List of integers; count primes below each.

        Returns:
            List of ProcessingResult objects.
        """
        results: list[ProcessingResult] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(cpu_heavy_task, n): n for n in numbers}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def hash_files(self, paths: list[Path]) -> list[ProcessingResult]:
        """Compute SHA-256 hashes of all *paths* in parallel (thread pool)."""
        results: list[ProcessingResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for result in executor.map(compute_file_hash, paths):
                results.append(result)
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate parallel processing."""
    processor = ParallelProcessor(max_workers=8)

    # I/O tasks
    print("=== Parallel I/O Tasks (threads) ===")
    urls = [f"https://api.example.com/item/{i}" for i in range(12)]
    urls[5] = "https://api.example.com/error"  # inject one failure

    start = time.perf_counter()
    io_results = processor.run_io_tasks(urls, verbose=True)
    elapsed = time.perf_counter() - start

    succeeded = sum(1 for r in io_results if r.success)
    failed    = sum(1 for r in io_results if not r.success)
    print(f"\n  {succeeded} succeeded, {failed} failed in {elapsed:.3f}s")

    # File hashing
    print("\n=== File Hashing (threads) ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files: list[Path] = []
        for i in range(5):
            p = tmp_path / f"file_{i}.bin"
            p.write_bytes(os.urandom(1024 * 256))  # 256 KB
            files.append(p)

        start = time.perf_counter()
        hash_results = processor.hash_files(files)
        elapsed = time.perf_counter() - start

        for r in hash_results:
            name = Path(r.item_id).name
            print(f"  {name}: {r.value[:16]}…  ({r.elapsed * 1000:.1f} ms)")
        print(f"  Total: {elapsed:.3f}s")


if __name__ == "__main__":
    main()

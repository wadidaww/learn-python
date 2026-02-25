"""
os_interaction/process_manager.py
===================================
Process management utilities using the subprocess module.

Features:
  - Run commands with timeout and error handling
  - Capture stdout/stderr
  - Pipeline (pipe output of one command to another)
  - Background process management

Run:  python process_manager.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of running a subprocess command."""

    command:     list[str]
    returncode:  int
    stdout:      str
    stderr:      str
    elapsed:     float

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.returncode})"
        return f"[{status}] {' '.join(self.command)} ({self.elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = 30.0,
    env: dict[str, str] | None = None,
    input_data: str | None = None,
) -> RunResult:
    """
    Run *command* as a subprocess with captured I/O.

    Args:
        command:    Argument list, e.g. ["ls", "-la"].
        cwd:        Working directory override.
        timeout:    Kill process after this many seconds (None = no limit).
        env:        Environment variables (merged with os.environ by default).
        input_data: Text to write to the process's stdin.

    Returns:
        RunResult with captured stdout/stderr and exit code.

    Raises:
        subprocess.TimeoutExpired: Propagated after killing the process.
    """
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=env,
            input=input_data,
        )
        elapsed = time.perf_counter() - start
        return RunResult(
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        raise


def run_pipeline(commands: list[list[str]], input_data: str | None = None) -> RunResult:
    """
    Execute a pipeline of commands, piping stdout of each to stdin of the next.

    Returns the result of the last command.
    """
    if not commands:
        raise ValueError("commands must not be empty")

    procs: list[subprocess.Popen[str]] = []
    prev_stdout: Any = subprocess.PIPE

    # Provide initial stdin data for the first process
    if input_data is not None:
        import io as _io
        first_stdin: Any = subprocess.PIPE
    else:
        first_stdin = None

    for i, cmd in enumerate(commands):
        stdin = first_stdin if i == 0 else procs[-1].stdout
        proc: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append(proc)

    # Write stdin to first process if provided
    if input_data is not None and procs:
        assert procs[0].stdin is not None
        procs[0].stdin.write(input_data)
        procs[0].stdin.close()

    # Close intermediate stdout handles to avoid deadlocks
    for i in range(len(procs) - 1):
        if procs[i].stdout:
            procs[i].stdout.close()  # type: ignore[union-attr]

    start = time.perf_counter()
    last = procs[-1]
    stdout, stderr = last.communicate()
    elapsed = time.perf_counter() - start

    for proc in procs[:-1]:
        proc.wait()

    return RunResult(
        command=commands[-1],
        returncode=last.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Higher-level utilities
# ---------------------------------------------------------------------------

def python_run(script: str, *args: str, timeout: float = 10.0) -> RunResult:
    """Execute a Python snippet using the current interpreter."""
    return run([sys.executable, "-c", script, *args], timeout=timeout)


def check_command_available(cmd: str) -> bool:
    """Return True if *cmd* is available on PATH."""
    try:
        result = run([cmd, "--version"], timeout=2.0)
        return result.success or result.returncode != 127
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class BackgroundProcess:
    """Manage a long-running background subprocess."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        """Launch the process in the background."""
        self._proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("Started PID %d: %s", self._proc.pid, " ".join(self.command))

    def stop(self, timeout: float = 3.0) -> int | None:
        """Terminate the process gracefully (SIGTERM), then SIGKILL if needed."""
        if self._proc is None:
            return None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        return self._proc.returncode

    def is_running(self) -> bool:
        """Return True if the process is still running."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate process management utilities."""
    print("=== Run command ===")
    result = run(["python3", "-c", "print('hello from subprocess')"])
    print(f"  {result}")
    print(f"  stdout: {result.stdout.strip()!r}")

    print("\n=== Run with timeout ===")
    try:
        run(["python3", "-c", "import time; time.sleep(10)"], timeout=0.5)
    except subprocess.TimeoutExpired:
        print("  Timed out as expected")

    print("\n=== Python runner ===")
    r = python_run("import sys; print(sys.version.split()[0])")
    print(f"  Python version: {r.stdout.strip()}")

    print("\n=== Background process ===")
    bg = BackgroundProcess(["python3", "-c", "import time; time.sleep(5)"])
    bg.start()
    print(f"  PID: {bg.pid}, running: {bg.is_running()}")
    bg.stop()
    print(f"  After stop, running: {bg.is_running()}")

    print("\n=== Pipeline ===")
    r = run_pipeline(
        [["python3", "-c", "print('line 1\\nline 2\\nline 3')"]],
    )
    print(f"  pipeline stdout: {r.stdout.strip()!r}")


if __name__ == "__main__":
    main()

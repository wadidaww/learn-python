"""
Module 01 – Error Handling
============================
Demonstrates custom exceptions, context managers, retry logic, and
structured error-handling patterns in Python 3.11+.

Run directly:  python 04_error_handling.py
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Generator
from typing import Any, TypeVar

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# 1. Custom exception hierarchy
# ---------------------------------------------------------------------------

class AppError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class ValidationError(AppError):
    """Raised when input data fails validation."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"Validation failed for '{field}': {message}", code=400)
        self.field = field


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, key: Any) -> None:
        super().__init__(f"{resource} with key={key!r} not found", code=404)
        self.resource = resource
        self.key = key


class RateLimitError(AppError):
    """Raised when a rate limit is exceeded."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s", code=429)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# 2. Context managers
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that measures elapsed wall time."""

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("%s took %.4fs", self.label, self.elapsed)
        return False  # do not suppress exceptions


class DatabaseConnection:
    """Simulated database connection context manager."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._connected = False

    def __enter__(self) -> DatabaseConnection:
        logger.debug("Connecting to %s", self.dsn)
        self._connected = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        logger.debug("Disconnecting from %s", self.dsn)
        self._connected = False
        if exc_type is not None:
            logger.error("Connection closed due to error: %s", exc_val)
        return False

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a fake query and return mock results."""
        if not self._connected:
            raise RuntimeError("Not connected")
        logger.debug("Executing: %s", sql)
        return [{"id": 1, "value": "mock"}]


@contextlib.contextmanager
def managed_resource(name: str) -> Generator[dict[str, str], None, None]:
    """Generator-based context manager for a named resource."""
    logger.debug("Acquiring resource: %s", name)
    resource: dict[str, str] = {"name": name, "status": "active"}
    try:
        yield resource
    except Exception as exc:
        resource["status"] = "error"
        logger.error("Resource %s encountered error: %s", name, exc)
        raise
    finally:
        resource["status"] = "released"
        logger.debug("Released resource: %s", name)


# ---------------------------------------------------------------------------
# 3. Retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function on specified exceptions.

    Args:
        max_attempts: Maximum number of attempts (including first call).
        delay: Seconds to wait between retries.
        exceptions: Exception types that trigger a retry.
    """
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt, max_attempts, func.__name__, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(
                f"All {max_attempts} attempts failed for {func.__name__}"
            ) from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# 4. Result type pattern (no exceptions for control flow)
# ---------------------------------------------------------------------------

class Result[T]:
    """
    A simple Result monad that holds either a success value or an error.

    Generic over the success type T.
    """

    def __init__(self, value: T | None, error: Exception | None) -> None:
        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> Result[T]:
        """Construct a successful result."""
        return cls(value, None)

    @classmethod
    def err(cls, error: Exception) -> Result[T]:
        """Construct a failed result."""
        return cls(None, error)

    @property
    def is_ok(self) -> bool:
        return self._error is None

    def unwrap(self) -> T:
        """Return the value, raising the stored error if failed."""
        if self._error is not None:
            raise self._error
        assert self._value is not None
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Return the value or *default* if failed."""
        if self._error is not None:
            return default
        assert self._value is not None
        return self._value

    def __repr__(self) -> str:
        if self.is_ok:
            return f"Result.ok({self._value!r})"
        return f"Result.err({self._error!r})"


def safe_divide(a: float, b: float) -> Result[float]:
    """Return a/b wrapped in Result; never raises."""
    if b == 0:
        return Result.err(ZeroDivisionError("division by zero"))
    return Result.ok(a / b)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate error handling patterns."""
    # Custom exceptions
    print("=== Custom Exceptions ===")
    try:
        raise ValidationError("email", "must contain @")
    except ValidationError as e:
        print(f"  Caught: {e} (field={e.field})")

    try:
        raise NotFoundError("User", 42)
    except NotFoundError as e:
        print(f"  Caught: {e}")

    # Context manager – timer
    print("\n=== Timer ===")
    with Timer("computation") as t:
        total = sum(range(1_000_000))
    print(f"  sum={total}, elapsed={t.elapsed:.4f}s")

    # Context manager – DB
    print("\n=== Database Connection ===")
    with DatabaseConnection("sqlite:///:memory:") as db:
        rows = db.query("SELECT 1")
        print("  Rows:", rows)

    # Generator context manager
    print("\n=== Managed Resource ===")
    with managed_resource("cache") as res:
        print("  Resource:", res)

    # Result pattern
    print("\n=== Result Pattern ===")
    for a, b in [(10, 2), (5, 0)]:
        r = safe_divide(a, b)
        print(f"  {a}/{b} = {r}")
        print(f"  unwrap_or(0): {r.unwrap_or(0)}")


if __name__ == "__main__":
    main()

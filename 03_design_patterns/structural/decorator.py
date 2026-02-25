"""
structural/decorator.py
========================
Decorator pattern — both the class-based (structural) and the
function-based (functional) flavours.

Examples:
  - Logging decorator
  - Caching / memoisation
  - Rate limiter
  - Retry
  - Input validation
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# 1. Function decorators
# ---------------------------------------------------------------------------

def timer(func: F) -> F:
    """Decorator that logs the execution time of *func*."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__name__} took {elapsed * 1000:.2f} ms")
        return result
    return wrapper  # type: ignore[return-value]


def memoize(func: F) -> F:
    """Simple memoisation decorator using a dict cache."""
    cache: dict[tuple[Any, ...], Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache  # type: ignore[attr-defined]
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def retry(
    max_attempts: int = 3,
    delay: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Parameterised retry decorator."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if delay > 0:
                        time.sleep(delay)
            raise RuntimeError(f"Failed after {max_attempts} attempts") from last
        return wrapper  # type: ignore[return-value]
    return decorator


def validate_positive(*param_names: str) -> Callable[[F], F]:
    """Decorator that ensures named parameters are positive numbers."""
    def decorator(func: F) -> F:
        import inspect
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in param_names:
                if name in bound.arguments:
                    val = bound.arguments[name]
                    if not isinstance(val, (int, float)) or val <= 0:
                        raise ValueError(f"Parameter '{name}' must be positive, got {val!r}")
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# 2. Class-based Decorator (structural pattern)
# ---------------------------------------------------------------------------

class Component:
    """Base component interface."""

    def operation(self) -> str:
        return "Component"

    def cost(self) -> float:
        return 0.0


class ConcreteComponent(Component):
    """The real object being decorated."""

    def operation(self) -> str:
        return "ConcreteComponent"

    def cost(self) -> float:
        return 10.0


class ComponentDecorator(Component):
    """Base decorator that wraps a Component."""

    def __init__(self, component: Component) -> None:
        self._component = component

    def operation(self) -> str:
        return self._component.operation()

    def cost(self) -> float:
        return self._component.cost()


class LoggingDecorator(ComponentDecorator):
    """Decorator that logs every operation call."""

    def __init__(self, component: Component, log: list[str] | None = None) -> None:
        super().__init__(component)
        self._log = log if log is not None else []

    def operation(self) -> str:
        result = self._component.operation()
        self._log.append(f"called operation → {result!r}")
        return f"Logged({result})"

    def cost(self) -> float:
        return self._component.cost()


class CachingDecorator(ComponentDecorator):
    """Decorator that caches the result of operation()."""

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        self._cached: str | None = None

    def operation(self) -> str:
        if self._cached is None:
            self._cached = self._component.operation()
        return self._cached


class PricingDecorator(ComponentDecorator):
    """Adds a surcharge to the cost."""

    def __init__(self, component: Component, surcharge: float) -> None:
        super().__init__(component)
        self._surcharge = surcharge

    def cost(self) -> float:
        return self._component.cost() + self._surcharge

    def operation(self) -> str:
        return f"Priced({self._component.operation()}, +{self._surcharge})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate decorator patterns."""
    print("=== Function Decorators ===")

    @timer
    def slow_sum(n: int) -> int:
        return sum(range(n))

    result = slow_sum(1_000_000)
    print(f"  slow_sum result: {result}")

    @memoize
    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    print(f"  fib(35) = {fib(35)}")

    @validate_positive("x", "y")
    def rectangle_area(x: float, y: float) -> float:
        return x * y

    print(f"  area(3, 4) = {rectangle_area(3, 4)}")
    try:
        rectangle_area(-1, 4)
    except ValueError as e:
        print(f"  Caught: {e}")

    print("\n=== Class-based Decorators ===")
    comp: Component = ConcreteComponent()
    log: list[str] = []
    comp = LoggingDecorator(comp, log)
    comp = PricingDecorator(comp, surcharge=5.0)
    print(f"  operation: {comp.operation()}")
    print(f"  cost: ${comp.cost():.2f}")
    print(f"  log: {log}")


if __name__ == "__main__":
    main()

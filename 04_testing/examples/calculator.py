"""
examples/calculator.py
=======================
A calculator class that demonstrates testable code design:
  - Pure functions with no side effects
  - Well-defined error conditions
  - An optional history feature
"""

from __future__ import annotations

import math
from enum import Enum, auto


class Operation(Enum):
    ADD      = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE   = auto()
    POWER    = auto()
    SQRT     = auto()
    MOD      = auto()


class CalculatorError(Exception):
    """Base exception for calculator errors."""


class DivisionByZeroError(CalculatorError):
    """Raised on division by zero."""


class NegativeSqrtError(CalculatorError):
    """Raised when taking the square root of a negative number."""


class Calculator:
    """
    A stateful calculator with operation history.

    All arithmetic methods return float results and record the operation
    in the history log.

    Example::

        calc = Calculator()
        calc.add(3, 4)       # 7.0
        calc.divide(10, 2)   # 5.0
        calc.history         # [(ADD, (3, 4), 7.0), (DIVIDE, (10, 2), 5.0)]
    """

    def __init__(self) -> None:
        self._history: list[tuple[Operation, tuple[float, ...], float]] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, a: float, b: float) -> float:
        """Return a + b."""
        result = a + b
        self._record(Operation.ADD, (a, b), result)
        return result

    def subtract(self, a: float, b: float) -> float:
        """Return a - b."""
        result = a - b
        self._record(Operation.SUBTRACT, (a, b), result)
        return result

    def multiply(self, a: float, b: float) -> float:
        """Return a * b."""
        result = a * b
        self._record(Operation.MULTIPLY, (a, b), result)
        return result

    def divide(self, a: float, b: float) -> float:
        """
        Return a / b.

        Raises:
            DivisionByZeroError: If b is zero.
        """
        if b == 0:
            raise DivisionByZeroError(f"Cannot divide {a} by zero")
        result = a / b
        self._record(Operation.DIVIDE, (a, b), result)
        return result

    def power(self, base: float, exp: float) -> float:
        """Return base ** exp."""
        result = base ** exp
        self._record(Operation.POWER, (base, exp), result)
        return result

    def sqrt(self, x: float) -> float:
        """
        Return the square root of x.

        Raises:
            NegativeSqrtError: If x < 0.
        """
        if x < 0:
            raise NegativeSqrtError(f"Cannot take sqrt of {x}")
        result = math.sqrt(x)
        self._record(Operation.SQRT, (x,), result)
        return result

    def modulo(self, a: float, b: float) -> float:
        """
        Return a % b.

        Raises:
            DivisionByZeroError: If b is zero.
        """
        if b == 0:
            raise DivisionByZeroError("Modulo by zero is undefined")
        result = a % b
        self._record(Operation.MOD, (a, b), result)
        return result

    # ------------------------------------------------------------------
    # Chained / utility methods
    # ------------------------------------------------------------------

    def percent(self, value: float, percent: float) -> float:
        """Return *percent* % of *value*."""
        return self.divide(self.multiply(value, percent), 100)

    def average(self, numbers: list[float]) -> float:
        """Return the arithmetic mean of *numbers*."""
        if not numbers:
            raise ValueError("Cannot take average of empty list")
        return sum(numbers) / len(numbers)

    def factorial(self, n: int) -> int:
        """Return n! for non-negative integer n."""
        if n < 0:
            raise ValueError("Factorial is undefined for negative integers")
        return math.factorial(n)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[tuple[Operation, tuple[float, ...], float]]:
        """Read-only snapshot of operation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the operation history."""
        self._history.clear()

    def last_result(self) -> float | None:
        """Return the result of the most recent operation, or None."""
        if not self._history:
            return None
        return self._history[-1][2]

    def _record(
        self,
        op: Operation,
        operands: tuple[float, ...],
        result: float,
    ) -> None:
        self._history.append((op, operands, result))

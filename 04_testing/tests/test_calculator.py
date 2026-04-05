"""
tests/test_calculator.py
=========================
Comprehensive tests for the Calculator class:
  - Unit tests
  - Parametrised tests
  - Exception tests
  - Mock-based tests
  - Fixture usage
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from examples.calculator import (
    Calculator,
    CalculatorError,
    DivisionByZeroError,
    NegativeSqrtError,
    Operation,
)


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

class TestAdd:
    def test_positive(self, calc: Calculator) -> None:
        assert calc.add(2, 3) == 5

    def test_negative(self, calc: Calculator) -> None:
        assert calc.add(-2, -3) == -5

    def test_mixed(self, calc: Calculator) -> None:
        assert calc.add(-2, 3) == 1

    def test_float(self, calc: Calculator) -> None:
        assert calc.add(0.1, 0.2) == pytest.approx(0.3)

    def test_identity(self, calc: Calculator) -> None:
        assert calc.add(42, 0) == 42


class TestSubtract:
    @pytest.mark.parametrize("a, b, expected", [
        (10, 3, 7),
        (0, 0, 0),
        (-5, -3, -2),
        (1.5, 0.5, 1.0),
    ])
    def test_subtract(self, calc: Calculator, a: float, b: float, expected: float) -> None:
        assert calc.subtract(a, b) == pytest.approx(expected)


class TestMultiply:
    @pytest.mark.parametrize("a, b, expected", [
        (3, 4, 12),
        (0, 100, 0),
        (-3, 4, -12),
        (2.5, 4, 10.0),
    ])
    def test_multiply(self, calc: Calculator, a: float, b: float, expected: float) -> None:
        assert calc.multiply(a, b) == pytest.approx(expected)


class TestDivide:
    def test_basic(self, calc: Calculator) -> None:
        assert calc.divide(10, 2) == 5.0

    def test_float_result(self, calc: Calculator) -> None:
        assert calc.divide(1, 3) == pytest.approx(1 / 3)

    def test_division_by_zero(self, calc: Calculator) -> None:
        with pytest.raises(DivisionByZeroError, match="divide"):
            calc.divide(10, 0)

    def test_division_by_zero_is_calculator_error(self, calc: Calculator) -> None:
        with pytest.raises(CalculatorError):
            calc.divide(5, 0)

    @pytest.mark.parametrize("a, b", [(0, 5), (0, -1)])
    def test_zero_numerator(self, calc: Calculator, a: float, b: float) -> None:
        assert calc.divide(a, b) == 0


class TestPower:
    @pytest.mark.parametrize("base, exp, expected", [
        (2, 10, 1024),
        (2, 0, 1),
        (5, -1, 0.2),
        (9, 0.5, 3.0),
    ])
    def test_power(self, calc: Calculator, base: float, exp: float, expected: float) -> None:
        assert calc.power(base, exp) == pytest.approx(expected)


class TestSqrt:
    def test_perfect_square(self, calc: Calculator) -> None:
        assert calc.sqrt(16) == 4.0

    def test_irrational(self, calc: Calculator) -> None:
        assert calc.sqrt(2) == pytest.approx(math.sqrt(2))

    def test_zero(self, calc: Calculator) -> None:
        assert calc.sqrt(0) == 0.0

    def test_negative_raises(self, calc: Calculator) -> None:
        with pytest.raises(NegativeSqrtError):
            calc.sqrt(-1)


class TestModulo:
    def test_basic(self, calc: Calculator) -> None:
        assert calc.modulo(10, 3) == 1

    def test_zero_divisor(self, calc: Calculator) -> None:
        with pytest.raises(DivisionByZeroError):
            calc.modulo(5, 0)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_percent(self, calc: Calculator) -> None:
        assert calc.percent(200, 25) == 50.0

    def test_average_basic(self, calc: Calculator) -> None:
        assert calc.average([1, 2, 3, 4, 5]) == 3.0

    def test_average_empty(self, calc: Calculator) -> None:
        with pytest.raises(ValueError):
            calc.average([])

    def test_factorial(self, calc: Calculator) -> None:
        assert calc.factorial(5) == 120
        assert calc.factorial(0) == 1

    def test_factorial_negative(self, calc: Calculator) -> None:
        with pytest.raises(ValueError):
            calc.factorial(-1)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_records_operations(self, calc_with_history: Calculator) -> None:
        assert len(calc_with_history.history) == 3

    def test_history_immutable(self, calc: Calculator) -> None:
        calc.add(1, 2)
        h = calc.history
        h.clear()
        assert len(calc.history) == 1

    def test_clear_history(self, calc_with_history: Calculator) -> None:
        calc_with_history.clear_history()
        assert len(calc_with_history.history) == 0

    def test_last_result(self, calc: Calculator) -> None:
        assert calc.last_result() is None
        calc.add(3, 4)
        assert calc.last_result() == 7.0

    def test_history_contains_correct_operations(self, calc: Calculator) -> None:
        calc.add(1, 2)
        calc.multiply(3, 4)
        ops = [h[0] for h in calc.history]
        assert ops == [Operation.ADD, Operation.MULTIPLY]


# ---------------------------------------------------------------------------
# Mock-based tests
# ---------------------------------------------------------------------------

class TestWithMocks:
    def test_mock_divide_called(self) -> None:
        """Verify that a collaborator's divide method is called correctly."""
        mock_calc = MagicMock(spec=Calculator)
        mock_calc.divide.return_value = 5.0

        result = mock_calc.divide(10, 2)

        mock_calc.divide.assert_called_once_with(10, 2)
        assert result == 5.0

    def test_patch_math_sqrt(self, calc: Calculator) -> None:
        """Patch math.sqrt to return a fixed value."""
        with patch("math.sqrt", return_value=99.0):
            result = calc.sqrt(16)
        assert result == 99.0

    def test_multiple_calls_tracked(self) -> None:
        mock_calc = MagicMock(spec=Calculator)
        mock_calc.add.side_effect = [1, 2, 3]

        assert mock_calc.add(0, 0) == 1
        assert mock_calc.add(0, 0) == 2
        assert mock_calc.add(0, 0) == 3
        assert mock_calc.add.call_count == 3

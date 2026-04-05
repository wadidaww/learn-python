"""
tests/test_property_based.py
=============================
Property-based tests implemented without any external library.

We manually generate random inputs and assert mathematical properties
that must hold for all inputs.
"""

from __future__ import annotations

import math
import random

import pytest

from examples.calculator import Calculator, DivisionByZeroError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_floats(
    n: int = 50,
    low: float = -1_000.0,
    high: float = 1_000.0,
    seed: int = 42,
) -> list[float]:
    """Generate *n* random floats in [low, high] deterministically."""
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(n)]


def random_nonzero_floats(n: int = 50, seed: int = 42) -> list[float]:
    """Generate *n* random floats, none equal to zero."""
    return [f for f in random_floats(n, seed=seed) if abs(f) > 1e-9]


# ---------------------------------------------------------------------------
# Arithmetic properties
# ---------------------------------------------------------------------------

class TestCommutativity:
    """a + b == b + a  and  a * b == b * a."""

    def test_add_commutative(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=1):
            b = a * 1.1 + 3.0
            assert calc.add(a, b) == pytest.approx(calc.add(b, a))

    def test_multiply_commutative(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=2):
            b = a - 5.0
            assert calc.multiply(a, b) == pytest.approx(calc.multiply(b, a))


class TestAssociativity:
    """(a + b) + c == a + (b + c)."""

    def test_add_associative(self) -> None:
        calc = Calculator()
        triples = [(a, a + 1, a - 1) for a in random_floats(20, seed=3)]
        for a, b, c in triples:
            lhs = calc.add(calc.add(a, b), c)
            rhs = calc.add(a, calc.add(b, c))
            assert lhs == pytest.approx(rhs, rel=1e-9)


class TestIdentityElements:
    """a + 0 == a  and  a * 1 == a."""

    def test_add_identity(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=4):
            assert calc.add(a, 0) == pytest.approx(a)

    def test_multiply_identity(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=5):
            assert calc.multiply(a, 1) == pytest.approx(a)

    def test_multiply_zero_absorbing(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=6):
            assert calc.multiply(a, 0) == pytest.approx(0.0)


class TestInverses:
    """a - a == 0  and  a / a == 1 (for a != 0)."""

    def test_subtract_self(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=7):
            assert calc.subtract(a, a) == pytest.approx(0.0)

    def test_divide_self(self) -> None:
        calc = Calculator()
        for a in random_nonzero_floats(30, seed=8):
            assert calc.divide(a, a) == pytest.approx(1.0)


class TestDivisionProperties:
    """a / b * b == a."""

    def test_divide_multiply_inverse(self) -> None:
        calc = Calculator()
        for a in random_floats(30, seed=9):
            for b in random_nonzero_floats(5, seed=10):
                result = calc.multiply(calc.divide(a, b), b)
                assert result == pytest.approx(a, rel=1e-9)

    def test_divide_by_zero_always_raises(self) -> None:
        calc = Calculator()
        for a in random_floats(20, seed=11):
            with pytest.raises(DivisionByZeroError):
                calc.divide(a, 0)


class TestSqrtProperties:
    """sqrt(x) ** 2 == x  and  sqrt(x * x) == |x|."""

    def test_sqrt_squared(self) -> None:
        calc = Calculator()
        for x in random_floats(30, low=0, high=10_000, seed=12):
            s = calc.sqrt(x)
            assert s ** 2 == pytest.approx(x, rel=1e-9)

    def test_sqrt_of_square(self) -> None:
        calc = Calculator()
        for x in random_floats(30, seed=13):
            assert calc.sqrt(x * x) == pytest.approx(abs(x), rel=1e-9)


class TestPowerProperties:
    """x ** (a + b) == x ** a * x ** b."""

    def test_power_addition_rule(self) -> None:
        calc = Calculator()
        bases = random_floats(20, low=0.1, high=10.0, seed=14)
        for x in bases:
            for a in range(1, 4):
                for b in range(1, 4):
                    lhs = calc.power(x, a + b)
                    rhs = calc.multiply(calc.power(x, a), calc.power(x, b))
                    assert lhs == pytest.approx(rhs, rel=1e-9)


class TestFactorialProperties:
    """n! == n * (n-1)!"""

    def test_factorial_recurrence(self) -> None:
        calc = Calculator()
        for n in range(1, 12):
            assert calc.factorial(n) == n * calc.factorial(n - 1)

    def test_factorial_non_negative(self) -> None:
        calc = Calculator()
        for n in range(10):
            assert calc.factorial(n) >= 1

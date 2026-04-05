"""
conftest.py
============
Shared pytest fixtures for Module 04.
"""

from __future__ import annotations

import pytest

from examples.calculator import Calculator


@pytest.fixture
def calc() -> Calculator:
    """Return a fresh Calculator instance."""
    return Calculator()


@pytest.fixture
def calc_with_history(calc: Calculator) -> Calculator:
    """Return a Calculator pre-loaded with some operations."""
    calc.add(10, 5)
    calc.subtract(20, 8)
    calc.multiply(3, 7)
    return calc

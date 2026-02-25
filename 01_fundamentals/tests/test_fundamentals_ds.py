"""
tests/test_data_structures.py
==============================
pytest tests for 01_fundamentals/02_data_structures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import importlib.util

import pytest

# Modules starting with digits can't be imported directly – use importlib
_spec = importlib.util.spec_from_file_location(
    "data_structures",
    Path(__file__).parent.parent / "02_data_structures.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

build_inventory = _mod.build_inventory
low_stock = _mod.low_stock
word_frequency = _mod.word_frequency


class TestWordFrequency:
    def test_basic_counts(self) -> None:
        freq = word_frequency("a b a c a b")
        assert freq["a"] == 3
        assert freq["b"] == 2
        assert freq["c"] == 1

    def test_case_insensitive(self) -> None:
        freq = word_frequency("Hello hello HELLO")
        assert freq["hello"] == 3

    def test_strips_punctuation(self) -> None:
        freq = word_frequency("hello, world! hello.")
        assert freq.get("hello") == 2
        assert freq.get("world") == 1

    def test_empty_string(self) -> None:
        assert word_frequency("") == {}

    def test_sorted_by_frequency_descending(self) -> None:
        freq = word_frequency("a a a b b c")
        values = list(freq.values())
        assert values == sorted(values, reverse=True)


class TestBuildInventory:
    def test_basic(self) -> None:
        inv = build_inventory([("apple", 10, 1.0)])
        assert "apple" in inv
        assert inv["apple"]["qty"] == 10
        assert inv["apple"]["price"] == 1.0
        assert inv["apple"]["total"] == 10.0

    def test_total_rounding(self) -> None:
        inv = build_inventory([("item", 3, 0.333)])
        assert inv["item"]["total"] == round(3 * 0.333, 2)

    def test_multiple_items(self) -> None:
        items = [("a", 5, 2.0), ("b", 3, 3.0), ("c", 1, 10.0)]
        inv = build_inventory(items)
        assert len(inv) == 3

    def test_empty(self) -> None:
        assert build_inventory([]) == {}


class TestLowStock:
    def test_below_threshold(self) -> None:
        inv = build_inventory([("apple", 2, 1.0), ("mango", 10, 2.0)])
        assert low_stock(inv) == ["apple"]

    def test_custom_threshold(self) -> None:
        inv = build_inventory([("a", 3, 1.0), ("b", 7, 1.0)])
        assert low_stock(inv, threshold=10) == ["a", "b"]

    def test_none_below(self) -> None:
        inv = build_inventory([("a", 50, 1.0)])
        assert low_stock(inv) == []

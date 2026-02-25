"""
tests/test_pipeline.py
=======================
Tests for data engineering pipeline components.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add 08_data_engineering to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.extractor import (
    CSVStringExtractor,
    JSONStringExtractor,
    MemoryExtractor,
)
from pipeline.transformer import (
    AddFieldTransformer,
    DeduplicateTransformer,
    DropNullTransformer,
    FilterTransformer,
    NormalizeStringTransformer,
    RenameTransformer,
    SelectTransformer,
    TransformPipeline,
    TypeCoercionTransformer,
    ValidationTransformer,
)
from pipeline.loader import MemoryLoader, SQLiteLoader
from pipeline.orchestrator import ETLPipeline


CSV_DATA = """id,name,price,qty
1,Widget A,9.99,10
2,Widget B,19.99,5
3,Widget C,,3
4,Widget A,9.99,10
"""

JSON_DATA = json.dumps([
    {"id": "1", "value": "100"},
    {"id": "2", "value": "200"},
])


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class TestCSVExtractor:
    def test_basic(self) -> None:
        ext = CSVStringExtractor(CSV_DATA)
        records = ext.extract()
        assert len(records) == 4
        assert records[0]["name"] == "Widget A"

    def test_headers_present(self) -> None:
        records = CSVStringExtractor(CSV_DATA).extract()
        assert set(records[0].keys()) == {"id", "name", "price", "qty"}


class TestJSONExtractor:
    def test_basic(self) -> None:
        ext = JSONStringExtractor(JSON_DATA)
        records = ext.extract()
        assert len(records) == 2

    def test_root_key(self) -> None:
        data = json.dumps({"items": [{"x": 1}]})
        ext = JSONStringExtractor(data, root_key="items")
        assert ext.extract() == [{"x": 1}]


class TestMemoryExtractor:
    def test_extract(self) -> None:
        data = [{"a": 1}, {"a": 2}]
        ext = MemoryExtractor(data)
        result = ext.extract()
        assert result == data
        assert result is not data  # should be a copy


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    def test_cast_int(self) -> None:
        t = TypeCoercionTransformer({"qty": int})
        result = t.transform([{"qty": "5", "name": "x"}])
        assert result[0]["qty"] == 5
        assert isinstance(result[0]["qty"], int)

    def test_cast_float(self) -> None:
        t = TypeCoercionTransformer({"price": float})
        result = t.transform([{"price": "9.99"}])
        assert result[0]["price"] == pytest.approx(9.99)

    def test_cast_error_raises(self) -> None:
        t = TypeCoercionTransformer({"qty": int})
        with pytest.raises(ValueError):
            t.transform([{"qty": "not-a-number"}])

    def test_skip_errors(self) -> None:
        t = TypeCoercionTransformer({"qty": int}, skip_errors=True)
        result = t.transform([{"qty": "bad"}])
        assert result[0]["qty"] == "bad"  # unchanged


class TestFilterTransformer:
    def test_filter(self) -> None:
        t = FilterTransformer(lambda r: int(r["qty"]) > 5)
        records = [{"qty": "10"}, {"qty": "3"}]
        result = t.transform(records)
        assert len(result) == 1
        assert result[0]["qty"] == "10"


class TestDropNull:
    def test_drops_empty(self) -> None:
        t = DropNullTransformer(["price"])
        records = [{"price": "9.99"}, {"price": ""}, {"price": None}]
        result = t.transform(records)
        assert len(result) == 1

    def test_drops_null_string(self) -> None:
        t = DropNullTransformer(["price"])
        records = [{"price": "NULL"}]
        assert t.transform(records) == []


class TestRename:
    def test_rename(self) -> None:
        t = RenameTransformer({"old_name": "new_name"})
        result = t.transform([{"old_name": "val", "other": 1}])
        assert "new_name" in result[0]
        assert "old_name" not in result[0]


class TestSelect:
    def test_select(self) -> None:
        t = SelectTransformer(["a", "b"])
        result = t.transform([{"a": 1, "b": 2, "c": 3}])
        assert set(result[0].keys()) == {"a", "b"}


class TestAddField:
    def test_computed_field(self) -> None:
        t = AddFieldTransformer("total", lambda r: r["price"] * r["qty"])
        records = [{"price": 10.0, "qty": 3}]
        result = t.transform(records)
        assert result[0]["total"] == 30.0


class TestDeduplicate:
    def test_removes_duplicates(self) -> None:
        t = DeduplicateTransformer(["id"])
        records = [{"id": 1, "v": "a"}, {"id": 1, "v": "b"}, {"id": 2, "v": "c"}]
        result = t.transform(records)
        assert len(result) == 2
        assert result[0]["v"] == "a"  # first occurrence kept


class TestValidation:
    def test_drops_invalid(self) -> None:
        t = ValidationTransformer({"price": lambda v: float(v) > 0})
        records = [{"price": "10"}, {"price": "-1"}]
        result = t.transform(records)
        assert len(result) == 1

    def test_raises_on_invalid(self) -> None:
        t = ValidationTransformer(
            {"price": lambda v: float(v) > 0},
            drop_invalid=False,
        )
        with pytest.raises(ValueError):
            t.transform([{"price": "-1"}])


class TestTransformPipeline:
    def test_chained(self) -> None:
        pipeline = TransformPipeline([
            TypeCoercionTransformer({"qty": int, "price": float}),
            FilterTransformer(lambda r: r["qty"] > 0),
            AddFieldTransformer("total", lambda r: r["price"] * r["qty"]),
        ])
        records = [
            {"qty": "5", "price": "9.99"},
            {"qty": "0", "price": "19.99"},
        ]
        result = pipeline.transform(records)
        assert len(result) == 1
        assert result[0]["total"] == pytest.approx(49.95)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestMemoryLoader:
    def test_load(self) -> None:
        loader = MemoryLoader()
        n = loader.load([{"a": 1}, {"a": 2}])
        assert n == 2
        assert len(loader.records) == 2


class TestSQLiteLoader:
    def test_load_and_query(self) -> None:
        loader = SQLiteLoader(":memory:", "test_table")
        records = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
        n = loader.load(records)
        assert n == 2
        rows = loader.query("SELECT * FROM test_table")
        assert len(rows) == 2
        names = {r["name"] for r in rows}
        assert names == {"Alice", "Bob"}
        loader.close()


# ---------------------------------------------------------------------------
# ETLPipeline
# ---------------------------------------------------------------------------

class TestETLPipeline:
    def test_full_pipeline(self) -> None:
        memory = MemoryLoader()
        pipeline = (
            ETLPipeline("test")
            .extract(MemoryExtractor([
                {"qty": "5", "price": "9.99"},
                {"qty": "0", "price": "19.99"},
                {"qty": "3", "price": ""},
            ]))
            .transform(DropNullTransformer(["price"]))
            .transform(TypeCoercionTransformer({"qty": int, "price": float}))
            .transform(FilterTransformer(lambda r: r["qty"] > 0))
            .load(memory)
        )
        result = pipeline.run()
        assert result.success
        assert len(memory.records) == 1
        assert memory.records[0]["qty"] == 5

    def test_pipeline_reports_steps(self) -> None:
        memory = MemoryLoader()
        pipeline = (
            ETLPipeline("test")
            .extract(MemoryExtractor([{"x": 1}]))
            .load(memory)
        )
        result = pipeline.run()
        assert len(result.steps) >= 2  # at least 1 extract + 1 load

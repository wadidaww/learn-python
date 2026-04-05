"""
pipeline/transformer.py
========================
Data transformation and validation layer.

Transformers are composable: chain them with Pipeline.
Each transformer takes a list of records and returns a list of records.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Base transformer
# ---------------------------------------------------------------------------

class Transformer(ABC):
    """Abstract base class for all transformers."""

    @abstractmethod
    def transform(self, records: list[Record]) -> list[Record]:
        """Transform *records* and return the result."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

class TypeCoercionTransformer(Transformer):
    """
    Cast column values to specified Python types.

    Example::

        t = TypeCoercionTransformer({"qty": int, "price": float, "date": str})
        records = t.transform(raw_records)
    """

    def __init__(
        self,
        schema: dict[str, Callable[[Any], Any]],
        skip_errors: bool = False,
    ) -> None:
        self.schema = schema
        self.skip_errors = skip_errors

    def transform(self, records: list[Record]) -> list[Record]:
        result: list[Record] = []
        for rec in records:
            out = dict(rec)
            for col, cast in self.schema.items():
                if col in out:
                    try:
                        out[col] = cast(out[col])
                    except (ValueError, TypeError) as exc:
                        if not self.skip_errors:
                            raise ValueError(
                                f"Cannot cast {col}={out[col]!r} using {cast.__name__}: {exc}"
                            ) from exc
                        # Leave original value on error
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class FilterTransformer(Transformer):
    """
    Keep only records matching *predicate*.

    Example::

        t = FilterTransformer(lambda r: r["price"] > 0)
    """

    def __init__(self, predicate: Callable[[Record], bool]) -> None:
        self.predicate = predicate

    def transform(self, records: list[Record]) -> list[Record]:
        return [r for r in records if self.predicate(r)]


class DropNullTransformer(Transformer):
    """Remove records that have None / empty-string in any of *required_fields*."""

    def __init__(self, required_fields: list[str]) -> None:
        self.required_fields = required_fields

    def transform(self, records: list[Record]) -> list[Record]:
        def is_valid(rec: Record) -> bool:
            return all(
                rec.get(f) not in (None, "", "NULL", "N/A")
                for f in self.required_fields
            )
        return [r for r in records if is_valid(r)]


# ---------------------------------------------------------------------------
# Field manipulation
# ---------------------------------------------------------------------------

class RenameTransformer(Transformer):
    """Rename columns according to *mapping* (old_name → new_name)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def transform(self, records: list[Record]) -> list[Record]:
        result: list[Record] = []
        for rec in records:
            out = {self.mapping.get(k, k): v for k, v in rec.items()}
            result.append(out)
        return result


class SelectTransformer(Transformer):
    """Keep only the specified *columns*."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def transform(self, records: list[Record]) -> list[Record]:
        return [{k: r[k] for k in self.columns if k in r} for r in records]


class AddFieldTransformer(Transformer):
    """Add a computed field using *name* = *func(record)*."""

    def __init__(self, name: str, func: Callable[[Record], Any]) -> None:
        self.name = name
        self.func = func

    def transform(self, records: list[Record]) -> list[Record]:
        result: list[Record] = []
        for rec in records:
            out = dict(rec)
            out[self.name] = self.func(rec)
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

class NormalizeStringTransformer(Transformer):
    """Strip whitespace and optionally lowercase string fields."""

    def __init__(self, columns: list[str], lowercase: bool = False) -> None:
        self.columns = columns
        self.lowercase = lowercase

    def transform(self, records: list[Record]) -> list[Record]:
        result: list[Record] = []
        for rec in records:
            out = dict(rec)
            for col in self.columns:
                if col in out and isinstance(out[col], str):
                    out[col] = out[col].strip()
                    if self.lowercase:
                        out[col] = out[col].lower()
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class DeduplicateTransformer(Transformer):
    """
    Remove duplicate records based on *key_fields*.

    Keeps the first occurrence.
    """

    def __init__(self, key_fields: list[str]) -> None:
        self.key_fields = key_fields

    def transform(self, records: list[Record]) -> list[Record]:
        seen: set[tuple[Any, ...]] = set()
        result: list[Record] = []
        for rec in records:
            key = tuple(rec.get(f) for f in self.key_fields)
            if key not in seen:
                seen.add(key)
                result.append(rec)
        return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationTransformer(Transformer):
    """
    Validate records against a schema of {field: validator_func}.

    Validators return True/False. Invalid records are dropped (or raise).
    """

    def __init__(
        self,
        validators: dict[str, Callable[[Any], bool]],
        drop_invalid: bool = True,
    ) -> None:
        self.validators = validators
        self.drop_invalid = drop_invalid

    def transform(self, records: list[Record]) -> list[Record]:
        result: list[Record] = []
        for rec in records:
            valid = True
            for field, validator in self.validators.items():
                value = rec.get(field)
                if not validator(value):
                    if not self.drop_invalid:
                        raise ValueError(
                            f"Validation failed: {field}={value!r}"
                        )
                    valid = False
                    break
            if valid:
                result.append(rec)
        return result


# ---------------------------------------------------------------------------
# Pipeline combinator
# ---------------------------------------------------------------------------

class TransformPipeline(Transformer):
    """
    Compose multiple transformers into a single transformer.

    Example::

        pipeline = TransformPipeline([
            TypeCoercionTransformer({"price": float}),
            FilterTransformer(lambda r: r["price"] > 0),
            AddFieldTransformer("total", lambda r: r["price"] * r["qty"]),
        ])
    """

    def __init__(self, transformers: list[Transformer]) -> None:
        self.transformers = transformers

    def transform(self, records: list[Record]) -> list[Record]:
        result = records
        for t in self.transformers:
            result = t.transform(result)
        return result

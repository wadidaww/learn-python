"""
pipeline/loader.py
==================
Data loading to various sinks:
  - CSV file
  - JSON file
  - SQLite database (stdlib sqlite3)
  - In-memory list (for testing)
"""

from __future__ import annotations

import csv
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Base loader
# ---------------------------------------------------------------------------

class Loader(ABC):
    """Abstract base class for all loaders."""

    @abstractmethod
    def load(self, records: list[Record]) -> int:
        """
        Persist *records* to the sink.

        Returns:
            Number of records written.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

class CSVLoader(Loader):
    """
    Write records to a CSV file.

    Example::

        loader = CSVLoader("output.csv")
        n = loader.load(records)
    """

    def __init__(
        self,
        path: Path | str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        mode: str = "w",
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.mode = mode

    def load(self, records: list[Record]) -> int:
        if not records:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys())
        with self.path.open(self.mode, newline="", encoding=self.encoding) as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=self.delimiter)
            writer.writeheader()
            writer.writerows(records)
        return len(records)


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

class JSONLoader(Loader):
    """
    Write records to a JSON file as an array.
    """

    def __init__(
        self,
        path: Path | str,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.indent = indent
        self.encoding = encoding

    def load(self, records: list[Record]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure JSON-serialisable (convert non-JSON types)
        def _default(obj: Any) -> str:
            return str(obj)

        with self.path.open("w", encoding=self.encoding) as fh:
            json.dump(records, fh, indent=self.indent, default=_default)
        return len(records)


# ---------------------------------------------------------------------------
# SQLite loader
# ---------------------------------------------------------------------------

class SQLiteLoader(Loader):
    """
    Write records to a SQLite table using sqlite3.

    The table is created (or its columns extended) automatically based on
    the first record's keys.

    Example::

        loader = SQLiteLoader(":memory:", table="sales")
        loader.load(records)
        with sqlite3.connect(":memory:") as conn:
            ...
    """

    def __init__(
        self,
        database: str,
        table: str,
        if_exists: str = "append",  # "append" | "replace"
    ) -> None:
        self.database = database
        self.table = table
        self.if_exists = if_exists
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.database)
        return self._conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_table(self, conn: sqlite3.Connection, columns: list[str]) -> None:
        """Create table if it doesn't exist."""
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.table}" ({col_defs})'
        )

    def load(self, records: list[Record]) -> int:
        if not records:
            return 0

        conn = self._get_conn()
        columns = list(records[0].keys())

        if self.if_exists == "replace":
            conn.execute(f'DROP TABLE IF EXISTS "{self.table}"')

        self._ensure_table(conn, columns)

        placeholders = ", ".join("?" * len(columns))
        col_names    = ", ".join(f'"{c}"' for c in columns)
        sql = f'INSERT INTO "{self.table}" ({col_names}) VALUES ({placeholders})'

        rows = [tuple(str(r.get(c, "")) for c in columns) for r in records]
        conn.executemany(sql, rows)
        conn.commit()
        return len(records)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute *sql* and return results as a list of dicts."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# In-memory loader (for testing)
# ---------------------------------------------------------------------------

class MemoryLoader(Loader):
    """Append records to an in-memory list."""

    def __init__(self) -> None:
        self.records: list[Record] = []

    def load(self, records: list[Record]) -> int:
        self.records.extend(records)
        return len(records)

    def clear(self) -> None:
        self.records.clear()

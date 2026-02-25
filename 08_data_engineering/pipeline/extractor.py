"""
pipeline/extractor.py
======================
Data extraction from various sources:
  - CSV files
  - JSON files / strings
  - Simulated API (URL placeholder with urllib)
  - In-memory iterables

All extractors implement the same protocol:
    def extract() -> list[dict[str, Any]]
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------

class Extractor(ABC):
    """Abstract base class for all data extractors."""

    @abstractmethod
    def extract(self) -> list[dict[str, Any]]:
        """Extract and return records as a list of dicts."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# CSV extractor
# ---------------------------------------------------------------------------

class CSVExtractor(Extractor):
    """
    Extract records from a CSV file.

    Example::

        ext = CSVExtractor("/data/sales.csv", delimiter=",")
        records = ext.extract()
    """

    def __init__(
        self,
        path: Path | str,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding

    def extract(self) -> list[dict[str, Any]]:
        """Read all rows as dicts keyed by header names."""
        with self.path.open(newline="", encoding=self.encoding) as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            return [dict(row) for row in reader]


class CSVStringExtractor(Extractor):
    """Extract records from a CSV-formatted string."""

    def __init__(self, csv_text: str, delimiter: str = ",") -> None:
        self.csv_text = csv_text
        self.delimiter = delimiter

    def extract(self) -> list[dict[str, Any]]:
        reader = csv.DictReader(io.StringIO(self.csv_text), delimiter=self.delimiter)
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# JSON extractor
# ---------------------------------------------------------------------------

class JSONExtractor(Extractor):
    """
    Extract records from a JSON file.

    The JSON file must contain either a list of objects, or an object
    with a list stored under *root_key*.
    """

    def __init__(self, path: Path | str, root_key: str | None = None) -> None:
        self.path = Path(path)
        self.root_key = root_key

    def extract(self) -> list[dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.root_key:
            data = data[self.root_key]
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array, got {type(data).__name__}"
            )
        return [dict(row) for row in data]


class JSONStringExtractor(Extractor):
    """Extract records from a JSON string."""

    def __init__(self, json_text: str, root_key: str | None = None) -> None:
        self.json_text = json_text
        self.root_key = root_key

    def extract(self) -> list[dict[str, Any]]:
        data = json.loads(self.json_text)
        if self.root_key:
            data = data[self.root_key]
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")
        return list(data)


# ---------------------------------------------------------------------------
# URL / API extractor
# ---------------------------------------------------------------------------

class URLExtractor(Extractor):
    """
    Fetch JSON data from a URL.

    Falls back to an empty list on network errors so pipelines can still run
    in offline / test environments.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        root_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.root_key = root_key
        self.headers = headers or {}

    def extract(self) -> list[dict[str, Any]]:
        try:
            req = urllib.request.Request(self.url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw: str = resp.read().decode("utf-8")
            data: Any = json.loads(raw)
            if self.root_key:
                data = data[self.root_key]
            return list(data)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            import logging
            logging.getLogger(__name__).warning("URLExtractor failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# In-memory extractor
# ---------------------------------------------------------------------------

class MemoryExtractor(Extractor):
    """Extract records from an in-memory list (useful for testing)."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def extract(self) -> list[dict[str, Any]]:
        return list(self._records)

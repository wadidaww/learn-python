"""
Module 01 – File I/O
======================
Demonstrates reading and writing files using pathlib, CSV, JSON, and
basic binary I/O in Python 3.11+.

Run directly:  python 05_file_io.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 1. Text files with pathlib
# ---------------------------------------------------------------------------

def write_text_file(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text_file(path: Path) -> str:
    """Read and return the text content of *path*."""
    return path.read_text(encoding="utf-8")


def count_lines(path: Path) -> int:
    """Count lines in a text file without loading it entirely."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for _ in fh:
            count += 1
    return count


def grep_file(path: Path, pattern: str) -> list[tuple[int, str]]:
    """Return (line_number, line) pairs where *pattern* appears in *path*."""
    results: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if pattern in line:
                results.append((lineno, line.rstrip()))
    return results


# ---------------------------------------------------------------------------
# 2. CSV
# ---------------------------------------------------------------------------

def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    """Write *rows* to a CSV file at *path*."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def read_csv_as_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def csv_to_typed(
    path: Path,
    converters: dict[str, type],
) -> list[dict[str, Any]]:
    """
    Read CSV and apply type converters to each column.

    Args:
        path: Path to the CSV file.
        converters: Mapping of column name → callable for conversion.

    Returns:
        List of row dicts with converted types.
    """
    rows = read_csv_as_dicts(path)
    result: list[dict[str, Any]] = []
    for row in rows:
        typed: dict[str, Any] = {}
        for key, value in row.items():
            converter = converters.get(key)
            typed[key] = converter(value) if converter else value
        result.append(typed)
    return result


# ---------------------------------------------------------------------------
# 3. JSON
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialise *data* to a JSON file at *path*."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def read_json(path: Path) -> Any:
    """Deserialise a JSON file at *path*."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def merge_json_files(paths: list[Path]) -> dict[str, Any]:
    """
    Load multiple JSON object files and merge them left-to-right.

    Later files override keys from earlier files.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        data = read_json(p)
        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON object in {p}, got {type(data).__name__}")
        merged |= data
    return merged


# ---------------------------------------------------------------------------
# 4. Binary files
# ---------------------------------------------------------------------------

def write_binary(path: Path, data: bytes) -> None:
    """Write raw bytes to *path*."""
    path.write_bytes(data)


def read_binary(path: Path) -> bytes:
    """Read raw bytes from *path*."""
    return path.read_bytes()


# ---------------------------------------------------------------------------
# 5. Pathlib utilities
# ---------------------------------------------------------------------------

def list_files(directory: Path, suffix: str = "") -> list[Path]:
    """
    List all files in *directory* (recursively) with an optional suffix filter.
    """
    pattern = f"**/*{suffix}" if suffix else "**/*"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def safe_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst*, creating parent dirs and never overwriting."""
    if dst.exists():
        raise FileExistsError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run I/O demonstrations using a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Text file
        print("=== Text File ===")
        poem = tmp / "poem.txt"
        write_text_file(poem, "Roses are red\nViolets are blue\nPython is great\n")
        print(f"  Lines: {count_lines(poem)}")
        print(f"  Grep 'blue': {grep_file(poem, 'blue')}")

        # CSV
        print("\n=== CSV ===")
        sales = tmp / "sales.csv"
        headers = ["date", "product", "quantity", "price"]
        rows: list[list[Any]] = [
            ["2024-01-01", "Widget A", 10, 9.99],
            ["2024-01-02", "Widget B", 5, 19.99],
            ["2024-01-03", "Widget A", 7, 9.99],
        ]
        write_csv(sales, headers, rows)
        typed_rows = csv_to_typed(sales, {"quantity": int, "price": float})
        for r in typed_rows:
            print(f"  {r}")

        # JSON
        print("\n=== JSON ===")
        config = tmp / "config.json"
        write_json(config, {"debug": True, "host": "localhost", "port": 8080})
        loaded = read_json(config)
        print(f"  Loaded: {loaded}")

        override = tmp / "override.json"
        write_json(override, {"port": 9090, "workers": 4})
        merged = merge_json_files([config, override])
        print(f"  Merged: {merged}")

        # Binary
        print("\n=== Binary ===")
        bin_file = tmp / "data.bin"
        write_binary(bin_file, bytes(range(256)))
        data = read_binary(bin_file)
        print(f"  Read {len(data)} bytes, first 8: {list(data[:8])}")

        # Pathlib
        print("\n=== Pathlib ===")
        all_files = list_files(tmp)
        for f in all_files:
            size = f.stat().st_size
            print(f"  {f.name}: {size} bytes")


if __name__ == "__main__":
    main()

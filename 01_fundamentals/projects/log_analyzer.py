"""
projects/log_analyzer.py
=========================
Parse structured log files, extract patterns, count events by level,
and generate summary reports.

Supported log format (common/combined-style):
    LEVEL TIMESTAMP [module] message

Run directly:  python log_analyzer.py <log_file>
               python log_analyzer.py --demo
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Matches lines like:
#   ERROR 2024-01-15T10:23:01 [auth] Invalid token received
LOG_PATTERN = re.compile(
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?P<timestamp>\S+)\s+"
    r"\[(?P<module>[^\]]+)\]\s+"
    r"(?P<message>.+)"
)

DEMO_LOG = """\
INFO 2024-01-15T08:00:00 [app] Application started
DEBUG 2024-01-15T08:00:01 [db] Connected to database
INFO 2024-01-15T08:01:00 [auth] User alice logged in
WARNING 2024-01-15T08:05:00 [app] High memory usage: 85%
ERROR 2024-01-15T08:10:00 [auth] Invalid token received from 192.168.1.50
INFO 2024-01-15T08:11:00 [auth] User bob logged in
ERROR 2024-01-15T08:15:00 [db] Query timeout after 30s
CRITICAL 2024-01-15T08:16:00 [db] Database connection lost
ERROR 2024-01-15T08:16:01 [app] Failed to process request: DB unavailable
INFO 2024-01-15T08:20:00 [db] Reconnected to database
WARNING 2024-01-15T08:22:00 [app] Slow response time: 4.2s
INFO 2024-01-15T09:00:00 [app] Daily backup started
INFO 2024-01-15T09:15:00 [app] Daily backup completed
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single parsed log entry."""

    level: str
    timestamp: datetime
    module: str
    message: str
    line_number: int

    @classmethod
    def parse(cls, line: str, line_number: int) -> LogEntry | None:
        """Parse a raw log line; return None if it doesn't match."""
        m = LOG_PATTERN.match(line.strip())
        if not m:
            return None
        try:
            ts = datetime.fromisoformat(m.group("timestamp"))
        except ValueError:
            ts = datetime.min
        return cls(
            level=m.group("level"),
            timestamp=ts,
            module=m.group("module"),
            message=m.group("message"),
            line_number=line_number,
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """Summary of a log file analysis."""

    total_lines: int
    parsed_lines: int
    skipped_lines: int
    level_counts: dict[str, int]
    module_counts: dict[str, int]
    errors: list[LogEntry]
    warnings: list[LogEntry]
    time_range: tuple[datetime, datetime] | None


class LogAnalyzer:
    """Analyse a collection of log lines and produce a report."""

    LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def parse_stream(self, stream: io.TextIOBase | io.StringIO) -> None:
        """Parse all lines from a text stream."""
        for lineno, line in enumerate(stream, start=1):
            entry = LogEntry.parse(line, lineno)
            if entry:
                self._entries.append(entry)

    def parse_file(self, path: Path) -> None:
        """Parse log entries from a file on disk."""
        with path.open(encoding="utf-8") as fh:
            self.parse_stream(fh)  # type: ignore[arg-type]

    def parse_string(self, text: str) -> None:
        """Parse log entries from a multi-line string."""
        self.parse_stream(io.StringIO(text))  # type: ignore[arg-type]

    def analyze(self) -> AnalysisReport:
        """Return a complete analysis report."""
        if not self._entries:
            return AnalysisReport(
                total_lines=0,
                parsed_lines=0,
                skipped_lines=0,
                level_counts={},
                module_counts={},
                errors=[],
                warnings=[],
                time_range=None,
            )

        level_counts = Counter(e.level for e in self._entries)
        module_counts = Counter(e.module for e in self._entries)
        errors   = [e for e in self._entries if e.level in ("ERROR", "CRITICAL")]
        warnings = [e for e in self._entries if e.level == "WARNING"]
        timestamps = [e.timestamp for e in self._entries if e.timestamp != datetime.min]
        time_range = (min(timestamps), max(timestamps)) if timestamps else None

        return AnalysisReport(
            total_lines=self._entries[-1].line_number if self._entries else 0,
            parsed_lines=len(self._entries),
            skipped_lines=0,
            level_counts=dict(level_counts),
            module_counts=dict(module_counts),
            errors=errors,
            warnings=warnings,
            time_range=time_range,
        )

    def module_error_rate(self) -> dict[str, float]:
        """Return fraction of ERROR/CRITICAL entries per module."""
        by_module: defaultdict[str, list[str]] = defaultdict(list)
        for e in self._entries:
            by_module[e.module].append(e.level)
        rates: dict[str, float] = {}
        for module, levels in by_module.items():
            err_count = sum(1 for lv in levels if lv in ("ERROR", "CRITICAL"))
            rates[module] = err_count / len(levels)
        return dict(sorted(rates.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: AnalysisReport) -> None:
    """Pretty-print an AnalysisReport to stdout."""
    sep = "=" * 55
    print(sep)
    print("  LOG ANALYSIS REPORT")
    print(sep)

    if report.time_range:
        start, end = report.time_range
        duration = end - start
        print(f"  Period:  {start.isoformat()} → {end.isoformat()}")
        print(f"  Duration: {duration}")

    print(f"  Parsed:  {report.parsed_lines} entries")
    print()
    print("  Level breakdown:")
    for level in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"):
        count = report.level_counts.get(level, 0)
        bar = "█" * count
        print(f"    {level:<10} {count:>4}  {bar}")

    print()
    print("  Module breakdown:")
    for module, count in sorted(
        report.module_counts.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"    {module:<15} {count:>4}")

    if report.errors:
        print()
        print("  ⚠  Errors / Critical:")
        for e in report.errors:
            print(f"    [{e.level}] {e.timestamp.isoformat()} [{e.module}] {e.message}")

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="log_analyzer",
        description="Parse and summarize log files",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("log_file", nargs="?", type=Path, help="Path to the log file")
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo log data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    analyzer = LogAnalyzer()

    if args.demo:
        analyzer.parse_string(DEMO_LOG)
    else:
        path: Path = args.log_file
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        analyzer.parse_file(path)

    report = analyzer.analyze()
    print_report(report)

    err_rates = analyzer.module_error_rate()
    print("\nModule error rates:")
    for module, rate in err_rates.items():
        print(f"  {module:<15} {rate * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

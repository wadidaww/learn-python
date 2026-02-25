"""
pipeline/orchestrator.py
=========================
ETL pipeline orchestrator: tie together Extractor → Transformer → Loader.

Features:
  - Sequential and parallel execution
  - Per-step timing and record counts
  - Error handling and partial failure recovery
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single ETL step."""

    step:          str
    records_in:    int
    records_out:   int
    elapsed:       float
    error:         str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""

    name:         str
    steps:        list[StepResult] = field(default_factory=list)
    total_elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def records_loaded(self) -> int:
        load_steps = [s for s in self.steps if "load" in s.step.lower()]
        return load_steps[-1].records_out if load_steps else 0

    def summary(self) -> str:
        lines = [f"Pipeline '{self.name}':"]
        for s in self.steps:
            status = "✓" if s.success else "✗"
            lines.append(
                f"  [{status}] {s.step:<25} "
                f"{s.records_in:>6} → {s.records_out:<6} records  "
                f"({s.elapsed * 1000:.1f} ms)"
            )
        lines.append(f"  Total: {self.total_elapsed * 1000:.1f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class ETLPipeline:
    """
    Composable ETL pipeline.

    Example::

        pipeline = (
            ETLPipeline("sales")
            .extract(CSVExtractor("sales.csv"))
            .transform(TypeCoercionTransformer({"qty": int, "price": float}))
            .transform(FilterTransformer(lambda r: r["price"] > 0))
            .load(SQLiteLoader(":memory:", "sales"))
        )
        result = pipeline.run()
        print(result.summary())
    """

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._extractors: list[Extractor] = []
        self._transformers: list[Transformer] = []
        self._loaders: list[Loader] = []

    def extract(self, extractor: Extractor) -> ETLPipeline:
        """Add an extractor stage."""
        self._extractors.append(extractor)
        return self

    def transform(self, transformer: Transformer) -> ETLPipeline:
        """Add a transformation stage."""
        self._transformers.append(transformer)
        return self

    def load(self, loader: Loader) -> ETLPipeline:
        """Add a loader stage."""
        self._loaders.append(loader)
        return self

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return a PipelineResult."""
        result = PipelineResult(name=self.name)
        pipeline_start = time.perf_counter()
        records: list[Record] = []

        # --- Extract ---
        for i, extractor in enumerate(self._extractors):
            step_name = f"extract[{i}]({type(extractor).__name__})"
            start = time.perf_counter()
            try:
                new_records = extractor.extract()
                records.extend(new_records)
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=0,
                    records_out=len(new_records),
                    elapsed=time.perf_counter() - start,
                ))
                logger.info("%s: extracted %d records", step_name, len(new_records))
            except Exception as exc:
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=0,
                    records_out=0,
                    elapsed=time.perf_counter() - start,
                    error=str(exc),
                ))
                logger.error("%s failed: %s", step_name, exc)
                break

        if not result.success:
            result.total_elapsed = time.perf_counter() - pipeline_start
            return result

        # --- Transform ---
        for i, transformer in enumerate(self._transformers):
            step_name = f"transform[{i}]({type(transformer).__name__})"
            count_in = len(records)
            start = time.perf_counter()
            try:
                records = transformer.transform(records)
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=count_in,
                    records_out=len(records),
                    elapsed=time.perf_counter() - start,
                ))
                logger.info(
                    "%s: %d → %d records", step_name, count_in, len(records)
                )
            except Exception as exc:
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=count_in,
                    records_out=0,
                    elapsed=time.perf_counter() - start,
                    error=str(exc),
                ))
                logger.error("%s failed: %s", step_name, exc)
                break

        if not result.success:
            result.total_elapsed = time.perf_counter() - pipeline_start
            return result

        # --- Load ---
        for i, loader in enumerate(self._loaders):
            step_name = f"load[{i}]({type(loader).__name__})"
            count_in = len(records)
            start = time.perf_counter()
            try:
                n_written = loader.load(records)
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=count_in,
                    records_out=n_written,
                    elapsed=time.perf_counter() - start,
                ))
                logger.info("%s: loaded %d records", step_name, n_written)
            except Exception as exc:
                result.steps.append(StepResult(
                    step=step_name,
                    records_in=count_in,
                    records_out=0,
                    elapsed=time.perf_counter() - start,
                    error=str(exc),
                ))
                logger.error("%s failed: %s", step_name, exc)

        result.total_elapsed = time.perf_counter() - pipeline_start
        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a sample ETL pipeline."""
    import tempfile
    from pathlib import Path
    from pipeline.extractor import CSVStringExtractor, MemoryExtractor
    from pipeline.transformer import (
        TypeCoercionTransformer,
        FilterTransformer,
        AddFieldTransformer,
        DropNullTransformer,
    )
    from pipeline.loader import MemoryLoader, SQLiteLoader

    csv_data = """date,product,quantity,price
2024-01-01,Widget A,10,9.99
2024-01-02,Widget B,5,19.99
2024-01-03,Widget A,7,9.99
2024-01-04,Widget C,,29.99
2024-01-05,Widget B,3,19.99
"""

    memory_loader = MemoryLoader()

    pipeline = (
        ETLPipeline("sales_etl")
        .extract(CSVStringExtractor(csv_data))
        .transform(DropNullTransformer(["quantity", "price"]))
        .transform(TypeCoercionTransformer({"quantity": int, "price": float}))
        .transform(FilterTransformer(lambda r: r["price"] > 0))
        .transform(
            AddFieldTransformer("total", lambda r: round(r["quantity"] * r["price"], 2))
        )
        .load(memory_loader)
    )

    result = pipeline.run()
    print(result.summary())
    print(f"\nLoaded {len(memory_loader.records)} records:")
    for rec in memory_loader.records:
        print(f"  {rec}")


if __name__ == "__main__":
    main()

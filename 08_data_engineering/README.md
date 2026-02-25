# Module 08: Data Engineering

Build ETL (Extract, Transform, Load) pipelines using only the Python standard library.

## Topics Covered

| File | Concepts |
|------|----------|
| `pipeline/extractor.py` | CSV, JSON, and URL data extraction |
| `pipeline/transformer.py` | Data cleaning, validation, type coercion |
| `pipeline/loader.py` | Loading to CSV, JSON, and SQLite sinks |
| `pipeline/orchestrator.py` | Pipeline composition and execution |

## Running

```bash
python pipeline/orchestrator.py
pytest tests/ -v
```

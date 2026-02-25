# Learn Python: A Comprehensive Learning Ecosystem

A structured, production-grade Python learning path covering fundamentals through expert-level topics.

## Prerequisites

- Python 3.11+
- Git
- `make` (optional, for convenience targets)

## Quick Start

```bash
git clone https://github.com/your-org/learn-python.git
cd learn-python
pip install -e ".[dev]"
make test
```

## Learning Path

Work through modules **in order**. Each module builds on the previous.

| # | Module | Topics |
|---|--------|--------|
| 01 | **Fundamentals** | Syntax, data structures, OOP, error handling, file I/O |
| 02 | **Algorithms & Data Structures** | Hash tables, heaps, tries, LRU cache, sorting, graphs, DP |
| 03 | **Design Patterns** | Singleton, Factory, Observer, Strategy, Decorator, DI |
| 04 | **Testing & Code Quality** | pytest, fixtures, mocks, parametrize, property-based testing |
| 05 | **Performance & Optimization** | Profiling, asyncio, multiprocessing, concurrent.futures |
| 06 | **Systems Programming** | Sockets, HTTP server, subprocess, task scheduling |
| 07 | **Backend Production** | FastAPI, Pydantic, JWT auth, Docker |
| 08 | **Data Engineering** | ETL pipelines, CSV/JSON/API extraction, transformation |
| 09 | **ML Engineering** | Feature engineering, model training, evaluation, serving |
| 10 | **Expert Level** | Async web framework, routing, middleware, task queue |

## How to Navigate

Each module has:
- A `README.md` explaining concepts
- Python source files with rich docstrings and type hints
- A `tests/` directory with runnable pytest tests
- A `projects/` directory (where applicable) with real mini-apps

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest 01_fundamentals/tests/ -v

# With coverage
pytest --cov=. --cov-report=html
```

## Linting & Formatting

```bash
make lint        # ruff check
make format      # black + isort
make type-check  # mypy
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/add-example`)
3. Commit your changes with clear messages
4. Open a Pull Request

Please follow the existing code style: type hints everywhere, docstrings on all public APIs, tests for new functionality.

## License

MIT — see [LICENSE](LICENSE).

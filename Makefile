.PHONY: install test lint format type-check clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project with dev dependencies
	pip install -e ".[dev]"

test: ## Run all tests with coverage
	pytest --cov=. --cov-report=term-missing --cov-report=html -q

test-fast: ## Run tests without coverage (faster)
	pytest -x -q

lint: ## Lint code with ruff
	ruff check .

format: ## Format code with black and isort
	black .
	isort .

type-check: ## Run mypy type checker
	mypy 01_fundamentals 02_algorithms 03_design_patterns 04_testing \
	     05_performance 08_data_engineering 09_ml_engineering 10_expert \
	     --ignore-missing-imports

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov      -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

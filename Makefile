.PHONY: help install format lint typecheck test check clean run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv (10-100x faster than pip)
	uv sync --extra dev
	uv run pre-commit install

format:  ## Format code with Ruff
	uv run ruff format .

lint:  ## Run Ruff linter (with auto-fix)
	uv run ruff check . --fix

typecheck:  ## Run Mypy type checker on critical modules
	uv run mypy code_validator.py gemini_client.py config.py

test:  ## Run tests with coverage
	uv run pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without integration tests
	uv run pytest tests/ -m "not integration" --cov=. --cov-report=term-missing

check:  ## Run all checks (format, lint, typecheck, test)
	@echo "Running format check..."
	uv run ruff format --check .
	@echo "\nRunning linter..."
	uv run ruff check .
	@echo "\nRunning type checker..."
	uv run mypy code_validator.py gemini_client.py config.py
	@echo "\nRunning tests..."
	uv run pytest tests/ -m "not integration" --cov=. --cov-report=term-missing

clean:  ## Remove cache and generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run:  ## Run the evolution prototype
	uv run python prototype.py

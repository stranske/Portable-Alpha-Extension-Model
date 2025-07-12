.PHONY: help setup install install-dev test lint typecheck ci clean demo dashboard docs

# Default target
help:
	@echo "Portable Alpha Extension Model - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  install    - Install package in development mode"
	@echo "  install-dev- Install package with development dependencies"
	@echo "  test       - Run test suite"
	@echo "  lint       - Run linting checks"
	@echo "  typecheck  - Run type checking"
	@echo "  ci         - Run full CI pipeline"
	@echo "  clean      - Clean up build artifacts"
	@echo "  demo       - Run CLI demo with sample data"
	@echo "  dashboard  - Start Streamlit dashboard"
	@echo "  docs       - Generate documentation"
	@echo ""

# Environment setup
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Testing and quality checks
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=pa_core --cov-report=html --cov-report=term

lint:
	python -m ruff check pa_core

lint-fix:
	python -m ruff check pa_core --fix

typecheck:
	python -m pyright

ci: lint typecheck test

# Development utilities
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/

demo:
	python -m pa_core.cli --params parameters.csv --index sp500tr_fred_divyield.csv

dashboard:
	python -m streamlit run dashboard/app.py

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Security check
security:
	python -m bandit -r pa_core/

# Format code
format:
	python -m black pa_core/ tests/
	python -m isort pa_core/ tests/

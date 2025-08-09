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
	@echo "  sync       - Sync with remote main branch"
	@echo "  check-updates - Check for remote updates from Codex"
	@echo "  dev-check  - Run all development checks (format+lint+test)"
	@echo "  hooks      - Install git pre-commit hooks"
	@echo "  hooks-run  - Run pre-commit on all files"
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
	@if [ ! -d "dev-env" ]; then \
		echo "🔧 Setting up development environment..."; \
		python3 -m venv dev-env; \
		dev-env/bin/pip install black isort flake8 mypy; \
	fi
	@echo "🔍 Linting code..."
	dev-env/bin/flake8 pa_core/ tests/ dashboard/ --max-line-length=88 --ignore=E203,W503

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
	@if [ ! -d "dev-env" ]; then \
		echo "🔧 Setting up development environment..."; \
		python3 -m venv dev-env; \
		dev-env/bin/pip install black isort flake8 mypy; \
	fi
	@echo "🎨 Formatting code..."
	dev-env/bin/black pa_core/ tests/ dashboard/
	dev-env/bin/isort pa_core/ tests/ dashboard/

# Pre-commit
hooks:
	PRE_COMMIT_HOME=$(PWD)/.cache/pre-commit pre-commit install

hooks-run:
	PRE_COMMIT_HOME=$(PWD)/.cache/pre-commit pre-commit run --all-files

# Git workflow commands
sync:
	@echo "🔄 Syncing with remote main..."
	git fetch origin
	git checkout main
	git pull origin main
	@echo "✅ Synced with remote main"

sync-rebase:
	@echo "🔄 Rebasing current branch on main..."
	git fetch origin
	git rebase origin/main
	@echo "✅ Rebased on latest main"

dev-check: format lint typecheck test
	@echo "✅ All development checks passed!"

# Check for changes from Codex
check-updates:
	@echo "📡 Checking for remote updates..."
	git fetch origin
	@if [ "$$(git rev-list HEAD..origin/main --count)" -gt 0 ]; then \
		echo "🚨 Remote updates available! Run 'make sync' to update."; \
		git log --oneline HEAD..origin/main; \
	else \
		echo "✅ Up to date with remote."; \
	fi

# Methodical debugging workflow for Codex PRs
debug-codex:
	@echo "🎯 Running methodical Codex PR debugging..."
	python scripts/debug_codex_pr.py --branch=$(shell git branch --show-current) --max-iterations=3

debug-codex-fix:
	@echo "🔧 Running methodical Codex PR debugging with auto-commit..."
	python scripts/debug_codex_pr.py --branch=$(shell git branch --show-current) --max-iterations=3 --commit

debug-codex-report:
	@echo "📄 Generating methodical Codex PR debugging report..."
	python scripts/debug_codex_pr.py --branch=$(shell git branch --show-current) --max-iterations=3 --report=debug_report.md

# Quick CI/CD validation workflow
validate-pr: debug-codex dev-check
	@echo "✅ PR validation complete!"

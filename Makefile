.PHONY: help install install-dev test format lint type-check clean

help:
	@echo "Available targets:"
	@echo "  install      - Install package in production mode"
	@echo "  install-dev  - Install package with dev dependencies"
	@echo "  test         - Run pytest tests"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run ruff linter"
	@echo "  type-check   - Run mypy type checker"
	@echo "  clean        - Remove build artifacts and cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -v

format:
	black src/ tests/
	isort src/ tests/

lint:
	ruff check src/ tests/

type-check:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

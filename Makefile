.PHONY: setup lint lint-fix test test-coverage clean

# Default target
all: lint test

# Setup development environment
setup:
	python scripts/setup_dev_environment.py

# Run code quality checks
lint:
	python scripts/run_code_checks.py check

# Fix code quality issues automatically
lint-fix:
	python scripts/run_code_checks.py fix

# Run tests
test:
	pytest tests/

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=yahoofinance --cov-report=term --cov-report=html

# Clean up temporary files
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make lint           - Run code quality checks"
	@echo "  make lint-fix       - Fix code quality issues automatically"
	@echo "  make test           - Run tests"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make all            - Run lint and test (default)"
	@echo "  make help           - Show this help message"
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
	rm -rf .temp/*

# Clean cache files
clean-cache:
	rm -rf .cache/*
	rm -rf yahoofinance/data/cache/*
	rm -rf yahoofinance/output/backtest/cache/*

# Clean logs (keep last 5)
clean-logs:
	cd logs && ls -t *.log 2>/dev/null | tail -n +6 | xargs -r rm --

# Clean output files
clean-output:
	rm -f yahoofinance/output/*.html
	rm -f yahoofinance/output/*.csv

# Deep clean - removes all generated files
clean-all: clean clean-cache clean-logs clean-output
	rm -rf output/*

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make lint           - Run code quality checks"
	@echo "  make lint-fix       - Fix code quality issues automatically"
	@echo "  make test           - Run tests"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make clean-cache    - Clean all cache files"
	@echo "  make clean-logs     - Clean old log files (keep last 5)"
	@echo "  make clean-output   - Clean output HTML/CSV files"
	@echo "  make clean-all      - Deep clean all generated files"
	@echo "  make all            - Run lint and test (default)"
	@echo "  make help           - Show this help message"
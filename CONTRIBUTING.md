# Contributing to etorotrade

Thank you for considering contributing to etorotrade! This document outlines the guidelines and workflow for contributing to the project.

## Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/etorotrade.git
   cd etorotrade
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Set up development environment**:
   ```bash
   # Quick setup (installs dependencies and sets up pre-commit hooks)
   python scripts/setup_dev_environment.py
   
   # Or manually:
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   pre-commit install
   ```

5. **Run tests to verify setup**:
   ```bash
   pytest tests/
   ```

## Code Style and Standards

We follow these coding standards:

1. **PEP 8** style guide with some modifications:
   - Line length: 100 characters
   - Use 4 spaces for indentation
   - Use meaningful variable and function names

2. **Documentation**:
   - Use Google/NumPy style docstrings
   - Document all public classes, methods, and functions
   - Include type hints for all function parameters and return values

3. **Imports**:
   - Sort imports using isort (automatically handled by pre-commit)
   - Use absolute imports from the package root
   - Group imports: standard library, third-party, and local

4. **Error Handling**:
   - Use the custom error hierarchy in `yahoofinance.core.errors`
   - Include context information in error messages
   - Always catch specific exceptions, not generic `Exception`

5. **Logging**:
   - Use the standardized logging configuration in `yahoofinance.core.logging_config`
   - Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - Include context information in log messages

6. **Testing**:
   - Write tests for all new features and bug fixes
   - Aim for at least 80% code coverage
   - Use pytest fixtures and parametrization for more comprehensive tests

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code that adheres to the standards above
   - Add or update tests as necessary
   - Keep commits focused and atomic

3. **Run tests and linting**:
   ```bash
   # Makefile commands (recommended)
   make lint        # Run all code quality checks
   make lint-fix    # Auto-fix code quality issues
   make test        # Run tests
   make test-coverage # Run tests with coverage report
   
   # Alternatively, use the provided scripts
   python scripts/run_code_checks.py        # Run all checks
   python scripts/run_code_checks.py --fix  # Fix issues
   
   # Bash script is also available
   ./scripts/lint.sh      # Run checks
   ./scripts/lint.sh fix  # Fix issues
   
   # Run individual tools manually
   pytest tests/
   black yahoofinance trade.py tests
   isort yahoofinance trade.py tests
   flake8 yahoofinance trade.py tests
   mypy yahoofinance trade.py tests
   ```

4. **Submit a pull request**:
   - Include a clear description of the changes
   - Reference any related issues
   - Ensure all tests and checks pass

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate.
2. Update the version number in relevant files following [Semantic Versioning](https://semver.org/).
3. Your pull request will be reviewed by at least one maintainer.
4. Once approved, your pull request will be merged.

## Provider Pattern

When working with financial data sources:

1. Always use the provider pattern through `get_provider()` function
2. Avoid direct usage of third-party libraries like `yfinance` or `yahooquery`
3. Implement proper rate limiting and circuit breaker patterns
4. Consider both synchronous and asynchronous usage patterns

## Error Handling

Always use the custom error hierarchy:

```python
from yahoofinance.core.errors import YFinanceError, APIError, ValidationError

try:
    # Code that might fail
    pass
except APIError as e:
    # Handle API-specific errors
    logger.error(f"API error: {str(e)}")
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Validation error: {str(e)}")
except YFinanceError as e:
    # Handle other Yahoo Finance errors
    logger.error(f"Yahoo Finance error: {str(e)}")
```

## Logging

Use the standardized logging configuration:

```python
from yahoofinance.core.logging_config import get_logger

logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")

# For ticker-specific logging
ticker = "AAPL"
ticker_logger = get_ticker_logger(logger, ticker)
ticker_logger.info(f"Processing data for {ticker}")
```

## Questions?

If you have any questions or need help, please create an issue in the repository.
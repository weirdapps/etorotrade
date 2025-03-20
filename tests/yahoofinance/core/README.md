# Core Module Tests

This directory contains tests for the core module components:

- `test_client.py`: Tests for YFinanceClient (moved from /tests/test_client.py)
- `test_cache.py`: Tests for caching system (moved from /tests/test_cache.py)
- `test_errors.py`: Tests for error hierarchy (moved from /tests/test_errors.py)
- `test_types.py`: Tests for data types (moved from /tests/test_types.py)
- `test_config.py`: Tests for configuration system

## Organization

Tests mirror the structure of the `yahoofinance.core` package to make it easier to locate tests for specific components.

## Running Tests

Run all core tests:
```
pytest tests/yahoofinance/core/
```

Run a specific test file:
```
pytest tests/yahoofinance/core/test_client.py
```

Run a specific test:
```
pytest tests/yahoofinance/core/test_client.py::TestYFinanceClient::test_get_ticker_info
```
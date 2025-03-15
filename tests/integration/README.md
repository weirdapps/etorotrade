# Integration Tests

This directory contains integration tests that verify multiple components working together.
Unlike unit tests that focus on isolated components, these tests ensure proper interaction
between different modules.

## Test Organization

Integration tests should be organized by feature area:

- `test_api_integration.py` - Tests for API client, rate limiting, and caching
- `test_market_integration.py` - Tests for market data analysis pipeline
- `test_portfolio_integration.py` - Tests for portfolio analysis
- `test_trade_integration.py` - Tests for trading recommendations

## Writing Integration Tests

When writing integration tests:

1. Focus on component interactions rather than individual methods
2. Use real (or realistic) data when possible
3. Minimize mocking to essential external services
4. Test complete workflows from input to output
5. Verify that data moves correctly between components

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run specific integration test
pytest tests/integration/test_api_integration.py
```
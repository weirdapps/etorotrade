# etorotrade Test Suite

This directory contains the test suite for the etorotrade application. The tests are organized according to their type and the module they test.

## Test Organization

The tests are undergoing reorganization to better align with best practices and the codebase structure. The target directory structure is:

- `unit/`: Unit tests testing individual components in isolation
  - `core/`: Tests for core modules like client, cache, errors, rate limiters
  - `api/`: Tests for API providers and interfaces
  - `utils/`: Tests for utility functions and helpers
    - `async/`: Tests for async utilities
    - `data/`: Tests for data formatting utilities
    - `market/`: Tests for market utilities
    - `network/`: Tests for network utilities (rate limiting, pagination)
  - `modules/`: Tests for specific feature modules
- `integration/`: Integration tests between components
  - `api/`: Tests for API client and provider integrations
  - `display/`: Tests for display integrations
- `e2e/`: End-to-end tests of complete workflows
- `performance/`: Performance and benchmark tests
- `fixtures/`: Reusable test fixtures
  - `market_data/`: Market data fixtures
  - `api_responses/`: API response fixtures
- `utils/`: Test utilities and helpers

**Note on Migration**: Currently, many test files are still in the root directory. New tests should be added to the appropriate subdirectory, and existing tests are being gradually migrated to this structure.

## Test Naming Conventions

- Test files: `test_<module>_<component>.py`
- Test classes: `Test<Component><Functionality>`
- Test methods: `test_<functionality>_<scenario>`

## Running Tests

### Running All Tests

```bash
pytest tests/
```

### Running Tests with Coverage

```bash
pytest tests/ --cov=yahoofinance
```

### Running Specific Tests

```bash
# Run tests in a specific directory
pytest tests/unit/

# Run a specific test file
pytest tests/unit/api/test_providers.py

# Run a specific test class
pytest tests/unit/api/test_providers.py::TestYahooFinanceProvider

# Run a specific test method
pytest tests/unit/api/test_providers.py::TestYahooFinanceProvider::test_get_ticker_info
```

### Test Categories

The tests use pytest markers to categorize tests. You can run tests with specific markers:

```bash
# Run unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run end-to-end tests
pytest -m e2e

# Run slow tests
pytest -m slow

# Run tests that require API access
pytest -m api

# Run tests that require network connectivity
pytest -m network
```

## Test Fixtures

The test suite uses fixtures to provide reusable test data and objects. Fixtures are defined in:

- `conftest.py`: Global fixtures available to all tests
- `fixtures/`: Directory containing specialized fixtures by category

### Key Fixtures

- `mock_client`: A mock YFinanceClient object
- `mock_stock_data`: Mock stock data with reasonable defaults
- `test_dataframe`: A test DataFrame with market data
- Market scenario fixtures:
  - `bull_market_data`, `bear_market_data`, `volatile_market_data`: Different market conditions
  - `bull_market_provider_data`, `bear_market_provider_data`, `volatile_market_provider_data`: Provider response versions

## Test Best Practices

1. **Test Isolation**: Ensure tests don't depend on global state
2. **Clear Intent**: Each test should have a clear purpose described in its docstring
3. **Arrange-Act-Assert**: Structure tests with setup, action, and verification phases
4. **Appropriate Mocking**: Mock external dependencies but not the functionality under test
5. **Descriptive Names**: Use clear, descriptive names for test methods
6. **Comprehensive Assertions**: Verify all relevant aspects of the expected outcome
7. **Edge Cases**: Include tests for edge cases and error conditions
8. **Performance**: Keep tests fast, using appropriate markers for slow tests

## Common Test Patterns

### Testing API Components

```python
def test_api_function(mock_client):
    # Arrange: Set up mock responses
    mock_client.get_data.return_value = {"key": "value"}
    
    # Act: Call the function under test
    result = function_using_api(mock_client)
    
    # Assert: Verify the result
    assert result["processed_key"] == "processed_value"
```

### Testing Error Handling

```python
def test_error_handling(mock_client):
    # Arrange: Set up mock to raise an exception
    mock_client.get_data.side_effect = APIError("Test error")
    
    # Act & Assert: Verify exception handling
    with pytest.raises(APIError):
        function_using_api(mock_client)
```

### Testing with Parametrization

```python
@pytest.mark.parametrize("input_data,expected", [
    ({"ticker": "AAPL", "price": 150}, True),
    ({"ticker": "MSFT", "price": 250}, False),
])
def test_parameterized_function(input_data, expected):
    assert evaluate_condition(input_data) == expected
```
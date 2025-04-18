# etorotrade Test Suite

This directory contains the test suite for the etorotrade application. The tests are organized to mirror the package structure for better discoverability and maintenance.

## Test Organization

Tests are structured to mirror the main package organization for easy navigation and maintenance. This update:

1. Makes it easier to find tests for specific components
2. Clarifies the scope and relationships between tests
3. Improves test cohesion by grouping related tests
4. Establishes a standard pattern for adding new tests
5. Simplifies navigation between implementation and tests

### New Structure

Tests are organized in a hierarchical structure mirroring the package:

- `yahoofinance/`: Tests for the yahoofinance package components
  - `analysis/`: Tests for market analysis modules
    - `market/`: Tests for market-specific analysis
  - `api/`: Tests for API interfaces
    - `providers/`: Tests for data providers (Yahoo Finance, etc.)
  - `core/`: Tests for core functionality
    - Cache system, client, error handling, types, config
  - `data/`: Tests for data handling
  - `presentation/`: Tests for output formatting and display
  - `utils/`: Tests for utility modules
    - `async/`: Tests for async utilities and rate limiting
    - `data/`: Tests for data formatting utilities
    - `date/`: Tests for date utilities
    - `market/`: Tests for market-specific utilities
    - `network/`: Tests for network operations
      - `async_utils/`: Tests for async network utilities
  - `validators/`: Tests for validation functions
- `trade/`: Tests for main trade module functionality
- `debug/`: Scripts for debugging issues (previously scattered in project root)
- `e2e/`: End-to-end tests for complete workflows
- `integration/`: Integration tests for component interactions
- `unit/`: Unit tests organized by module
  - `api/`: Unit tests for API components
  - `core/`: Unit tests for core functionality
  - `trade/`: Unit tests for trade functionality
  - `utils/`: Unit tests for utility modules
- `fixtures/`: Shared test fixtures
- `conftest.py`: Pytest configuration and global fixtures

### Categorization

Tests are also categorized by type using pytest markers:

- `@pytest.mark.unit`: Unit tests for isolated components
- `@pytest.mark.integration`: Tests verifying component interactions
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.api`: Tests requiring API access
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.network`: Tests requiring network connectivity
- `@pytest.mark.asyncio`: Tests for async functionality

**Migration Status**: This restructuring is in progress. New tests should follow this pattern, and existing tests are being gradually migrated.

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

### Testing Provider Pattern Components

```python
def test_provider_function(mock_provider):
    # Arrange: Set up mock provider responses
    mock_provider.get_ticker_info.return_value = {
        "ticker": "AAPL",
        "price": 150.0,
        "name": "Apple Inc."
    }
    
    # Act: Call the function under test with provider
    result = function_using_provider(mock_provider)
    
    # Assert: Verify the result
    assert result["ticker_name"] == "Apple Inc."
    assert result["price_formatted"] == "$150.00"
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

### Testing Retry Logic

```python
def test_retry_logic(mock_provider):
    # Arrange: Set up mock to fail twice then succeed
    mock_provider.get_ticker_info.side_effect = [
        RateLimitError("Rate limit exceeded", retry_after=1),
        ConnectionError("Network failure"),
        {"ticker": "AAPL", "price": 150.0}
    ]
    
    # Act: Call the function that should implement retries
    result = get_data_with_retries(mock_provider, "AAPL")
    
    # Assert: Verify final success
    assert result["ticker"] == "AAPL"
    assert result["price"] == 150.0
    assert mock_provider.get_ticker_info.call_count == 3
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
# etorotrade Test Suite

This directory contains the test suite for the etorotrade application. The tests are organized to mirror the package structure for better discoverability and maintenance.

## Test Organization

Tests are structured to mirror the main package organization for easy navigation and maintenance. This structure:

1. Makes it easier to find tests for specific components
2. Clarifies the scope and relationships between tests
3. Improves test cohesion by grouping related tests
4. Establishes a standard pattern for adding new tests
5. Simplifies navigation between implementation and tests

**Recent Updates (2025-01-18)**: All tests have been updated to reflect the recent performance optimizations and trade analysis fixes, ensuring comprehensive coverage of the enhanced functionality.

### Structure

Tests are organized in a hierarchical structure mirroring the package:

- `yahoofinance/`: Tests for the yahoofinance package components
  - `analysis/`: Tests for market analysis modules
  - `api/`: Tests for API interfaces and providers
  - `core/`: Tests for core functionality (cache, client, errors, types)
  - `data/`: Tests for data handling
  - `presentation/`: Tests for output formatting and display
  - `utils/`: Tests for utility modules
    - `network/`: Tests for network operations, rate limiting, and circuit breakers
- `integration/`: Integration tests for component interactions
- `unit/`: Unit tests organized by module
  - `trade/`: Tests for main trade module functionality
- `standalone/`: Standalone test files moved from project root
- `benchmarks/`: Performance benchmarking tests
- `debug/`: Debug utilities and test helpers
- `fixtures/`: Shared test fixtures
- `conftest.py`: Pytest configuration and global fixtures

### Categorization

Tests are categorized by type using pytest markers:

- `@pytest.mark.unit`: Unit tests for isolated components
- `@pytest.mark.integration`: Tests verifying component interactions
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.api`: Tests requiring API access
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.network`: Tests requiring network connectivity
- `@pytest.mark.asyncio`: Tests for async functionality (requires pytest-asyncio plugin)

## Test Naming Conventions

- Test files: `test_<module>.py` or `test_<module>_<component>.py`
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

# Run asyncio tests
pytest -m asyncio
```

## Test Fixtures

The test suite uses fixtures to provide reusable test data and objects. Fixtures are defined in:

- `conftest.py`: Global fixtures available to all tests
- `fixtures/`: Directory containing specialized fixtures by category

### Key Fixtures

- `mock_client`: A mock YFinanceClient object
- `mock_stock_data`: Mock stock data with reasonable defaults
- `test_dataframe`: A test DataFrame with market data

## Test Best Practices

Based on our experience building and maintaining this test suite, we recommend these critical practices:

1. **Test Isolation**: Ensure tests don't affect global state or other tests
   - Reset global variables before and after tests
   - Use unique resource names for each test
   - Clean up all resources in finally blocks
   - Use pytest fixtures with proper setup/teardown

2. **Thread Safety**: Ensure thread-safe access to shared resources
   - Use locks when accessing shared mutable data
   - Add delays between operations to avoid race conditions
   - Use thread-safe data structures

3. **Async Testing**: Properly test async code
   - Use the `pytest.mark.asyncio` decorator for async tests
   - Make sure all coroutines are properly awaited
   - Cancel remaining tasks when tests complete
   - Explicitly close resources with `await resource.close()`
   - Set up an event loop for tests with asyncio primitives: `set_event_loop(new_event_loop())`
   - Be cautious with asyncio.Lock() and similar primitives that require an event loop

4. **Mocking**: Use appropriate mocking techniques
   - Mock external dependencies but not the code under test
   - Ensure mocks have all required attributes/methods
   - Consider using real objects instead of mocks for complex behavior
   - Reset mocks between tests

5. **Global State Management**: Properly handle components with global state
   - Make copies of global state before modifying
   - Restore original state after tests
   - Use unique identifiers to prevent test interference
   - Register and clean up global resources explicitly

## Common Patterns

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

### Testing Global State Components

```python
def test_with_global_state():
    # Save original state
    original_state = dict(global_registry)
    
    try:
        # Clear for this test
        global_registry.clear()
        
        # Create test instance with unique name
        instance_name = f"test_instance_{uuid.uuid4()}"
        instance = TestClass(instance_name)
        
        # Register in global registry
        global_registry[instance_name] = instance
        
        # Test logic
        assert instance_name in global_registry
        assert global_registry[instance_name] is instance
        
    finally:
        # Clean up this test's data
        if instance_name in global_registry:
            del global_registry[instance_name]
            
        # Restore original state
        global_registry.clear()
        global_registry.update(original_state)
```

### Testing Async Components

```python
@pytest.mark.asyncio
async def test_async_function(event_loop):
    # Arrange
    provider = get_provider(async_api=True)
    
    try:
        # Act
        result = await provider.get_ticker_info("AAPL")
        
        # Assert
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        
    finally:
        # Clean up resources
        if hasattr(provider, 'close') and callable(provider.close()):
            cleanup_task = provider.close()
            if cleanup_task is not None and asyncio.iscoroutine(cleanup_task):
                await cleanup_task
```

### Testing with AsyncIO Primitives

When testing components that use asyncio primitives like Lock, Event, etc., you need to ensure an event loop is set:

```python
import asyncio
from asyncio import set_event_loop, new_event_loop

# For test functions that use asyncio primitives
def test_async_primitives():
    # Create and set an event loop for this test
    set_event_loop(new_event_loop())
    
    # Now you can safely create asyncio primitives
    lock = asyncio.Lock()
    event = asyncio.Event()
    
    # Test your component that uses these primitives
    component = AsyncComponent(lock=lock, event=event)
    assert component is not None

# For fixture-based setup
@pytest.fixture
def async_rate_limiter():
    """Create an async rate limiter with its own event loop."""
    # Create and set a new event loop for this fixture
    set_event_loop(new_event_loop())
    
    # Create the rate limiter (which uses asyncio.Lock internally)
    limiter = AsyncRateLimiter(
        window_size=5,
        max_calls=20,
        base_delay=0.01
    )
    return limiter
```

### Testing Rate Limiting

```python
def test_rate_limiter():
    # Create isolated rate limiter for testing
    test_limiter = RateLimiter(window_size=10, max_calls=100)
    
    # Reset to clean state with thread safety
    with test_limiter.lock:
        test_limiter.call_timestamps = []
        test_limiter.success_streak = 0
        test_limiter.failure_streak = 0
        test_limiter.delay = test_limiter.base_delay
    
    # Define function with rate limiter
    @rate_limited(limiter=test_limiter)
    def test_function(x):
        return x * 2
    
    # Test with small delays to avoid rate limiting during tests
    results = []
    for i in range(5):
        results.append(test_function(i))
        time.sleep(0.01)
    
    # Verify results
    assert results == [0, 2, 4, 6, 8]
```

### Testing Circuit Breakers

```python
def test_circuit_breaker():
    # Use a unique circuit name with UUID to avoid conflicts
    circuit_name = f"test_circuit_{uuid.uuid4()}"
    
    try:
        # Create isolated circuit breaker
        cb = CircuitBreaker(circuit_name, failure_threshold=2)
        circuit_breakers[circuit_name] = cb
        
        # Test function that fails
        test_func = MagicMock(side_effect=APIError("test error"))
        
        # First failure - circuit stays closed
        with pytest.raises(APIError):
            cb.execute(test_func)
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 1
        
        # Second failure - circuit opens
        with pytest.raises(APIError):
            cb.execute(test_func)
            
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2
        
    finally:
        # Clean up to prevent affecting other tests
        if circuit_name in circuit_breakers:
            del circuit_breakers[circuit_name]
```

### Testing Timeout and Probability-Based Components

When testing components that involve timeouts or probabilistic behavior:

```python
def test_with_timeout():
    """Test timeout behavior in circuit breaker."""
    # Use a unique circuit name
    circuit_name = f"test_timeout_{uuid.uuid4()}"
    
    # Create the circuit breaker with a very short timeout
    # Make timeout shorter than the sleep time in the test function
    cb = CircuitBreaker(circuit_name, timeout=0.05)  # 50ms timeout
    
    try:
        # Define a test function that sleeps longer than the timeout
        def slow_func():
            time.sleep(0.2)  # 200ms sleep, exceeds timeout
            return "success"
        
        # Execute the function with the circuit breaker, which should raise
        with pytest.raises(CircuitBreakerError, match="Circuit breaker timeout"):
            cb.execute(slow_func)
    finally:
        # Clean up
        if circuit_name in circuit_breakers:
            del circuit_breakers[circuit_name]
```

For components with probabilistic behavior, use fixed seeds:

```python
def test_probability_based_component():
    """Test component with probabilistic behavior."""
    # Use a fixed seed for reproducible tests
    import random
    random.seed(42)
    
    # Create the component
    component = ProbabilityBasedComponent(probability=0.5)
    
    # Run multiple iterations
    results = [component.execute() for _ in range(100)]
    
    # Assert on a range rather than exact value
    success_count = sum(results)
    assert 40 <= success_count <= 65  # Allow a reasonable range
```

### Testing with Floating Point Values

When testing code that involves floating point calculations, avoid using exact equality:

```python
def test_rate_limiter_delay():
    """Test delay calculation in rate limiter."""
    # Create a rate limiter for testing
    limiter = RateLimiter(base_delay=0.3)
    
    # Get the calculated delay
    delay = limiter.get_delay_for_ticker()
    
    # DON'T use exact equality (this is fragile)
    # assert delay == limiter.base_delay  # BAD!
    
    # GOOD: Use one of these approaches instead:
    
    # Option 1: Allow a delta for floating point comparisons
    assert abs(delay - limiter.base_delay) < 0.2
    
    # Option 2: Use pytest's approx function
    from pytest import approx
    assert delay == approx(limiter.base_delay, abs=0.2)
    
    # Option 3: For unittest.TestCase classes, use assertAlmostEqual
    self.assertAlmostEqual(delay, limiter.base_delay, delta=0.2)
```
# Test Fixtures

This directory contains reusable test fixtures and helpers for testing the etorotrade application.

## Available Fixtures

### Pagination Fixtures (`pagination.py`)

- `create_paginated_data(num_pages, items_per_page)`: Creates mock paginated data responses
- `create_mock_fetcher(pages)`: Creates a mock page fetcher function 
- `create_bulk_fetch_mocks()`: Creates mock objects for bulk fetching tests

### Async Fixtures (`async_fixtures.py`)

- `create_flaky_function(fail_count)`: Creates a mock async function that initially fails then succeeds
- `create_async_processor_mock(error_item)`: Creates a mock async processor function for testing batch processing

### Rate Limiter Fixtures

- `create_test_rate_limiter()`: Creates an isolated rate limiter instance for testing
- `reset_rate_limiter(limiter)`: Resets a rate limiter to a clean state with thread safety

### Circuit Breaker Fixtures

- `create_test_circuit_breaker(name=None)`: Creates an isolated circuit breaker with a unique name
- `save_and_restore_circuit_state(reset=True)`: Context manager that saves and restores circuit breaker global state

## Best Practices for Using Fixtures

1. **Isolation**: Always use isolated fixtures that don't affect global state
   ```python
   # Good
   test_limiter = create_test_rate_limiter()
   
   # Bad - affects global state
   from yahoofinance.utils.network.rate_limiter import global_rate_limiter
   ```

2. **Thread Safety**: Use thread-safe approaches when fixtures access shared resources
   ```python
   with test_limiter.lock:
       test_limiter.delay = 0.1
   ```

3. **Unique Naming**: Use unique identifiers for resources to prevent test interference
   ```python
   import uuid
   circuit_name = f"test_circuit_{uuid.uuid4()}"
   ```

4. **Proper Cleanup**: Always clean up fixture resources, especially when using finally blocks
   ```python
   try:
       # Use fixture
   finally:
       # Clean up
       reset_rate_limiter(test_limiter)
   ```

5. **Context Managers**: Prefer context managers for fixtures that need setup/teardown
   ```python
   with save_and_restore_circuit_state():
       # Test code that uses circuit breakers
   # Global circuit breaker state is automatically restored here
   ```

## Usage Examples

### Pagination Fixtures

```python
def test_pagination():
    # Create mock paginated data with 3 pages, 5 items per page
    pages = create_paginated_data(3, 5)
    
    # Create a mock fetcher that returns this data
    mock_fetcher = create_mock_fetcher(pages)
    
    # Test pagination logic
    result = paginate_all_results(mock_fetcher)
    
    # Verify results
    assert len(result) == 15  # 3 pages * 5 items
```

### Async Fixtures

```python
@pytest.mark.asyncio
async def test_retry_mechanism():
    # Create a function that fails twice then succeeds
    flaky_func = create_flaky_function(fail_count=2)
    
    # Test retry logic
    result = await retry_async_operation(flaky_func, max_retries=3)
    
    # Verification
    assert result == "success"
    assert flaky_func.call_count == 3  # Called 3 times
```

### Rate Limiter Fixtures

```python
def test_rate_limiting():
    # Create an isolated test rate limiter
    test_limiter = create_test_rate_limiter(window_size=10, max_calls=5)
    
    try:
        # Test code using the rate limiter
        # ...
    finally:
        # Clean up
        reset_rate_limiter(test_limiter)
```

### Circuit Breaker Fixtures

```python
def test_circuit_breaker_behavior():
    with save_and_restore_circuit_state():
        # Create test circuit breaker with unique name
        cb = create_test_circuit_breaker()
        
        # Test circuit breaker behavior
        # ...
        
    # Circuit breaker state is automatically restored here
```
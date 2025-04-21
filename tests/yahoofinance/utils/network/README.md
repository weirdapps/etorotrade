# Network Utilities Tests

This directory contains tests for the network utilities, including rate limiting, circuit breakers, and pagination.

## Test Files

- `test_rate_limiter.py`: Tests for synchronous rate limiting 
- `test_circuit_breaker.py`: Tests for the circuit breaker pattern
- `test_async_circuit_breaker.py`: Tests for async circuit breaker implementation
- `test_pagination.py`: Tests for pagination utilities

## Testing Considerations

### Rate Limiter Testing

When testing rate limiters, keep these considerations in mind:

1. **Thread Safety**: Always use locks when accessing shared state in rate limiters
   ```python
   with rate_limiter.lock:
       rate_limiter.call_timestamps = []
       rate_limiter.delay = rate_limiter.base_delay
   ```

2. **Test Isolation**: Create a dedicated rate limiter instance for each test
   ```python
   test_limiter = RateLimiter()  # Don't use the global rate limiter
   ```

3. **Avoid Race Conditions**: Add small delays between calls to prevent race conditions
   ```python
   for i in range(5):
       results.append(test_function(i))
       time.sleep(0.01)  # Small delay to avoid race conditions
   ```

### Circuit Breaker Testing

When testing circuit breakers, consider these practices:

1. **Global State Management**: Circuit breakers use a global registry that must be properly managed
   ```python
   # Save original state
   original_circuits = dict(_circuit_breakers)
   
   # Clear for test
   _circuit_breakers.clear()
   
   # After test
   _circuit_breakers.clear()
   # Restore original
   for name, circuit in original_circuits.items():
       _circuit_breakers[name] = circuit
   ```

2. **Unique Test Instances**: Use UUIDs to create unique circuit names for each test
   ```python
   import uuid
   circuit_name = f"test_circuit_{uuid.uuid4()}"
   ```

3. **Explicit Registration**: Explicitly register circuit breakers in the global registry
   ```python
   cb = CircuitBreaker(circuit_name, timeout=0.1)
   _circuit_breakers[circuit_name] = cb  # Explicit registration
   ```

4. **Thorough Cleanup**: Always clean up in a finally block
   ```python
   finally:
       if circuit_name in _circuit_breakers:
           del _circuit_breakers[circuit_name]
       reset_all_circuits()  # Reset everything to be extra safe
   ```

### Async Testing

For async network utilities:

1. **Use pytest.mark.asyncio**: Mark async tests properly
   ```python
   @pytest.mark.asyncio
   async def test_async_function():
       # Test code
   ```

2. **Await All Coroutines**: Always await coroutines, including cleanup
   ```python
   cleanup_task = provider.close()
   if asyncio.iscoroutine(cleanup_task):
       await cleanup_task
   ```

## Running Tests

Run all network utility tests:
```
pytest tests/yahoofinance/utils/network/
```

Run a specific test file:
```
pytest tests/yahoofinance/utils/network/test_circuit_breaker.py
```

Run a specific test:
```
pytest tests/yahoofinance/utils/network/test_circuit_breaker.py::TestCircuitBreaker::test_with_timeout
```
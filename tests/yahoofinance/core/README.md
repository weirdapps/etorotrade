# Core Module Tests

This directory contains tests for the core module components of the yahoofinance package.

## Test Files

- `test_client.py`: Tests for YFinanceClient and API request handling
- `test_cache.py`: Tests for the caching system
- `test_cache_example.py`: Example tests demonstrating cache usage patterns
- `test_error_handling.py`: Tests for error handling utilities (retry, context enrichment)
- `test_errors.py`: Tests for the error hierarchy and classification
- `test_types.py`: Tests for data type definitions and conversions
- `test_config.py`: Tests for configuration system

## Testing Best Practices

### Error Handling Tests

When testing error handling components:

1. **Test the Error Hierarchy**: Verify error inheritance and type relationships
   ```python
   def test_error_hierarchy():
       # Verify inheritance
       assert issubclass(APIError, YFinanceError)
       assert issubclass(ValidationError, YFinanceError)
       
       # Test basic behavior
       err = APIError("Test message")
       assert str(err) == "Test message"
       assert isinstance(err, YFinanceError)
   ```

2. **Test Error Context Enrichment**: Verify additional context can be added to errors
   ```python
   def test_enrich_error_context():
       # Create error
       error = APIError("API request failed")
       
       # Enrich context
       enriched = enrich_error_context(error, {"ticker": "AAPL", "attempt": 3})
       
       # Verify context was added
       assert enriched.details["ticker"] == "AAPL"
       assert enriched.details["attempt"] == 3
   ```

3. **Test Error Translations**: Verify appropriate error types are generated
   ```python
   def test_translate_error():
       # Test ValueError translation
       val_err = ValueError("Invalid input")
       translated = translate_error(val_err)
       assert isinstance(translated, ValidationError)
       
       # Test KeyError translation
       key_err = KeyError("Missing key")
       translated = translate_error(key_err)
       assert isinstance(translated, DataError)
   ```

4. **Test Retry Logic**: Verify retry behavior with different error types
   ```python
   def test_with_retry_decorator():
       mock_func = MagicMock()
       mock_func.side_effect = [
           ConnectionError("Network error"),
           TimeoutError("Request timed out"),
           "success"
       ]
       
       # Apply retry decorator
       func_with_retry = with_retry(max_retries=3)(mock_func)
       
       # Should eventually succeed
       result = func_with_retry()
       assert result == "success"
       assert mock_func.call_count == 3
   ```

### Cache Testing

When testing the caching system:

1. **Clean Up Test Files**: Always clean up cache files in finally blocks
   ```python
   def test_cache_operations():
       cache = Cache("test_cache")
       try:
           # Test cache operations
           cache.set("key1", "value1")
           assert cache.get("key1") == "value1"
       finally:
           # Clean up
           cache.clear()
           if os.path.exists(cache.cache_dir):
               shutil.rmtree(cache.cache_dir)
   ```

2. **Test Expiration**: Verify cache expiration works correctly
   ```python
   def test_cache_expiration():
       cache = Cache("test_cache", expire_seconds=1)
       try:
           cache.set("key1", "value1")
           assert cache.get("key1") == "value1"
           
           # Wait for expiration
           time.sleep(1.1)
           assert cache.get("key1") is None
       finally:
           cache.clear()
   ```

### Type Testing

Verify types behave correctly, especially for data conversions:

```python
def test_from_dict_method():
    data = {
        "ticker": "AAPL",
        "price": 150.0,
        "volume": 1000000,
        "extras": {"pe_ratio": 25.0}
    }
    
    # Convert dictionary to object
    stock = StockData.from_dict(data)
    
    # Verify fields
    assert stock.ticker == "AAPL"
    assert stock.price == 150.0
    assert stock.volume == 1000000
    assert stock.extras["pe_ratio"] == 25.0
    
    # Test back to dict
    dict_data = stock.to_dict()
    assert dict_data["ticker"] == "AAPL"
    assert dict_data["price"] == 150.0
```

## Running Tests

Run all core tests:
```
pytest tests/yahoofinance/core/
```

Run a specific test file:
```
pytest tests/yahoofinance/core/test_error_handling.py
```

Run a specific test:
```
pytest tests/yahoofinance/core/test_error_handling.py::TestErrorHandling::test_with_retry_decorator
```

Run with coverage:
```
pytest tests/yahoofinance/core/ --cov=yahoofinance.core
```
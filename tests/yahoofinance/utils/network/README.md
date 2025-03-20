# Network Utilities Tests

This directory contains tests for the network utilities:

- `test_rate_limiter.py`: Tests for synchronous rate limiting (moved from /tests/test_rate_limiter.py)
- `test_pagination.py`: Tests for pagination utilities (moved from /tests/test_pagination_utils.py)

## Async Utilities

The `async_utils` subdirectory contains tests for asynchronous networking utilities:

- `test_async_rate_limiter.py`: Tests for AsyncRateLimiter
- `test_batch_processing.py`: Tests for async batch processing functions

## Running Tests

Run all network utility tests:
```
pytest tests/yahoofinance/utils/network/
```

Run only async tests:
```
pytest tests/yahoofinance/utils/network/async_utils/
```
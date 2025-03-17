# Async Yahoo Finance API

This document provides a guide on using the asynchronous API for Yahoo Finance data.

## Overview

The asynchronous API allows for concurrent fetching of financial data, which can significantly improve performance when retrieving data for multiple tickers. It uses Python's asyncio library and provides proper rate limiting to avoid overwhelming the Yahoo Finance servers.

## Key Components

- `AsyncFinanceDataProvider`: The base interface for async data providers
- `AsyncYahooFinanceProvider`: The Yahoo Finance implementation of the async provider
- `async_rate_limited`: Decorator for rate limiting async functions
- `gather_with_rate_limit`: Helper for gathering async tasks with rate limiting
- `process_batch_async`: Helper for processing batches of items asynchronously

## Getting Started

### Basic Usage

```python
import asyncio
from yahoofinance.api import AsyncYahooFinanceProvider

async def get_apple_info():
    provider = AsyncYahooFinanceProvider()
    data = await provider.get_ticker_info("AAPL")
    print(f"Apple Inc. current price: {data['current_price']}")

# Run the async function
asyncio.run(get_apple_info())
```

### Concurrent Requests

```python
import asyncio
from yahoofinance.api import AsyncYahooFinanceProvider

async def get_multiple_tickers():
    provider = AsyncYahooFinanceProvider()
    
    # Fetch data for multiple tickers concurrently
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    results = await provider.batch_get_ticker_info(tickers)
    
    for ticker, data in results.items():
        if data:
            print(f"{ticker}: {data['name']} - {data['current_price']}")
        else:
            print(f"{ticker}: Failed to fetch data")

# Run the async function
asyncio.run(get_multiple_tickers())
```

### Advanced Concurrent Operations

```python
import asyncio
from yahoofinance.api import AsyncYahooFinanceProvider

async def get_detailed_data(ticker):
    provider = AsyncYahooFinanceProvider()
    
    # Fetch different types of data concurrently
    info_task = asyncio.create_task(provider.get_ticker_info(ticker))
    price_task = asyncio.create_task(provider.get_price_data(ticker))
    hist_task = asyncio.create_task(provider.get_historical_data(ticker, period="1mo"))
    
    # Await all tasks
    info = await info_task
    price = await price_task
    hist = await hist_task
    
    return {
        "info": info,
        "price": price,
        "history": hist
    }

# Run the async function
data = asyncio.run(get_detailed_data("AAPL"))
print(f"Company: {data['info']['name']}")
print(f"Current price: {data['price']['current_price']}")
print(f"Data points: {len(data['history'])}")
```

## Rate Limiting

The async API includes built-in rate limiting to prevent overwhelming the Yahoo Finance servers. This is handled automatically by the `AsyncRateLimiter` class and the `async_rate_limited` decorator.

- `max_concurrency`: Controls the maximum number of concurrent requests (default: 4)
- Each API call is properly rate limited based on the global rate limiter configuration

## Error Handling

Errors are handled gracefully with proper retries and fallbacks:

```python
import asyncio
from yahoofinance.api import AsyncYahooFinanceProvider
from yahoofinance.core.errors import YFinanceError

async def safely_get_ticker_info(ticker):
    provider = AsyncYahooFinanceProvider()
    try:
        return await provider.get_ticker_info(ticker)
    except YFinanceError as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# Run the async function
asyncio.run(safely_get_ticker_info("INVALID_TICKER"))
```

## Example

For a complete working example, see `examples/async_api_example.py`.
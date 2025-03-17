# Provider Pattern: Finance Data API

This document explains how to use the provider pattern to access Yahoo Finance data in both synchronous and asynchronous ways.

## Overview

The provider pattern is a design pattern that abstracts the implementation details of data access behind a consistent interface. This allows for:

- Multiple implementations (e.g., Yahoo Finance, another data source)
- Both synchronous and asynchronous variants
- Centralized configuration
- Simpler testing through mocking

## Getting Started

### Sync Usage

The simplest way to use the provider pattern is to get a provider instance and use it directly:

```python
from yahoofinance import get_provider

# Get default provider (synchronous Yahoo Finance implementation)
provider = get_provider()

# Get data for a ticker
info = provider.get_ticker_info("AAPL")
print(f"Company: {info['name']}")
print(f"Current price: ${info['current_price']}")
```

### Async Usage

For asynchronous operations, specify the `async_mode` parameter:

```python
from yahoofinance import get_provider
import asyncio

async def fetch_data():
    # Get async provider
    provider = get_provider(async_mode=True)
    
    # Get data for a ticker
    info = await provider.get_ticker_info("MSFT")
    print(f"Company: {info['name']}")
    print(f"Current price: ${info['current_price']}")
    
    # Batch process multiple tickers efficiently
    batch_results = await provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])
    for ticker, data in batch_results.items():
        if data:  # Data might be None if ticker lookup failed
            print(f"{ticker}: {data['name']} - ${data['current_price']}")

# Run the async function
asyncio.run(fetch_data())
```

## Provider Interface

All providers implement the same interface, ensuring consistent usage:

### Synchronous Interface (FinanceDataProvider)

```python
class FinanceDataProvider:
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker"""
        
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker"""
        
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker"""
        
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker"""
        
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker"""
        
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query"""
```

### Asynchronous Interface (AsyncFinanceDataProvider)

```python
class AsyncFinanceDataProvider:
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker asynchronously"""
        
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker asynchronously"""
        
    async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker asynchronously"""
        
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker asynchronously"""
        
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker asynchronously"""
        
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query asynchronously"""
        
    async def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple symbols in a batch"""
```

## Return Data Formats

All provider methods return structured data with consistent formats:

### get_ticker_info

Returns a dictionary with company information:

```python
{
    'ticker': 'AAPL',
    'name': 'Apple Inc.',
    'sector': 'Technology',
    'market_cap': 2740000000000.0,
    'beta': 1.28,
    'pe_trailing': 27.5,
    'pe_forward': 24.8,
    'dividend_yield': 0.0058,
    'current_price': 173.57,
    'analyst_count': 42,
    'peg_ratio': 2.1,
    'short_float_pct': 0.67,
    'last_earnings': '2023-07-27',
    'previous_earnings': '2023-05-04'
}
```

### get_price_data

Returns price-specific information:

```python
{
    'current_price': 173.57,
    'target_price': 195.24,
    'upside_potential': 12.48,
    'price_change': 1.34,
    'price_change_percentage': 0.78
}
```

### get_historical_data

Returns a pandas DataFrame with historical price data:

```
                  Open        High         Low       Close    Volume
Date                                                                
2023-07-03  193.780000  193.880000  191.759999  192.460000  18246500
2023-07-05  191.570000  192.979999  190.619999  191.330000  20583900
...
```

### get_analyst_ratings

Returns analyst rating information:

```python
{
    'positive_percentage': 85,
    'total_ratings': 42,
    'ratings_type': 'buy_sell_hold',
    'recommendations': {
        'buy': 36,
        'hold': 5,
        'sell': 1
    }
}
```

### search_tickers

Returns matching ticker symbols:

```python
[
    {
        'symbol': 'AAPL',
        'name': 'Apple Inc.',
        'exchange': 'NASDAQ',
        'type': 'EQUITY',
        'score': 1.0
    },
    # ... more results
]
```

## Creating Custom Providers

To create a custom provider, implement the `FinanceDataProvider` or `AsyncFinanceDataProvider` interface:

```python
from yahoofinance.api.providers.base import FinanceDataProvider

class MyCustomProvider(FinanceDataProvider):
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        # Custom implementation...
        return {
            'ticker': ticker,
            'name': 'Custom Data',
            'current_price': 100.0,
            # ... other required fields
        }
    
    # Implement other methods...
```

## Legacy Client vs Provider

The provider pattern is the recommended approach for new code, but legacy code using `YFinanceClient` directly is still supported:

```python
# Legacy approach - using YFinanceClient directly
from yahoofinance import YFinanceClient
client = YFinanceClient()
stock_data = client.get_ticker_info("AAPL")  # Returns StockData object

# New approach - using provider pattern
from yahoofinance import get_provider
provider = get_provider()
stock_data = provider.get_ticker_info("AAPL")  # Returns dictionary
```

### Key Differences

1. **Return Types**: 
   - Client: Returns custom objects like `StockData`
   - Provider: Returns dictionaries and standard Python types

2. **Method Signatures**: 
   - Client: Some methods have different parameters
   - Provider: Consistent interface across all implementations

3. **Async Support**:
   - Client: Only synchronous operations
   - Provider: Both sync and async implementations

## Performance Considerations

The provider pattern includes the same rate limiting and caching as the direct client approach, ensuring optimal performance:

- Adaptive rate limiting to prevent API throttling
- Batch processing for multiple tickers
- Controlled concurrency in async mode
- Proper backoff strategies for errors

For high-volume applications, the async provider offers significant performance advantages through:

- Concurrent API requests
- Batch processing
- Non-blocking I/O

## Best Practices

1. Use the provider pattern for all new code
2. Prefer async mode for processing multiple tickers
3. Use batch methods when available
4. Handle potential None values in batch results
5. Use try/except blocks to handle potential errors

## Complete Example

Here's a complete example showing how to use the provider pattern effectively:

```python
import asyncio
from yahoofinance import get_provider

async def analyze_sector(sector_tickers):
    # Get async provider
    provider = get_provider(async_mode=True)
    
    # Process tickers in batch
    results = await provider.batch_get_ticker_info(sector_tickers)
    
    # Filter and analyze results
    valid_results = {ticker: data for ticker, data in results.items() if data is not None}
    
    # Calculate sector averages
    pe_ratios = [data['pe_forward'] for ticker, data in valid_results.items() 
                 if data.get('pe_forward') is not None and data['pe_forward'] > 0]
    
    avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else None
    
    return {
        'tickers_processed': len(sector_tickers),
        'valid_results': len(valid_results),
        'average_pe': avg_pe
    }

# Example usage
async def main():
    tech_tickers = ["AAPL", "MSFT", "GOOG", "META", "AMZN"]
    finance_tickers = ["JPM", "BAC", "GS", "WFC", "C"]
    
    # Run analyses concurrently
    tech_analysis, finance_analysis = await asyncio.gather(
        analyze_sector(tech_tickers),
        analyze_sector(finance_tickers)
    )
    
    print(f"Tech sector: Average P/E = {tech_analysis['average_pe']:.2f}")
    print(f"Finance sector: Average P/E = {finance_analysis['average_pe']:.2f}")

# Run the async function
asyncio.run(main())
```
# CLAUDE.md - Guide for Coding Agents

## Table of Contents
- [Commands](#commands)
- [Code Style](#code-style)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Data Formats](#data-formats)
- [Trading Criteria](#trading-criteria)
- [Ticker Formats](#ticker-formats)
- [Display Formatting](#display-formatting)
- [Performance Optimizations](#performance-optimizations)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Asynchronous Capabilities](#asynchronous-capabilities)
- [Provider Pattern](#provider-pattern)
- [Performance Tracking](#performance-tracking)
- [Troubleshooting](#troubleshooting)
- [Pending Improvements](#pending-improvements)
- [File Directory Documentation](#file-directory-documentation)

## Commands

### Application Commands
- `python trade.py` - Run main app
  - Select 'P' for Portfolio analysis - Shows analysis of current portfolio holdings
  - Select 'M' for Market analysis - Analyzes all stocks in the market list
  - Select 'E' for eToro Market analysis - Analyzes stocks available on eToro platform
  - Select 'T' for Trade analysis - Provides actionable trading recommendations
    - Select 'B' for Buy opportunities - Shows new stocks to consider buying
    - Select 'S' for Sell candidates - Shows portfolio stocks to consider selling
    - Select 'H' for Hold candidates - Shows stocks with neutral outlook
  - Select 'I' for Manual ticker input - Analyze specific tickers

### Analysis Module Commands
- `python -m yahoofinance.analysis.stock validate` - Validate tickers against Yahoo Finance API
- `python -m yahoofinance.analysis.news` - Show latest news with sentiment analysis
- `python -m yahoofinance.analysis.portfolio` - Show portfolio performance metrics
- `python -m yahoofinance.analysis.metrics` - View economic indicators and metrics
- `python -m yahoofinance.analysis.earnings` - View upcoming earnings dates and surprises
- `python -m yahoofinance.analysis.performance` - View market and portfolio performance tracking
- `python -m yahoofinance.analysis.analyst` - View analyst ratings and recommendations
- `python -m yahoofinance.analysis.insiders` - Analyze insider transactions

### Testing Commands
- `pytest tests/` - Run all tests
- `pytest tests/test_file.py::TestClass::test_method` - Run specific test
- `pytest tests/ --cov=yahoofinance` - Run tests with coverage
- `pytest tests/test_trade.py --cov=trade --cov-report=term-missing` - Run specific module tests with coverage
- `pytest -xvs tests/test_specific.py` - Run verbose, no capture
- `pytest tests/test_utils.py` - Test core utilities and improvements
- `pytest tests/test_errors.py` - Test error handling and hierarchy
- `pytest tests/test_async.py` - Test async utilities and pagination
- `pytest tests/test_rate.py` - Test rate limiting functionality

## Code Style
- **Imports**: Standard library first, third-party packages, then local modules
- **Types**: Use type hints (List, Dict, Optional, Any, Tuple) from typing module
- **Classes**: Use dataclasses for data containers, proper error hierarchy
- **Naming**: snake_case for variables/functions, PascalCase for classes, ALL_CAPS for constants
- **Documentation**: Docstrings with Args/Returns/Raises sections for all functions/classes
- **Error Handling**: Custom exception hierarchy (YFinanceError → APIError, ValidationError, etc.)
- **Rate Limiting**: Always use rate limiting for API calls with adaptive delays
- **Formatting**: Format numbers with proper precision (1-2 decimals), handle None values, display company names in ALL CAPS
- **Thread Safety**: Use proper locks when modifying shared state
- **Asyncio**: Use appropriate async patterns with rate limiting protection

## Configuration
- **config.py**: Central configuration file with application settings
- **Rate Limiting Settings**:
  - `WINDOW_SIZE`: Time window for rate limiting in seconds (default: 60)
  - `MAX_CALLS`: Maximum API calls per window (default: 60)
  - `BASE_DELAY`: Starting delay between calls in seconds (default: 1.0)
  - `MIN_DELAY`: Minimum delay after many successful calls (default: 0.5)
  - `MAX_DELAY`: Maximum delay after errors (default: 30.0)
  - `BATCH_SIZE`: Number of items per batch (default: 15)
  - `BATCH_DELAY`: Delay between batches in seconds (default: 15.0)
  - **Rate Limit Handling**:
    - Automatic detection of rate limits
    - Smart backoff strategies
    - Success/failure tracking
    - Adaptive delay calculation
- **Caching Settings**:
  - `MARKET_CACHE_TTL`: Market data cache timeout (default: 300 seconds / 5 minutes)
  - `NEWS_CACHE_TTL`: News cache timeout (default: 900 seconds / 15 minutes)
  - `EARNINGS_CACHE_TTL`: Earnings cache timeout (default: 3600 seconds / 1 hour)
  - `CACHE_SIZE_LIMIT`: Maximum cache size (default: 1000 items)
- **Trading Criteria**: Trading rules defined in `TRADING_CRITERIA` dictionary

## Project Structure

### High-Level Organization
- `yahoofinance/` - Main package with modular components
- `tests/` - Test files with module-based organization
- `logs/` - Log files
- `assets/` - Application assets (images, etc.)
- `myenv/` - Python virtual environment

### Package Structure
The codebase follows a modern provider-based architecture with five key layers:

```
yahoofinance/               # Main package
├── api/                    # Provider interfaces and implementations
│   └── providers/          # Data provider implementations
│       ├── base_provider.py            # Provider interfaces
│       ├── yahoo_finance.py            # Sync implementation
│       ├── async_yahoo_finance.py      # Async implementation
│       └── enhanced_async_yahoo_finance.py # Enhanced async
├── analysis/               # Analysis modules
│   ├── analyst.py          # Analyst ratings
│   ├── earnings.py         # Earnings data
│   ├── insiders.py         # Insider transactions
│   ├── market.py           # Market analysis
│   ├── metrics.py          # Financial metrics
│   ├── news.py             # News analysis
│   ├── performance.py      # Performance tracking
│   ├── portfolio.py        # Portfolio analysis
│   └── stock.py            # Stock data analysis
├── compat/                 # Compatibility layer
│   ├── client.py           # Legacy client compatibility
│   └── display.py          # Legacy display compatibility
├── core/                   # Core functionality
│   ├── client.py           # Core client implementation
│   ├── config.py           # Configuration settings
│   ├── errors.py           # Error hierarchy
│   ├── logging.py          # Logging configuration
│   └── types.py            # Type definitions
├── data/                   # Data management
│   ├── cache.py            # Caching implementation
│   ├── cache/              # Cache storage
│   └── download.py         # Data download functionality
├── presentation/           # Presentation components
│   ├── console.py          # Console output
│   ├── formatter.py        # Data formatting
│   ├── html.py             # HTML generation
│   └── templates.py        # HTML templates
├── utils/                  # Utility modules
│   ├── async/              # Basic async utilities
│   ├── async_utils/        # Enhanced async utilities
│   │   ├── enhanced.py     # Enhanced async features
│   │   └── helpers.py      # Async helpers
│   ├── data/               # Data formatting utilities
│   │   ├── format_utils.py            # Formatting utilities
│   │   └── market_cap_formatter.py    # Market cap formatting
│   ├── date/               # Date utilities
│   │   └── date_utils.py   # Date handling
│   ├── market/             # Market-specific utilities
│   │   ├── filter_utils.py # Market filtering
│   │   └── ticker_utils.py # Ticker validation
│   └── network/            # Network utilities
│       ├── batch.py        # Batch processing
│       ├── circuit_breaker.py # Circuit breaker pattern
│       ├── pagination.py   # Paginated response handling
│       └── rate_limiter.py # Rate limiting
├── input/                  # Input data files
└── output/                 # Generated output files
```

### Key Components

1. **Provider Layer**: 
   - `api/providers/base_provider.py` - Defines the interfaces `FinanceDataProvider` and `AsyncFinanceDataProvider`
   - `api/providers/yahoo_finance.py` - Synchronous Yahoo Finance provider
   - `api/providers/async_yahoo_finance.py` - Asynchronous Yahoo Finance provider
   - `api/providers/enhanced_async_yahoo_finance.py` - Enhanced asynchronous provider with batch operations

2. **Core Layer**:
   - `core/errors.py` - Comprehensive error hierarchy
   - `core/config.py` - Centralized configuration
   - `core/client.py` - Core API client implementation
   - `core/types.py` - Type definitions and data classes
   - `core/logging.py` - Logging configuration

3. **Utility Modules**:
   - Network utilities - Rate limiting, pagination, circuit breaker pattern
   - Data formatting utilities - Consistent presentation of financial data
   - Market utilities - Ticker validation and normalization
   - Date utilities - Date handling and formatting
   - Async utilities - Asynchronous operations with rate limiting and concurrency control

4. **Analysis Layer**:
   - Finance-specific analysis modules
   - Stock data analysis and validation
   - Performance tracking for portfolios and indices
   - News sentiment analysis

5. **Presentation Layer**:
   - Console output formatting
   - HTML dashboard generation
   - Data visualization components

### Compatibility Layer

The codebase includes a compatibility layer that ensures backward compatibility:

- `compat/client.py` - Legacy client interface
- `compat/display.py` - Legacy display interface
- Top-level utility modules serve as compatibility layers that re-export functionality from their specialized submodule implementations

### Test Organization
- `tests/` - Test files with module-based organization
  - `tests/test_utils.py` - Tests for core utility functions and market utilities
  - `tests/test_rate.py` - Tests for rate limiting functionality
  - `tests/test_async.py` - Tests for async helpers and pagination
  - `tests/test_errors.py` - Tests for error hierarchy and handling
  - `tests/test_market_display.py` - Tests for market display functionality
  - `tests/test_market_display_batch.py` - Tests for batch processing
  - `tests/test_market_display_html.py` - Tests for HTML output
  - Other module-specific test files (test_client.py, test_display.py, etc.)
  - `tests/fixtures/` - Reusable test fixtures for complex objects
  - `tests/utils/test_fixtures.py` - Common test utilities

## Data Formats

### Data Directories
- `yahoofinance/input/` - Input data files (.csv)
  - `market.csv` - All market tickers for analysis
  - `etoro.csv` - Filtered list of tickers available on eToro
  - `portfolio.csv` - Current portfolio holdings
  - `yfinance.csv` - Valid tickers that pass Yahoo Finance API validation
  - `notrade.csv` - Tickers to exclude from trading recommendations
  - `cons.csv` - Consolidated list of important tickers
  - `china.csv` - China market tickers
  - `europe.csv` - Europe market tickers
  - `usa.csv` - USA market tickers
  - `usindex.csv` - US market indices
- `yahoofinance/output/` - Generated output files
  - `buy.csv` - Generated buy recommendations
  - `sell.csv` - Generated sell recommendations
  - `hold.csv` - Generated hold recommendations
  - `market.csv` - Analysis results from market or eToro tickers
  - `portfolio.csv` - Analysis results from portfolio
  - `index.html` - Main HTML dashboard
  - `portfolio_dashboard.html` - Portfolio HTML dashboard
  - `manual.csv` - Results from manual ticker input
  - `monthly_performance.json` - Monthly performance data
  - `portfolio_performance.json` - Portfolio performance data
  - `weekly_performance.json` - Weekly performance data
  - `script.js` - Dashboard JavaScript
  - `styles.css` - Dashboard styles

### Input File Formats

#### portfolio.csv
```
symbol,shares,cost,date
AAPL,10,150.25,2022-03-15
MSFT,5,280.75,2022-04-20
...
```

#### market.csv
```
symbol,sector
AAPL,Technology
MSFT,Technology
...
```

#### etoro.csv
```
symbol,name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
...
```

#### notrade.csv
```
symbol
AAPL
MSFT
...
```

## Trading Criteria

- **INCONCLUSIVE**:
  - Less than 5 price targets OR
  - Less than 5 analyst ratings

For stocks that pass the confidence threshold (5+ price targets and 5+ analyst ratings):

- **SELL** (Checked first for risk management, triggered if ANY of these conditions are met):
  - Less than 5% upside OR
  - Less than 65% buy ratings OR
  - PEF > PET (deteriorating earnings outlook, when both are positive) OR
  - PEF > 45.0 (extremely high valuation) OR
  - PEG > 3.0 (overvalued relative to growth) OR
  - SI > 4% (high short interest) OR
  - Beta > 3.0 (excessive volatility) OR
  - EXRET < 10.0 (insufficient expected return)

- **BUY** (ALL of these conditions must be met):
  - 20% or more upside AND
  - 82% or more buy ratings AND
  - Beta <= 3.0 (acceptable volatility) AND
  - Beta > 0.2 (sufficient volatility) AND
  - PEF < PET (improving earnings outlook) OR Trailing P/E ≤ 0 (negative) AND
  - PEF > 0.5 (positive earnings projection) AND
  - PEF <= 45.0 (reasonable valuation) AND
  - PEG < 3.0 (reasonable valuation relative to growth) - *ignored if PEG data not available* AND
  - SI <= 3% (acceptable short interest) - *ignored if SI data not available*

- **HOLD**:
  - Stocks that pass confidence threshold
  - Don't meet SELL criteria
  - Don't meet BUY criteria

- **EXRET Calculation**:
  - Expected Return = Upside Potential × Buy Percentage / 100

**Important Note**: These exact same criteria are used consistently in both:
1. Determining the coloring in market and portfolio views (`trade m`, `trade p`, `trade e`)
2. Filtering stocks into the buy/sell/hold lists (`trade t b`, `trade t s`, `trade t h`)

The system ensures perfect alignment between the color a stock receives in the main views and which list it appears in with the trade command. Green-colored stocks appear in buy lists, red-colored stocks appear in sell lists, and white/neutral-colored stocks appear in hold lists.

## Ticker Formats

- **US Ticker Detection**:
  - US tickers have no suffix or end with .US
  - Special cases: BRK.A, BRK.B, BF.A, BF.B are US stocks with dots

- **Hong Kong (HK) Stocks**:
  - Program automatically fixes eToro HK ticker formats
  - Leading zeros are removed from tickers with 5+ digits
  - Example: `03690.HK` → `3690.HK`
  - 4-digit tickers remain unchanged (e.g., `0700.HK`)

- **Crypto Tickers**:
  - Standardized to the `-USD` format (e.g., `BTC-USD`, `ETH-USD`)

- **Ticker Length Validation**:
  - Standard tickers: up to 10 characters
  - Exchange-specific tickers: up to 20 characters (allows for longer exchange formats)
  - Handles special formats like 'MAERSK-A.CO'

## Display Formatting

### Text Formatting

- **Company Names**: 
  - Always displayed in ALL CAPS for readability
  - Truncated to maximum 14 characters if needed
  - Left-aligned in all table outputs

- **Market Cap Formatting**:
  - Trillion-scale values use "T" suffix (e.g., "2.75T")
    - ≥ 10T: 1 decimal place (e.g., "12.5T")
    - < 10T: 2 decimal places (e.g., "2.75T")
  - Billion-scale values use "B" suffix (e.g., "175B")
    - ≥ 100B: No decimals (e.g., "175B")
    - ≥ 10B and < 100B: 1 decimal place (e.g., "25.5B")
    - < 10B: 2 decimal places (e.g., "5.25B")
  - Right-aligned in all table outputs
  
- **Percentage Formatting**:
  - Upside, EXRET, SI use 1 decimal place with % suffix (e.g., "27.9%")
  - Buy percentage uses 0 decimal places with % suffix (e.g., "85%")
  - Dividend yield uses 2 decimal places with % suffix (e.g., "0.84%")
  
- **Other Number Formatting**:
  - Price, target price, beta, PET, PEF, PEG use 1 decimal place
  - Rankings (# column) included in all views for consistent display

### HTML Dashboard Generation
- **Templates**: Defined in `templates.py`
- **Components**:
  - Base HTML structure with responsive design
  - CSS styles for metrics and data visualization
  - JavaScript for dynamic content
- **Dashboard Types**:
  - Market dashboard (index.html)
  - Portfolio dashboard (portfolio.html)
- **Update Process**:
  - Data retrieved through API or from CSVs
  - Formatted using FormatUtils for consistent presentation
  - Inserted into templates
  - Written to output directory

## Performance Optimizations

- **Ticker Validation**:
  - Validates tickers against Yahoo Finance API
  - Filters out invalid or delisted tickers 
  - Saves valid tickers to yfinance.csv
  - Improves batch processing reliability
  - Reduces API errors and failed requests
  
- **API Optimization for Non-US Markets**:
  - Skips analyst ratings API calls for non-US markets
  - Skips insider transaction API calls for non-US markets
  - Skips short interest API calls for non-US markets
  - Falls back to alternative data sources for non-US tickers
  
- **Rate Limiting Optimizations**:
  - Thread-safe API call tracking
  - Adaptive delays based on API response patterns
  - Individual ticker tracking for problematic symbols
  - Exponential backoff for rate limit errors
  - Batch processing with controlled concurrency
  
- **Caching Improvements**:
  - LRU (Least Recently Used) eviction policy
  - Size-limited cache to prevent unbounded growth
  - Different TTL values for different data types
  - Automatic cache cleanup and maintenance
  
- **Pagination & Bulk Processing**:
  - Efficient handling of large result sets
  - Rate-limiting aware pagination
  - Memory-efficient result processing

## Error Handling
- **Centralized Error System**:
  - Comprehensive exception hierarchy
  - Detailed error reporting with context
  - Structured error classification
  - Contextual error recovery
  - API-specific error handling
  
- **Error Hierarchy**:
  - `YFinanceError` - Base class for all errors
  - `ValidationError` - Input validation issues
  - `APIError` - General API communication errors
  - `RateLimitError` - Rate limiting detected
  - `NetworkError` - Network connectivity issues
  - `DataError` - Problems with data quality or availability
  - `ConfigError` - Configuration-related errors

## Testing

### Testing Approach
- **Unit Testing**: Tests individual components in isolation
- **Mock-Based Testing**: Uses mock objects to isolate components
- **Test Coverage**: Aims for high coverage of critical components
- **Component Testing**: Tests interactions between related components
- **Test Fixtures**: Reusable test data and setup utilities
- **Pytest Patterns**: Uses pytest fixtures and parameterization

## Asynchronous Capabilities
- **Safe Async Operations**:
  - Rate limiting for async functions
  - Controlled concurrency
  - Exponential backoff for failures
  - Resource-efficient batch processing
- **AsyncRateLimiter**: Thread-safe rate limiting for async operations
- **async_rate_limited Decorator**: Easy application to async functions
- **Semaphore Usage**: Controls concurrent API access

## Troubleshooting

### Common Issues
- **Rate Limiting Errors**:
  - The system implements automatic backoff
  - Check if you're using rate_limited decorator or AdaptiveRateLimiter
  - Increase BASE_DELAY if experiencing frequent errors
  
- **Invalid Ticker Errors**:
  - Run `python -m yahoofinance.validate` to update valid tickers
  - Check ticker formats for exchange-specific requirements
  
- **Missing Data for Non-US Stocks**:
  - Expected behavior - some data isn't available for international markets
  - System will skip those API calls automatically
  
- **Input File Format Errors**:
  - Verify CSV headers match expected format
  - Check for BOM markers or encoding issues in input files

## Code Duplication Cleanup
The codebase has been reorganized to eliminate duplications:

- **Format Utilities**:
  - Consolidated duplicate code between `format_utils.py` and `utils/data/format_utils.py`
  - Maintained backward compatibility via imports
  
- **Rate Limiting**:
  - Unified implementations between `rate_limiter.py` and `utils/network/rate_limiter.py`
  - Ensured proper patching for tests
  
- **Pagination**:
  - Merged duplicated functionality between `pagination.py` and `utils/network/pagination.py`
  - Standardized interfaces for consistent usage
  
- **Market Utilities**:
  - Combined functionality from `market_utils.py` and `utils/market/ticker_utils.py`
  - Preserved API compatibility

## Completed Improvements

- **Testing Consolidation**: 
  - Combined rate limiter tests (`test_rate_limiter.py`, `test_rate_limiter_unified.py`, `test_rate_limiter_advanced.py`) into a single comprehensive test file
  - Consolidated market display tests (`test_market_display.py`, `test_market_display_batch.py`, `test_market_display_html.py`, `test_market_display_unified.py`) into a single test file

- **API Architecture**:
  - Completed the `api/providers/` directory implementations
  - Added proper provider interfaces and implementations
  - Created a centralized `get_provider()` function for easy access
  - Implemented missing client methods to complete provider API compatibility

- **Code Organization**:
  - Fixed compatibility layers to properly re-export functionality
  - Updated imports to use `core.*` modules
  - Fixed import paths in `network/pagination.py` and other modules
  - Ensured proper adherence to module boundaries
  - Normalized error imports from core.errors instead of types.py

- **Async Utilities**:
  - Properly structured async utilities following the same pattern as other utilities
  - Created clear module boundaries and responsibilities
  - Consolidated duplicate async utilities into a single implementation
  - Updated imports to use core.errors instead of relative imports

- **Error Handling**:
  - Consolidated error type definitions to core.errors module
  - Ensured client code uses proper error imports
  - Added documentation to clarify import patterns

## Recently Completed Improvements

- **Full Provider Integration**:
  - Added provider pattern as the main access method for finance data
  - Updated MarketDisplay class to support both provider and direct client instantiation
  - Modified importing code to use proper imports for core modules
  - Added detailed examples in documentation

- **Testing Enhancement**:
  - Created integration tests for provider pattern in test_api_integration.py
  - Added dedicated test file for async provider (test_async_api.py)
  - Added compatibility tests to ensure consistent behavior between client and provider

- **Documentation Updates**:
  - Created comprehensive async_api.md documentation
  - Included detailed examples for both sync and async usage
  - Added section on creating custom providers
  - Provided complete reference for all provider methods and return formats
  
## Provider Pattern

The provider pattern is a design pattern that abstracts the implementation details of data access behind a consistent interface, allowing for:

- Multiple implementations (e.g., Yahoo Finance, another data source)
- Both synchronous and asynchronous variants
- Centralized configuration
- Simpler testing through mocking

### Getting Started

#### Sync Usage

```python
from yahoofinance import get_provider

# Get default provider (synchronous Yahoo Finance implementation)
provider = get_provider()

# Get data for a ticker
info = provider.get_ticker_info("AAPL")
print(f"Company: {info['name']}")
print(f"Current price: ${info['current_price']}")
```

#### Async Usage

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

### Provider Interfaces

All providers implement the same interface, ensuring consistent usage:

#### Synchronous Interface (FinanceDataProvider)

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

#### Asynchronous Interface (AsyncFinanceDataProvider)

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

### Return Data Formats

All provider methods return structured data with consistent formats:

#### get_ticker_info

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

#### get_price_data

```python
{
    'current_price': 173.57,
    'target_price': 195.24,
    'upside_potential': 12.48,
    'price_change': 1.34,
    'price_change_percentage': 0.78
}
```

#### get_historical_data

Returns a pandas DataFrame with historical price data:

```
              Open        High         Low       Close    Volume
Date                                                                
2023-07-03  193.780000  193.880000  191.759999  192.460000  18246500
2023-07-05  191.570000  192.979999  190.619999  191.330000  20583900
...
```

#### get_analyst_ratings

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

### Creating Custom Providers

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

### Legacy Client vs Provider

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

Key Differences:

1. **Return Types**: 
   - Client: Returns custom objects like `StockData`
   - Provider: Returns dictionaries and standard Python types

2. **Method Signatures**: 
   - Client: Some methods have different parameters
   - Provider: Consistent interface across all implementations

3. **Async Support**:
   - Client: Only synchronous operations
   - Provider: Both sync and async implementations

### Best Practices

1. Use the provider pattern for all new code
2. Import utilities directly from their canonical sources
3. Prefer async mode for processing multiple tickers
4. Use batch methods when available
5. Handle potential None values in batch results
6. Use try/except blocks to handle potential errors

### Complete Example

```python
import asyncio
from yahoofinance import get_provider
from yahoofinance.utils.network.async_utils.rate_limiter import gather_with_rate_limit

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
    tech_analysis, finance_analysis = await gather_with_rate_limit([
        analyze_sector(tech_tickers),
        analyze_sector(finance_tickers)
    ])
    
    print(f"Tech sector: Average P/E = {tech_analysis['average_pe']:.2f}")
    print(f"Finance sector: Average P/E = {finance_analysis['average_pe']:.2f}")

# Run the async function
asyncio.run(main())
```

## Performance Tracking

The performance tracking module provides functionality for tracking market index performance and portfolio performance metrics from external sources with these key features:

1. **Market Index Performance Tracking**
   - Tracks weekly and monthly performance of major market indices
   - Generates HTML dashboards with performance metrics
   - Provides both synchronous and asynchronous implementations

2. **Portfolio Performance Web Scraping**
   - Scrapes portfolio performance metrics from external web sources
   - Extracts key metrics like MTD/YTD performance, beta, Sharpe ratio, etc.
   - Generates HTML dashboards and saves performance data
   - Includes circuit breaker pattern for resilient web scraping

### Usage Examples

#### Tracking Market Index Performance

```python
from yahoofinance.analysis.performance import track_index_performance

# Track weekly performance
track_index_performance(period_type="weekly")

# Track monthly performance
track_index_performance(period_type="monthly")
```

#### Tracking Portfolio Performance

```python
from yahoofinance.analysis.performance import track_portfolio_performance

# Use default URL
track_portfolio_performance()

# Specify a custom URL
track_portfolio_performance(url="https://your-portfolio-url.com")
```

#### Asynchronous Tracking (Both Index and Portfolio)

```python
import asyncio
from yahoofinance.analysis.performance import track_performance_async

# Track both weekly index performance and portfolio performance
asyncio.run(track_performance_async(
    period_type="weekly",
    portfolio_url="https://your-portfolio-url.com"
))
```

### Advanced Usage with PerformanceTracker Class

For more control over the tracking process, you can use the `PerformanceTracker` class directly:

```python
from yahoofinance.analysis.performance import PerformanceTracker
from yahoofinance.api import get_provider

# Synchronous usage
tracker = PerformanceTracker(
    provider=get_provider(),      # Optional: specify a provider
    output_dir="custom/output"    # Optional: specify output directory
)

# Get index performance
performances = tracker.get_index_performance(period_type="weekly")

# Get portfolio performance
portfolio_perf = tracker.get_portfolio_performance_web(url="https://your-portfolio-url.com")

# Generate HTML dashboards
tracker.generate_index_performance_html(performances, title="Weekly Market Performance")
tracker.generate_portfolio_performance_html(portfolio_perf)

# Save data to JSON files
tracker.save_performance_data(performances, file_name="weekly_performance.json")
tracker.save_performance_data(portfolio_perf, file_name="portfolio_performance.json")
```

### Data Classes

The module provides two data classes for representing performance data:

1. `IndexPerformance` - Represents performance metrics for a market index
2. `PortfolioPerformance` - Represents portfolio performance metrics

These classes provide convenient access to performance metrics and can be easily converted to dictionaries for HTML generation or JSON serialization.

### Circuit Breaker Pattern

The web scraping functionality includes the circuit breaker pattern for resilience:

- Automatically detects and handles web scraping failures
- Prevents excessive retries when the target site is unavailable
- Gradually recovers when the target site becomes available again

## Future Opportunities
- **Extend Provider Pattern**: Create additional provider implementations for other data sources
- **Full Migration**: Gradually migrate all direct YFinanceClient usage to provider pattern
- **Performance Optimization**: Enhance batch processing capabilities for async providers

## File Directory Documentation

### Root Directory

- **trade.py**: Main entry point for the application. Provides menu options for Portfolio analysis, Market analysis, eToro Market analysis, and Trade analysis.
- **README.md**: Project overview and documentation.
- **LICENSE**: License information for the project.
- **CLAUDE.md**: Guide for coding agents with commands, code style guidelines, and project organization.
- **requirements.txt**: Python package dependencies for the project.

### yahoofinance/ (Main Package)

#### Provider Layer (yahoofinance/api/)

- **api/__init__.py**: Exports provider factory function and interfaces.
- **api/providers/base_provider.py**: Defines `FinanceDataProvider` and `AsyncFinanceDataProvider` interfaces.
- **api/providers/yahoo_finance.py**: Synchronous Yahoo Finance provider implementation.
- **api/providers/async_yahoo_finance.py**: Asynchronous Yahoo Finance provider implementation.
- **api/providers/enhanced_async_yahoo_finance.py**: Enhanced async provider with batch operations.

#### Analysis Layer (yahoofinance/analysis/)

- **analysis/__init__.py**: Exports analysis module functionality.
- **analysis/analyst.py**: Analyst ratings and recommendations.
- **analysis/earnings.py**: Earnings dates and surprises.
- **analysis/insiders.py**: Insider transactions analysis.
- **analysis/market.py**: Market analysis functionality.
- **analysis/metrics.py**: Financial metrics calculations.
- **analysis/news.py**: News with sentiment analysis.
- **analysis/performance.py**: Performance tracking for indices and portfolios.
- **analysis/portfolio.py**: Portfolio analysis and management.
- **analysis/stock.py**: Stock data analysis and validation.

#### Core Layer (yahoofinance/core/)

- **core/__init__.py**: Core package initialization.
- **core/client.py**: Core client implementation.
- **core/config.py**: Configuration settings and constants.
- **core/errors.py**: Centralized error hierarchy.
- **core/logging.py**: Logging configuration.
- **core/types.py**: Common type definitions.

#### Data Management (yahoofinance/data/)

- **data/__init__.py**: Data package initialization.
- **data/cache.py**: Cache implementation with size limits.
- **data/download.py**: Data download functionality.

#### Presentation Layer (yahoofinance/presentation/)

- **presentation/__init__.py**: Presentation package initialization.
- **presentation/console.py**: Console output formatting.
- **presentation/formatter.py**: Data formatting utilities.
- **presentation/html.py**: HTML generation.
- **presentation/templates.py**: HTML dashboard templates.

#### Compatibility Layer (yahoofinance/compat/)

- **compat/__init__.py**: Compatibility package initialization.
- **compat/client.py**: Legacy client interface.
- **compat/display.py**: Legacy display interface.

#### Utility Modules (yahoofinance/utils/)

- **utils/__init__.py**: Utilities package initialization.
- **utils/async/**: Basic async utilities (compatibility layer).
- **utils/async_utils/**: Enhanced async utilities.
  - **async_utils/enhanced.py**: Enhanced async functionality.
  - **async_utils/helpers.py**: Async helper functions.
- **utils/data/**: Data formatting utilities.
  - **data/format_utils.py**: Core formatting implementation.
  - **data/market_cap_formatter.py**: Market cap formatting.
- **utils/date/**: Date utilities.
  - **date/date_utils.py**: Date formatting and processing.
- **utils/market/**: Market-specific utilities.
  - **market/filter_utils.py**: Market filtering utilities.
  - **market/ticker_utils.py**: Ticker validation and normalization.
- **utils/network/**: Network utilities.
  - **network/batch.py**: Batch processing.
  - **network/circuit_breaker.py**: Circuit breaker pattern.
  - **network/pagination.py**: Paginated response handling.
  - **network/rate_limiter.py**: Thread-safe rate limiting.

### tests/ (Test Directory)

- **__init__.py**: Test package initialization.
- **conftest.py**: Pytest configuration and shared fixtures.

#### Test Organization

The test directory follows a modular structure matching the main package:

```
tests/
├── e2e/                    # End-to-end tests
│   └── test_trade_workflows.py
├── integration/            # Integration tests
│   ├── api/                # API integration tests
│   │   ├── test_api_integration.py
│   │   ├── test_async_api.py
│   │   └── test_circuit_breaker_integration.py
├── trade/                  # Trade module tests
│   └── test_trade.py
├── unit/                   # Unit tests
│   ├── api/                # API unit tests
│   │   └── providers/
│   │       └── test_enhanced_async_provider.py
│   ├── core/               # Core unit tests
│   └── utils/              # Utils unit tests
│       └── async/
│           └── test_enhanced.py
│       └── network/
│           ├── test_async_circuit_breaker.py
│           └── test_circuit_breaker.py
└── yahoofinance/           # Package tests
    ├── analysis/           # Analysis module tests
    │   ├── test_analyst.py
    │   ├── test_earnings.py
    │   ├── test_insiders.py
    │   └── ...
    ├── core/               # Core module tests
    │   ├── test_cache.py
    │   ├── test_client.py
    │   ├── test_errors.py
    │   └── test_types.py
    └── utils/              # Utils tests
        ├── async/
        │   ├── test_async.py
        │   └── test_async_helpers.py
        ├── data/
        │   ├── test_format_utils.py
        │   └── test_formatting.py
        └── network/
            ├── test_pagination.py
            ├── test_rate_limiter.py
            └── test_rate.py
```

#### Key Test Files

- **e2e/test_trade_workflows.py**: Tests for end-to-end trading workflows.
- **integration/api/test_api_integration.py**: Tests for API integration.
- **integration/api/test_async_api.py**: Tests for async API functionality.
- **trade/test_trade.py**: Tests for main trade module functionality.
- **unit/api/providers/test_enhanced_async_provider.py**: Tests for enhanced async provider.
- **unit/utils/network/test_circuit_breaker.py**: Tests for circuit breaker pattern.
- **yahoofinance/analysis/test_analyst.py**: Tests for analyst module.
- **yahoofinance/core/test_cache.py**: Tests for cache functionality.
- **yahoofinance/utils/async/test_async.py**: Tests for async utilities.

#### test/fixtures/ (Test Fixtures)

- **__init__.py**: Fixtures package initialization.
- **README.md**: Documentation for test fixtures.
- **async_fixtures.py**: Fixtures for async testing.
- **api_responses/api_errors.py**: Mock API error responses.
- **market_data/stock_data.py**: Sample stock data for testing.
- **pagination.py**: Fixtures for pagination testing.
- **rate_limiter_fixtures.py**: Rate limiter testing utilities.
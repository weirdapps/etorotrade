# CLAUDE.md - Guide for Coding Agents

## Commands
- `python trade.py` - Run main app
  - Select 'P' for Portfolio analysis
  - Select 'M' for Market analysis
  - Select 'E' for eToro Market analysis (filtered tickers available on eToro)
  - Select 'T' for Trade analysis
    - Select 'B' for Buy opportunities
    - Select 'S' for Sell candidates
  - Select 'I' for Manual ticker input
- `python -m yahoofinance.validate` - Validate tickers against Yahoo Finance API
- `python -m yahoofinance.cons` - Generate market constituents (filtered by yfinance.csv when available)
- `pytest tests/` - Run all tests
- `pytest tests/test_file.py::TestClass::test_method` - Run specific test
- `pytest tests/ --cov=yahoofinance` - Run tests with coverage
- `pytest tests/test_cons.py tests/test_trade.py --cov=yahoofinance.cons --cov=trade --cov-report=term-missing` - Run specific module tests with coverage
- `pytest -xvs tests/test_specific.py` - Run verbose, no capture
- `python -m yahoofinance.module_name` - Run specific module (news, portfolio, econ)
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
- **Formatting**: Format numbers with proper precision (1-2 decimals), handle None values
- **Thread Safety**: Use proper locks when modifying shared state
- **Asyncio**: Use appropriate async patterns with rate limiting protection

## Project Organization
- `yahoofinance/` - Main package with modular components
- `yahoofinance/client.py` - API client with rate limiting and caching
- `yahoofinance/display.py` - Output handling and batch processing
- `yahoofinance/formatting.py` - Data formatting and colorization
- `yahoofinance/cons.py` - Market constituents management
- `yahoofinance/validate.py` - Ticker validation against Yahoo Finance API
- `yahoofinance/errors.py` - Centralized error handling
- `yahoofinance/types.py` - Common types and data structures
- `yahoofinance/cache.py` - LRU caching system with size limits

### Utility Modules
- `yahoofinance/utils/` - Utility modules for core functionality
- `yahoofinance/utils/market_utils.py` - Ticker validation and normalization
- `yahoofinance/utils/rate_limiter.py` - Thread-safe adaptive rate limiting
- `yahoofinance/utils/pagination.py` - Paginated API result handling
- `yahoofinance/utils/async_helpers.py` - Async utilities with rate limiting
- `yahoofinance/utils/format_utils.py` - HTML and output formatting utilities

### Test Organization
- `tests/` - Test files with module-based organization
  - `tests/test_utils.py` - Tests for core utility functions and market utilities
  - `tests/test_rate.py` - Tests for rate limiting functionality
  - `tests/test_async.py` - Tests for async helpers and pagination
  - `tests/test_errors.py` - Tests for error hierarchy and handling
  - Other module-specific test files (test_client.py, test_display.py, etc.)

### Data Directories
- `yahoofinance/input/` - Input data files (.csv)
  - `market.csv` - All market tickers for analysis
  - `etoro.csv` - Filtered list of tickers available on eToro
  - `portfolio.csv` - Current portfolio holdings
  - `cons.csv` - Market constituent data (filtered by yfinance.csv when available)
  - `yfinance.csv` - Valid tickers that pass Yahoo Finance API validation
- `yahoofinance/output/` - Generated output files
  - `buy.csv` - Generated buy recommendations
  - `sell.csv` - Generated sell recommendations
  - `market.csv` - Analysis results from market or eToro tickers
  - `portfolio.csv` - Analysis results from portfolio
  - `index.html`, `portfolio.html` - HTML dashboards
  
## Trading Criteria
- **Buy Signal**:
  - More than 5 price targets (# T)
  - More than 5 analyst ratings (# A)
  - More than 20% upside potential
  - More than 85% of analysts recommend buying
  
- **Sell Signal**:
  - More than 5 price targets (# T)
  - More than 5 analyst ratings (# A)
  - AND either:
    - Less than 5% upside potential, OR
    - Less than 55% of analysts recommend buying

- **Hold Signal**:
  - More than 5 price targets (# T)
  - More than 5 analyst ratings (# A)
  - Between 5-20% upside potential
  - Between 55-85% of analysts recommend buying

- **Low Confidence/Insufficient Data**:
  - 5 or fewer price targets OR
  - 5 or fewer analyst ratings
  
- **EXRET Calculation**:
  - Expected Return = Upside Potential × Buy Percentage / 100

## Exchange Ticker Formats
- **Hong Kong (HK) Stocks**:
  - Program automatically fixes eToro HK ticker formats
  - Leading zeros are removed from tickers with 5+ digits
  - Example: `03690.HK` → `3690.HK`
  - 4-digit tickers remain unchanged (e.g., `0700.HK`)

## Performance Optimizations
- **Ticker Validation**:
  - Validates tickers against Yahoo Finance API
  - Filters out invalid or delisted tickers 
  - Saves valid tickers to yfinance.csv
  - Improves batch processing reliability
  - Reduces API errors and failed requests

- **US vs Non-US Market Detection**:
  - Automatically detects US vs non-US tickers based on exchange suffix
  - US tickers have no suffix or end with .US
  - Special cases handled: BRK.A, BRK.B, BF.A, BF.B are US stocks with dots
  
- **API Optimization for Non-US Markets**:
  - Skips analyst ratings API calls for non-US markets
  - Skips insider transaction API calls for non-US markets
  - Skips short interest API calls for non-US markets
  - Falls back to alternative data sources for non-US tickers
  
- **Ticker Length Validation**:
  - Standard tickers: up to 10 characters
  - Exchange-specific tickers: up to 20 characters (allows for longer exchange formats)
  - Handles special formats like 'MAERSK-A.CO'
  
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
  
- **Rate Limit Handling**:
  - Automatic detection of rate limits
  - Smart backoff strategies
  - Success/failure tracking
  - Adaptive delay calculation

## Asynchronous Capabilities
- **Safe Async Operations**:
  - Rate limiting for async functions
  - Controlled concurrency
  - Exponential backoff for failures
  - Resource-efficient batch processing
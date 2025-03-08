# CLAUDE.md - Guide for Coding Agents

## Commands
- `python trade.py` - Run main app
  - Select 'P' for Portfolio analysis - Shows analysis of current portfolio holdings
  - Select 'M' for Market analysis - Analyzes all stocks in the market list
  - Select 'E' for eToro Market analysis - Analyzes stocks available on eToro platform
  - Select 'T' for Trade analysis - Provides actionable trading recommendations
    - Select 'B' for Buy opportunities - Shows new stocks to consider buying
    - Select 'S' for Sell candidates - Shows portfolio stocks to consider selling
    - Select 'H' for Hold candidates - Shows stocks with neutral outlook
  - Select 'I' for Manual ticker input - Analyze specific tickers
- `python -m yahoofinance.validate` - Validate tickers against Yahoo Finance API
- `pytest tests/` - Run all tests
- `pytest tests/test_file.py::TestClass::test_method` - Run specific test
- `pytest tests/ --cov=yahoofinance` - Run tests with coverage
- `pytest tests/test_trade.py --cov=trade --cov-report=term-missing` - Run specific module tests with coverage
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
- **Formatting**: Format numbers with proper precision (1-2 decimals), handle None values, display company names in ALL CAPS
- **Thread Safety**: Use proper locks when modifying shared state
- **Asyncio**: Use appropriate async patterns with rate limiting protection

## Project Organization
- `yahoofinance/` - Main package with modular components
- `yahoofinance/client.py` - API client with rate limiting and caching
- `yahoofinance/display.py` - Output handling and batch processing
- `yahoofinance/formatting.py` - Data formatting, colorization, and display style rules
- `yahoofinance/validate.py` - Ticker validation against Yahoo Finance API
- `yahoofinance/errors.py` - Centralized error handling
- `yahoofinance/types.py` - Common types and data structures
- `yahoofinance/cache.py` - LRU caching system with size limits

### Utility Modules
- `yahoofinance/utils/` - Utility modules for core functionality
- `yahoofinance/utils/market_utils.py` - Ticker validation and normalization (compatibility layer)
- `yahoofinance/utils/rate_limiter.py` - Thread-safe adaptive rate limiting (compatibility layer)
- `yahoofinance/utils/pagination.py` - Paginated API result handling (compatibility layer)
- `yahoofinance/utils/async_helpers.py` - Async utilities with rate limiting
- `yahoofinance/utils/format_utils.py` - HTML and output formatting utilities (compatibility layer)

### Modular Design
The codebase follows a modular design with clear separation of concerns:

- **Core Functionality**: Main modules in the root directory provide high-level features
- **Utility Modules**: Common utilities in the utils/ directory for reusable functionality
- **Specialized Submodules**:
  - `utils/data/` - Data formatting and transformation
    - `format_utils.py` - Core implementation of formatting utilities
  - `utils/network/` - Rate limiting and API communication
    - `rate_limiter.py` - Core implementation of rate limiting functionality
    - `pagination.py` - Core implementation of pagination functionality
  - `utils/market/` - Market-specific utilities like ticker validation
    - `ticker_utils.py` - Core implementation of ticker validation/normalization
  - `utils/date/` - Date manipulation and formatting
    - `date_utils.py` - Date formatting and processing utilities
  - `utils/async/` - Asynchronous operation helpers

The top-level utils files (`rate_limiter.py`, `pagination.py`, etc.) serve as compatibility layers that re-export functionality from their specialized submodule implementations, ensuring backward compatibility while allowing for a more organized code structure.

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
  - `yfinance.csv` - Valid tickers that pass Yahoo Finance API validation
- `yahoofinance/output/` - Generated output files
  - `buy.csv` - Generated buy recommendations
  - `sell.csv` - Generated sell recommendations
  - `hold.csv` - Generated hold recommendations
  - `market.csv` - Analysis results from market or eToro tickers
  - `portfolio.csv` - Analysis results from portfolio
  - `index.html`, `portfolio.html` - HTML dashboards
  
## Trading Criteria

- **INCONCLUSIVE**:
  - Less than 5 price targets OR
  - Less than 5 analyst ratings

For stocks that pass the confidence threshold (5+ price targets and 5+ analyst ratings):

- **SELL** (Checked first for risk management, triggered if ANY of these conditions are met):
  - Less than 5% upside OR
  - Less than 65% buy ratings OR
  - PEF > PET (deteriorating earnings outlook, when both are positive) OR
  - PEG > 3.0 (overvalued relative to growth) OR
  - SI > 5% (high short interest) OR
  - Beta > 3.0 (excessive volatility)

- **BUY** (ALL of these conditions must be met):
  - 20% or more upside AND
  - 82% or more buy ratings AND
  - Beta <= 3.0 (acceptable volatility) AND
  - Beta > 0.2 (sufficient volatility) AND
  - PEF < PET (improving earnings outlook) OR Trailing P/E ≤ 0 (negative) AND
  - PEF > 0.5 (positive earnings projection) AND
  - PEG < 3.0 (reasonable valuation relative to growth) - *ignored if PEG data not available* AND
  - SI <= 5% (acceptable short interest) - *ignored if SI data not available*

- **HOLD**:
  - Stocks that pass confidence threshold
  - Don't meet SELL criteria
  - Don't meet BUY criteria

- **EXRET Calculation**:
  - Expected Return = Upside Potential × Buy Percentage / 100

## Exchange Ticker Formats
- **Hong Kong (HK) Stocks**:
  - Program automatically fixes eToro HK ticker formats
  - Leading zeros are removed from tickers with 5+ digits
  - Example: `03690.HK` → `3690.HK`
  - 4-digit tickers remain unchanged (e.g., `0700.HK`)

## Display Formatting
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
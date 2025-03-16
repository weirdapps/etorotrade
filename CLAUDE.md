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
- `python -m yahoofinance.validate` - Validate tickers against Yahoo Finance API
- `python -m yahoofinance.news` - Show latest news with sentiment analysis
- `python -m yahoofinance.portfolio` - Show portfolio performance metrics
- `python -m yahoofinance.econ` - View economic indicators from FRED API
- `python -m yahoofinance.earnings` - View upcoming earnings dates and surprises
- `python -m yahoofinance.index` - View market index performance (weekly/monthly)
- `python -m yahoofinance.monthly` - View monthly market index performance
- `python -m yahoofinance.weekly` - View weekly market index performance
- `python -m yahoofinance.holders` - Analyze institutional ownership
- `python -m yahoofinance.insiders` - Analyze insider transactions

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

### Core Modules
- `yahoofinance/client.py` - API client with rate limiting and caching
- `yahoofinance/display.py` - Output handling and batch processing
- `yahoofinance/formatting.py` - Data formatting, colorization, and display style rules
- `yahoofinance/validate.py` - Ticker validation against Yahoo Finance API
- `yahoofinance/errors.py` - Centralized error handling
- `yahoofinance/types.py` - Common types and data structures
- `yahoofinance/cache.py` - LRU caching system with size limits
- `yahoofinance/config.py` - Configuration settings and constants

### Analysis Modules
- `yahoofinance/analyst.py` - Analyst ratings and recommendations
- `yahoofinance/earnings.py` - Earnings dates and surprises
- `yahoofinance/econ.py` - Economic indicators from FRED
- `yahoofinance/holders.py` - Institutional ownership analysis
- `yahoofinance/insiders.py` - Insider transactions analysis
- `yahoofinance/monthly.py` - Monthly market performance
- `yahoofinance/news.py` - News with sentiment analysis
- `yahoofinance/portfolio.py` - Portfolio performance tracking
- `yahoofinance/pricing.py` - Stock price and target analysis
- `yahoofinance/weekly.py` - Weekly market performance
- `yahoofinance/index.py` - Combined market index performance
- `yahoofinance/metrics.py` - Internal metrics calculations

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
  - `utils/async/` - Asynchronous operation helpers (currently missing, planned)

The top-level utils files (`rate_limiter.py`, `pagination.py`, etc.) serve as compatibility layers that re-export functionality from their specialized submodule implementations, ensuring backward compatibility while allowing for a more organized code structure.

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
  - `us_tickers.csv` - US market tickers
- `yahoofinance/output/` - Generated output files
  - `buy.csv` - Generated buy recommendations
  - `sell.csv` - Generated sell recommendations
  - `hold.csv` - Generated hold recommendations
  - `market.csv` - Analysis results from market or eToro tickers
  - `portfolio.csv` - Analysis results from portfolio
  - `index.html`, `portfolio.html` - HTML dashboards
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

## Pending Improvements
- **Async Directory Structure**: Create proper `utils/async/` directory with implementations
- **Testing Consolidation**: Combine similar test files (rate limiter, market display)
- **API Architecture**: Implement or remove empty `api/providers/` directory
- **Trading Directory**: Implement or remove empty `trading/` directory
- **Metrics Naming**: Rename `_metrics.py` to follow standard naming conventions

## File Directory Documentation

### Root Directory

- **trade.py**: Main entry point for the application. Provides menu options for Portfolio analysis, Market analysis, eToro Market analysis, and Trade analysis.
- **README.md**: Project overview and documentation.
- **LICENSE**: License information for the project.
- **CLAUDE.md**: Guide for coding agents with commands, code style guidelines, and project organization.
- **requirements.txt**: Python package dependencies for the project.

### yahoofinance/ (Main Package)

#### Core Files

- **__init__.py**: Package initialization for yahoofinance, exports key functionality.
- **client.py**: API client with rate limiting and caching capabilities for Yahoo Finance data.
- **display.py**: Handles output formatting and batch processing of financial data.
- **formatting.py**: Utility functions for data formatting, colorization, and display style rules.
- **validate.py**: Functions to validate tickers against Yahoo Finance API.
- **errors.py**: Centralized error handling system with custom exception hierarchy.
- **types.py**: Common type definitions and data structures.
- **cache.py**: LRU caching system with size limits for API responses.
- **config.py**: Configuration settings and constants for the application.
- **logging_config.py**: Logging configuration for the application.
- **templates.py**: Templates for HTML dashboard generation.

#### Analysis Modules

- **analyst.py**: Handles analyst ratings and recommendations.
- **earnings.py**: Manages earnings dates and surprises information.
- **econ.py**: Retrieves economic indicators from FRED.
- **holders.py**: Analyzes institutional ownership.
- **insiders.py**: Analyzes insider transactions.
- **monthly.py**: Processes monthly market performance.
- **news.py**: Retrieves news with sentiment analysis.
- **portfolio.py**: Tracks portfolio performance.
- **pricing.py**: Analyzes stock price and target data.
- **weekly.py**: Processes weekly market performance.
- **index.py**: Combined market index performance.
- **download.py**: File download functionality.
- **metrics.py**: Internal metrics calculations.

#### yahoofinance/utils/ (Utility Modules)

- **__init__.py**: Initializes the utils package.
- **market_utils.py**: Compatibility layer for ticker validation and normalization.
- **rate_limiter.py**: Compatibility layer for thread-safe adaptive rate limiting.
- **pagination.py**: Compatibility layer for paginated API result handling.
- **async_helpers.py**: Async utilities with rate limiting.
- **format_utils.py**: Compatibility layer for HTML and output formatting utilities.

##### yahoofinance/utils/async/ (Async Utilities)
- **__init__.py**: Package initialization.
- **async_utils.py**: Core async utility implementations.

##### yahoofinance/utils/data/ (Data Formatting)
- **__init__.py**: Package initialization.
- **format_utils.py**: Core implementation of formatting utilities.

##### yahoofinance/utils/date/ (Date Utilities)
- **__init__.py**: Package initialization.
- **date_utils.py**: Date formatting and processing utilities.

##### yahoofinance/utils/market/ (Market Utilities)
- **__init__.py**: Package initialization.
- **ticker_utils.py**: Core implementation of ticker validation/normalization.

##### yahoofinance/utils/network/ (Network Utilities)
- **__init__.py**: Package initialization.
- **pagination.py**: Core implementation of pagination functionality.
- **rate_limiter.py**: Core implementation of rate limiting functionality.

#### yahoofinance/core/ (Core Functionality)

- **__init__.py**: Core package initialization.
- **cache.py**: Core caching implementation.
- **client.py**: Core client implementation.
- **config.py**: Core configuration.
- **errors.py**: Core error handling.
- **logging.py**: Core logging functionality.
- **types.py**: Core type definitions.

#### yahoofinance/api/ (API Interface)

- **__init__.py**: API package initialization.
- **providers/__init__.py**: Initialization for API providers.

### tests/ (Test Directory)

- **__init__.py**: Test package initialization.
- **conftest.py**: Pytest configuration and shared fixtures.

#### Test Files for Core Functionality

- **test_advanced_utils.py**: Tests for advanced utility functions.
- **test_analyst.py**: Tests for analyst module.
- **test_async.py**: Tests for async utilities and pagination.
- **test_cache.py**: Tests for caching functionality.
- **test_client.py**: Tests for API client.
- **test_compatibility.py**: Tests for compatibility between old and new interfaces.
- **test_display.py**: Tests for display functionality.
- **test_download.py**: Tests for download functionality.
- **test_earnings.py**: Tests for earnings module.
- **test_econ.py**: Tests for economic indicators.
- **test_error_handling.py**: Tests for error handling.
- **test_errors.py**: Tests for error hierarchy.
- **test_format_utils.py**: Tests for format utilities.
- **test_formatting.py**: Tests for formatting module.
- **test_holders.py**: Tests for institutional holders module.
- **test_improvements.py**: Tests for code improvements.
- **test_index.py**: Tests for index module.
- **test_insiders.py**: Tests for insider transactions.
- **test_market_display.py**: Tests for market display functionality.
- **test_market_display_batch.py**: Tests for batch processing in market display.
- **test_market_display_html.py**: Tests for HTML output in market display.
- **test_market_display_unified.py**: Tests for unified market display.
- **test_market_utils.py**: Tests for market utilities.
- **test_metrics.py**: Tests for metrics calculations.
- **test_monthly.py**: Tests for monthly performance.
- **test_news.py**: Tests for news module.
- **test_pagination_utils.py**: Tests for pagination utilities.
- **test_portfolio.py**: Tests for portfolio module.
- **test_pricing.py**: Tests for pricing module.
- **test_rate.py**: Tests for rate limiting.
- **test_rate_limiter.py**: Tests for rate limiter functionality.
- **test_rate_limiter_advanced.py**: Tests for advanced rate limiter features.
- **test_rate_limiter_unified.py**: Tests for unified rate limiter.
- **test_templates.py**: Tests for HTML templates.
- **test_trade.py**: Tests for trade analysis.
- **test_types.py**: Tests for type definitions.
- **test_utils.py**: Tests for utility functions.
- **test_utils_refactor.py**: Tests for refactored utilities.
- **test_validate.py**: Tests for ticker validation.
- **test_weekly.py**: Tests for weekly performance.

#### test/fixtures/ (Test Fixtures)

- **__init__.py**: Fixtures package initialization.
- **README.md**: Documentation for test fixtures.
- **async_fixtures.py**: Fixtures for async testing.
- **pagination.py**: Fixtures for pagination testing.

#### test/utils/ (Test Utilities)

- **__init__.py**: Test utilities package initialization.
- **test_fixtures.py**: Common test utilities and fixtures.
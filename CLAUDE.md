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
- `pytest tests/` - Run all tests
- `pytest tests/test_file.py::TestClass::test_method` - Run specific test
- `pytest tests/ --cov=yahoofinance` - Run tests with coverage
- `pytest tests/test_cons.py tests/test_trade.py --cov=yahoofinance.cons --cov=trade --cov-report=term-missing` - Run specific module tests with coverage
- `pytest -xvs tests/test_specific.py` - Run verbose, no capture
- `python -m yahoofinance.module_name` - Run specific module (news, portfolio, econ)

## Code Style
- **Imports**: Standard library first, third-party packages, then local modules
- **Types**: Use type hints (List, Dict, Optional, Any, Tuple) from typing module
- **Classes**: Use dataclasses for data containers, proper error hierarchy
- **Naming**: snake_case for variables/functions, PascalCase for classes, ALL_CAPS for constants
- **Documentation**: Docstrings with Args/Returns/Raises sections for all functions/classes
- **Error Handling**: Custom exception hierarchy (YFinanceError → APIError, ValidationError)
- **Rate Limiting**: Always use RateLimitTracker for API calls with adaptive delays
- **Formatting**: Format numbers with proper precision (1-2 decimals), handle None values

## Project Organization
- `yahoofinance/` - Main package with modular components
- `yahoofinance/client.py` - API client with rate limiting and caching
- `yahoofinance/display.py` - Output handling and batch processing
- `yahoofinance/formatting.py` - Data formatting and colorization
- `yahoofinance/input/` - Input data files (.csv)
  - `market.csv` - All market tickers for analysis
  - `etoro.csv` - Filtered list of tickers available on eToro
  - `portfolio.csv` - Current portfolio holdings
  - `cons.csv` - Sector/industry data
- `yahoofinance/output/` - Generated output files
  - `buy.csv` - Generated buy recommendations
  - `sell.csv` - Generated sell recommendations
  - `market.csv` - Analysis results from market or eToro tickers
  - `portfolio.csv` - Analysis results from portfolio
  - `index.html`, `portfolio.html` - HTML dashboards
- `tests/` - Test files with comprehensive coverage
  - `test_trade.py` - Tests for CLI functionality (>70% coverage)
  - `test_cons.py` - Tests for market constituents (>70% coverage)
  - `test_formatting.py` - Tests for data formatting utilities
  - Other test modules focusing on specific functionality

## Trading Criteria
- **Buy Signal**:
  - 5 or more analysts covering the stock
  - 20% or greater upside potential
  - 75% or more of analysts recommend buying
  
- **Sell Signal**:
  - 5 or more analysts covering the stock AND either:
    - Less than 5% upside potential, OR
    - Less than 50% of analysts recommend buying

- **Low Confidence**:
  - Fewer than 5 analysts covering the stock
  
- **EXRET Calculation**:
  - Expected Return = Upside Potential × Buy Percentage / 100

## Exchange Ticker Formats
- **Hong Kong (HK) Stocks**:
  - Program automatically fixes eToro HK ticker formats
  - Leading zeros are removed from tickers with 5+ digits
  - Example: `03690.HK` → `3690.HK`
  - 4-digit tickers remain unchanged (e.g., `0700.HK`)

## Performance Optimizations
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
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

## Trading Criteria
- **Strong Buy Signal**:
  - More than 4 analysts covering the stock
  - Greater than 15% upside potential
  - More than 65% of analysts recommend buying
  
- **Strong Sell Signal**:
  - More than 4 analysts covering the stock AND either:
    - Less than 5% upside potential, OR
    - Less than 50% of analysts recommend buying

- **Low Confidence**:
  - Fewer than 4 analysts covering the stock
  
- **EXRET Calculation**:
  - Expected Return = Upside Potential × Buy Percentage / 100

## Exchange Ticker Formats
- **Hong Kong (HK) Stocks**:
  - Program automatically fixes eToro HK ticker formats
  - Leading zeros are removed from tickers with 5+ digits
  - Example: `03690.HK` → `3690.HK`
  - 4-digit tickers remain unchanged (e.g., `0700.HK`)
# Market Analysis and Portfolio Management Tool

A robust Python-based market analysis system that leverages Yahoo Finance data to provide comprehensive stock analysis, portfolio management, and market intelligence. The system features advanced rate limiting, intelligent caching, and multiple output formats.

## Key Features

### 1. Intelligent Data Fetching
- **Advanced Rate Limiting**
  * Adaptive batch processing (15 tickers per batch)
  * Smart delay system (1-30s) based on success rates
  * Thread-safe API call tracking
  * Error pattern detection and handling
  * Success streak monitoring
  * Exponential backoff on errors
  * Ticker-specific error tracking
  * Exchange-specific optimizations for non-US markets
- **Smart Caching**
  * 5-minute TTL for market data
  * 15-minute TTL for news
  * 60-minute TTL for earnings data
  * LRU cache with size limiting (prevents unbounded growth)
  * Memory-efficient storage
  * Automatic cache cleanup

### 2. Comprehensive Analysis
- **Market Data**
  * Real-time price monitoring
  * Target price analysis
  * Analyst recommendations
  * Risk metrics calculation (Beta, Alpha, Sharpe, Sortino)
  * Insider trading patterns
  * Expected return calculations (EXRET)
- **Portfolio Analysis**
  * Performance tracking (Daily, MTD, YTD, 2YR)
  * Risk metrics (Beta, Alpha, Sharpe, Sortino)
  * Position monitoring
  * Returns analysis
- **Trade Recommendations**
  * Buy opportunity identification
  * Portfolio-based sell signals
  * Cross-analysis with existing holdings
  * EXRET-based ranking of opportunities
- **Market Intelligence**
  * News aggregation with sentiment analysis
  * Earnings calendar and tracking
  * Economic indicators
  * Institutional holdings

### 3. Advanced Utilities
- **Rate Limiting Utilities**
  * Thread-safe API call tracking
  * Adaptive delays based on API response patterns
  * Support for both synchronous and asynchronous operations
  * Function decorators for easy application to any API call
- **Pagination Support**
  * Automatic handling of paginated API responses
  * Rate-limiting aware iteration
  * Memory-efficient result buffering
- **Async Capabilities**
  * Asynchronous operations with rate limiting
  * Controlled concurrency to prevent API throttling
  * Safe alternatives to standard asyncio functions
  * Retry mechanisms with exponential backoff

### 4. Multiple Output Formats
- **Console Display**
  * Color-coded metrics based on analysis
  * Progress tracking with tqdm
  * Batch processing feedback
  * Tabulated data with consistent formatting
- **Data Export**
  * CSV data files with comprehensive metrics
  * HTML dashboards with performance indicators
  * Automatic file organization
- **Web Interface**
  * Interactive performance dashboards
  * Risk metric visualization
  * Market index tracking

## Installation

```bash
# Clone repository
git clone https://github.com/weirdapps/etorotrade
cd etorotrade

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Unix
myenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### 1. Market Analysis
```bash
python trade.py
```
Select data source:
- P: Portfolio file (yahoofinance/input/portfolio.csv)
- M: Market file (yahoofinance/input/market.csv)
- E: eToro Market file (yahoofinance/input/etoro.csv) - filtered for eToro-available tickers
- T: Trade analysis (buy/sell recommendations)
- I: Manual ticker(s) input (comma separated)

When selecting Trade Analysis (T), you can:
- B: Explore Buy opportunities (not in current portfolio)
- S: Explore Sell candidates in current portfolio
- H: Explore Hold candidates (stocks that meet neither buy nor sell criteria)

The analysis automatically saves results to CSV files:
- Buy recommendations: yahoofinance/output/buy.csv
- Sell recommendations: yahoofinance/output/sell.csv
- Market analysis: yahoofinance/output/market.csv

### Ticker Management

#### Yahoo Finance Validation
The system has built-in ticker validation to filter out invalid or delisted tickers. This prevents errors during batch processing and improves overall reliability.

To validate tickers:
```bash
# Run this to validate tickers against Yahoo Finance API
python -m yahoofinance.validate
```

The validation process:
1. Asks for tickers to be validated 
2. Checks each ticker against Yahoo Finance API
3. Saves valid tickers to `yahoofinance/input/yfinance.csv`

This significantly improves processing time and reduces API errors when running market analysis.

#### eToro Ticker Management
The yahoofinance/input/etoro.csv file contains a subset of tickers that are available for trading on eToro. By using the 'E' option in the main program, you can analyze only these eToro-tradable stocks, significantly reducing processing time from potentially thousands of tickers to just the ones you can actually trade.

Format of etoro.csv:
```
symbol,name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
AMZN,Amazon.com Inc.
...
```

You can manually edit this file to add or remove tickers as you discover what's available on eToro. This focused approach makes analysis much faster and more relevant for your trading activities.

### 2. News Analysis
```bash
python -m yahoofinance.news
```
Features:
- Latest news with sentiment scores (-1 to +1)
- Color-coded sentiment indicators
- Source attribution and timestamps
- Full article links

### 3. Portfolio Tracking
```bash
python -m yahoofinance.portfolio
```
Displays:
- Current performance metrics
- Risk indicators (Beta, Alpha, Sharpe, Sortino)
- Position updates
- Historical returns (Daily, MTD, YTD, 2YR)

### 4. Economic Calendar
```bash
python -m yahoofinance.econ
```
Tracks:
- GDP Growth Rate
- Unemployment Rate
- CPI (Month-over-Month)
- Federal Funds Rate
- And more...

## Configuration

### Required Files
```
yahoofinance/
├── input/
│   ├── portfolio.csv  # Portfolio holdings
│   ├── market.csv     # Market watchlist
└── .env              # API keys and settings
```

### Environment Variables
```env
FRED_API_KEY=your_fred_api_key
NEWS_API_KEY=your_news_api_key  # Optional
```

## Metrics Guide

### Price Metrics
- **PRICE**: Current stock price
- **TARGET**: Average analyst target
- **UPSIDE**: Target vs current price (%)
- **EXRET**: Expected return (Upside × Buy%)

### Analyst Coverage
- **# T**: Number of price targets
- **% BUY**: Buy rating percentage
- **# A**: Number of ratings
- **A**: Rating source (E: post-earnings, A: all-time)

### Valuation
- **PET**: Trailing P/E ratio
- **PEF**: Forward P/E ratio
- **PEG**: Price/Earnings to Growth
- **DIV%**: Dividend yield

### Risk Metrics
- **BETA**: Market volatility comparison
- **SI**: Short interest percentage
- **INS%**: Insider buy percentage
- **# INS**: Insider transaction count

## Trading Recommendation Criteria

The system classifies stocks into four categories:

- 🟡 **INCONCLUSIVE** (Low Confidence/Insufficient Data)
  * Less than 5 price targets OR
  * Less than 5 analyst ratings

For stocks that pass the confidence threshold (5+ price targets and 5+ analyst ratings):

- 🔴 **SELL** - Checked first due to risk management priority, triggered if ANY of these conditions are met:
  * Less than 5% upside OR
  * Less than 65% buy ratings OR
  * PEF > PET (deteriorating earnings outlook, when both are positive) OR
  * PEG > 3.0 (overvalued relative to growth) OR
  * SI > 5% (high short interest) OR
  * Beta > 3.0 (excessive volatility)

- 🟢 **BUY** - Checked after eliminating sell candidates, ALL of these conditions must be met:
  * 20% or more upside AND
  * 82% or more buy ratings AND
  * Beta <= 3.0 (acceptable volatility) AND
  * Beta > 0.2 (sufficient volatility) AND
  * PEF < PET (improving earnings outlook) OR Trailing P/E ≤ 0 (negative) AND
  * PEF > 0.5 (positive earnings projection) AND
  * PEG < 3.0 (reasonable valuation relative to growth) - *ignored if PEG data not available* AND
  * SI <= 5% (acceptable short interest) - *ignored if SI data not available*

- ⚪ **HOLD** - Stocks with balanced risk profile
  * Stocks that pass the confidence check
  * Don't meet the criteria for Buy or Sell recommendations

## Architecture

The system uses a modular architecture:
1. **Client Layer** (client.py): API interactions with rate limiting
2. **Utilities Layer**: Reusable components for different operations
   - **Specialized Submodules**:
     - `utils/data/` - Data formatting utilities
     - `utils/network/` - Rate limiting and API communication
     - `utils/market/` - Market-specific utilities like ticker validation
     - `utils/date/` - Date manipulation and formatting
     - `utils/async/` - Asynchronous operation helpers
3. **Analysis Layer** (Multiple modules): Data processing and calculations
4. **Display Layer** (display.py): Output formatting and presentation

Key components:
- **YFinanceError Hierarchy**: Comprehensive error handling system
- **AdaptiveRateLimiter**: Advanced rate limiting with thread safety
- **YFinanceClient**: Handles Yahoo Finance API interactions
- **Cache**: LRU caching system with size limiting
- **StockData**: Core data structure for stock information
- **DisplayFormatter**: Formats data for console output
- **MarketDisplay**: Manages batch processing and report generation
- **PaginatedResults**: Handles paginated API responses efficiently

## Trading Platform Integration

### eToro Compatibility
- **Hong Kong Stocks**: Automatic normalization of HK stock tickers
  * Removes leading zeros from 5+ digit ticker numbers
  * Example: eToro's `03690.HK` format is automatically converted to `3690.HK`
  * Compatible with portfolio.csv imports from eToro
- **Crypto Tickers**: Standardizes eToro's crypto tickers
  * Automatically converts tickers to the `-USD` format (e.g., `BTC-USD`, `ETH-USD`)

### Performance Optimizations
- **Market-Specific Data Fetching**
  * Automatically detects US vs non-US tickers based on exchange suffix
  * Skips known-to-fail API calls for non-US exchanges (ratings, insider data, short interest)
  * Reduces API load and speeds up batch processing by 20-30% for international stocks
  * Falls back to available data sources for non-US market analysis
- **Extended Ticker Format Support**
  * Standard tickers: validated up to 10 characters
  * Exchange-specific tickers: validated up to 20 characters
  * Supports complex formats like `MAERSK-A.CO` (Danish market)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=yahoofinance tests/

# Run specific tests
pytest tests/test_market_display.py

# Run tests for utility modules
pytest tests/test_utils.py tests/test_rate.py tests/test_async.py tests/test_errors.py

# Run specific test case
pytest tests/test_market_display.py::TestMarketDisplay::test_display_report

# Run tests with coverage for specific modules
pytest tests/test_trade.py --cov=trade --cov-report=term-missing
```

The codebase includes extensive test coverage for critical components:
- Core modules have >70% test coverage
- Utility modules have >90% test coverage
- 60+ comprehensive test cases covering both normal operations and edge cases
- Test mocking for network calls and API interactions
- Integration tests for end-to-end workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow code style guidelines in CLAUDE.md
4. Add tests for new features
5. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Yahoo Finance API (via yfinance package)
- FRED Economic Data
- Pandas, Tabulate, and other great Python libraries
- Contributors and maintainers


[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

### FOLLOW OR COPY PLESSAS ON ETORO
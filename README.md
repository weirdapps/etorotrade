# Market Analysis and Portfolio Management Tool

A robust Python-based market analysis system that leverages Yahoo Finance data to provide comprehensive stock analysis, portfolio management, and market intelligence. The system features advanced rate limiting, intelligent caching, and multiple output formats.

## Key Features

### 1. Intelligent Data Fetching
- **Advanced Rate Limiting**
  * Adaptive batch processing (15 tickers per batch)
  * Smart delay system (1-30s) based on success rates
  * Error pattern detection and handling
  * Success streak monitoring
  * Exponential backoff on errors
  * Ticker-specific error tracking
  * Exchange-specific optimizations for non-US markets
- **Smart Caching**
  * 5-minute TTL for market data
  * 15-minute TTL for news
  * LRU cache for frequently accessed tickers
  * Memory-efficient storage

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
  * Major Market Indices Coverage:
    - US: S&P 500
    - UK: FTSE 100
    - France: CAC 40
    - Germany: DAX
    - Spain: IBEX 35
    - Italy: FTSE MIB
    - Portugal: PSI
    - Switzerland: SMI
    - Denmark: OMXC25
    - Greece: ATHEX
    - Japan: Nikkei 225
    - Hong Kong: Hang Seng

### 3. Multiple Output Formats
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
git clone https://github.com/yourusername/trade
cd trade

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

The analysis automatically saves results to CSV files:
- Buy recommendations: yahoofinance/output/buy.csv
- Sell recommendations: yahoofinance/output/sell.csv
- Market analysis: yahoofinance/output/market.csv

### Ticker Management

#### Yahoo Finance Validation
The system has built-in ticker validation to filter out invalid or delisted tickers. This prevents errors during batch processing and improves overall reliability.

To validate tickers and create a filtered list:
```bash
# Run this once to validate all tickers against Yahoo Finance API
python -m yahoofinance.validate

# After validation, cons.py will automatically filter against valid tickers
python -m yahoofinance.cons
```

The validation process:
1. Checks each ticker against Yahoo Finance API
2. Saves valid tickers to `yahoofinance/input/yfinance.csv`
3. Subsequent runs of `cons.py` use this list to filter out invalid tickers

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

## Color Coding System

- 🟢 **Green** (Buy)
  * 5+ price targets (# T)
  * 5+ analyst ratings (# A)
  * 20%+ upside
  * 80%+ buy ratings

- 🔴 **Red** (Sell)
  * 5+ price targets (# T)
  * 5+ analyst ratings (# A) AND
  * < 5% upside OR
  * < 60% buy ratings

- 🟡 **Yellow** (Low Confidence)
  * < 5 price targets OR
  * < 5 analyst ratings OR
  * Limited data

- ⚪ **White** (Hold)
  * Metrics between buy/sell thresholds

## Architecture

The system uses a modular architecture:
1. **Client Layer** (client.py): API interactions with rate limiting
2. **Analysis Layer** (Multiple modules): Data processing and calculations
3. **Display Layer** (display.py): Output formatting and presentation

Key components:
- **RateLimitTracker**: Manages API call timing and adaptive delays
- **YFinanceClient**: Handles Yahoo Finance API interactions
- **StockData**: Core data structure for stock information
- **DisplayFormatter**: Formats data for console output
- **MarketDisplay**: Manages batch processing and report generation

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

# Run specific test case
pytest tests/test_market_display.py::TestMarketDisplay::test_display_report

# Run tests with coverage for specific modules
pytest tests/test_cons.py tests/test_trade.py --cov=yahoofinance.cons --cov=trade --cov-report=term-missing
```

The codebase includes extensive test coverage for critical components:
- Core modules have >70% test coverage
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
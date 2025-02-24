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
- I: Manual ticker(s) input (comma separated)

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

- 🟢 **Green** (Strong Buy)
  * 4+ analysts
  * 15%+ upside
  * 65%+ buy ratings

- 🔴 **Red** (Sell)
  * 4+ analysts AND
  * < 5% upside OR
  * < 50% buy ratings

- 🟡 **Yellow** (Low Confidence)
  * < 4 analysts OR
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
```

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


### FOLLOW OR COPY PLESSAS ON ETORO
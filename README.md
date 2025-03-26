# Market Analysis and Portfolio Management Tool

A robust Python-based market analysis system that leverages Yahoo Finance data to provide comprehensive stock analysis, portfolio management, and market intelligence. The system features advanced rate limiting, intelligent caching, and multiple output formats.

![eToro Trade Analysis Tool](assets/etorotrade.png)

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
  * Company names in ALL CAPS for readability
  * Market cap values with T/B suffixes (e.g., "2.75T", "175B")
  * Consistent column alignment and number formatting
- **Data Export**
  * CSV data files with comprehensive metrics
  * HTML dashboards with performance indicators
  * Automatic file organization
- **HTML Dashboards**
  * Performance dashboards with metrics visualization
  * Risk metric visualization
  * Market index tracking
  * Generated from templates in templates.py

## Installation

```bash
# Clone repository
git clone https://github.com/weirdapps/etorotrade
cd etorotrade

# Create virtual environment (Python 3.8+ recommended)
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
- Hold recommendations: yahoofinance/output/hold.csv
- Market analysis: yahoofinance/output/market.csv
- Portfolio analysis: yahoofinance/output/portfolio.csv

### Custom Analysis Using Provider Pattern

You can also use the provider pattern to create custom analyses:

```python
from yahoofinance import get_provider

# Get the default provider
provider = get_provider()

# Get data for a specific ticker
info = provider.get_ticker_info("AAPL")
print(f"Company: {info['name']}")
print(f"Current price: ${info['current_price']}")
print(f"Target price: ${info['target_price']}")
print(f"Upside potential: {info['upside_potential']}%")

# For async operations
import asyncio
from yahoofinance import get_provider

async def analyze_portfolio():
    # Get async provider
    provider = get_provider(async_mode=True)
    
    # Analyze multiple tickers concurrently
    portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    results = await provider.batch_get_ticker_info(portfolio)
    
    # Process results
    for ticker, data in results.items():
        if data:
            print(f"{ticker}: {data['name']} - ${data['current_price']}")

# Run async function
asyncio.run(analyze_portfolio())
```

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

#### "No Trade" List
The system supports a list of tickers to exclude from trading recommendations (yahoofinance/input/notrade.csv). Tickers in this list will still be analyzed but won't appear in buy/sell recommendations.

Format of notrade.csv:
```
symbol
AAPL
MSFT
...
```

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

### 5. Earnings Calendar
```bash
python -m yahoofinance.earnings
```
Features:
- Upcoming earnings dates
- Historical earnings surprises
- Earnings estimate trends

### 6. Market Index Tracking
```bash
python -m yahoofinance.index
```
Features:
- Weekly and monthly index performance
- Major US and international indices
- Customizable time periods

## Configuration

### Project Structure
```
/
├── yahoofinance/           # Main package
│   ├── api/                # Provider interfaces and implementations
│   │   └── providers/      # Data provider implementations
│   ├── core/               # Core functionality
│   │   ├── errors.py       # Error hierarchy
│   │   ├── config.py       # Configuration settings
│   │   └── types.py        # Type definitions
│   ├── analysis/           # Analysis modules
│   │   ├── analyst.py      # Analyst ratings
│   │   ├── performance.py  # Performance tracking
│   │   └── ...
│   ├── utils/              # Utility modules
│   │   ├── network/        # Network utilities (rate limiting, pagination)
│   │   ├── data/           # Data formatting utilities
│   │   ├── market/         # Market-specific utilities
│   │   ├── date/           # Date utilities
│   │   └── async/          # Async utilities
│   ├── presentation/       # Presentation components
│   │   ├── console.py      # Console output
│   │   ├── html.py         # HTML generation
│   │   └── templates.py    # HTML templates
│   ├── input/              # Input data files
│   │   ├── portfolio.csv   # Portfolio holdings
│   │   ├── market.csv      # Market watchlist
│   │   ├── etoro.csv       # eToro available tickers
│   │   ├── notrade.csv     # Tickers to exclude from recommendations
│   │   ├── cons.csv        # Consolidated list of important tickers
│   │   ├── us_tickers.csv  # US market tickers
│   │   └── yfinance.csv    # Validated tickers
│   └── output/             # Generated output files
│       ├── buy.csv         # Buy recommendations
│       ├── sell.csv        # Sell recommendations
│       ├── hold.csv        # Hold recommendations
│       ├── market.csv      # Market analysis results
│       ├── portfolio.csv   # Portfolio analysis results
│       ├── index.html      # HTML dashboard
│       ├── portfolio.html  # Portfolio HTML dashboard
│       ├── script.js       # Dashboard JavaScript
│       └── styles.css      # Dashboard CSS
├── tests/                  # Test files
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── scripts/                # Utility scripts
├── assets/                 # Static assets
├── trade.py                # Main entry point
├── CLAUDE.md               # Comprehensive documentation
├── requirements.txt        # Dependencies
└── .env                    # API keys and settings
```

### Input File Formats

**portfolio.csv** - Your current holdings:
```
symbol,shares,cost,date
AAPL,10,150.25,2022-03-15
MSFT,5,280.75,2022-04-20
...
```

**market.csv** - Full watchlist of tickers to analyze:
```
symbol,sector
AAPL,Technology
MSFT,Technology
...
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
  * PEF > 45.0 (extremely high valuation) OR
  * PEG > 3.0 (overvalued relative to growth) OR
  * SI > 4% (high short interest) OR
  * Beta > 3.0 (excessive volatility) OR
  * EXRET < 10.0 (insufficient expected return)

- 🟢 **BUY** - Checked after eliminating sell candidates, ALL of these conditions must be met:
  * 20% or more upside AND
  * 82% or more buy ratings AND
  * Beta <= 3.0 (acceptable volatility) AND
  * Beta > 0.2 (sufficient volatility) AND
  * PEF < PET (improving earnings outlook) OR Trailing P/E ≤ 0 (negative) AND
  * PEF > 0.5 (positive earnings projection) AND
  * PEF <= 45.0 (reasonable valuation) AND
  * PEG < 3.0 (reasonable valuation relative to growth) - *ignored if PEG data not available* AND
  * SI <= 3% (acceptable short interest) - *ignored if SI data not available*

- ⚪ **HOLD** - Stocks with balanced risk profile
  * Stocks that pass the confidence check
  * Don't meet the criteria for Buy or Sell recommendations

**Note:** These exact same criteria are used for both coloring in market/portfolio views and for filtering stocks into the buy/sell/hold lists. There is perfect alignment between the color a stock receives in the main views and which list it appears in with the trade command.

## Architecture

The system uses a modern provider-based architecture with five key layers:

1. **Provider Layer**: Abstract interfaces and implementations for data access
   - Base provider interfaces (`FinanceDataProvider`, `AsyncFinanceDataProvider`)
   - Implementations (`YahooFinanceProvider`, `AsyncYahooFinanceProvider`, `EnhancedAsyncYahooFinanceProvider`)
   - Factory function (`get_provider()`) for obtaining appropriate provider instances

2. **Core Layer**: Fundamental services and definitions
   - Error handling (`core.errors`)
   - Configuration (`core.config`)
   - Logging (`core.logging`)
   - Type definitions (`core.types`)

3. **Utilities Layer**: Reusable components with specialized submodules
   - Network utilities: Rate limiting, pagination, circuit breakers
   - Data utilities: Formatting, transformation
   - Market utilities: Ticker validation and normalization
   - Date utilities: Date handling and formatting
   - Async utilities: Asynchronous operations support

4. **Analysis Layer**: Domain-specific processing modules
   - Finance-specific analysis components
   - Performance tracking
   - Market data processing

5. **Presentation Layer**: Output formatting and display
   - Console output
   - HTML generation
   - Data visualization

The codebase follows a clean, organized structure:
- Core functionality in canonical source modules (e.g., `utils/network/async_utils/rate_limiter.py`)
- Compatibility layers provide backward compatibility (e.g., `utils/async_helpers.py`)
- Clear provider interfaces ensure consistent data access patterns

Key components:
- **Provider Pattern**: Abstracts data access behind consistent interfaces
- **YFinanceError Hierarchy**: Comprehensive error handling system
- **AdaptiveRateLimiter**: Advanced rate limiting with thread safety
- **AsyncRateLimiter**: Async-compatible rate limiting
- **Circuit Breaker Pattern**: Resilient handling of external service failures
- **Cache**: LRU caching system with size limiting
- **PerformanceTracker**: Tracks performance metrics and generates dashboards
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

## Display Formatting

The system follows consistent formatting rules:

- **Company Names**: 
  * Always displayed in ALL CAPS for readability
  * Truncated to maximum 14 characters if needed
  * Left-aligned in all table outputs

- **Market Cap Formatting**:
  * Trillion-scale values use "T" suffix (e.g., "2.75T")
    * ≥ 10T: 1 decimal place (e.g., "12.5T")
    * < 10T: 2 decimal places (e.g., "2.75T")
  * Billion-scale values use "B" suffix (e.g., "175B")
    * ≥ 100B: No decimals (e.g., "175B")
    * ≥ 10B and < 100B: 1 decimal place (e.g., "25.5B")
    * < 10B: 2 decimal places (e.g., "5.25B")
  * Right-aligned in all table outputs
  
- **Percentage Formatting**:
  * Upside, EXRET, SI use 1 decimal place with % suffix (e.g., "27.9%")
  * Buy percentage uses 0 decimal places with % suffix (e.g., "85%")
  * Dividend yield uses 2 decimal places with % suffix (e.g., "0.84%")

## Troubleshooting

### Common Issues

1. **Rate Limiting Errors**
   * Symptom: "API rate limit exceeded" errors
   * Solution: The system implements automatic backoff. Simply wait and retry.

2. **Invalid Ticker Errors**
   * Symptom: "Invalid ticker" warnings during batch processing
   * Solution: Run `python -m yahoofinance.validate` to update the valid tickers list

3. **Missing Data for Non-US Stocks**
   * Symptom: Missing analyst ratings or insider data for international tickers
   * Reason: These data points are often unavailable through the API for non-US markets
   * Solution: The system automatically adapts by skipping unavailable data sources

4. **Format Errors in Input Files**
   * Symptom: "KeyError" or "Expected column X" errors
   * Solution: Check the format of your input CSV files against the examples provided

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

# Run verbosely with no output capture
pytest -xvs tests/test_specific.py
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
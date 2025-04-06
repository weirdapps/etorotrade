# etorotrade - Market Analysis & Portfolio Management

A powerful Python-based analysis system that provides actionable trading recommendations for stocks, portfolios, and market intelligence using Yahoo Finance data. The tool features sophisticated rate limiting, intelligent caching, and multiple output formats to help you make informed investment decisions.

![eToro Trade Analysis Tool](assets/etorotrade.png)


## What's This Tool For?

etoroTRADE helps you:
- **Analyze your portfolio** for potential sell candidates
- **Discover new buy opportunities** based on upside potential and analyst consensus
- **Monitor market conditions** across US, European, and Chinese markets
- **Track news sentiment** to stay ahead of market-moving events
- **Follow insider transactions** and institutional activity
- **Generate actionable trade recommendations** based on comprehensive criteria
- **Backtest trading strategies** to optimize criteria parameters
- **Track performance** of market indices and your portfolio

## Trade.py - Main Application

The primary interface is `trade.py`, which provides several analysis options:

```bash
python trade.py
```

> **Note:** To run non-interactively (e.g., in scripts), you can pipe the menu selections. For example, to run Portfolio Analysis (P) using the existing file (E):
> ```bash
> echo "p\ne" | python trade.py
> ```

### Menu Options

- **P: Portfolio Analysis**
  - Analyzes your current holdings from portfolio.csv
  - Shows performance metrics, risk factors, and recommendations
  - Outputs to yahoofinance/output/portfolio.csv and HTML dashboard

- **M: Market Analysis**
  - Prompts for market selection (USA, Europe, China, or Manual)
  - Analyzes selected market for investment opportunities
  - Outputs to yahoofinance/output/market.csv and HTML dashboard

- **E: eToro Market Analysis**
  - Analyzes tickers available on eToro platform
  - Perfect for eToro users to find opportunities within available assets
  - Outputs to yahoofinance/output/market.csv and HTML dashboard

- **T: Trade Analysis**
  - Provides actionable trading recommendations with sub-options:
    - **B: Buy Opportunities** - New stocks to consider purchasing
    - **S: Sell Candidates** - Portfolio stocks to consider selling
    - **H: Hold Candidates** - Stocks with neutral outlook
  - Outputs to yahoofinance/output/buy.csv, sell.csv, or hold.csv with HTML dashboards

- **I: Manual Ticker Input**
  - Analyze specific tickers entered manually
  - Outputs to yahoofinance/output/manual.csv and HTML dashboard

## Trading Classification Criteria

The system uses these criteria to classify stocks into four categories:

### 🟡 INCONCLUSIVE
- Low confidence due to:
  - Less than 5 price targets OR
  - Less than 5 analyst ratings

### 🔴 SELL (Risk Management)
Triggered if ANY of these conditions are met:
- Less than 5% upside potential
- Less than 65% buy ratings from analysts
- Forward P/E (PEF) > Trailing P/E (PET) - deteriorating earnings outlook
- Forward P/E (PEF) > 50.0 (extremely high valuation)
- PEG ratio > 3.0 (overvalued relative to growth)
- Short Interest (SI) > 2% (high short interest)
- Beta > 3.0 (excessive volatility)
- Expected Return (EXRET) < 5.0 (insufficient potential return)

### 🟢 BUY (Growth Opportunity)
ALL of these conditions must be met:
- 20% or more upside potential
- 85% or more buy ratings from analysts
- Beta data must be available (primary required criterion)
- Acceptable volatility (0.25 < Beta ≤ 2.5)
- PE Forward (PEF) and PE Trailing (PET) data must be available (primary required criteria)
  - The hybrid provider ensures PEF data is available for 90%+ international stocks
- Improving earnings outlook (PEF < PET) OR Negative trailing P/E
- Positive earnings projection (0.5 < PEF ≤ 45.0)
- Reasonable valuation relative to growth (PEG < 2.5) - if PEG data available (secondary criterion)
- Acceptable short interest (SI ≤ 1.5%) - if SI data available (secondary criterion) 
- Strong expected return (EXRET ≥ 15.0)

### ⚪ HOLD
- Stocks that pass confidence threshold
- Don't meet SELL criteria
- Don't meet BUY criteria

## Analysis Modules

### News & Market Intelligence
```bash
# News with sentiment analysis
python -m yahoofinance.analysis.news

# Economic indicators and metrics
python -m yahoofinance.analysis.metrics
```

### Performance Tracking
```bash
# Portfolio performance metrics
python -m yahoofinance.analysis.portfolio

# Market and portfolio performance tracking
python -m yahoofinance.analysis.performance
```

### Analyst & Earnings
```bash
# Analyst ratings and recommendations
python -m yahoofinance.analysis.analyst

# Upcoming earnings dates and surprises
python -m yahoofinance.analysis.earnings
```

### Insider Transactions
```bash
# Insider transactions analysis
python -m yahoofinance.analysis.insiders
```

### Backtesting & Optimization
```bash
# Run a backtest with default settings
python scripts/optimize_criteria.py --mode backtest

# Optimize trading criteria parameters
python scripts/optimize_criteria.py --mode optimize --metric sharpe_ratio
```

## Backtesting Framework

The backtesting framework allows you to:

1. Test your trading criteria against historical data
2. Optimize criteria parameters to find the best performing combination
3. Generate performance reports comparing to benchmark indices
4. Perform what-if analysis with different market conditions
5. Use market cap-based position weighting (1% to 10% per position) for realistic portfolio allocation
6. Analyze detailed portfolio synthesis tables with complete metrics for each position

For detailed instructions on using the backtesting framework, see [scripts/BACKTEST_README.md](scripts/BACKTEST_README.md).

### Basic Examples

```bash
# Run a 2-year backtest on your portfolio
python scripts/optimize_criteria.py --mode backtest --period 2y --source portfolio

# Optimize sell criteria parameters for maximum Sharpe ratio
python scripts/optimize_criteria.py --mode optimize --param-file scripts/sample_parameters.json

# Test specific tickers with weekly rebalancing
python scripts/optimize_criteria.py --mode backtest --tickers AAPL,MSFT,GOOGL,AMZN --rebalance weekly
```

## Architecture & Design

etorotrade follows a modern provider-based architecture with five key layers:

1. **Provider Layer**: Abstract interfaces and implementations for data access
   - Standardized interfaces for both synchronous and asynchronous operations
   - Multiple provider implementations (Yahoo Finance, YahooQuery, Hybrid, Enhanced Async)
   - Factory function for obtaining provider instances: `get_provider()`
   - Default hybrid provider uses YahooFinance + YahooQuery for optimal data completeness

2. **Core Layer**: Fundamental services and definitions
   - Error handling hierarchy for consistent error management
   - Centralized configuration for application settings
   - Type definitions and logging configuration

3. **Analysis Layer**: Domain-specific processing modules
   - Stock data analysis and market intelligence
   - Portfolio performance tracking
   - Trading recommendations based on configurable criteria

4. **Utilities Layer**: Reusable components with specialized functionality
   - Network utilities with rate limiting and circuit breaker patterns
   - Data formatting utilities for consistent presentation
   - Market utilities for ticker validation and normalization
   - Advanced hybrid data provider that combines YFinance with YahooQuery for optimal data quality

5. **Presentation Layer**: Output formatting and display
   - Consistent console output with standardized formatting
   - HTML dashboard generation with responsive design
   - Visual indicators for buy/sell/hold recommendations

### Provider Pattern

The provider pattern abstracts data access behind consistent interfaces:

```python
# Using the provider pattern (recommended)
from yahoofinance import get_provider

# Get the default hybrid provider (combines YahooFinance + YahooQuery for best data)
provider = get_provider()
ticker_info = provider.get_ticker_info("AAPL")

# Specify a particular provider implementation if needed
yf_provider = get_provider(provider_name='yahoo')
yq_provider = get_provider(provider_name='yahooquery')

# Get asynchronous provider
async_provider = get_provider(async_api=True)
ticker_info = await async_provider.get_ticker_info("MSFT")

# Batch processing with async provider
batch_results = await async_provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])
```

All providers implement the same interface, ensuring consistent usage regardless of the underlying implementation. This design allows for easy testing, mocking, and future expansion to other data sources.

### YahooQuery Integration Toggle

By default, the hybrid provider uses both YahooFinance (yfinance) and YahooQuery (yahooquery) libraries to provide the most complete data, especially for PEG ratios and P/E Forward values. However, in case of Yahoo Finance API changes or to reduce API rate limiting issues, you can disable YahooQuery integration with a configuration flag:

```python
# In yahoofinance/core/config.py
PROVIDER_CONFIG = {
    # Enable yahooquery supplementation in hybrid provider
    "ENABLE_YAHOOQUERY": False,  # Set to False to disable yahooquery and prevent crumb errors
}
```

Setting `ENABLE_YAHOOQUERY` to `False` will prevent all yahooquery API calls and rely solely on yfinance data. This can be useful when:

1. You encounter "Failed to obtain crumb" errors from yahooquery
2. You need to reduce API call volume
3. Yahoo Finance makes API changes that temporarily break yahooquery

The providers still maintain the same interface and functionality, but certain fields like PEG ratio might be less available for some tickers when yahooquery is disabled.

For more details on the codebase architecture, see [CLAUDE.md](CLAUDE.md).

## Setup Instructions

### Installation

```bash
# Clone the repository
git clone https://github.com/weirdapps/etorotrade
cd etorotrade

# Create and activate virtual environment
python -m venv myenv

# On Windows
myenv\Scripts\activate

# On macOS/Linux
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Input Files

Create/modify these files in the yahoofinance/input/ directory:

1. **portfolio.csv** - Your current holdings
   ```
   symbol,shares,cost,date
   AAPL,10,150.25,2022-03-15
   MSFT,5,280.75,2022-04-20
   ```

2. **market.csv** - General market watchlist
   ```
   symbol,sector
   AAPL,Technology
   MSFT,Technology
   ```

3. **etoro.csv** - Tickers available on eToro (pre-populated)
4. **usa.csv**, **europe.csv**, **china.csv** - Regional ticker lists (pre-populated)
5. **notrade.csv** - Tickers to exclude from trading recommendations

## Project Structure

```
etorotrade/
├── CLAUDE.md             # Detailed technical documentation
├── LICENSE               # MIT License
├── README.md             # This file
├── assets/               # Application assets
│   └── etorotrade.png    # Screenshot
├── logs/                 # Log files directory
├── requirements.txt      # Python dependencies
├── scripts/              # Utility scripts
│   ├── BACKTEST_README.md         # Backtesting documentation
│   ├── optimize_criteria.py       # Backtesting & optimization CLI
│   └── sample_parameters.json     # Sample parameter ranges
├── trade.py              # Main application entry point
├── yahoofinance/         # Core package
    ├── analysis/         # Analysis modules
    │   ├── backtest.py   # Backtesting framework
    │   └── ...           # Other analysis modules
    ├── api/              # Provider interfaces
    │   └── providers/    # Data providers
    ├── core/             # Core functionality
    ├── data/             # Data management
    ├── input/            # Input data files
    ├── output/           # Generated outputs
    │   └── backtest/     # Backtesting results
    ├── presentation/     # Display formatting
    └── utils/            # Utility modules
        ├── async_utils/  # Enhanced async utilities
        ├── data/         # Data formatting utilities
        ├── date/         # Date handling utilities
        ├── market/       # Market-specific utilities
        └── network/      # Network utilities (rate limiting, circuit breaker)
```

## Real-World Investment Performance

I personally use this script to power my eToro investment decisions. For real-world results and validation of this approach, you can follow or copy my eToro portfolio:

👉 [@plessas on eToro](https://www.etoro.com/people/plessas)

---

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
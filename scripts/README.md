# Scripts Usage Guide

This directory contains utility scripts for the etorotrade project. These scripts provide additional functionality beyond the main application, such as portfolio optimization, data processing, system monitoring, and development tools.

## Available Scripts

### Download Portfolio Data

**File**: `download_portfolio_data.py`

Downloads historical price data for all tickers in your portfolio and saves it to cache files for faster portfolio optimization.

```bash
python scripts/download_portfolio_data.py [options]
```

Options:
- `--portfolio`: Path to portfolio CSV file (default: yahoofinance/input/portfolio.csv)
- `--max-years`: Maximum number of years of historical data (default: 6)
- `--batch-size`: Number of tickers per batch (default: 10)
- `--output`: Cache output file path (default: yahoofinance/data/portfolio_cache.pkl)
- `--price-output`: Price output file path (default: yahoofinance/data/portfolio_prices.json)
- `--delay`: Base delay between API calls in seconds (default: 1.0)
- `--verbose`: Enable verbose logging

### Code Quality Checks

**File**: `lint.sh`

Runs code quality checks on the codebase using black, isort, flake8, and mypy.

```bash
./scripts/lint.sh [fix]
```

Parameters:
- `fix`: Optional. When provided, automatically fixes formatting issues where possible.

### Optimize Trading Criteria

**File**: `optimize_criteria.py`

Backtests and optimizes trading criteria parameters to find the best performing combination.

```bash
python scripts/optimize_criteria.py [options]
```

Key options:
- `--mode`: Mode to run (`backtest` or `optimize`, default: backtest)
- `--period`: Backtest period (e.g., '1y', '3y', '5y', default: '3y')
- `--metric`: Metric to optimize (default: sharpe_ratio)
- `--tickers`: Comma-separated list of tickers to backtest
- `--source`: Source of tickers (default: portfolio)
- `--param-file`: JSON file with parameter ranges (default: use built-in ranges)
- `--output`: Output file for results
- `--ticker-limit`: Limit number of tickers for faster execution

### Enhanced Monitoring

**File**: `run_enhanced_monitoring.py`

Runs enhanced monitoring with structured logging, health endpoints, and a real-time dashboard.

```bash
python scripts/run_enhanced_monitoring.py [options]
```

Key options:
- `--health-port`: Port for health check endpoints (default: 8081)
- `--dashboard-port`: Port for monitoring dashboard (default: 8000)
- `--refresh`: Dashboard refresh interval in seconds (default: 30)
- `--log-level`: Logging level (default: INFO)
- `--log-file`: Log file path (default: None, logs to console only)
- `--timeout`: Time in seconds to run before exiting (default: 300)

### Basic Monitoring Dashboard

**File**: `run_monitoring.py`

Runs a monitoring dashboard for system performance.

```bash
python scripts/run_monitoring.py [options]
```

Key options:
- `--port`: Port to run server on (default: 8000)
- `--refresh`: Dashboard refresh interval in seconds (default: 30)
- `--no-browser`: Do not open dashboard in browser
- `--timeout`: Time in seconds to run the dashboard (default: 60)
- `--max-updates`: Maximum number of dashboard updates (default: 5)

### Portfolio Optimizer

**File**: `run_optimizer.py`

Runs the portfolio optimizer to find optimal position sizes based on historical performance.

```bash
python scripts/run_optimizer.py [options]
```

Key options:
- `--min`: Minimum position size in USD (default: 1000.0)
- `--max`: Maximum position size in USD (default: 25000.0)
- `--periods`: Time periods in years to analyze (default: 1 3 4 5)
- `--limit`: Limit the number of tickers to process (0 = no limit)
- `--use-cache`: Use cached historical data and prices
- `--cache-path`: Path to cached historical data
- `--price-cache-path`: Path to cached price data

### Split eToro by Region

**File**: `split_etoro_by_region.py`

Splits the eToro CSV file into regional files (China, Europe, USA) based on ticker patterns.

```bash
python scripts/split_etoro_by_region.py
```

This script:
- Reads from yahoofinance/input/etoro.csv
- Creates separate files for:
  - China (.HK tickers) in china.csv
  - Europe (other .XX tickers) in europe.csv
  - USA (no .XX suffix) in usa.csv
- Preserves the original etoro.csv file

### Parameter Configuration

**File**: `sample_parameters.json`

Sample parameter ranges for trading criteria optimization. Used by optimize_criteria.py to define the search space for finding optimal trading parameters.

Key sections:
- `SELL`: Parameters for sell criteria
- `BUY`: Parameters for buy criteria
- `CONFIDENCE`: Thresholds for confidence in recommendations

## Common Usage Patterns

### Full Optimization Workflow

```bash
# 1. Download historical data
python scripts/download_portfolio_data.py --max-years 5

# 2. Run portfolio optimization using cached data
python scripts/run_optimizer.py --use-cache --min 1000 --max 25000

# 3. Optimize trading criteria
python scripts/optimize_criteria.py --mode optimize --period 3y --metric sharpe_ratio
```

### Development and Testing

```bash
# Check code quality
./scripts/lint.sh

# Fix formatting issues
./scripts/lint.sh fix

# Run basic monitoring
python scripts/run_monitoring.py --timeout 60
```

### Regional Market Analysis

```bash
# Split eToro tickers by region
python scripts/split_etoro_by_region.py

# Then use the main trade.py application to analyze regional markets
```
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

## Main Application

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

- **O: Monitoring Dashboard**
  - Launch the monitoring and observability dashboard
  - Real-time metrics, health checks, and performance stats
  - Visual charts for request rates, response times, and memory usage

## Trading Classification Criteria

The system uses these criteria to classify stocks into four categories:

### 🟡 INCONCLUSIVE
- Low confidence due to insufficient analyst coverage (< 5 price targets or < 5 analyst ratings)

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
- Beta data available: 0.25 < Beta ≤ 2.5
- PE Forward (PEF) and PE Trailing (PET) data available
- Improving earnings outlook (PEF < PET) OR Negative trailing P/E
- Positive earnings projection (0.5 < PEF ≤ 45.0)
- Reasonable valuation relative to growth (PEG < 2.5) - if PEG data available
- Acceptable short interest (SI ≤ 1.5%) - if SI data available
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

# Market performance tracking - multiple time periods available
python -m yahoofinance.analysis.performance weekly    # Week-over-week comparison
python -m yahoofinance.analysis.performance monthly   # Month-over-month comparison
python -m yahoofinance.analysis.performance ytd       # Year-to-date (Dec 31 to now)
python -m yahoofinance.analysis.performance mtd       # Month-to-date (prev month end to now)
python -m yahoofinance.analysis.performance portfolio # Portfolio metrics

# Track both market and portfolio performance together
python -m yahoofinance.analysis.performance all
```

For detailed documentation on the performance module, see [yahoofinance/analysis/PERFORMANCE_README.md](yahoofinance/analysis/PERFORMANCE_README.md).

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

### Monitoring & Observability
```bash
# Start the monitoring dashboard (with timeout to prevent hanging)
python scripts/run_monitoring.py --timeout 60 --max-updates 5

# Generate example monitoring data (simplified version)
python scripts/monitoring_examples_limited.py

# Generate simple test data
python scripts/simple_monitoring_test.py

# Static dashboard generation (no server)
python scripts/simple_dashboard.py

# Custom dashboard options
python scripts/run_monitoring.py --port 8080 --refresh 15 --export-interval 30 --timeout 120 --max-updates 10
```

For detailed documentation on the monitoring system, see [scripts/MONITORING_README.md](scripts/MONITORING_README.md).

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

## Monitoring and Observability System

The application includes a comprehensive monitoring system that provides visibility into:

- **API Performance**: Request rates, response times, and error rates
- **Component Health**: System and component health status
- **Resource Usage**: Memory and CPU tracking
- **Circuit Breakers**: API failure detection and protection
- **Alerts**: Threshold-based alerting for key metrics

The monitoring dashboard provides real-time visualization of these metrics, making it easy to identify issues and track system performance.

```bash
# Launch the monitoring dashboard (with timeout)
python scripts/run_monitoring.py --timeout 60 --max-updates 5

# View in browser at http://localhost:8000/dashboard.html
# Or view directly at yahoofinance/output/monitoring/dashboard.html
```

**Note:** The dashboard is designed to run for a limited time to prevent resource exhaustion, then can be accessed statically. For long-running monitoring, consider using the script with appropriate timeout values or the simpler alternatives described in the [Monitoring README](scripts/MONITORING_README.md).

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

### Development Setup

For development, we provide tools to ensure code quality and consistency:

```bash
# Quick setup for development environment 
python scripts/setup_dev_environment.py

# Or set up manually:
pip install -r dev-requirements.txt
pre-commit install
```

#### Docker Development Environment

For a consistent development environment, you can use Docker:

```bash
# Build and run the application
docker-compose up etorotrade

# Run tests in Docker
docker-compose up tests

# Run a specific command in the Docker container
docker-compose run etorotrade python -m yahoofinance.analysis.portfolio
```

### CI/CD and Testing Tools

We've implemented a lightweight CI/CD approach suitable for locally-run projects with GitHub integration:

```bash
# Run all tests and benchmarks
./run_tests.sh --all

# Run specific test types
./run_tests.sh --unit        # Run unit tests
./run_tests.sh --integration # Run integration tests
./run_tests.sh --memory      # Run memory leak tests
./run_tests.sh --performance # Run performance benchmarks
./run_tests.sh --priority    # Test priority rate limiting
./run_tests.sh --monitoring  # Test monitoring components
```

#### Version Management

```bash
# Tag a new version
./tag_version.sh 1.0.0 "Initial stable release"
```

#### GitHub Integration

The repository includes:
- GitHub Actions workflow for automated testing
- Issue and feature request templates
- Pre-commit hooks configuration

See [CI_CD.md](CI_CD.md) for complete documentation on the CI/CD setup.

### Code Quality Tools

```bash
# Run all code quality checks
make lint

# Auto-fix issues where possible
make lint-fix
```

### Environment Variables

You can configure the application behavior using environment variables:

- `ETOROTRADE_LOG_LEVEL`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `ETOROTRADE_LOG_FILE`: Path to log file (defaults to `logs/yahoofinance.log`)
- `ETOROTRADE_DEBUG`: Set to `true` to enable debug mode with more verbose logging
- `YAHOOFINANCE_LOG_LEVEL`: Set logging level for the yahoofinance package
- `YAHOOFINANCE_DEBUG`: Set to `true` to enable debug mode for the yahoofinance package
- `YAHOOFINANCE_MONITORING_ENABLED`: Set to `true` to enable monitoring system (default: true)
- `YAHOOFINANCE_MONITORING_EXPORT_INTERVAL`: Set export interval in seconds (default: 60)

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

## Technical Documentation

For comprehensive technical documentation on the architecture, design patterns, and implementation details, please refer to:

- [CLAUDE.md](CLAUDE.md) - Technical reference for developers
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [scripts/MONITORING_README.md](scripts/MONITORING_README.md) - Monitoring system documentation

## Real-World Investment Performance

I personally use this script to power my eToro investment decisions. For real-world results and validation of this approach, you can follow or copy my eToro portfolio:

👉 [@plessas on eToro](https://www.etoro.com/people/plessas)

---

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
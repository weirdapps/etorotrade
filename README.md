# etorotrade 👨🏻‍💻 An Investment Analysis Tool

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=coverage)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

**Data-driven investment decisions powered by analyst consensus and financial metrics**

![eToro Trade Analysis Tool](docs/assets/etorotrade.png)

etorotrade is a Python-based investment analysis system that helps you make smarter trading decisions by analyzing financial data, analyst ratings, and technical indicators from Yahoo Finance. Whether you're managing a portfolio or searching for new opportunities, etorotrade provides actionable insights with clear BUY, SELL, or HOLD recommendations.

## 🎩 Features

- **Portfolio Analysis**: Identify risks and opportunities in your current holdings
- **Market Screening**: Discover promising stocks across US, European, and Chinese markets
- **Multi-Asset Support**: Analyze stocks, ETFs, commodities (Gold, Oil, Silver), and cryptocurrencies
- **Trade Recommendations**: Get clear BUY, SELL, or HOLD guidance with detailed reasoning
- **News & Sentiment**: Track market-moving news and sentiment analysis
- **Insider Activity**: Follow institutional and insider transactions
- **Backtesting**: Test your strategies against historical data to optimize performance
- **Performance Tracking**: Monitor how your portfolio and markets are performing

## 🏁 Quick Start

```bash
# Clone and setup
git clone https://github.com/weirdapps/etorotrade
cd etorotrade
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install -r requirements.txt

# Run the main application
python trade.py
```

After running, you'll be prompted to select an analysis type:

```
Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)?
```

Simply enter the letter that corresponds to your desired analysis type.

### 🎯 Portfolio Analysis from eToro

For Portfolio analysis (P), you can now fetch your live portfolio data directly from eToro:

1. Select **P** (Portfolio)
2. Choose **E** (Use existing portfolio file) or **N** (Download new portfolio from eToro)

**Features:**
- 🔄 Automatic ticker format fixing for Yahoo Finance compatibility
- 🏢 Hong Kong tickers properly formatted (e.g., `700.HK` → `0700.HK`)
- 💰 Cryptocurrency symbols get USD suffix (e.g., `BTC` → `BTC-USD`)
- 📊 Positions grouped by symbol with aggregated metrics

**Setup eToro API Access:**
Add your eToro API credentials to the `.env` file:
```bash
ETORO_API_KEY=your-etoro-api-key
ETORO_USER_KEY=your-etoro-user-key
ETORO_USERNAME=your-etoro-username
```

## 💲 Trading Criteria

etorotrade uses a sophisticated classification system based on financial metrics, analyst consensus, and technical indicators. **All trading criteria are centralized in a single configuration file** for consistency across all analysis types.

### 🟢 BUY Recommendations
A stock must meet ALL of these criteria:
- **Strong Upside**: 20%+ potential upside
- **Analyst Consensus**: 85%+ buy ratings
- **Reasonable Volatility**: Beta between 0.25 and 2.5
- **Attractive Valuation**: 
  - Improving earnings outlook (Forward P/E < Trailing P/E OR Trailing P/E ≤ 0)
  - Reasonable Forward P/E (0.5 < PEF ≤ 45.0)
  - Good growth-adjusted value (PEG < 2.5)
- **Limited Risk Factors**: 
  - Low short interest (≤ 1.5%)
  - Strong expected return (≥ 15.0)

### 🔴 SELL Signals
A stock triggers a SELL if ANY of these warning signs appear:
- **Limited Upside**: Less than 5% upside potential
- **Weak Analyst Support**: Less than 65% buy ratings
- **Deteriorating Earnings**: Forward P/E > Trailing P/E (worsening outlook)
- **Overvaluation**: Forward P/E > 50.0 or PEG > 3.0
- **High Risk Factors**: Short interest > 2% or Beta > 3.0
- **Poor Expected Return**: EXRET < 5.0

### ⚪ HOLD Recommendations
- Passes confidence thresholds but doesn't meet full BUY or SELL criteria
- May have mixed signals or be fairly valued at current price

### 🟡 INCONCLUSIVE Classification
- Insufficient analyst coverage (< 5 price targets or < 5 analyst ratings)
- Not enough data for confident decision-making

### 🎯 Robust Target Price Mechanism

etorotrade features a sophisticated **quality-validated target price system** that goes beyond simple analyst medians to provide more reliable upside calculations:

#### Quality Assessment System
Every stock's analyst price targets are automatically graded on a **5-tier quality scale**:

- **Grade A (85-100)**: Excellent consensus with tight target ranges
- **Grade B (70-84)**: Good quality with minor disagreements
- **Grade C (55-69)**: Moderate quality with some outliers
- **Grade D (40-54)**: Poor quality requiring manual review
- **Grade F (<40)**: Excluded due to unreliable data

#### Robustness Scoring Factors
The system evaluates price target quality using multiple criteria:

- **Spread Analysis**: Penalizes excessive target ranges (>50% concerning, >75% severe)
- **Outlier Detection**: Identifies extreme predictions that skew averages
- **Consensus Validation**: Rewards tight agreement between mean and median
- **Coverage Assessment**: Considers analyst count and expertise level
- **Price Reasonableness**: Flags targets >300% from current price

#### Trading Decision Impact
- **Grades A-C**: Uses quality-validated median target with confidence indicators
- **Grade D**: Still participates in trading but flagged for manual review
- **Grade F**: Excluded from buy/sell recommendations entirely
- **Fallback Protection**: Reverts to simple median if robust calculation fails

The robust mechanism **prioritizes data quality over raw numbers**, ensuring trading decisions are based on reliable analyst consensus rather than potentially misleading outliers.

### ⚙️ Customizing Trading Criteria
All trading criteria can be customized by editing a single file: `yahoofinance/core/trade_criteria_config.py`

This centralized configuration ensures consistency across:
- ACT column values (B/S/H/I)
- Color coding (green/red/yellow highlighting)
- Buy/Sell/Hold opportunity filtering
- All recommendation outputs

Example customization:
```python
# In yahoofinance/core/trade_criteria_config.py
class TradingCriteria:
    # Make BUY criteria more aggressive
    BUY_MIN_UPSIDE = 25.0              # Require 25% upside (was 20%)
    BUY_MIN_BUY_PERCENTAGE = 90.0      # Require 90% buy rating (was 85%)
    
    # Make SELL criteria more conservative  
    SELL_MAX_UPSIDE = 3.0              # Sell if upside < 3% (was 5%)
```

### 💰 Position Size Calculation
The system automatically calculates optimal position sizes based on market cap and expected return:
- Formula: `position_size = market_cap * EXRET / 5,000,000,000`
- Uses default EXRET values (10-15%) when actual EXRET is unavailable
- Result is rounded up to the nearest thousand with minimum size of $1,000

## 📁 Input Files

Create or edit these CSV files in the `yahoofinance/input/` directory:

```
# portfolio.csv - Your current holdings
symbol,shares,cost,date
AAPL,10,150.25,2022-03-15
MSFT,5,280.75,2022-04-20

# market.csv - General market watchlist
symbol,sector
AAPL,Technology
MSFT,Technology
```

Pre-populated files include:
- `etoro.csv`: Tickers available to invest on eToro (GR account)
- `usa.csv`, `europe.csv`, `china.csv`: Regional market lists
- `notrade.csv`: Tickers to exclude from recommendations

## 🧰 Analysis Tools

### Portfolio Analysis
```bash
# Interactive analysis
python trade.py  # Then select "P" for Portfolio Analysis

# Non-interactive for scripts/automation
echo "p\ne" | python trade.py  # P for portfolio, E for existing file
```

### Market Screening
```bash
# Find opportunities in specific markets
python trade.py  # Then select "M" and choose a market

# Or analyze eToro-available stocks
python trade.py  # Then select "E" for eToro Market Analysis
```

### Trade Recommendations
```bash
# Get actionable BUY/SELL/HOLD guidance
python trade.py  # Then select "T" and choose recommendation type

# Run directly from command line (for BUY recommendations)
echo "t\nb" | python trade.py
```

### Monitoring Dashboard
```bash
# Start the basic monitoring dashboard with a timeout
python scripts/run_monitoring.py --timeout 60 --max-updates 5

# Enhanced monitoring with health endpoints and structured logging
python scripts/run_enhanced_monitoring.py --timeout 300 --health-port 8081
```

### Portfolio Optimization
```bash
# Download historical data for portfolio optimization
python scripts/download_portfolio_data.py --max-years 5 --batch-size 10

# Run portfolio optimizer with custom position size constraints
python scripts/run_optimizer.py --min 1000 --max 25000 --periods 1 3 5 --use-cache

# Optimize trading criteria with backtesting
python scripts/optimize_criteria.py --mode optimize --period 2y --metric sharpe_ratio
```

### Utility Scripts
```bash
# Run code quality checks
./scripts/lint.sh          # Check code quality
./scripts/lint.sh fix      # Fix formatting issues automatically

# Split eToro tickers by region
python scripts/split_etoro_by_region.py  # Creates usa.csv, europe.csv, and china.csv
```

### Specialized Analysis
```bash
# Analyst ratings and recommendations
python -m yahoofinance.analysis.analyst

# News with sentiment analysis
python -m yahoofinance.analysis.news

# Insider transactions
python -m yahoofinance.analysis.insiders

# Performance tracking
python -m yahoofinance.analysis.performance weekly    # Week-over-week
python -m yahoofinance.analysis.performance monthly   # Month-over-month
python -m yahoofinance.analysis.performance portfolio # Portfolio metrics
```

## 📝 Output Examples

The analysis generates both CSV files and interactive HTML dashboards:

- **Portfolio Analysis**: `yahoofinance/output/portfolio.csv` + HTML dashboard
- **Market Analysis**: `yahoofinance/output/market.csv` + HTML dashboard
- **Trade Recommendations**: 
  - `yahoofinance/output/buy.csv`
  - `yahoofinance/output/sell.csv`
  - `yahoofinance/output/hold.csv`

## 🚢 Docker Support

For a consistent environment or deployment:

```bash
# Run the application in Docker
docker-compose up etorotrade

# Run a specific analysis in Docker
docker-compose run etorotrade python -m yahoofinance.analysis.portfolio
```

## 🔍 Configuration

Customize behavior with environment variables:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export ETOROTRADE_LOG_LEVEL=DEBUG

# Enable debug mode
export ETOROTRADE_DEBUG=true

# Configure log file location
export ETOROTRADE_LOG_FILE=logs/custom.log
```

## 🌟 Real-World Results

I personally use this tool to power my own eToro investment decisions. For real-world validation of this approach, you can follow or copy my eToro portfolio:

👉 [@plessas on eToro](https://www.etoro.com/people/plessas)

---

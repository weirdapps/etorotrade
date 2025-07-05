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

### 💰 Intelligent Position Sizing

etorotrade features a sophisticated position sizing system that calculates optimal trade sizes based on portfolio allocation, risk management, and expected returns.

#### Position Sizing Strategy
- **Portfolio Value**: $450,000 (configurable in `yahoofinance/core/config.py`)
- **Base Position**: 0.5% of portfolio = $2,250 for standard opportunities  
- **High Conviction**: 2% of portfolio = $9,000 for exceptional opportunities
- **Position Limits**: $1,000 minimum, $40,000 maximum (8.9% max allocation)

#### Smart Sizing Logic
1. **Expected Return Adjustment**: Higher EXRET = larger positions
   - EXRET ≥ 15%: High conviction multiplier (2-4x base position)
   - EXRET 10-15%: Moderate increase (1.5-2x base position)  
   - EXRET < 10%: Standard or reduced position

2. **Market Cap Scaling**: Position size scales with company size
   - Large cap (>$50B): Can support larger positions
   - Mid cap ($10-50B): Standard scaling
   - Small cap (<$10B): Reduced positions for higher risk

3. **Risk Management**: Automatic scaling for volatility and other risk factors

#### Display Format
Position sizes are shown in the SIZE column with intuitive formatting:
- $2,000 → "2k"
- $7,500 → "7.5k" 
- $15,000 → "15k"

#### Exclusions
- ETFs and commodities: No position sizing (SIZE shows "--")
- Stocks under $500M market cap: Excluded for liquidity concerns
- Missing data: No position size without EXRET or market cap

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

### Core Trading Analysis
```bash
# Interactive analysis with clean progress display
python trade.py  # Then select:
                 # P - Portfolio Analysis
                 # M - Market Screening  
                 # E - eToro Market Analysis
                 # T - Trade Recommendations
                 # I - Manual ticker input

# Non-interactive automation examples
echo "p\ne" | python trade.py  # Portfolio analysis with existing file
echo "t\nb" | python trade.py  # BUY recommendations
echo "m" | python trade.py     # Market screening
```

**Recent Improvements (2025-01-05)**:
- ✅ **Clean Progress Display**: Errors collected and shown in summary at end
- ✅ **Position Sizing Fixed**: SIZE column now shows actual calculated values (2k, 7.5k, etc.)
- ✅ **Silent Processing**: yfinance errors and rate limit warnings suppressed during progress
- ✅ **Connection Pooling**: HTTP performance optimized with shared session management

### Monitoring Dashboard
```bash
# Start the basic monitoring dashboard with a timeout
python tools/run_monitoring.py --timeout 60 --max-updates 5

# Enhanced monitoring with health endpoints and structured logging
python tools/run_enhanced_monitoring.py --timeout 300 --health-port 8081
```

### Portfolio Optimization
```bash
# Download historical data for portfolio optimization
python tools/download_portfolio_data.py --max-years 5 --batch-size 10

# Run portfolio optimizer with custom position size constraints
python tools/run_optimizer.py --min 1000 --max 25000 --periods 1 3 5 --use-cache

# Optimize trading criteria with backtesting
python tools/optimize_criteria.py --mode optimize --period 2y --metric sharpe_ratio
```

### Utility Scripts
```bash
# Run code quality checks
./tools/lint.sh          # Check code quality
./tools/lint.sh fix      # Fix formatting issues automatically

# Split eToro tickers by region
python tools/split_etoro_by_region.py  # Creates usa.csv, europe.csv, and china.csv
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

# etorotrade 👨🏻‍💻 An Investment Analysis Tool

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=coverage)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

**Data-driven investment decisions powered by analyst consensus and financial metrics**

![eToro Trade Analysis Tool](docs/assets/etorotrade.png)

etorotrade is a comprehensive Python-based investment analysis system that helps you make smarter trading decisions by analyzing financial data, analyst ratings, and technical indicators from Yahoo Finance. The system features a sophisticated hybrid provider architecture, comprehensive dual-listed stock support, and advanced trading criteria to provide actionable insights with clear BUY, SELL, or HOLD recommendations.

## 🎩 Features

- **Portfolio Analysis**: Identify risks and opportunities in your current holdings
- **Market Screening**: Discover promising stocks across US, European, and Chinese markets
- **Multi-Asset Support**: Analyze stocks, ETFs, commodities (Gold, Oil, Silver), and cryptocurrencies
- **Trade Recommendations**: Get clear BUY, SELL, or HOLD guidance with detailed reasoning
- **Dual-Listed Stock Support**: Comprehensive handling of stocks listed on multiple exchanges (e.g., ASML.NV)
- **Hybrid Data Provider**: Intelligent fallback system combining YFinance and YahooQuery for maximum data reliability
- **Advanced Trading Criteria**: Sophisticated multi-tier criteria system (Value, Growth, Bets tiers)
- **News & Sentiment**: Track market-moving news and sentiment analysis
- **Insider Activity**: Follow institutional and insider transactions
- **Backtesting**: Test your strategies against historical data to optimize performance
- **Performance Tracking**: Monitor how your portfolio and markets are performing

## 🔄 Recent Improvements

**Latest (2025-08-11)**:
- **HTML Action Display Fix**: Fixed critical bug where "I" (Inconclusive) actions were incorrectly displayed as "H" (Hold) in HTML output
  - Updated HTML generator validation in `yahoofinance/presentation/html.py` to include "I" and "" actions
  - TSM and other stocks with insufficient analyst coverage now correctly display as "I" with yellow styling
  - Fix applies to all trade options and sub-options without special case handling
- **Codebase Organization**: Moved portfolio analysis scripts to `tools/` directory for better organization
- **Test File Cleanup**: Removed temporary test files from output directory for production readiness

**Previous (2025-01-08)**:
- **Portfolio Performance System**: Comprehensive portfolio tracking with market comparison
  - Real-time market indices comparison (DJI30, SP500, NQ100)
  - Multiple time horizons (Today, This Week, This Month, YTD, Annualized)
  - Advanced portfolio metrics (Alpha, Beta, Sharpe, Sortino ratios)
  - Geographic and sector allocation analysis
- **Production Ready Codebase**: Cleaned and organized codebase with all tests passing
- **Enhanced Date Logic**: Fixed monthly date calculations and improved trading day handling
- **Error Handling**: Comprehensive FutureWarning fixes for yfinance API integration

**Previous**:
- **Fixed ASML Ticker Display Issue**: Resolved ticker normalization in display pipeline to show proper tickers (e.g., "ASML.NV" instead of "ASML")
- **Enhanced Test Coverage**: Comprehensive test suite with vectorized action calculation tests
- **Code Cleanup**: Removed legacy files and improved codebase organization
- **Improved Error Handling**: Better handling of edge cases in trading criteria evaluation

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

## 💲 Three-Tier Trading System

etorotrade uses a sophisticated **three-tier trading system** that classifies stocks by market cap and applies appropriate risk-adjusted criteria. This approach recognizes that large-cap blue chips require different analysis than small-cap speculative plays.

### 🏗️ Market Cap Tier Classification

**VALUE Tier (≥$100B)**: Large-cap quality companies
- **Philosophy**: Conservative approach for established blue chips
- **Examples**: Apple, Microsoft, Google, Amazon

**GROWTH Tier ($5B-$100B)**: Mid-cap growth companies  
- **Philosophy**: Balanced risk/reward for established growers
- **Examples**: Most established companies with growth potential

**BETS Tier (<$5B)**: Small-cap speculative positions
- **Philosophy**: Higher bar for speculative investments
- **Examples**: Small-cap growth stocks, emerging companies

**M Column**: Every output shows tier classification (V/G/B) for instant risk assessment

### 🟢 BUY Recommendations (Tier-Specific)

**VALUE Tier BUY** (≥$100B market cap):
- **Conservative Upside**: 15%+ potential upside (reasonable for large-caps)
- **Analyst Consensus**: 70%+ buy ratings (strong support needed)  
- **Expected Return**: ≥10% (balanced threshold for stability)
- **Risk Management**: Higher position size tolerance due to stability

**GROWTH Tier BUY** ($5B-$100B market cap):
- **Standard Upside**: 20%+ potential upside
- **Analyst Consensus**: 75%+ buy ratings
- **Expected Return**: ≥15% (standard requirement)
- **Risk Management**: Balanced position sizing and thresholds

**BETS Tier BUY** (<$5B market cap):
- **Aggressive Upside**: 25%+ potential upside (high returns for risk)
- **Strong Consensus**: 80%+ buy ratings (conviction required)
- **High Return**: ≥20% expected return (compensation for risk)
- **Risk Management**: Smaller positions, stricter fundamentals

**Common BUY Requirements** (All Tiers):
- **Reasonable Volatility**: Beta between 0.25 and 3.0
- **Attractive Valuation**: Forward P/E 0.5-65.0, Trailing P/E required
- **Quality Factors**: PEG < 2.5, Low short interest (if available)
- **Sufficient Coverage**: ≥5 analyst ratings and ≥5 price targets

### 🔴 SELL Signals (Tier-Specific)

**VALUE Tier SELL** (≥$100B market cap):
- **Limited Upside**: <5% upside potential
- **Weak Support**: <50% buy ratings
- **Poor Returns**: <5% expected return

**GROWTH Tier SELL** ($5B-$100B market cap):
- **Limited Upside**: <8% upside potential  
- **Weak Support**: <60% buy ratings
- **Poor Returns**: <8% expected return

**BETS Tier SELL** (<$5B market cap):
- **Limited Upside**: <12% upside potential
- **Weak Support**: <70% buy ratings
- **Poor Returns**: <10% expected return

**Common SELL Triggers** (All Tiers):
- **Overvaluation**: Forward P/E >65.0, PEG >3.0, or excessive PE expansion
- **Poor Fundamentals**: Earnings growth <-15%, Price performance <-35%
- **High Risk**: Short interest >3%, Beta >3.0, Expected return <2.5%

### ⚪ HOLD & 🟡 INCONCLUSIVE
- **HOLD**: Passes confidence thresholds but doesn't meet tier-specific BUY/SELL criteria
- **INCONCLUSIVE**: Insufficient analyst coverage (<5 targets or ratings)

## 📊 Enhanced Data Columns

etorotrade now includes additional fundamental analysis columns for more comprehensive stock evaluation:

### New Data Columns
- **EG (Earnings Growth)**: Year-over-year earnings growth percentage from quarterly data
- **PP (Price Performance)**: 3-month price performance showing recent momentum (shows `--` when data unavailable)
- **Enhanced Position Sizing**: Updated to consider high conviction criteria

### Data Availability Notes
- **PP Column**: Shows `--` when price performance data is unavailable from Yahoo Finance API
- **Common for**: ETFs, crypto, international stocks, and smaller stocks
- **This is expected behavior** - the system correctly shows `--` instead of misleading information

### High Conviction Position Sizing
The system now identifies high conviction opportunities using multiple criteria:
- **High Conviction Criteria**: EG >15% AND PP >0% AND EXRET >20%
- **Position Range**: 0.5% to 10% of portfolio ($2,250 to $45,000)
- **Smart Scaling**: Higher conviction = larger position sizes

### 🎯 Dual-Listed Stock Handling (New 2025-01-28)

etorotrade now features **intelligent dual-listed stock normalization** that handles stocks trading on multiple exchanges, ensuring consistent analysis and preventing duplicate portfolio positions.

### Key Features
- **Automatic Normalization**: US tickers (ADRs, cross-listings) are automatically converted to their original exchange tickers
- **Portfolio Deduplication**: Prevents suggesting stocks you already own under different ticker symbols  
- **Geographic Risk Assessment**: Applies correct geographic multipliers based on the original exchange
- **Consistent Display**: All outputs use the canonical ticker for clarity

### Supported Dual-Listed Stocks
The system handles major dual-listings including:

**European Stocks with US ADRs:**
- NVO → NOVO-B.CO (Novo Nordisk)
- SNY → SAN.PA (Sanofi)  
- ASML → ASML.NV (ASML Netherlands)
- SHEL → SHEL.L (Shell London)

**Asian Stocks with US ADRs:**
- JD → 9618.HK (JD.com)
- BABA → 9988.HK (Alibaba)
- TCEHY → 0700.HK (Tencent)

**Share Class Normalization:**
- GOOGL → GOOG (Google Class C as main ticker)

### Portfolio Impact
If you own **NVO** (Novo Nordisk ADR), the system will:
1. Display it as **NOVO-B.CO** in all outputs
2. Apply **European (0.75x)** geographic risk multiplier
3. **Prevent** suggesting NOVO-B.CO as a buy opportunity
4. Use the **same analysis** whether you input NVO or NOVO-B.CO

*For complete documentation, see: [docs/ticker-normalization.md](docs/ticker-normalization.md)*

## 🎯 Robust Target Price Mechanism

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
- M column tier classification (V/G/B)
- Color coding (green/red/yellow highlighting)
- Buy/Sell/Hold opportunity filtering
- All recommendation outputs

Example tier customization:
```python
# In yahoofinance/core/trade_criteria_config.py
class TradingCriteria:
    # Adjust market cap tier boundaries
    VALUE_TIER_MIN_CAP = 50_000_000_000   # $50B threshold (was $100B)
    GROWTH_TIER_MIN_CAP = 2_000_000_000   # $2B threshold (was $5B)
    
    # Customize VALUE tier BUY criteria (example: more conservative)
    VALUE_BUY_MIN_UPSIDE = 10.0           # Lower upside needed (current: 15%)
    VALUE_BUY_MIN_BUY_PERCENTAGE = 65.0   # Lower consensus (current: 70%)
    
    # Customize BETS tier BUY criteria (example: more aggressive)
    BETS_BUY_MIN_UPSIDE = 30.0            # Higher upside required (current: 25%)
    BETS_BUY_MIN_BUY_PERCENTAGE = 85.0    # Higher consensus (current: 80%)
```

### 💰 Intelligent Position Sizing (Updated 2025-01-27)

etorotrade features a sophisticated position sizing system that calculates optimal trade sizes based on portfolio allocation, risk management, and expected returns with tier-based and geographic adjustments.

#### Position Sizing Strategy
- **Portfolio Value**: $450,000 (configurable in `yahoofinance/core/config.py`)
- **Base Position**: 0.5% of portfolio = $2,250 for standard opportunities  
- **Position Limits**: $1,000 minimum, $40,000 maximum (8.9% max allocation)
- **Extended EXRET Range**: Multipliers from 0.5x to 5.0x for better differentiation

#### Enhanced Sizing Logic
1. **EXRET-Based Scaling**: Extended multiplier range for better opportunity differentiation
   - EXRET ≥ 40%: Exceptional opportunity (5.0x multiplier)
   - EXRET ≥ 30%: High opportunity (4.0x multiplier)
   - EXRET ≥ 25%: Good opportunity (3.0x multiplier)
   - EXRET ≥ 20%: Standard opportunity (2.0x multiplier)
   - EXRET ≥ 15%: Lower opportunity (1.5x multiplier)
   - EXRET ≥ 10%: Base position (1.0x multiplier)
   - EXRET < 10%: Conservative (0.5x multiplier)

2. **Tier-Based Market Cap Scaling**: Rewards stability and appropriate risk
   - **VALUE tier** (≥$100B): 2.5x multiplier (enhanced for stability)
   - **GROWTH tier** ($5B-$100B): 1.5x multiplier (standard allocation)
   - **BETS tier** (<$5B): 0.5x multiplier (conservative small-cap sizing)

3. **Geographic Risk Management**: Concentration risk mitigation (Updated 2025-01-28)
   - **Hong Kong** (.HK): 0.5x multiplier (enhanced reduction for overexposure)
   - **Europe** (.L, .PA, .DE, etc.): 0.75x multiplier (moderate reduction)
   - **All other markets**: 1.0x multiplier (no adjustment)

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
                 # P - Portfolio Analysis (comprehensive with dual-listed handling)
                 # M - Market Screening (across US/EU/Asia markets)
                 # E - eToro Market Analysis
                 # T - Trade Recommendations (BUY/SELL/HOLD with tier-based criteria)
                 # I - Manual ticker input (supports all exchange formats)

# Non-interactive automation examples
echo "p\ne" | python trade.py  # Portfolio analysis with existing file
echo "t\nb" | python trade.py  # BUY recommendations with position sizing
echo "t\ns" | python trade.py  # SELL recommendations with risk assessment
echo "t\nh" | python trade.py  # HOLD recommendations
echo "m" | python trade.py     # Market screening with geographic filtering

# Command line arguments (fastest execution)
python trade.py -o t -t b      # BUY opportunities
python trade.py -o t -t s      # SELL opportunities  
python trade.py -o t -t h      # HOLD opportunities
python trade.py -o p -t n      # Portfolio analysis with new eToro download
python trade.py -o m -t 50     # Market analysis for 50 stocks
python trade.py -o i -t AAPL,MSFT,NOVO-B.CO  # Manual input analysis (auto-normalizes tickers)
```

**Recent Improvements (2025-01-28)**:
- 🎯 **Dual-Listed Stock System**: Intelligent ticker normalization prevents duplicate holdings (NVO→NOVO-B.CO, GOOGL→GOOG)
- 🌍 **Geographic Risk Update**: HK stocks now use 0.5x multiplier (down from 0.75x) for better risk management
- 🧹 **Codebase Cleanup**: Removed 3 legacy analysis scripts, consolidated functionality into main analysis engine
- 📊 **Enhanced Portfolio Filtering**: Prevents suggesting equivalent stocks already owned under different ticker symbols

**Performance Improvements (2025-01-18)**:
- 🚀 **Major Performance Boost**: 127% faster API processing (390 vs 171 tickers/minute)
- ⚡ **Vectorized Operations**: DataFrame calculations now >1.7M rows/second 
- 📊 **Smart Rate Limiting**: Optimized batch sizes (25 vs 10) and reduced delays (0.15s vs 0.3s)
- 🧪 **Comprehensive Testing**: 19 new test cases with performance benchmarks
- 🏗️ **Modular Architecture**: Clean separation into specialized trade_modules
- ✅ **Enterprise Performance**: 56% faster processing for large datasets (100+ tickers)
- 📈 **Performance Monitoring**: Built-in benchmarking tools for real-time metrics
- 🔧 **Trade Analysis Fixed**: Display formatting and HOLD analysis logic corrected
- 📊 **Enhanced Trading Criteria**: More selective BUY signals (20% upside, 15% EXRET) and conservative SELL triggers

**UI/UX Enhancements (2025-01-06)**:
- 🎨 **Ultra-Clean Display**: Completely silent processing with zero debug/info/warning messages
- 📊 **Professional Output**: Tables display cleanly without progress bars or processing noise
- 🚫 **Comprehensive Error Filtering**: Suppressed all irrelevant delisting/earnings/HTTP warnings
- ⚡ **Streamlined Experience**: Focus purely on data with minimal interface distractions
- 🧹 **Production-Ready**: Enterprise-level clean output suitable for automated workflows

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

### Performance Benchmarking
```bash
# Run comprehensive performance benchmark
python tools/performance_benchmark.py

# Example output shows dramatic improvements:
# - API throughput: 390 tickers/minute (vs 171 before optimization)
# - DataFrame processing: >1.7M rows/second with vectorized operations
# - Memory usage: Optimized with reduced copying and efficient calculations
# - Realistic portfolio (50 tickers): <0.002s processing time
```

### Specialized Analysis
```bash
# Analyst ratings and recommendations (with dual-listed normalization)
python -m yahoofinance.analysis.analyst

# News with sentiment analysis
python -m yahoofinance.analysis.news

# Insider transactions (tracks original exchange tickers)
python -m yahoofinance.analysis.insiders

# Performance tracking with geographic risk assessment
python -m yahoofinance.analysis.performance weekly    # Week-over-week
python -m yahoofinance.analysis.performance monthly   # Month-over-month
python -m yahoofinance.analysis.performance portfolio # Portfolio metrics with tier analysis
```

## 📝 Output Examples

The analysis generates both CSV files and interactive HTML dashboards:

- **Portfolio Analysis**: `yahoofinance/output/portfolio.csv` + HTML dashboard
- **Market Analysis**: `yahoofinance/output/market.csv` + HTML dashboard
- **Manual Input Analysis**: `yahoofinance/output/manual.csv` + HTML dashboard
- **Trade Recommendations**: 
  - `yahoofinance/output/buy.csv`
  - `yahoofinance/output/sell.csv`
  - `yahoofinance/output/hold.csv`

## 🚢 Docker Support

For a consistent environment or deployment:

```bash
# Run the application in Docker (infrastructure files in .config/)
docker-compose -f .config/docker/docker-compose.yml up etorotrade

# Run a specific analysis in Docker
docker-compose -f .config/docker/docker-compose.yml run etorotrade python -m yahoofinance.analysis.portfolio

# Build Docker image
docker build -f .config/docker/Dockerfile -t etorotrade .

# Use Makefile for common tasks
make -f .config/Makefile test
make -f .config/Makefile lint
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

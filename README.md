# Market Analysis and Portfolio Management Tool

A Python-based analysis system for stocks, portfolios, and market intelligence using Yahoo Finance data. Features rate limiting, caching, and multiple output formats.

![eToro Trade Analysis Tool](assets/etorotrade.png)

## Quick Start Guide

1. **Setup:**
   ```bash
   # Clone and install
   git clone https://github.com/weirdapps/etorotrade
   cd etorotrade
   python -m venv myenv
   source myenv/bin/activate  # Unix
   pip install -r requirements.txt
   ```

2. **Run Main Application:**
   ```bash
   python trade.py
   ```
   - Choose from menu options:
     - **P**: Portfolio analysis
     - **M**: Market analysis
     - **E**: eToro market analysis
     - **T**: Trading recommendations
     - **I**: Manual ticker input

3. **Run Specific Analysis Modules:**
   ```bash
   # Stock analysis and recommendations
   python trade.py
   
   # Portfolio performance
   python -m yahoofinance.analysis.portfolio
   
   # News with sentiment
   python -m yahoofinance.analysis.news
   
   # Analyst ratings
   python -m yahoofinance.analysis.analyst
   
   # Earnings information
   python -m yahoofinance.analysis.earnings
   
   # Performance tracking
   python -m yahoofinance.analysis.performance
   ```

## Key Features

### Analysis Tools
- **Portfolio Analysis**: Tracking, metrics, position monitoring
- **Market Analysis**: Price targets, recommendations, risk metrics
- **Trade Recommendations**: Buy/sell signals, opportunity ranking
- **Market Intelligence**: News, earnings, economic indicators

### Output Options
- **Console Display**: Color-coded metrics based on analysis
- **CSV Exports**: Comprehensive data in structured format
- **HTML Dashboards**: Visual performance metrics and rankings

## Required Files and Setup

### Input Files
- **portfolio.csv**: Your current portfolio holdings 
- **market.csv**: Market watchlist for analysis
- **etoro.csv**: Tickers available on eToro platform
- **notrade.csv**: Tickers to exclude from recommendations

### Environment Setup
```bash
# Configure environment
cp .env.example .env
# Edit .env with your API keys if needed
```

## File Formats

### portfolio.csv
```
symbol,shares,cost,date
AAPL,10,150.25,2022-03-15
MSFT,5,280.75,2022-04-20
```

### market.csv
```
symbol,sector
AAPL,Technology
MSFT,Technology
```

## Trading Tools

### 1. Ticker Validation
```bash
# Validate tickers against Yahoo Finance API
python -m yahoofinance.analysis.stock validate
```
- Checks tickers against Yahoo Finance API
- Saves valid tickers to `yahoofinance/input/yfinance.csv`
- Improves batch processing reliability

### 2. eToro-Specific Features
- **HK Ticker Support**: Automatically converts `03690.HK` to `3690.HK`
- **Crypto Support**: Standardizes to `-USD` format (e.g., `BTC-USD`)
- **Regional Analysis**: Separate files for USA, Europe, China

## Trading Criteria

The system classifies stocks into four categories:

- 🟡 **INCONCLUSIVE** - Low confidence (< 5 price targets OR < 5 analyst ratings)
- 🔴 **SELL** - ANY of these conditions:
  * < 5% upside OR < 65% buy ratings OR PEF > PET OR PEF > 45.0 OR
  * PEG > 3.0 OR SI > 4% OR Beta > 3.0 OR EXRET < 10.0
- 🟢 **BUY** - ALL of these conditions:
  * ≥ 20% upside AND ≥ 82% buy ratings AND 0.2 < Beta ≤ 3.0 AND
  * PEF < PET (or P/E ≤ 0) AND 0.5 < PEF ≤ 45.0 AND PEG < 3.0 AND SI ≤ 3%
- ⚪ **HOLD** - Stocks passing confidence check but not meeting BUY or SELL criteria

## Key Metrics

### Price Metrics
- **PRICE**: Current stock price
- **TARGET**: Average analyst target
- **UPSIDE**: Target vs current price (%)
- **EXRET**: Expected return (Upside × Buy%)

### Analyst Coverage
- **# T**: Number of price targets
- **% BUY**: Buy rating percentage
- **# A**: Number of ratings

### Valuation
- **PET**: Trailing P/E ratio
- **PEF**: Forward P/E ratio
- **PEG**: Price/Earnings to Growth

### Risk Metrics
- **BETA**: Market volatility comparison
- **SI**: Short interest percentage
- **INS%**: Insider buy percentage

## Advanced Usage

### Custom Analysis with Provider Pattern

```python
from yahoofinance import get_provider

# For synchronous operations
provider = get_provider()

# Get data for a specific ticker
info = provider.get_ticker_info("AAPL")
print(f"Company: {info['name']}")
print(f"Current price: ${info['current_price']}")

# For asynchronous operations
import asyncio
from yahoofinance import get_provider

async def analyze_portfolio():
    provider = get_provider(async_mode=True)
    portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    results = await provider.batch_get_ticker_info(portfolio)
    
    for ticker, data in results.items():
        if data:
            print(f"{ticker}: {data['name']} - ${data['current_price']}")

# Run async function
asyncio.run(analyze_portfolio())
```

## Troubleshooting

- **Rate Limiting**: System implements automatic backoff - wait and retry
- **Invalid Tickers**: Run ticker validation to update valid tickers list
- **Non-US Markets**: Some data may be unavailable - system adapts automatically
- **Input File Errors**: Check CSV formats match examples in documentation

For detailed technical documentation, see CLAUDE.md

## License

This project is licensed under the MIT License - see LICENSE file for details.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
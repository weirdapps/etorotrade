# eToro Trade Analysis Tool 🚀

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=coverage)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

**Professional investment analysis that turns market data into clear BUY/SELL/HOLD recommendations based on institutional-grade algorithms and real analyst consensus.**

## What It Does

This tool analyzes stocks using the same methods used by institutional investors:
- 📊 **Analyst Consensus** - Aggregates recommendations from major banks
- 🎯 **Price Targets** - Calculates upside potential based on analyst targets
- 💰 **Smart Position Sizing** - Suggests position sizes based on risk/reward
- 🌍 **Portfolio X-Ray** - See through ETFs to understand true geographic/sector exposure
- ⚡ **Real-Time Data** - Live market data from Yahoo Finance APIs

## Quick Start (2 Minutes)

```bash
# Clone and setup
git clone https://github.com/weirdapps/etorotrade
cd etorotrade
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate
pip install -r requirements.txt

# Run your first analysis
python trade.py
```

## How to Use

### Interactive Mode (Easiest)
```bash
python trade.py
```
Then choose:
- **P** - Analyze your portfolio
- **M** - Screen the market
- **E** - Analyze eToro market (5500+ stocks)
- **T** - Get trade recommendations (BUY/SELL/HOLD)
- **I** - Analyze specific tickers

### Power User Commands
```bash
# Find BUY opportunities
python trade.py -o t -t b

# Check for SELL signals
python trade.py -o t -t s

# Analyze portfolio with $50K value
python trade.py -o p -pv 50000

# Deep portfolio analysis
python scripts/analyze_geography.py  # Geographic exposure
python scripts/analyze_industry.py   # Sector allocation

# Analyze specific stocks
python trade.py -o i -t AAPL,MSFT,NVDA
```

## Understanding the Output

### Trade Signals
| Signal | Meaning | Action |
|--------|---------|--------|
| **BUY** 🟢 | Strong upside + analyst consensus | Consider opening position |
| **SELL** 🔴 | Overvalued or deteriorating | Consider taking profits |
| **HOLD** 🟡 | Fair value | Keep existing position |
| **INCONCLUSIVE** ⚪ | Insufficient data | Do more research |

### Key Metrics
- **UPSIDE** - % gain to analyst price target
- **%BUY** - % of analysts recommending BUY
- **EXRET** - Expected return (upside × buy%)
- **SIZE** - Suggested position size in USD
- **PP** - 12-month price performance
- **EG** - Earnings growth rate

### Market Cap Tiers
The system uses sophisticated tier-based analysis:
- **MEGA** (≥$500B) - Blue-chip giants like AAPL, MSFT
- **LARGE** ($100-500B) - Established leaders
- **MID** ($10-100B) - Growth companies
- **SMALL** ($2-10B) - Emerging opportunities
- **MICRO** (<$2B) - High risk/reward

## Configuration

### Add Your Portfolio (Optional)
Export from eToro to `yahoofinance/input/portfolio.csv`:
```csv
symbol,totalInvestmentPct,totalNetProfitPct,instrumentDisplayName
AAPL,5.2,12.5,Apple Inc
MSFT,4.8,8.3,Microsoft Corp
```

### Trading Thresholds
Edit `config.yaml` to customize buy/sell criteria by market cap tier and region (US/EU/HK).

## Output Files

The tool generates several CSV and HTML reports in `yahoofinance/output/`:
- `portfolio.csv/html` - Your portfolio analysis
- `market.csv/html` - Market screening results
- `buy.csv/html` - BUY recommendations
- `sell.csv/html` - SELL recommendations

## Real-World Usage

I use this system for my personal eToro investments:
👉 **[@plessas on eToro](https://www.etoro.com/people/plessas)**

## Requirements

- Python 3.8+
- Internet connection for real-time data
- Optional: eToro portfolio export

## Support

- **Issues**: [GitHub Issues](https://github.com/weirdapps/etorotrade/issues)
- **Documentation**: See `docs/` folder for technical details

## License

MIT License - See LICENSE file for details

---
*Built with modern software engineering practices for reliable investment analysis*
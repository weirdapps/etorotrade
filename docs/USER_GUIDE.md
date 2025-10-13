# 📈 eToro Trading Analysis - User Guide

Simple, powerful portfolio analysis for your eToro investments.

**Created by:** [plessas](https://www.etoro.com/people/plessas) - eToro Popular Investor

**What this tool does:** Aggregates analyst recommendations, price targets, and fundamental metrics from 20+ investment banks into a single, clear display—helping you make data-driven investment decisions.

**What this is NOT:** Investment advice. All investment decisions are your own responsibility.

---

## 🚀 Quick Start

### 1. Export Your Portfolio from eToro

**Detailed steps:**

1. **Log into eToro:**
   - Go to [www.etoro.com](https://www.etoro.com)
   - Sign in with your credentials

2. **Navigate to Portfolio:**
   - Click "Portfolio" in the left sidebar
   - You'll see your current holdings

3. **Export to CSV:**
   - Click the **gear icon** (⚙️) in the top right of the portfolio table
   - Select **"Export to CSV"** from the dropdown menu
   - Your browser will download a file named something like `AccountStatement.csv` or `Portfolio.csv`

4. **Prepare the file:**
   ```bash
   # Create input directory if it doesn't exist
   mkdir -p yahoofinance/input

   # Move and rename your downloaded file
   mv ~/Downloads/AccountStatement.csv yahoofinance/input/portfolio.csv
   # Or on Windows:
   # move %USERPROFILE%\Downloads\AccountStatement.csv yahoofinance\input\portfolio.csv
   ```

5. **Verify file location:**
   ```bash
   # On Mac/Linux
   ls -l yahoofinance/input/portfolio.csv

   # On Windows
   dir yahoofinance\input\portfolio.csv
   ```

**File format example:**
```csv
symbol,totalInvestmentPct,totalNetProfitPct,instrumentDisplayName
AAPL,5.2,12.5,Apple Inc
MSFT,4.8,8.3,Microsoft Corporation
GOOGL,3.9,15.2,Alphabet Inc
```

**Required column:** At minimum, the file must have a `symbol` or `ticker` column with stock symbols.

### 2. Run Analysis

**Basic portfolio analysis:**
```bash
python trade.py -o p
```

**What happens:**
- Tool reads your `portfolio.csv`
- Fetches analyst data from Yahoo Finance
- Applies trading criteria from `config.yaml`
- Generates BUY/SELL/HOLD/INCONCLUSIVE signals
- Creates output files

**Expected time:**
- 10 stocks: ~10 seconds
- 20 stocks: ~20 seconds
- 50 stocks: ~1 minute

**First run is slower** (no cache). Subsequent runs within 48 hours use cached data.

### 3. View Results

**Console output:**
```
┌────┬─────────┬────────────┬──────┬───────┬────────┬────────┬──────┬────┬─────┐
│ #  │ TICKER  │ COMPANY    │ CAP  │ PRICE │ TARGET │ UPSIDE │ %BUY │ BS │ ... │
├────┼─────────┼────────────┼──────┼───────┼────────┼────────┼──────┼────┼─────┤
│ 1  │ AAPL    │ Apple Inc  │ 3.1T │ 185.5 │ 210.0  │ 13.2%  │ 76%  │ 🟢B│ ... │
│ 2  │ MSFT    │ Microsoft  │ 2.8T │ 380.2 │ 420.0  │ 10.5%  │ 82%  │ 🟢B│ ... │
│ 3  │ NVDA    │ NVIDIA     │ 1.2T │ 495.3 │ 470.0  │ -5.1%  │ 45%  │ 🔴S│ ... │
└────┴─────────┴────────────┴──────┴───────┴────────┴────────┴──────┴────┴─────┘
```

**Output files in `yahoofinance/output/`:**
- **portfolio.csv** - Full analysis results
- **portfolio.html** - Browser-friendly report with color coding
- **buy.csv** - Stocks meeting BUY criteria
- **sell.csv** - Stocks meeting SELL criteria

**Open HTML report:**
```bash
# Mac
open yahoofinance/output/portfolio.html

# Windows
start yahoofinance\output\portfolio.html

# Linux
xdg-open yahoofinance/output/portfolio.html
```

---

## 📊 Understanding the Output

### Color Coding
- 🟢 **Green (B)**: Strong BUY recommendation
- 🔴 **Red (S)**: Strong SELL recommendation  
- 🟡 **Yellow (H)**: HOLD - wait and see
- ⚪ **Gray (I)**: INCONCLUSIVE - insufficient data

### Key Columns
| Column | Meaning | Notes |
|--------|---------|-------|
| **UPSIDE** | Price target vs current | Higher = more potential |
| **%BUY** | Analyst buy ratings | 80%+ is strong consensus |
| **M** | Market cap tier | MEGA/LARGE/MID/SMALL/MICRO |
| **BS** | Recommendation | B/S/H/I for Buy/Sell/Hold/Inconclusive |

### Market Cap Tiers (5-Tier System)
- **MEGA**: $500B+ companies (e.g., AAPL, MSFT)
- **LARGE**: $100B-$500B companies (e.g., NFLX, DIS)
- **MID**: $10B-$100B companies (e.g., ROKU, SNAP)
- **SMALL**: $2B-$10B companies (smaller growth stocks)
- **MICRO**: <$2B companies (higher risk opportunities)

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# 5-Tier Market Cap System
tiers:
  mega:
    market_cap_min: 500_000_000_000  # $500B+
    min_upside: 10
    min_buy_percentage: 65
  large:
    market_cap_min: 100_000_000_000  # $100B-$500B
    min_upside: 12
    min_buy_percentage: 70
  mid:
    market_cap_min: 10_000_000_000   # $10B-$100B
    min_upside: 15
    min_buy_percentage: 70
  small:
    market_cap_min: 2_000_000_000    # $2B-$10B
    min_upside: 20
    min_buy_percentage: 75
  micro:
    market_cap_min: 0                 # <$2B
    min_upside: 25
    min_buy_percentage: 80

# See config.yaml for full configuration
```

---

## 🔧 Common Commands

### Portfolio Analysis
```bash
# Full portfolio analysis
python trade.py -o p -t e

# Quick portfolio check (no external data)
python trade.py -o p -t n
```

### Market Analysis
```bash
# Market overview (top stocks)
python trade.py -o m -t e

# Market with specific number of stocks
python trade.py -o m -t 20
```

### Trading Recommendations
```bash
# Buy opportunities
python trade.py -o t -t b

# Sell candidates  
python trade.py -o t -t s

# Hold recommendations
python trade.py -o t -t h
```

### Individual Stock Analysis
```bash
# Analyze specific tickers
python trade.py -o i -t AAPL,MSFT,GOOG
```

---

## 🛠️ Troubleshooting

### Quick Fixes

| Issue | Quick Solution |
|-------|----------------|
| Portfolio file not found | Check `yahoofinance/input/portfolio.csv` exists |
| API timeout | Wait and retry, or reduce number of tickers |
| Rate limiting | Wait 1 hour between large analyses |
| All SELL signals | Test with `python trade.py -o i -t AAPL` |
| Slow performance | First run is slower, cache helps next time |

### Detailed Troubleshooting

For comprehensive troubleshooting, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

Topics covered:
- Portfolio file issues
- API connection errors
- Rate limiting
- Incorrect trading signals
- Installation issues
- Performance problems
- Output file problems

---

## 📁 File Structure

```
etorotrade/
├── config.yaml              # Your preferences
├── trade.py                 # Main analysis script
├── yahoofinance/
│   ├── input/
│   │   └── portfolio.csv    # Your eToro export
│   └── output/             # Analysis results
│       ├── portfolio.csv   # Portfolio analysis
│       ├── buy.csv        # Buy opportunities  
│       └── sell.csv       # Sell candidates
└── logs/                   # Error logs
    └── trading_analysis.log
```

---

## 💡 Tips for Best Results

### Data Quality
- Ensure portfolio CSV is recent (exported today)
- Focus on stocks with 5+ analyst ratings
- Higher analyst coverage = more reliable recommendations

### Portfolio Management
- Review SELL recommendations weekly
- Consider BUY opportunities for diversification
- Pay attention to market cap tier classifications

### Performance
- Enable caching for faster subsequent runs
- Run analysis during market hours for latest data
- Use `-t n` flag for quick checks without external API calls

---

## 🆘 Getting Help

### Documentation Resources

- **[README.md](../README.md)** - Project overview and technical specifications
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive problem-solving guide
- **[EXAMPLES.md](EXAMPLES.md)** - Real-world usage scenarios
- **[FAQ.md](FAQ.md)** - Frequently asked questions
- **[TECHNICAL.md](TECHNICAL.md)** - Technical architecture for developers
- **[POSITION_SIZING.md](POSITION_SIZING.md)** - Position sizing methodology

### Common Issues

1. **"Portfolio file not found"**
   - See [TROUBLESHOOTING.md - Portfolio File Issues](TROUBLESHOOTING.md#portfolio-file-issues)

2. **"All stocks showing SELL"**
   - See [TROUBLESHOOTING.md - Incorrect Trading Signals](TROUBLESHOOTING.md#incorrect-trading-signals)

3. **Rate limiting / API timeouts**
   - See [TROUBLESHOOTING.md - Rate Limiting](TROUBLESHOOTING.md#rate-limiting)

4. **"How do I use this for X?"**
   - See [EXAMPLES.md](EXAMPLES.md) for real-world scenarios

5. **General questions**
   - See [FAQ.md](FAQ.md) for answers

### Community Support

**GitHub Issues:**
- Visit [GitHub Issues](https://github.com/weirdapps/etorotrade/issues)
- Search for similar problems
- Create new issue with details

**Before asking for help:**
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Search GitHub issues
3. Review [FAQ.md](FAQ.md)
4. Collect error messages and system info

---

## 💡 Next Steps

**New users:**
1. Complete Quick Start above
2. Read [EXAMPLES.md](EXAMPLES.md) - "Getting Started with eToro"
3. Review [FAQ.md](FAQ.md) - General Questions

**Experienced users:**
1. Explore advanced commands below
2. Customize `config.yaml` thresholds
3. Run geographic/sector analysis scripts

**Developers:**
1. Read [TECHNICAL.md](TECHNICAL.md) for architecture
2. Review test suite: `pytest tests/`
3. Check [CI/CD.md](CI_CD.md) for contribution guidelines

---

*Happy trading! 📈*

**Disclaimer:** This tool provides analysis only, not investment advice. All investment decisions are your own responsibility.
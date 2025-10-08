# 📈 eToro Trading Analysis - User Guide

Simple, powerful portfolio analysis for your eToro investments.

## 🚀 Quick Start

### 1. Export Your Portfolio
1. Log into eToro
2. Go to Portfolio → Export to CSV
3. Save as `portfolio.csv`
4. Place in: `yahoofinance/input/portfolio.csv`

### 2. Run Analysis
```bash
# Analyze your portfolio
python trade.py -o p -t e

# View market opportunities  
python trade.py -o m -t e

# Get buy recommendations
python trade.py -o t -t b
```

### 3. View Results
- **Console**: Color-coded recommendations
- **CSV files**: Saved in `yahoofinance/output/`
- **HTML reports**: Browser-friendly format

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

### "Portfolio file not found"
- Export portfolio from eToro as CSV
- Save as `portfolio.csv` in `yahoofinance/input/`
- Check filename is exactly `portfolio.csv`

### "API timeout" or "Connection error"  
- Check internet connection
- Try again in a few minutes (rate limiting)
- Reduce portfolio size if very large

### "Error parsing CSV"
- Open CSV in Excel/Numbers to check format
- Re-export from eToro if corrupted
- Ensure proper UTF-8 encoding

### Slow performance
- Enable caching in `config.yaml`
- Reduce `max_concurrent_requests` 
- Analyze smaller batches of stocks

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

### Log Files
Check `logs/trading_analysis.log` for detailed error information.

### Common Solutions
1. **Restart**: Try running the command again
2. **Cache**: Delete old cache files if data seems stale
3. **Network**: Check internet connection for API access
4. **Files**: Verify all input files exist and are readable

### Contact
For technical issues, check the error logs first, then review this guide.

---

*Happy trading! 📈*
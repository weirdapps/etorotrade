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
| **M** | Market cap tier | V=Value, G=Growth, B=Bets |
| **BS** | Recommendation | B/S/H/I for Buy/Sell/Hold/Inconclusive |

### Market Cap Tiers
- **V (VALUE)**: $100B+ companies, 15% upside needed for BUY
- **G (GROWTH)**: $5B-$100B companies, 20% upside needed  
- **B (BETS)**: <$5B companies, 25% upside needed

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Trading criteria
trading:
  value_threshold: 100    # VALUE tier threshold ($100B)
  growth_threshold: 5     # GROWTH tier threshold ($5B)
  min_analysts: 5         # Minimum analyst coverage

# Performance  
performance:
  max_concurrent_requests: 10  # API request limit
  cache_ttl_hours: 24         # Cache data for 24 hours

# Output
output:
  display_colors: true        # Enable color coding
  max_display_rows: 50       # Limit console output
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
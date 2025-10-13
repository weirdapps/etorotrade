# Troubleshooting Guide
## eToro Trade Analysis Tool

This guide helps you diagnose and fix common issues when using the eToro Trade Analysis Tool.

---

## Table of Contents

1. [Portfolio File Issues](#portfolio-file-issues)
2. [API Connection Errors](#api-connection-errors)
3. [Rate Limiting](#rate-limiting)
4. [Incorrect Trading Signals](#incorrect-trading-signals)
5. [Installation Issues](#installation-issues)
6. [Performance Issues](#performance-issues)
7. [Output File Problems](#output-file-problems)

---

## Portfolio File Issues

### Error: "Portfolio file not found"

**What you see:**
```
❌ Portfolio file not found

💡 Suggestions:
• Export your eToro portfolio to CSV format
• Place the file in: yahoofinance/input/portfolio.csv
• Make sure the filename is exactly 'portfolio.csv'
```

**How to fix:**

1. **Export from eToro:**
   - Log in to eToro
   - Go to Portfolio section
   - Click "Export" or "Download Portfolio"
   - Save as CSV format

2. **Place in correct location:**
   ```
   etorotrade/
   └── yahoofinance/
       └── input/
           └── portfolio.csv    ← File must be here
   ```

3. **Check filename:**
   - Must be exactly `portfolio.csv` (lowercase)
   - Not `Portfolio.csv` or `portfolio (1).csv`
   - Not inside a subfolder

4. **Verify file exists:**
   ```bash
   # On Mac/Linux
   ls -l yahoofinance/input/portfolio.csv

   # On Windows
   dir yahoofinance\input\portfolio.csv
   ```

### Error: "Error reading CSV file"

**What you see:**
```
❌ Error reading CSV file: yahoofinance/input/portfolio.csv

Details: [specific parsing error]

💡 Suggestions:
• Check the file format is valid CSV
• Try opening in Excel/Numbers to verify
• Re-export from eToro if corrupted
```

**How to fix:**

1. **Open file in Excel/Numbers:**
   - Does it open correctly?
   - Are columns aligned properly?
   - Is there any corrupted data?

2. **Check required columns:**
   ```csv
   symbol,totalInvestmentPct,totalNetProfitPct,instrumentDisplayName
   AAPL,5.2,12.5,Apple Inc
   ```
   - Must have `symbol` column with ticker symbols
   - May have additional eToro-specific columns

3. **Common issues:**
   - Extra commas in company names
   - Missing header row
   - File saved as Excel format (.xlsx) instead of CSV
   - Special characters causing encoding issues

4. **Re-export:**
   - Go back to eToro
   - Export portfolio again
   - Make sure to select CSV format

---

## API Connection Errors

### Error: "Yahoo Finance API error"

**What you see:**
```
❌ Yahoo Finance API error

Details: [connection error details]

💡 Suggestions:
• Check internet connection
• Try again in a few minutes
• Reduce number of tickers if analyzing large portfolio
```

**How to fix:**

1. **Check internet connection:**
   ```bash
   # Test connection to Yahoo Finance
   ping finance.yahoo.com
   ```

2. **Wait and retry:**
   - Yahoo Finance may be temporarily down
   - Wait 5-10 minutes and try again
   - Check [Yahoo Finance status](https://finance.yahoo.com/)

3. **For large portfolios:**
   - Analyze in smaller batches
   - Use specific tickers instead of full market scan
   ```bash
   # Instead of full market analysis
   python trade.py -o m

   # Try specific tickers
   python trade.py -o i -t AAPL,MSFT,GOOGL
   ```

### Error: "Operation timed out"

**What happens:**
- Analysis stops mid-processing
- Long wait times with no progress
- Timeout errors in logs

**How to fix:**

1. **Increase timeout (if using programmatically):**
   - Default timeout is typically 30-60 seconds per ticker
   - Large portfolios may need more time

2. **Check network speed:**
   ```bash
   # Test download speed
   speedtest-cli
   ```

3. **Retry with smaller batch:**
   - Process 10-20 tickers at a time
   - Combine results manually if needed

---

## Rate Limiting

### What is rate limiting?

Yahoo Finance limits API requests to prevent abuse:
- Approximately **2,000 requests per hour**
- Analyzing 100+ tickers may hit this limit
- Tool includes built-in rate limiting to prevent this

### Symptoms:

- Analysis slows down significantly
- "Rate limit" warnings in output
- Some tickers return no data

### How the tool handles it:

1. **Automatic rate limiting:**
   - Max 15 concurrent requests
   - Adaptive delays between batches
   - Exponential backoff on errors

2. **Caching:**
   - 48-hour cache for market data
   - Reduces redundant API calls by ~80%

### If you still hit limits:

1. **Wait before retrying:**
   ```bash
   # Wait 1 hour, then try again
   ```

2. **Use cached data:**
   - Run analysis, let it cache data
   - Subsequent runs will use cache
   - Cache location: in-memory (clears on restart)

3. **Analyze smaller sets:**
   ```bash
   # Split large portfolio into batches
   python trade.py -o i -t AAPL,MSFT,GOOGL    # Batch 1
   python trade.py -o i -t NFLX,DIS,UBER      # Batch 2
   ```

---

## Incorrect Trading Signals

### All stocks showing SELL signal

**Problem:** After running portfolio analysis, everything shows as SELL

**How to diagnose:**

1. **Check data quality:**
   ```bash
   # Test with known stocks
   python trade.py -o i -t AAPL,MSFT
   ```
   - If these show SELL, there's likely a data issue
   - If these show correct signals, portfolio data may be corrupted

2. **Verify analyst data:**
   - Signals require minimum 4 analysts + 4 price targets
   - Stocks with less coverage show "INCONCLUSIVE"

3. **Check thresholds:**
   - Review `config.yaml` for your tier thresholds
   - Different tiers (MEGA, LARGE, MID, SMALL, MICRO) have different criteria

**Common causes:**

1. **Missing price target data:**
   - Yahoo Finance occasionally returns incomplete data
   - Wait and retry
   - Check stock has analyst coverage

2. **Market conditions:**
   - During market downturns, more stocks genuinely show SELL
   - Tool reflects analyst consensus, not market timing

3. **Configuration issues:**
   - Custom `config.yaml` modifications
   - Reset to defaults: backup and restore original config

### INCONCLUSIVE signals for everything

**Problem:** Most/all stocks show "INCONCLUSIVE" instead of BUY/SELL/HOLD

**Cause:** Insufficient analyst coverage

**Requirements for signal generation:**
- Minimum **4 analysts** providing recommendations
- Minimum **4 price targets**

**How to fix:**

1. **Check stock coverage:**
   - Small-cap stocks often lack analyst coverage
   - Stick to large-cap stocks (market cap > $10B)

2. **Verify data availability:**
   ```bash
   # Test with well-covered stock
   python trade.py -o i -t AAPL
   ```
   - AAPL should show 20+ analysts
   - If not, API data may be incomplete

3. **Not a bug:**
   - Tool correctly identifies insufficient data
   - Manual research required for these stocks

---

## Installation Issues

### Error: "ModuleNotFoundError"

**What you see:**
```
ModuleNotFoundError: No module named 'yfinance'
```

**How to fix:**

1. **Verify virtual environment is active:**
   ```bash
   # On Mac/Linux
   source myenv/bin/activate

   # On Windows
   myenv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   pip list | grep yfinance
   ```

### Error: "Python version too old"

**What you see:**
```
SyntaxError: invalid syntax
```
or
```
This package requires Python 3.9+
```

**How to fix:**

1. **Check Python version:**
   ```bash
   python --version
   # or
   python3 --version
   ```

2. **Upgrade Python:**
   - Download from [python.org](https://python.org)
   - Minimum version: **Python 3.9**
   - Recommended: **Python 3.10+**

3. **Create new virtual environment with correct version:**
   ```bash
   python3.10 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

---

## Performance Issues

### Analysis taking too long

**Expected times:**
- **10 tickers:** 5-10 seconds
- **50 tickers:** 30-60 seconds
- **100 tickers:** 2-3 minutes
- **500 tickers:** 10-15 minutes
- **5,544 tickers** (full eToro market): 15-20 minutes

**If slower than expected:**

1. **First run is always slower:**
   - No cached data yet
   - API calls for all tickers
   - Subsequent runs use 48-hour cache

2. **Network speed:**
   - Slow connection affects API calls
   - Run speed test
   - Consider running during off-peak hours

3. **System resources:**
   ```bash
   # Check CPU/memory usage
   top          # Mac/Linux
   Task Manager # Windows
   ```

4. **Progress tracking:**
   - Tool shows progress bars
   - Displays estimated time remaining
   - Shows tickers processed/total

### Memory issues

**What you see:**
```
MemoryError: Unable to allocate array
```

**How to fix:**

1. **Process in batches:**
   - Don't analyze 1000+ tickers at once
   - Break into chunks of 100-200 tickers

2. **Close other applications:**
   - Free up RAM
   - Especially browser tabs with heavy sites

3. **Increase system memory:**
   - Analysis of 500+ tickers needs 4GB+ RAM
   - Full eToro market (5,544 tickers) needs 8GB+ RAM

---

## Output File Problems

### HTML report not opening

**Problem:** Double-clicking `portfolio.html` doesn't work or shows blank page

**How to fix:**

1. **Manual open:**
   ```bash
   # Mac
   open yahoofinance/output/portfolio.html

   # Linux
   xdg-open yahoofinance/output/portfolio.html

   # Windows
   start yahoofinance\output\portfolio.html
   ```

2. **Browser security:**
   - Some browsers block local HTML files
   - Try different browser
   - Or use "Open File" from browser menu

3. **File permissions:**
   ```bash
   # Check file exists and is readable
   ls -l yahoofinance/output/*.html
   ```

### CSV file corrupted/empty

**Problem:** Output CSV files are empty or won't open

**How to fix:**

1. **Check file exists:**
   ```bash
   ls -l yahoofinance/output/
   ```

2. **Verify file size:**
   ```bash
   # Files should be > 0 bytes
   ls -lh yahoofinance/output/*.csv
   ```

3. **Check logs for errors:**
   - Look for "Failed to generate" messages
   - Check write permissions on output directory

4. **Re-run analysis:**
   ```bash
   # Clean and re-run
   rm yahoofinance/output/*.csv
   python trade.py -o p
   ```

---

## Getting Additional Help

### Enable debug logging

For detailed error information:

```bash
# Set log level to DEBUG (if supported by your version)
# Check trade.py or config files for logging configuration
```

### Check log files

Look in project directory for:
- `trade.log`
- Console output showing errors

### Common commands for diagnosis

```bash
# Test specific ticker
python trade.py -o i -t AAPL

# Check installation
pip list | grep yfinance

# Verify Python version
python --version

# Test file access
ls -l yahoofinance/input/portfolio.csv
ls -l yahoofinance/output/
```

### Report issues

If problem persists:

1. **Check GitHub Issues:**
   - Visit [GitHub Issues](https://github.com/weirdapps/etorotrade/issues)
   - Search for similar problems

2. **Create new issue:**
   - Describe the problem
   - Include error messages
   - Specify Python version
   - Operating system details
   - Steps to reproduce

3. **Include relevant information:**
   ```
   - Python version: 3.10.5
   - Operating system: macOS 13.2
   - Command run: python trade.py -o p
   - Error message: [copy error here]
   - Number of tickers: 50
   ```

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Portfolio not found | Check `yahoofinance/input/portfolio.csv` exists |
| API timeout | Wait and retry, or reduce tickers |
| Rate limiting | Wait 1 hour between large analyses |
| All SELL signals | Test with `python trade.py -o i -t AAPL` |
| INCONCLUSIVE signals | Stock lacks analyst coverage (normal) |
| Slow performance | First run is slower, cache helps next time |
| Import errors | Run `pip install -r requirements.txt` |

---

*Last Updated: October 2025*

**Remember:** This tool provides analysis only, not investment advice. All investment decisions are your responsibility.

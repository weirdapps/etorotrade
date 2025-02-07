=== Follow or Copy PLESSAS on etoro ===

# Market Analysis and Portfolio Management Tool

A Python-based tool for analyzing stocks using data from Yahoo Finance. The tool provides a comprehensive market analysis report with various metrics to help in stock selection and portfolio management.

## Features

- Real-time stock data from Yahoo Finance
- Multiple data  options (Portfolio, Market, Manual)
- Comprehensive analysis with multiple metrics
- Color-coded output for quick insights
- Support for multiple stocks analysis

## Usage

Run the tool using:
```bash
python trade.py
```

You'll be prompted to choose a data source:
- P: Load tickers from portfolio file (yahoofinance/input/portfolio.csv)
- M: Load tickers from market file (yahoofinance/input/market.csv)
- I: Manually input tickers (comma-separated)

## Metrics Explanation

### Price and Target
- **PRICE**: Current stock price
- **TARGET**: Average analyst price target
- **UPSIDE**: Percentage difference between target and current price
- **EXRET**: Expected return (Upside × Buy %) - Used for ranking table

### Analyst Coverage
- **# T**: Number of analysts providing price targets
- **% BUY**: Percentage of analysts recommending Buy
- **# A**: Number of analysts providing ratings

### Valuation Metrics
- **PET**: Trailing P/E ratio (Price / Last 12 months earnings)
  - High: Stock might be expensive
  - Low: Might indicate value or problems
  - Industry comparison is important

- **PEF**: Forward P/E ratio (Price / Next 12 months expected earnings)
  - Lower than PET suggests expected earnings growth
  - Higher than PET suggests expected earnings decline

- **PEG**: Price/Earnings to Growth ratio
  - < 1: Potentially undervalued
  - > 1: Potentially overvalued
  - ~1: Fairly valued

### Risk Metrics
- **BETA**: Stock's volatility compared to the market
  - > 1: More volatile than market (e.g., 1.5 = 50% more volatile)
  - < 1: Less volatile than market (e.g., 0.5 = 50% less volatile)
  - = 1: Same volatility as market

### Income & Ownership
- **DIV %**: Dividend yield percentage
- **SI**: Short Interest - % of float shares sold short
  - High: Indicates bearish sentiment
  - Low: Less bearish sentiment
- **INS %**: Percentage of insider buy transactions
- **# INS**: Number of insider transactions

### Timing
- **EARNINGS**: Date of last earnings report
  - Used to track analyst reports and insider trading since last earnings

## Color Coding

The tool uses color coding for quick visual analysis:
- 🟢 **Green**: Potential buy signal (examine also other factors)
  - High upside potential (>15%)
  - Strong analyst consensus (>65% buy)
  - Sufficient analyst coverage (>4)

- 🔴 **Red**: Potential Sell signal (examine also other factors)
  - Low upside potential (<5%) or
  - Low analyst consensus (<50%)
  - Sufficient analyst coverage (>4)

- 🟡 **Yellow**: Low confidence
  - Limited analyst coverage (<4 analysts)
  - Use additional research

- ⚪ **White**: Hold/Neutral
  - Moderate metrics (examine also other factors)
  - Neither strong buy nor sell signals

## Data Sources

All data is sourced from Yahoo Finance API through the yfinance Python package. The data includes:
- Real-time and historical price data
- Analyst recommendations and price targets
- Financial metrics and ratios
- Insider trading information
- Company fundamentals

## Dependencies

- Python 3.x
- yfinance
- pandas
- tabulate
- tqdm

=== Follow or Copy PLESSAS on etoro ===

### === FOLLOW OR COPY plessas ON ETORO ===

# Stock Market Analysis Tools

This repository contains two independent tools for stock market analysis:

## 1. Market Analysis Tool (trade.py & yahoofinance/)

A standalone command-line tool for analyzing stock market data using Yahoo Finance. This component is completely independent and can be used without any other parts of the repository.

### Features
- Real-time stock price tracking
- Analyst ratings and recommendations analysis
- Price target tracking
- Key financial metrics (PE ratio, PEG ratio, dividend yield)
- Comprehensive market reporting with customizable display formats

### Usage
```bash
python trade.py
```

You'll be prompted to choose your data source:
- P: Load from portfolio file (yahoofinance/input/portfolio.csv)
- M: Load from market watchlist (yahoofinance/input/market.csv)
- I: Manual ticker input

### Structure
```
.
├── trade.py                 # Main CLI interface
└── yahoofinance/           # Self-contained Yahoo Finance analysis module
    ├── analyst.py          # Analyst ratings handling
    ├── client.py           # Yahoo Finance API client
    ├── display.py          # Market data display
    ├── formatting.py       # Output formatting
    ├── pricing.py          # Price analysis
    └── input/              # Input data files
        ├── market.csv      # Market watchlist
        └── portfolio.csv   # Portfolio data
```

### Requirements
- Python 3.x
- pandas
- yfinance
- tabulate
- tqdm

## 2. Financial Preparation System (finprep/)

A separate, comprehensive stock screening and portfolio quality tracking system. See [finprep/README.md](finprep/README.md) for details.

Features include:
- DCF valuation
- Analyst price targets and recommendations
- Institutional ownership changes
- Insider and senate transactions
- Piotroski score
- Comprehensive company ratings

## Error Handling

The Market Analysis Tool includes comprehensive error handling:
- Validation of input data
- Graceful handling of API failures
- Logging of errors and warnings
- Fallback values for missing data

## Output

The Market Analysis Tool generates detailed reports including:
- Stock rankings
- Price metrics
- Analyst recommendations
- Key financial ratios
- Performance indicators

Reports are displayed in a clear, tabulated format with proper alignment and formatting for easy reading.

## Note on Project Structure

This repository contains two independent tools that can be used separately:
1. The Market Analysis Tool (trade.py & yahoofinance/) is a standalone component
2. The Financial Preparation System (finprep/) is a separate tool with its own functionality

Each tool has its own complete implementation and can be used independently of the other.

### === FOLLOW OR COPY plessas ON ETORO ===

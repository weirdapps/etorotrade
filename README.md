# Trade Analysis Project

This project provides market analysis capabilities with multiple implementations using different data sources.

## Project Structure

- `trade.py` - Main entry point for running market analysis
- `fmp/` - Financial Modeling Prep API implementation
  - Original implementation using FMP API
  - Contains all utility scripts, support modules, and output files
- `yfinOLD/` - Original Yahoo Finance implementation
  - Legacy implementation using yfinance library
  - Basic market analysis features
- `yfin2/` - Enhanced Yahoo Finance implementation
  - Improved implementation with better error handling
  - Enhanced sorting and display capabilities
  - Proper color coding and formatting

## Usage

Run the main analysis script:

```bash
python trade.py
```

Then select your data source:
- `P` for Portfolio analysis
- `M` for Market analysis
- `I` for Manual ticker input

## Implementations

### FMP Implementation (fmp/)
Original implementation using Financial Modeling Prep API, including:
- Portfolio tracking
- Market analysis
- News tracking
- Earnings analysis
- Custom screens

### Original YFinance (yfinOLD/)
First Yahoo Finance implementation with basic features:
- Stock analysis
- Price targets
- Analyst ratings
- Insider transactions

### Enhanced YFinance (yfin2/)
Improved implementation with:
- Better error handling
- Enhanced display formatting
- Proper sorting of results
- Improved color coding
- More reliable data fetching

## Data Sources
- Financial Modeling Prep API (fmp implementation)
- Yahoo Finance API (yfinOLD and yfin2 implementations)
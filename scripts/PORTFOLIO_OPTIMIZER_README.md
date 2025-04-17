# Portfolio Optimizer

This is a Modern Portfolio Theory (MPT) based optimizer for your investment portfolio. It determines optimal allocations for your portfolio that maximize the Sharpe ratio (risk-adjusted return) while respecting minimum and maximum position constraints.

## Key Features

- **Modern Portfolio Theory**: Uses MPT to find the optimal weights that maximize the Sharpe ratio
- **Position Constraints**: Set minimum and maximum position sizes to create realistic portfolios
- **Multiple Time Horizons**: Analyze over different time periods (1, 3, 4, 5 years)
- **International Tickers**: Fully supports international ticker symbols with extensions (.L, .HK, .DE, etc.)
- **2-Step Approach**: First download historical data, then run optimization without hitting API limits
- **Rate Limiting**: Uses the project's rate limiter to avoid Yahoo Finance API throttling
- **Built-in Caching**: Cache historical data and current prices for faster optimization runs

## How to Use

The portfolio optimization process is divided into two steps:

1. **Download Data**: First download all required historical data and current prices
2. **Run Optimization**: Then run the optimizer using the downloaded data

This approach prevents rate limiting issues with Yahoo Finance and allows for faster optimization runs.

### Step 1: Download Historical Data and Current Prices

Run the following command to download historical price data and current prices for all tickers in your portfolio:

```bash
python -m scripts.download_portfolio_data
```

**Options:**
- `--portfolio PATH`: Path to portfolio CSV file (default: yahoofinance/input/portfolio.csv)
- `--max-years N`: Maximum years of historical data to retrieve (default: 6)
- `--batch-size N`: Number of tickers per batch (default: 10)
- `--delay N`: Base delay between API calls in seconds (default: 1.0)
- `--output PATH`: Output file for cached data (default: yahoofinance/data/portfolio_cache.pkl)
- `--price-output PATH`: Output file for current prices (default: yahoofinance/data/portfolio_prices.json)
- `--verbose`: Enable verbose logging

**Example:**
```bash
python -m scripts.download_portfolio_data --batch-size 5 --delay 2.0 --max-years 3
```

### Step 2: Run Portfolio Optimization

After downloading the data, run the optimizer using the cached data:

```bash
python -m scripts.run_optimizer --use-cache
```

**Options:**
- `--min N`: Minimum position size in USD (default: 1000.0)
- `--max N`: Maximum position size in USD (default: 25000.0)
- `--periods N [N ...]`: Time periods in years to analyze (default: 1 3 4 5)
- `--use-cache`: Use cached historical data and prices (default: False)
- `--cache-path PATH`: Path to cached historical data (default: yahoofinance/data/portfolio_cache.pkl)
- `--price-cache-path PATH`: Path to cached price data (default: yahoofinance/data/portfolio_prices.json)
- `--limit N`: Limit the number of tickers to process (0 = no limit, default: 0)
- `--verbose`: Enable verbose logging

**Example:**
```bash
python -m scripts.run_optimizer --use-cache --min 2000 --max 15000 --periods 1 3 4 5
```

## Understanding the Results

For each time period, the optimizer outputs:

1. **Expected Annual Return**: The expected annual return of the optimal portfolio
2. **Expected Annual Volatility**: The expected annual volatility (risk) of the optimal portfolio
3. **Sharpe Ratio**: The risk-adjusted return (higher is better)
4. **Optimal Allocation**: A table showing the optimal weights, dollar amounts, and share counts for each ticker

The optimizer respects your constraints:
- Positions below the minimum investment amount are excluded
- Positions above the maximum investment amount are capped at the maximum

## Tips for Better Results

- **Use multiple time horizons**: Different time periods can provide different insights
- **Adjust min/max constraints**: Different constraints can yield different portfolios
- **Run with and without certain asset classes**: Try excluding certain asset classes to see the impact
- **Larger batch sizes**: For reliable connections, increase batch size to speed up data collection
- **Refresh data periodically**: Re-run the data download step periodically to get fresh prices

## Troubleshooting

- **Rate limiting errors**: Increase the `--delay` parameter in the download step
- **Missing tickers**: Some tickers may not be available through Yahoo Finance - check ticker symbols
- **Long runtime**: Use the `--limit` parameter to test with fewer tickers first
- **Memory issues**: Reduce the `--max-years` parameter to collect less historical data
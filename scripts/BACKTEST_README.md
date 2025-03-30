# Trading Criteria Backtesting

This module provides backtesting functionality to evaluate different trading criteria parameters against historical data.

## Features

- Test trading criteria against historical market data
- Simulate portfolio performance with realistic assumptions
- Smart market cap-based position weighting (1-10% per position)
- Smart data filtering to maximize backtest periods
- Optimize parameters to find the best trading criteria
- Generate HTML reports with performance metrics
- Detailed portfolio synthesis tables showing all position metrics
- Compare strategy performance against S&P 500 benchmark
- Visualize portfolio growth over time
- Interactive progress bars for tracking optimization progress

## Usage

### Basic Backtesting

To run a basic backtest with default settings:

```bash
python scripts/optimize_criteria.py --mode backtest
```

This will test the default trading criteria on your portfolio stocks for a 3-year period.

### Parameter Optimization

To find optimal trading criteria parameters:

```bash
python scripts/optimize_criteria.py --mode optimize --metric sharpe_ratio
```

This will test different parameter combinations and find the set that maximizes the Sharpe ratio.

### Command-line Options

```
  --mode {backtest,optimize}
                        Mode to run (backtest or optimize)
  --period {1y,2y,3y,5y,max}
                        Backtest period
  --tickers TICKERS     Comma-separated list of tickers to backtest
  --ticker-limit TICKER_LIMIT
                        Limit number of tickers to test (for faster execution)
  --data-coverage-threshold DATA_COVERAGE_THRESHOLD
                        Data coverage threshold (0.0-1.0). Lower values allow longer backtests 
                        by excluding tickers with limited history.
  --source {portfolio,market,etoro,yfinance,usa,europe,china,usindex}
                        Source of tickers
  --capital CAPITAL     Initial capital
  --position-size POSITION_SIZE
                        Target position size percentage (note: actual allocation uses
                        market cap-based weighting between 1% and 10% per position)
  --max-positions MAX_POSITIONS
                        Maximum number of positions
  --commission COMMISSION
                        Commission percentage
  --rebalance {daily,weekly,monthly}
                        Rebalance frequency
  --metric {total_return,annualized_return,sharpe_ratio,max_drawdown,volatility}
                        Metric to optimize
  --max-combinations MAX_COMBINATIONS
                        Maximum number of parameter combinations to test
  --param-file PARAM_FILE
                        JSON file with parameter ranges for optimization
  --output OUTPUT       Output file for results (CSV)
  --quiet               Suppress progress bars and detailed output (for batch processing)
```

### Parameter Configuration

You can specify parameter ranges to test in a JSON file. Here's an example of what this file might look like:

```json
{
  "SELL": {
    "SELL_MIN_PEG": [2.0, 2.5, 3.0, 3.5],
    "SELL_MIN_SHORT_INTEREST": [1.0, 1.5, 2.0, 2.5],
    "SELL_MIN_BETA": [2.5, 3.0, 3.5, 4.0],
    "SELL_MAX_EXRET": [2.5, 5.0, 7.5, 10.0],
    "SELL_MAX_UPSIDE": [3.0, 5.0, 7.5],
    "SELL_MIN_BUY_PERCENTAGE": [60.0, 65.0, 70.0],
    "SELL_MIN_FORWARD_PE": [45.0, 50.0, 55.0]
  },
  "BUY": {
    "BUY_MIN_UPSIDE": [20.0, 25.0, 30.0],
    "BUY_MIN_BUY_PERCENTAGE": [85.0, 87.5, 90.0],
    "BUY_MAX_PEG": [2.0, 2.5, 3.0],
    "BUY_MAX_SHORT_INTEREST": [1.0, 1.5, 2.0],
    "BUY_MIN_EXRET": [15.0, 20.0, 25.0],
    "BUY_MIN_BETA": [0.2, 0.25, 0.3],
    "BUY_MAX_BETA": [2.5, 3.0, 3.5],
    "BUY_MIN_FORWARD_PE": [0.3, 0.5, 0.7],
    "BUY_MAX_FORWARD_PE": [40.0, 45.0, 50.0]
  },
  "CONFIDENCE": {
    "MIN_ANALYST_COUNT": [3, 5, 7],
    "MIN_PRICE_TARGETS": [3, 5, 7]
  }
}
```

You can also test specific parameter ranges programmatically:

```python
parameter_ranges = {
    "SELL.SELL_MIN_SHORT_INTEREST": [2.0, 3.0, 4.0],
    "BUY.BUY_MIN_UPSIDE": [20.0, 25.0, 30.0],
    "BUY.BUY_MIN_EXRET": [15.0, 20.0, 25.0]
}
```

### Examples

Here are some useful example commands:

1. Test a single set of parameters on 10 US stocks:
   ```bash
   python scripts/optimize_criteria.py --mode backtest --source usa --ticker-limit 10 --period 1y
   ```

2. Run a 2-year backtest on major tech and financial stocks:
   ```bash
   python scripts/optimize_criteria.py --mode backtest --tickers "AAPL,MSFT,AMZN,META,GOOGL,NVDA,TSLA,JPM,BAC,UNH,LLY,WMT,KO,MCD" --period 2y
   ```

3. Run a 5-year backtest on index stocks, using only 60% of tickers with the best data availability:
   ```bash
   python scripts/optimize_criteria.py --mode backtest --source usindex --period 5y --data-coverage-threshold 0.6
   ```

4. Optimize parameters for best return:
   ```bash
   python scripts/optimize_criteria.py --mode optimize --source usa --ticker-limit 15 --period 1y --metric total_return
   ```

5. Find strategy with lowest drawdown:
   ```bash
   python scripts/optimize_criteria.py --mode optimize --metric max_drawdown --max-combinations 20
   ```

6. Run in quiet mode for automated/background usage:
   ```bash
   python scripts/optimize_criteria.py --mode optimize --metric sharpe_ratio --quiet --output results.json
   ```

### Progress Bars

The backtester now features interactive progress bars showing:

1. **Parameter Combination Progress**: Shows which parameter set is being tested and tracks the best metric value found so far
2. **Data Loading Progress**: Shows the loading progress for stock historical data
3. **Prefetching Historical Data**: Shows the progress of prefetching historical data for synthetic analyst data generation
4. **Synthetic Data Generation**: Shows the progress of generating synthetic analyst data for backtesting
5. **Portfolio Simulation Progress**: Shows simulated portfolio value and positions as the backtest runs

To disable progress bars for automated or background processing, use the `--quiet` flag.

### Performance Optimizations

The backtester includes several optimizations to reduce API calls and improve performance:

1. **Historical Data Caching**: All historical data is cached to avoid repeated API calls
2. **Smart Data Filtering**: Excludes tickers with limited history to maximize backtest periods
3. **Batch Processing**: Ticker data is processed in batches to reduce API call overhead
4. **Prefetching**: Historical data is prefetched before generating synthetic analyst data
5. **Adaptive Batch Sizing**: Batch size automatically adjusts based on the number of tickers
6. **Parallel Processing**: Tickers are processed in parallel batches for maximum efficiency
7. **Smart Caching**: Each ticker's data is cached only once, regardless of how many times it's used
8. **Progress Visualization**: Color-coded progress bars help monitor different stages of processing:
   - Cyan: Initial data loading
   - Magenta: Prefetching historical data
   - Yellow: Synthetic data generation
   - Blue: Portfolio simulation
   - Green: Parameter optimization

With these optimizations, backtesting performance has improved significantly:
- Reduced API calls by up to 90% during backtesting
- Eliminated rate limiting warnings during backtest runs
- Improved performance for synthetic data generation
- Better visibility into progress with detailed, color-coded progress bars
- Maximized backtesting periods by intelligently filtering out tickers with limited data
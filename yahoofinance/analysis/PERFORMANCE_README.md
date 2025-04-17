# Performance Tracker Module

This module provides tools for tracking market index performance and portfolio performance over different time periods.

## Features

- Track market index performance across multiple time periods:
  - **Weekly**: Week-over-week comparison
  - **Monthly**: Month-over-month comparison 
  - **Year-to-date (YTD)**: From December 31st of previous year to present
  - **Month-to-date (MTD)**: From last day of previous month to present
- Generate HTML dashboards with performance metrics
- Save performance data in JSON format for further analysis
- Display colorized performance metrics in the console
- Track portfolio performance metrics including:
  - Month-to-date performance
  - Year-to-date performance
  - 2-year performance
  - Risk metrics (Beta, Sharpe ratio, Alpha, Sortino ratio)

## Usage

The performance tracker can be run directly from the command line:

```bash
# Track weekly market performance
python -m yahoofinance.analysis.performance weekly
# or the shorthand version
python -m yahoofinance.analysis.performance w

# Track monthly market performance
python -m yahoofinance.analysis.performance monthly
# or the shorthand version
python -m yahoofinance.analysis.performance m

# Track year-to-date market performance
python -m yahoofinance.analysis.performance yeartodate
# or the shorthand versions
python -m yahoofinance.analysis.performance ytd
python -m yahoofinance.analysis.performance y

# Track month-to-date market performance
python -m yahoofinance.analysis.performance monthtodate
# or the shorthand versions
python -m yahoofinance.analysis.performance mtd
python -m yahoofinance.analysis.performance md

# Track portfolio performance
python -m yahoofinance.analysis.performance portfolio
# or the shorthand version
python -m yahoofinance.analysis.performance p

# Track both market and portfolio performance asynchronously
python -m yahoofinance.analysis.performance all
# You can specify which market period to include
python -m yahoofinance.analysis.performance all weekly
python -m yahoofinance.analysis.performance all monthly
python -m yahoofinance.analysis.performance all yeartodate
python -m yahoofinance.analysis.performance all monthtodate
```

## Output

The performance tracker generates the following outputs:

1. Console output with formatted tables showing:
   - Index values for different time periods
   - Percentage changes (colored green for positive, red for negative)
   - Portfolio performance metrics when tracking portfolio

2. HTML dashboard (`/yahoofinance/output/performance.html`) with:
   - Interactive charts
   - Formatted tables
   - Visual indicators for performance

3. JSON data file (`/yahoofinance/output/performance.json`) containing:
   - Raw performance data for programmatic access
   - Timestamps for when the data was collected

## Time Period Definitions

The performance tracker uses these specific date ranges for different period types:

- **Weekly**: Most recent completed week vs. the week before that
  - Example: Week of Apr 8-12 vs. Week of Apr 1-5
  
- **Monthly**: Most recent completed month vs. the month before that
  - Example: March 2025 vs. February 2025
  
- **Year-to-date (YTD)**: December 31, 2024 to present
  - Measures performance from the beginning of the calendar year
  
- **Month-to-date (MTD)**: March 31, 2025 to present (if current month is April)
  - Measures performance from the last day of the previous month
  - For January, it measures from December 31 of the previous year

## Integration in Your Code

You can also use the performance tracker programmatically in your Python code:

```python
from yahoofinance.analysis.performance import PerformanceTracker

# Create a performance tracker
tracker = PerformanceTracker()

# Get weekly index performance
weekly_performance = tracker.get_index_performance(period_type="weekly")

# Get monthly index performance
monthly_performance = tracker.get_index_performance(period_type="monthly")

# Get year-to-date index performance
ytd_performance = tracker.get_index_performance(period_type="yeartodate")

# Get month-to-date index performance
mtd_performance = tracker.get_index_performance(period_type="monthtodate")

# Get portfolio performance
portfolio_performance = tracker.get_portfolio_performance_web(
    url="https://bullaware.com/etoro/plessas"
)

# Generate HTML dashboard for index performance
tracker.generate_index_performance_html(
    weekly_performance,
    title="Weekly Market Performance"
)

# Generate HTML dashboard for portfolio performance
tracker.generate_portfolio_performance_html(
    portfolio_performance,
    title="Portfolio Performance Summary"
)

# Save performance data to JSON
tracker.save_performance_data(
    weekly_performance,
    file_name="weekly_performance.json"
)
```

## Asynchronous Operation

The performance tracker also supports asynchronous operation for improved performance:

```python
import asyncio
from yahoofinance.analysis.performance import track_performance_async

# Track both market and portfolio performance asynchronously
asyncio.run(track_performance_async(period_type="weekly"))
```
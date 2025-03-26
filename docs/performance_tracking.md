# Performance Tracking Module

The `yahoofinance_v2.analysis.performance` module provides functionality for tracking market index performance and portfolio performance metrics from external sources. This module implements two key features that were previously available in yahoofinance v1:

1. **Market Index Performance Tracking** (formerly `index.py`)
   - Tracks weekly and monthly performance of major market indices
   - Generates HTML dashboards with performance metrics
   - Provides both synchronous and asynchronous implementations

2. **Portfolio Performance Web Scraping** (formerly `portfolio.py`)
   - Scrapes portfolio performance metrics from external web sources
   - Extracts key metrics like MTD/YTD performance, beta, Sharpe ratio, etc.
   - Generates HTML dashboards and saves performance data
   - Includes circuit breaker pattern for resilient web scraping

## Usage Examples

### Tracking Market Index Performance

```python
from yahoofinance_v2.analysis.performance import track_index_performance

# Track weekly performance
track_index_performance(period_type="weekly")

# Track monthly performance
track_index_performance(period_type="monthly")
```

### Tracking Portfolio Performance

```python
from yahoofinance_v2.analysis.performance import track_portfolio_performance

# Use default URL
track_portfolio_performance()

# Specify a custom URL
track_portfolio_performance(url="https://your-portfolio-url.com")
```

### Asynchronous Tracking (Both Index and Portfolio)

```python
import asyncio
from yahoofinance_v2.analysis.performance import track_performance_async

# Track both weekly index performance and portfolio performance
asyncio.run(track_performance_async(
    period_type="weekly",
    portfolio_url="https://your-portfolio-url.com"
))
```

## Advanced Usage with PerformanceTracker Class

For more control over the tracking process, you can use the `PerformanceTracker` class directly:

```python
from yahoofinance_v2.analysis.performance import PerformanceTracker
from yahoofinance_v2.api import get_provider, get_async_provider

# Synchronous usage
tracker = PerformanceTracker(
    provider=get_provider(),      # Optional: specify a provider
    output_dir="custom/output"    # Optional: specify output directory
)

# Get index performance
performances = tracker.get_index_performance(period_type="weekly")

# Get portfolio performance
portfolio_perf = tracker.get_portfolio_performance_web(url="https://your-portfolio-url.com")

# Generate HTML dashboards
tracker.generate_index_performance_html(performances, title="Weekly Market Performance")
tracker.generate_portfolio_performance_html(portfolio_perf)

# Save data to JSON files
tracker.save_performance_data(performances, file_name="weekly_performance.json")
tracker.save_performance_data(portfolio_perf, file_name="portfolio_performance.json")
```

## Data Classes

The module provides two data classes for representing performance data:

1. `IndexPerformance` - Represents performance metrics for a market index
2. `PortfolioPerformance` - Represents portfolio performance metrics

These classes provide convenient access to performance metrics and can be easily converted to dictionaries for HTML generation or JSON serialization.

## Integration with HTML Generation

The performance tracking module integrates with the HTML generation functionality in `yahoofinance_v2.presentation.html` to create dashboards for performance metrics.

## Example Application

A complete example application is available in `examples/performance_example.py`, which demonstrates all features of the performance tracking module.

## Circuit Breaker Pattern

The web scraping functionality includes the circuit breaker pattern for resilience:

- Automatically detects and handles web scraping failures
- Prevents excessive retries when the target site is unavailable
- Gradually recovers when the target site becomes available again

## Asynchronous Support

Both market index tracking and portfolio performance tracking support asynchronous operation for improved performance.
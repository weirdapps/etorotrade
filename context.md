# Trade Project: Market Analysis and Portfolio Management System

## Overview

Trade is a sophisticated Python-based market analysis system that leverages Yahoo Finance data through the yfinance package to provide comprehensive stock analysis, portfolio management, and market intelligence. The system features advanced rate limiting, intelligent caching, and multiple output formats.

## Core Architecture

### 1. Data Pipeline

```
User Input → Rate Limiter → Yahoo Finance API → Data Processing → Multiple Output Formats
```

The system follows a robust data pipeline:
1. Input handling (CSV files, manual input)
2. Rate-limited API requests with adaptive delays
3. Data processing and color-coded analysis
4. Multi-format output (Console, CSV, HTML)

### 2. Key Components

#### Client Layer (client.py)
- YFinanceClient class for API interactions
- StockData class for structured data storage
- Custom exception hierarchy for error handling
- Caching with LRU for frequently accessed tickers
- Advanced rate limiting with exponential backoff
- Risk metrics calculation (Alpha, Beta, Sharpe, Sortino)

#### Analysis Layer
- **Analyst Module** (analyst.py)
  * Analyst ratings and recommendations
  * Buy/Sell/Hold percentage calculations
  * Target price analysis
  * Earnings impact assessment

- **Pricing Module** (pricing.py)
  * Price metrics calculation
  * Upside potential analysis
  * Historical price tracking
  * Expected return computation

- **Insiders Module** (insiders.py)
  * Insider transaction monitoring
  * Buy/Sell percentage tracking
  * Transaction count analysis
  * Insider sentiment metrics

- **Market Intelligence**
  * News aggregation with sentiment analysis
  * Earnings calendar management
  * Economic indicators tracking
  * Institutional holdings analysis

#### Display Layer (display.py)
- MarketDisplay class for output management
- RateLimitTracker for API call management
- Batch processing with progress tracking
- Color-coded output based on analysis metrics
- CSV data export with standardized formatting
- HTML dashboard generation with performance indicators

### 3. Data Structures

#### StockData Class
Comprehensive dataclass containing:
- Basic Info: name, sector, market_cap
- Price Data: current_price, target_price, change percentages
- Analyst Data: recommendations, ratings, analyst counts
- Financial Ratios: PE (trailing/forward), PEG, Beta
- Risk Metrics: short_float, Alpha, Sharpe, Sortino
- Dividend Data: dividend_yield
- Event Data: earnings dates
- Insider Info: insider_buy_pct, insider_transactions

#### DisplayConfig Class
Configuration for display formatting:
- Color coding enabled/disabled
- Date and number formatting options
- Display thresholds for buy/sell signals
- Minimum analyst coverage for high confidence

## Key Features

### 1. Rate Limiting System
- **RateLimitTracker**
  * Window-based tracking (60-second window)
  * Adaptive delays (1-30 seconds)
  * Success streak monitoring
  * Error pattern detection
  * Ticker-specific error tracking
  * Batch processing optimization (15 tickers/batch)
  * Batch success rate monitoring

### 2. Data Analysis
- **Comprehensive Metrics**
  * Price and target analysis
  * Upside potential calculation
  * Analyst recommendation aggregation
  * Expected return calculation (EXRET)
  * Risk metrics (Beta, Alpha, Sharpe, Sortino)
  * Insider trading patterns
  * Valuation metrics (PE, PEG, etc.)

- **Color Coding System**
  * 🟢 Green: Strong Buy (4+ analysts, 15%+ upside, 65%+ buy ratings)
  * 🔴 Red: Sell (4+ analysts AND <5% upside OR <50% buy ratings)
  * 🟡 Yellow: Low Confidence (<4 analysts or limited data)
  * ⚪ White: Hold (metrics between buy/sell thresholds)

### 3. Portfolio Management
- Performance tracking (Daily, MTD, YTD, 2YR)
- Risk analysis (Beta, Alpha, Sharpe, Sortino)
- Position monitoring
- HTML dashboard generation

### 4. Market Intelligence
- News aggregation with sentiment scoring
- Earnings date tracking
- Economic indicator monitoring
- Major market indices coverage (US, Europe, Asia)

## Implementation Details

### 1. Rate Limiting Implementation
```python
class RateLimitTracker:
    def __init__(self):
        self.window_size = 60  # Time window in seconds
        self.max_calls = 100   # Maximum calls per window
        self.calls = deque(maxlen=1000) # Timestamp queue
        self.errors = deque(maxlen=20)  # Recent errors
        self.base_delay = 2.0  # Base delay between calls
        self.min_delay = 1.0   # Minimum delay
        self.max_delay = 30.0  # Maximum delay
        self.batch_delay = 5.0 # Delay between batches
        self.error_counts = {} # Ticker-specific error tracking
        self.success_streak = 0 # Consecutive successful calls
```

### 2. Data Processing Pipeline
```python
def display_report(self, tickers, source=None):
    # 1. Process tickers in batches
    reports = self._process_tickers(tickers, batch_size=15)
    
    # 2. Save raw data to CSV if source specified
    if source in ['M', 'P']:
        self._save_to_csv(reports, source)
    
    # 3. Format data for display
    formatted_reports = [report['formatted'] for report in reports]
    df = pd.DataFrame(formatted_reports)
    
    # 4. Sort and format DataFrame
    df = self._sort_market_data(df)
    df = self._format_dataframe(df)
    
    # 5. Display the final report
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
```

### 3. Error Handling Strategy
- Custom exception hierarchy (YFinanceError → APIError, ValidationError)
- Automatic retry with exponential backoff
- Skip tickers with persistent errors (after 5 failures)
- Empty report generation for problematic tickers
- Comprehensive logging at appropriate levels

## Configuration

### 1. Input Files
- portfolio.csv: Portfolio holdings (ticker column)
- market.csv: Market watchlist (symbol column)
- cons.csv: Constants and configurations

### 2. Output Files
- portfolio.html: Portfolio dashboard
- index.html: Market performance dashboard
- market.csv: Raw market data export
- portfolio.csv: Portfolio analysis export

### 3. Environment Variables
Required:
- FRED_API_KEY: For economic data

Optional:
- NEWS_API_KEY: For additional news sources

## Best Practices

### 1. Rate Limiting
- Always use RateLimitTracker for API calls
- Respect batch sizes (15 tickers per batch)
- Include appropriate delays between API calls
- Monitor success rates and adjust delays accordingly
- Implement ticker-specific delay handling for problematic tickers

### 2. Error Handling
- Use the custom exception hierarchy
- Validate input data thoroughly
- Implement retries with exponential backoff
- Log errors at appropriate levels
- Provide fallback data when possible

### 3. Data Formatting
- Use DisplayFormatter for consistent formatting
- Handle None/missing values gracefully
- Apply color coding based on analysis metrics
- Use appropriate precision for different data types
- Sort data meaningfully (by expected return, then earnings date)

## Common Tasks

### 1. Adding New Features
1. Identify appropriate module
2. Implement rate-limited data fetching
3. Add error handling with custom exceptions
4. Update display formatting in formatting.py
5. Add tests for new functionality

### 2. Modifying Analysis
1. Update relevant analysis module
2. Adjust display formatting and thresholds
3. Update display.py to include new metrics
4. Add validation for new metrics in types.py
5. Update tests to cover changes

### 3. Adding Data Sources
1. Create new client class if needed
2. Implement rate limiting
3. Create data processor
4. Add to display layer
5. Update tests

## Testing

### 1. Test Structure
- Unit tests for each module
- Integration tests for data flow
- Mock objects for API responses
- Edge case testing for error handling

### 2. Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=yahoofinance tests/

# Run specific test file
pytest tests/test_market_display.py

# Run specific test
pytest tests/test_market_display.py::TestMarketDisplay::test_display_report
```

### 3. Test Coverage
- API interaction
- Rate limiting behavior
- Data processing logic
- Display formatting
- Error handling
- Edge cases

## Troubleshooting

### 1. API Issues
- Check rate limiting logs
- Look for patterns in errors (specific tickers)
- Try with smaller batch sizes
- Increase delay between calls
- Verify network connectivity

### 2. Data Quality
- Validate input data format
- Check for missing values
- Verify calculations
- Monitor formatting issues

### 3. Performance
- Check batch size (smaller batches for reliability)
- Monitor base delay settings
- Check for excessive error rates
- Analyze success rates by batch

## Future Development

### 1. Planned Features
- Technical analysis indicators
- Portfolio optimization
- Enhanced caching mechanisms
- Additional market metrics

### 2. Architecture Evolution
- Parallel processing
- Additional data sources integration
- Real-time data streaming
- Advanced visualization

This context provides a comprehensive understanding of the Trade project's architecture, implementation details, and best practices for development.
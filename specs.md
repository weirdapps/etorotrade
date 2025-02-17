# Market Analysis Tool Technical Specification

## Project Overview

This project is a command-line market analysis tool that fetches and analyzes stock market data from Yahoo Finance. It provides comprehensive analysis of stocks including price metrics, analyst ratings, insider trading information, and various financial ratios.

## System Requirements

### Hardware Requirements
- CPU: 1+ core
- RAM: 2GB minimum, 4GB recommended
- Storage: 100MB for installation and cache

### Software Requirements
- Python 3.x
- Operating System: Cross-platform (Windows, macOS, Linux)
- Internet connection for API access

### Dependencies
- beautifulsoup4 >= 4.12.3: Web scraping functionality
- pandas >= 2.2.2: Data manipulation and analysis
- python-dotenv >= 1.0.1: Environment variable management
- pytz >= 2024.1: Timezone handling
- requests >= 2.32.3: HTTP requests
- tabulate >= 0.9.0: Table formatting
- tqdm >= 4.66.4: Progress bars
- yfinance >= 0.2.52: Yahoo Finance API client
- pytest >= 8.0.0: Testing framework
- pytest-cov >= 4.1.0: Test coverage reporting

## Installation and Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys:
   - Create `.env` file in yahoofinance directory
   - Add Google News API key: `GOOGLE_NEWS_API_KEY=your_api_key`
   - Add FRED API key: `FRED_API_KEY=your_api_key` (get from https://fred.stlouisfed.org/docs/api/api_key.html)
4. Set up input files in yahoofinance/input/:
   - portfolio.csv: Portfolio tickers
   - market.csv: Market watchlist tickers

## Module Documentation

### 1. trade.py (Main Entry Point)

#### Functions

- `main()`: Command line interface entry point
  - Purpose: Handles user input and orchestrates the market analysis process
  - Parameters: None
  - Returns: None
  - Error Handling: Catches and logs KeyboardInterrupt and general exceptions

#### Variables

- `logger`: Logging instance for the module
  - Type: logging.Logger
  - Purpose: Handles logging with INFO level configuration

### 2. yahoofinance/download.py

#### Functions

- `setup_driver() -> webdriver.Chrome`
  - Purpose: Configure and initialize Chrome WebDriver with appropriate options
  - Returns: Configured Chrome WebDriver instance
  - Options:
    * No sandbox
    * Disabled dev shm usage
    * Disabled web security
    * Disabled GPU
    * Window size 1200x800

- `wait_and_find_element(driver: webdriver.Chrome, by: By, value: str, timeout: int = 10, check_visibility: bool = True) -> Optional[WebElement]`
  - Purpose: Helper function to wait for and find an element
  - Parameters:
    * driver: Chrome WebDriver instance
    * by: Selenium By locator strategy
    * value: Element locator value
    * timeout: Maximum wait time in seconds
    * check_visibility: Whether to check for visibility or just presence
  - Returns: Found WebElement or None if not found

- `login(driver: webdriver.Chrome, email: str, password: str) -> None`
  - Purpose: Handle login process for pi-screener.com
  - Parameters:
    * driver: Chrome WebDriver instance
    * email: Login email
    * password: Login password
  - Raises: Exception if login fails

- `process_portfolio() -> None`
  - Purpose: Process downloaded portfolio CSV file
  - Features:
    * Reads most recent CSV from Downloads folder
    * Updates crypto ticker symbols
    * Saves to yahoofinance/input/portfolio.csv
    * Cleans up downloaded file

- `download_portfolio() -> bool`
  - Purpose: Main function to download and process portfolio
  - Returns: True if successful, False if failed
  - Features:
    * Sets up Chrome WebDriver
    * Navigates to pi-screener.com
    * Handles login
    * Downloads portfolio
    * Processes downloaded file
    * Cleans up resources

### 3. yahoofinance/client.py

#### Classes

##### YFinanceError (Exception)

Base exception class for YFinance client errors

##### APIError (YFinanceError)

Exception raised when API calls fail

##### ValidationError (YFinanceError)

Exception raised when data validation fails

##### StockData (dataclass)

Data structure for stock information

###### Fields

- `name`: Company name (str)
- `sector`: Company sector (str)
- `market_cap`: Market capitalization (Optional[float])
- `current_price`: Current stock price (Optional[float])
- `target_price`: Analyst target price (Optional[float])
- `recommendation_mean`: Mean analyst recommendation (Optional[float])
- `recommendation_key`: Recommendation category (str)
- `analyst_count`: Number of analysts covering stock (Optional[int])
- `pe_trailing`: Trailing P/E ratio (Optional[float])
- `pe_forward`: Forward P/E ratio (Optional[float])
- `peg_ratio`: PEG ratio (Optional[float])
- `quick_ratio`: Quick ratio (Optional[float])
- `current_ratio`: Current ratio (Optional[float])
- `debt_to_equity`: Debt to equity ratio (Optional[float])
- `short_float_pct`: Short float percentage (Optional[float])
- `short_ratio`: Short ratio (Optional[float])
- `beta`: Beta value (Optional[float])
- `dividend_yield`: Dividend yield (Optional[float])
- `last_earnings`: Most recent earnings date (Optional[str])
- `previous_earnings`: Second most recent earnings date (Optional[str])
- `insider_buy_pct`: Percentage of insider buying (Optional[float])
- `insider_transactions`: Total insider transactions (Optional[int])
- `ticker_object`: Underlying yfinance Ticker object (Any)

##### YFinanceClient

###### Methods

- `__init__(retry_attempts: int = 3, timeout: int = 10, cache_ttl: int = 300)`
  - Purpose: Initialize client with retry and timeout settings
  - Parameters:
    - retry_attempts: Number of API retry attempts
    - timeout: API timeout in seconds
    - cache_ttl: Cache time-to-live in seconds

- `_validate_ticker(ticker: str) -> None`
  - Purpose: Validate ticker symbol format
  - Parameters:
    - ticker: Stock ticker symbol
  - Raises: ValidationError for invalid tickers

- `get_past_earnings_dates(ticker: str) -> List[pd.Timestamp]`
  - Purpose: Retrieve historical earnings dates
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: List of past earnings dates in descending order

- `get_earnings_dates(ticker: str) -> Tuple[Optional[str], Optional[str]]`
  - Purpose: Get last two earnings dates
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: Tuple of (most_recent_date, previous_date)

- `get_ticker_info(ticker: str, skip_insider_metrics: bool = False) -> StockData`
  - Purpose: Fetch comprehensive stock information
  - Parameters:
    - ticker: Stock ticker symbol
    - skip_insider_metrics: Whether to skip insider trading data
  - Returns: StockData object with company information
  - Cache: Implements LRU cache with maxsize=100

### 3. yahoofinance/analyst.py

#### Constants

- `POSITIVE_GRADES`: Set of analyst ratings considered positive
  - Type: Set[str]
  - Values: "Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive"

#### Classes

##### AnalystData

###### Methods

- `__init__(client: YFinanceClient)`
  - Purpose: Initialize with YFinance client
  - Parameters:
    - client: YFinanceClient instance

- `_validate_date(date: Optional[str]) -> None`
  - Purpose: Validate date string format
  - Parameters:
    - date: Date string in YYYY-MM-DD format
  - Raises: ValidationError for invalid dates

- `_safe_float_conversion(value: Any) -> Optional[float]`
  - Purpose: Safely convert values to float
  - Parameters:
    - value: Value to convert
  - Returns: Float value or None if conversion fails

- `fetch_ratings_data(ticker: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]`
  - Purpose: Fetch analyst ratings history
  - Parameters:
    - ticker: Stock ticker symbol
    - start_date: Optional start date for filtering
  - Returns: DataFrame with ratings history or None

- `get_ratings_summary(ticker: str, start_date: Optional[str] = None, use_earnings_date: bool = True) -> Dict[str, Optional[float]]`
  - Purpose: Calculate ratings summary metrics
  - Parameters:
    - ticker: Stock ticker symbol
    - start_date: Optional start date
    - use_earnings_date: Whether to use last earnings date as start
  - Returns: Dictionary with positive_percentage and total_ratings

- `get_recent_changes(ticker: str, days: int = 30) -> List[Dict[str, str]]`
  - Purpose: Get recent rating changes
  - Parameters:
    - ticker: Stock ticker symbol
    - days: Number of days to look back
  - Returns: List of recent rating changes

### 4. yahoofinance/pricing.py

#### Classes

##### PriceTarget (NamedTuple)

###### Fields

- `mean`: Mean target price (Optional[float])
- `high`: Highest target price (Optional[float])
- `low`: Lowest target price (Optional[float])
- `num_analysts`: Number of analysts (int)

##### PriceData (NamedTuple)

###### Fields

- `date`: Price date (datetime)
- `open`: Opening price (float)
- `high`: High price (float)
- `low`: Low price (float)
- `close`: Closing price (float)
- `volume`: Trading volume (int)
- `adjusted_close`: Adjusted closing price (float)

##### PricingAnalyzer

###### Methods

- `__init__(client: YFinanceClient)`
  - Purpose: Initialize with YFinance client
  - Parameters:
    - client: YFinanceClient instance

- `_safe_float_conversion(value: Any) -> Optional[float]`
  - Purpose: Safely convert values to float
  - Parameters:
    - value: Value to convert
  - Returns: Float value or None if conversion fails

- `get_current_price(ticker: str) -> Optional[float]`
  - Purpose: Get current stock price
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: Current price or None if unavailable

- `get_historical_prices(ticker: str, period: str = "1mo") -> List[PriceData]`
  - Purpose: Get historical price data
  - Parameters:
    - ticker: Stock ticker symbol
    - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
  - Returns: List of PriceData objects

- `get_price_targets(ticker: str) -> PriceTarget`
  - Purpose: Get analyst price targets
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: PriceTarget object with target information

- `calculate_price_metrics(ticker: str) -> Dict[str, Optional[float]]`
  - Purpose: Calculate price-related metrics
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: Dictionary with current_price, target_price, and upside_potential

### 5. yahoofinance/formatting.py

#### Classes

##### Color (Enum)

###### Values

- `GREEN`: ANSI code for buy signal ("\033[92m")
- `YELLOW`: ANSI code for low confidence ("\033[93m")
- `RED`: ANSI code for sell signal ("\033[91m")
- `RESET`: ANSI code to reset formatting ("\033[0m")
- `DEFAULT`: Empty string for hold signal ("")

##### DisplayConfig (dataclass)

###### Fields

- `use_colors`: Enable color output (bool, default=True)
- `date_format`: Date display format (str, default="%Y-%m-%d")
- `float_precision`: Decimal places for floats (int, default=2)
- `percentage_precision`: Decimal places for percentages (int, default=1)
- `table_format`: Table display format (str, default="fancy_grid")
- `min_analysts`: Minimum analysts for high confidence (int, default=4)
- `high_upside`: Threshold for buy signal (float, default=15.0)
- `low_upside`: Threshold for sell signal (float, default=5.0)
- `high_buy_percent`: Threshold for strong buy (float, default=65.0)

##### DisplayFormatter

###### Methods

- `__init__(config: DisplayConfig = DisplayConfig())`
  - Purpose: Initialize formatter with configuration
  - Parameters:
    - config: DisplayConfig instance

- `_get_color_code(num_targets: Any, upside: Any, total_ratings: Any, percent_buy: Any) -> Color`
  - Purpose: Determine color coding based on metrics
  - Parameters:
    - num_targets: Number of price targets
    - upside: Price upside percentage
    - total_ratings: Total analyst ratings
    - percent_buy: Percentage of buy ratings
  - Returns: Color enum value

- `format_value(value: Any, decimals: int = 1, percent: bool = False) -> str`
  - Purpose: Format numeric values
  - Parameters:
    - value: Value to format
    - decimals: Decimal places
    - percent: Whether to add % symbol
  - Returns: Formatted string

- `format_date(date_str: str) -> str`
  - Purpose: Format date strings
  - Parameters:
    - date_str: Date string
  - Returns: Formatted date string

- `remove_ansi(text: str) -> str`
  - Purpose: Remove ANSI color codes
  - Parameters:
    - text: Text with ANSI codes
  - Returns: Clean text

- `colorize(text: str, color: Color) -> str`
  - Purpose: Apply color to text
  - Parameters:
    - text: Text to colorize
    - color: Color enum value
  - Returns: Colored text string

- `format_stock_row(data: Dict[str, Any]) -> Dict[str, Any]`
  - Purpose: Format stock data for display
  - Parameters:
    - data: Stock data dictionary
  - Returns: Formatted data dictionary

- `create_sortable_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame`
  - Purpose: Create sorted DataFrame for display
  - Parameters:
    - rows: List of stock data dictionaries
  - Returns: Sorted pandas DataFrame

### 6. yahoofinance/news.py

#### Classes

##### NewsAggregator

###### Methods

- `__init__(api_key: Optional[str] = None)`
  - Purpose: Initialize news aggregator with optional Google News API key
  - Parameters:
    - api_key: Google News API key from .env file

- `calculate_sentiment(title: str, summary: str) -> float`
  - Purpose: Calculate sentiment score for news article
  - Parameters:
    - title: Article title
    - summary: Article description/summary
  - Returns: Float between -1 (most negative) and +1 (most positive)
  - Implementation:
    - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
    - Specifically tuned for social media and short texts
    - Weights: 60% title, 40% summary
    - Handles missing summary gracefully
    - Returns compound score normalized between -1 and 1

- `get_sentiment_color(sentiment: float) -> str`
  - Purpose: Get color code based on sentiment value
  - Parameters:
    - sentiment: Float value between -1 and +1
  - Returns: ANSI color code
  - Thresholds:
    - Red: sentiment < -0.05 (negative)
    - Yellow: -0.05 ≤ sentiment ≤ 0.05 (neutral)
    - Green: sentiment > 0.05 (positive)

- `fetch_google_news(ticker: str, max_articles: int = 5) -> List[Dict[str, str]]`
  - Purpose: Fetch news from Google News API
  - Parameters:
    - ticker: Stock ticker symbol
    - max_articles: Maximum number of articles to return
  - Returns: List of news articles with title, description, url, date, and sentiment score

- `fetch_yahoo_news(ticker: str, max_articles: int = 5) -> List[Dict[str, str]]`
  - Purpose: Fetch news from Yahoo Finance
  - Parameters:
    - ticker: Stock ticker symbol
    - max_articles: Maximum number of articles to return
  - Returns: List of news articles with title, description, url, date, and sentiment score

### 7. yahoofinance/insiders.py

#### Classes

##### InsiderTrading

###### Methods

- `__init__(client: YFinanceClient)`
  - Purpose: Initialize with YFinance client
  - Parameters:
    - client: YFinanceClient instance

- `get_insider_transactions(ticker: str, days: int = 90) -> List[Dict[str, Any]]`
  - Purpose: Get insider trading data
  - Parameters:
    - ticker: Stock ticker symbol
    - days: Number of days to look back
  - Returns: List of insider transactions with date, insider name, role, shares, and value

- `calculate_insider_metrics(ticker: str) -> Dict[str, float]`
  - Purpose: Calculate insider trading metrics
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: Dictionary with buy_percentage and total_transactions

### 8. yahoofinance/holders.py

#### Functions

- `format_percentage(value: float) -> str`
  - Purpose: Format numeric values as percentages or comma-separated numbers
  - Parameters:
    - value: Number to format
  - Returns: Formatted string with % for values ≤ 1, comma-separated for values > 1

- `format_billions(value: float) -> str`
  - Purpose: Format large numbers in billions with dollar sign
  - Parameters:
    - value: Number to format
  - Returns: Formatted string in $B format

- `analyze_holders(ticker: str) -> None`
  - Purpose: Analyze and display institutional holders for a ticker
  - Parameters:
    - ticker: Stock ticker symbol
  - Output: Prints detailed analysis of institutional ownership

- `main() -> None`
  - Purpose: Command line interface for holders analysis
  - Features:
    - Handles multiple comma-separated tickers
    - Interactive input loop
    - Error handling for invalid tickers
    - Graceful exit with 'q' command

#### Data Structures

##### Major Holders DataFrame
- Index: ['insidersPercentHeld', 'institutionsPercentHeld', 'institutionsFloatPercentHeld', 'institutionsCount']
- Columns: ['Value']
- Types: float64

##### Institutional Holders DataFrame
- Columns:
  * Date Reported (datetime64[ns])
  * Holder (object)
  * pctHeld (float64)
  * Shares (int64)
  * Value (int64)

###### Methods

- `__init__(client: YFinanceClient)`
  - Purpose: Initialize with YFinance client
  - Parameters:
    - client: YFinanceClient instance

- `get_insider_transactions(ticker: str, days: int = 90) -> List[Dict[str, Any]]`
  - Purpose: Get insider trading data
  - Parameters:
    - ticker: Stock ticker symbol
    - days: Number of days to look back
  - Returns: List of insider transactions with date, insider name, role, shares, and value

- `calculate_insider_metrics(ticker: str) -> Dict[str, float]`
  - Purpose: Calculate insider trading metrics
  - Parameters:
    - ticker: Stock ticker symbol
  - Returns: Dictionary with buy_percentage and total_transactions

### 8. yahoofinance/index.py

#### Classes

##### None (Module Level Functions)

###### Functions

- `get_previous_trading_day_close(ticker: str, date: datetime) -> Tuple[pd.Series, datetime.date]`
  - Purpose: Get the closing price for the last trading day before the given date
  - Parameters:
    - ticker: Stock ticker symbol
    - date: Target date
  - Returns: Tuple of (price_series, actual_date)

- `calculate_weekly_dates() -> Tuple[datetime, datetime]`
  - Purpose: Calculate last Friday and the previous Friday
  - Returns: Tuple of (previous_friday, last_friday)

- `get_previous_month_ends() -> Tuple[datetime.date, datetime.date]`
  - Purpose: Calculate last business days of previous and previous previous month
  - Returns: Tuple of (previous_previous_month_end, previous_month_end)

- `fetch_changes(start_date: datetime.date, end_date: datetime.date) -> List[Dict]`
  - Purpose: Fetch price changes for indices between two dates
  - Parameters:
    - start_date: Start date for comparison
    - end_date: End date for comparison
  - Returns: List of dictionaries containing index changes

- `display_results(data: List[Dict]) -> None`
  - Purpose: Display results in a formatted table
  - Parameters:
    - data: List of dictionaries containing index data

- `main() -> None`
  - Purpose: Main function that handles user input and orchestrates the workflow
  - Features:
    - Interactive prompt for weekly/monthly choice
    - Automatic date calculation
    - Data fetching and display
    - HTML file updates

### 9. yahoofinance/economics.py

#### Classes

##### EconomicCalendar

###### Methods

- `__init__()`
  - Purpose: Initialize economic calendar with FRED API key
  - Environment Variables:
    - FRED_API_KEY: Required for FRED API access

- `validate_date_format(date_str: str) -> bool`
  - Purpose: Validate if date string matches YYYY-MM-DD format
  - Parameters:
    - date_str: Date string to validate
  - Returns: True if valid, False otherwise

- `get_economic_calendar(start_date: str, end_date: str) -> Optional[pd.DataFrame]`
  - Purpose: Get economic calendar for specified date range
  - Parameters:
    - start_date: Start date in YYYY-MM-DD format
    - end_date: End date in YYYY-MM-DD format
  - Returns: DataFrame with economic calendar information if available, None otherwise

- `_get_releases(start_date: str, end_date: str) -> List[Dict]`
  - Purpose: Get releases from FRED API
  - Parameters:
    - start_date: Start date
    - end_date: End date
  - Returns: List of release data

- `_get_release_series(release_id: str) -> List[Dict]`
  - Purpose: Get series for a specific release
  - Parameters:
    - release_id: FRED release ID
  - Returns: List of series data

- `_get_latest_value(series_id: str) -> str`
  - Purpose: Get latest value for a series
  - Parameters:
    - series_id: FRED series ID
  - Returns: Latest value formatted as string

#### Functions

- `format_economic_table(df: pd.DataFrame, start_date: str, end_date: str) -> None`
  - Purpose: Format and display economic calendar table
  - Parameters:
    - df: DataFrame with economic calendar data
    - start_date: Start date in YYYY-MM-DD format
    - end_date: End date in YYYY-MM-DD format

- `get_user_dates() -> Tuple[str, str]`
  - Purpose: Get start and end dates from user input
  - Returns: Tuple of start_date and end_date strings

#### Constants

- `indicators`: Dictionary mapping economic indicators to FRED series IDs
  - Categories:
    - Employment (Nonfarm Payrolls, Unemployment Rate, Initial Claims)
    - Inflation (CPI, Core CPI, PPI)
    - Growth (GDP, Retail Sales, Industrial Production)
  - Fields per indicator:
    - id: FRED series ID
    - impact: High/Medium importance
    - description: Human-readable description

## File Formats

### Input Files

#### portfolio.csv
```csv
ticker
AAPL
MSFT
GOOGL
```
- Required columns: ticker (stock symbol)
- Optional columns: None
- Format: UTF-8 encoded CSV
- Header row required

#### market.csv
```csv
symbol
AAPL
MSFT
GOOGL
```
- Required columns: symbol (stock symbol)
- Optional columns: None
- Format: UTF-8 encoded CSV
- Header row required

### Output Files

#### performance.html
- Purpose: Portfolio performance dashboard
- Format: HTML with embedded CSS
- Updates: Automatic on portfolio tracking execution
- Content: Portfolio metrics, returns, and risk measures

## API Rate Limiting

### Yahoo Finance API
- Default: 2000 requests per hour per IP
- Implemented backoff strategy:
  - Initial delay: 1 second
  - Maximum delay: 60 seconds
  - Exponential backoff multiplier: 2

### Google News API
- Free tier: 100 requests per day
- Paid tier: Based on subscription
- Error handling for rate limits with 429 status code

### FRED API
- Default: 120 requests per minute
- Error handling for rate limits with 429 status code
- Automatic retry with exponential backoff
- Cache implementation to minimize API calls

## Caching Strategy

1. File-Based Cache Implementation (cache.py)
   - JSON-based storage for persistence
   - Configurable cache directory
   - Default TTL: 15 minutes for news data
   - Automatic directory creation
   - Thread-safe file operations

2. Cache Keys
   - Format for Google News: f"google_news_{ticker}_{limit}"
   - Format for Yahoo Finance: f"yahoo_news_{ticker}"
   - Safe key hashing for filesystem compatibility
   - Includes timestamp for TTL calculation

3. Cache Invalidation
   - Automatic on TTL expiration
   - Manual clearing via clear() method
   - Automatic cleanup of expired/corrupted entries
   - Memory-efficient JSON storage

4. Error Handling
   - Graceful handling of corrupted cache files
   - Automatic cleanup of invalid entries
   - Safe concurrent access
   - Fallback to fresh data on cache miss

5. Implementation Details
   - Cache class with configurable parameters
   - Supports any JSON-serializable data
   - Transparent integration with news fetching
   - Maintains data freshness with TTL

## Security Considerations

1. API Key Protection
   - Store in .env file (not in version control)
   - Use environment variables
   - Implement key rotation capability

2. Data Validation
   - Input sanitization for all user inputs
   - Ticker symbol validation
   - Date format validation

3. Error Handling
   - Secure error messages (no sensitive data)
   - Logging without credentials
   - Rate limit adherence

## Troubleshooting Guide

1. API Connection Issues
   - Verify internet connection
   - Check API key validity
   - Confirm rate limit status
   - Review proxy settings if applicable

2. Data Quality Issues
   - Verify ticker symbols
   - Check for API service status
   - Confirm data freshness (cache status)
   - Review error logs

3. Performance Issues
   - Monitor memory usage
   - Check cache effectiveness
   - Review concurrent requests
   - Analyze response times

## Configuration Options

1. Display Configuration
   - Color output toggle
   - Date format customization
   - Number precision settings
   - Table format selection

2. Analysis Parameters
   - Minimum analyst coverage
   - Buy/Sell signal thresholds
   - Lookback periods
   - Cache TTL settings

3. API Configuration
   - Retry attempts
   - Timeout settings
   - Rate limit adjustments
   - Cache size control

## Data Flow

1. User Input Processing:
   - User selects data source (Portfolio, Market, Manual)
   - System loads tickers from selected source
   - Validates and deduplicates ticker symbols

2. Data Fetching:
   - YFinanceClient fetches raw data from Yahoo Finance API
   - Implements retry mechanism and caching
   - Validates and processes API responses

3. Data Analysis:
   - PricingAnalyzer calculates price metrics
   - AnalystData processes analyst ratings
   - Combines multiple data points for comprehensive analysis

4. Display Formatting:
   - DisplayFormatter applies formatting rules
   - Adds color coding based on analysis
   - Creates sortable DataFrame for display

5. Output Generation:
   - Generates formatted table output
   - Applies color coding for visual analysis
   - Sorts results by expected return and earnings dates

## Error Handling

1. Custom Exceptions:
   - YFinanceError: Base exception
   - APIError: API communication issues
   - ValidationError: Data validation failures

2. Retry Mechanism:
   - Configurable retry attempts
   - Exponential backoff
   - Detailed error logging

3. Data Validation:
   - Ticker symbol validation
   - Date format validation
   - Numeric value validation

## Performance Optimizations

1. Caching:
   - LRU cache for API calls
   - Configurable cache TTL
   - Memory-efficient storage

2. Data Processing:
   - Efficient DataFrame operations
   - Minimal data transformations
   - Optimized sorting algorithms

## Contributing Guidelines

1. Code Style:
   - Use type hints
   - Follow existing error handling patterns
   - Maintain comprehensive docstrings

2. Testing:
   - Add unit tests for new features
   - Test error handling
   - Verify cache behavior

3. Documentation:
   - Update technical specification
   - Document new functions
   - Include usage examples

## Future Enhancements

1. Additional Features:
   - Real-time data streaming
   - Technical analysis indicators
   - Portfolio optimization

2. Performance Improvements:
   - Parallel data fetching
   - Enhanced caching strategies
   - Optimized data structures

3. User Interface:
   - Web interface
   - Interactive charts
   - Custom alerts system

4. Data Sources:
   - Additional market data providers
   - Alternative data integration
   - Custom data source support

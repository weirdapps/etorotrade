# Market Analysis Tool Technical Specification

## Project Overview
This project is a command-line market analysis tool that fetches and analyzes stock market data from Yahoo Finance. It provides comprehensive analysis of stocks including price metrics, analyst ratings, insider trading information, and various financial ratios.

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

### 2. yahoofinance/client.py

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
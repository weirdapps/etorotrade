# Market Analysis Tool Technical Specification

## System Architecture

### 1. Core Components
```
Client Layer → Analysis Layer → Display Layer
     ↑              ↑               ↑
     └──── Rate Limiting System ────┘
```

#### Client Layer (yahoofinance/client.py)
- Handles all Yahoo Finance API interactions with yfinance
- Implements rate limiting, caching, and exponential backoff
- Manages data validation and error handling
- Provides StockData objects with comprehensive metrics

#### Analysis Layer (Multiple Modules)
- Processes raw data into actionable insights
- Implements business logic and metric calculations
- Handles data transformation and aggregation
- Specialized modules for analysts, pricing, insiders, earnings, etc.

#### Display Layer (yahoofinance/display.py)
- Manages output formatting and presentation
- Handles batch processing and progress tracking
- Implements multiple output formats (console, CSV, HTML)
- Color-codes results based on analysis metrics

### 2. Data Flow
```
Input → Rate Limiting → API Fetch → Processing → Output
  ↑          ↑             ↑           ↑          ↑
  └────────── Error Handling System ──────────────┘
```

#### Market Coverage
The system supports data fetching and analysis for major global indices:
- North America: S&P 500
- Western Europe:
  * UK: FTSE 100
  * France: CAC 40
  * Germany: DAX
  * Spain: IBEX 35
  * Italy: FTSE MIB
  * Portugal: PSI
  * Switzerland: SMI
  * Denmark: OMXC25
  * Greece: ATHEX
- Asia:
  * Japan: Nikkei 225
  * Hong Kong: Hang Seng

## Implementation Details

### 1. Rate Limiting System

#### RateLimitTracker Class
```python
class RateLimitTracker:
    def __init__(self):
        self.window_size = 60    # Time window in seconds
        self.max_calls = 100     # Maximum calls per window
        self.calls = deque(maxlen=1000) # Timestamp queue
        self.errors = deque(maxlen=20)  # Recent errors
        self.base_delay = 2.0    # Base delay between calls
        self.min_delay = 1.0     # Minimum delay
        self.max_delay = 30.0    # Maximum delay
        self.batch_delay = 5.0   # Delay between batches
        self.error_counts = {}   # Track errors per ticker
        self.success_streak = 0  # Track successful calls
```

#### Adaptive Delay System
- Base delay: 2 seconds (adjustable based on success)
- Success streak monitoring (reduces delay on consecutive successes)
- Error pattern detection (increases delay on repeated errors)
- Exponential backoff (doubles delay after significant failures)
- Batch processing optimization (15 tickers per batch)
- Ticker-specific delay adjustments

### 2. Data Structures

#### StockData Class
```python
@dataclass
class StockData:
    # Basic Info (Required)
    name: str
    sector: str
    recommendation_key: str
    
    # Market Data (Optional)
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    price_change_percentage: Optional[float] = None
    mtd_change: Optional[float] = None
    ytd_change: Optional[float] = None
    two_year_change: Optional[float] = None
    
    # Analyst Coverage (Optional)
    recommendation_mean: Optional[float] = None
    analyst_count: Optional[int] = None
    
    # Valuation Metrics (Optional)
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    
    # Financial Health (Optional)
    quick_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    # Risk Metrics (Optional)
    short_float_pct: Optional[float] = None
    short_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    cash_percentage: Optional[float] = None
    
    # Dividends (Optional)
    dividend_yield: Optional[float] = None
    
    # Events (Optional)
    last_earnings: Optional[str] = None
    previous_earnings: Optional[str] = None
    
    # Insider Activity (Optional)
    insider_buy_pct: Optional[float] = None
    insider_transactions: Optional[int] = None
    
    # Internal (Optional)
    ticker_object: Optional[yf.Ticker] = field(default=None)
```

#### Display Formatting
```python
@dataclass
class DisplayConfig:
    use_colors: bool = True
    date_format: str = "%Y-%m-%d"
    float_precision: int = 2
    percentage_precision: int = 1
    table_format: str = "fancy_grid"
    min_analysts: int = 4       # Minimum analysts for high confidence
    high_upside: float = 15.0   # Threshold for buy signal (%)
    low_upside: float = 5.0     # Threshold for sell signal (%)
    high_buy_percent: float = 65.0  # Threshold for strong buy (%)
```

### 3. Error Handling

#### Exception Hierarchy
```python
class YFinanceError(Exception):
    """Base exception for YFinance operations"""

class APIError(YFinanceError):
    """API communication errors"""

class ValidationError(YFinanceError):
    """Data validation errors"""
```

#### Error Recovery Strategy
1. Retry mechanism with exponential backoff
2. Fallback to empty reports for problematic tickers
3. Comprehensive error logging
4. Skip tickers with persistent errors
5. Batch success rate monitoring

## Technical Requirements

### 1. System Requirements
- CPU: 1+ core
- RAM: 2GB minimum, 4GB recommended
- Storage: 100MB for installation and cache
- Internet: Stable connection required

### 2. Software Requirements
- Python 3.8+
- Operating System: Cross-platform
- Dependencies: See requirements.txt

### 3. API Requirements
- Yahoo Finance API access (via yfinance package)
- FRED API key for economic data
- Optional: News API key

## Dependencies

### Core Dependencies
```
beautifulsoup4>=4.13.3  # Web scraping
pandas>=2.2.3          # Data manipulation
pytz>=2025.1          # Timezone handling
requests>=2.32.3      # HTTP requests
tabulate>=0.9.0       # Table formatting
tqdm>=4.67.1          # Progress bars
yfinance>=0.2.54      # Yahoo Finance API
nltk>=3.9.1          # Natural language processing
vaderSentiment>=3.3.2 # Sentiment analysis
```

### Development Dependencies
```
pytest>=8.3.4         # Testing framework
pytest-cov>=6.0.0     # Coverage reporting
pytest-mock>=3.14.0   # Test mocking
```

## File Structure

### Input/Output
```
yahoofinance/
├── input/
│   ├── portfolio.csv    # Portfolio holdings
│   ├── market.csv      # Market watchlist
│   └── cons.csv        # Constants
├── output/
│   ├── portfolio.html  # Portfolio dashboard
│   ├── index.html     # Market performance
│   ├── market.csv     # Market data export
│   └── portfolio.csv  # Portfolio data export
└── cache/             # API response cache
```

### Source Code
```
yahoofinance/
├── __init__.py
├── _metrics.py         # Internal metrics calculations
├── client.py          # API client
├── types.py           # Data structures & exceptions
├── analyst.py         # Analyst data processing
├── cache.py           # Caching system
├── cons.py            # Constants
├── display.py         # Output handling
├── download.py        # Data downloading
├── earnings.py        # Earnings data
├── econ.py            # Economic indicators
├── formatting.py      # Data formatting
├── holders.py         # Institutional holders
├── index.py           # Market indices
├── insiders.py        # Insider transactions
├── monthly.py         # Monthly data
├── news.py            # News and sentiment
├── portfolio.py       # Portfolio management
├── pricing.py         # Price analysis
├── templates.py       # HTML templates
├── utils.py           # Utilities
└── weekly.py          # Weekly data
```

## Performance Specifications

### 1. Rate Limiting
- Maximum calls: 2000/hour per IP
- Batch size: 15 tickers
- Base delay: 2 seconds between calls
- Batch delay: 5-30 seconds between batches
- Maximum delay: 30 seconds (during API stress)
- Success threshold: 80% (for delay adjustment)

### 2. Caching
- Market data TTL: 300 seconds (5 minutes)
- News data TTL: 900 seconds (15 minutes)
- LRU cache for ticker info (maxsize=50)

### 3. Response Times
- Single API call: < 5 seconds
- Batch processing: 30-60 seconds per batch
- Data export: < 10 seconds
- HTML generation: < 5 seconds

## Security Considerations

### 1. API Key Protection
- Environment variable storage
- Access logging
- Rate monitoring

### 2. Data Validation
- Input sanitization
- Type checking and validation
- Range validation
- Comprehensive error boundaries

### 3. Error Handling
- Secure error messages
- Logging without credentials
- Rate limit adherence
- Retry mechanisms with backoff

## Testing Strategy

### 1. Unit Tests
- Component isolation
- Mock API responses
- Error condition testing
- Edge case validation

### 2. Integration Tests
- API interaction
- Data flow validation
- Rate limiting verification
- Display formatting

### 3. Specialized Tests
- Market display formatting
- Batch processing
- HTML output
- CSV generation

## Monitoring and Logging

### 1. Performance Metrics
- API response times
- Success/error rates
- Cache hit rates
- Batch success rates

### 2. Error Tracking
- API errors
- Rate limit violations
- Data validation failures
- Display formatting issues

### 3. User Feedback
- Progress bars
- Error messages
- Completion notifications
- Performance statistics

## Future Considerations

### 1. Technical Improvements
- Parallel processing
- Enhanced caching
- WebSocket support
- Alternative data sources

### 2. Feature Additions
- Machine learning analysis
- Technical indicators
- Portfolio optimization
- Real-time alerts

### 3. Interface Enhancements
- Interactive dashboards
- Data visualization
- Custom metrics
- Mobile support

This technical specification provides a comprehensive guide for development, maintenance, and future enhancement of the Trade project.
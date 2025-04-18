"""
Configuration settings for Yahoo Finance data access.

This module defines configuration settings for rate limiting, caching,
API timeouts, and more. It provides a central location for all configuration
values used throughout the package.
"""

import os
from typing import Dict, Any, List, Set

# Define set of analyst grades considered positive
POSITIVE_GRADES = {
    "Buy", "Outperform", "Strong Buy", "Overweight", "Accumulate", 
    "Add", "Conviction Buy", "Top Pick", "Positive"
}

# Provider configuration
PROVIDER_CONFIG = {
    # Enable yahooquery supplementation in hybrid provider
    "ENABLE_YAHOOQUERY": False,  # Set to False to disable yahooquery and prevent crumb errors
}

# Rate limiting configuration - OPTIMIZED FOR DIRECT API ACCESS
RATE_LIMIT = {
    # Time window for rate limiting in seconds (60s = 1 minute window)
    "WINDOW_SIZE": 60,
    
    # Maximum API calls per minute window - Yahoo Finance generally allows 100-120
    # Using 75 as a conservative value to prevent rate limiting
    "MAX_CALLS": 75,  # Increased from 60
    
    # Base delay between calls in seconds - reduced for faster API access
    # This is a starting point and will adaptively adjust based on API response
    "BASE_DELAY": 0.3,  # Reduced from 0.5
    
    # Minimum delay after many successful calls in seconds
    # Can go lower with direct API access (no cache overhead)
    "MIN_DELAY": 0.1,  # Reduced from 0.2
    
    # Maximum delay after errors in seconds - kept high to prevent bans
    "MAX_DELAY": 30.0,
    
    # Success threshold for delay reduction - after this many consecutive 
    # successful calls, we'll reduce the delay
    "SUCCESS_THRESHOLD": 5,  # New setting to reduce delay more quickly
    
    # Delay reduction factor - multiplier applied to current delay after
    # SUCCESS_THRESHOLD consecutive successful API calls
    "SUCCESS_DELAY_REDUCTION": 0.8,  # New setting (20% reduction)
    
    # Number of items per batch - increased for more parallel processing
    "BATCH_SIZE": 15,  # Increased from 10
    
    # Delay between batches in seconds - minimized for faster processing
    "BATCH_DELAY": 0.5,  # Reduced from 1.0
    
    # Maximum retry attempts for API calls
    "MAX_RETRY_ATTEMPTS": 3,
    
    # API request timeout in seconds
    "API_TIMEOUT": 60,  # Keep longer timeout for stability
    
    # Maximum concurrent API calls (for async)
    "MAX_CONCURRENT_CALLS": 15,  # Increased from 10
    
    # Jitter factor for randomizing delays (helps prevent rate limit detection)
    "JITTER_FACTOR": 0.2,  # New setting - adds ±20% randomness to delays
    
    # Error count threshold - after this many errors, we'll increase delay
    "ERROR_THRESHOLD": 2,  # New setting
    
    # Error delay increase factor - multiplier applied to current delay after
    # ERROR_THRESHOLD consecutive failed API calls
    "ERROR_DELAY_INCREASE": 1.5,  # New setting
    
    # Rate limit error delay increase factor - applied when a rate limit error is detected
    "RATE_LIMIT_DELAY_INCREASE": 2.0,  # New setting
    
    # Ticker priority tiers - HIGH priority tickers get processed faster with lower delays
    # MEDIUM priority tickers use standard delays
    # LOW priority tickers use higher delays
    "TICKER_PRIORITY": {
        "HIGH": 0.7,    # 30% delay reduction
        "MEDIUM": 1.0,  # Standard delay
        "LOW": 1.5,     # 50% delay increase
    },
    
    # Problematic tickers that should use longer delays - updated to be tightly focused
    "SLOW_TICKERS": set(),
    
    # VIP tickers that should always process with highest priority
    "VIP_TICKERS": set(),
    
    # Cache-aware settings - still relevant for API health even without caching
    "CACHE_AWARE_RATE_LIMITING": False,  # Disabled since cache is off
    
    # Market hours delay multipliers
    "MARKET_HOURS_DELAY_MULTIPLIER": 1.0,  # Regular delay during market hours
    "OFF_MARKET_DELAY_MULTIPLIER": 1.5,    # Reduced from 2.0 for faster processing
    
    # Region-specific delay multipliers - optimized values
    "US_DELAY_MULTIPLIER": 1.0,        # Standard delay for US tickers
    "EUROPE_DELAY_MULTIPLIER": 1.1,    # Slight adjustment (reduced from 1.2)
    "ASIA_DELAY_MULTIPLIER": 1.2,      # Reduced from 1.5 for faster overall processing
    
    # Adaptive strategy settings - new advanced configuration
    "ENABLE_ADAPTIVE_STRATEGY": True,  # New setting - enables runtime strategy adaptation
    "MONITOR_INTERVAL": 60,           # Seconds between rate limiting strategy adjustments
    "MAX_ERROR_RATE": 0.05,           # 5% maximum allowable error rate before adjusting
    "MIN_SUCCESS_RATE": 0.95,         # 95% minimum success rate target
}

# Circuit breaker configuration
CIRCUIT_BREAKER = {
    # Failure threshold to trip the circuit breaker
    "FAILURE_THRESHOLD": 5,
    
    # Time window in seconds to count failures
    "FAILURE_WINDOW": 60,
    
    # Recovery timeout in seconds before circuit half-opens
    "RECOVERY_TIMEOUT": 300,
    
    # Maximum consecutive successes required to close circuit
    "SUCCESS_THRESHOLD": 3,
    
    # Percentage of requests to allow through in half-open state
    "HALF_OPEN_ALLOW_PERCENTAGE": 10,
    
    # Maximum time in seconds a circuit can stay open
    "MAX_OPEN_TIMEOUT": 1800,  # 30 minutes
    
    # Enable circuit breaker by default
    "ENABLED": True,
    
    # Path to persistent circuit state file
    "STATE_FILE": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "circuit_state.json"
    ),
}

# Caching configuration - COMPLETELY DISABLED FOR BETTER PERFORMANCE
# Testing showed that direct API calls without caching provide equal or better performance
CACHE_CONFIG = {
    # Memory cache disabled - direct API calls performing better
    "ENABLE_MEMORY_CACHE": False,
    
    # Disk cache disabled - avoids I/O overhead
    "ENABLE_DISK_CACHE": False,
    
    # Memory-only mode (not relevant when both caches disabled)
    "MEMORY_ONLY_MODE": True,
    
    # Memory cache size (items) - significantly increased for better hit rates
    "MEMORY_CACHE_SIZE": 10000,
    
    # Default memory cache TTL (seconds)
    "MEMORY_CACHE_TTL": 300,  # 5 minutes
    
    # Thread-local cache size for frequently accessed keys
    "THREAD_LOCAL_CACHE_SIZE": 100,
    
    # Enable ultra-fast path optimizations
    "ENABLE_ULTRA_FAST_PATH": True,
    
    # Batch update threshold (minimum items for using batch updates)
    "BATCH_UPDATE_THRESHOLD": 5,
    
    # Error caching (cache error responses to avoid redundant failed requests)
    "CACHE_ERRORS": True,
    
    # Error cache TTL (shorter TTL for error responses)
    "ERROR_CACHE_TTL": 60,  # 1 minute
    
    # Disk cache size (MB) - not used when disabled
    "DISK_CACHE_SIZE_MB": 100,
    
    # Default disk cache TTL (seconds) - not used when disabled
    "DISK_CACHE_TTL": 3600,  # 1 hour
    
    # Disk cache directory
    "DISK_CACHE_DIR": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "cache"
    ),
    
    # TTL settings by data type (seconds)
    # Basic company information
    "TICKER_INFO_MEMORY_TTL": 86400,    # 1 day
    "TICKER_INFO_DISK_TTL": 604800,     # 1 week
    
    # Current price, volume data
    "MARKET_DATA_MEMORY_TTL": 60,       # 1 minute
    "MARKET_DATA_DISK_TTL": 180,        # 3 minutes
    
    # PE ratios, PEG, financial metrics
    "FUNDAMENTALS_MEMORY_TTL": 60,      # 1 minute
    "FUNDAMENTALS_DISK_TTL": 180,       # 3 minutes
    
    # News articles and headlines
    "NEWS_MEMORY_TTL": 600,             # 10 minutes
    "NEWS_DISK_TTL": 1200,              # 20 minutes
    
    # Analyst ratings and recommendations
    "ANALYSIS_MEMORY_TTL": 600,         # 10 minutes
    "ANALYSIS_DISK_TTL": 1200,          # 20 minutes
    
    # Historical price data
    "HISTORICAL_DATA_MEMORY_TTL": 86400,    # 1 day
    "HISTORICAL_DATA_DISK_TTL": 172800,     # 2 days
    
    # Earnings dates and data
    "EARNINGS_DATA_MEMORY_TTL": 600,    # 10 minutes
    "EARNINGS_DATA_DISK_TTL": 1200,     # 20 minutes
    
    # Insider trading information
    "INSIDER_TRADES_MEMORY_TTL": 86400, # 1 day
    "INSIDER_TRADES_DISK_TTL": 172800,  # 2 days
    
    # Dividend information
    "DIVIDEND_DATA_MEMORY_TTL": 86400,  # 1 day
    "DIVIDEND_DATA_DISK_TTL": 172800,   # 2 days
    
    # Target price information
    "TARGET_PRICE_MEMORY_TTL": 600,     # 10 minutes
    "TARGET_PRICE_DISK_TTL": 1200,      # 20 minutes
    
    # Missing data cache (special longer TTL for known missing data)
    "MISSING_DATA_MEMORY_TTL": 259200,  # 3 days
    "MISSING_DATA_DISK_TTL": 604800,    # 7 days
    
    # Differential TTL based on stock origin
    "US_STOCK_TTL_MULTIPLIER": 1.0,     # Standard TTL for US stocks
    "NON_US_STOCK_TTL_MULTIPLIER": 2.0, # Double TTL for non-US stocks that update less frequently
}

# Risk metrics configuration
RISK_METRICS = {
    # Risk-free rate (annual)
    "RISK_FREE_RATE": 0.03,
    
    # Trading days per year
    "TRADING_DAYS_PER_YEAR": 252,
}

# Pagination configuration
PAGINATION = {
    # Default page size
    "DEFAULT_PAGE_SIZE": 20,
    
    # Maximum page size
    "MAX_PAGE_SIZE": 100,
    
    # Default first page index
    "DEFAULT_FIRST_PAGE": 1,
    
    # Default sort order
    "DEFAULT_SORT_ORDER": "desc",
    
    # Default page size for internal API calls
    "PAGE_SIZE": 20,
    
    # Maximum number of pages to fetch
    "MAX_PAGES": 10,
    
    # Maximum number of retry attempts
    "MAX_RETRIES": 3,
    
    # Delay in seconds between retries
    "RETRY_DELAY": 1.0,
}

# Trading criteria configuration
TRADING_CRITERIA = {
    "CONFIDENCE": {
        # Minimum number of analysts covering the stock
        "MIN_ANALYST_COUNT": 5,
        
        # Minimum number of price targets
        "MIN_PRICE_TARGETS": 5,
    },
    "SELL": {
        # Maximum upside potential for sell recommendation (sell if below this)
        "SELL_MAX_UPSIDE": 5.0,
        
        # Minimum buy percentage for sell recommendation (sell if below this)
        "SELL_MIN_BUY_PERCENTAGE": 65.0,
        
        # Minimum forward P/E for sell recommendation (sell if above this)
        "SELL_MIN_FORWARD_PE": 50.0,
        
        # Minimum PEG ratio for sell recommendation (sell if above this)
        "SELL_MIN_PEG": 3.0,
        
        # Minimum short interest for sell recommendation (sell if above this)
        "SELL_MIN_SHORT_INTEREST": 2.0,
        
        # Minimum beta for sell recommendation (sell if above this)
        "SELL_MIN_BETA": 3.0,
        
        # Maximum expected return for sell recommendation (sell if below this)
        "SELL_MAX_EXRET": 5.0,
    },
    "BUY": {
        # Minimum upside potential for buy recommendation (buy if above this)
        "BUY_MIN_UPSIDE": 20.0,
        
        # Minimum buy percentage for buy recommendation (buy if above this)
        "BUY_MIN_BUY_PERCENTAGE": 85.0,
        
        # Minimum beta for buy recommendation (buy if above this)
        "BUY_MIN_BETA": 0.25,
        
        # Maximum beta for buy recommendation (buy if below this)
        "BUY_MAX_BETA": 2.5,
        
        # Minimum forward P/E for buy recommendation (buy if above this)
        "BUY_MIN_FORWARD_PE": 0.5,
        
        # Maximum forward P/E for buy recommendation (buy if below this)
        "BUY_MAX_FORWARD_PE": 45.0,
        
        # Maximum PEG ratio for buy recommendation (buy if below this)
        "BUY_MAX_PEG": 2.5,
        
        # Maximum short interest for buy recommendation (buy if below this)
        "BUY_MAX_SHORT_INTEREST": 1.5,
        
        # Minimum expected return for buy recommendation (buy if above this)
        "BUY_MIN_EXRET": 15.0,
    },
}

# Display configuration
DISPLAY = {
    # Maximum company name length
    "MAX_COMPANY_NAME_LENGTH": 14,
    
    # Default display columns
    "DEFAULT_COLUMNS": [
        "ticker", 
        "company", 
        "market_cap", 
        "price", 
        "target_price", 
        "upside", 
        "analyst_count",
        "buy_percentage", 
        "total_ratings", 
        "beta",
        "pe_trailing", 
        "pe_forward", 
        "peg_ratio", 
        "dividend_yield",
        "short_float_pct"
    ],
    
    # Column formatters
    "FORMATTERS": {
        "price": {"precision": 2},
        "target_price": {"precision": 2},
        "upside": {"precision": 1, "as_percentage": True},
        "buy_percentage": {"precision": 0, "as_percentage": True},
        "beta": {"precision": 2},
        "pe_trailing": {"precision": 1},
        "pe_forward": {"precision": 1},
        "peg_ratio": {"precision": 1},
        "dividend_yield": {"precision": 2, "as_percentage": True},
        "short_float_pct": {"precision": 1, "as_percentage": True},
    },
}

# Message constants
MESSAGES = {
    # Error messages
    "NO_DATA_FOUND_TICKER": "No data found for {ticker}",
    "NO_TICKERS_FOUND": "No tickers found or provided.",
    "NO_NEWS_FOUND_TICKER": "No news found for {ticker}",
    "NO_PORTFOLIO_TICKERS_FOUND": "No portfolio tickers found.",
    "NO_MARKET_TICKERS_FOUND": "No market tickers found.",
    "NO_PROVIDER_AVAILABLE": "No provider available. Please initialize with a provider.",
    "NO_RESULTS_AVAILABLE": "No results available.",
    "ANALYSIS_FAILED": "Analysis failed: {error}",
    
    # Common error messages
    "ERROR_FETCHING_DATA": "Error fetching data for {ticker}: {error}",
    "ERROR_FETCHING_NEWS": "Error fetching news for {ticker}: {error}",
    "ERROR_FETCHING_NEWS_ASYNC": "Error fetching news async for {ticker}: {error}",
    "ERROR_ANALYZING_TICKER": "Error analyzing {ticker}: {error}",
    "ERROR_PROCESSING_NEWS": "Error processing news item: {error}",
    "ERROR_LOADING_TICKERS": "Error loading tickers: {error}",
    "ERROR_READING_PORTFOLIO": "Error reading portfolio file: {error}",
    "ERROR_ANALYZING_PORTFOLIO": "Error analyzing portfolio: {error}",
    "ERROR_GENERATING_BUY_RECOMMENDATIONS": "Error generating buy recommendations: {error}",
    "ERROR_GENERATING_SELL_RECOMMENDATIONS": "Error generating sell recommendations: {error}",
    "ERROR_GENERATING_HOLD_RECOMMENDATIONS": "Error generating hold recommendations: {error}",
    "ERROR_BATCH_FETCH": "Error fetching batch data: {error}",
    "ERROR_BATCH_FETCH_ASYNC": "Error fetching batch data asynchronously: {error}",
    "ERROR_INVALID_SOURCE": "Invalid source: {source}. Must be one of: {valid_sources} or I",
    "ERROR_LOADING_FILE": "Error loading tickers from {file_path}: {error}",
    "ERROR_TICKER_COLUMN_NOT_FOUND": "Ticker column not found in {file_path}. Expected one of: {columns}",
    
    # User prompts
    "PROMPT_ENTER_TICKERS": "Enter comma-separated tickers (e.g., AAPL,MSFT,GOOGL): ",
    "PROMPT_ENTER_TICKERS_DISPLAY": "Enter tickers separated by commas: ",
    "PROMPT_TICKER_SOURCE": "\nSelect ticker input method:",
    "PROMPT_TICKER_SOURCE_OPTIONS": "P - Load tickers from portfolio.csv\nI - Enter tickers manually",
    "PROMPT_TICKER_SOURCE_CHOICE": "\nEnter your choice (P/I): ",
    "PROMPT_INVALID_CHOICE": "Invalid choice. Please enter 'P' or 'I'.",
    
    # Informational messages
    "INFO_FETCHING_DATA": "Fetching market data...",
    "INFO_FETCHING_NEWS": "\nFetching news for: {tickers}",
    "INFO_ANALYZING_PORTFOLIO": "Analyzing {count} portfolio tickers...",
    "INFO_ANALYZING_MARKET": "Analyzing {count} market tickers...",
    "INFO_CIRCUIT_BREAKER_STATUS": "Circuit breaker status: {status}",
    "INFO_MARKET_ANALYSIS_COMPLETE": "Market analysis complete. Results saved to {path}",
    "INFO_PORTFOLIO_ANALYSIS_COMPLETE": "Portfolio analysis complete. Results saved to {path}",
    "INFO_FETCHING_TICKER_DATA": "Fetching data for ticker: {ticker}",
    "INFO_TICKERS_LOADED": "Loaded {count} tickers from {file_path}",
    "INFO_PROCESSING_TICKERS": "Processing {count} tickers...",
}

# Paths configuration
PATHS = {
    # Input directory
    "INPUT_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "input"),
    
    # Output directory
    "OUTPUT_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
    
    # Log directory
    "LOG_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"),
    
    # Default log file
    "DEFAULT_LOG_FILE": os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "logs", 
        "yahoofinance.log"
    ),
}

# File paths for data files
FILE_PATHS = {
    # Input files
    "MARKET_FILE": os.path.join(PATHS["INPUT_DIR"], "market.csv"),
    "PORTFOLIO_FILE": os.path.join(PATHS["INPUT_DIR"], "portfolio.csv"),
    "ETORO_FILE": os.path.join(PATHS["INPUT_DIR"], "etoro.csv"),
    "YFINANCE_FILE": os.path.join(PATHS["INPUT_DIR"], "yfinance.csv"),
    "NOTRADE_FILE": os.path.join(PATHS["INPUT_DIR"], "notrade.csv"),
    "CONS_FILE": os.path.join(PATHS["INPUT_DIR"], "cons.csv"),
    "US_TICKERS_FILE": os.path.join(PATHS["INPUT_DIR"], "us_tickers.csv"),
    "EUROPE_FILE": os.path.join(PATHS["INPUT_DIR"], "europe.csv"),
    "CHINA_FILE": os.path.join(PATHS["INPUT_DIR"], "china.csv"),
    "USA_FILE": os.path.join(PATHS["INPUT_DIR"], "usa.csv"),
    "USINDEX_FILE": os.path.join(PATHS["INPUT_DIR"], "usindex.csv"),
    
    # Output files
    "MARKET_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "market.csv"),
    "PORTFOLIO_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "portfolio.csv"),
    "BUY_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "buy.csv"),
    "SELL_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "sell.csv"),
    "HOLD_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "hold.csv"),
    "MANUAL_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "manual.csv"),
    "HTML_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "index.html"),
    "PORTFOLIO_HTML": os.path.join(PATHS["OUTPUT_DIR"], "portfolio_dashboard.html"),
    "PORTFOLIO_PERFORMANCE_JSON": os.path.join(PATHS["OUTPUT_DIR"], "portfolio_performance.json"),
    "MONTHLY_PERFORMANCE_JSON": os.path.join(PATHS["OUTPUT_DIR"], "monthly_performance.json"),
    "WEEKLY_PERFORMANCE_JSON": os.path.join(PATHS["OUTPUT_DIR"], "weekly_performance.json"),
    "CSS_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "styles.css"),
    "JS_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "script.js"),
}

# Special tickers configuration
SPECIAL_TICKERS = {
    # US stocks with dots in their symbols
    "US_SPECIAL_CASES": {
        'BRK.A', 'BRK.B',  # Berkshire Hathaway
        'BF.A', 'BF.B',    # Brown-Forman
    },
}

# Column and field name constants
COLUMN_NAMES = {
    # Display column names
    "EARNINGS_DATE": "Earnings Date",
    "BUY_PERCENTAGE": "% BUY",
    "DIVIDEND_YIELD": "DIV %",
    "COMPANY_NAME": "COMPANY",
    "TICKER": "TICKER",
    "MARKET_CAP": "CAP",
    "PRICE": "PRICE",
    "TARGET_PRICE": "TARGET",
    "UPSIDE": "UPSIDE",
    "ANALYST_COUNT": "# T",
    "TOTAL_RATINGS": "# A",
    "ACTION": "ACT",
    "POSITION_SIZE": "SIZE",
    "RATING_TYPE": "A",
    "EXPECTED_RETURN": "EXRET",
    "BETA": "BETA",
    "PE_TRAILING": "PET",
    "PE_FORWARD": "PEF",
    "PEG_RATIO": "PEG",
    "SHORT_INTEREST": "SI",
    "EARNINGS": "EARNINGS",
    "RANKING": "#",
}

# Standard display column order for all views
# This is the canonical column order that must be used for all displays
STANDARD_DISPLAY_COLUMNS = [
    "#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE",
    "# T", "% BUY", "# A", "A", "EXRET", "BETA", "PET", "PEF", "PEG",
    "DIV %", "SI", "EARNINGS", "SIZE", "ACT"
]

# Load environment variables if needed
def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary containing configuration values from environment variables
    """
    config = {}
    
    # Rate limit settings
    if 'YFINANCE_MAX_CALLS' in os.environ:
        config['RATE_LIMIT.MAX_CALLS'] = int(os.environ['YFINANCE_MAX_CALLS'])
    
    # Cache settings
    if 'YFINANCE_CACHE_TTL' in os.environ:
        config['CACHE_CONFIG.MEMORY_CACHE_TTL'] = int(os.environ['YFINANCE_CACHE_TTL'])
    
    # API settings
    if 'YFINANCE_API_TIMEOUT' in os.environ:
        config['RATE_LIMIT.API_TIMEOUT'] = int(os.environ['YFINANCE_API_TIMEOUT'])
    
    # Circuit breaker settings
    if 'YFINANCE_CIRCUIT_BREAKER_ENABLED' in os.environ:
        config['CIRCUIT_BREAKER.ENABLED'] = os.environ['YFINANCE_CIRCUIT_BREAKER_ENABLED'].lower() == 'true'
    
    return config

# Apply environment variable configuration
ENV_CONFIG = load_env_config()

# Update configuration with environment variables
def apply_env_config(env_config: Dict[str, Any]) -> None:
    """
    Apply environment variable configuration.
    
    Args:
        env_config: Dictionary containing configuration values from environment variables
    """
    for key, value in env_config.items():
        parts = key.split('.')
        if len(parts) == 2:
            module_name, setting_name = parts
            if module_name in globals() and setting_name in globals()[module_name]:
                globals()[module_name][setting_name] = value

# Apply environment configuration
apply_env_config(ENV_CONFIG)

# Performance benchmarking configuration
PERFORMANCE_CONFIG = {
    # Enable memory profiling for benchmarks
    "ENABLE_MEMORY_PROFILING": True,
    
    # Directory for benchmark results
    "RESULTS_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "benchmarks"),
    
    # Directory for baseline results
    "BASELINE_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "benchmarks"),
    
    # Benchmark settings
    "BENCHMARK": {
        "BENCHMARK_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "benchmarks"),
        "SAMPLE_TICKERS": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"],
        "BASELINE_FILE": "baseline_performance.json",
        "MEMORY_PROFILE_THRESHOLD": 1.2,  # 20% increase is concerning
        "RESOURCE_MONITOR_INTERVAL": 0.5,  # seconds
        "MAX_BENCHMARK_DURATION": 300,  # 5 minutes max for any benchmark
        "DEFAULT_ITERATIONS": 3,
        "DEFAULT_WARMUP_ITERATIONS": 1,
    }
}
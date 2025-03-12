"""
Centralized configuration for the Yahoo Finance API client.

This module contains constants and configuration settings used throughout the package,
including rate limiting parameters, cache settings, API timeouts, and finance-specific constants.
"""

from typing import Dict, Set

# Rate Limiting Configuration
RATE_LIMIT = {
    "WINDOW_SIZE": 60,           # Time window in seconds
    "MAX_CALLS": 100,            # Maximum calls per window
    "BASE_DELAY": 2.0,           # Base delay between calls
    "MIN_DELAY": 1.0,            # Minimum delay
    "MAX_DELAY": 30.0,           # Maximum delay
    "BATCH_DELAY": 5.0,          # Delay between batches
    "MAX_RETRY_ATTEMPTS": 3,     # Maximum retry attempts
    "API_TIMEOUT": 10,           # Timeout in seconds for API calls
    "BATCH_SIZE": 15,            # Default batch size
}

# Cache Configuration
CACHE = {
    "MARKET_DATA_TTL": 5,        # Market data cache expiration in minutes
    "NEWS_DATA_TTL": 15,         # News cache expiration in minutes
    "EARNINGS_DATA_TTL": 60,     # Earnings data cache expiration in minutes
    "DEFAULT_TTL": 15,           # Default cache expiration in minutes
    "MAX_CACHE_ENTRIES": 500,    # Maximum number of cache entries
}

# Risk Metrics
RISK_METRICS = {
    "RISK_FREE_RATE": 0.05,      # 5% annual risk-free rate
    "TRADING_DAYS_PER_YEAR": 252,# Trading days in a year
}

# Analyst Ratings
POSITIVE_GRADES: Set[str] = {
    "Buy", 
    "Overweight", 
    "Outperform", 
    "Strong Buy", 
    "Long-Term Buy", 
    "Positive"
}

# Trading Criteria
TRADING_CRITERIA = {
    # Common criteria for all trading recommendations
    "COMMON": {
        "MIN_ANALYST_COUNT": 5,      # Minimum number of analyst price targets
        "MIN_RATINGS_COUNT": 5,      # Minimum number of analyst ratings
    },
    
    # Buy criteria
    "BUY": {
        "MIN_UPSIDE": 20.0,          # Minimum upside percentage (>=)
        "MIN_BUY_PERCENTAGE": 82.0,  # Minimum buy percentage (>=)
        "MAX_BETA": 3.0,             # Maximum beta value (<=)
        "MIN_BETA": 0.2,             # Minimum beta value (>)
        "MIN_PE_FORWARD": 0.5,       # Minimum forward P/E (>)
        "MAX_PE_FORWARD": 45.0,      # Maximum forward P/E (<=)
        "MAX_PEG_RATIO": 3.0,        # Maximum PEG ratio (<)
        "MAX_SHORT_INTEREST": 3.0,   # Maximum short interest percentage (<=)
        "PE_CONDITION": "PEF < PET"  # PE Forward should be less than PE Trailing
    },
    
    # Sell criteria - any of these conditions trigger a sell recommendation
    "SELL": {
        "MAX_UPSIDE": 5.0,           # Maximum upside percentage (<)
        "MAX_BUY_PERCENTAGE": 65.0,  # Maximum buy percentage (<)
        # "MIN_PE_FORWARD": 0.5,     # Removed - No longer consider low PEF as a sell signal
        "MAX_PEG_RATIO": 3.0,        # Maximum PEG ratio - above this is a sell signal (>)
        "MIN_SHORT_INTEREST": 4.0,   # Minimum short interest - above this is a sell signal (>)
        "MAX_BETA": 3.0,             # Maximum beta - above this is a sell signal (>)
        "MIN_PE_FORWARD": 45.0,      # Minimum PE Forward - above this is a sell signal (>)
        "MAX_EXRET": 10.0,           # Maximum EXRET - below this is a sell signal (<)
        "PE_CONDITION": "PEF > PET"  # PE Forward greater than PE Trailing is a sell signal (when both positive)
    },
    
    # Hold criteria - stocks that pass confidence threshold but don't meet buy or sell criteria
    "HOLD": {
        # No specific parameters, defined by not meeting buy or sell criteria
    },
    
    # Inconclusive - insufficient analyst coverage
    "INCONCLUSIVE": {
        # Defined by not meeting the common criteria
    }
}

# API Endpoints (reserved for future use)
API_ENDPOINTS: Dict[str, str] = {
    "BASE_URL": "https://query1.finance.yahoo.com/v8/finance",
    # Add more endpoints as needed
}

# Display Configuration
DISPLAY = {
    "DEFAULT_TABLE_FORMAT": "fancy_grid",
    "MAX_COMPANY_NAME_LENGTH": 20,
    "PRICE_DECIMALS": 2,
    "PERCENTAGE_DECIMALS": 1,
    "RATIO_DECIMALS": 2,
}

# File Paths Configuration
FILE_PATHS = {
    "INPUT_DIR": "yahoofinance/input",
    "OUTPUT_DIR": "yahoofinance/output",
    "CACHE_DIR": "yahoofinance/cache",
}
#!/usr/bin/env python3
"""
Trading Analysis Constants

This module contains all magic numbers and thresholds used throughout the trading analysis system.
Extracting these constants improves maintainability and makes the system more configurable.
"""

# Market Cap Tier Thresholds (in dollars)
MARKET_CAP_SMALL_THRESHOLD = 2_000_000_000        # $2B - Small cap threshold
MARKET_CAP_LARGE_THRESHOLD = 10_000_000_000       # $10B - Large cap threshold

# Market Cap Multipliers
TRILLION_MULTIPLIER = 1_000_000_000_000  # 1T
BILLION_MULTIPLIER = 1_000_000_000       # 1B  
MILLION_MULTIPLIER = 1_000_000           # 1M

# Financial Ratio Thresholds
PE_RATIO_LOW_THRESHOLD = 15.0            # Low PE ratio threshold
PE_RATIO_HIGH_THRESHOLD = 25.0           # High PE ratio threshold
PEG_RATIO_GOOD_THRESHOLD = 1.0           # Good PEG ratio threshold
PEG_RATIO_EXPENSIVE_THRESHOLD = 2.0      # Expensive PEG ratio threshold

# Risk Metrics
BETA_LOW_RISK_THRESHOLD = 1.0            # Low risk beta threshold
BETA_HIGH_RISK_THRESHOLD = 1.5           # High risk beta threshold
SHORT_PERCENT_HIGH_THRESHOLD = 10.0      # High short interest threshold

# Performance Thresholds
UPSIDE_STRONG_THRESHOLD = 20.0           # Strong upside potential threshold
UPSIDE_MODERATE_THRESHOLD = 10.0         # Moderate upside potential threshold
BUY_PERCENTAGE_HIGH_THRESHOLD = 70.0     # High analyst buy percentage threshold
BUY_PERCENTAGE_MODERATE_THRESHOLD = 50.0 # Moderate analyst buy percentage threshold

# EXRET (Expected Return) Calculation
EXRET_PERCENTAGE_DIVISOR = 100.0         # Divisor for percentage calculations

# Analyst Coverage Thresholds  
MIN_ANALYST_COUNT = 3                    # Minimum analyst coverage for reliability
STRONG_ANALYST_COUNT = 10                # Strong analyst coverage threshold

# Default Values for Missing Data
DEFAULT_NUMERIC_VALUE = 0.0              # Default for missing numeric values
DEFAULT_PE_RATIO = 20.0                  # Default PE ratio when missing
DEFAULT_PEG_RATIO = 1.5                  # Default PEG ratio when missing
DEFAULT_BETA = 1.0                       # Default beta when missing

# String Cleaning Constants
PERCENTAGE_SUFFIX = '%'                  # Percentage string suffix
EMPTY_STRING = ''                        # Empty string constant

# Action Classifications
ACTION_BUY = 'B'                         # Buy action code
ACTION_SELL = 'S'                        # Sell action code  
ACTION_HOLD = 'H'                        # Hold action code
ACTION_IDEA = 'I'                        # Idea action code

# Column Name Constants
TICKER_COLUMN = 'ticker'                 # Standard ticker column name
MARKET_CAP_COLUMN = 'market_cap'         # Market cap column name
ACTION_COLUMN = 'action'                 # Action column name
BS_COLUMN = 'BS'                         # Buy/Sell column name

# File Extensions
CSV_EXTENSION = '.csv'                   # CSV file extension
HTML_EXTENSION = '.html'                 # HTML file extension

# Cache and Performance Constants
DEFAULT_CACHE_TTL = 300                  # Default cache TTL in seconds (5 minutes)
BATCH_SIZE_DEFAULT = 50                  # Default batch size for API calls
MAX_CONCURRENT_REQUESTS = 10             # Maximum concurrent API requests

# Validation Thresholds
MAX_REASONABLE_PE = 1000.0               # Maximum reasonable PE ratio
MAX_REASONABLE_MARKET_CAP = 10e12        # Maximum reasonable market cap ($10T)
MIN_REASONABLE_PRICE = 0.01              # Minimum reasonable stock price

# Display and Formatting
DECIMAL_PLACES_CURRENCY = 2             # Decimal places for currency display
DECIMAL_PLACES_PERCENTAGE = 1           # Decimal places for percentage display
DECIMAL_PLACES_RATIO = 2                # Decimal places for ratio display

# Error Handling
MAX_RETRY_ATTEMPTS = 3                   # Maximum retry attempts for API calls
RETRY_DELAY_SECONDS = 1.0                # Delay between retry attempts

# Data Quality Thresholds
MIN_DATA_COMPLETENESS = 0.8              # Minimum data completeness ratio (80%)
MAX_OUTLIER_FACTOR = 3.0                 # Maximum outlier detection factor

# Performance Benchmarks
TARGET_PROCESSING_RATE = 1000            # Target stocks processed per second
PERFORMANCE_WARNING_THRESHOLD = 500     # Performance warning threshold (stocks/sec)

# Trading Signal Weights (for future use in weighted scoring)
UPSIDE_WEIGHT = 0.4                      # Weight for upside potential in scoring
BUY_PERCENTAGE_WEIGHT = 0.3              # Weight for analyst buy percentage
PE_RATIO_WEIGHT = 0.2                    # Weight for PE ratio in scoring
PEG_RATIO_WEIGHT = 0.1                   # Weight for PEG ratio in scoring
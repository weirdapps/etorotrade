"""
Rate limiting configuration.
"""

from typing import Any, Dict, Set


class RateLimitConfig:
    """Rate limiting configuration with immutable settings."""
    
    def __init__(self):
        """Initialize rate limiting configuration."""
        # Time window configuration
        self.window_size = 60  # 1 minute sliding window
        self.max_calls = 60    # Max 60 calls per minute window
        
        # Delay configuration - SIMPLIFIED FIXED RATE STRATEGY
        self.base_delay = 0.3     # 300ms between API calls
        self.min_delay = 0.3      # Fixed minimum delay
        self.max_delay = 0.3      # Fixed maximum delay (except for rate limit errors)
        
        # Batch configuration
        self.batch_size = 10       # Items per batch
        self.batch_delay = 0.2     # Delay between batches in seconds
        
        # Retry and timeout configuration
        self.max_retry_attempts = 3
        self.api_timeout = 60
        self.max_concurrent_calls = 15
        
        # Jitter and adaptation - DISABLED for fixed rate strategy
        self.jitter_factor = 0.0
        self.error_threshold = 999999  # Effectively disabled
        self.error_delay_increase = 1.0
        self.rate_limit_delay_increase = 1.0
        
        # Ticker priorities - ALL EQUAL for fixed rate strategy
        self.ticker_priority = {
            "HIGH": 1.0,
            "MEDIUM": 1.0,
            "LOW": 1.0,
        }
        
        # Special ticker sets - EMPTY for fixed rate strategy
        self.slow_tickers: Set[str] = set()
        self.vip_tickers: Set[str] = set()
        
        # Feature flags - DISABLED for simplified strategy
        self.cache_aware_rate_limiting = False
        self.enable_adaptive_strategy = False
        self.monitor_interval = 0
        
        # Market hours and regional adjustments - DISABLED
        self.market_hours_delay_multiplier = 1.0
        self.off_market_delay_multiplier = 1.0
        self.us_delay_multiplier = 1.0
        self.europe_delay_multiplier = 1.0
        self.asia_delay_multiplier = 1.0
        
        # Performance thresholds
        self.max_error_rate = 1.0
        self.min_success_rate = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "WINDOW_SIZE": self.window_size,
            "MAX_CALLS": self.max_calls,
            "BASE_DELAY": self.base_delay,
            "MIN_DELAY": self.min_delay,
            "MAX_DELAY": self.max_delay,
            "BATCH_SIZE": self.batch_size,
            "BATCH_DELAY": self.batch_delay,
            "MAX_RETRY_ATTEMPTS": self.max_retry_attempts,
            "API_TIMEOUT": self.api_timeout,
            "MAX_CONCURRENT_CALLS": self.max_concurrent_calls,
            "JITTER_FACTOR": self.jitter_factor,
            "ERROR_THRESHOLD": self.error_threshold,
            "ERROR_DELAY_INCREASE": self.error_delay_increase,
            "RATE_LIMIT_DELAY_INCREASE": self.rate_limit_delay_increase,
            "TICKER_PRIORITY": self.ticker_priority.copy(),
            "SLOW_TICKERS": self.slow_tickers.copy(),
            "VIP_TICKERS": self.vip_tickers.copy(),
            "CACHE_AWARE_RATE_LIMITING": self.cache_aware_rate_limiting,
            "MARKET_HOURS_DELAY_MULTIPLIER": self.market_hours_delay_multiplier,
            "OFF_MARKET_DELAY_MULTIPLIER": self.off_market_delay_multiplier,
            "US_DELAY_MULTIPLIER": self.us_delay_multiplier,
            "EUROPE_DELAY_MULTIPLIER": self.europe_delay_multiplier,
            "ASIA_DELAY_MULTIPLIER": self.asia_delay_multiplier,
            "ENABLE_ADAPTIVE_STRATEGY": self.enable_adaptive_strategy,
            "MONITOR_INTERVAL": self.monitor_interval,
            "MAX_ERROR_RATE": self.max_error_rate,
            "MIN_SUCCESS_RATE": self.min_success_rate,
        }
    
    def update_for_testing(self, **kwargs) -> None:
        """Update configuration for testing purposes.
        
        WARNING: Only use this in tests!
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown rate limit config key: {key}")
    
    def create_test_copy(self, **overrides) -> 'RateLimitConfig':
        """Create a copy with test overrides."""
        copy = RateLimitConfig()
        copy.update_for_testing(**overrides)
        return copy
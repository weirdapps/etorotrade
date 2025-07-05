"""
Production environment configuration.
"""

from .base import BaseConfig


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    
    def get_log_level(self) -> str:
        """Get the log level for production environment."""
        return "INFO"
    
    def _configure_environment(self) -> None:
        """Configure production-specific settings."""
        # Optimized rate limiting for production
        self.rate_limit.base_delay = 0.3     # Standard production delay
        self.rate_limit.batch_size = 10      # Standard batch size
        self.rate_limit.max_concurrent_calls = 15  # Higher concurrency for production
        
        # Disable debugging features for performance
        self.rate_limit.enable_adaptive_strategy = False
        self.rate_limit.monitor_interval = 0
        
        # Production trading criteria (from original config)
        self.trading_criteria.buy_min_upside = 20.0
        self.trading_criteria.buy_min_buy_percentage = 85.0
        
        # Disable yahooquery in production to prevent crumb errors
        self.providers.enable_yahooquery = False
        
        # Production portfolio settings (full amounts)
        self.portfolio.update({
            "PORTFOLIO_VALUE": 450_000,
            "MIN_POSITION_USD": 1_000,
            "MAX_POSITION_USD": 40_000,
        })
        
        # More aggressive timeouts for production
        self.rate_limit.api_timeout = 45  # Shorter timeout in production
        
        # Stricter error handling in production
        self.rate_limit.max_retry_attempts = 2  # Fewer retries in production
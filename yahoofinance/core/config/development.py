"""
Development environment configuration.
"""

from .base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    def get_log_level(self) -> str:
        """Get the log level for development environment."""
        return "DEBUG"
    
    def _configure_environment(self) -> None:
        """Configure development-specific settings."""
        # More aggressive rate limiting for development to be safer
        self.rate_limit.base_delay = 0.5  # Slower in development
        self.rate_limit.batch_size = 5    # Smaller batches in development
        self.rate_limit.max_concurrent_calls = 5  # Fewer concurrent calls
        
        # Enable debugging features
        self.rate_limit.enable_adaptive_strategy = True
        self.rate_limit.monitor_interval = 30
        
        # More conservative trading criteria for development/testing
        self.trading_criteria.buy_min_upside = 25.0  # Higher threshold in dev
        self.trading_criteria.buy_min_buy_percentage = 90.0  # More conservative
        
        # Enable yahooquery in development for testing
        self.providers.enable_yahooquery = True
        
        # Development-specific portfolio settings (smaller amounts)
        self.portfolio.update({
            "PORTFOLIO_VALUE": 100_000,  # Smaller dev portfolio
            "MIN_POSITION_USD": 500,     # Smaller minimum
            "MAX_POSITION_USD": 10_000,  # Smaller maximum
        })
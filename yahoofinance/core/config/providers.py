"""
Provider configuration.
"""

from typing import Any, Dict


class ProviderConfig:
    """Provider configuration with immutable settings."""
    
    def __init__(self):
        """Initialize provider configuration."""
        # YahooQuery integration toggle
        self.enable_yahooquery = False  # Set to False to disable yahooquery and prevent crumb errors
        
        # Default provider settings
        self.default_provider = "hybrid"
        self.default_async_provider = "async_hybrid"
        
        # Provider-specific settings
        self.yahoo_finance = {
            "MAX_RETRIES": 3,
            "TIMEOUT": 30,
            "ENABLE_CACHING": True,
        }
        
        self.yahooquery = {
            "MAX_RETRIES": 3,
            "TIMEOUT": 30,
            "ENABLE_CACHING": True,
            "CRUMB_RETRY_ATTEMPTS": 2,
        }
        
        self.hybrid = {
            "PRIORITIZE_YFINANCE": True,
            "SUPPLEMENT_WITH_YAHOOQUERY": True,
            "FALLBACK_ON_ERROR": True,
        }
        
        # Async provider settings
        self.async_settings = {
            "MAX_CONCURRENT_REQUESTS": 10,
            "BATCH_SIZE": 20,
            "CONNECTION_POOL_SIZE": 10,
            "CONNECTION_TIMEOUT": 30,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "ENABLE_YAHOOQUERY": self.enable_yahooquery,
            "DEFAULT_PROVIDER": self.default_provider,
            "DEFAULT_ASYNC_PROVIDER": self.default_async_provider,
            "YAHOO_FINANCE": self.yahoo_finance.copy(),
            "YAHOOQUERY": self.yahooquery.copy(),
            "HYBRID": self.hybrid.copy(),
            "ASYNC_SETTINGS": self.async_settings.copy(),
        }
    
    def update_for_testing(self, **kwargs) -> None:
        """Update configuration for testing purposes.
        
        WARNING: Only use this in tests!
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown provider config key: {key}")
    
    def create_test_copy(self, **overrides) -> 'ProviderConfig':
        """Create a copy with test overrides."""
        copy = ProviderConfig()
        copy.update_for_testing(**overrides)
        return copy
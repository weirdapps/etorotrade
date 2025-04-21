import pytest

# Import tests to ensure imports are working correctly
def test_api_imports():
    """Test that all necessary API imports work correctly."""
    # Core imports
    from yahoofinance.core.errors import YFinanceError, ValidationError, APIError
    from yahoofinance.core.types import StockData
    
    # Provider imports
    from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
    from yahoofinance.api.providers.base_provider import FinanceDataProvider
    
    # Utility imports
    from yahoofinance.utils.network.rate_limiter import RateLimiter
    from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, circuit_protected
    from yahoofinance.utils.data.format_utils import format_number, format_market_cap
    
    # Main module imports
    from yahoofinance import get_provider
    
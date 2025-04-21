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
    
    assert YFinanceError is not None
    assert ValidationError is not None
    assert APIError is not None
    assert StockData is not None
    assert YahooFinanceProvider is not None
    assert FinanceDataProvider is not None
    assert RateLimiter is not None
    assert CircuitBreaker is not None
    assert circuit_protected is not None
    assert format_number is not None
    assert format_market_cap is not None
    assert get_provider is not None
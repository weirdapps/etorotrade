# Import tests to ensure imports are working correctly
def test_api_imports():
    """Test that all necessary API imports work correctly."""
    # Core imports
    # Main module imports
    from yahoofinance.core.errors import ValidationError, YFinanceError
    from yahoofinance.core.types import StockData
    from yahoofinance.utils.data.format_utils import format_market_cap, format_number
    from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, circuit_protected

    # Utility imports
    from yahoofinance.utils.network.rate_limiter import RateLimiter

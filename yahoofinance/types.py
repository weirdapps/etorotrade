"""Common types and data structures used across the package"""

from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf

# Import errors from the centralized errors module
from .errors import (
    YFinanceError,
    APIError,
    ValidationError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    ResourceNotFoundError,
    DataError
)

@dataclass
class StockData:
    """
    Comprehensive stock information data class.
    
    Contains fundamental data, technical indicators, analyst ratings,
    and market metrics for a given stock. All numeric fields are
    optional as they may not be available for all stocks.
    
    Fields are grouped by category:
    - Basic Info: name, sector
    - Market Data: market_cap, current_price, target_price
    - Analyst Coverage: recommendation_mean, recommendation_key, analyst_count
    - Valuation Metrics: pe_trailing, pe_forward, peg_ratio
    - Financial Health: quick_ratio, current_ratio, debt_to_equity
    - Risk Metrics: short_float_pct, short_ratio, beta
    - Dividends: dividend_yield
    - Events: last_earnings, previous_earnings
    - Insider Activity: insider_buy_pct, insider_transactions
    """
    # Basic Info (Required)
    name: str
    sector: str
    recommendation_key: str
    
    # Market Data (Optional)
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    price_change_percentage: Optional[float] = None
    mtd_change: Optional[float] = None
    ytd_change: Optional[float] = None
    two_year_change: Optional[float] = None
    
    # Analyst Coverage (Optional)
    recommendation_mean: Optional[float] = None
    analyst_count: Optional[int] = None
    
    # Valuation Metrics (Optional)
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    
    # Financial Health (Optional)
    quick_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    # Risk Metrics (Optional)
    short_float_pct: Optional[float] = None
    short_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    cash_percentage: Optional[float] = None
    
    # Dividends (Optional)
    dividend_yield: Optional[float] = None
    
    # Events (Optional)
    last_earnings: Optional[str] = None
    previous_earnings: Optional[str] = None
    
    # Insider Activity (Optional)
    insider_buy_pct: Optional[float] = None
    insider_transactions: Optional[int] = None
    
    # Internal (Optional)
    ticker_object: Optional[yf.Ticker] = field(default=None, repr=False)  # Don't include in string representation

    def __post_init__(self):
        """Validate types after initialization"""
        self._validate_required_fields()
        self._validate_numeric_fields()
        self._validate_string_fields()
        self._validate_ticker_object()
        
    def _validate_required_fields(self):
        """Validate required string fields"""
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not isinstance(self.sector, str):
            raise TypeError("sector must be a string")
        if not isinstance(self.recommendation_key, str):
            raise TypeError("recommendation_key must be a string")
    
    def _validate_numeric_fields(self):
        """Validate and convert numeric fields"""
        numeric_fields = {
            'market_cap': float, 'current_price': float, 'target_price': float,
            'price_change_percentage': float, 'mtd_change': float, 'ytd_change': float,
            'two_year_change': float, 'recommendation_mean': float, 'analyst_count': int,
            'pe_trailing': float, 'pe_forward': float, 'peg_ratio': float,
            'quick_ratio': float, 'current_ratio': float, 'debt_to_equity': float,
            'short_float_pct': float, 'short_ratio': float, 'beta': float,
            'alpha': float, 'sharpe_ratio': float, 'sortino_ratio': float,
            'cash_percentage': float, 'dividend_yield': float,
            'insider_buy_pct': float, 'insider_transactions': int
        }
        
        for field_name, expected_type in numeric_fields.items():
            value = getattr(self, field_name)
            if value is not None:
                try:
                    # Convert to expected type if needed
                    if not isinstance(value, expected_type):
                        setattr(self, field_name, expected_type(value))
                except (ValueError, TypeError):
                    raise TypeError(f"{field_name} must be convertible to {expected_type.__name__}")
    
    def _validate_string_fields(self):
        """Validate optional string fields"""
        string_fields = ['last_earnings', 'previous_earnings']
        for field_name in string_fields:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
    
    def _validate_ticker_object(self):
        """Validate ticker object if present"""
        if self.ticker_object is not None:
            # Skip validation in test environment (when Mock objects are used)
            if not hasattr(self.ticker_object, '_mock_return_value'):  # Not a mock object
                if not isinstance(self.ticker_object, yf.Ticker):
                    raise TypeError("ticker_object must be a yfinance.Ticker instance")

    @property
    def _stock(self) -> yf.Ticker:
        """
        Access the underlying yfinance Ticker object.
        
        This property provides access to the raw yfinance Ticker object,
        which can be used for additional API calls not covered by the
        standard properties.
        
        Returns:
            yfinance.Ticker object for additional API access
            
        Raises:
            AttributeError: If ticker_object is None
        """
        if self.ticker_object is None:
            raise AttributeError("No ticker object available")
        return self.ticker_object
"""
Compatibility module for pricing analyzer from v1.

This module provides the PricingAnalyzer class that mirrors the interface of
the v1 pricing analyzer but uses the v2 implementation under the hood.

DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
Use the canonical import path instead:
from yahoofinance.analysis.market import MarketAnalyzer
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union

# Show deprecation warning
warnings.warn(
    "The yahoofinance.compat.pricing module is deprecated and will be removed in a future version. "
    "Use 'from yahoofinance.analysis.market import MarketAnalyzer' instead.",
    DeprecationWarning,
    stacklevel=2
)

from ..api import get_provider

logger = logging.getLogger(__name__)

class PricingAnalyzer:
    """
    Compatibility class for v1 PricingAnalyzer.
    
    Uses v2 provider pattern under the hood.
    """
    
    def __init__(self):
        """Initialize v1-compatible pricing analyzer."""
        self.provider = get_provider()
    
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with price data
        """
        try:
            data = self.provider.get_ticker_info(ticker)
            return {
                "ticker": ticker,
                "price": data.get("price"),
                "target_price": data.get("target_price"),
                "upside": data.get("upside")
            }
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "price": None,
                "target_price": None,
                "upside": None
            }
    
    def calculate_upside(self, current_price: float, target_price: float) -> float:
        """
        Calculate upside potential as a percentage.
        
        Args:
            current_price: Current stock price
            target_price: Target stock price
            
        Returns:
            Upside potential as a percentage
        """
        if not current_price or not target_price or current_price <= 0:
            return 0.0
        
        return ((target_price - current_price) / current_price) * 100
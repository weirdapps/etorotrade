"""
Compatibility module for analyst data classes from v1.

This module provides the AnalystData class that mirrors the interface of
the v1 analyst data classes but uses the v2 implementation under the hood.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

from ..analysis.analyst import AnalystData as V2AnalystData

logger = logging.getLogger(__name__)

@dataclass
class AnalystData:
    """
    Compatibility class for v1 AnalystData.
    
    Mirrors the interface of the v1 analyst data class.
    """
    ticker: str
    count: Optional[int] = None
    buy_percentage: Optional[float] = None
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    upside: Optional[float] = None
    
    @classmethod
    def from_v2_data(cls, data: V2AnalystData) -> 'AnalystData':
        """
        Create a v1-compatible AnalystData from a v2 analyst data object.
        
        Args:
            data: V2 analyst data object
            
        Returns:
            V1-compatible AnalystData
        """
        return cls(
            ticker=data.ticker,
            count=data.count,
            buy_percentage=data.buy_percentage,
            target_price=data.target_price,
            current_price=data.current_price,
            upside=data.upside
        )
    
    @classmethod
    def from_provider_data(cls, ticker: str, data: Dict[str, Any]) -> 'AnalystData':
        """
        Create AnalystData from provider data dictionary.
        
        Args:
            ticker: Stock ticker symbol
            data: Provider data dictionary
            
        Returns:
            AnalystData instance
        """
        return cls(
            ticker=ticker,
            count=data.get("recommendations"),
            buy_percentage=data.get("buy_percentage"),
            target_price=data.get("target_price"),
            current_price=data.get("price"),
            upside=data.get("upside")
        )
"""
Compatibility module for display formatting classes from v1.

This module provides the DisplayFormatter and DisplayConfig classes that mirror
the interface of the v1 formatting classes but use the v2 implementation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

from ..presentation.formatter import DisplayFormatter as V2Formatter
from ..presentation.formatter import DisplayConfig as V2Config

logger = logging.getLogger(__name__)

@dataclass
class DisplayConfig:
    """Compatibility class for v1 DisplayConfig"""
    use_colors: bool = True
    date_format: str = "%Y-%m-%d"
    float_precision: int = 2
    percentage_precision: int = 1
    table_format: str = "grid"
    
    @classmethod
    def from_v2_config(cls, config: V2Config) -> 'DisplayConfig':
        """
        Create a v1-compatible config from a v2 config.
        
        Args:
            config: V2 configuration object
            
        Returns:
            V1-compatible DisplayConfig
        """
        return cls(
            use_colors=config.use_colors,
            date_format=config.date_format,
            float_precision=config.float_precision,
            percentage_precision=config.percentage_precision,
            table_format=config.table_format
        )

class DisplayFormatter:
    """
    Compatibility class for v1 DisplayFormatter.
    
    Uses v2 formatter under the hood.
    """
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        """
        Initialize v1-compatible formatter.
        
        Args:
            config: V1-compatible display configuration
        """
        # Create v2 config from v1 config
        v2_config = None
        if config:
            v2_config = V2Config(
                use_colors=config.use_colors,
                date_format=config.date_format,
                float_precision=config.float_precision,
                percentage_precision=config.percentage_precision,
                table_format=config.table_format
            )
            
        # Create v2 formatter
        self.v2_formatter = V2Formatter(v2_config)
    
    def format_stock_row(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format stock data for display.
        
        Args:
            stock_data: Dictionary containing stock metrics
            
        Returns:
            Dictionary with formatted values
        """
        return self.v2_formatter.format_stock_row(stock_data)
    
    def format_price(self, price: Union[float, int, None]) -> str:
        """
        Format price with appropriate precision.
        
        Args:
            price: Price to format
            
        Returns:
            Formatted price string
        """
        return self.v2_formatter.format_price(price)
    
    def format_percentage(self, percentage: Union[float, int, None]) -> str:
        """
        Format percentage with appropriate precision.
        
        Args:
            percentage: Percentage to format
            
        Returns:
            Formatted percentage string
        """
        return self.v2_formatter.format_percentage(percentage)
    
    def format_market_cap(self, market_cap: Union[float, int, None]) -> str:
        """
        Format market cap with appropriate abbreviation.
        
        Args:
            market_cap: Market cap to format
            
        Returns:
            Formatted market cap string
        """
        return self.v2_formatter.format_market_cap(market_cap)
    
    def format_date(self, date_str: Optional[str]) -> str:
        """
        Format date with appropriate format.
        
        Args:
            date_str: Date string to format
            
        Returns:
            Formatted date string
        """
        return self.v2_formatter.format_date(date_str)
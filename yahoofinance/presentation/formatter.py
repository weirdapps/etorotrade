"""
Display formatting utilities for presenting financial data.

This module provides utilities for formatting financial data for display
in both console and HTML output. It handles market cap formatting, 
percentage formatting, and coloring of financial metrics.
"""

from enum import Enum
from typing import Optional, Any, Union, Dict, List, Tuple
from dataclasses import dataclass

class Color(Enum):
    """Color constants for terminal output."""
    GREEN = "\033[92m"      # Green color
    RED = "\033[91m"        # Red color
    YELLOW = "\033[93m"     # Yellow color
    BLUE = "\033[94m"       # Blue color
    PURPLE = "\033[95m"     # Purple color
    CYAN = "\033[96m"       # Cyan color
    WHITE = "\033[97m"      # White color
    RESET = "\033[0m"       # Reset color
    BOLD = "\033[1m"        # Bold text
    UNDERLINE = "\033[4m"   # Underlined text
    
@dataclass
class DisplayConfig:
    """Configuration for display formatting."""
    compact_mode: bool = False
    show_colors: bool = True
    max_name_length: int = 14
    date_format: str = "%Y-%m-%d"
    show_headers: bool = True
    max_columns: Optional[int] = None
    sort_column: Optional[str] = None
    reverse_sort: bool = False

class DisplayFormatter:
    """
    Format financial data for display with appropriate styling.
    
    This class provides methods for formatting market caps, prices,
    percentages, and other financial metrics with consistent styling.
    It also applies coloration based on financial signals.
    """
    
    def __init__(self, compact_mode: bool = False):
        """
        Initialize a display formatter.
        
        Args:
            compact_mode: Use more compact formatting (fewer decimals)
        """
        self.compact_mode = compact_mode
    
    def format_market_cap(self, value: Optional[float]) -> Optional[str]:
        """
        Format market cap value with appropriate suffix (T, B, M).
        
        Args:
            value: Market cap value
            
        Returns:
            Formatted market cap string or None if value is None
        """
        if value is None:
            return None
        
        # Trillions
        if value >= 1e12:
            if value >= 10e12:
                # Above 10T, use 1 decimal place
                return f"{value / 1e12:.1f}T"
            else:
                # Below 10T, use 2 decimal places
                return f"{value / 1e12:.2f}T"
        
        # Billions
        elif value >= 1e9:
            if value >= 100e9:
                # Above 100B, use no decimals
                return f"{int(value / 1e9)}B"
            elif value >= 10e9:
                # Above 10B but below 100B, use 1 decimal place
                return f"{value / 1e9:.1f}B"
            else:
                # Below 10B, use 2 decimal places
                return f"{value / 1e9:.2f}B"
        
        # Millions
        elif value >= 1e6:
            if value >= 100e6:
                # Above 100M, use no decimals
                return f"{int(value / 1e6)}M"
            elif value >= 10e6:
                # Above 10M but below 100M, use 1 decimal place
                return f"{value / 1e6:.1f}M"
            else:
                # Below 10M, use 2 decimal places
                return f"{value / 1e6:.2f}M"
        
        # Less than a million, format with commas
        else:
            return f"{int(value):,}"
    
    def format_price(self, value: Optional[float], decimals: int = 2) -> str:
        """
        Format a price value.
        
        Args:
            value: Price value
            decimals: Number of decimal places
            
        Returns:
            Formatted price string
        """
        if value is None:
            return "--"
        
        # Use fewer decimals in compact mode
        if self.compact_mode and value >= 100:
            decimals = max(0, decimals - 1)
        
        # Format price with appropriate decimals
        return f"${value:,.{decimals}f}"
    
    def format_percentage(self, value: Optional[float], decimals: int = 1) -> str:
        """
        Format a percentage value.
        
        Args:
            value: Percentage value (should be pre-multiplied by 100)
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if value is None:
            return "--"
        
        # Use fewer decimals in compact mode
        if self.compact_mode:
            decimals = max(0, decimals - 1)
        
        # Format percentage with appropriate decimals
        return f"{value:,.{decimals}f}%"
    
    def format_ratio(self, value: Optional[float], decimals: int = 2) -> str:
        """
        Format a ratio value.
        
        Args:
            value: Ratio value
            decimals: Number of decimal places
            
        Returns:
            Formatted ratio string
        """
        if value is None:
            return "--"
        
        # Use fewer decimals in compact mode
        if self.compact_mode:
            decimals = max(1, decimals - 1)
        
        # Format ratio with appropriate decimals
        return f"{value:,.{decimals}f}"
    
    def color_value(self, value: str, condition: str, cutoff: float, reverse: bool = False) -> str:
        """
        Apply color to a value based on a condition.
        
        Args:
            value: Value to color
            condition: Condition type (e.g., 'gt', 'lt', 'eq')
            cutoff: Cutoff value for the condition
            reverse: Whether to reverse the coloring logic
            
        Returns:
            Colored value string
        """
        # Don't try to evaluate placeholders
        if value == "--":
            return value
        
        # Try to convert string to float for comparison
        try:
            numeric_value = float(value.replace('$', '').replace('%', '').replace(',', ''))
        except (ValueError, AttributeError):
            # If conversion fails, return the original value without coloring
            return value
        
        # Determine whether condition is met
        condition_met = False
        if condition == 'gt':
            condition_met = numeric_value > cutoff
        elif condition == 'lt':
            condition_met = numeric_value < cutoff
        elif condition == 'eq':
            condition_met = numeric_value == cutoff
        elif condition == 'gte':
            condition_met = numeric_value >= cutoff
        elif condition == 'lte':
            condition_met = numeric_value <= cutoff
        
        # Reverse the condition if requested
        if reverse:
            condition_met = not condition_met
        
        # Apply appropriate color
        if condition_met:
            return f"{Color.GREEN.value}{value}{Color.RESET.value}"
        else:
            return f"{Color.RED.value}{value}{Color.RESET.value}"
    
    def color_buy_percentage(self, value: str) -> str:
        """
        Color buy percentage based on analyst rating thresholds.
        
        Args:
            value: Buy percentage string
            
        Returns:
            Colored buy percentage string
        """
        return self.color_value(value, 'gt', 80)
    
    def color_upside(self, value: str) -> str:
        """
        Color upside potential based on thresholds.
        
        Args:
            value: Upside potential string
            
        Returns:
            Colored upside string
        """
        return self.color_value(value, 'gt', 15)
    
    def color_peg(self, value: str) -> str:
        """
        Color PEG ratio based on thresholds (lower is better).
        
        Args:
            value: PEG ratio string
            
        Returns:
            Colored PEG ratio string
        """
        return self.color_value(value, 'lt', 1.5, reverse=False)
    
    def color_beta(self, value: str) -> str:
        """
        Color beta value based on thresholds.
        
        Args:
            value: Beta value string
            
        Returns:
            Colored beta string
        """
        # Try to convert to float first
        try:
            beta = float(value)
            
            # Apply different colors based on beta range
            if beta < 0.8:
                return f"{Color.BLUE.value}{value}{Color.RESET.value}"  # Low volatility
            elif beta > 2.0:
                return f"{Color.RED.value}{value}{Color.RESET.value}"   # High volatility
            else:
                return f"{Color.GREEN.value}{value}{Color.RESET.value}" # Moderate volatility
        except ValueError:
            return value
    
    def color_pe_ratio(self, value: str, is_forward: bool = False) -> str:
        """
        Color PE ratio based on thresholds.
        
        Args:
            value: PE ratio string
            is_forward: Whether this is a forward PE ratio
            
        Returns:
            Colored PE ratio string
        """
        # Try to convert to float first
        try:
            pe = float(value)
            
            # Different thresholds for forward vs trailing PE
            if is_forward:
                if pe < 15:
                    return f"{Color.GREEN.value}{value}{Color.RESET.value}"  # Attractive
                elif pe > 30:
                    return f"{Color.RED.value}{value}{Color.RESET.value}"    # Expensive
                else:
                    return f"{Color.YELLOW.value}{value}{Color.RESET.value}" # Moderate
            else:
                if pe < 18:
                    return f"{Color.GREEN.value}{value}{Color.RESET.value}"  # Attractive
                elif pe > 35:
                    return f"{Color.RED.value}{value}{Color.RESET.value}"    # Expensive
                else:
                    return f"{Color.YELLOW.value}{value}{Color.RESET.value}" # Moderate
        except ValueError:
            # For special values like "--" or non-numeric strings
            if value in ["--", "N/A"]:
                return value
            # Try to handle negative PE specifically
            if "-" in value:
                return f"{Color.PURPLE.value}{value}{Color.RESET.value}"
            return value
    
    def color_by_signal(self, ticker_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine the overall buy/sell/hold signal color based on analysis.
        
        Args:
            ticker_data: Dictionary of ticker data
            
        Returns:
            Dictionary mapping column names to colored values
        """
        colored_data = {}
        
        # Extract and convert relevant metrics
        upside = ticker_data.get('upside')
        buy_percentage = ticker_data.get('buy_percentage')
        pe_forward = ticker_data.get('pe_forward')
        pe_trailing = ticker_data.get('pe_trailing')
        peg = ticker_data.get('peg_ratio')
        beta = ticker_data.get('beta')
        short_interest = ticker_data.get('short_float_pct')
        
        # Confidence check - require both analyst metrics to be present
        if (ticker_data.get('analyst_count') is None or 
            ticker_data.get('total_ratings') is None or
            upside is None or 
            buy_percentage is None):
            signal = "NEUTRAL"
        else:
            # Calculate expected return
            expected_return = (upside * buy_percentage / 100) if upside is not None and buy_percentage is not None else None
            
            # Check sell signals first (any trigger a sell)
            if any([
                upside < 5,  # Low upside (upside is already confirmed not None above)
                buy_percentage < 65,  # Low buy rating (buy_percentage is already confirmed not None above)
                # PE deteriorating (both positive, but forward > trailing)
                pe_forward is not None and pe_trailing is not None and pe_forward > 0 and pe_trailing > 0 and pe_forward > pe_trailing,
                pe_forward is not None and pe_forward > 45,  # Extremely high forward PE
                peg is not None and peg > 3.0,  # High PEG ratio
                short_interest is not None and short_interest > 4.0,  # High short interest
                beta is not None and beta > 3.0,  # Excessive volatility
                expected_return is not None and expected_return < 10.0  # Low expected return
            ]):
                signal = "SELL"
            # Then check buy signals (all criteria must be met)
            elif all([
                upside >= 20,  # Strong upside (upside is already confirmed not None above)
                buy_percentage >= 82,  # Strong buy consensus (buy_percentage is already confirmed not None above)
                beta is None or (beta > 0.2 and beta <= 3.0),  # Reasonable volatility
                # PE improvement or negative trailing PE (growth stock)
                (pe_forward is None or pe_trailing is None or
                 pe_forward <= 0 or  # Negative future earnings still allowed
                 (pe_forward > 0.5 and pe_forward <= 45.0 and  # Positive and reasonable forward PE
                  (pe_trailing <= 0 or pe_forward < pe_trailing))),  # Either negative trailing or improving ratio
                peg is None or peg < 3.0,  # Reasonable PEG ratio
                short_interest is None or short_interest <= 3.0  # Low short interest
            ]):
                signal = "BUY"
            else:
                signal = "HOLD"
        
        # Apply color based on signal
        for key, value in ticker_data.items():
            if value is None:
                colored_data[key] = "--"
                continue
                
            # Convert to string representation
            str_value = str(value)
            
            # Apply signal-based coloring
            if signal == "BUY":
                colored_data[key] = f"{Color.GREEN.value}{str_value}{Color.RESET.value}"
            elif signal == "SELL":
                colored_data[key] = f"{Color.RED.value}{str_value}{Color.RESET.value}"
            elif signal == "HOLD":
                colored_data[key] = f"{Color.YELLOW.value}{str_value}{Color.RESET.value}"
            else:
                colored_data[key] = str_value
        
        # Return colored data
        return colored_data
    
    def get_signal(self, ticker_data: Dict[str, Any]) -> str:
        """
        Determine the overall buy/sell/hold signal based on analysis.
        
        Args:
            ticker_data: Dictionary of ticker data
            
        Returns:
            Signal string ("BUY", "SELL", "HOLD", or "NEUTRAL")
        """
        # Extract and convert relevant metrics
        upside = ticker_data.get('upside')
        buy_percentage = ticker_data.get('buy_percentage')
        pe_forward = ticker_data.get('pe_forward')
        pe_trailing = ticker_data.get('pe_trailing')
        peg = ticker_data.get('peg_ratio')
        beta = ticker_data.get('beta')
        short_interest = ticker_data.get('short_float_pct')
        
        # Confidence check - require both analyst metrics to be present
        if (ticker_data.get('analyst_count') is None or 
            ticker_data.get('total_ratings') is None or
            upside is None or 
            buy_percentage is None):
            return "NEUTRAL"
        
        # Calculate expected return
        expected_return = (upside * buy_percentage / 100) # upside and buy_percentage are confirmed not None in the condition above
        
        # Check sell signals first (any trigger a sell)
        if any([
            upside < 5,  # Low upside
            buy_percentage < 65,  # Low buy rating
            # PE deteriorating (both positive, but forward > trailing)
            pe_forward is not None and pe_trailing is not None and pe_forward > 0 and pe_trailing > 0 and pe_forward > pe_trailing,
            pe_forward is not None and pe_forward > 45,  # Extremely high forward PE
            peg is not None and peg > 3.0,  # High PEG ratio
            short_interest is not None and short_interest > 4.0,  # High short interest
            beta is not None and beta > 3.0,  # Excessive volatility
            expected_return < 10.0  # Low expected return (expected_return is calculated above and is not None)
        ]):
            return "SELL"
        # Then check buy signals (all criteria must be met)
        elif all([
            upside >= 20,  # Strong upside
            buy_percentage >= 82,  # Strong buy consensus
            beta is None or (beta > 0.2 and beta <= 3.0),  # Reasonable volatility
            # PE improvement or negative trailing PE (growth stock)
            (pe_forward is None or pe_trailing is None or
             pe_forward <= 0 or  # Negative future earnings still allowed
             (pe_forward > 0.5 and pe_forward <= 45.0 and  # Positive and reasonable forward PE
              (pe_trailing <= 0 or pe_forward < pe_trailing))),  # Either negative trailing or improving ratio
            peg is None or peg < 3.0,  # Reasonable PEG ratio
            short_interest is None or short_interest <= 3.0  # Low short interest
        ]):
            return "BUY"
        else:
            return "HOLD"
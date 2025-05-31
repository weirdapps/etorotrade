"""
Display formatting utilities for presenting financial data.

This module provides utilities for formatting financial data for display
in both console and HTML output. It handles market cap formatting,
percentage formatting, and coloring of financial metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..utils.data.format_utils import format_market_cap as utils_format_market_cap


class Color(Enum):
    """Color constants for terminal output."""

    GREEN = "\033[92m"  # Green color
    RED = "\033[91m"  # Red color
    YELLOW = "\033[93m"  # Yellow color
    BLUE = "\033[94m"  # Blue color
    PURPLE = "\033[95m"  # Purple color
    CYAN = "\033[96m"  # Cyan color
    WHITE = "\033[97m"  # White color
    RESET = "\033[0m"  # Reset color
    BOLD = "\033[1m"  # Bold text
    UNDERLINE = "\033[4m"  # Underlined text


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
    # Standard column order that should be used for all displays
    reorder_columns: List[str] = None

    def __post_init__(self):
        # Import here to avoid circular import
        from ..core.config import STANDARD_DISPLAY_COLUMNS

        self.reorder_columns = STANDARD_DISPLAY_COLUMNS


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
        self.config = DisplayConfig()

    def format_market_cap(self, value: Optional[float]) -> Optional[str]:
        """
        Format market cap value with appropriate suffix (T, B, M).

        Uses the canonical market cap formatter from utils.data.format_utils.

        Args:
            value: Market cap value

        Returns:
            Formatted market cap string or None if value is None
        """
        # Use the canonical format_market_cap function from format_utils
        return utils_format_market_cap(value)

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
            numeric_value = float(value.replace("$", "").replace("%", "").replace(",", ""))
        except (ValueError, AttributeError):
            # If conversion fails, return the original value without coloring
            return value

        # Determine whether condition is met
        condition_met = False
        if condition == "gt":
            condition_met = numeric_value > cutoff
        elif condition == "lt":
            condition_met = numeric_value < cutoff
        elif condition == "eq":
            condition_met = numeric_value == cutoff
        elif condition == "gte":
            condition_met = numeric_value >= cutoff
        elif condition == "lte":
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
        return self.color_value(value, "gt", 80)

    def color_upside(self, value: str) -> str:
        """
        Color upside potential based on thresholds.

        Args:
            value: Upside potential string

        Returns:
            Colored upside string
        """
        return self.color_value(value, "gt", 15)

    def color_peg(self, value: str) -> str:
        """
        Color PEG ratio based on thresholds (lower is better).

        Args:
            value: PEG ratio string

        Returns:
            Colored PEG ratio string
        """
        return self.color_value(value, "lt", 1.5, reverse=False)

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
                return f"{Color.RED.value}{value}{Color.RESET.value}"  # High volatility
            else:
                return f"{Color.GREEN.value}{value}{Color.RESET.value}"  # Moderate volatility
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
                    return f"{Color.RED.value}{value}{Color.RESET.value}"  # Expensive
                else:
                    return f"{Color.YELLOW.value}{value}{Color.RESET.value}"  # Moderate
            else:
                if pe < 18:
                    return f"{Color.GREEN.value}{value}{Color.RESET.value}"  # Attractive
                elif pe > 35:
                    return f"{Color.RED.value}{value}{Color.RESET.value}"  # Expensive
                else:
                    return f"{Color.YELLOW.value}{value}{Color.RESET.value}"  # Moderate
        except ValueError:
            # For special values like "--" or non-numeric strings
            if value in ["--", "N/A"]:
                return "--"  # Normalize both to "--"
            # Try to handle negative PE specifically
            if "-" in value:
                return f"{Color.PURPLE.value}{value}{Color.RESET.value}"
            return value

    def _calculate_signal(self, ticker_data: Dict[str, Any]) -> str:
        """
        Calculate the trading signal (BUY, SELL, HOLD, NEUTRAL) based on financial metrics.

        This shared method is used by both get_signal and color_by_signal to ensure consistent
        signal determination and reduce code duplication.

        Args:
            ticker_data: Dictionary of ticker data

        Returns:
            Signal string ("BUY", "SELL", "HOLD", or "NEUTRAL")
        """
        # Import trading criteria from centralized config
        from yahoofinance.core.trade_criteria_config import TradingCriteria

        # Extract and convert relevant metrics
        upside = ticker_data.get("upside")
        buy_percentage = ticker_data.get("buy_percentage")
        pe_forward = ticker_data.get("pe_forward")
        pe_trailing = ticker_data.get("pe_trailing")
        peg = ticker_data.get("peg_ratio")
        beta = ticker_data.get("beta")
        short_interest = ticker_data.get("short_float_pct")
        analyst_count = ticker_data.get("analyst_count")
        total_ratings = ticker_data.get("total_ratings")

        # Get the criteria values from the central config
        # Use centralized TradingCriteria class
        sell_criteria = TradingCriteria
        buy_criteria = TradingCriteria
        confidence = TradingCriteria

        # Confidence check - require both analyst metrics to be present with minimum values
        if (
            analyst_count is None
            or total_ratings is None
            or upside is None
            or buy_percentage is None
            or analyst_count < confidence.MIN_ANALYST_COUNT
            or total_ratings < confidence.MIN_PRICE_TARGETS
        ):
            return "NEUTRAL"

        # Calculate expected return
        expected_return = upside * buy_percentage / 100

        # Check sell signals first (any trigger a sell)
        # Each condition checks if the metric exists before applying criteria
        sell_signals = []

        # Primary criteria - always check these
        sell_signals.append(upside < sell_criteria.SELL_MAX_UPSIDE)
        sell_signals.append(buy_percentage < sell_criteria.SELL_MIN_BUY_PERCENTAGE)
        sell_signals.append(expected_return < sell_criteria.SELL_MAX_EXRET)

        # Secondary criteria - only check if the data is available
        # PE deteriorating (both positive, but forward > trailing)
        if (
            pe_forward is not None
            and pe_trailing is not None
            and pe_forward > 0
            and pe_trailing > 0
        ):
            sell_signals.append(pe_forward > pe_trailing)

        # Extremely high forward PE or negative forward PE
        if pe_forward is not None:
            sell_signals.append(pe_forward > sell_criteria.SELL_MIN_FORWARD_PE or pe_forward < 0)

        # High PEG ratio (optional secondary criterion)
        if peg is not None:
            try:
                peg_val = float(peg)
                sell_signals.append(peg_val > sell_criteria.SELL_MIN_PEG)
            except (ValueError, TypeError):
                pass  # Ignore conversion errors

        # High short interest (optional secondary criterion)
        if short_interest is not None:
            try:
                si_val = float(short_interest)
                sell_signals.append(si_val > sell_criteria.SELL_MIN_SHORT_INTEREST)
            except (ValueError, TypeError):
                pass  # Ignore conversion errors

        # Excessive volatility
        if beta is not None:
            try:
                beta_val = float(beta)
                sell_signals.append(beta_val > sell_criteria.SELL_MIN_BETA)
            except (ValueError, TypeError):
                pass  # Ignore conversion errors

        if any(sell_signals):
            return "SELL"

        # Then check buy signals

        # PRIMARY CRITERIA - Check if required fields are present and valid
        # Beta, PET, and PEF are primary (required) criteria
        if (
            upside is None
            or buy_percentage is None
            or pe_forward is None
            or pe_trailing is None
            or beta is None
            or upside < buy_criteria.BUY_MIN_UPSIDE
            or buy_percentage < buy_criteria.BUY_MIN_BUY_PERCENTAGE
            or expected_return < buy_criteria.BUY_MIN_EXRET
        ):
            return "HOLD"  # Missing required data or basic criteria not met

        # Check PE condition (required - primary criterion)
        pe_condition = False
        if (
            pe_forward < buy_criteria.BUY_MIN_FORWARD_PE
            or pe_forward > buy_criteria.BUY_MAX_FORWARD_PE
        ):
            return "HOLD"  # PE outside acceptable range

        # Check trailing PE condition (required - primary criterion)
        if pe_trailing > 0:
            pe_condition = pe_forward < pe_trailing  # Improving PE
        else:
            pe_condition = True  # Trailing is negative/zero (growth case)

        if not pe_condition:
            return "HOLD"  # PE condition not met

        # Beta range check (required - primary criterion)
        try:
            beta_val = float(beta)
            if not (
                beta_val >= buy_criteria.BUY_MIN_BETA
                and beta_val <= buy_criteria.BUY_MAX_BETA
            ):
                return "HOLD"  # Beta outside acceptable range
        except (ValueError, TypeError):
            return "HOLD"  # Invalid beta value

        # SECONDARY CRITERIA - only check if available (optional)

        # PEG check (optional secondary criterion)
        if peg is not None and peg != "--":
            try:
                peg_val = float(peg)
                if peg_val > 0 and peg_val >= buy_criteria.BUY_MAX_PEG:
                    return "HOLD"  # PEG too high
            except (ValueError, TypeError):
                pass  # Ignore conversion errors

        # Short interest check (optional secondary criterion)
        if short_interest is not None:
            try:
                si_val = float(short_interest)
                if si_val > buy_criteria.BUY_MAX_SHORT_INTEREST:
                    return "HOLD"  # Short interest too high
            except (ValueError, TypeError):
                pass  # Ignore conversion errors

        # If we got here, all criteria are met
        return "BUY"

    def color_by_signal(self, ticker_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine the overall buy/sell/hold signal color based on analysis.

        Args:
            ticker_data: Dictionary of ticker data

        Returns:
            Dictionary mapping column names to colored values
        """
        colored_data = {}

        # Get signal using shared method
        signal = self._calculate_signal(ticker_data)

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
        # Use the shared signal calculation method
        return self._calculate_signal(ticker_data)


def create_formatter(
    output_format: str = "console", compact_mode: bool = False, show_colors: bool = True, **kwargs
) -> Union[DisplayFormatter, Any]:
    """
    Factory function to create an appropriate formatter based on output format.

    This function creates and returns a formatter appropriate for the
    specified output format. It's designed to be used with the dependency
    injection system.

    Args:
        output_format: The desired output format ('console', 'html', etc.)
        compact_mode: Whether to use compact formatting
        show_colors: Whether to include colors in output
        **kwargs: Additional keyword arguments passed to formatter constructor

    Returns:
        An appropriate formatter instance for the specified format

    Raises:
        ValidationError: When the output format is invalid or not supported
    """
    from yahoofinance.core.errors import ValidationError
    from yahoofinance.core.logging import get_logger

    logger = get_logger(__name__)

    try:
        if output_format == "console":
            # Return standard display formatter for console output
            formatter = DisplayFormatter(compact_mode=compact_mode)
            # Apply additional configuration
            formatter.config.show_colors = show_colors
            for key, value in kwargs.items():
                if hasattr(formatter.config, key):
                    setattr(formatter.config, key, value)
            return formatter

        elif output_format == "html":
            # Import HTML formatter dynamically to avoid circular imports
            try:
                from yahoofinance.presentation.html import HTMLFormatter

                return HTMLFormatter(compact_mode=compact_mode, **kwargs)
            except ImportError:
                logger.error("HTML formatter not available")
                raise ValidationError("HTML formatter not available")

        elif output_format == "json":
            # Import JSON formatter dynamically to avoid circular imports
            # This is a placeholder for potential JSON formatting
            try:
                from yahoofinance.presentation.json_format import JSONFormatter

                return JSONFormatter(**kwargs)
            except ImportError:
                logger.error("JSON formatter not available")
                raise ValidationError("JSON formatter not available")

        else:
            logger.error(f"Invalid output format: {output_format}")
            raise ValidationError(f"Invalid output format: {output_format}")

    except Exception as e:
        logger.error(f"Error creating formatter: {str(e)}")
        # Fall back to basic formatter if possible
        logger.info("Falling back to basic display formatter")
        return DisplayFormatter(compact_mode=compact_mode)

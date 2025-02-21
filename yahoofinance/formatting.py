from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ColorCode:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"   # Strong buy signal
    YELLOW = "\033[93m"  # Low confidence rating
    RED = "\033[91m"     # Strong sell signal
    RESET = "\033[0m"    # Reset color
    DEFAULT = ""         # Neutral/hold rating

class Color(Enum):
    """Color enum with semantic meanings"""
    STRONG_BUY = ColorCode.GREEN
    LOW_CONFIDENCE = ColorCode.YELLOW
    STRONG_SELL = ColorCode.RED
    RESET = ColorCode.RESET
    NEUTRAL = ColorCode.DEFAULT

@dataclass
class DisplayConfig:
    """Configuration for display formatting"""
    use_colors: bool = True
    date_format: str = "%Y-%m-%d"
    float_precision: int = 2
    percentage_precision: int = 1
    table_format: str = "fancy_grid"
    min_analysts: int = 4  # Minimum analysts for high confidence rating
    high_upside: float = 15.0  # Threshold for buy signal
    low_upside: float = 5.0   # Threshold for sell signal
    high_buy_percent: float = 65.0  # Threshold for strong buy signal

class DisplayFormatter:
    """Handles formatting of display output"""
    
    def __init__(self, config: DisplayConfig = DisplayConfig()):
        self.config = config
        self.ANSI_ESCAPE = re.compile(r'(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def _convert_numeric(self, value: Any, default: float = 0.0) -> float:
        """
        Convert value to float, handling invalid values.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Converted float value or default
        """
        try:
            if value not in [None, "N/A", "--"]:
                return float(value)
            return default
        except (ValueError, TypeError):
            return default

    def _get_color_code(self,
                       num_targets: Optional[int],
                       upside: Optional[float],
                       total_ratings: Optional[int],
                       percent_buy: Optional[float]) -> Color:
        """
        Determine color coding based on analyst metrics.
        
        Args:
            num_targets: Number of analyst price targets
            upside: Calculated upside percentage
            total_ratings: Total number of analyst ratings
            percent_buy: Percentage of buy ratings
            
        Returns:
            Color enum value based on metrics analysis
        """
        if not self.config.use_colors:
            return Color.NEUTRAL

        try:
            # Convert values with proper type hints
            num_targets = int(self._convert_numeric(num_targets))
            total_ratings = int(self._convert_numeric(total_ratings))
            upside = self._convert_numeric(upside)
            percent_buy = self._convert_numeric(percent_buy)

            # Low confidence check
            if num_targets <= self.config.min_analysts or total_ratings <= self.config.min_analysts:
                return Color.LOW_CONFIDENCE

            # Strong buy signal
            if (num_targets > self.config.min_analysts and
                upside > self.config.high_upside and
                total_ratings > self.config.min_analysts and
                percent_buy > self.config.high_buy_percent):
                return Color.STRONG_BUY

            # Strong sell signal
            if ((num_targets > self.config.min_analysts and upside < self.config.low_upside) or
                (total_ratings > self.config.min_analysts and percent_buy < 50)):
                return Color.STRONG_SELL

            # Default to neutral
            return Color.NEUTRAL

        except Exception as e:
            logger.warning(f"Error determining color code: {str(e)}")
            return Color.NEUTRAL

    def format_value(self, value: Any, decimals: int = 1, percent: bool = False) -> str:
        """
        Format numeric values with proper handling.
        
        Args:
            value: Value to format
            decimals: Number of decimal places
            percent: Whether to format as percentage
            
        Returns:
            Formatted string representation
        """
        if value is None or value in ["N/A", "--", ""]:
            return "--"
            
        try:
            # Remove commas and convert to float
            value = float(str(value).replace(',', ''))
            
            # Format with specified precision
            formatted = f"{value:.{decimals}f}"
            
            # Add percentage symbol if needed
            return f"{formatted}%" if percent else formatted
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Value formatting failed: {str(e)}")
            return str(value)

    def format_date(self, date_str: Optional[str]) -> str:
        """
        Format date strings consistently.
        
        Args:
            date_str: Date string to format
            
        Returns:
            Formatted date string or placeholder
        """
        if not date_str or date_str == "--":
            return "--"
            
        try:
            date = pd.to_datetime(date_str)
            return date.strftime(self.config.date_format)
        except (ValueError, TypeError) as e:
            logger.debug(f"Date formatting failed: {str(e)}")
            return "--"

    def remove_ansi(self, text: Union[str, Any]) -> str:
        """
        Remove ANSI color codes from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with ANSI codes removed
        """
        if not isinstance(text, str):
            return str(text)
        return self.ANSI_ESCAPE.sub("", text)

    def colorize(self, text: Union[str, Any], color: Color) -> str:
        """
        Apply color to text if colors are enabled.
        
        Args:
            text: Text to colorize
            color: Color enum value to apply
            
        Returns:
            Colorized text string
        """
        text_str = str(text)
        if not self.config.use_colors or color == Color.NEUTRAL:
            return text_str
        return f"{color.value}{text_str}{Color.RESET.value}"

    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate derived metrics from raw data."""
        price = data.get("price")
        target = data.get("target_price")
        percent_buy = data.get("buy_percentage")
        
        # Calculate upside
        upside = None
        if price is not None and target is not None:
            upside = 0 if price == 0 else ((target / price - 1) * 100)
            
        # Calculate expected return
        ex_ret = None
        if upside is not None and percent_buy is not None:
            ex_ret = (upside * percent_buy / 100)
            
        return {
            "upside": upside,
            "ex_ret": ex_ret
        }
        
    def _format_row_fields(self, data: Dict[str, Any], metrics: Dict[str, Optional[float]], color: Color) -> Dict[str, str]:
        """Format all fields with proper formatting and colors."""
        return {
            "TICKER": self.colorize(data.get("ticker", ""), color),
            "PRICE": self.colorize(self.format_value(data.get("price"), 2), color),
            "TARGET": self.colorize(self.format_value(data.get("target_price"), 1), color),
            "UPSIDE": self.colorize(self.format_value(metrics["upside"], 1, True), color),
            "# T": self.colorize(self.format_value(data.get("analyst_count"), 0), color),
            "% BUY": self.colorize(self.format_value(data.get("buy_percentage"), 1, True), color),
            "# A": self.colorize(self.format_value(data.get("total_ratings"), 0), color),
            "A": self.colorize(data.get("A", ""), color),  # Add A column after # A
            "EXRET": self.colorize(self.format_value(metrics["ex_ret"], 1, True), color),
            "BETA": self.colorize(self.format_value(data.get("beta")), color),
            "PET": self.colorize(self.format_value(data.get("pe_trailing")), color),
            "PEF": self.colorize(self.format_value(data.get("pe_forward")), color),
            "PEG": self.colorize(self.format_value(data.get("peg_ratio")), color),
            "DIV %": self.colorize(self.format_value(data.get("dividend_yield"), 2, True), color),
            "SI": self.colorize(self.format_value(data.get("short_float_pct"), 1, True), color),
            "INS %": self.colorize(self.format_value(data.get("insider_buy_pct"), 1, True), color),
            "# INS": self.colorize(self.format_value(data.get("insider_transactions"), 0), color),
            "EARNINGS": self.colorize(self.format_date(data.get("last_earnings")), color)
        }
        
    def _get_sort_values(self, data: Dict[str, Any], metrics: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Get values used for sorting."""
        return {
            "_sort_exret": metrics["ex_ret"],
            "_sort_earnings": pd.to_datetime(data.get("last_earnings")) if data.get("last_earnings") else pd.NaT
        }

    def format_stock_row(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single stock's data row with proper formatting and colors.
        
        Args:
            data: Raw stock data dictionary
            
        Returns:
            Dictionary containing formatted fields with colors and sort values
        """
        try:
            # Calculate derived metrics
            metrics = self._calculate_metrics(data)
            
            # Get color based on metrics
            color = self._get_color_code(
                data.get("analyst_count"),
                metrics["upside"],
                data.get("total_ratings"),
                data.get("buy_percentage")
            )
            
            # Format all fields
            formatted = self._format_row_fields(data, metrics, color)
            
            # Add sort values
            formatted.update(self._get_sort_values(data, metrics))
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting stock row: {str(e)}")
            return {field: "--" for field in [
                "TICKER", "PRICE", "TARGET", "UPSIDE", "# T",
                "% BUY", "# A", "A", "EXRET", "BETA", "PET", "PEF", "PEG", "DIV %", "SI",
                "INS %", "# INS", "EARNINGS", "_sort_exret", "_sort_earnings"
            ]}

    def _validate_dataframe_input(self, rows: List[Dict[str, Any]]) -> None:
        """
        Validate input rows for DataFrame creation.
        
        Args:
            rows: List of dictionaries containing stock data
            
        Raises:
            ValueError: If input is invalid or missing required fields
        """
        if not rows:
            raise ValueError("No rows provided to create DataFrame")
            
        required_cols = ['_sort_exret', '_sort_earnings']
        if not all(col in rows[0] for col in required_cols):
            raise ValueError(f"Input rows missing required sort columns: {required_cols}")

    def create_sortable_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create DataFrame with proper sorting and ranking.
        
        Args:
            rows: List of dictionaries containing stock data
                 Each dictionary must contain '_sort_exret' and '_sort_earnings'
                 for sorting purposes
            
        Returns:
            pandas DataFrame with:
                - Rows sorted by expected return and earnings date
                - Ranking column added
                - Sort helper columns removed
                
        Raises:
            ValueError: If input rows are invalid or missing required fields
        """
        try:
            # Validate input
            self._validate_dataframe_input(rows)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            if not df.empty:
                # Sort using the raw numeric values
                df = df.sort_values(
                    by=['_sort_exret', '_sort_earnings'],
                    ascending=[False, False],
                    na_position='last'
                ).reset_index(drop=True)
                
                # Add ranking
                df.insert(0, "#", range(1, len(df) + 1))
                
                # Remove sorting columns
                df = df.drop(columns=['_sort_exret', '_sort_earnings'])
                
            return df
            
        except ValueError as e:
            logger.error(f"Validation error creating DataFrame: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating DataFrame: {str(e)}")
            raise ValueError(f"Failed to create sortable DataFrame: {str(e)}") from e
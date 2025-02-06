from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ANSI color codes
class Color(Enum):
    GREEN = "\033[92m"  # Buy
    YELLOW = "\033[93m"  # Low Confidence
    RED = "\033[91m"    # Sell
    RESET = "\033[0m"
    DEFAULT = ""        # Hold

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

    def _get_color_code(self, 
                       num_targets: Any, 
                       upside: Any, 
                       total_ratings: Any, 
                       percent_buy: Any) -> Color:
        """Determine color coding based on analyst metrics"""
        if not self.config.use_colors:
            return Color.DEFAULT

        try:
            # Convert values to appropriate types
            num_targets = int(float(num_targets)) if num_targets not in [None, "N/A", "--"] else 0
            total_ratings = int(float(total_ratings)) if total_ratings not in [None, "N/A", "--"] else 0
            upside = float(upside) if upside not in [None, "N/A", "--"] else 0
            percent_buy = float(percent_buy) if percent_buy not in [None, "N/A", "--"] else 0

            # Apply color coding logic matching original implementation
            if num_targets < self.config.min_analysts or total_ratings < self.config.min_analysts:
                return Color.YELLOW  # Low confidence
            if (num_targets > self.config.min_analysts and upside > self.config.high_upside and 
                total_ratings > self.config.min_analysts and percent_buy > self.config.high_buy_percent):
                return Color.GREEN   # Buy signal
            if (num_targets > self.config.min_analysts and upside < self.config.low_upside) or \
               (total_ratings > self.config.min_analysts and percent_buy < 50):
                return Color.RED     # Sell signal
            return Color.DEFAULT     # Hold

        except (ValueError, TypeError):
            return Color.DEFAULT

    def format_value(self, value: Any, decimals: int = 1, percent: bool = False) -> str:
        """Format numeric values with proper handling"""
        if value is None or value in ["N/A", "--", ""]:
            return "--"
            
        try:
            value = float(str(value).replace(',', ''))
            if percent:
                return f"{value:.{decimals}f}%"
            return f"{value:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def format_date(self, date_str: str) -> str:
        """Format date strings consistently"""
        if not date_str or date_str == "--":
            return "--"
        try:
            date = pd.to_datetime(date_str)
            return date.strftime(self.config.date_format)
        except (ValueError, TypeError):
            return "--"

    def remove_ansi(self, text: str) -> str:
        """Remove ANSI color codes from text"""
        return self.ANSI_ESCAPE.sub("", text) if isinstance(text, str) else str(text)

    def colorize(self, text: str, color: Color) -> str:
        """Apply color to text if colors are enabled"""
        if not self.config.use_colors or color == Color.DEFAULT:
            return str(text)
        return f"{color.value}{text}{Color.RESET.value}"

    def format_stock_row(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single stock's data row with proper formatting and colors"""
        try:
            # First, extract and convert numeric values (before formatting)
            price = float(data.get("price", 0) or 0)
            target = float(data.get("target_price", 0) or 0)
            upside = ((target / price - 1) * 100) if price and target else 0
            num_targets = int(float(data.get("analyst_count", 0) or 0))
            percent_buy = float(data.get("buy_percentage", 0) or 0)
            total_ratings = int(float(data.get("total_ratings", 0) or 0))
            ex_ret = (upside * percent_buy / 100) if upside is not None and percent_buy is not None else None

            # Get color based on raw numeric values
            color = self._get_color_code(num_targets, upside, total_ratings, percent_buy)

            # Format values and apply color
            return {
                "TICKER": self.colorize(data.get("ticker", ""), color),
                "PRICE": self.colorize(self.format_value(price, 2), color),
                "TARGET": self.colorize(self.format_value(target, 1), color),
                "UPSIDE": self.colorize(self.format_value(upside, 1, True), color),
                "# T": self.colorize(self.format_value(num_targets, 0), color),
                "% BUY": self.colorize(self.format_value(percent_buy, 1, True), color),
                "# A": self.colorize(self.format_value(total_ratings, 0), color),
                "EXRET": self.colorize(self.format_value(ex_ret, 1, True), color),
                "PE": self.colorize(self.format_value(data.get("pe_trailing")), color),
                "PEG": self.colorize(self.format_value(data.get("peg_ratio")), color),
                "DIV %": self.colorize(self.format_value(data.get("dividend_yield"), 2, True), color),
                "INS %": self.colorize(self.format_value(data.get("insider_buy_pct"), 1, True), color),
                "# INS": self.colorize(self.format_value(data.get("insider_transactions"), 0), color),
                "EARNINGS": self.colorize(self.format_date(data.get("last_earnings")), color),
                "_sort_exret": ex_ret,  # Raw value for sorting
                "_sort_earnings": pd.to_datetime(data.get("last_earnings")) if data.get("last_earnings") else pd.NaT
            }
        except Exception as e:
            logger.error(f"Error formatting stock row: {str(e)}")
            return {field: "--" for field in [
                "TICKER", "PRICE", "TARGET", "UPSIDE", "# T",
                "% BUY", "# A", "EXRET", "PE", "PEG", "DIV %",
                "INS %", "# INS", "EARNINGS", "_sort_exret", "_sort_earnings"
            ]}

    def create_sortable_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame with proper sorting"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating sortable DataFrame: {str(e)}")
            return pd.DataFrame()
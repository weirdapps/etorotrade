from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
import pandas as pd
import logging
from .config import TRADING_CRITERIA

logger = logging.getLogger(__name__)

class ColorCode:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"   # Buy signal
    YELLOW = "\033[93m"  # Low confidence rating
    RED = "\033[91m"     # Sell signal
    RESET = "\033[0m"    # Reset color
    DEFAULT = ""         # Neutral/hold rating

class Color(Enum):
    """Color enum with semantic meanings"""
    BUY = ColorCode.GREEN
    LOW_CONFIDENCE = ColorCode.YELLOW
    SELL = ColorCode.RED
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
    
    # Confidence thresholds - align with TRADING_CRITERIA
    min_analysts: int = TRADING_CRITERIA["COMMON"]["MIN_ANALYST_COUNT"]
    
    # Buy signal thresholds - align with TRADING_CRITERIA
    high_upside: float = TRADING_CRITERIA["BUY"]["MIN_UPSIDE"]
    high_buy_percent: float = TRADING_CRITERIA["BUY"]["MIN_BUY_PERCENTAGE"]
    max_beta_buy: float = TRADING_CRITERIA["BUY"]["MAX_BETA"] 
    min_beta_buy: float = TRADING_CRITERIA["BUY"]["MIN_BETA"]
    max_peg_buy: float = TRADING_CRITERIA["BUY"]["MAX_PEG_RATIO"]
    max_si_buy: float = TRADING_CRITERIA["BUY"]["MAX_SHORT_INTEREST"]
    
    # Sell signal thresholds - align with TRADING_CRITERIA
    low_upside: float = TRADING_CRITERIA["SELL"]["MAX_UPSIDE"]
    low_buy_percent: float = TRADING_CRITERIA["SELL"]["MAX_BUY_PERCENTAGE"]
    max_peg_sell: float = TRADING_CRITERIA["SELL"]["MAX_PEG_RATIO"]
    max_si_sell: float = TRADING_CRITERIA["SELL"]["MIN_SHORT_INTEREST"]
    max_exret: float = TRADING_CRITERIA["SELL"]["MAX_EXRET"]

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

    def _get_color_code(self, data: Dict[str, Any], metrics: Dict[str, Any]) -> Color:
        """
        Determine color coding based on comprehensive financial metrics.
        
        Args:
            data: Dictionary containing all stock metrics
            metrics: Dictionary containing calculated metrics (upside, ex_ret)
            
        Returns:
            Color enum value based on metrics analysis
        """
        if not self.config.use_colors:
            return Color.NEUTRAL

        try:
            # Extract required metrics
            num_targets = int(self._convert_numeric(data.get("analyst_count")))
            total_ratings = int(self._convert_numeric(data.get("total_ratings")))
            upside = self._convert_numeric(metrics.get("upside"))
            percent_buy = self._convert_numeric(data.get("buy_percentage"))
            beta = self._convert_numeric(data.get("beta"))
            pef = self._convert_numeric(data.get("pe_forward"))
            pet = self._convert_numeric(data.get("pe_trailing"))
            peg = self._convert_numeric(data.get("peg_ratio"))
            si = self._convert_numeric(data.get("short_float_pct"))
            
            # Check if SI data is missing
            si_missing = data.get("short_float_pct") in [None, "N/A", "--", ""]
            
            # 1. First check: Low Confidence/Inconclusive (Yellow)
            if num_targets < self.config.min_analysts or total_ratings < self.config.min_analysts:
                return Color.LOW_CONFIDENCE
                
            # From here, we know we have sufficient analyst coverage
            
            # 2. Second check: Sell Signal (Red)
            # Get EXRET for evaluation
            ex_ret = self._convert_numeric(metrics.get("ex_ret"))
            
            if (upside < self.config.low_upside or
                percent_buy <= self.config.low_buy_percent or
                (pef > pet and pef > 0 and pet > 0) or  # PEF > PET (if both are positive)
                peg > self.config.max_peg_sell or  # PEG too high
                (not si_missing and si > self.config.max_si_sell) or  # High short interest
                ex_ret < self.config.max_exret):  # EXRET too low
                return Color.SELL
                
            # 3. Third check: Buy Signal (Green)
            # Check if PEG is missing
            peg_missing = data.get("peg_ratio") in [None, "N/A", "--", ""]
            
            # Get minimum PE Forward threshold from TRADING_CRITERIA
            min_pe_forward = TRADING_CRITERIA["BUY"]["MIN_PE_FORWARD"]
            
            # Make sure PEF > min_pe_forward is strictly enforced
            if (upside >= self.config.high_upside and
            percent_buy >= self.config.high_buy_percent and
            beta <= self.config.max_beta_buy and
            beta > self.config.min_beta_buy and
            (pef < pet or pet <= 0) and  # PEF < PET or PET non-positive
            pef > min_pe_forward and  # PE Forward must be > MIN_PE_FORWARD (0.5)
            (peg_missing or peg < self.config.max_peg_buy) and  # PEG must be < max_peg_buy or missing
            (si_missing or si <= self.config.max_si_buy)):  # Low or missing short interest
                # Add debug logging to diagnose PEG condition failures
                logger.debug(f"Buy condition check: Ticker: {data.get('ticker')}, "
                            f"PEG: {peg}, PEG missing: {peg_missing}, "
                            f"Max PEG buy: {self.config.max_peg_buy}, "
                            f"PEG condition: {peg_missing or peg < self.config.max_peg_buy}")
                return Color.BUY
                
            # 4. Fourth check: Hold Signal (Neutral/White) - everything else
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
            
    def format_market_cap(self, value: Any) -> str:
        """
        Format market cap in trillions or billions with a 'T' or 'B' suffix.
        
        Formatting rules:
        - >= 10T: 1 decimal (e.g. "10.5T")
        - >= 1T and < 10T: 2 decimals (e.g. "2.75T")
        - >= 100B: No decimals (e.g. "100B")
        - >= 10B and < 100B: 1 decimal (e.g. "50.5B")
        - < 10B: 2 decimals (e.g. "5.25B")
        - No dollar sign
        
        Args:
            value: Market cap value
            
        Returns:
            Formatted market cap string according to rules
        """
        if value is None or value in ["N/A", "--", ""]:
            return "--"
        try:
            # Convert to a float value
            value_float = float(str(value).replace(',', ''))
            
            # Check if trillion formatting is needed (>= 1T)
            if value_float >= 1_000_000_000_000:
                value_trillions = value_float / 1_000_000_000_000
                if value_trillions >= 10:
                    return f"{value_trillions:.1f}T"
                else:
                    return f"{value_trillions:.2f}T"
            else:
                # Format in billions
                value_billions = value_float / 1_000_000_000
                
                # Apply formatting rules based on size
                if value_billions >= 100:
                    return f"{value_billions:.0f}B"
                elif value_billions >= 10:
                    return f"{value_billions:.1f}B"
                else:
                    return f"{value_billions:.2f}B"
        except (ValueError, TypeError):
            return "--"

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
        
        fields = {
            # Always add TICKER first
            "TICKER": self.colorize(data.get("ticker", ""), color),
        }
        
        # Add COMPANY NAME after TICKER (truncated to 14 characters) - ALL CAPS
        company_name = data.get("company_name", "")
        if company_name and company_name != data.get("ticker", ""):
            # Convert company name to ALL CAPS
            company_name = str(company_name).upper()
            if len(company_name) > 14:
                company_name = company_name[:14]
            fields["COMPANY"] = self.colorize(company_name, color)
        else:
            fields["COMPANY"] = self.colorize("", color)
        
        # Add CAP column after COMPANY
        fields["CAP"] = self.colorize(self.format_market_cap(data.get("market_cap")), color)
            
        # Add remaining columns
        fields.update({
            "PRICE": self.colorize(self.format_value(data.get("price"), 1), color),
            "TARGET": self.colorize(self.format_value(data.get("target_price"), 1), color),
            "UPSIDE": self.colorize(self.format_value(metrics["upside"], 1, True), color),
            "# T": self.colorize(self.format_value(data.get("analyst_count"), 0), color),
            "% BUY": self.colorize(self.format_value(data.get("buy_percentage"), 0, True), color),
            "# A": self.colorize(self.format_value(data.get("total_ratings"), 0), color),
            "A": self.colorize(data.get("A", ""), color),
            "EXRET": self.colorize(self.format_value(metrics["ex_ret"], 1, True), color),  # EXRET right after A
            "BETA": self.colorize(self.format_value(data.get("beta"), 1), color),
            "PET": self.colorize(self.format_value(data.get("pe_trailing"), 1), color),
            "PEF": self.colorize(self.format_value(data.get("pe_forward"), 1), color),
            "PEG": self.colorize(self.format_value(data.get("peg_ratio"), 1), color),
            "DIV %": self.colorize(self.format_value(data.get("dividend_yield"), 2, True), color),
            "SI": self.colorize(self.format_value(data.get("short_float_pct"), 1, True), color),
            "EARNINGS": self.colorize(self.format_date(data.get("last_earnings")), color)
        })
        
        return fields
        
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
            
            # Get color based on comprehensive metrics
            color = self._get_color_code(data, metrics)
            
            # Format all fields
            formatted = self._format_row_fields(data, metrics, color)
            
            # Add sort values
            formatted.update(self._get_sort_values(data, metrics))
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting stock row: {str(e)}")
            return {field: "--" for field in [
                "TICKER", "CAP", "COMPANY", "PRICE", "TARGET", "UPSIDE", "# T",
                "% BUY", "# A", "A", "EXRET", "BETA", "PET", "PEF", "PEG", "DIV %", "SI", 
                "EARNINGS", "_sort_exret", "_sort_earnings"
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
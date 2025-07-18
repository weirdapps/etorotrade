"""
Centralized trade criteria configuration.

This module provides a single source of truth for all trading criteria used in:
- ACT column calculation
- Buy/Sell/Hold opportunity filtering
- Display coloring
- Trade recommendations

The criteria ensure consistency across all components of the system.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd


# Trading action constants
BUY_ACTION = "B"
SELL_ACTION = "S"
HOLD_ACTION = "H"
INCONCLUSIVE_ACTION = "I"  # For insufficient data/confidence
NO_ACTION = ""  # For no data

# Display color constants
GREEN_COLOR = "92"  # Buy
RED_COLOR = "91"    # Sell
YELLOW_COLOR = "93"  # Inconclusive
# No color for Hold - stays default

# Color mapping for actions
ACTION_COLORS = {
    BUY_ACTION: GREEN_COLOR,
    SELL_ACTION: RED_COLOR,
    HOLD_ACTION: None,  # No color
    INCONCLUSIVE_ACTION: YELLOW_COLOR,
    NO_ACTION: None,
}


class TradingCriteria:
    """Centralized trading criteria configuration."""

    # Confidence thresholds
    MIN_ANALYST_COUNT = 5
    MIN_PRICE_TARGETS = 5

    # SELL criteria thresholds
    SELL_MAX_UPSIDE = 5.0              # Sell if upside < 5%
    SELL_MIN_BUY_PERCENTAGE = 65.0     # Sell if buy% < 65%
    SELL_MIN_FORWARD_PE = 65.0         # Sell if PEF > 65
    SELL_MIN_PEG = 3.0                 # Sell if PEG > 3
    SELL_MIN_SHORT_INTEREST = 3.0      # Sell if SI > 3.0%
    SELL_MIN_BETA = 3.0                # Sell if Beta > 3
    SELL_MAX_EXRET = 0.025             # Sell if EXRET < 2.5% (stored as decimal)
    SELL_MAX_EARNINGS_GROWTH = -15.0   # Sell if EG < -15%
    SELL_MAX_PRICE_PERFORMANCE = -35.0 # Sell if PP < -35%

    # BUY criteria thresholds (Adjusted for realistic market conditions)
    BUY_MIN_UPSIDE = 20.0              # Buy if upside >= 20%
    BUY_MIN_BUY_PERCENTAGE = 75.0      # Buy if buy% >= 75%
    BUY_MIN_BETA = 0.25                # Buy if beta > 0.25
    BUY_MAX_BETA = 2.5                 # Buy if beta <= 2.5
    BUY_MIN_FORWARD_PE = 0.5           # Buy if PEF > 0.5
    BUY_MAX_FORWARD_PE = 65.0          # Buy if PEF <= 65
    BUY_MIN_TRAILING_PE = 0.5          # Buy if PET > 0.5  
    BUY_MAX_TRAILING_PE = 80.0         # Buy if PET <= 80
    BUY_MAX_PEG = 2.5                  # Buy if PEG < 2.5 (conditional)
    BUY_MAX_SHORT_INTEREST = 2.0       # Buy if SI <= 2.0%
    BUY_MIN_EXRET = 0.15               # Buy if EXRET >= 15% (stored as decimal)
    BUY_MIN_MARKET_CAP = 1_000_000_000 # Buy if market cap >= $1B
    BUY_MIN_EARNINGS_GROWTH = -10.0    # Buy if EG >= -10% (conditional)
    BUY_MIN_PRICE_PERFORMANCE = -10.0  # Buy if PP >= -10% (conditional)

    @classmethod
    def check_confidence(cls, analyst_count: Optional[float], total_ratings: Optional[float]) -> bool:
        """Check if stock has sufficient analyst coverage for confident recommendations."""
        if analyst_count is None or total_ratings is None:
            return False

        try:
            # Handle string values
            if isinstance(analyst_count, str):
                if analyst_count == "--" or not analyst_count.replace(".", "", 1).isdigit():
                    return False
                analyst_count = float(analyst_count)

            if isinstance(total_ratings, str):
                if total_ratings == "--" or not total_ratings.replace(".", "", 1).isdigit():
                    return False
                total_ratings = float(total_ratings)

            return analyst_count >= cls.MIN_ANALYST_COUNT and total_ratings >= cls.MIN_PRICE_TARGETS
        except (ValueError, TypeError):
            return False

    @classmethod
    def check_sell_criteria(cls, row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a stock meets SELL criteria (ANY condition triggers sell).

        Returns:
            Tuple of (is_sell, reason)
        """
        # 1. Low upside OR low buy percentage
        upside = cls._get_numeric_value(row.get("upside"))
        buy_pct = cls._get_numeric_value(row.get("buy_percentage"))

        if upside is not None and upside < cls.SELL_MAX_UPSIDE:
            return True, f"Low upside ({upside:.1f}% < {cls.SELL_MAX_UPSIDE}%)"

        if buy_pct is not None and buy_pct < cls.SELL_MIN_BUY_PERCENTAGE:
            return True, f"Low buy percentage ({buy_pct:.1f}% < {cls.SELL_MIN_BUY_PERCENTAGE}%)"

        # 2. Deteriorating PE (PEF > PET) OR high forward PE
        pe_forward = cls._get_numeric_value(row.get("pe_forward"))
        pe_trailing = cls._get_numeric_value(row.get("pe_trailing"))

        if pe_forward is not None and pe_trailing is not None:
            if pe_forward > 0 and pe_trailing > 0 and (pe_forward - pe_trailing) > 10:
                return True, f"Worsening P/E (PEF {pe_forward:.1f} - PET {pe_trailing:.1f} > 10)"

        if pe_forward is not None:
            if pe_forward < 0.5:
                return True, f"Low forward P/E ({pe_forward:.1f} < 0.5)"
            elif pe_forward > cls.SELL_MIN_FORWARD_PE:
                return True, f"High forward P/E ({pe_forward:.1f} > {cls.SELL_MIN_FORWARD_PE})"

        # 3. High PEG ratio
        peg = cls._get_numeric_value(row.get("peg_ratio"))
        if peg is not None and peg >= cls.SELL_MIN_PEG:
            return True, f"High PEG ratio ({peg:.1f} >= {cls.SELL_MIN_PEG})"

        # 4. High short interest
        si = cls._get_numeric_value(row.get("short_percent", row.get("SI")))
        if si is not None and si > cls.SELL_MIN_SHORT_INTEREST:
            return True, f"High short interest ({si:.1f}% > {cls.SELL_MIN_SHORT_INTEREST}%)"

        # 5. High beta
        beta = cls._get_numeric_value(row.get("beta"))
        if beta is not None and beta > cls.SELL_MIN_BETA:
            return True, f"High beta ({beta:.1f} > {cls.SELL_MIN_BETA})"

        # 6. Low expected return
        exret = cls._get_numeric_value(row.get("EXRET"))
        if exret is not None:
            # Handle both decimal (0.05) and percentage (5.0) formats
            if exret > 1.0:  # If > 1, assume it's in percentage format, convert to decimal
                exret = exret / 100
            if exret < cls.SELL_MAX_EXRET:
                return True, f"Low expected return ({exret*100:.1f}% < {cls.SELL_MAX_EXRET*100:.0f}%)"

        # 7. Poor earnings growth
        eg = cls._get_numeric_value(row.get("earnings_growth", row.get("EG")))
        if eg is not None and eg < cls.SELL_MAX_EARNINGS_GROWTH:
            return True, f"Poor earnings growth ({eg:.1f}% < {cls.SELL_MAX_EARNINGS_GROWTH}%)"

        # 8. Poor price performance
        pp = cls._get_numeric_value(row.get("price_performance", row.get("PP")))
        if pp is not None and pp < cls.SELL_MAX_PRICE_PERFORMANCE:
            return True, f"Poor price performance ({pp:.1f}% < {cls.SELL_MAX_PRICE_PERFORMANCE}%)"

        return False, None

    @classmethod
    def check_buy_criteria(cls, row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a stock meets BUY criteria (ALL conditions must be met).

        Returns:
            Tuple of (is_buy, failure_reason)
        """
        # 1. High upside AND high buy percentage (both required)
        upside = cls._get_numeric_value(row.get("upside"))
        buy_pct = cls._get_numeric_value(row.get("buy_percentage"))

        if upside is None:
            return False, "Upside data not available"
        if upside < cls.BUY_MIN_UPSIDE:
            return False, f"Insufficient upside ({upside:.1f}% < {cls.BUY_MIN_UPSIDE}%)"

        if buy_pct is None:
            return False, "Buy percentage not available"
        if buy_pct < cls.BUY_MIN_BUY_PERCENTAGE:
            return False, f"Insufficient buy percentage ({buy_pct:.1f}% < {cls.BUY_MIN_BUY_PERCENTAGE}%)"

        # 2. Beta in valid range (required)
        beta = cls._get_numeric_value(row.get("beta"))
        if beta is None:
            return False, "Beta data not available"
        if beta <= cls.BUY_MIN_BETA:
            return False, f"Beta too low ({beta:.1f} <= {cls.BUY_MIN_BETA})"
        if beta > cls.BUY_MAX_BETA:
            return False, f"Beta too high ({beta:.1f} > {cls.BUY_MAX_BETA})"

        # 3. PE conditions (required)
        pe_forward = cls._get_numeric_value(row.get("pe_forward"))
        pe_trailing = cls._get_numeric_value(row.get("pe_trailing"))

        if pe_forward is None:
            return False, "Forward P/E not available"

        # MANDATORY: PET (trailing P/E) is now required for BUY signals
        if pe_trailing is None:
            return False, "Trailing P/E (PET) not available - required for BUY"

        # Check if PE is in valid range
        if not (cls.BUY_MIN_FORWARD_PE <= pe_forward <= cls.BUY_MAX_FORWARD_PE):
            return False, f"Forward P/E out of range ({pe_forward:.1f})"

        # Check if trailing PE is in valid range (companies with negative earnings can't get BUY)
        if not (cls.BUY_MIN_TRAILING_PE <= pe_trailing <= cls.BUY_MAX_TRAILING_PE):
            return False, f"Trailing P/E out of range ({pe_trailing:.1f}) - must be between {cls.BUY_MIN_TRAILING_PE} and {cls.BUY_MAX_TRAILING_PE}"

        # PE condition: PEF - PET <= 10 (PE not expanding too much)
        # This allows for reasonable PE expansion but not excessive growth
        pe_difference = pe_forward - pe_trailing
        if pe_difference > 10:
            return False, f"P/E expanding too much (PEF {pe_forward:.1f} - PET {pe_trailing:.1f} = {pe_difference:.1f} > 10)"

        # 4. Secondary criteria (conditional - only checked if data available)
        peg = cls._get_numeric_value(row.get("peg_ratio"))
        if peg is not None and peg >= cls.BUY_MAX_PEG:
            return False, f"PEG ratio too high ({peg:.1f} >= {cls.BUY_MAX_PEG})"

        si = cls._get_numeric_value(row.get("short_percent", row.get("SI")))
        if si is not None and si > cls.BUY_MAX_SHORT_INTEREST:
            return False, f"Short interest too high ({si:.1f}% > {cls.BUY_MAX_SHORT_INTEREST}%)"

        # 5. Market cap (required)
        market_cap = cls._get_numeric_value(row.get("market_cap"))
        if market_cap is None:
            return False, "Market cap not available"
        if market_cap < cls.BUY_MIN_MARKET_CAP:
            market_cap_formatted = cls._format_market_cap_short(market_cap)
            min_cap_formatted = cls._format_market_cap_short(cls.BUY_MIN_MARKET_CAP)
            return False, f"Market cap too small ({market_cap_formatted} < {min_cap_formatted})"

        # 6. Expected return (required)
        exret = cls._get_numeric_value(row.get("EXRET"))
        if exret is None:
            return False, "Expected return not available"
        
        # Handle both decimal (0.15) and percentage (15.0) formats
        if exret > 1.0:  # If > 1, assume it's in percentage format, convert to decimal
            exret = exret / 100
            
        if exret < cls.BUY_MIN_EXRET:
            return False, f"Expected return too low ({exret*100:.1f}% < {cls.BUY_MIN_EXRET*100:.0f}%)"

        # 7. Earnings growth (conditional - only checked if data available)
        eg = cls._get_numeric_value(row.get("earnings_growth", row.get("EG")))
        if eg is not None and eg < cls.BUY_MIN_EARNINGS_GROWTH:
            return False, f"Earnings growth too low ({eg:.1f}% < {cls.BUY_MIN_EARNINGS_GROWTH}%)"
        
        # 8. Price performance (conditional - only checked if data available)
        pp = cls._get_numeric_value(row.get("price_performance", row.get("PP")))
        if pp is not None and pp < cls.BUY_MIN_PRICE_PERFORMANCE:
            return False, f"Price performance too low ({pp:.1f}% < {cls.BUY_MIN_PRICE_PERFORMANCE}%)"

        return True, None

    @classmethod
    def calculate_action(cls, row: Dict[str, Any]) -> Tuple[str, str]:
        """
        Calculate trading action for a stock based on criteria.

        Returns:
            Tuple of (action, reason)
        """
        # Check confidence first
        analyst_count = row.get("analyst_count", row.get("# T"))
        total_ratings = row.get("total_ratings", row.get("# A"))

        if not cls.check_confidence(analyst_count, total_ratings):
            return INCONCLUSIVE_ACTION, "Insufficient analyst coverage"

        # Check SELL criteria (any condition triggers)
        is_sell, sell_reason = cls.check_sell_criteria(row)
        if is_sell:
            return SELL_ACTION, sell_reason

        # Check BUY criteria (all conditions required)
        is_buy, buy_reason = cls.check_buy_criteria(row)
        if is_buy:
            return BUY_ACTION, "Meets all buy criteria"

        # Default to HOLD
        return HOLD_ACTION, buy_reason or "Does not meet buy or sell criteria"

    @staticmethod
    def _get_numeric_value(value: Any) -> Optional[float]:
        """Convert various value formats to float."""
        if value is None or pd.isna(value):
            return None

        if isinstance(value, str):
            # Handle special values
            if value == "--" or not value.strip():
                return None

            # Remove percentage sign and convert
            try:
                return float(value.replace("%", "").replace(",", ""))
            except (ValueError, TypeError):
                return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _format_market_cap_short(value: float) -> str:
        """Format market cap for display in error messages."""
        if value >= 1_000_000_000_000:  # Trillion
            return f"{value / 1_000_000_000_000:.1f}T"
        elif value >= 1_000_000_000:  # Billion
            return f"{value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:  # Million
            return f"{value / 1_000_000:.0f}M"
        else:
            return f"{value:,.0f}"


def get_action_color(action: str) -> Optional[str]:
    """Get the color code for a given action."""
    return ACTION_COLORS.get(action)


def normalize_row_for_criteria(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize row column names for criteria evaluation.

    Maps display column names to internal names used by criteria.
    """
    mapping = {
        "UPSIDE": "upside",
        "% BUY": "buy_percentage",
        "BETA": "beta",
        "PET": "pe_trailing",
        "PEF": "pe_forward",
        "PEG": "peg_ratio",
        "SI": "short_percent",
        "# T": "analyst_count",
        "# A": "total_ratings",
        "EXRET": "EXRET",  # Keep as is
        "CAP": "market_cap",
        "EG": "earnings_growth",  # Map EG display column to internal name
        "PP": "price_performance",  # Map PP display column to internal name
    }

    normalized = {}

    # Copy all original keys first
    for key, value in row.items():
        normalized[key] = value

    # Apply mapping
    for display_name, internal_name in mapping.items():
        if display_name in row and internal_name not in row:
            value = row[display_name]
            
            # Special handling for market cap - convert formatted string to numeric
            if internal_name == "market_cap" and isinstance(value, str):
                value = _parse_market_cap_string(value)
            
            normalized[internal_name] = value

    # Calculate upside dynamically from price and target_price if not already present
    if "upside" not in normalized and "price" in row:
        # Try to use validated upside calculation with price target quality assessment
        try:
            from yahoofinance.utils.data.format_utils import calculate_validated_upside, calculate_upside
            
            # First try validated upside with quality assessment
            validated_upside, quality_desc = calculate_validated_upside(normalized)
            if validated_upside is not None:
                normalized["upside"] = validated_upside
                normalized["upside_quality"] = quality_desc
            # Fallback to simple calculation if we have target_price
            elif "target_price" in row and row.get("target_price"):
                normalized["upside"] = calculate_upside(row.get("price"), row.get("target_price"))
                normalized["upside_quality"] = "simple_calculation"
        except ImportError:
            # Fallback for backward compatibility
            if "target_price" in row:
                from yahoofinance.utils.data.format_utils import calculate_upside
                normalized["upside"] = calculate_upside(row.get("price"), row.get("target_price"))
                normalized["upside_quality"] = "legacy_calculation"

    return normalized


def _parse_market_cap_string(cap_str: str) -> Optional[float]:
    """Parse market cap string (e.g., '3.67B') to numeric value."""
    if cap_str == "--" or not cap_str or cap_str.strip() == "":
        return None
    
    try:
        cap_str = cap_str.upper().strip()
        if cap_str.endswith('T'):
            return float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            return float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            return float(cap_str[:-1]) * 1_000_000
        else:
            return float(cap_str)
    except (ValueError, TypeError):
        return None

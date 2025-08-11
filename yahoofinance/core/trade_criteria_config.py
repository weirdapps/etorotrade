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

    # Market cap tier thresholds (in dollars)
    VALUE_TIER_MIN_CAP = 100_000_000_000  # $100B
    GROWTH_TIER_MIN_CAP = 5_000_000_000   # $5B
    
    # Tier classification constants
    VALUE_TIER = "V"
    GROWTH_TIER = "G"
    BETS_TIER = "B"

    # Confidence thresholds - Strict criteria for reliable trading decisions
    MIN_ANALYST_COUNT = 5
    MIN_PRICE_TARGETS = 5

    # SELL criteria thresholds (shared across tiers, but can be overridden per tier)
    SELL_MAX_UPSIDE = 5.0              # Default sell if upside < 5%
    SELL_MIN_BUY_PERCENTAGE = 65.0     # Default sell if buy% < 65%
    SELL_MIN_FORWARD_PE = 65.0         # Sell if PEF > 65
    SELL_MIN_PEG = 3.0                 # Sell if PEG > 3
    SELL_MIN_SHORT_INTEREST = 3.0      # Sell if SI > 3.0%
    SELL_MIN_BETA = 3.0                # Sell if Beta > 3
    SELL_MAX_EXRET = 0.025             # Sell if EXRET < 2.5% (stored as decimal)
    SELL_MAX_EARNINGS_GROWTH = -15.0   # Sell if EG < -15%
    SELL_MAX_PRICE_PERFORMANCE = -35.0 # Sell if PP < -35%

    # VALUE tier BUY criteria (≥$100B market cap) - Conservative quality large-caps
    VALUE_BUY_MIN_UPSIDE = 15.0              # Reasonable upside for large-caps
    VALUE_BUY_MIN_BUY_PERCENTAGE = 70.0      # Strong analyst consensus required
    VALUE_BUY_MIN_BETA = 0.25                # Minimum beta allowed
    VALUE_BUY_MAX_BETA = 3.0                 # Maximum beta allowed
    VALUE_BUY_MIN_FORWARD_PE = 0.5           # Minimum forward PE
    VALUE_BUY_MAX_FORWARD_PE = 65.0          # Maximum forward PE
    VALUE_BUY_MIN_TRAILING_PE = 0.5          # Minimum trailing PE
    VALUE_BUY_MAX_TRAILING_PE = 85.0         # Higher trailing PE allowed for stability
    VALUE_BUY_MAX_PEG = 2.5                  # PEG requirement
    VALUE_BUY_MAX_SHORT_INTEREST = 2.0       # Short interest tolerance
    VALUE_BUY_MIN_EXRET = 0.10               # Expected return threshold (10%)
    VALUE_BUY_MIN_EARNINGS_GROWTH = -5.0     # Stricter requirement for large companies
    VALUE_BUY_MIN_PRICE_PERFORMANCE = -20.0  # Most tolerance for large company underperformance
    
    # GROWTH tier BUY criteria ($5B-$100B market cap) - Standard criteria
    GROWTH_BUY_MIN_UPSIDE = 20.0             # Standard upside requirement
    GROWTH_BUY_MIN_BUY_PERCENTAGE = 75.0     # Standard analyst consensus
    GROWTH_BUY_MIN_BETA = 0.25               # Standard beta range
    GROWTH_BUY_MAX_BETA = 3.0                # Higher beta limit
    GROWTH_BUY_MIN_FORWARD_PE = 0.5          # Standard PE requirements
    GROWTH_BUY_MAX_FORWARD_PE = 65.0         # Standard forward PE limit
    GROWTH_BUY_MIN_TRAILING_PE = 0.5         # Standard trailing PE requirement
    GROWTH_BUY_MAX_TRAILING_PE = 80.0        # Standard trailing PE limit
    GROWTH_BUY_MAX_PEG = 2.5                 # Standard PEG requirement
    GROWTH_BUY_MAX_SHORT_INTEREST = 2.5      # Higher short interest tolerance
    GROWTH_BUY_MIN_EXRET = 0.15              # Standard expected return (15%)
    GROWTH_BUY_MIN_EARNINGS_GROWTH = -10.0   # Standard earnings growth requirement
    GROWTH_BUY_MIN_PRICE_PERFORMANCE = -15.0 # Moderate tolerance for underperformance
    
    # BETS tier BUY criteria (<$5B market cap) - High-conviction speculative positions
    BETS_BUY_MIN_UPSIDE = 25.0               # High upside required for small caps
    BETS_BUY_MIN_BUY_PERCENTAGE = 80.0       # Strong analyst consensus required
    BETS_BUY_MIN_BETA = 0.25                 # Minimum beta allowed
    BETS_BUY_MAX_BETA = 3.0                  # Maximum beta allowed
    BETS_BUY_MIN_FORWARD_PE = 0.5            # Minimum forward PE
    BETS_BUY_MAX_FORWARD_PE = 60.0           # Stricter forward PE limit
    BETS_BUY_MIN_TRAILING_PE = 0.5           # Minimum trailing PE
    BETS_BUY_MAX_TRAILING_PE = 60.0          # Stricter trailing PE limit
    BETS_BUY_MAX_PEG = 2.0                   # Stricter PEG requirement
    BETS_BUY_MAX_SHORT_INTEREST = 2.0        # Short interest tolerance
    BETS_BUY_MIN_EXRET = 0.20                # Higher expected return required (20%)
    BETS_BUY_MIN_EARNINGS_GROWTH = -15.0     # More tolerance for small company volatility
    BETS_BUY_MIN_PRICE_PERFORMANCE = -10.0   # Stricter requirement for small companies

    # Tier-specific SELL criteria (logically consistent progression)
    # VALUE tier SELL criteria (≥$100B market cap) - Lower tolerance for large-caps
    VALUE_SELL_MAX_UPSIDE = 5.0              # Sell if upside < 5%
    VALUE_SELL_MIN_BUY_PERCENTAGE = 50.0     # Sell if buy% < 50%
    VALUE_SELL_MAX_FORWARD_PE = 65.0         # Sell if PEF > 65
    VALUE_SELL_MAX_EXRET = 0.05              # Sell if EXRET < 5%
    
    # GROWTH tier SELL criteria ($5B-$100B market cap) - Moderate tolerance
    GROWTH_SELL_MAX_UPSIDE = 8.0             # Sell if upside < 8%
    GROWTH_SELL_MIN_BUY_PERCENTAGE = 60.0    # Sell if buy% < 60%
    GROWTH_SELL_MAX_FORWARD_PE = 65.0        # Sell if PEF > 65
    GROWTH_SELL_MAX_EXRET = 0.08             # Sell if EXRET < 8%
    
    # BETS tier SELL criteria (<$5B market cap) - Higher tolerance for small-caps
    BETS_SELL_MAX_UPSIDE = 12.0              # Sell if upside < 12%
    BETS_SELL_MIN_BUY_PERCENTAGE = 70.0      # Sell if buy% < 70%
    BETS_SELL_MAX_FORWARD_PE = 65.0          # Sell if PEF > 65
    BETS_SELL_MAX_EXRET = 0.10               # Sell if EXRET < 10%

    # Legacy single-tier criteria (for backward compatibility)
    BUY_MIN_UPSIDE = GROWTH_BUY_MIN_UPSIDE
    BUY_MIN_BUY_PERCENTAGE = GROWTH_BUY_MIN_BUY_PERCENTAGE
    BUY_MIN_BETA = GROWTH_BUY_MIN_BETA
    BUY_MAX_BETA = GROWTH_BUY_MAX_BETA
    BUY_MIN_FORWARD_PE = GROWTH_BUY_MIN_FORWARD_PE
    BUY_MAX_FORWARD_PE = GROWTH_BUY_MAX_FORWARD_PE
    BUY_MIN_TRAILING_PE = GROWTH_BUY_MIN_TRAILING_PE
    BUY_MAX_TRAILING_PE = GROWTH_BUY_MAX_TRAILING_PE
    BUY_MAX_PEG = GROWTH_BUY_MAX_PEG
    BUY_MAX_SHORT_INTEREST = GROWTH_BUY_MAX_SHORT_INTEREST
    BUY_MIN_EXRET = GROWTH_BUY_MIN_EXRET
    BUY_MIN_MARKET_CAP = 1_000_000_000       # Minimum market cap for any buy
    BUY_MIN_EARNINGS_GROWTH = GROWTH_BUY_MIN_EARNINGS_GROWTH
    BUY_MIN_PRICE_PERFORMANCE = GROWTH_BUY_MIN_PRICE_PERFORMANCE

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
        # Get the tier-specific sell criteria
        tier = cls.get_market_cap_tier(row)
        if tier == cls.VALUE_TIER:
            max_upside = cls.VALUE_SELL_MAX_UPSIDE
            min_buy_pct = cls.VALUE_SELL_MIN_BUY_PERCENTAGE
            max_forward_pe = cls.VALUE_SELL_MAX_FORWARD_PE
            max_exret = cls.VALUE_SELL_MAX_EXRET
        elif tier == cls.GROWTH_TIER:
            max_upside = cls.GROWTH_SELL_MAX_UPSIDE
            min_buy_pct = cls.GROWTH_SELL_MIN_BUY_PERCENTAGE
            max_forward_pe = cls.GROWTH_SELL_MAX_FORWARD_PE
            max_exret = cls.GROWTH_SELL_MAX_EXRET
        else:  # BETS_TIER
            max_upside = cls.BETS_SELL_MAX_UPSIDE
            min_buy_pct = cls.BETS_SELL_MIN_BUY_PERCENTAGE
            max_forward_pe = cls.BETS_SELL_MAX_FORWARD_PE
            max_exret = cls.BETS_SELL_MAX_EXRET

        # 1. Low upside OR low buy percentage (tier-specific thresholds)
        upside = cls._get_numeric_value(row.get("upside"))
        buy_pct = cls._get_numeric_value(row.get("buy_percentage"))

        if upside is not None and upside < max_upside:
            return True, f"Low upside ({upside:.1f}% < {max_upside}% for {tier} tier)"

        if buy_pct is not None and buy_pct < min_buy_pct:
            return True, f"Low buy percentage ({buy_pct:.1f}% < {min_buy_pct}% for {tier} tier)"

        # 2. Deteriorating PE (PEF > PET) OR high forward PE
        pe_forward = cls._get_numeric_value(row.get("pe_forward"))
        pe_trailing = cls._get_numeric_value(row.get("pe_trailing"))

        if pe_forward is not None and pe_trailing is not None:
            if pe_forward > 0 and pe_trailing > 0 and (pe_forward - pe_trailing) > 5:
                return True, f"Worsening P/E (PEF {pe_forward:.1f} - PET {pe_trailing:.1f} > 5)"

        if pe_forward is not None:
            if pe_forward < 0.5:
                return True, f"Low forward P/E ({pe_forward:.1f} < 0.5)"
            elif pe_forward > max_forward_pe:
                return True, f"High forward P/E ({pe_forward:.1f} > {max_forward_pe} for {tier} tier)"

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

        # 6. Low expected return (tier-specific threshold)
        exret = cls._get_numeric_value(row.get("EXRET"))
        if exret is not None:
            # Handle both decimal (0.05) and percentage (5.0) formats
            if exret > 1.0:  # If > 1, assume it's in percentage format, convert to decimal
                exret = exret / 100
            if exret < max_exret:
                return True, f"Low expected return ({exret*100:.1f}% < {max_exret*100:.0f}% for {tier} tier)"

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
    def get_tier_criteria(cls, tier: str) -> Dict[str, float]:
        """Get criteria thresholds for a specific tier."""
        if tier == cls.VALUE_TIER:
            return {
                "min_upside": cls.VALUE_BUY_MIN_UPSIDE,
                "min_buy_percentage": cls.VALUE_BUY_MIN_BUY_PERCENTAGE,
                "min_beta": cls.VALUE_BUY_MIN_BETA,
                "max_beta": cls.VALUE_BUY_MAX_BETA,
                "min_forward_pe": cls.VALUE_BUY_MIN_FORWARD_PE,
                "max_forward_pe": cls.VALUE_BUY_MAX_FORWARD_PE,
                "min_trailing_pe": cls.VALUE_BUY_MIN_TRAILING_PE,
                "max_trailing_pe": cls.VALUE_BUY_MAX_TRAILING_PE,
                "max_peg": cls.VALUE_BUY_MAX_PEG,
                "max_short_interest": cls.VALUE_BUY_MAX_SHORT_INTEREST,
                "min_exret": cls.VALUE_BUY_MIN_EXRET,
                "min_earnings_growth": cls.VALUE_BUY_MIN_EARNINGS_GROWTH,
                "min_price_performance": cls.VALUE_BUY_MIN_PRICE_PERFORMANCE,
            }
        elif tier == cls.GROWTH_TIER:
            return {
                "min_upside": cls.GROWTH_BUY_MIN_UPSIDE,
                "min_buy_percentage": cls.GROWTH_BUY_MIN_BUY_PERCENTAGE,
                "min_beta": cls.GROWTH_BUY_MIN_BETA,
                "max_beta": cls.GROWTH_BUY_MAX_BETA,
                "min_forward_pe": cls.GROWTH_BUY_MIN_FORWARD_PE,
                "max_forward_pe": cls.GROWTH_BUY_MAX_FORWARD_PE,
                "min_trailing_pe": cls.GROWTH_BUY_MIN_TRAILING_PE,
                "max_trailing_pe": cls.GROWTH_BUY_MAX_TRAILING_PE,
                "max_peg": cls.GROWTH_BUY_MAX_PEG,
                "max_short_interest": cls.GROWTH_BUY_MAX_SHORT_INTEREST,
                "min_exret": cls.GROWTH_BUY_MIN_EXRET,
                "min_earnings_growth": cls.GROWTH_BUY_MIN_EARNINGS_GROWTH,
                "min_price_performance": cls.GROWTH_BUY_MIN_PRICE_PERFORMANCE,
            }
        else:  # BETS_TIER
            return {
                "min_upside": cls.BETS_BUY_MIN_UPSIDE,
                "min_buy_percentage": cls.BETS_BUY_MIN_BUY_PERCENTAGE,
                "min_beta": cls.BETS_BUY_MIN_BETA,
                "max_beta": cls.BETS_BUY_MAX_BETA,
                "min_forward_pe": cls.BETS_BUY_MIN_FORWARD_PE,
                "max_forward_pe": cls.BETS_BUY_MAX_FORWARD_PE,
                "min_trailing_pe": cls.BETS_BUY_MIN_TRAILING_PE,
                "max_trailing_pe": cls.BETS_BUY_MAX_TRAILING_PE,
                "max_peg": cls.BETS_BUY_MAX_PEG,
                "max_short_interest": cls.BETS_BUY_MAX_SHORT_INTEREST,
                "min_exret": cls.BETS_BUY_MIN_EXRET,
                "min_earnings_growth": cls.BETS_BUY_MIN_EARNINGS_GROWTH,
                "min_price_performance": cls.BETS_BUY_MIN_PRICE_PERFORMANCE,
            }

    @classmethod
    def check_buy_criteria(cls, row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a stock meets BUY criteria using tier-specific thresholds.

        Returns:
            Tuple of (is_buy, failure_reason)
        """
        # Determine tier and get appropriate criteria
        tier = cls.get_market_cap_tier(row)
        criteria = cls.get_tier_criteria(tier)

        # 1. High upside AND high buy percentage (both required)
        upside = cls._get_numeric_value(row.get("upside"))
        buy_pct = cls._get_numeric_value(row.get("buy_percentage"))

        if upside is None:
            return False, "Upside data not available"
        if upside < criteria["min_upside"]:
            return False, f"Insufficient upside ({upside:.1f}% < {criteria['min_upside']}% for {tier} tier)"

        if buy_pct is None:
            return False, "Buy percentage not available"
        if buy_pct < criteria["min_buy_percentage"]:
            return False, f"Insufficient buy percentage ({buy_pct:.1f}% < {criteria['min_buy_percentage']}% for {tier} tier)"

        # 2. Beta in valid range (required)
        beta = cls._get_numeric_value(row.get("beta"))
        if beta is None:
            return False, "Beta data not available"
        if beta <= criteria["min_beta"]:
            return False, f"Beta too low ({beta:.1f} <= {criteria['min_beta']} for {tier} tier)"
        if beta > criteria["max_beta"]:
            return False, f"Beta too high ({beta:.1f} > {criteria['max_beta']} for {tier} tier)"

        # 3. PE conditions (required)
        pe_forward = cls._get_numeric_value(row.get("pe_forward"))
        pe_trailing = cls._get_numeric_value(row.get("pe_trailing"))

        if pe_forward is None:
            return False, "Forward P/E not available"

        # MANDATORY: PET (trailing P/E) is now required for BUY signals
        if pe_trailing is None:
            return False, "Trailing P/E (PET) not available - required for BUY"

        # Check if PE is in valid range
        if not (criteria["min_forward_pe"] <= pe_forward <= criteria["max_forward_pe"]):
            return False, f"Forward P/E out of range ({pe_forward:.1f}) for {tier} tier"

        # Check if trailing PE is in valid range (companies with negative earnings can't get BUY)
        if not (criteria["min_trailing_pe"] <= pe_trailing <= criteria["max_trailing_pe"]):
            return False, f"Trailing P/E out of range ({pe_trailing:.1f}) for {tier} tier"

        # PE condition: PEF - PET <= 5 (PE not expanding too much)
        # This allows for reasonable PE expansion but not excessive growth
        pe_difference = pe_forward - pe_trailing
        if pe_difference > 5:
            return False, f"P/E expanding too much (PEF {pe_forward:.1f} - PET {pe_trailing:.1f} = {pe_difference:.1f} > 5)"

        # 4. Secondary criteria (conditional - only checked if data available)
        peg = cls._get_numeric_value(row.get("peg_ratio"))
        if peg is not None and peg >= criteria["max_peg"]:
            return False, f"PEG ratio too high ({peg:.1f} >= {criteria['max_peg']} for {tier} tier)"

        si = cls._get_numeric_value(row.get("short_percent", row.get("SI")))
        if si is not None and si > criteria["max_short_interest"]:
            return False, f"Short interest too high ({si:.1f}% > {criteria['max_short_interest']}% for {tier} tier)"

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
            
        if exret < criteria["min_exret"]:
            return False, f"Expected return too low ({exret*100:.1f}% < {criteria['min_exret']*100:.0f}% for {tier} tier)"

        # 7. Earnings growth (conditional - only checked if data available)
        eg = cls._get_numeric_value(row.get("earnings_growth", row.get("EG")))
        if eg is not None and eg < criteria["min_earnings_growth"]:
            return False, f"Earnings growth too low ({eg:.1f}% < {criteria['min_earnings_growth']}% for {tier} tier)"
        
        # 8. Price performance (conditional - only checked if data available)
        pp = cls._get_numeric_value(row.get("price_performance", row.get("PP")))
        if pp is not None and pp < criteria["min_price_performance"]:
            return False, f"Price performance too low ({pp:.1f}% < {criteria['min_price_performance']}% for {tier} tier)"

        return True, None

    @classmethod
    def get_market_cap_tier(cls, row: Dict[str, Any]) -> str:
        """
        Determine market cap tier (VALUE/GROWTH/BETS) for a stock.
        
        Args:
            row: Stock data row
            
        Returns:
            str: Tier classification (V/G/B)
        """
        market_cap = cls._get_numeric_value(row.get("market_cap"))
        if market_cap is None:
            # Try to parse from CAP column if market_cap not available
            cap_str = row.get("CAP")
            if cap_str:
                market_cap = _parse_market_cap_string(cap_str)
        
        if market_cap is None:
            return cls.BETS_TIER  # Default to BETS if no market cap data
            
        if market_cap >= cls.VALUE_TIER_MIN_CAP:
            return cls.VALUE_TIER
        elif market_cap >= cls.GROWTH_TIER_MIN_CAP:
            return cls.GROWTH_TIER
        else:
            return cls.BETS_TIER

    @classmethod
    def calculate_action(cls, row: Dict[str, Any]) -> Tuple[str, str]:
        """
        Calculate trading action for a stock based on criteria.

        Returns:
            Tuple of (action, reason)
        """
        # Check confidence first - but differentiate between asset types
        analyst_count = row.get("analyst_count", row.get("# T"))
        total_ratings = row.get("total_ratings", row.get("# A"))

        if not cls.check_confidence(analyst_count, total_ratings):
            # Determine if this asset type naturally lacks analyst coverage
            ticker = row.get("TICKER", row.get("ticker", ""))
            company_name = row.get("COMPANY", row.get("company_name", ""))
            
            # Import asset classification utility
            try:
                from ..utils.data.asset_type_utils import classify_asset_type
                asset_type = classify_asset_type(ticker, None, company_name)
                
                # ETFs, crypto, and commodities naturally don't have analyst coverage
                if asset_type in ["etf", "crypto", "commodity"]:
                    # Skip analyst coverage check and proceed to other criteria
                    pass  # Continue to sell/buy criteria checks
                else:
                    # For stocks, insufficient analyst coverage is inconclusive
                    return INCONCLUSIVE_ACTION, "Insufficient analyst coverage"
            except ImportError:
                # Fallback: treat as inconclusive if we can't classify
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
        "%BUY": "buy_percentage",     # Handle both formats
        "BETA": "beta",
        "PET": "pe_trailing",
        "PEF": "pe_forward",
        "PEG": "peg_ratio",
        "SI": "short_percent",
        "#T": "analyst_count",        # Current format (no space)
        "# T": "analyst_count",       # Legacy format (with space)
        "#A": "total_ratings",        # Current format (no space)
        "# A": "total_ratings",       # Legacy format (with space)
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

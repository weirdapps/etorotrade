"""
VIX Regime Provider

Fetches live VIX data to adjust trading criteria based on market volatility regime.
During high volatility, we become more conservative; during low volatility, more aggressive.

P1 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class VixRegime(Enum):
    """Market volatility regimes based on VIX levels."""
    LOW = "low"            # VIX < 15: Complacent market
    NORMAL = "normal"      # VIX 15-25: Normal volatility
    ELEVATED = "elevated"  # VIX 25-35: Elevated concern
    HIGH = "high"          # VIX > 35: High fear


# VIX threshold boundaries
VIX_THRESHOLDS = {
    "low_max": 15.0,
    "normal_max": 25.0,
    "elevated_max": 35.0,
}

# Threshold adjustments per regime (multipliers for criteria)
# Values < 1.0 make criteria stricter, > 1.0 make criteria looser
REGIME_ADJUSTMENTS: Dict[VixRegime, Dict[str, float]] = {
    VixRegime.LOW: {
        "min_upside_multiplier": 0.9,      # Slightly stricter (less upside needed when VIX low)
        "min_buy_pct_multiplier": 1.0,     # No change
        "max_upside_sell_offset": 0.0,     # No change
    },
    VixRegime.NORMAL: {
        "min_upside_multiplier": 1.0,      # Normal thresholds
        "min_buy_pct_multiplier": 1.0,     # No change
        "max_upside_sell_offset": 0.0,     # No change
    },
    VixRegime.ELEVATED: {
        "min_upside_multiplier": 1.15,     # Require more upside (15% higher threshold)
        "min_buy_pct_multiplier": 1.05,    # Require slightly higher consensus
        "max_upside_sell_offset": 2.0,     # Less aggressive on sells (add 2% buffer)
    },
    VixRegime.HIGH: {
        "min_upside_multiplier": 1.3,      # Require significantly more upside (30% higher)
        "min_buy_pct_multiplier": 1.1,     # Require stronger consensus
        "max_upside_sell_offset": 5.0,     # Much less aggressive on sells
    },
}

# Cache for VIX data
_vix_cache: Optional[float] = None
_vix_cache_timestamp: Optional[datetime] = None
_vix_cache_lock = threading.Lock()
_VIX_CACHE_TTL_MINUTES = 30  # Refresh every 30 minutes (VIX is more volatile)


def _fetch_vix() -> Optional[float]:
    """Fetch current VIX level from Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="1d")
        if hist is not None and not hist.empty:
            vix = hist["Close"].iloc[-1]
            if vix and vix > 0:
                return round(float(vix), 2)
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch VIX: {e}")
        return None


def _is_cache_valid() -> bool:
    """Check if VIX cache is still valid."""
    if _vix_cache_timestamp is None:
        return False
    return datetime.now() - _vix_cache_timestamp < timedelta(minutes=_VIX_CACHE_TTL_MINUTES)


def get_current_vix() -> Optional[float]:
    """
    Get current VIX level.

    Uses cached value with 30-minute TTL.

    Returns:
        Current VIX level, or None if unavailable
    """
    global _vix_cache, _vix_cache_timestamp

    with _vix_cache_lock:
        if not _is_cache_valid():
            try:
                vix = _fetch_vix()
                if vix is not None:
                    _vix_cache = vix
                    _vix_cache_timestamp = datetime.now()
                    logger.info(f"VIX updated: {vix}")
            except Exception as e:
                logger.warning(f"VIX fetch failed: {e}")

        return _vix_cache


def get_vix_regime() -> VixRegime:
    """
    Determine current VIX regime.

    Returns:
        VixRegime enum indicating current market volatility state
    """
    vix = get_current_vix()

    if vix is None:
        # Default to normal if VIX unavailable
        return VixRegime.NORMAL

    if vix < VIX_THRESHOLDS["low_max"]:
        return VixRegime.LOW
    elif vix < VIX_THRESHOLDS["normal_max"]:
        return VixRegime.NORMAL
    elif vix < VIX_THRESHOLDS["elevated_max"]:
        return VixRegime.ELEVATED
    else:
        return VixRegime.HIGH


def get_regime_adjustments() -> Dict[str, float]:
    """
    Get threshold adjustment multipliers for current VIX regime.

    Returns:
        Dictionary of adjustment multipliers/offsets
    """
    regime = get_vix_regime()
    return REGIME_ADJUSTMENTS[regime].copy()


def adjust_buy_criteria(criteria: Dict, apply_adjustments: bool = True) -> Dict:
    """
    Adjust buy criteria based on current VIX regime.

    Args:
        criteria: Original buy criteria dictionary
        apply_adjustments: Whether to apply VIX-based adjustments

    Returns:
        Adjusted criteria dictionary
    """
    if not apply_adjustments:
        return criteria.copy()

    adjustments = get_regime_adjustments()
    adjusted = criteria.copy()

    # Adjust min_upside
    if "min_upside" in adjusted:
        original = adjusted["min_upside"]
        adjusted["min_upside"] = round(original * adjustments["min_upside_multiplier"], 1)
        if adjusted["min_upside"] != original:
            logger.debug(f"VIX regime: min_upside adjusted {original} -> {adjusted['min_upside']}")

    # Adjust min_buy_percentage
    if "min_buy_percentage" in adjusted:
        original = adjusted["min_buy_percentage"]
        adjusted["min_buy_percentage"] = round(original * adjustments["min_buy_pct_multiplier"], 1)
        # Cap at reasonable maximum
        adjusted["min_buy_percentage"] = min(adjusted["min_buy_percentage"], 95.0)
        if adjusted["min_buy_percentage"] != original:
            logger.debug(f"VIX regime: min_buy_percentage adjusted {original} -> {adjusted['min_buy_percentage']}")

    return adjusted


def adjust_sell_criteria(criteria: Dict, apply_adjustments: bool = True) -> Dict:
    """
    Adjust sell criteria based on current VIX regime.

    Args:
        criteria: Original sell criteria dictionary
        apply_adjustments: Whether to apply VIX-based adjustments

    Returns:
        Adjusted criteria dictionary
    """
    if not apply_adjustments:
        return criteria.copy()

    adjustments = get_regime_adjustments()
    adjusted = criteria.copy()

    # Adjust max_upside for sells (add offset to be less aggressive in high VIX)
    if "max_upside" in adjusted:
        original = adjusted["max_upside"]
        adjusted["max_upside"] = round(original + adjustments["max_upside_sell_offset"], 1)
        if adjusted["max_upside"] != original:
            logger.debug(f"VIX regime: sell max_upside adjusted {original} -> {adjusted['max_upside']}")

    return adjusted


def get_regime_status() -> Tuple[VixRegime, float, str]:
    """
    Get current VIX regime status for display.

    Returns:
        Tuple of (regime, vix_level, description)
    """
    vix = get_current_vix()
    regime = get_vix_regime()

    descriptions = {
        VixRegime.LOW: "Low volatility - market complacent",
        VixRegime.NORMAL: "Normal volatility",
        VixRegime.ELEVATED: "Elevated volatility - increased caution",
        VixRegime.HIGH: "High volatility - defensive mode",
    }

    return regime, vix if vix else 0.0, descriptions[regime]


def invalidate_cache() -> None:
    """Force VIX cache invalidation (for testing)."""
    global _vix_cache_timestamp
    with _vix_cache_lock:
        _vix_cache_timestamp = None

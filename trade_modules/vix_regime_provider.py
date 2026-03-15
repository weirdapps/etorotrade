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
#
# CIO Review Finding #7: Comprehensive regime-aware threshold adjustment
# In bull markets (low VIX), forward estimates are optimistic -> tighten BUY criteria
# In bear markets (high VIX), everything looks expensive -> loosen BUY criteria
REGIME_ADJUSTMENTS: Dict[VixRegime, Dict[str, float]] = {
    VixRegime.LOW: {
        # Risk-on: Everything looks cheap on forward estimates, be MORE selective
        "min_upside_multiplier": 1.10,     # Require 10% more upside (avoid buying at consensus peak)
        "min_buy_pct_multiplier": 1.05,    # Slightly higher consensus required
        "min_exret_multiplier": 1.10,      # Require higher expected return
        "min_analysts_offset": 1,          # Require 1 more analyst for conviction
        "max_upside_sell_offset": -1.0,    # Slightly more aggressive on sells
        "max_pct_52w_buy_multiplier": 1.05,  # Tolerate closer to highs
        "max_pe_multiplier": 0.90,         # Tighten PE caps (valuations stretched in low-vol)
    },
    VixRegime.NORMAL: {
        "min_upside_multiplier": 1.0,
        "min_buy_pct_multiplier": 1.0,
        "min_exret_multiplier": 1.0,
        "min_analysts_offset": 0,
        "max_upside_sell_offset": 0.0,
        "max_pct_52w_buy_multiplier": 1.0,
        "max_pe_multiplier": 1.0,
    },
    VixRegime.ELEVATED: {
        # Risk-off lite: Good stocks look expensive, be more forgiving on BUY
        "min_upside_multiplier": 0.90,     # Accept 10% less upside (stocks beaten down)
        "min_buy_pct_multiplier": 0.95,    # Accept slightly lower consensus
        "min_exret_multiplier": 0.90,      # Accept lower expected return
        "min_analysts_offset": 0,
        "max_upside_sell_offset": 2.0,     # Less aggressive on sells (add 2% buffer)
        "max_pct_52w_buy_multiplier": 0.90,  # Accept larger drawdowns from highs
        "max_pe_multiplier": 1.10,         # Loosen PE caps (depressed earnings inflate PE)
    },
    VixRegime.HIGH: {
        # Risk-off: Deep fear, be significantly more forgiving on BUY criteria
        "min_upside_multiplier": 0.80,     # Accept 20% less upside (quality at a discount)
        "min_buy_pct_multiplier": 0.90,    # Accept 10% lower consensus
        "min_exret_multiplier": 0.80,      # Accept lower expected return
        "min_analysts_offset": -1,         # Require 1 fewer analyst (coverage drops in crisis)
        "max_upside_sell_offset": 5.0,     # Much less aggressive on sells
        "max_pct_52w_buy_multiplier": 0.80,  # Accept large drawdowns (everything is down)
        "max_pe_multiplier": 1.20,         # Loosen PE caps significantly
    },
}

# CIO Review Finding M2: Position sizing multipliers per regime
# In high volatility, reduce position sizes to manage risk.
# The signal engine loosens BUY criteria to capture quality at a discount,
# but we must simultaneously reduce position sizes.
REGIME_POSITION_MULTIPLIERS: Dict[VixRegime, float] = {
    VixRegime.LOW: 1.00,       # Normal sizing
    VixRegime.NORMAL: 1.00,    # Normal sizing
    VixRegime.ELEVATED: 0.75,  # Reduce 25%
    VixRegime.HIGH: 0.50,      # Reduce 50%
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

    Applies comprehensive threshold adjustments per CIO Review Finding #7:
    - In low VIX (risk-on): tighten criteria (everything looks cheap, be selective)
    - In high VIX (risk-off): loosen criteria (quality at a discount)

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
        adjusted["min_buy_percentage"] = min(adjusted["min_buy_percentage"], 95.0)
        if adjusted["min_buy_percentage"] != original:
            logger.debug(f"VIX regime: min_buy_percentage adjusted {original} -> {adjusted['min_buy_percentage']}")

    # Adjust min_exret
    if "min_exret" in adjusted:
        original = adjusted["min_exret"]
        adjusted["min_exret"] = round(original * adjustments["min_exret_multiplier"], 1)
        if adjusted["min_exret"] != original:
            logger.debug(f"VIX regime: min_exret adjusted {original} -> {adjusted['min_exret']}")

    # Adjust min_analysts (integer offset)
    if "min_analysts" in adjusted:
        original = adjusted["min_analysts"]
        adjusted["min_analysts"] = max(4, original + adjustments["min_analysts_offset"])
        if adjusted["min_analysts"] != original:
            logger.debug(f"VIX regime: min_analysts adjusted {original} -> {adjusted['min_analysts']}")

    # Adjust min_pct_from_52w_high
    if "min_pct_from_52w_high" in adjusted:
        original = adjusted["min_pct_from_52w_high"]
        adjusted["min_pct_from_52w_high"] = round(
            original * adjustments["max_pct_52w_buy_multiplier"], 1
        )
        if adjusted["min_pct_from_52w_high"] != original:
            logger.debug(f"VIX regime: min_pct_from_52w_high adjusted {original} -> {adjusted['min_pct_from_52w_high']}")

    # Adjust PE caps
    pe_multiplier = adjustments["max_pe_multiplier"]
    for pe_key in ["max_forward_pe", "max_trailing_pe"]:
        if pe_key in adjusted:
            original = adjusted[pe_key]
            adjusted[pe_key] = round(original * pe_multiplier, 1)
            if adjusted[pe_key] != original:
                logger.debug(f"VIX regime: {pe_key} adjusted {original} -> {adjusted[pe_key]}")

    return adjusted


def adjust_sell_criteria(criteria: Dict, apply_adjustments: bool = True) -> Dict:
    """
    Adjust sell criteria based on current VIX regime.

    In high VIX environments, we become less aggressive on sells to avoid
    panic-selling quality stocks during temporary market dislocations.

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

    # Adjust max_exret for sells (in high VIX, require worse exret to trigger sell)
    if "max_exret" in adjusted:
        original = adjusted["max_exret"]
        # In risk-off, increase the exret threshold (harder to trigger sell)
        adjusted["max_exret"] = round(
            original + adjustments["max_upside_sell_offset"] * 0.4, 1
        )
        if adjusted["max_exret"] != original:
            logger.debug(f"VIX regime: sell max_exret adjusted {original} -> {adjusted['max_exret']}")

    return adjusted


def get_adjusted_thresholds(base_config: Dict, config_type: str = "buy") -> Dict:
    """
    Get regime-adjusted thresholds for a tier-region config block.

    This is the primary interface for the signal engine to apply regime-aware
    adjustments. Pass the buy or sell config dict from config.yaml and get
    back adjusted thresholds based on current VIX regime.

    Args:
        base_config: Buy or sell config dict from config.yaml
        config_type: "buy" or "sell"

    Returns:
        Adjusted config dict with regime-appropriate thresholds
    """
    if config_type == "sell":
        return adjust_sell_criteria(base_config)
    return adjust_buy_criteria(base_config)


def get_position_size_multiplier() -> float:
    """
    Get position size multiplier for current VIX regime.

    CIO Review Finding M2: In high volatility environments,
    reduce position sizes even when BUY criteria are loosened.

    Returns:
        Multiplier between 0.5 and 1.0
    """
    regime = get_vix_regime()
    multiplier = REGIME_POSITION_MULTIPLIERS.get(regime, 1.0)
    if multiplier < 1.0:
        logger.info(
            f"VIX regime {regime.value}: position size multiplier = {multiplier:.2f}"
        )
    return multiplier


def get_regime_context() -> Dict:
    """
    Get regime context summary for inclusion in committee reports.

    Returns dict with:
        regime: Current regime name
        vix: Current VIX level
        description: Human-readable description
        adjustments: Active threshold adjustments
        implications: List of regime implications for the committee
    """
    regime, vix, description = get_regime_status()
    adjustments = get_regime_adjustments()

    implications = []
    if regime == VixRegime.LOW:
        implications = [
            "Market complacency - forward estimates likely optimistic",
            "BUY thresholds tightened 10% to avoid buying at consensus peak",
            "PE caps tightened 10% - valuations stretched in low-vol environments",
            "SELL thresholds slightly more aggressive",
        ]
    elif regime == VixRegime.ELEVATED:
        implications = [
            "Elevated fear - quality stocks trading at discounts",
            "BUY thresholds loosened 10% to capture beaten-down quality",
            "PE caps loosened 10% - depressed earnings inflate PE ratios",
            "SELL triggers relaxed with 2% upside buffer",
        ]
    elif regime == VixRegime.HIGH:
        implications = [
            "High fear / crisis mode - significant market dislocation",
            "BUY thresholds loosened 20% to capture deep value",
            "Analyst requirements reduced by 1 (coverage drops in crisis)",
            "SELL triggers significantly relaxed to prevent panic selling",
            "52-week high threshold loosened 20% (everything is down)",
        ]

    return {
        "regime": regime.value,
        "vix": vix,
        "description": description,
        "adjustments": adjustments,
        "implications": implications,
    }


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

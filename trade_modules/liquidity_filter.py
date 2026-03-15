"""
Liquidity Filter Module

Filters stocks by Average Daily Volume (ADV) and estimates transaction costs.
Stocks with insufficient liquidity are flagged as INCONCLUSIVE regardless of
other signal criteria — you can't trade what you can't exit.

CIO Review Finding S5: No transaction cost or liquidity model.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum ADV in USD by tier
TIER_MIN_ADV: Dict[str, float] = {
    "MEGA": 50_000_000,
    "LARGE": 20_000_000,
    "MID": 10_000_000,
    "SMALL": 5_000_000,
    "MICRO": 2_000_000,
}

# Estimated spread costs by tier (basis points)
# Larger stocks have tighter spreads
TIER_SPREAD_BPS: Dict[str, float] = {
    "MEGA": 2.0,
    "LARGE": 5.0,
    "MID": 10.0,
    "SMALL": 20.0,
    "MICRO": 40.0,
}

# eToro daily overnight fee rate (annualized ~6.4% for long positions)
ETORO_OVERNIGHT_ANNUAL_RATE = 0.064

# Cache for ADV data
_adv_cache: Dict[str, Tuple[float, datetime]] = {}
_adv_cache_lock = threading.Lock()
_ADV_CACHE_TTL_HOURS = 4


def _fetch_adv(ticker: str, period_days: int = 30) -> Optional[float]:
    """
    Fetch average daily dollar volume for a ticker.

    Args:
        ticker: Stock ticker symbol
        period_days: Number of days to average over

    Returns:
        Average daily dollar volume, or None if unavailable
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{period_days}d")
        if hist is None or hist.empty or len(hist) < 5:
            return None
        # Dollar volume = close price * volume
        dollar_volume = (hist["Close"] * hist["Volume"]).mean()
        if dollar_volume > 0:
            return float(dollar_volume)
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch ADV for {ticker}: {e}")
        return None


def get_adv(ticker: str) -> Optional[float]:
    """
    Get average daily dollar volume with caching.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Average daily dollar volume in USD, or None
    """
    with _adv_cache_lock:
        if ticker in _adv_cache:
            value, timestamp = _adv_cache[ticker]
            if datetime.now() - timestamp < timedelta(hours=_ADV_CACHE_TTL_HOURS):
                return value

    adv = _fetch_adv(ticker)
    if adv is not None:
        with _adv_cache_lock:
            _adv_cache[ticker] = (adv, datetime.now())

    return adv


def check_liquidity(ticker: str, tier: str) -> Dict[str, Any]:
    """
    Check if a stock meets minimum liquidity requirements for its tier.

    Args:
        ticker: Stock ticker symbol
        tier: Market cap tier (MEGA, LARGE, MID, SMALL, MICRO)

    Returns:
        Dict with:
            passes: bool - whether stock meets liquidity requirements
            adv: float or None - average daily dollar volume
            min_adv: float - minimum required for this tier
            spread_cost_bps: float - estimated spread cost
    """
    tier_upper = tier.upper() if tier else "MID"
    min_adv = TIER_MIN_ADV.get(tier_upper, TIER_MIN_ADV["MID"])
    spread_bps = TIER_SPREAD_BPS.get(tier_upper, TIER_SPREAD_BPS["MID"])

    adv = get_adv(ticker)

    if adv is None:
        # Can't determine liquidity — pass by default (don't block on missing data)
        return {
            "passes": True,
            "adv": None,
            "min_adv": min_adv,
            "spread_cost_bps": spread_bps,
            "reason": "adv_unavailable",
        }

    passes = adv >= min_adv
    reason = "sufficient" if passes else f"adv_{adv/1e6:.1f}M_below_{min_adv/1e6:.0f}M"

    return {
        "passes": passes,
        "adv": adv,
        "min_adv": min_adv,
        "spread_cost_bps": spread_bps,
        "reason": reason,
    }


def estimate_transaction_cost(
    position_size: float,
    tier: str,
    holding_period_days: int = 90,
) -> Dict[str, float]:
    """
    Estimate total transaction cost for a position.

    Includes:
    - Entry spread cost
    - Exit spread cost
    - eToro overnight financing fees for holding period

    Args:
        position_size: Position size in USD
        tier: Market cap tier
        holding_period_days: Expected holding period in calendar days

    Returns:
        Dict with spread_cost, financing_cost, total_cost, total_cost_pct
    """
    tier_upper = tier.upper() if tier else "MID"
    spread_bps = TIER_SPREAD_BPS.get(tier_upper, TIER_SPREAD_BPS["MID"])

    # Entry + exit spread costs
    spread_cost = position_size * (spread_bps / 10000) * 2  # Round trip

    # eToro overnight financing
    financing_cost = position_size * ETORO_OVERNIGHT_ANNUAL_RATE * (holding_period_days / 365)

    total_cost = spread_cost + financing_cost
    total_cost_pct = (total_cost / position_size * 100) if position_size > 0 else 0.0

    return {
        "spread_cost": round(spread_cost, 2),
        "financing_cost": round(financing_cost, 2),
        "total_cost": round(total_cost, 2),
        "total_cost_pct": round(total_cost_pct, 2),
    }


def calculate_cost_adjusted_return(
    expected_return_pct: float,
    position_size: float,
    tier: str,
    holding_period_days: int = 90,
) -> float:
    """
    Calculate expected return after transaction costs.

    A signal is only actionable if expected return exceeds costs.

    Args:
        expected_return_pct: Raw expected return (e.g., EXRET or upside %)
        position_size: Position size in USD
        tier: Market cap tier
        holding_period_days: Expected holding period

    Returns:
        Cost-adjusted expected return percentage
    """
    costs = estimate_transaction_cost(position_size, tier, holding_period_days)
    return round(expected_return_pct - costs["total_cost_pct"], 2)


def filter_by_liquidity(
    df: pd.DataFrame,
    tier_col: str = "tier",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Filter DataFrame to only include stocks meeting liquidity requirements.

    Stocks that fail are not removed but flagged — the signal engine
    will mark them INCONCLUSIVE.

    Args:
        df: DataFrame with stock data
        tier_col: Column containing tier classification
        ticker_col: Column containing ticker symbols

    Returns:
        DataFrame with added columns: adv, liquidity_pass, spread_cost_bps
    """
    if df.empty:
        return df

    result = df.copy()

    # Find ticker column
    tkr_col = None
    for col in [ticker_col, "TKR", "TICKER", "symbol"]:
        if col in result.columns:
            tkr_col = col
            break
    if tkr_col is None and result.index.name in ("TKR", "ticker", "TICKER"):
        result["_ticker"] = result.index
        tkr_col = "_ticker"

    if tkr_col is None:
        logger.warning("No ticker column found for liquidity filter")
        return result

    # Find tier column
    t_col = None
    for col in [tier_col, "TIER", "CAP"]:
        if col in result.columns:
            t_col = col
            break

    # Initialize new columns
    result["adv"] = np.nan
    result["liquidity_pass"] = True
    result["spread_cost_bps"] = np.nan

    for idx in result.index:
        ticker = str(result.at[idx, tkr_col]) if tkr_col in result.columns else str(idx)
        tier = str(result.at[idx, t_col]) if t_col and t_col in result.columns else "MID"

        liquidity = check_liquidity(ticker, tier)
        result.at[idx, "adv"] = liquidity["adv"] if liquidity["adv"] else np.nan
        result.at[idx, "liquidity_pass"] = liquidity["passes"]
        result.at[idx, "spread_cost_bps"] = liquidity["spread_cost_bps"]

    # Clean up temp column
    if "_ticker" in result.columns:
        result = result.drop(columns=["_ticker"])

    failed_count = (~result["liquidity_pass"]).sum()
    if failed_count > 0:
        logger.info(f"Liquidity filter: {failed_count} stocks below ADV threshold")

    return result


def invalidate_cache() -> None:
    """Clear ADV cache (for testing)."""
    global _adv_cache
    with _adv_cache_lock:
        _adv_cache = {}

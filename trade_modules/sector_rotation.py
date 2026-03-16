"""
Sector Rotation Detection Module

Detects sector rotation by tracking changes in sector-level signal
distributions over time using the signal log. When a sector's BUY rate
drops significantly while another's rises, this indicates money rotating
between sectors.

CIO Review Finding M6: No sector rotation detection.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Rotation detection thresholds
ROTATION_THRESHOLD_PP = 15  # Minimum change in BUY% to flag rotation (percentage points)
ROTATION_LOOKBACK_DAYS = 30  # Days to look back for comparison
MIN_SECTOR_STOCKS = 3  # Minimum stocks in sector to be meaningful

# Signal log path (same as backtest engine)
SIGNAL_LOG_PATH = Path.home() / ".weirdapps-trading" / "signals" / "signal_log.jsonl"


def _load_sector_signals(
    signal_log_path: Optional[Path] = None,
    days_back: int = 60,
) -> pd.DataFrame:
    """
    Load signal log entries with sector data.

    Args:
        signal_log_path: Path to signal log file
        days_back: How many days of history to load

    Returns:
        DataFrame with columns: date, ticker, signal, sector
    """
    log_path = signal_log_path or SIGNAL_LOG_PATH
    if not log_path.exists():
        logger.warning(f"Signal log not found: {log_path}")
        return pd.DataFrame()

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    records = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = data.get("timestamp", "")[:10]
            if timestamp < cutoff:
                continue

            sector = data.get("sector")
            signal = data.get("signal")
            ticker = data.get("ticker", "")

            # Skip test tickers and missing data
            if not sector or not signal or signal not in ("B", "S", "H"):
                continue
            if ticker.startswith("TEST") or ticker.startswith("FAKE"):
                continue

            records.append({
                "date": timestamp,
                "ticker": ticker,
                "signal": signal,
                "sector": sector,
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _calculate_sector_buy_rates(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate BUY rate per sector for a date range.

    Args:
        df: Signal log DataFrame
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dict mapping sector -> {buy_rate, count, buy_count}
    """
    period_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if period_df.empty:
        return {}

    # Use latest signal per ticker in the period
    latest = period_df.sort_values("date").groupby("ticker").last().reset_index()

    result = {}
    for sector, group in latest.groupby("sector"):
        count = len(group)
        if count < MIN_SECTOR_STOCKS:
            continue
        buy_count = (group["signal"] == "B").sum()
        sell_count = (group["signal"] == "S").sum()
        result[sector] = {
            "buy_rate": round(buy_count / count * 100, 1),
            "sell_rate": round(sell_count / count * 100, 1),
            "count": count,
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
        }

    return result


def detect_sector_rotation(
    signal_log_path: Optional[Path] = None,
    lookback_days: int = ROTATION_LOOKBACK_DAYS,
    threshold_pp: float = ROTATION_THRESHOLD_PP,
) -> Dict[str, Any]:
    """
    Detect sector rotation by comparing current vs prior signal distributions.

    Args:
        signal_log_path: Path to signal log
        lookback_days: Days between comparison periods
        threshold_pp: Minimum change in BUY% to flag (percentage points)

    Returns:
        Dict with:
            rotation_detected: bool
            rotations: list of rotation descriptions
            gaining_sectors: sectors with rising BUY%
            losing_sectors: sectors with falling BUY%
            current_rates: current sector BUY rates
            prior_rates: prior period sector BUY rates
    """
    df = _load_sector_signals(signal_log_path, days_back=lookback_days * 3)

    if df.empty:
        return {
            "rotation_detected": False,
            "rotations": [],
            "gaining_sectors": [],
            "losing_sectors": [],
            "current_rates": {},
            "prior_rates": {},
        }

    # Define periods
    today = datetime.now().strftime("%Y-%m-%d")
    current_start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    prior_end = current_start
    prior_start = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

    current_rates = _calculate_sector_buy_rates(df, current_start, today)
    prior_rates = _calculate_sector_buy_rates(df, prior_start, prior_end)

    # Find sectors in both periods
    common_sectors = set(current_rates.keys()) & set(prior_rates.keys())

    gaining = []
    losing = []
    rotations = []

    for sector in common_sectors:
        curr_buy = current_rates[sector]["buy_rate"]
        prior_buy = prior_rates[sector]["buy_rate"]
        delta = curr_buy - prior_buy

        if delta >= threshold_pp:
            gaining.append({
                "sector": sector,
                "current_buy_rate": curr_buy,
                "prior_buy_rate": prior_buy,
                "delta_pp": round(delta, 1),
                "stock_count": current_rates[sector]["count"],
            })
        elif delta <= -threshold_pp:
            losing.append({
                "sector": sector,
                "current_buy_rate": curr_buy,
                "prior_buy_rate": prior_buy,
                "delta_pp": round(delta, 1),
                "stock_count": current_rates[sector]["count"],
            })

    # Sort by magnitude
    gaining.sort(key=lambda x: x["delta_pp"], reverse=True)
    losing.sort(key=lambda x: x["delta_pp"])

    # Generate rotation descriptions
    for g in gaining:
        for l in losing:
            rotations.append(
                f"Rotation: {l['sector']} ({l['delta_pp']:+.0f}pp) -> "
                f"{g['sector']} ({g['delta_pp']:+.0f}pp)"
            )

    rotation_detected = len(gaining) > 0 and len(losing) > 0

    return {
        "rotation_detected": rotation_detected,
        "rotations": rotations,
        "gaining_sectors": gaining,
        "losing_sectors": losing,
        "current_rates": current_rates,
        "prior_rates": prior_rates,
    }


def detect_price_based_rotation(
    sector_returns: Dict[str, Dict[str, float]],
    threshold_pp: float = 5.0,
) -> Dict[str, Any]:
    """
    Detect sector rotation using price-based relative strength.

    CIO v3 F5: Price-based rotation leads analyst consensus by 2-4 weeks.
    Compares recent sector ETF/index returns against prior period to detect
    money flows before they show up in analyst recommendation changes.

    Args:
        sector_returns: Dict mapping sector name to:
            - 'current_return': return over recent period (e.g., last 21 trading days), as %
            - 'prior_return': return over prior period (e.g., previous 21 trading days), as %
            Example: {"Technology": {"current_return": 3.2, "prior_return": -1.5}}
        threshold_pp: Minimum change in momentum to flag rotation (percentage points)

    Returns:
        Dict with rotation_detected, gaining_sectors, losing_sectors, rotations
    """
    gaining = []
    losing = []
    rotations = []

    for sector, returns in sector_returns.items():
        current = returns.get("current_return", 0.0)
        prior = returns.get("prior_return", 0.0)
        momentum_change = current - prior

        if momentum_change >= threshold_pp:
            gaining.append({
                "sector": sector,
                "current_return": round(current, 1),
                "prior_return": round(prior, 1),
                "momentum_change_pp": round(momentum_change, 1),
            })
        elif momentum_change <= -threshold_pp:
            losing.append({
                "sector": sector,
                "current_return": round(current, 1),
                "prior_return": round(prior, 1),
                "momentum_change_pp": round(momentum_change, 1),
            })

    gaining.sort(key=lambda x: x["momentum_change_pp"], reverse=True)
    losing.sort(key=lambda x: x["momentum_change_pp"])

    for g in gaining:
        for l in losing:
            rotations.append(
                f"Price rotation: {l['sector']} ({l['momentum_change_pp']:+.1f}pp) -> "
                f"{g['sector']} ({g['momentum_change_pp']:+.1f}pp)"
            )

    rotation_detected = len(gaining) > 0 and len(losing) > 0

    return {
        "rotation_detected": rotation_detected,
        "rotations": rotations,
        "gaining_sectors": gaining,
        "losing_sectors": losing,
        "detection_method": "price_based",
    }


def detect_combined_rotation(
    signal_log_path: Optional[Path] = None,
    sector_returns: Optional[Dict[str, Dict[str, float]]] = None,
    lookback_days: int = ROTATION_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """
    Combine signal-based and price-based rotation detection.

    CIO v3 F5: When both methods agree, conviction is highest.
    Price-based leads by 2-4 weeks; signal-based confirms the rotation.

    Args:
        signal_log_path: Path to signal log
        sector_returns: Sector ETF returns (for price-based detection)
        lookback_days: Days for signal-based lookback

    Returns:
        Dict with combined rotation assessment and confidence level
    """
    signal_result = detect_sector_rotation(signal_log_path, lookback_days)

    if sector_returns:
        price_result = detect_price_based_rotation(sector_returns)
    else:
        price_result = {
            "rotation_detected": False,
            "gaining_sectors": [],
            "losing_sectors": [],
        }

    # Determine sectors gaining/losing in each method
    signal_gaining = {s["sector"] for s in signal_result.get("gaining_sectors", [])}
    signal_losing = {s["sector"] for s in signal_result.get("losing_sectors", [])}
    price_gaining = {s["sector"] for s in price_result.get("gaining_sectors", [])}
    price_losing = {s["sector"] for s in price_result.get("losing_sectors", [])}

    # Sectors confirmed by both methods get highest confidence
    confirmed_gaining = signal_gaining & price_gaining
    confirmed_losing = signal_losing & price_losing

    # Price-only signals are early warnings
    early_gaining = price_gaining - signal_gaining
    early_losing = price_losing - signal_losing

    has_price_signals = bool(price_gaining or price_losing)

    if confirmed_gaining or confirmed_losing:
        confidence = "HIGH"
    elif has_price_signals:
        confidence = "MEDIUM"
    elif signal_result["rotation_detected"]:
        confidence = "LOW"
    else:
        confidence = "NONE"

    return {
        "rotation_detected": signal_result["rotation_detected"] or price_result["rotation_detected"],
        "confidence": confidence,
        "confirmed_gaining": sorted(confirmed_gaining),
        "confirmed_losing": sorted(confirmed_losing),
        "early_warning_gaining": sorted(early_gaining),
        "early_warning_losing": sorted(early_losing),
        "signal_based": signal_result,
        "price_based": price_result,
    }


def get_rotation_context() -> Dict[str, Any]:
    """
    Get sector rotation context for inclusion in committee reports and briefings.

    Returns:
        Dict with rotation_detected, summary, details
    """
    result = detect_sector_rotation()

    if not result["rotation_detected"]:
        return {
            "rotation_detected": False,
            "summary": "No significant sector rotation detected in the past 30 days.",
            "details": result,
        }

    gaining_names = [s["sector"] for s in result["gaining_sectors"]]
    losing_names = [s["sector"] for s in result["losing_sectors"]]

    summary = (
        f"Sector rotation detected: money moving FROM {', '.join(losing_names)} "
        f"TO {', '.join(gaining_names)}. "
        f"Adjust conviction scores accordingly — favor gaining sectors, "
        f"be cautious on losing sectors."
    )

    return {
        "rotation_detected": True,
        "summary": summary,
        "gaining": gaining_names,
        "losing": losing_names,
        "details": result,
    }

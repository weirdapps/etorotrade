"""
Committee Performance Scorecard

CIO Review v3 Finding F13: Track committee recommendation performance
to enable feedback loops and system calibration.

Logs committee action items with timestamps, then evaluates hit rates
at T+7, T+30 against actual price movements.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

COMMITTEE_LOG_DIR = Path.home() / ".weirdapps-trading" / "committee"
COMMITTEE_LOG_PATH = COMMITTEE_LOG_DIR / "action_log.jsonl"
SCORECARD_OUTPUT_PATH = COMMITTEE_LOG_DIR / "committee_scorecard.json"


def log_committee_actions(
    date: str,
    actions: List[Dict[str, Any]],
    log_path: Optional[Path] = None,
) -> None:
    """
    Log committee action items for future performance tracking.

    Args:
        date: Committee date (YYYY-MM-DD)
        actions: List of action dicts with ticker, action, conviction, etc.
        log_path: Path to action log file
    """
    path = log_path or COMMITTEE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    for action in actions:
        entry = {
            "committee_date": date,
            "timestamp": datetime.now().isoformat(),
            "ticker": action.get("ticker"),
            "action": action.get("action"),  # BUY, SELL, HOLD, REDUCE, ADD
            "conviction": action.get("conviction"),
            "size": action.get("size"),
            "agents_agreeing": action.get("agents_agreeing"),
            "dissenting_agents": action.get("dissenting_agents"),
            "signal": action.get("signal"),  # From signal engine
            "price_at_recommendation": action.get("price"),
        }

        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Logged {len(actions)} committee actions for {date}")


def load_committee_actions(
    months_back: int = 3,
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load committee actions from the log file."""
    path = log_path or COMMITTEE_LOG_PATH
    if not path.exists():
        return []

    cutoff = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
    actions = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("committee_date", "") >= cutoff:
                    actions.append(entry)
            except json.JSONDecodeError:
                continue

    return actions


def generate_committee_scorecard(
    months_back: int = 3,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate performance scorecard for committee recommendations.

    Evaluates BUY/SELL recommendations against actual price movements
    at T+7 and T+30 horizons.

    Args:
        months_back: How many months of history to evaluate
        log_path: Path to action log

    Returns:
        Dict with hit rates, agent accuracy, and calibration insights
    """
    actions = load_committee_actions(months_back, log_path)

    if not actions:
        return _empty_scorecard(months_back)

    # Fetch current prices for performance calculation
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for committee scorecard")
        return _empty_scorecard(months_back)

    buy_results = []
    sell_results = []

    for action in actions:
        ticker = action.get("ticker")
        action_type = action.get("action", "").upper()
        entry_price = action.get("price_at_recommendation")
        committee_date = action.get("committee_date")

        if not ticker or not entry_price or not committee_date:
            continue

        try:
            entry_price = float(entry_price)
        except (ValueError, TypeError):
            continue

        # Fetch price history since recommendation
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=committee_date, period="3mo")
            if hist.empty:
                continue

            close = hist["Close"]

            # Calculate returns at T+7 and T+30
            result = {
                "ticker": ticker,
                "action": action_type,
                "conviction": action.get("conviction"),
                "committee_date": committee_date,
                "entry_price": entry_price,
            }

            if len(close) >= 5:
                t7_price = float(close.iloc[min(5, len(close) - 1)])
                result["return_7d"] = round(
                    (t7_price - entry_price) / entry_price * 100, 2
                )

            if len(close) >= 21:
                t30_price = float(close.iloc[min(21, len(close) - 1)])
                result["return_30d"] = round(
                    (t30_price - entry_price) / entry_price * 100, 2
                )

            if action_type in ("BUY", "ADD"):
                buy_results.append(result)
            elif action_type in ("SELL", "REDUCE"):
                sell_results.append(result)

        except Exception as e:
            logger.debug(f"Failed to fetch data for {ticker}: {e}")
            continue

    # Calculate aggregate statistics
    scorecard = {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "period_months": months_back,
        "buy_recommendations": _summarize_buys(buy_results),
        "sell_recommendations": _summarize_sells(sell_results),
        "conviction_calibration": _calibrate_conviction(buy_results),
        "details": {
            "buys": buy_results,
            "sells": sell_results,
        },
    }

    # Save
    _save_scorecard(scorecard)

    return scorecard


def _summarize_buys(results: List[Dict]) -> Dict[str, Any]:
    """Summarize BUY recommendation performance."""
    if not results:
        return {"total": 0}

    returns_7d = [r["return_7d"] for r in results if "return_7d" in r]
    returns_30d = [r["return_30d"] for r in results if "return_30d" in r]

    summary: Dict[str, Any] = {"total": len(results)}

    if returns_7d:
        summary["hit_rate_7d"] = round(
            sum(1 for r in returns_7d if r > 0) / len(returns_7d) * 100, 1
        )
        summary["avg_return_7d"] = round(np.mean(returns_7d), 2)

    if returns_30d:
        summary["hit_rate_30d"] = round(
            sum(1 for r in returns_30d if r > 0) / len(returns_30d) * 100, 1
        )
        summary["avg_return_30d"] = round(np.mean(returns_30d), 2)

        # Best and worst
        best = max(results, key=lambda r: r.get("return_30d", -999))
        worst = min(results, key=lambda r: r.get("return_30d", 999))
        summary["best_30d"] = f"{best['ticker']} {best.get('return_30d', 0):+.1f}%"
        summary["worst_30d"] = f"{worst['ticker']} {worst.get('return_30d', 0):+.1f}%"

    return summary


def _summarize_sells(results: List[Dict]) -> Dict[str, Any]:
    """Summarize SELL recommendation performance (validated if price fell)."""
    if not results:
        return {"total": 0}

    returns_30d = [r["return_30d"] for r in results if "return_30d" in r]

    summary: Dict[str, Any] = {"total": len(results)}

    if returns_30d:
        # For SELLs, a validated recommendation means the stock went down
        summary["validated_30d"] = round(
            sum(1 for r in returns_30d if r < 0) / len(returns_30d) * 100, 1
        )
        summary["avg_avoided_loss"] = round(np.mean(returns_30d), 2)

    return summary


def _calibrate_conviction(results: List[Dict]) -> Dict[str, Any]:
    """
    Check if conviction scores predict return magnitude.

    High conviction should correlate with higher returns.
    """
    if not results or len(results) < 5:
        return {"sufficient_data": False}

    scored = [
        r for r in results
        if r.get("conviction") is not None and "return_30d" in r
    ]

    if len(scored) < 5:
        return {"sufficient_data": False}

    # Split into high/low conviction
    high_conv = [r for r in scored if r["conviction"] >= 65]
    low_conv = [r for r in scored if r["conviction"] < 65]

    calibration: Dict[str, Any] = {"sufficient_data": True}

    if high_conv:
        calibration["high_conviction_avg_return"] = round(
            np.mean([r["return_30d"] for r in high_conv]), 2
        )
        calibration["high_conviction_count"] = len(high_conv)

    if low_conv:
        calibration["low_conviction_avg_return"] = round(
            np.mean([r["return_30d"] for r in low_conv]), 2
        )
        calibration["low_conviction_count"] = len(low_conv)

    # Check if high conviction outperforms low conviction
    if high_conv and low_conv:
        high_avg = np.mean([r["return_30d"] for r in high_conv])
        low_avg = np.mean([r["return_30d"] for r in low_conv])
        calibration["conviction_predictive"] = high_avg > low_avg
        calibration["conviction_spread"] = round(high_avg - low_avg, 2)

    return calibration


def _save_scorecard(scorecard: Dict[str, Any]) -> None:
    """Save scorecard to JSON file."""
    COMMITTEE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORECARD_OUTPUT_PATH, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    logger.info(f"Committee scorecard saved to {SCORECARD_OUTPUT_PATH}")


def _empty_scorecard(months_back: int) -> Dict[str, Any]:
    """Return empty scorecard structure."""
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "period_months": months_back,
        "buy_recommendations": {"total": 0},
        "sell_recommendations": {"total": 0},
        "conviction_calibration": {"sufficient_data": False},
        "details": {"buys": [], "sells": []},
    }

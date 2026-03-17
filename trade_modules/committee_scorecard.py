"""
Committee Performance Scorecard

CIO Review v3 Finding F13: Track committee recommendation performance
to enable feedback loops and system calibration.

CIO Review v4 Finding F5: Persistent opportunity tracking across
committee runs to reward conviction consistency.

CIO Review v4 Finding F10: Automated performance check of previous
committee recommendations with current prices.

CIO Review v4 Finding F11: Kill thesis monitoring — structured storage
and automated heuristic checking of kill theses for BUY/ADD positions.

Logs committee action items with timestamps, then evaluates hit rates
at T+7, T+30 against actual price movements.
"""

import csv
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
OPPORTUNITY_HISTORY_PATH = COMMITTEE_LOG_DIR / "opportunity_history.json"
KILL_THESIS_LOG_PATH = COMMITTEE_LOG_DIR / "kill_thesis_log.json"

# F5: Maximum conviction bonus from consecutive appearances
MAX_CONSECUTIVE_BONUS = 9
BONUS_PER_APPEARANCE = 3

# F11: Default portfolio signals path
DEFAULT_PORTFOLIO_SIGNALS_PATH = (
    Path.home() / "SourceCode" / "etorotrade"
    / "yahoofinance" / "output" / "portfolio.csv"
)

# F11: Kill thesis expiry default (days)
KILL_THESIS_DEFAULT_EXPIRY_DAYS = 90


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


def track_opportunities(
    opportunity_tickers: List[str],
    committee_date: str,
    history_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Track opportunity persistence across consecutive committee runs.

    CIO Review v4 Finding F5: Opportunities that appear in multiple
    consecutive committee runs get a conviction bonus, rewarding
    consistency and persistence of the signal.

    Args:
        opportunity_tickers: List of ticker symbols flagged as opportunities
        committee_date: Committee date (YYYY-MM-DD)
        history_path: Path to opportunity history JSON file

    Returns:
        Dict mapping ticker -> {
            "consecutive_appearances": int,
            "first_seen": str (YYYY-MM-DD),
            "conviction_bonus": int (0 to MAX_CONSECUTIVE_BONUS)
        }
    """
    path = history_path or OPPORTUNITY_HISTORY_PATH
    history = _load_opportunity_history(path)

    previous_date = history.get("last_committee_date")
    previous_tickers = set(history.get("active_tickers", []))
    ticker_records = history.get("ticker_records", {})

    result: Dict[str, Dict[str, Any]] = {}

    for ticker in opportunity_tickers:
        record = ticker_records.get(ticker, {})

        if ticker in previous_tickers and previous_date is not None:
            # Ticker appeared in previous run -- increment streak
            consecutive = record.get("consecutive_appearances", 1) + 1
            first_seen = record.get("first_seen", committee_date)
        else:
            # New ticker or gap in appearances -- reset streak
            consecutive = 1
            first_seen = record.get("first_seen", committee_date)

        bonus = min(
            (consecutive - 1) * BONUS_PER_APPEARANCE,
            MAX_CONSECUTIVE_BONUS,
        )

        result[ticker] = {
            "consecutive_appearances": consecutive,
            "first_seen": first_seen,
            "conviction_bonus": bonus,
        }

        # Update the stored record
        ticker_records[ticker] = {
            "consecutive_appearances": consecutive,
            "first_seen": first_seen,
            "last_seen": committee_date,
        }

    # Update history state for next run
    history["last_committee_date"] = committee_date
    history["active_tickers"] = list(opportunity_tickers)
    history["ticker_records"] = ticker_records

    # Persist
    save_opportunity_history(history, path)

    logger.info(
        f"Tracked {len(opportunity_tickers)} opportunities for {committee_date}: "
        f"{sum(1 for v in result.values() if v['conviction_bonus'] > 0)} with bonus"
    )

    return result


def save_opportunity_history(
    history: Dict[str, Any],
    history_path: Optional[Path] = None,
) -> None:
    """
    Save opportunity history to disk.

    Args:
        history: The full history dict to persist
        history_path: Path to opportunity history JSON file
    """
    path = history_path or OPPORTUNITY_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        logger.debug(f"Opportunity history saved to {path}")
    except OSError as e:
        logger.error(f"Failed to save opportunity history: {e}")


def _load_opportunity_history(path: Path) -> Dict[str, Any]:
    """Load opportunity history from disk, returning empty structure on failure."""
    if not path.exists():
        return {
            "last_committee_date": None,
            "active_tickers": [],
            "ticker_records": {},
        }

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load opportunity history: {e}")
        return {
            "last_committee_date": None,
            "active_tickers": [],
            "ticker_records": {},
        }


def check_previous_recommendations(
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Check performance of most recent committee BUY/ADD recommendations.

    CIO Review v4 Finding F10: Automated performance check that fetches
    current prices for the most recent committee run's BUY/ADD actions
    and computes returns since recommendation.

    Args:
        log_path: Path to action log JSONL file

    Returns:
        Dict with:
            - total_recommendations: int
            - active_buys: int (with valid price data)
            - avg_return_to_date: float (percentage)
            - best_performer: {"ticker": str, "return_pct": float}
            - worst_performer: {"ticker": str, "return_pct": float}
            - triggered_kill_theses: list (placeholder)
            - recommendations: list of individual results
    """
    path = log_path or COMMITTEE_LOG_PATH
    if not path.exists():
        return _empty_performance_check()

    # Load all actions and find the most recent committee date
    all_actions = _load_all_actions(path)
    if not all_actions:
        return _empty_performance_check()

    # Get the most recent committee date
    dates = sorted(set(a.get("committee_date", "") for a in all_actions))
    if not dates:
        return _empty_performance_check()

    latest_date = dates[-1]

    # Filter to BUY/ADD from the most recent run
    buy_actions = [
        a for a in all_actions
        if a.get("committee_date") == latest_date
        and a.get("action", "").upper() in ("BUY", "ADD")
    ]

    if not buy_actions:
        return _empty_performance_check()

    # Fetch current prices
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for performance check")
        return _empty_performance_check()

    results = []
    for action in buy_actions:
        ticker = action.get("ticker")
        entry_price = action.get("price_at_recommendation")

        if not ticker or not entry_price:
            continue

        try:
            entry_price = float(entry_price)
        except (ValueError, TypeError):
            continue

        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            current_price = float(info.get("lastPrice", 0) or info.get("last_price", 0))

            if current_price <= 0:
                # Fallback: try history
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])

            if current_price <= 0:
                continue

            return_pct = round(
                (current_price - entry_price) / entry_price * 100, 2
            )

            results.append({
                "ticker": ticker,
                "entry_price": entry_price,
                "current_price": current_price,
                "return_pct": return_pct,
                "conviction": action.get("conviction"),
                "committee_date": latest_date,
            })

        except Exception as e:
            logger.debug(f"Failed to fetch price for {ticker}: {e}")
            continue

    if not results:
        return _empty_performance_check()

    returns = [r["return_pct"] for r in results]
    best = max(results, key=lambda r: r["return_pct"])
    worst = min(results, key=lambda r: r["return_pct"])

    return {
        "committee_date": latest_date,
        "total_recommendations": len(buy_actions),
        "active_buys": len(results),
        "avg_return_to_date": round(float(np.mean(returns)), 2),
        "best_performer": {
            "ticker": best["ticker"],
            "return_pct": best["return_pct"],
        },
        "worst_performer": {
            "ticker": worst["ticker"],
            "return_pct": worst["return_pct"],
        },
        "triggered_kill_theses": [],  # Placeholder for future implementation
        "recommendations": results,
    }


def get_track_record_summary(
    log_path: Optional[Path] = None,
) -> str:
    """
    Return a one-line track record summary for display in committee reports.

    CIO Review v4 Finding F10: Provides a concise summary string like:
    "Track record: XX% BUY hit rate at T+30 (n=YY)"

    Uses the committee scorecard if available, otherwise returns
    a placeholder indicating insufficient data.

    Args:
        log_path: Path to committee scorecard JSON file

    Returns:
        One-line summary string
    """
    scorecard_path = SCORECARD_OUTPUT_PATH

    if not scorecard_path.exists():
        return "Track record: insufficient data (no scorecard generated yet)"

    try:
        with open(scorecard_path, "r") as f:
            scorecard = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "Track record: insufficient data (scorecard unreadable)"

    buy_recs = scorecard.get("buy_recommendations", {})
    total = buy_recs.get("total", 0)

    if total == 0:
        return "Track record: insufficient data (n=0)"

    hit_rate = buy_recs.get("hit_rate_30d")
    if hit_rate is not None:
        return f"Track record: {hit_rate:.0f}% BUY hit rate at T+30 (n={total})"

    hit_rate_7d = buy_recs.get("hit_rate_7d")
    if hit_rate_7d is not None:
        return f"Track record: {hit_rate_7d:.0f}% BUY hit rate at T+7 (n={total})"

    return f"Track record: {total} BUY recommendations logged, awaiting T+30 data"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, stripping % signs and handling edge cases."""
    if value is None:
        return default
    try:
        s = str(value).strip().rstrip("%")
        if s in ("", "--", "N/A", "nan", "NaN"):
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


# ============================================================
# F11: Kill Thesis Monitoring
# ============================================================


def log_kill_theses(
    date: str,
    theses: List[Dict[str, Any]],
    log_path: Optional[Path] = None,
) -> None:
    """
    Store kill theses as structured JSON.

    CIO Review v4 Finding F11: Persist kill theses from committee
    BUY/ADD recommendations for automated monitoring.

    Args:
        date: Committee date (YYYY-MM-DD)
        theses: List of thesis dicts, each with at least:
            - ticker: str
            - kill_thesis: str
            Optional:
            - status: str (default "active")
            - expiry_date: str or None
        log_path: Path to kill thesis log JSON file
    """
    path = log_path or KILL_THESIS_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing theses
    existing = _load_kill_theses(path)

    # Build a set of (ticker, committee_date) for dedup
    existing_keys = {
        (t["ticker"], t["committee_date"])
        for t in existing
    }

    new_count = 0
    for thesis in theses:
        ticker = thesis.get("ticker")
        if not ticker:
            continue

        key = (ticker, date)
        if key in existing_keys:
            continue  # Skip duplicate

        entry = {
            "ticker": ticker,
            "kill_thesis": thesis.get("kill_thesis", ""),
            "committee_date": date,
            "status": thesis.get("status", "active"),
            "expiry_date": thesis.get("expiry_date"),
            # CIO Legacy D2: Custom machine-checkable conditions
            "conditions": thesis.get("conditions", []),
        }

        existing.append(entry)
        existing_keys.add(key)
        new_count += 1

    # Save back
    _save_kill_theses(existing, path)
    logger.info(
        f"Logged {new_count} kill theses for {date} "
        f"({len(existing)} total)"
    )


def check_kill_theses(
    portfolio_signals_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Check if any active kill theses have triggered based on signal data.

    CIO Review v4 Finding F11: Heuristic kill thesis monitoring using
    available signal data to flag deteriorating positions.

    Heuristic triggers:
        - signal_deteriorated: ticker signal changed to S (SELL)
        - price_collapsed: 52-week performance dropped below 40
        - analyst_downgrade: analyst momentum (AM) strongly negative (< -5)

    Args:
        portfolio_signals_path: Path to portfolio.csv with current signals
        log_path: Path to kill thesis log JSON file

    Returns:
        Dict with:
            - active_theses: list of still-active theses
            - triggered_theses: list of theses where heuristic triggers fired
            - expired_theses: list of theses past 90-day expiry
    """
    path = log_path or KILL_THESIS_LOG_PATH
    signals_path = portfolio_signals_path or DEFAULT_PORTFOLIO_SIGNALS_PATH

    all_theses = _load_kill_theses(path)

    if not all_theses:
        return {
            "active_theses": [],
            "triggered_theses": [],
            "expired_theses": [],
        }

    # Load portfolio signals
    signals = _load_portfolio_signals(signals_path)

    now = datetime.now()
    cutoff = now - timedelta(days=KILL_THESIS_DEFAULT_EXPIRY_DAYS)

    active = []
    triggered = []
    expired = []

    for thesis in all_theses:
        status = thesis.get("status", "active")

        # Already resolved
        if status in ("triggered", "expired"):
            if status == "expired":
                expired.append(thesis)
            else:
                triggered.append(thesis)
            continue

        # Check expiry
        committee_date_str = thesis.get("committee_date", "")
        try:
            committee_dt = datetime.strptime(committee_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            active.append(thesis)
            continue

        # Check explicit expiry_date first
        expiry_str = thesis.get("expiry_date")
        if expiry_str:
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
                if now > expiry_dt:
                    thesis["status"] = "expired"
                    expired.append(thesis)
                    continue
            except (ValueError, TypeError):
                pass

        # Check 90-day default expiry
        if committee_dt < cutoff:
            thesis["status"] = "expired"
            expired.append(thesis)
            continue

        # Check heuristic triggers against signal data
        ticker = thesis.get("ticker", "")
        signal_data = signals.get(ticker)

        triggers = []
        if signal_data:
            # Signal deterioration: ticker now has SELL signal
            if signal_data.get("BS") == "S":
                triggers.append("signal_deteriorated")

            # Price collapse: 52-week performance below 40
            perf_52w = _safe_float(signal_data.get("52W"), default=100.0)
            if perf_52w < 40:
                triggers.append("price_collapsed")

            # Analyst downgrade: AM strongly negative
            am = _safe_float(signal_data.get("AM"), default=0.0)
            if am < -5:
                triggers.append("analyst_downgrade")

            # CIO Legacy D2: Evaluate custom conditions
            for cond in thesis.get("conditions", []):
                metric = cond.get("metric", "")
                operator = cond.get("operator", "")
                threshold = cond.get("threshold")
                if not metric or not operator or threshold is None:
                    continue
                actual = _safe_float(signal_data.get(metric), default=None)
                if actual is None:
                    continue
                cond_met = False
                if operator == "lt" and actual < threshold:
                    cond_met = True
                elif operator == "gt" and actual > threshold:
                    cond_met = True
                elif operator == "eq" and actual == threshold:
                    cond_met = True
                elif operator == "le" and actual <= threshold:
                    cond_met = True
                elif operator == "ge" and actual >= threshold:
                    cond_met = True
                if cond_met:
                    triggers.append(f"custom:{metric} {operator} {threshold}")

        if triggers:
            thesis["status"] = "triggered"
            thesis["triggers"] = triggers
            triggered.append(thesis)
        else:
            active.append(thesis)

    # Persist updated statuses
    _save_kill_theses(all_theses, path)

    logger.info(
        f"Kill thesis check: {len(active)} active, "
        f"{len(triggered)} triggered, {len(expired)} expired"
    )

    return {
        "active_theses": active,
        "triggered_theses": triggered,
        "expired_theses": expired,
    }


def expire_old_theses(
    days: int = 90,
    log_path: Optional[Path] = None,
) -> int:
    """
    Mark theses older than ``days`` as expired.

    CIO Review v4 Finding F11: Housekeeping to prevent stale kill
    theses from accumulating indefinitely.

    Args:
        days: Age threshold in days (default 90)
        log_path: Path to kill thesis log JSON file

    Returns:
        Number of theses that were expired
    """
    path = log_path or KILL_THESIS_LOG_PATH
    all_theses = _load_kill_theses(path)

    if not all_theses:
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    expired_count = 0

    for thesis in all_theses:
        if thesis.get("status") != "active":
            continue

        committee_date_str = thesis.get("committee_date", "")
        try:
            committee_dt = datetime.strptime(committee_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        if committee_dt < cutoff:
            thesis["status"] = "expired"
            expired_count += 1

    if expired_count > 0:
        _save_kill_theses(all_theses, path)
        logger.info(f"Expired {expired_count} kill theses older than {days} days")

    return expired_count


def _load_kill_theses(path: Path) -> List[Dict[str, Any]]:
    """Load kill theses from JSON file."""
    if not path.exists():
        return []

    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load kill theses: {e}")
        return []


def _save_kill_theses(
    theses: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save kill theses to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(theses, f, indent=2, default=str)
    except OSError as e:
        logger.error(f"Failed to save kill theses: {e}")


def _load_portfolio_signals(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load portfolio signals from CSV into a dict keyed by ticker.

    The CSV has headers:
    TKR,NAME,CAP,PRC,TGT,UP%,#T,%B,#A,AM,A,EXR,B,52W,2H,...,BS

    Returns:
        Dict mapping ticker -> dict of column values
    """
    if not path.exists():
        return {}

    try:
        signals: Dict[str, Dict[str, str]] = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("TKR", "").strip()
                if ticker:
                    signals[ticker] = dict(row)
        return signals
    except (OSError, csv.Error) as e:
        logger.warning(f"Failed to load portfolio signals: {e}")
        return {}


def _load_all_actions(path: Path) -> List[Dict[str, Any]]:
    """Load all committee actions from log without date filtering."""
    if not path.exists():
        return []

    actions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return actions


def _empty_performance_check() -> Dict[str, Any]:
    """Return empty performance check structure."""
    return {
        "committee_date": None,
        "total_recommendations": 0,
        "active_buys": 0,
        "avg_return_to_date": 0.0,
        "best_performer": None,
        "worst_performer": None,
        "triggered_kill_theses": [],
        "recommendations": [],
    }


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

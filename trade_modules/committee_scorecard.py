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

# Primary: repo-local path (works in CI and local dev)
_REPO_ROOT = Path(__file__).parent.parent
COMMITTEE_LOG_DIR = _REPO_ROOT / "data" / "committee"
COMMITTEE_LOG_PATH = COMMITTEE_LOG_DIR / "action_log.jsonl"

# Derived outputs stay in ~/.weirdapps-trading (not committed)
_USER_COMMITTEE_DIR = Path.home() / ".weirdapps-trading" / "committee"
SCORECARD_OUTPUT_PATH = _USER_COMMITTEE_DIR / "committee_scorecard.json"
OPPORTUNITY_HISTORY_PATH = _USER_COMMITTEE_DIR / "opportunity_history.json"
KILL_THESIS_LOG_PATH = _USER_COMMITTEE_DIR / "kill_thesis_log.json"

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
    portfolio_signals: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log committee action items for future performance tracking.

    Args:
        date: Committee date (YYYY-MM-DD)
        actions: List of action dicts with ticker, action, conviction, etc.
        log_path: Path to action log file
        portfolio_signals: Dict of ticker -> signal data for price fallback
    """
    path = log_path or COMMITTEE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    for action in actions:
        # Price with fallback chain: action price → portfolio signals → warning
        price = action.get("price")
        if price is None or (isinstance(price, (int, float)) and price <= 0):
            ticker = action.get("ticker", "")
            if portfolio_signals and ticker in portfolio_signals:
                sig = portfolio_signals[ticker]
                price = sig.get("price") or sig.get("PRC")
                if price is not None:
                    try:
                        price = float(str(price).replace(",", ""))
                    except (ValueError, TypeError):
                        price = None
            if price is None or (isinstance(price, (int, float)) and price <= 0):
                logger.warning(
                    f"No price available for {action.get('ticker')} on {date} — "
                    f"scorecard will skip this entry until backfilled"
                )

        entry = {
            "committee_date": date,
            "timestamp": datetime.now().isoformat(),
            "ticker": action.get("ticker"),
            "action": action.get("action"),  # BUY, ADD, HOLD, TRIM, SELL
            "conviction": action.get("conviction"),
            "size": action.get("size"),
            "agents_agreeing": action.get("agents_agreeing"),
            "dissenting_agents": action.get("dissenting_agents"),
            "signal": action.get("signal"),  # From signal engine
            "price_at_recommendation": price,
            # CIO v22.0: Log new indicators for future calibration
            "adx": action.get("adx"),
            "adx_trend": action.get("adx_trend"),
            "divergence": action.get("divergence"),
            "piotroski_score": action.get("piotroski_score"),
            "entry_timing": action.get("entry_timing"),
            # CIO v21.0 R1-R5: Codified conviction modifiers
            "rs_vs_spy": action.get("rs_vs_spy"),
            "revenue_growth_class": action.get("revenue_growth_class"),
            "atr_pct": action.get("atr_pct"),
            # CIO v23.0 S1-S2: Quick-win modifiers
            "short_interest_pct": action.get("short_interest_pct"),
            "target_dispersion": action.get("target_dispersion"),
            # CIO v23.1 R6: Currency risk
            "currency_zone": action.get("currency_zone"),
            "fx_impact": action.get("fx_impact"),
            # CIO v23.3: Signal quality modifiers
            "confluence_score": action.get("confluence_score"),
            "obv_trend": action.get("obv_trend"),
            "relative_volume": action.get("relative_volume"),
            "eps_revisions": action.get("eps_revisions"),
            # CIO v23.4: Advanced fundamentals & timing
            "iv_rank": action.get("iv_rank"),
            "volatility_regime": action.get("volatility_regime"),
            "fcf_classification": action.get("fcf_classification"),
            "debt_risk": action.get("debt_risk"),
            # CIO v25.0: Conviction waterfall for calibration
            "conviction_waterfall": action.get("conviction_waterfall"),
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
    hold_results = []

    # Fetch SPY benchmark once for alpha computation
    spy_close = None
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="6mo")
        if not spy_hist.empty:
            spy_close = spy_hist["Close"]
    except Exception as e:
        logger.debug(f"Failed to fetch SPY benchmark: {e}")

    for action in actions:
        ticker = action.get("ticker")
        action_type = action.get("action", "").upper()
        entry_price = action.get("price_at_recommendation")
        committee_date = action.get("committee_date")

        if not ticker or entry_price is None or not committee_date:
            continue

        try:
            entry_price = float(entry_price)
            if entry_price <= 0:
                continue
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

            # Get SPY returns for the same horizons
            spy_return_7d = None
            spy_return_30d = None
            if spy_close is not None:
                spy_from_date = spy_close[spy_close.index >= committee_date]
                if len(spy_from_date) >= 1:
                    spy_base = float(spy_from_date.iloc[0])
                    if spy_base > 0:
                        if len(spy_from_date) >= 5:
                            spy_t7 = float(spy_from_date.iloc[min(5, len(spy_from_date) - 1)])
                            spy_return_7d = (spy_t7 - spy_base) / spy_base * 100
                        if len(spy_from_date) >= 21:
                            spy_t30 = float(spy_from_date.iloc[min(21, len(spy_from_date) - 1)])
                            spy_return_30d = (spy_t30 - spy_base) / spy_base * 100

            if len(close) >= 5:
                t7_price = float(close.iloc[min(5, len(close) - 1)])
                result["return_7d"] = round(
                    (t7_price - entry_price) / entry_price * 100, 2
                )
                if spy_return_7d is not None:
                    result["alpha_7d"] = round(result["return_7d"] - spy_return_7d, 2)

            if len(close) >= 21:
                t30_price = float(close.iloc[min(21, len(close) - 1)])
                result["return_30d"] = round(
                    (t30_price - entry_price) / entry_price * 100, 2
                )
                if spy_return_30d is not None:
                    result["alpha_30d"] = round(result["return_30d"] - spy_return_30d, 2)

            if action_type in ("BUY", "ADD"):
                buy_results.append(result)
            elif action_type in ("SELL", "TRIM"):
                sell_results.append(result)
            elif action_type == "HOLD":
                hold_results.append(result)

        except Exception as e:
            logger.debug(f"Failed to fetch data for {ticker}: {e}")
            continue

    # Calculate aggregate statistics
    scorecard = {
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "period_months": months_back,
        "buy_recommendations": _summarize_buys(buy_results),
        "sell_recommendations": _summarize_sells(sell_results),
        "hold_recommendations": _summarize_holds(hold_results),
        "conviction_calibration": _calibrate_conviction(
            buy_results + hold_results
        ),
        "benchmark_comparison": compute_benchmark_comparison(months_back, log_path),
        "modifier_attribution": attribute_performance_to_modifiers(months_back, log_path),
        "details": {
            "buys": buy_results,
            "sells": sell_results,
            "holds": hold_results,
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

        if not ticker or entry_price is None:
            continue

        try:
            entry_price = float(entry_price)
            if entry_price <= 0:
                continue
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
        return f"Track record: {hit_rate:.0f}% BUY alpha hit rate at T+30 (n={total}, vs SPY)"

    hit_rate_7d = buy_recs.get("hit_rate_7d")
    if hit_rate_7d is not None:
        return f"Track record: {hit_rate_7d:.0f}% BUY alpha hit rate at T+7 (n={total}, vs SPY)"

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

    # CIO v22.0: Load latest concordance for ADX/divergence/Piotroski triggers
    concordance_map = {}
    concordance_path = _USER_COMMITTEE_DIR / "concordance.json"
    if concordance_path.exists():
        try:
            with open(concordance_path, "r") as f:
                conc_data = json.load(f)
            # Handle both list format and dict-with-stocks format
            entries = conc_data if isinstance(conc_data, list) else conc_data.get("concordance", [])
            if isinstance(entries, dict):
                concordance_map = entries
            else:
                for entry in entries:
                    tkr = entry.get("ticker", "")
                    if tkr:
                        concordance_map[tkr] = entry
        except (json.JSONDecodeError, OSError):
            pass

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
        concordance_entry = concordance_map.get(ticker, {})

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

        # CIO v22.0: Check concordance-sourced indicators
        if concordance_entry:
            # Bearish RSI divergence on a BUY position = reversal warning
            if concordance_entry.get("divergence") == "bearish":
                triggers.append("bearish_divergence")

            # Strong downtrend confirmed by ADX >= 30 with -DI dominant
            adx = concordance_entry.get("adx")
            if adx and adx >= 30 and concordance_entry.get("entry_timing") in ("AVOID", "EXIT_SOON"):
                triggers.append("strong_downtrend_confirmed")

            # Piotroski quality deterioration (score <= 2 = severe weakness)
            pio = concordance_entry.get("piotroski_score", 0)
            if pio and pio <= 2:
                triggers.append("piotroski_quality_failure")

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
    """Summarize BUY recommendation performance using alpha (vs SPY)."""
    if not results:
        return {"total": 0}

    alphas_7d = [r["alpha_7d"] for r in results if "alpha_7d" in r]
    alphas_30d = [r["alpha_30d"] for r in results if "alpha_30d" in r]
    returns_7d = [r["return_7d"] for r in results if "return_7d" in r]
    returns_30d = [r["return_30d"] for r in results if "return_30d" in r]

    summary: Dict[str, Any] = {"total": len(results)}

    if alphas_7d:
        summary["hit_rate_7d"] = round(
            sum(1 for a in alphas_7d if a > 0) / len(alphas_7d) * 100, 1
        )
        summary["avg_alpha_7d"] = round(np.mean(alphas_7d), 2)
    if returns_7d:
        summary["avg_return_7d"] = round(np.mean(returns_7d), 2)

    if alphas_30d:
        summary["hit_rate_30d"] = round(
            sum(1 for a in alphas_30d if a > 0) / len(alphas_30d) * 100, 1
        )
        summary["avg_alpha_30d"] = round(np.mean(alphas_30d), 2)
    if returns_30d:
        summary["avg_return_30d"] = round(np.mean(returns_30d), 2)

        # Best and worst by alpha
        with_alpha = [r for r in results if "alpha_30d" in r]
        if with_alpha:
            best = max(with_alpha, key=lambda r: r["alpha_30d"])
            worst = min(with_alpha, key=lambda r: r["alpha_30d"])
            summary["best_30d"] = f"{best['ticker']} α={best['alpha_30d']:+.1f}%"
            summary["worst_30d"] = f"{worst['ticker']} α={worst['alpha_30d']:+.1f}%"

    return summary


def _summarize_sells(results: List[Dict]) -> Dict[str, Any]:
    """Summarize SELL recommendation performance (validated if underperformed SPY)."""
    if not results:
        return {"total": 0}

    alphas_30d = [r["alpha_30d"] for r in results if "alpha_30d" in r]
    returns_30d = [r["return_30d"] for r in results if "return_30d" in r]

    summary: Dict[str, Any] = {"total": len(results)}

    if alphas_30d:
        # For SELLs, validated = stock underperformed SPY (negative alpha)
        summary["validated_30d"] = round(
            sum(1 for a in alphas_30d if a < 0) / len(alphas_30d) * 100, 1
        )
        summary["avg_alpha_30d"] = round(np.mean(alphas_30d), 2)
    if returns_30d:
        summary["avg_return_30d"] = round(np.mean(returns_30d), 2)

    return summary


def _summarize_holds(results: List[Dict]) -> Dict[str, Any]:
    """Summarize HOLD recommendation performance.

    A HOLD is validated if it tracked the market (alpha within ±3%).
    A HOLD with alpha < -8% is a clear miss (should have been SELL).
    """
    if not results:
        return {"total": 0}

    alphas_30d = [r["alpha_30d"] for r in results if "alpha_30d" in r]
    returns_30d = [r["return_30d"] for r in results if "return_30d" in r]

    summary: Dict[str, Any] = {"total": len(results)}

    if alphas_30d:
        summary["validated_30d"] = round(
            sum(1 for a in alphas_30d if a > -3) / len(alphas_30d) * 100, 1
        )
        summary["avg_alpha_30d"] = round(float(np.mean(alphas_30d)), 2)
        summary["missed_30d"] = sum(1 for a in alphas_30d if a < -8)
    if returns_30d:
        summary["avg_return_30d"] = round(float(np.mean(returns_30d)), 2)

    alphas_7d = [r["alpha_7d"] for r in results if "alpha_7d" in r]
    if alphas_7d:
        summary["avg_alpha_7d"] = round(float(np.mean(alphas_7d)), 2)

    return summary


def _calibrate_conviction(results: List[Dict]) -> Dict[str, Any]:
    """
    Check if conviction scores predict alpha (outperformance vs SPY).

    High conviction should correlate with higher alpha.
    """
    if not results or len(results) < 5:
        return {"sufficient_data": False}

    scored = [
        r for r in results
        if r.get("conviction") is not None and "alpha_30d" in r
    ]

    if len(scored) < 5:
        return {"sufficient_data": False}

    # Split into high/low conviction
    high_conv = [r for r in scored if r["conviction"] >= 65]
    low_conv = [r for r in scored if r["conviction"] < 65]

    calibration: Dict[str, Any] = {"sufficient_data": True}

    if high_conv:
        calibration["high_conviction_avg_alpha"] = round(
            np.mean([r["alpha_30d"] for r in high_conv]), 2
        )
        calibration["high_conviction_count"] = len(high_conv)

    if low_conv:
        calibration["low_conviction_avg_alpha"] = round(
            np.mean([r["alpha_30d"] for r in low_conv]), 2
        )
        calibration["low_conviction_count"] = len(low_conv)

    # Check if high conviction generates more alpha than low conviction
    if high_conv and low_conv:
        high_avg = np.mean([r["alpha_30d"] for r in high_conv])
        low_avg = np.mean([r["alpha_30d"] for r in low_conv])
        calibration["conviction_predictive"] = high_avg > low_avg
        calibration["conviction_spread"] = round(high_avg - low_avg, 2)

    # CIO v22.0 + v21.0: Indicator effectiveness tracking (alpha-based)
    indicator_groups = {
        "strong_adx": [r for r in scored if r.get("adx") and r["adx"] >= 30],
        "bullish_divergence": [r for r in scored if r.get("divergence") == "bullish"],
        "high_piotroski": [r for r in scored if r.get("piotroski_score") and r["piotroski_score"] >= 7],
        "rs_outperforming": [r for r in scored if r.get("rs_vs_spy") and r["rs_vs_spy"] > 1.0],
        "rs_underperforming": [r for r in scored if r.get("rs_vs_spy") and r["rs_vs_spy"] < 0.95],
        "rev_accelerating": [r for r in scored if r.get("revenue_growth_class", "").upper() in ("ACCELERATING", "STRONG_GROWTH")],
        "high_atr_pct": [r for r in scored if r.get("atr_pct") and r["atr_pct"] > 5.0],
        "high_short_interest": [r for r in scored if r.get("short_interest_pct") and r["short_interest_pct"] > 10],
        "high_dispersion": [r for r in scored if r.get("target_dispersion") and r["target_dispersion"] > 30],
        "low_dispersion": [r for r in scored if r.get("target_dispersion") and r["target_dispersion"] < 10],
        "usd_zone": [r for r in scored if r.get("currency_zone") == "USD"],
        "eur_zone": [r for r in scored if r.get("currency_zone") == "EUR"],
        "other_fx_zone": [r for r in scored if r.get("currency_zone") not in (None, "USD", "EUR", "CRYPTO")],
        "strong_confluence": [r for r in scored if r.get("confluence_score") and r["confluence_score"] >= 5],
        "weak_confluence": [r for r in scored if r.get("confluence_score") and r["confluence_score"] <= -3],
        "volume_confirmed": [r for r in scored if r.get("relative_volume") and r["relative_volume"] > 1.5 and r.get("obv_trend") == "RISING"],
        "eps_revisions_up": [r for r in scored if r.get("eps_revisions") == "REVISIONS_UP"],
        "low_iv_rank": [r for r in scored if r.get("iv_rank") is not None and r["iv_rank"] < 20],
        "high_iv_rank": [r for r in scored if r.get("iv_rank") is not None and r["iv_rank"] > 80],
        "strong_fcf": [r for r in scored if r.get("fcf_classification") == "STRONG"],
        "weak_fcf": [r for r in scored if r.get("fcf_classification") == "WEAK"],
        "high_debt_risk": [r for r in scored if r.get("debt_risk") == "HIGH_RISK"],
    }
    for label, group in indicator_groups.items():
        if len(group) >= 3:
            avg_alpha = float(np.mean([r["alpha_30d"] for r in group]))
            alpha_hit = sum(1 for r in group if r["alpha_30d"] > 0) / len(group) * 100
            calibration[f"{label}_avg_alpha"] = round(avg_alpha, 2)
            calibration[f"{label}_alpha_hit_rate"] = round(alpha_hit, 1)
            calibration[f"{label}_count"] = len(group)

    return calibration


def _save_scorecard(scorecard: Dict[str, Any]) -> None:
    """Save scorecard to JSON file."""
    _USER_COMMITTEE_DIR.mkdir(parents=True, exist_ok=True)
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
        "hold_recommendations": {"total": 0},
        "conviction_calibration": {"sufficient_data": False},
        "details": {"buys": [], "sells": [], "holds": []},
    }


# =============================================================================
# CIO v23.6: Self-Calibration Infrastructure
# =============================================================================

def calibrate_modifiers(
    months_back: int = 3,
    min_observations: int = 10,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    CIO v23.6: Auto-calibration from scorecard data.

    For each modifier/indicator group, computes:
    - Hit rate with vs without the modifier applied
    - Average return delta
    - Statistical significance (n >= min_observations)
    - Recommendation: KEEP, ADJUST, or REMOVE

    Returns calibration report with per-modifier analysis.
    """
    actions = load_committee_actions(months_back, log_path)
    if not actions:
        return {"sufficient_data": False, "total_actions": 0}

    try:
        import yfinance as yf
    except ImportError:
        return {"sufficient_data": False, "error": "yfinance not available"}

    # Fetch current prices + SPY benchmark for alpha calculation
    tickers = list(set(a["ticker"] for a in actions if a.get("ticker")))
    if not tickers:
        return {"sufficient_data": False, "total_actions": 0}

    # Ensure SPY is fetched alongside stock prices
    fetch_tickers = list(set(tickers + ["SPY"]))

    scored = []
    try:
        data = yf.download(fetch_tickers, period="3mo", progress=False, auto_adjust=True)
        prices = {}
        if not data.empty:
            for t in fetch_tickers:
                try:
                    closes = data['Close'][t].dropna() if len(fetch_tickers) > 1 else data['Close'].dropna()
                    if not closes.empty:
                        prices[t] = closes  # Keep full series for date-matched alpha
                except (KeyError, IndexError):
                    pass
    except Exception:
        prices = {}

    spy_series = prices.get("SPY")

    for a in actions:
        t = a.get("ticker")
        rec_price = a.get("price_at_recommendation")
        committee_date = a.get("committee_date")
        t_series = prices.get(t)
        if rec_price is None or t_series is None or float(rec_price) <= 0:
            continue

        rec_price = float(rec_price)
        curr = float(t_series.iloc[-1])
        ret = (curr - rec_price) / rec_price * 100

        # Compute alpha: stock return minus SPY return over same period
        alpha = ret  # Fallback to absolute return if SPY unavailable
        if spy_series is not None and committee_date:
            spy_from_date = spy_series[spy_series.index >= committee_date]
            if len(spy_from_date) >= 2:
                spy_base = float(spy_from_date.iloc[0])
                spy_curr = float(spy_series.iloc[-1])
                if spy_base > 0:
                    spy_ret = (spy_curr - spy_base) / spy_base * 100
                    alpha = ret - spy_ret

        scored.append({**a, "return_pct": round(ret, 2), "alpha_pct": round(alpha, 2)})

    if len(scored) < min_observations:
        return {"sufficient_data": False, "total_scored": len(scored),
                "min_required": min_observations}

    # Define modifier groups to calibrate
    modifier_defs = {
        "strong_adx": lambda r: r.get("adx") and r["adx"] >= 30,
        "bullish_divergence": lambda r: r.get("divergence") == "bullish",
        "high_piotroski": lambda r: r.get("piotroski_score") and r["piotroski_score"] >= 7,
        "rs_outperforming": lambda r: r.get("rs_vs_spy") and r["rs_vs_spy"] > 1.0,
        "high_short_interest": lambda r: r.get("short_interest_pct") and r["short_interest_pct"] > 10,
        "strong_confluence": lambda r: r.get("confluence_score") and r["confluence_score"] >= 5,
        "volume_confirmed": lambda r: r.get("relative_volume") and r["relative_volume"] > 1.5,
        "eps_revisions_up": lambda r: r.get("eps_revisions") == "REVISIONS_UP",
        "strong_fcf": lambda r: r.get("fcf_classification") == "STRONG",
        # CIO v27.0: Adversarial debate signal calibration
        "debate_bearish": lambda r: r.get("debate_signal") in ("WEAKEN_BULL", "STRENGTHEN_BEAR"),
        "debate_bullish": lambda r: r.get("debate_signal") in ("STRENGTHEN_BULL", "WEAKEN_BEAR"),
    }

    calibration = {"sufficient_data": True, "total_scored": len(scored), "modifiers": {}}
    buy_scored = [r for r in scored if r.get("action") == "BUY"]
    baseline_avg = float(np.mean([r["alpha_pct"] for r in buy_scored])) if buy_scored else 0
    baseline_hit = (sum(1 for r in buy_scored if r["alpha_pct"] > 0) / len(buy_scored) * 100) if buy_scored else 0

    for mod_name, mod_fn in modifier_defs.items():
        with_mod = [r for r in buy_scored if mod_fn(r)]
        without_mod = [r for r in buy_scored if not mod_fn(r)]

        if len(with_mod) >= 3 and len(without_mod) >= 3:
            with_avg = float(np.mean([r["alpha_pct"] for r in with_mod]))
            without_avg = float(np.mean([r["alpha_pct"] for r in without_mod]))
            with_hit = sum(1 for r in with_mod if r["alpha_pct"] > 0) / len(with_mod) * 100
            without_hit = sum(1 for r in without_mod if r["alpha_pct"] > 0) / len(without_mod) * 100

            delta = with_avg - without_avg
            if delta > 2 and with_hit > without_hit:
                recommendation = "KEEP"
            elif delta < -2 and with_hit < without_hit:
                recommendation = "REMOVE"
            else:
                recommendation = "ADJUST"

            calibration["modifiers"][mod_name] = {
                "with_count": len(with_mod),
                "without_count": len(without_mod),
                "with_avg_alpha": round(with_avg, 2),
                "without_avg_alpha": round(without_avg, 2),
                "alpha_delta": round(delta, 2),
                "with_alpha_hit_rate": round(with_hit, 1),
                "without_alpha_hit_rate": round(without_hit, 1),
                "recommendation": recommendation,
            }

    calibration["baseline"] = {
        "avg_alpha": round(baseline_avg, 2),
        "alpha_hit_rate": round(baseline_hit, 1),
        "total_buys": len(buy_scored),
    }

    # CIO v25.0: Waterfall-based calibration — uses actual conviction deltas
    # correlated with alpha (regime-neutral) rather than absolute returns.
    wf_stats: Dict[str, Dict[str, Any]] = {}
    for r in scored:
        wf = r.get("conviction_waterfall")
        if not isinstance(wf, dict):
            continue
        for mod_key, delta in wf.items():
            if mod_key.startswith("_"):
                continue
            if mod_key not in wf_stats:
                wf_stats[mod_key] = {"deltas": [], "alphas": []}
            wf_stats[mod_key]["deltas"].append(delta)
            wf_stats[mod_key]["alphas"].append(r.get("alpha_pct", 0))

    if wf_stats:
        calibration["waterfall_calibration"] = {}
        for mod_key, stats in wf_stats.items():
            if len(stats["deltas"]) < 3:
                continue
            avg_delta = float(np.mean(stats["deltas"]))
            avg_alpha = float(np.mean(stats["alphas"]))
            correlation = float(np.corrcoef(stats["deltas"], stats["alphas"])[0, 1]) \
                if len(stats["deltas"]) >= 5 else None
            calibration["waterfall_calibration"][mod_key] = {
                "count": len(stats["deltas"]),
                "avg_delta": round(avg_delta, 2),
                "avg_alpha": round(avg_alpha, 2),
                "delta_alpha_corr": round(correlation, 3) if correlation is not None else None,
            }

    return calibration


def attribute_performance_to_modifiers(
    months_back: int = 3,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    CIO v23.6: Performance attribution per modifier.

    For each modifier that fired, compute:
    - "Stocks that got [modifier] bonus returned X% vs Y% without"
    - Validates or invalidates each modifier individually.
    """
    cal = calibrate_modifiers(months_back, min_observations=5, log_path=log_path)
    if not cal.get("sufficient_data"):
        return {"sufficient_data": False}

    attribution = {"sufficient_data": True, "modifiers": {}}
    baseline = cal.get("baseline", {})

    for mod_name, mod_data in cal.get("modifiers", {}).items():
        delta = mod_data.get("alpha_delta", 0)
        with_hit = mod_data.get("with_alpha_hit_rate", 0)
        recommendation = mod_data.get("recommendation", "ADJUST")

        verdict = "EFFECTIVE" if recommendation == "KEEP" else (
            "INEFFECTIVE" if recommendation == "REMOVE" else "INCONCLUSIVE"
        )

        attribution["modifiers"][mod_name] = {
            "verdict": verdict,
            "alpha_delta": delta,
            "with_avg_alpha": mod_data.get("with_avg_alpha", 0),
            "without_avg_alpha": mod_data.get("without_avg_alpha", 0),
            "with_alpha_hit_rate": with_hit,
            "sample_size": mod_data.get("with_count", 0),
            "narrative": (
                f"Stocks with {mod_name}: α={mod_data.get('with_avg_alpha', 0):+.1f}% "
                f"vs α={mod_data.get('without_avg_alpha', 0):+.1f}% without "
                f"({delta:+.1f}% alpha delta, {with_hit:.0f}% beat SPY)"
            ),
        }

    return attribution


def compute_benchmark_comparison(
    months_back: int = 3,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    CIO v23.6: Committee BUY portfolio vs SPY buy-and-hold.

    Computes information ratio, tracking error, and excess return.
    """
    actions = load_committee_actions(months_back, log_path)
    buy_actions = [a for a in actions if a.get("action") == "BUY" and a.get("price_at_recommendation")]

    if len(buy_actions) < 3:
        return {"sufficient_data": False}

    try:
        import yfinance as yf
    except ImportError:
        return {"sufficient_data": False, "error": "yfinance not available"}

    # Get SPY performance over the same period
    try:
        spy = yf.download("SPY", period=f"{months_back * 30}d", progress=False, auto_adjust=True)
        if spy.empty:
            return {"sufficient_data": False}
        spy_start = float(spy['Close'].iloc[0])
        spy_end = float(spy['Close'].iloc[-1])
        spy_return = (spy_end - spy_start) / spy_start * 100
    except Exception:
        return {"sufficient_data": False, "error": "SPY data unavailable"}

    # Get current prices for BUY portfolio
    tickers = list(set(a["ticker"] for a in buy_actions))
    try:
        data = yf.download(tickers, period="3mo", progress=False, auto_adjust=True)
    except Exception:
        return {"sufficient_data": False}

    returns = []
    for a in buy_actions:
        t = a.get("ticker")
        rec_price = a.get("price_at_recommendation")
        if rec_price is None:
            continue
        try:
            rec_price = float(rec_price)
            if rec_price <= 0:
                continue
        except (ValueError, TypeError):
            continue
        try:
            closes = data['Close'][t].dropna() if len(tickers) > 1 else data['Close'].dropna()
            curr = float(closes.iloc[-1])
            ret = (curr - rec_price) / rec_price * 100
            returns.append(ret)
        except (KeyError, IndexError):
            pass

    if len(returns) < 3:
        return {"sufficient_data": False}

    portfolio_return = float(np.mean(returns))
    excess_return = portfolio_return - spy_return

    # Tracking error (std dev of return differences from SPY)
    return_diffs = [r - spy_return for r in returns]
    tracking_error = float(np.std(return_diffs)) if len(return_diffs) > 1 else 0

    # Information ratio
    info_ratio = excess_return / tracking_error if tracking_error > 0 else 0

    return {
        "sufficient_data": True,
        "portfolio_return": round(portfolio_return, 2),
        "spy_return": round(spy_return, 2),
        "excess_return": round(excess_return, 2),
        "tracking_error": round(tracking_error, 2),
        "information_ratio": round(info_ratio, 2),
        "total_buys_evaluated": len(returns),
        "period_months": months_back,
    }

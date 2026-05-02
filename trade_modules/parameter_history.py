"""Parameter History — Archive ALL parameters per stock per committee run.

Captures ~120 parameters per stock from every data source (signals, fundamental,
technical, macro, census, news, risk, synthesis) into a single JSONL file.
This enables continuous backtesting: correlate ANY parameter with T+7/T+30
forward returns to discover which parameters predict alpha.

CIO v35.0: Created as part of the data pipeline workstream.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HISTORY_PATH = Path.home() / ".weirdapps-trading" / "committee" / "parameter_history.jsonl"


def save_parameter_history(
    date: str,
    concordance: List[Dict[str, Any]],
    portfolio_signals: Dict[str, Dict],
    fund_report: Dict,
    tech_report: Dict,
    macro_report: Dict,
    census_report: Dict,
    news_report: Dict,
    risk_report: Dict,
) -> int:
    """Archive all parameters for every stock in this committee run."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    tickers = {e.get("ticker", "") for e in concordance}
    tickers.update(portfolio_signals.keys())
    tickers.discard("")

    lines = 0
    with open(HISTORY_PATH, "a") as f:
        for ticker in sorted(tickers):
            lines += _write_signals(f, date, ticker, portfolio_signals)
            lines += _write_fundamental(f, date, ticker, fund_report)
            lines += _write_technical(f, date, ticker, tech_report)
            lines += _write_macro(f, date, ticker, macro_report)
            lines += _write_census(f, date, ticker, census_report)
            lines += _write_news(f, date, ticker, news_report)
            lines += _write_risk(f, date, ticker, risk_report)
            lines += _write_synthesis(f, date, ticker, concordance)

    logger.info("Parameter history: %d lines for %d tickers on %s", lines, len(tickers), date)
    return lines


def _write_record(f, record: Dict) -> int:
    f.write(json.dumps(record, default=str) + "\n")
    return 1


def _write_signals(f, date: str, ticker: str, signals: Dict) -> int:
    sig = signals.get(ticker)
    if not sig:
        return 0
    record = {"date": date, "ticker": ticker, "source": "signals"}
    for key in ("price", "upside", "buy_pct", "exret", "beta", "pet", "pef",
                "pp", "52w", "am", "analyst_type", "signal", "short_interest",
                "roe", "de", "fcf", "num_targets"):
        val = sig.get(key)
        if val is not None:
            record[key] = val
    return _write_record(f, record)


def _write_fundamental(f, date: str, ticker: str, fund: Dict) -> int:
    stock = fund.get("stocks", {}).get(ticker)
    if not stock or not isinstance(stock, dict):
        return 0
    record = {"date": date, "ticker": ticker, "source": "fundamental"}
    for key in ("fundamental_score", "outlook", "pe_trajectory", "quality_trap",
                "piotroski_score", "revenue_growth_class", "eps_revisions",
                "insider_sentiment", "earnings_quality"):
        val = stock.get(key)
        if val is not None:
            record[key] = val
    return _write_record(f, record)


def _write_technical(f, date: str, ticker: str, tech: Dict) -> int:
    stock = tech.get("stocks", {}).get(ticker)
    if not stock or not isinstance(stock, dict):
        return 0
    record = {"date": date, "ticker": ticker, "source": "technical"}
    for key in ("price", "rsi", "macd_signal", "macd_histogram", "bb_position",
                "above_sma50", "above_sma200", "golden_cross", "vol_ratio",
                "support", "resistance", "momentum_score", "trend",
                "timing_signal", "relative_strength_vs_spy", "atr_pct",
                "adx", "adx_trend"):
        val = stock.get(key)
        if val is not None:
            record[key] = val
    return _write_record(f, record)


def _write_macro(f, date: str, ticker: str, macro: Dict) -> int:
    impl = macro.get("portfolio_implications", {}).get(ticker)
    if not impl or not isinstance(impl, dict):
        return 0
    record = {
        "date": date, "ticker": ticker, "source": "macro",
        "regime": macro.get("regime"),
        "macro_score": macro.get("macro_score"),
        "rotation_phase": macro.get("rotation_phase"),
    }
    for key in ("macro_fit", "rate_sensitive", "dollar_impact"):
        val = impl.get(key)
        if val is not None:
            record[key] = val
    return _write_record(f, record)


def _write_census(f, date: str, ticker: str, census: Dict) -> int:
    sentiment = census.get("sentiment", {})
    record = {
        "date": date, "ticker": ticker, "source": "census",
        "fg_top100": _num(sentiment.get("fg_top100")),
        "fg_broad": _num(sentiment.get("fg_broad")),
        "cash_top100": _num(sentiment.get("cash_top100")),
        "cash_broad": _num(sentiment.get("cash_broad")),
    }
    return _write_record(f, record)


def _write_news(f, date: str, ticker: str, news: Dict) -> int:
    port_news = news.get("portfolio_news", {}).get(ticker, [])
    impact = "NEUTRAL"
    if isinstance(port_news, list) and port_news:
        impacts = [n.get("impact", "NEUTRAL") for n in port_news if isinstance(n, dict)]
        if any("NEGATIVE" in i for i in impacts):
            impact = "NEGATIVE"
        elif any("POSITIVE" in i for i in impacts):
            impact = "POSITIVE"

    earnings_days = None
    for ear in news.get("earnings_calendar", []):
        if isinstance(ear, dict) and ear.get("ticker") == ticker:
            try:
                earnings_days = int(ear.get("days_away", 999))
            except (ValueError, TypeError):
                pass

    record = {
        "date": date, "ticker": ticker, "source": "news",
        "news_impact": impact,
        "earnings_days_away": earnings_days,
        "news_count": len(port_news) if isinstance(port_news, list) else 0,
    }
    return _write_record(f, record)


def _write_risk(f, date: str, ticker: str, risk: Dict) -> int:
    limits = risk.get("position_limits", {}).get(ticker, {})
    record = {
        "date": date, "ticker": ticker, "source": "risk",
        "position_limit_pct": limits.get("max_pct") if isinstance(limits, dict) else None,
        "portfolio_risk_score": risk.get("portfolio_risk", {}).get("risk_score"),
        "portfolio_beta": risk.get("portfolio_risk", {}).get("portfolio_beta"),
    }
    return _write_record(f, record)


def _write_synthesis(f, date: str, ticker: str, concordance: List[Dict]) -> int:
    entry = next((e for e in concordance if e.get("ticker") == ticker), None)
    if not entry:
        return 0
    record = {"date": date, "ticker": ticker, "source": "synthesis"}
    for key in ("conviction", "action", "base", "bonuses", "penalties",
                "bull_weight", "bear_weight", "bull_pct",
                "signal_velocity", "earnings_surprise", "days_held",
                "holding_review_flag", "position_size_pct", "entry_timing",
                "is_opportunity", "sector", "directional_confidence",
                "hold_tier", "max_pct", "fund_score", "tech_signal",
                "macro_fit", "census", "news_impact", "rsi", "exret",
                "buy_pct", "signal", "price"):
        val = entry.get(key)
        if val is not None:
            record[key] = val
    wf = entry.get("conviction_waterfall", {})
    if wf:
        record["waterfall"] = wf
    return _write_record(f, record)


def _num(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, dict):
        return val.get("value", val.get("score"))
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def get_parameter_history(
    ticker: Optional[str] = None,
    source: Optional[str] = None,
    days: int = 90,
) -> List[Dict]:
    """Read parameter history, optionally filtered by ticker and source."""
    if not HISTORY_PATH.exists():
        return []

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    results = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("date", "") < cutoff:
                continue
            if ticker and record.get("ticker") != ticker:
                continue
            if source and record.get("source") != source:
                continue
            results.append(record)
    return results

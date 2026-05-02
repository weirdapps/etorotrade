"""Parameter History — Archive ALL parameters per stock per committee run.

Captures ~120 parameters per stock from every data source (signals, fundamental,
technical, macro, census, news, risk, synthesis) into a single JSONL file.
This enables continuous backtesting: correlate ANY parameter with T+7/T+30
forward returns to discover which parameters predict alpha.

CIO v35.0: Created as part of the data pipeline workstream.
"""

import csv
import json
import logging
import os
from datetime import datetime
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
    """Archive all parameters for every stock in this committee run.

    Writes one JSONL line per (stock × source). Returns total lines written.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0

    tickers = set()
    for entry in concordance:
        tickers.add(entry.get("ticker", ""))
    for tkr in portfolio_signals:
        tickers.add(tkr)
    tickers.discard("")

    with open(HISTORY_PATH, "a") as f:
        for ticker in sorted(tickers):
            # Source: signals
            sig = portfolio_signals.get(ticker, {})
            if sig:
                record = {"date": date, "ticker": ticker, "source": "signals"}
                for key in ("price", "upside", "buy_pct", "exret", "beta",
                            "pet", "pef", "pp", "52w", "am", "analyst_type",
                            "signal", "short_interest", "roe", "de", "fcf",
                            "num_targets", "upside"):
                    if key in sig and sig[key] is not None:
                        record[key] = sig[key]
                f.write(json.dumps(record, default=str) + "\n")
                lines_written += 1

            # Source: fundamental
            fund_stock = fund_report.get("stocks", {}).get(ticker, {})
            if fund_stock and isinstance(fund_stock, dict):
                record = {"date": date, "ticker": ticker, "source": "fundamental"}
                for key in ("fundamental_score", "outlook", "pe_trajectory",
                            "quality_trap", "piotroski_score", "revenue_growth_class",
                            "eps_revisions", "insider_sentiment", "insider_detail",
                            "peer_rank", "ev_ebitda", "fcf_yield_pct", "debt_risk",
                            "earnings_quality"):
                    val = fund_stock.get(key)
                    if val is not None:
                        record[key] = val
                f.write(json.dumps(record, default=str) + "\n")
                lines_written += 1

            # Source: technical
            tech_stock = tech_report.get("stocks", {}).get(ticker, {})
            if tech_stock and isinstance(tech_stock, dict):
                record = {"date": date, "ticker": ticker, "source": "technical"}
                for key in ("price", "rsi", "macd_signal", "macd_histogram",
                            "bb_position", "above_sma50", "above_sma200",
                            "golden_cross", "vol_ratio", "support", "resistance",
                            "momentum_score", "trend", "timing_signal",
                            "relative_strength_vs_spy", "atr_pct", "adx",
                            "adx_trend", "volume_confirmation", "rsi_divergence"):
                    val = tech_stock.get(key)
                    if val is not None:
                        record[key] = val
                f.write(json.dumps(record, default=str) + "\n")
                lines_written += 1

            # Source: macro (per-stock implications)
            macro_impl = macro_report.get("portfolio_implications", {}).get(ticker, {})
            if macro_impl and isinstance(macro_impl, dict):
                record = {
                    "date": date, "ticker": ticker, "source": "macro",
                    "regime": macro_report.get("regime"),
                    "macro_score": macro_report.get("macro_score"),
                    "rotation_phase": macro_report.get("rotation_phase"),
                }
                for key in ("macro_fit", "rate_sensitive", "dollar_impact",
                            "reasoning"):
                    val = macro_impl.get(key)
                    if val is not None:
                        record[key] = val
                f.write(json.dumps(record, default=str) + "\n")
                lines_written += 1

            # Source: census
            sentiment = census_report.get("sentiment", {})
            census_divs = census_report.get("divergences", {})
            div_score = 0
            alignment = "NEUTRAL"
            for div_list in census_divs.values():
                if isinstance(div_list, list):
                    for item in div_list:
                        if isinstance(item, dict) and item.get("ticker") == ticker:
                            div_score = item.get("divergence_score", 0)

            record = {
                "date": date, "ticker": ticker, "source": "census",
                "fg_top100": _num(sentiment.get("fg_top100")),
                "fg_broad": _num(sentiment.get("fg_broad")),
                "cash_top100": _num(sentiment.get("cash_top100")),
                "cash_broad": _num(sentiment.get("cash_broad")),
                "divergence_score": div_score,
            }
            f.write(json.dumps(record, default=str) + "\n")
            lines_written += 1

            # Source: news
            port_news = news_report.get("portfolio_news", {}).get(ticker, [])
            news_impact = "NEUTRAL"
            if isinstance(port_news, list) and port_news:
                impacts = [n.get("impact", "NEUTRAL") for n in port_news if isinstance(n, dict)]
                if any("NEGATIVE" in i for i in impacts):
                    news_impact = "NEGATIVE"
                elif any("POSITIVE" in i for i in impacts):
                    news_impact = "POSITIVE"

            earnings_days = None
            for ear in news_report.get("earnings_calendar", []):
                if isinstance(ear, dict) and ear.get("ticker") == ticker:
                    try:
                        earnings_days = int(ear.get("days_away", 999))
                    except (ValueError, TypeError):
                        pass

            record = {
                "date": date, "ticker": ticker, "source": "news",
                "news_impact": news_impact,
                "earnings_days_away": earnings_days,
                "news_count": len(port_news) if isinstance(port_news, list) else 0,
            }
            f.write(json.dumps(record, default=str) + "\n")
            lines_written += 1

            # Source: risk
            risk_limits = risk_report.get("position_limits", {}).get(ticker, {})
            devils = risk_report.get("devils_advocate", {}).get(ticker, "")
            record = {
                "date": date, "ticker": ticker, "source": "risk",
                "position_limit_pct": risk_limits.get("max_pct") if isinstance(risk_limits, dict) else None,
                "portfolio_risk_score": risk_report.get("portfolio_risk", {}).get("risk_score"),
                "portfolio_beta": risk_report.get("portfolio_risk", {}).get("portfolio_beta"),
                "has_devils_advocate": bool(devils),
            }
            f.write(json.dumps(record, default=str) + "\n")
            lines_written += 1

            # Source: synthesis (from concordance)
            conc_entry = next((e for e in concordance if e.get("ticker") == ticker), None)
            if conc_entry:
                record = {"date": date, "ticker": ticker, "source": "synthesis"}
                for key in ("conviction", "action", "base", "bonuses", "penalties",
                            "bull_weight", "bear_weight", "bull_pct",
                            "signal_velocity", "earnings_surprise",
                            "days_held", "holding_review_flag",
                            "circuit_breaker_blocked", "position_size_pct",
                            "entry_timing", "is_opportunity", "sector",
                            "directional_confidence", "hold_tier", "max_pct",
                            "fund_score", "tech_signal", "macro_fit",
                            "census", "news_impact", "rsi", "exret",
                            "buy_pct", "signal", "price"):
                    val = conc_entry.get(key)
                    if val is not None:
                        record[key] = val
                # Save FULL waterfall (including shadow ~ modifiers)
                wf = conc_entry.get("conviction_waterfall", {})
                if wf:
                    record["waterfall"] = wf
                f.write(json.dumps(record, default=str) + "\n")
                lines_written += 1

    logger.info("Parameter history: %d lines written for %d tickers on %s",
                lines_written, len(tickers), date)
    return lines_written


def _num(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, dict):
        return val.get("value", val.get("score", None))
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

    from datetime import timedelta
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

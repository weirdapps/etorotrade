#!/usr/bin/env python3
"""
CIO v36 synthesis runner — loads the 8 (now-real) agent reports, builds
the concordance with the new ACTIVE_MODIFIERS set, applies the M4 BUY
gate, and writes synthesis.json + HTML report.

Usage:
    CIO_V36_ENABLE_EMPIRICAL_GATE=1 python run_v36_synthesis.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

# Make the repo importable
REPO = Path.home() / "SourceCode" / "etorotrade"
sys.path.insert(0, str(REPO))

from trade_modules.census_time_series import get_census_context  # noqa: E402
from trade_modules.committee_html import generate_report_from_files  # noqa: E402
from trade_modules.committee_qa import normalize_agent_reports  # noqa: E402
from trade_modules.committee_synthesis import (  # noqa: E402
    build_concordance,
    compute_changes,
    detect_sector_gaps,
    enrich_with_position_sizes,
    generate_synthesis_output,
)
from trade_modules.vol_targeting import (  # noqa: E402
    compute_vol_scale,
    estimate_portfolio_vol_from_history,
)

REPORTS_DIR = Path.home() / ".weirdapps-trading" / "committee" / "reports"
PORTFOLIO_CSV = REPO / "yahoofinance" / "output" / "portfolio.csv"
ETORO_CSV = REPO / "yahoofinance" / "output" / "etoro.csv"
SYNTHESIS_JSON = REPORTS_DIR / "synthesis.json"
PREV_CONCORDANCE = Path.home() / ".weirdapps-trading" / "committee" / "concordance.json"


def load_report(name: str) -> dict:
    p = REPORTS_DIR / f"{name}.json"
    with open(p) as f:
        return json.load(f)


def _f(value, default=0.0):
    if value is None:
        return default
    try:
        return float(str(value).replace("%", "").replace(",", "").replace("--", ""))
    except (ValueError, TypeError):
        return default


def build_portfolio_signals():
    """Read portfolio.csv → dict of ticker → signal-engine fields.

    Header: TKR,NAME,CAP,PRC,TGT,UP%,#T,%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS
    """
    if not PORTFOLIO_CSV.exists():
        raise SystemExit(f"Portfolio CSV missing: {PORTFOLIO_CSV}")

    signals = {}
    sector_map = {}
    with open(PORTFOLIO_CSV) as f:
        for row in csv.DictReader(f):
            t = row.get("TKR", "").strip()
            if not t:
                continue
            signals[t] = {
                "signal": row.get("BS", "H"),  # B/H/S/I
                "exret": _f(row.get("EXR")),
                "buy_pct": _f(row.get("%B")),
                "beta": _f(row.get("B"), 1.0),
                "pet": _f(row.get("PET")),
                "pef": _f(row.get("PEF")),
                "pp": _f(row.get("PP")),
                "52w": _f(row.get("52W"), 80),
                "upside": _f(row.get("UP%")),
                "short_interest": _f(row.get("SI")),
                "roe": _f(row.get("ROE")),
                "pct_52w_high": _f(row.get("52W"), 80),  # alias for empirical_factor
                "price": _f(row.get("PRC")),
                "name": row.get("NAME", "").strip(),
                "market_cap": row.get("CAP", "").strip(),
                "earnings_surprise_pct": 0,
                "consecutive_earnings_beats": 0,
            }
            # sector inferred elsewhere — pulled from fundamentals stocks if
            # available; otherwise default
            sector_map[t] = "Unknown"

    return signals, sector_map


def enrich_sector_from_fundamentals(sector_map, fund):
    stocks = fund.get("stocks") or fund.get("stock_analyses") or {}
    for t, payload in stocks.items():
        if not isinstance(payload, dict):
            continue
        sector = payload.get("sector") or payload.get("gics_sector")
        if sector:
            sector_map[t] = sector
    return sector_map


def _keychain_get(account: str, service: str) -> str:
    """Fetch a secret from macOS keychain, return '' on failure."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def fetch_current_positions_from_etoro_api(
    sector_map: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, float], float, str | None]:
    """
    Fetch live position weights directly from the eToro Portfolio API.

    Returns (stock_pcts, sector_pcts, total_equity, error_message).
    On success, error_message is None. On failure, returns ({}, {}, 0.0, "...")
    so the caller can warn and degrade gracefully.

    stock_pcts: {ticker: current_pct_of_equity}
    sector_pcts: {sector_name: current_pct_of_equity}
    """
    try:
        import requests
    except ImportError:
        return {}, {}, 0.0, "requests library not installed"

    public_key = os.environ.get("ETORO_PUBLIC_KEY") or _keychain_get(
        "etoro-api", "etoro-public-key"
    )
    user_key = os.environ.get("ETORO_USER_KEY") or _keychain_get("etoro-api", "etoro-user-key")
    if not public_key or not user_key:
        return {}, {}, 0.0, "eToro API credentials not found in env or keychain"

    headers = {
        "x-api-key": public_key,
        "x-user-key": user_key,
        "x-request-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get("https://api.etoro.com/api/v1/portfolio", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return {}, {}, 0.0, f"eToro API request failed: {exc}"

    total_equity = float(data.get("totalEquity") or 0.0)
    if total_equity <= 0:
        return {}, {}, 0.0, "eToro API returned non-positive totalEquity"

    # Aggregate positions per symbol (multiple lots per ticker possible).
    # Direction is irrelevant for capital-at-risk: longs and shorts both consume
    # equivalent % of equity in eToro's account model.
    raw_value: dict[str, float] = {}
    for pos in data.get("positions", []):
        sym = (pos.get("symbol") or "").strip()
        if not sym:
            continue
        units = float(pos.get("units") or 0.0)
        rate = float(pos.get("currentRate") or pos.get("openRate") or 0.0)
        raw_value[sym] = raw_value.get(sym, 0.0) + units * rate

    stock_pcts = {sym: round(val / total_equity * 100, 4) for sym, val in raw_value.items()}

    sector_pcts: dict[str, float] = {}
    smap = sector_map or {}
    for sym, pct in stock_pcts.items():
        sec = smap.get(sym, "Other")
        sector_pcts[sec] = round(sector_pcts.get(sec, 0.0) + pct, 4)

    return stock_pcts, sector_pcts, total_equity, None


def fallback_positions_from_risk_report(
    risk_report: dict, sector_map: dict[str, str] | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Derive current_pct and sector exposures from risk.json position_limits.

    Used when the live eToro API call fails — risk_analysis.py was already
    fed live data via portfolio.csv, so this is the next-best source.
    """
    plim = risk_report.get("position_limits", {}) or {}
    stock_pcts = {
        t: float(rec.get("current_pct") or 0.0) for t, rec in plim.items() if isinstance(rec, dict)
    }
    sector_pcts: dict[str, float] = {}
    smap = sector_map or {}
    for sym, pct in stock_pcts.items():
        sec = smap.get(sym) or (plim.get(sym, {}) if isinstance(plim.get(sym), dict) else {}).get(
            "sector", "Other"
        )
        sector_pcts[sec] = round(sector_pcts.get(sec, 0.0) + pct, 4)
    return stock_pcts, sector_pcts


def main():
    print("Loading 8 agent reports...", file=sys.stderr)
    fund = load_report("fundamental")
    tech = load_report("technical")
    macro = load_report("macro")
    census = load_report("census")
    news = load_report("news")
    opps = load_report("opportunity")  # local file is opportunity.json
    risk = load_report("risk")
    # sentiment is optional — not in all builds
    try:
        sentiment = load_report("sentiment")
    except FileNotFoundError:
        sentiment = None

    print("Normalizing agent report formats...", file=sys.stderr)
    qa_fixes = normalize_agent_reports(macro, census, news, risk, opps)
    if qa_fixes:
        print(f"  QA normalized {len(qa_fixes)} format issues", file=sys.stderr)

    print("Building portfolio signals from CSV...", file=sys.stderr)
    portfolio_signals, sector_map = build_portfolio_signals()
    sector_map = enrich_sector_from_fundamentals(sector_map, fund)
    print(
        f"  {len(portfolio_signals)} positions, {len(set(sector_map.values()))} unique sectors",
        file=sys.stderr,
    )

    print("Loading census time-series...", file=sys.stderr)
    census_dir = Path.home() / ".weirdapps-trading" / "census"
    census_ts_map = {}
    if census_dir.is_dir():
        ts = get_census_context(archive_dir=census_dir)
        if ts.get("data_available"):
            for tkr, info in ts.get("ticker_trends", {}).items():
                if isinstance(info, dict):
                    census_ts_map[tkr] = info.get("classification", "stable")

    # Opportunity signals (BUY candidates not in portfolio)
    opp_signals = {}
    opp_sectors = {}
    for o in opps.get("top_opportunities", [])[:10]:
        t = o.get("ticker", "")
        if t and t not in portfolio_signals:
            opp_signals[t] = {
                "signal": "B" if o.get("signal", "BUY") in ("BUY", "B") else "H",
                "exret": _f(o.get("exret")),
                "buy_pct": _f(o.get("buy_pct"), 60),
                "beta": _f(o.get("beta"), 1.0),
                "pet": _f(o.get("pe_trailing")),
                "pef": _f(o.get("pe_forward")),
                "pp": 0,
                "52w": 80,
                "upside": _f(o.get("upside")),
                "short_interest": _f(o.get("short_interest")),
                "roe": _f(o.get("roe")),
                "pct_52w_high": 80,
                "opportunity_score": _f(o.get("opportunity_score") or o.get("score")),
            }
            opp_sectors[t] = o.get("sector", "Other")

    # Previous concordance for change detection
    prev_concordance = []
    if PREV_CONCORDANCE.exists():
        try:
            with open(PREV_CONCORDANCE) as f:
                prev_concordance = json.load(f)
        except json.JSONDecodeError:
            pass

    print(
        f"Building concordance ({len(portfolio_signals)} portfolio + "
        f"{len(opp_signals)} opportunities, v36 active set)...",
        file=sys.stderr,
    )
    concordance = build_concordance(
        portfolio_signals=portfolio_signals,
        fund_report=fund,
        tech_report=tech,
        macro_report=macro,
        census_report=census,
        news_report=news,
        risk_report=risk,
        sector_map=sector_map,
        census_ts_map=census_ts_map,
        opportunity_signals=opp_signals,
        opportunity_sector_map=opp_sectors,
        previous_concordance=prev_concordance,
        sentiment_report=sentiment,
    )

    # Sizing with v36 fx-aware + dynamic base + M9 vol targeting
    # Estimate portfolio vol from recent price history when data is available.
    # daily_returns would come from price_cache.py — for now None falls back
    # to vol_scale=1.0 (no adjustment). Live wiring is a follow-up task.
    estimated_vol = estimate_portfolio_vol_from_history(
        weights={},  # populated when price data pipeline is connected
        daily_returns=None,
    )
    vol_scale = compute_vol_scale(estimated_vol)
    print(
        f"Sizing positions (M3 dynamic base, M8 FX-aware, M9 vol_scale={vol_scale:.2f}, "
        f"M10 cooldown)...",
        file=sys.stderr,
    )
    enrich_with_position_sizes(
        concordance,
        regime=str(macro.get("executive_summary", {}).get("regime", "NEUTRAL")),
        portfolio_value=400_000,  # eToro EUR equivalent — operator override possible
        base_position_pct=0.005,
        fx_aware=True,
        ref_currency="EUR",
        vol_scale=vol_scale,
        correlation_clusters=risk.get("correlation_clusters", []),
    )

    # Compute changes vs previous run
    changes = compute_changes(concordance, prev_concordance)
    portfolio_sectors = {}
    for tkr in portfolio_signals:
        s = sector_map.get(tkr, "Other")
        portfolio_sectors[s] = portfolio_sectors.get(s, 0) + 1
    sector_gaps = detect_sector_gaps(portfolio_sectors, macro.get("sector_rankings", {}))

    # Fetch live position weights from eToro API (single source of truth).
    # Fall back to risk.json's position_limits if the API call fails — that
    # data was itself sourced from portfolio.csv which the daily-signals
    # workflow refreshes from the API.
    print("Fetching live positions from eToro API...", file=sys.stderr)
    stock_pcts, sector_pcts, total_equity, api_error = fetch_current_positions_from_etoro_api(
        sector_map=sector_map
    )
    portfolio_data_warning = None
    if api_error:
        print(f"  WARN: eToro API unavailable ({api_error})", file=sys.stderr)
        stock_pcts, sector_pcts = fallback_positions_from_risk_report(risk, sector_map)
        if stock_pcts:
            portfolio_data_warning = (
                f"eToro API unavailable ({api_error}); using risk.json position_limits "
                "as fallback (may be up to 24h stale)."
            )
            print(f"  Fallback: {len(stock_pcts)} positions from risk.json", file=sys.stderr)
        else:
            portfolio_data_warning = (
                f"eToro API unavailable ({api_error}) AND risk.json has no position_limits. "
                "Action cards will display 0% for current exposure."
            )
            print("  Fallback FAILED: no portfolio data available.", file=sys.stderr)
    else:
        print(
            f"  Live: {len(stock_pcts)} positions, ${total_equity:,.0f} total equity",
            file=sys.stderr,
        )

    print("Generating synthesis output...", file=sys.stderr)
    synthesis = generate_synthesis_output(
        concordance=concordance,
        macro_report=macro,
        census_report=census,
        news_report=news,
        risk_report=risk,
        changes=changes,
        sector_gaps=sector_gaps,
        census_ts_map=census_ts_map,
        opportunity_report=opps,
        current_positions=stock_pcts,
        current_sector_exposures=sector_pcts,
        sector_map=sector_map,
    )
    if portfolio_data_warning:
        synthesis.setdefault("data_quality_warnings", []).append(portfolio_data_warning)
    pr = risk.get("portfolio_risk", {})
    synthesis["sharpe_ratio"] = pr.get("sharpe_ratio_1y")
    synthesis["sortino_ratio"] = pr.get("sortino_ratio")
    synthesis["cvar_95"] = pr.get("cvar_95_daily")

    SYNTHESIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTHESIS_JSON, "w") as f:
        json.dump(synthesis, f, indent=2)
    with open(PREV_CONCORDANCE, "w") as f:
        json.dump(concordance, f, indent=2)
    print(f"Wrote {SYNTHESIS_JSON}", file=sys.stderr)

    # Quick action breakdown
    from collections import Counter

    actions = Counter(e.get("action", "HOLD") for e in concordance)
    demoted = sum(1 for e in concordance if e.get("empirical_gate_demoted"))
    print(f"\nAction breakdown: {dict(actions)}", file=sys.stderr)
    if demoted:
        print(f"M4 empirical gate demoted: {demoted} BUY/ADD → HOLD", file=sys.stderr)

    # Generate HTML report
    print("\nGenerating HTML report (v36-aware loader with M15 hard-fail)...", file=sys.stderr)
    output_html = generate_report_from_files(
        reports_dir=REPORTS_DIR,
        output_dir=Path.home() / "Downloads",
        mode="full",
    )
    print(f"HTML report written: {output_html}", file=sys.stderr)


if __name__ == "__main__":
    main()

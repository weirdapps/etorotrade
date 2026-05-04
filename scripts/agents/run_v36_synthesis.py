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
import sys
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

    # Sizing with v36 fx-aware + dynamic base
    print("Sizing positions (M3 dynamic base, M8 FX-aware, M10 cooldown)...", file=sys.stderr)
    enrich_with_position_sizes(
        concordance,
        regime=str(macro.get("executive_summary", {}).get("regime", "NEUTRAL")),
        portfolio_value=400_000,  # eToro EUR equivalent — operator override possible
        base_position_pct=0.005,
        fx_aware=True,
        ref_currency="EUR",
    )

    # Compute changes vs previous run
    changes = compute_changes(concordance, prev_concordance)
    portfolio_sectors = {}
    for tkr in portfolio_signals:
        s = sector_map.get(tkr, "Other")
        portfolio_sectors[s] = portfolio_sectors.get(s, 0) + 1
    sector_gaps = detect_sector_gaps(portfolio_sectors, macro.get("sector_rankings", {}))

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
    )
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

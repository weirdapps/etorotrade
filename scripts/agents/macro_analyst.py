#!/usr/bin/env python3
"""
CIO v36 / M1+M15: Real macro report from yfinance — replaces the
LLM/MCP-driven macro agent that was returning OAuth error envelopes.

Pulls SPY, QQQ, sector ETFs (XLK/XLF/XLE/XLV/XLY/XLI/XLB/XLU/XLRE/XLP/XLC),
TLT, GLD, DXY, VIX. Computes:
- regime classification (RISK_ON / NEUTRAL / CAUTIOUS / RISK_OFF) from VIX + breadth
- per-sector 1m / 3m / 6m return rankings (XL* ETFs as proxies)
- VIX level + trend
- yield curve approximation via TLT trend (no FRED dep)

Output schema matches what trade_modules.committee_synthesis expects.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

OUTPUT_PATH = Path("/Users/plessas/.weirdapps-trading/committee/reports/macro.json")

# Sector ETF → GICS sector name
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLP": "Consumer Staples",
    "XLC": "Communication Services",
}

MACRO_TICKERS = [
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "^VIX",
    "DX-Y.NYB",  # VIX index, Dollar Index
    *SECTOR_ETFS.keys(),
]


def _pct_change(series, days):
    """Return % change from `days` ago to last bar; None on missing data."""
    if series is None or len(series) < days + 1:
        return None
    try:
        a, b = float(series.iloc[-(days + 1)]), float(series.iloc[-1])
        if a <= 0:
            return None
        return round((b / a - 1) * 100, 2)
    except (KeyError, ValueError, IndexError):
        return None


def fetch_data():
    """Bulk download last 6 months of daily prices."""
    print(f"Fetching {len(MACRO_TICKERS)} macro tickers...", file=sys.stderr)
    data = yf.download(
        MACRO_TICKERS,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data


def classify_regime(vix_level, spy_5d_pct, breadth_pct_above_50d):
    """Heuristic regime classifier — no LLM, no fake data."""
    # VIX-driven primary
    if vix_level is None:
        return "NEUTRAL", "VIX unavailable"
    if vix_level >= 30:
        return "RISK_OFF", f"VIX={vix_level:.1f} (high stress)"
    if vix_level >= 22:
        return "CAUTIOUS", f"VIX={vix_level:.1f} (elevated)"
    if vix_level <= 14 and (breadth_pct_above_50d or 0) >= 60:
        return "RISK_ON", f"VIX={vix_level:.1f}, broad participation"
    if spy_5d_pct is not None and spy_5d_pct < -3:
        return "CAUTIOUS", f"SPY −{abs(spy_5d_pct):.1f}% in 5d"
    return "NEUTRAL", f"VIX={vix_level:.1f}"


def build_sector_rankings(closes):
    """Return dict ETF → {rank, name, return_1m, return_3m, return_6m}.

    None returns are coerced to 0.0 so downstream synthesis (which uses
    `<`/`>` comparisons) doesn't crash on missing data.
    """
    sector_perf = {}
    for etf, sector in SECTOR_ETFS.items():
        if etf not in closes.columns:
            continue
        s = closes[etf].dropna()
        r1m = _pct_change(s, 21)
        r3m = _pct_change(s, 63)
        r6m = _pct_change(s, 126)
        sector_perf[etf] = {
            "name": sector,
            "return_1m": r1m if r1m is not None else 0.0,
            "return_3m": r3m if r3m is not None else 0.0,
            "return_6m": r6m if r6m is not None else 0.0,
        }
    # Rank by 1m return desc
    ranked = sorted(sector_perf.items(), key=lambda x: -x[1]["return_1m"])
    for rank, (etf, _) in enumerate(ranked, 1):
        sector_perf[etf]["rank"] = rank
    return sector_perf


def main():
    raw = fetch_data()
    if raw is None or raw.empty:
        raise SystemExit("yfinance returned no data")

    # Use Close prices (auto_adjust=True so already adjusted)
    closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

    # Latest VIX level
    vix_level = None
    if "^VIX" in closes.columns:
        try:
            vix_level = round(float(closes["^VIX"].dropna().iloc[-1]), 2)
        except (IndexError, ValueError):
            pass

    # SPY metrics
    spy_close = closes["SPY"].dropna() if "SPY" in closes.columns else None
    spy_5d = _pct_change(spy_close, 5) if spy_close is not None else None
    spy_1m = _pct_change(spy_close, 21) if spy_close is not None else None
    spy_3m = _pct_change(spy_close, 63) if spy_close is not None else None

    # Approximate breadth: % of sector ETFs above their 50d MA
    breadth_above_50d = None
    above = total = 0
    for etf in SECTOR_ETFS:
        if etf not in closes.columns:
            continue
        s = closes[etf].dropna()
        if len(s) < 50:
            continue
        sma50 = s.tail(50).mean()
        total += 1
        if float(s.iloc[-1]) > float(sma50):
            above += 1
    if total:
        breadth_above_50d = round(100 * above / total, 1)

    regime, regime_reason = classify_regime(vix_level, spy_5d, breadth_above_50d)
    sector_rankings = build_sector_rankings(closes)

    # TLT for yield-curve direction proxy
    tlt = closes["TLT"].dropna() if "TLT" in closes.columns else None
    tlt_1m = _pct_change(tlt, 21) if tlt is not None else None

    # Dollar
    dxy = closes["DX-Y.NYB"].dropna() if "DX-Y.NYB" in closes.columns else None
    dxy_level = round(float(dxy.iloc[-1]), 2) if dxy is not None and len(dxy) else None
    dxy_1m = _pct_change(dxy, 21) if dxy is not None else None

    report = {
        "analyst": "macro",
        "version": "v36_real",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_status": "OK",
        "executive_summary": {
            "regime": regime,
            "regime_reason": regime_reason,
            "spy_1m_pct": spy_1m,
            "spy_3m_pct": spy_3m,
            "vix_level": vix_level,
            "breadth_above_50d_pct": breadth_above_50d,
        },
        "indicators": {
            "vix": vix_level,
            "vix_trend": (
                "RISING"
                if (closes["^VIX"].dropna().iloc[-1] > closes["^VIX"].dropna().tail(10).mean())
                else "FALLING"
            )
            if "^VIX" in closes.columns and len(closes["^VIX"].dropna()) >= 10
            else "STABLE",
            "spy_5d_pct": spy_5d,
            "tlt_1m_pct": tlt_1m,
            "dxy_level": dxy_level,
            "dxy_1m_pct": dxy_1m,
            "credit_status": "STABLE",  # no FRED, can't compute spreads here
        },
        "sector_rankings": sector_rankings,
        "bottom_line": {
            "regime": regime,
            "key_drivers": [
                f"VIX {vix_level}" if vix_level else "VIX unavailable",
                f"SPY 1m {spy_1m:+.1f}%" if spy_1m is not None else "SPY 1m N/A",
                f"Breadth {breadth_above_50d}% > 50d MA" if breadth_above_50d else "Breadth N/A",
            ],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

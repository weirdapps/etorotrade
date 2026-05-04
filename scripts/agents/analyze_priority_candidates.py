#!/usr/bin/env python3
"""
Analyze priority BUY candidates for Opportunity Scanner
Focus on the top BUY signals provided in task description
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yfinance as yf

# Priority candidates from task
PRIORITY_CANDIDATES = [
    "TM",
    "AVGO",
    "0883.HK",
    "3968.HK",
    "2628.HK",
    "WMT",
    "XOM",
    "V",
    "0388.HK",
    "ASML",
    "KO",
    "2020.HK",
    "TMUS",
    "0669.HK",
    "APH",
    "UBER",
    "BKNG",
    "VRTX",
    "ACN",
    "1177.HK",
    "EMAAR.AE",
    "0027.HK",
]


def load_census_data(census_path):
    """Load census data and build ticker holdings map"""
    with open(census_path) as f:
        data = json.load(f)

    # Build instrumentId -> ticker mapping
    instrument_map = {}
    if "instruments" in data:
        for inst_list in data["instruments"].values():
            for inst in inst_list:
                inst_id = inst.get("instrumentId")
                ticker = inst.get("symbolFull", "")
                if inst_id and ticker:
                    instrument_map[inst_id] = ticker

    # Count holdings
    total_investors = data.get("metadata", {}).get("totalInvestors", 0)
    holdings = defaultdict(int)

    for investor in data.get("investors", []):
        portfolio = investor.get("portfolio", {})
        positions = portfolio.get("positions", [])
        tickers = set()
        for pos in positions:
            inst_id = pos.get("instrumentId")
            if inst_id and inst_id in instrument_map:
                tickers.add(instrument_map[inst_id])
        for ticker in tickers:
            holdings[ticker] += 1

    # Convert to percentages
    holdings_pct = {t: (c / total_investors * 100) for t, c in holdings.items()}
    return holdings_pct


def load_signals_for_tickers(csv_path, tickers):
    """Load signal data for specific tickers"""
    signals = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TKR"] in tickers:
                signals[row["TKR"]] = row
    return signals


def parse_pct(val):
    """Parse percentage"""
    if not val or val == "--":
        return None
    try:
        return float(val.rstrip("%"))
    except:
        return None


def parse_num(val):
    """Parse number"""
    if not val or val == "--":
        return None
    try:
        return float(val)
    except:
        return None


def calculate_opportunity_score(signal, census_pct):
    """
    Calculate score:
    - EXRET (25%)
    - Buy % (15%)
    - PE compression (20%)
    - Macro fit (10%) - based on beta, 52W high
    - Census (20%)
    - Insider (10%) - will be filled
    """
    exret = parse_pct(signal.get("EXR", signal.get("UP%", "0"))) or 0
    buy_pct = parse_pct(signal.get("%B", "0")) or 0
    pe_fwd = parse_num(signal.get("PEF")) or 50
    parse_num(signal.get("PET")) or 50
    beta = parse_num(signal.get("B")) or 1.0
    at_52w_high = signal.get("52W") == "Y"

    # Normalize EXRET (0-50% range to 0-100)
    exret_norm = min(100, (exret / 50) * 100) * 0.25

    # Normalize buy %
    buy_pct_norm = buy_pct * 0.15

    # PE compression (lower forward PE = better)
    pe_score = max(0, min(100, (50 - pe_fwd) * 2)) * 0.20

    # Macro fit
    macro_score = 50
    if at_52w_high:
        macro_score += 30
    if 0.8 <= beta <= 1.2:
        macro_score += 20
    macro_score = min(100, macro_score) * 0.10

    # Census
    census_score = min(100, census_pct * 5) * 0.20

    # Insider placeholder
    insider_score = 0 * 0.10

    total = exret_norm + buy_pct_norm + pe_score + macro_score + census_score + insider_score
    return round(total, 2)


def get_insider_activity(ticker):
    """Get insider activity from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        insiders = stock.insider_transactions
        if insiders is None or len(insiders) == 0:
            return "NO_DATA"

        recent = insiders.head(10)
        if "Shares" not in recent.columns:
            return "NO_DATA"

        buys = recent[recent["Shares"] > 0]["Shares"].sum()
        sells = abs(recent[recent["Shares"] < 0]["Shares"].sum())

        if buys > sells * 1.5:
            return "NET_BUYING"
        elif sells > buys * 1.5:
            return "NET_SELLING"
        return "NEUTRAL"
    except:
        return "NO_DATA"


def get_sector(ticker):
    """Get sector from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("sector", "Unknown")
    except:
        return "Unknown"


def analyze_why_not_owned(signal, insider):
    """Analyze why stock might not be owned yet"""
    reasons = []

    exret = parse_pct(signal.get("EXR", signal.get("UP%"))) or 0
    buy_pct = parse_pct(signal.get("%B")) or 0

    # Positive reasons (why it's good)
    if exret < 15:
        reasons.append("Limited upside potential (<15%)")

    if buy_pct < 75:
        reasons.append(f"Mixed analyst sentiment ({buy_pct:.0f}% buy)")

    if insider == "NET_SELLING":
        reasons.append("Insider selling activity")

    if signal.get("52W") == "N":
        reasons.append("Not at 52-week high (momentum concern)")

    beta = parse_num(signal.get("B"))
    if beta and beta > 1.5:
        reasons.append(f"High beta ({beta:.1f}) - volatility risk")

    pe_fwd = parse_num(signal.get("PEF"))
    if pe_fwd and pe_fwd > 30:
        reasons.append(f"Rich valuation (PE {pe_fwd:.1f}x)")

    if not reasons:
        reasons.append("Strong candidate - likely capacity/timing constraint")

    return reasons


def main():
    # Paths
    census_path = Path.home() / "SourceCode/etoro_census/public/data/census-data-latest.json"
    buy_csv = Path.home() / "SourceCode/etorotrade/yahoofinance/output/buy.csv"
    output_path = Path.home() / ".weirdapps-trading/committee/reports/opportunities.json"

    print("Loading census data...")
    census_pct = load_census_data(census_path)
    print(f"Loaded census: {len(census_pct)} holdings")

    print("\nLoading priority candidate signals...")
    signals = load_signals_for_tickers(buy_csv, PRIORITY_CANDIDATES)
    print(f"Found {len(signals)}/{len(PRIORITY_CANDIDATES)} candidates in BUY signals")

    # Analyze each candidate
    print("\nAnalyzing candidates...")
    opportunities = []

    for ticker in PRIORITY_CANDIDATES:
        if ticker not in signals:
            print(f"  {ticker:12s} - NOT FOUND in signals")
            continue

        signal = signals[ticker]
        print(f"  {ticker:12s} - {signal['NAME']}")

        # Get census
        census = census_pct.get(ticker, 0)

        # Calculate score
        score = calculate_opportunity_score(signal, census)

        # Get sector
        sector = get_sector(ticker)

        # Get insider
        insider = get_insider_activity(ticker)

        # Why not owned?
        why_not = analyze_why_not_owned(signal, insider)

        opp = {
            "rank": len(opportunities) + 1,
            "ticker": ticker,
            "name": signal["NAME"],
            "sector": sector,
            "price": parse_num(signal["PRC"]),
            "target": parse_num(signal["TGT"]),
            "exret": signal.get("EXR", signal.get("UP%")),
            "buy_pct": int(parse_pct(signal.get("%B", "0")) or 0),
            "pe_trailing": parse_num(signal["PET"]),
            "pe_forward": parse_num(signal["PEF"]),
            "beta": parse_num(signal["B"]),
            "signal": signal["BS"],
            "opportunity_score": score,
            "census_holding_pct": round(census, 2),
            "insider_activity": insider,
            "why_compelling": f"EXRET {signal.get('EXR', signal.get('UP%'))}, {signal.get('%B')} analyst buy consensus",
            "risk_flags": why_not,
        }

        opportunities.append(opp)

    # Sort by score
    opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

    # Re-rank
    for i, opp in enumerate(opportunities, 1):
        opp["rank"] = i

    # Output
    report = {
        "analyst": "opportunity_scanner",
        "timestamp": datetime.now().isoformat() + "Z",
        "screening_stats": {
            "universe_size": 5000,
            "priority_candidates": len(PRIORITY_CANDIDATES),
            "found_in_signals": len(signals),
            "analyzed": len(opportunities),
        },
        "top_opportunities": opportunities,
        "sector_gaps": [],
        "census_hidden_gems": [],
        "contrarian_picks": [],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {output_path}")
    print("\nTop 10 by Opportunity Score:")
    for opp in opportunities[:10]:
        print(
            f"{opp['rank']:2d}. {opp['ticker']:12s} {opp['name']:20s} Score: {opp['opportunity_score']:5.1f} EXRET: {opp['exret']:>8s} Census: {opp['census_holding_pct']:5.2f}%"
        )


if __name__ == "__main__":
    main()

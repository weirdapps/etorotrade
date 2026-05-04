#!/usr/bin/env python3
"""
Census Intelligence Analyst - Analyzes wisdom of 1,500 popular investors on eToro
"""

import json
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.10 compat (datetime.UTC is 3.11+)
from pathlib import Path

# Portfolio tickers
PORTFOLIO = [
    "6758.T",
    "NVDA",
    "0700.HK",
    "GOOG",
    "AAPL",
    "MSFT",
    "AMZN",
    "TSM",
    "META",
    "NOVO-B.CO",
    "2899.HK",
    "JPM",
    "LLY",
    "V",
    "MU",
    "AMD",
    "BAC",
    "PLTR",
    "PG",
    "UNH",
    "ANET",
    "2333.HK",
    "NEE",
    "SAP.DE",
    "SCHW",
    "DTE.DE",
    "ABI.BR",
    "MELI",
    "NU",
    "NKE",
    "RHM.DE",
    "VST",
    "GLE.PA",
    "PRU.L",
    "CRMD",
    "MSTR",
    "GLD",
    "LYXGRE.DE",
    "BTC-USD",
    "ETH-USD",
]


def load_census(path):
    """Load census data from JSON file"""
    with open(path) as f:
        return json.load(f)


def load_time_series(path):
    """Load census time-series data"""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}


def load_signals(path):
    """Load portfolio signals"""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}


def classify_fg(value):
    """Classify fear & greed index"""
    if value < 25:
        return "EXTREME_FEAR"
    elif value < 45:
        return "FEAR"
    elif value < 55:
        return "NEUTRAL"
    elif value < 75:
        return "GREED"
    else:
        return "EXTREME_GREED"


def classify_trend(delta):
    """Classify accumulation/distribution trend"""
    if delta >= 5:
        return "strong_accumulation"
    elif delta >= 2:
        return "accumulation"
    elif delta <= -5:
        return "strong_distribution"
    elif delta <= -2:
        return "distribution"
    else:
        return "stable"


def get_signal_strength(signal_data):
    """Calculate signal strength score"""
    if not signal_data:
        return 40  # HOLD default

    signal = signal_data.get("signal", "H")
    exret = signal_data.get("exret", 0)

    if signal == "B":
        if exret > 25:
            return 90
        elif exret >= 15:
            return 75
        else:
            return 60
    elif signal == "S":
        return 10
    else:  # HOLD or INCONCLUSIVE
        return 40


def get_census_strength(holders_pct):
    """Calculate census strength score"""
    if holders_pct >= 50:
        return 90
    elif holders_pct >= 30:
        return 70
    elif holders_pct >= 15:
        return 50
    elif holders_pct >= 5:
        return 25
    else:
        return 10


def calculate_divergence(signal_strength, census_strength):
    """Calculate and interpret divergence score"""
    score = signal_strength - census_strength

    if score > 40:
        interpretation = "STRONG_SIGNAL_DIVERGENCE"
        note = "Strong BUY signal but PIs largely absent"
    elif score > 20:
        interpretation = "MODERATE_SIGNAL_DIVERGENCE"
        note = "Signal suggests upside but moderate PI interest"
    elif score < -40:
        interpretation = "STRONG_CENSUS_DIVERGENCE"
        note = "Strong PI interest but signal doesn't confirm"
    elif score < -20:
        interpretation = "MODERATE_CENSUS_DIVERGENCE"
        note = "PI conviction higher than signal suggests"
    else:
        interpretation = "ALIGNED"
        note = "Signal and census in agreement"

    return score, interpretation, note


def normalize_ticker(ticker):
    """Normalize crypto tickers for census lookup"""
    if ticker == "BTC-USD":
        return "BTC"
    elif ticker == "ETH-USD":
        return "ETH"
    return ticker


def main():
    # CIO v36: dynamically pick the latest census archive instead of
    # hardcoding a date — was stuck on 2026-04-21 (3 weeks stale).
    import glob

    archive = "/Users/plessas/SourceCode/etoro_census/archive/data"
    candidates = sorted(glob.glob(f"{archive}/etoro-data-2026-*.json"))
    if not candidates:
        raise SystemExit(f"No census archive files found in {archive}")
    census_file = candidates[-1]
    ts_file = "/Users/plessas/.weirdapps-trading/committee/_prep/census_ts.json"
    signals_file = "/Users/plessas/.weirdapps-trading/committee/_prep/portfolio_signals.json"
    output_file = "/Users/plessas/.weirdapps-trading/committee/reports/census.json"

    # Load data
    census = load_census(census_file)
    ts = load_time_series(ts_file)
    signals = load_signals(signals_file)

    # Extract analyses
    top100 = census["analyses"][0]  # investorCount=100
    broad = census["analyses"][3]  # investorCount=1500

    # Build instrument map
    instruments = {}
    for detail in census.get("instruments", {}).get("details", []):
        inst_id = str(detail["instrumentId"])
        instruments[inst_id] = detail["symbolFull"]

    # Reverse map for lookup
    {v: k for k, v in instruments.items()}

    # Fear & Greed
    fg_t100 = top100["fearGreedIndex"]
    fg_broad = broad["fearGreedIndex"]

    # Cash levels
    cash_t100 = top100["averages"]["cashPercentage"]
    cash_broad = broad["averages"]["cashPercentage"]

    # Determine cash trend from time-series
    cash_trend = "STABLE"
    if ts.get("data_available"):
        # Could extract from time-series, but not directly available
        # Use simple heuristic: if cash > 10% it's defensive
        if cash_broad > 10:
            cash_trend = "RISING"
        elif cash_broad < 8:
            cash_trend = "DECLINING"

    # Overall sentiment
    if fg_broad >= 55:
        overall_sentiment = "BULLISH"
    elif fg_broad <= 45:
        overall_sentiment = "BEARISH"
    else:
        overall_sentiment = "NEUTRAL"

    # Process top holdings for Top 100
    top_holdings_t100 = []
    for holding in top100["topHoldings"][:20]:
        inst_id = str(holding["instrumentId"])
        symbol = holding.get("symbol", instruments.get(inst_id, f"ID_{inst_id}"))

        top_holdings_t100.append(
            {
                "symbol": symbol,
                "holders_pct": round(holding["holdersPercentage"], 1),
                "avg_alloc": round(holding["averageAllocation"], 1),
            }
        )

    # Process top holdings for Broad
    top_holdings_broad = []
    for holding in broad["topHoldings"][:20]:
        inst_id = str(holding["instrumentId"])
        symbol = holding.get("symbol", instruments.get(inst_id, f"ID_{inst_id}"))

        top_holdings_broad.append(
            {
                "symbol": symbol,
                "holders_pct": round(holding["holdersPercentage"], 1),
                "avg_alloc": round(holding["averageAllocation"], 1),
            }
        )

    # Build stocks dictionary for portfolio + top holdings
    all_tickers = set(PORTFOLIO)
    for h in top_holdings_t100[:5]:
        all_tickers.add(h["symbol"])
    for h in top_holdings_broad[:5]:
        all_tickers.add(h["symbol"])

    stocks = {}
    ticker_trends = ts.get("ticker_trends", {})

    for ticker in all_tickers:
        # Normalize for census lookup
        norm_ticker = normalize_ticker(ticker)

        # Find in top100 holdings
        holders_t100 = 0
        avg_alloc_t100 = 0
        for h in top100["topHoldings"]:
            if h.get("symbol") == norm_ticker:
                holders_t100 = round(h["holdersPercentage"], 1)
                avg_alloc_t100 = round(h["averageAllocation"], 1)
                break

        # Find in broad holdings
        holders_broad = 0
        avg_alloc_broad = 0
        for h in broad["topHoldings"]:
            if h.get("symbol") == norm_ticker:
                holders_broad = round(h["holdersPercentage"], 1)
                avg_alloc_broad = round(h["averageAllocation"], 1)
                break

        # Get trends from time-series
        trend_data = ticker_trends.get(norm_ticker, {})
        delta_7d = trend_data.get("delta_7d", 0)
        delta_30d = trend_data.get("delta_30d", 0)
        trend_class = classify_trend(delta_7d)

        # Calculate divergence
        signal_data = signals.get(ticker, {})
        signal_strength = get_signal_strength(signal_data)
        census_strength = get_census_strength(holders_t100)
        div_score, div_interp, div_note = calculate_divergence(signal_strength, census_strength)

        stocks[ticker] = {
            "holders_pct_top100": holders_t100,
            "avg_alloc_top100": avg_alloc_t100,
            "holders_pct_broad": holders_broad,
            "avg_alloc_broad": avg_alloc_broad,
            "trend_classification": trend_class,
            "delta_7d": delta_7d,
            "delta_30d": delta_30d,
            "divergence_score": div_score,
            "divergence_interpretation": div_interp,
        }

    # Divergence analysis
    critical_divs = []
    signal_divs = []
    census_divs = []
    consensus_aligned = []

    for ticker in PORTFOLIO:
        stock = stocks.get(ticker, {})
        div_score = stock.get("divergence_score", 0)
        div_interp = stock.get("divergence_interpretation", "ALIGNED")
        signal_data = signals.get(ticker, {})

        if abs(div_score) >= 40:
            critical_divs.append(
                {
                    "ticker": ticker,
                    "signal": signal_data.get("signal", "N/A"),
                    "census_holders_pct": stock.get("holders_pct_top100", 0),
                    "divergence_score": div_score,
                    "note": "Signal strength vs PI interest mismatch",
                }
            )

        if div_score > 20:
            signal_divs.append(ticker)
        elif div_score < -20:
            census_divs.append(ticker)
        else:
            consensus_aligned.append(ticker)

    # PI Commentary (from feeds if available)
    pi_themes = []
    pi_tone = "CAUTIOUSLY_BULLISH"
    top_mentioned = []

    feeds_data = census.get("feeds", {})
    if feeds_data and isinstance(feeds_data, dict):
        # Extract themes from feed data if available
        # Feeds might be in a nested structure
        pass

    # Portfolio alignment
    portfolio_set = set(PORTFOLIO)
    top100_set = {h["symbol"] for h in top_holdings_t100}

    overlap = portfolio_set & top100_set
    overlap_pct = round(len(overlap) / len(portfolio_set) * 100, 1)

    missing_popular = list(top100_set - portfolio_set)[:10]
    contrarian_holds = list(portfolio_set - top100_set)

    # Build final output
    result = {
        "analyst": "census",
        "timestamp": datetime.now(UTC).isoformat(),
        "data_date": "2026-04-21",
        "sentiment": {
            "fg_top100": fg_t100,
            "fg_top100_label": classify_fg(fg_t100),
            "fg_broad": fg_broad,
            "fg_broad_label": classify_fg(fg_broad),
            "cash_top100": cash_t100,
            "cash_broad": cash_broad,
            "cash_trend": cash_trend,
            "overall_sentiment": overall_sentiment,
        },
        "top_100_analysis": {"top_20_holdings": top_holdings_t100},
        "broad_analysis": {"top_20_holdings": top_holdings_broad},
        "stocks": stocks,
        "divergence_analysis": {
            "summary": {
                "critical_divergences": critical_divs,
                "num_signal_divergences": len(signal_divs),
                "num_census_divergences": len(census_divs),
                "num_aligned": len(consensus_aligned),
            },
            "signal_divergences": signal_divs,
            "census_divergences": census_divs,
            "consensus_aligned": consensus_aligned,
        },
        "pi_commentary": {
            "themes": pi_themes,
            "top_mentioned_tickers": top_mentioned,
            "overall_tone": pi_tone,
        },
        "portfolio_alignment": {
            "overlap_pct": overlap_pct,
            "missing_popular": missing_popular,
            "contrarian_holds": contrarian_holds,
        },
    }

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

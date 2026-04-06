"""
Committee Synthesis Engine

CIO Review v5.3 (Legacy Review): Codifies the CIO synthesis logic that was
previously implemented ad-hoc in /tmp scripts during committee runs. This
module provides a deterministic, testable, versioned implementation of:

1. Agent vote counting with freshness weights and synthetic data discount
2. Signal-aware base conviction with regime adjustment
3. BUY-side quality bonuses and conviction floors
4. Penalty/bonus adjustments with caps and contradiction detection
5. Action assignment with signal-aware thresholds
6. Concordance matrix construction with capital efficiency scoring
7. Exposure-weighted sector gap detection

Key design principles (from v3 live-run bug fixes):
- Neutral agent views are TRUE NEUTRAL (split evenly, do not lean bullish)
- BUY-side quality bonuses only apply to BUY-signal stocks
- BUY signal sets a conviction floor (the quant system verified criteria)
- SELL conviction is reduced when tech/macro agents disagree
- HOLD signal caps base to prevent easy escalation to ADD

CIO Legacy Review additions (v5.3):
- A1: Regime-adjusted conviction (RISK_OFF: -15%, CAUTIOUS: -8%)
- A2: Synthetic data discount (0.5x weight for fallback agent data)
- A3: Contradiction detection with penalty integration
- B2: Extreme EXRET penalty (>40% triggers staleness penalty)
- C1: Exposure-weighted sector gap detection
- C2: Capital efficiency score for within-group ranking

CIO v11.0 (Final Legacy Review):
- L1: Action/conviction desync fix — re-evaluate ALL actions after sector penalty
- L2: Signal velocity wired to previous concordance data
- L5: Sector concentration penalty cap reduced from 10 to 6 for existing holdings
- L6: Directional confidence penalty skipped for dual-synthetic stocks
- L7: Conviction floors reapplied after sector concentration penalty

CIO v12.0 (Architecture Audit):
- R1: Kill thesis floor escape fix — skip quality floors when kill thesis triggered
- M1: Asymmetric news weights — negative news 1.3x, positive 0.85x (prospect theory)
- R2: SELL bear ratio continuous interpolation (eliminates 0.65-0.79 dead zone)
- M2: Volume-weighted census signal — scales census weight by divergence magnitude
- M3: Earnings trajectory momentum — ACCELERATING/DECELERATING surprise modulation

CIO v14.0 (Conviction Effectiveness Review):
- V1: RSI-contextual tech vote — AVOID/EXIT_SOON at RSI<35 = neutral (oversold, not bearish)
- V2: BUY-tech disagree penalty scaling — RSI<35 → -2, RSI 35-50 → -5, RSI>50 → -8
- V3: Risk warning consensus override — buy_pct>=90 + fund>=70 → reduced risk weight
- V4: Excess EXRET staleness validation — only penalize when buy_pct<80 (confirms stale)
- V5: Penalty proportionality cap — total penalties capped at 60% of base conviction
- V6: Sector concentration scope — count by BUY/ADD actions, not total sector population

CIO v19.0 (Regime Momentum Overlay):
- R1: compute_regime_momentum() — forward-looking regime assessment using RSI breadth,
      insider sentiment, VIX trend, credit direction, and catalyst polarity. Classifies
      regime as IMPROVING, STABLE, or DETERIORATING. Addresses the fundamental limitation
      that 4 of 6 regime inputs are backward-looking (sector returns, yield curve, credit
      spreads, inflation) — by the time they classify RISK_OFF, the selloff may be 70% done.
- R2: Variable conviction cap in RISK_OFF — IMPROVING raises cap from 75 to 82,
      DETERIORATING lowers it to 70. This lets the system add to oversold quality names
      during late-stage RISK_OFF (when RSI breadth is capitulating and insiders are buying)
      while staying defensive when conditions are actively worsening.
- R3: Variable regime discount — IMPROVING softens the 15% RISK_OFF discount to 10%,
      DETERIORATING hardens it to 20%. Applied in determine_base_conviction().

CIO v20.0 (Conviction Effectiveness Review):
- D1: Conviction decay — signals lose conviction over time (0.98^weeks for BUY/HOLD held >14 days)
- D2: Census contrarian indicator — F&G > 80 penalizes new BUYs, F&G < 20 bonuses them
- D3: Earnings proximity guard — penalize BUYs within 7 days of earnings (-12 to -15)
      Enhanced PEAD bonus for serial beaters (+8 from +5)
- D4: Stale analyst target detection — Type A (>6mo) with no momentum change = -5
- D5: Regional calibration — European stocks -5 (38.6% hit rate), HK stocks +3 (66.4%)
- D6: Portfolio Sharpe consideration — penalize positions that create redundancy (-4 to -8)
- D7: Macro regime threshold tightening — RISK_OFF new positions need EXRET > 20 or -10
- D8: Position sizing output — Kelly-inspired sizing (1-5%) based on conviction/beta/correlation
- D9: Entry timing signal — tech timing recommendation in entry note field

CIO v23.0 (Quick-Win Conviction Modifiers):
- S1: Short interest — SI > 10% + bullish tech = +3 squeeze, SI > 10% + weak fundamentals = -3
- S2: Target price dispersion — wide (>30%) = -2 quality penalty, narrow (<10%) = +2 consensus boost
- S3: Deterministic earnings date — yfinance calendar replaces news agent web search for D3

CIO v23.1 (Currency Risk Codification — R6):
- R6: Currency risk for EUR-based investor — penalizes when EUR strengthens vs stock's currency
      EUR stocks = no penalty (home), USD stocks = -3/-5 on EUR strength,
      GBP/HKD/JPY = -2/-4 on weakness vs EUR, crypto = exempt

CIO v23.3 (Signal Quality & Timing):
- T1: Multi-timeframe confluence — daily+weekly aligned = +5, conflicting = -3
- T2: Volume confirmation — high relative volume + rising OBV + BUY = +3
- T3: EPS revision tracking — REVISIONS_UP for BUY = +3, REVISIONS_DOWN = -3
- T4: Staged watchlist — gated opportunities get re-entry triggers
- T5: Regime transition model — 3-run vs 7-run conviction MA crossover detection

CIO v21.0 (Codified Agent Rules — previously prompt-only in AGENT.md):
- R1: Relative strength vs SPY confirmation — +3 outperforming, -5 underperforming BUYs
- R2: Revenue growth trajectory — +5 accelerating, -5 declining BUYs
- R3: Dividend cut risk — -5 yield trap (div_yield > 4% with FCF < 50% of div)
- R4: ATR-based entry timing — ATR% > 5% forces WAIT_FOR_PULLBACK (late override)
- R5: Liquidity-adjusted sizing — daily dollar volume < $5M reduces position 40%

CIO v15.0 (Legacy CIO Review — Conviction Integrity):
- W1: Consensus-anchored vote injection — buy_pct >= 75% injects a synthetic
      "analyst consensus agent" vote that scales with consensus strength, preventing
      neutral agent dilution from suppressing stocks with overwhelming analyst agreement
- W2: Inviolable BUY floor after sector concentration — sector concentration penalty
      is capped so it cannot breach the graduated quality floor. Floor is the HARD minimum.
- W3: Regime discount cap for BUY signals — CAUTIOUS regime discount capped at 5%
      (was 8%) for BUY-signal stocks because the signal engine already validated criteria
      under current market conditions
- W4: Consensus premium in base conviction — buy_pct >= 80% adds a graduated premium
      to base conviction (separate from EXRET bonus), giving high-consensus stocks
      the differentiation they deserve
- W5: Opportunity BUY threshold reduction — opportunities use 50 (was 55) as BUY
      threshold to account for the systematic discounts already applied by the gate
- W6: Consensus warning cliff smoothing — added intermediate tier for
      excess_exret in [-10, 0) to eliminate the 7-point penalty cliff at
      excess_exret=0 that penalized stocks marginally below sector median
      the same as those far below
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent report normalizers (CIO v26.1)
#
# LLM agents produce slightly different JSON structures across runs.
# These functions normalize at the boundary so downstream code sees a
# consistent schema regardless of agent output variation.
# ---------------------------------------------------------------------------


def _normalize_fund_stocks(fund_report: Dict) -> Dict:
    """Ensure fund_report['stocks'] is dict of ticker -> data.

    Agents sometimes return a list of dicts with 'ticker' keys instead
    of a dict keyed by ticker.
    """
    stocks = fund_report.get("stocks", {})
    if isinstance(stocks, list):
        normalized = {}
        for item in stocks:
            if isinstance(item, dict):
                tkr = item.get("ticker", item.get("symbol", ""))
                if tkr:
                    normalized[tkr] = item
        fund_report["stocks"] = normalized
        logger.info("Normalized fund_report['stocks'] from list (%d items) to dict", len(normalized))
    return fund_report


def _normalize_census_divergences(census_report: Dict) -> Dict:
    """Ensure census_report['divergences'] is structured dict.

    Expected: {signal_divergences: [...], census_divergences: [...], consensus_aligned: [...]}
    Agents sometimes return a flat list with 'type' field on each item.
    """
    divs = census_report.get("divergences", {})
    if isinstance(divs, list):
        structured = {"signal_divergences": [], "census_divergences": [], "consensus_aligned": []}
        for item in divs:
            if not isinstance(item, dict):
                continue
            dtype = str(item.get("type", item.get("divergence_type", ""))).lower()
            if "signal" in dtype:
                structured["signal_divergences"].append(item)
            elif "census" in dtype:
                structured["census_divergences"].append(item)
            elif "aligned" in dtype or "consensus" in dtype:
                structured["consensus_aligned"].append(item)
            else:
                score = abs(item.get("divergence_score", 0))
                if score <= 10:
                    structured["consensus_aligned"].append(item)
                else:
                    structured["signal_divergences"].append(item)
        census_report["divergences"] = structured
        logger.info(
            "Normalized census divergences from flat list: %d signal, %d census, %d aligned",
            len(structured["signal_divergences"]),
            len(structured["census_divergences"]),
            len(structured["consensus_aligned"]),
        )
    return census_report


def _normalize_breaking_news(news_report: Dict) -> Dict:
    """Ensure news_report['breaking_news'] is a list of dicts.

    Agents sometimes return a dict of category_name -> list_of_items
    instead of a flat list.
    """
    bn = news_report.get("breaking_news", [])
    if isinstance(bn, dict):
        flat = []
        severity_map = {"CRITICAL": "HIGH_NEGATIVE", "HIGH": "LOW_NEGATIVE", "MEDIUM": "NEUTRAL"}
        for _category, items in bn.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                if "headline" not in entry:
                    entry["headline"] = entry.get("title", entry.get("event", _category))
                if "impact" not in entry:
                    sev = str(entry.get("severity", "NEUTRAL")).upper()
                    entry["impact"] = severity_map.get(sev, "NEUTRAL")
                if "affected_tickers" not in entry:
                    entry["affected_tickers"] = entry.get("tickers", [])
                if "affected_sectors" not in entry:
                    entry["affected_sectors"] = entry.get("sectors", [])
                flat.append(entry)
        news_report["breaking_news"] = flat
        logger.info("Normalized breaking_news from dict (%d categories) to list (%d items)",
                     len(bn), len(flat))
    return news_report


def _normalize_sector_rankings(macro_report: Dict) -> Dict:
    """Ensure macro_report['sector_rankings'] is dict of ETF/sector -> data.

    Agents sometimes return a list of dicts with 'sector'/'etf' keys
    instead of a dict keyed by ETF symbol.
    """
    sr = macro_report.get("sector_rankings", {})
    if isinstance(sr, list):
        sr_dict = {}
        for item in sr:
            if not isinstance(item, dict):
                continue
            key = item.get("etf", item.get("sector", item.get("name", "")))
            if key:
                sr_dict[key] = {
                    "return_1m": item.get("1m_return", item.get("return_1m", 0)),
                    "return_3m": item.get("3m_return", item.get("return_3m", 0)),
                    "relative_strength": item.get("status", item.get("relative_strength", "NEUTRAL")),
                    "rank": item.get("rank", 6),
                }
        macro_report["sector_rankings"] = sr_dict
        logger.info("Normalized sector_rankings from list (%d items) to dict", len(sr_dict))
    return macro_report


def _normalize_portfolio_news(news_report: Dict) -> Dict:
    """Ensure news_report['portfolio_news'] is dict of ticker -> list of news items.

    Agents sometimes return a list of dicts with 'ticker' keys instead
    of a dict keyed by ticker.
    """
    pn = news_report.get("portfolio_news", {})
    if isinstance(pn, list):
        pn_dict: Dict[str, list] = {}
        for item in pn:
            if not isinstance(item, dict):
                continue
            tkr = item.get("ticker", item.get("symbol", ""))
            if not tkr:
                continue
            news_items = item.get("news", [])
            if isinstance(news_items, list):
                pn_dict[tkr] = news_items
            else:
                pn_dict[tkr] = [{"headline": str(news_items), "impact": item.get("impact", "NEUTRAL")}]
        news_report["portfolio_news"] = pn_dict
        logger.info("Normalized portfolio_news from list (%d items) to dict (%d tickers)",
                     len(pn), len(pn_dict))
    return news_report


def _normalize_census_sentiment(census_report: Dict) -> Dict:
    """Ensure census sentiment values are plain numbers, not nested dicts.

    Agents sometimes return {fg_top100: {value: 59, classification: "Greed"}}
    instead of {fg_top100: 59}.
    """
    sentiment = census_report.get("sentiment", {})
    if not isinstance(sentiment, dict):
        return census_report
    for key in ("fg_top100", "fg_broad", "cash_top100", "cash_broad"):
        val = sentiment.get(key)
        if isinstance(val, dict):
            sentiment[key] = val.get("value", val.get("score", val.get("index", 0)))
            logger.info("Normalized census sentiment['%s'] from dict to %s", key, sentiment[key])
    census_report["sentiment"] = sentiment
    return census_report


def _normalize_economic_events(news_report: Dict) -> Dict:
    """Ensure news_report['economic_events'] is a flat list.

    Agents sometimes return {this_week: [...], key_context: "..."}
    instead of a flat list.
    """
    ee = news_report.get("economic_events", [])
    if isinstance(ee, dict):
        flat = ee.get("this_week", ee.get("events", []))
        if not isinstance(flat, list):
            flat = []
        news_report["economic_events"] = flat
        logger.info("Normalized economic_events from dict to list (%d items)", len(flat))
    return news_report


def _normalize_tech_stocks(tech_report: Dict) -> Dict:
    """Ensure tech_report['stocks'] is dict of ticker -> data.

    Agents sometimes return a list of dicts with 'ticker' keys instead
    of a dict keyed by ticker.
    """
    stocks = tech_report.get("stocks", {})
    if isinstance(stocks, list):
        normalized = {}
        for item in stocks:
            if isinstance(item, dict):
                tkr = item.get("ticker", item.get("symbol", ""))
                if tkr:
                    normalized[tkr] = item
        tech_report["stocks"] = normalized
        logger.info("Normalized tech_report['stocks'] from list (%d items) to dict", len(normalized))
    return tech_report


def normalize_agent_reports(
    fund_report: Dict,
    tech_report: Dict,
    macro_report: Dict,
    census_report: Dict,
    news_report: Dict,
    risk_report: Dict,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """Apply all normalizers to agent reports before synthesis.

    Call this once at the start of build_concordance() to ensure all
    downstream code sees a consistent schema.
    """
    fund_report = _normalize_fund_stocks(fund_report)
    tech_report = _normalize_tech_stocks(tech_report)
    census_report = _normalize_census_divergences(census_report)
    census_report = _normalize_census_sentiment(census_report)
    news_report = _normalize_breaking_news(news_report)
    news_report = _normalize_portfolio_news(news_report)
    news_report = _normalize_economic_events(news_report)
    macro_report = _normalize_sector_rankings(macro_report)
    return fund_report, tech_report, macro_report, census_report, news_report, risk_report


# Freshness multipliers per agent (CIO v3 F2)
AGENT_FRESHNESS = {
    "fundamental": 0.8,
    "technical": 1.0,
    "macro": 0.9,
    "census": 0.85,
    "news": 1.0,
    "opportunity": 0.8,
    "risk": 1.0,
}

# Conviction score classification
CONVICTION_TIERS = {
    "STRONG": (80, 100),
    "MODERATE": (60, 79),
    "LOW": (40, 59),
    "NONE": (0, 39),
}

# Sector to ETF mapping for rotation analysis
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# CIO v8.0 F1: Known ticker-to-GICS sector mapping.
# Eliminates 38% "Other" sector assignments caused by incomplete agent coverage.
# Agent reports often lack sector fields, leading to bad sector medians, broken
# rotation signals, and meaningless concentration analysis.
TICKER_GICS_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology",
    "PANW": "Technology", "NOW": "Technology", "PLTR": "Technology",
    "CRWD": "Technology", "SNPS": "Technology", "CDNS": "Technology",
    "SAP.DE": "Technology", "MRVL": "Technology", "MU": "Technology",
    "ANET": "Technology", "FTNT": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "WFC": "Financials",
    "SCHW": "Financials", "BLK": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "AMGN": "Healthcare",
    "ISRG": "Healthcare", "ELV": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "CMG": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "GOOG": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services",
    "SNAP": "Communication Services", "PINS": "Communication Services",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples", "CL": "Consumer Staples",
    "PM": "Consumer Staples",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "UNP": "Industrials",
    "HON": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "BA": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    "OXY": "Energy", "HAL": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "GLD": "Materials",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
    "EQIX": "Real Estate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities",
    # Crypto / Alt
    "BTC-USD": "Crypto/Alt", "ETH-USD": "Crypto/Alt",
    "MSTR": "Crypto/Alt", "COIN": "Crypto/Alt",
    # Semiconductors → Technology
    "TSM": "Technology", "ASML": "Technology",
    # European stocks
    "LYXGRE.DE": "Financials",  # Greek ETF proxy
    "NOVO-B.CO": "Healthcare",
    "SAP": "Technology",
    "AZN": "Healthcare", "AZN.L": "Healthcare",
    "SHEL": "Energy",
    "TTE": "Energy",
    "BABA": "Consumer Discretionary",
    "RHM.DE": "Industrials",  # Rheinmetall — defense
    "DTE.DE": "Communication Services",  # Deutsche Telekom
    "EDV.L": "Materials",  # Endeavour Mining
    "PRU.L": "Financials",  # Prudential plc
    "AIR.PA": "Industrials",  # Airbus
    "SU.PA": "Consumer Staples",  # Schneider / luxury (mapped broadly)
    # Asia-Pacific
    "6758.T": "Technology",  # Sony
    "0700.HK": "Communication Services",  # Tencent
    "0175.HK": "Consumer Discretionary",  # Geely
    "0388.HK": "Financials",  # HKEX
    "0027.HK": "Consumer Discretionary",  # Galaxy Entertainment
    # Americas
    "NU": "Financials",  # Nu Holdings
    "UBER": "Industrials",  # Uber — transport/logistics
    "SPGI": "Financials",  # S&P Global
    "TMUS": "Communication Services",  # T-Mobile
    "EMIRATESNBD.AE": "Financials",  # Emirates NBD
    "EW": "Healthcare",  # Edwards Lifesciences
    "APH": "Technology",  # Amphenol
}

# Sector normalization: merge agent-reported granular sectors into GICS standards.
# Agents may report "Semiconductors", "Consumer Electronics", etc. — these need
# to map to the 11 GICS sectors for consistent rotation analysis.
_SECTOR_NORMALIZE = {
    "Semiconductors": "Technology",
    "Consumer Electronics": "Technology",
    "Technology/Services": "Industrials",
    "Software": "Technology",
    "Semiconductor Equipment": "Technology",
    "Financial Services": "Financials",
    "Insurance": "Financials",
    "Banking": "Financials",
    "Capital Markets": "Financials",
    "Basic Materials": "Materials",
    "Metals & Mining": "Materials",
    "Gold": "Materials",
    "Casinos/Gaming": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Retail": "Consumer Discretionary",
    "Auto Manufacturers": "Consumer Discretionary",
    "Aerospace & Defense": "Industrials",
    "Defense": "Industrials",
    "Transportation": "Industrials",
    "Biotechnology": "Healthcare",
    "Pharma": "Healthcare",
    "Medical Devices": "Healthcare",
    "Oil & Gas": "Energy",
    "Telecom": "Communication Services",
    "Media": "Communication Services",
    "Interactive Media": "Communication Services",
    # CIO v13.0 F5: Catch bare "Consumer" and other orphan sectors
    "Consumer": "Consumer Discretionary",
    "Consumer Goods": "Consumer Staples",
    "Payments": "Financials",
    "Crypto": "Crypto/Alt",
    "Digital Assets": "Crypto/Alt",
    "Logistics": "Industrials",
    "Cloud": "Technology",
    "AI": "Technology",
    "Cybersecurity": "Technology",
    "Networking": "Technology",
    "Fintech": "Financials",
}

# CIO v25.2: Currency zone inference from ticker suffix.
# Used as fallback when sig_data doesn't include currency_zone from data_loader.
_EUR_SUFFIXES = ('.DE', '.PA', '.AS', '.MI', '.MC', '.BR', '.CO', '.ST', '.OL', '.HE')
_CRYPTO_BASES = {'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE'}


def _infer_currency_zone(ticker: str) -> str:
    """Infer currency denomination from ticker suffix."""
    if any(ticker.endswith(s) for s in _EUR_SUFFIXES):
        return 'EUR'
    if ticker.endswith('.L'):
        return 'GBP'
    if ticker.endswith('.HK'):
        return 'HKD'
    if ticker.endswith('.T'):
        return 'JPY'
    if ticker.endswith('.SW'):
        return 'CHF'
    if '-USD' in ticker and ticker.split('-')[0] in _CRYPTO_BASES:
        return 'CRYPTO'
    if ticker.endswith('.AE'):
        return 'AED'
    return 'USD'


def compute_regime_momentum(
    macro_report: Dict[str, Any],
    tech_report: Dict[str, Any],
    fund_report: Dict[str, Any],
    news_report: Dict[str, Any],
) -> str:
    """
    CIO v19.0 R1: Assess whether the current regime is IMPROVING, STABLE,
    or DETERIORATING by combining forward-looking signals that the regime
    classifier misses.

    The regime classifier (RISK_ON/CAUTIOUS/RISK_OFF) uses mostly backward-
    looking inputs (sector ETF returns, yield curve, credit spreads, inflation).
    By the time it declares RISK_OFF, the selloff may be 70% done. This function
    looks at leading indicators to detect regime *transitions*.

    Scoring: sum of 5 directional signals, each -2 to +2.
      >= +4: IMPROVING (regime turning, add to oversold)
      -3 to +3: STABLE (regime holding, maintain current posture)
      <= -4: DETERIORATING (regime worsening, increase defensiveness)
    """
    score = 0

    # Signal 1: RSI breadth — what fraction of stocks are deeply oversold?
    # High oversold breadth = capitulation = regime likely IMPROVING
    tech_stocks = tech_report.get("stocks", {})
    if tech_stocks:
        rsis = [d.get("rsi", 50) for d in tech_stocks.values()
                if isinstance(d, dict) and d.get("rsi") is not None]
        if rsis:
            oversold_pct = sum(1 for r in rsis if r < 30) / len(rsis)
            avg_rsi = sum(rsis) / len(rsis)
            if oversold_pct >= 0.20 or avg_rsi < 35:
                score += 2  # Extreme breadth = capitulation
            elif oversold_pct >= 0.10 or avg_rsi < 40:
                score += 1
            # Overbought breadth = regime peaking
            overbought_pct = sum(1 for r in rsis if r > 70) / len(rsis)
            if overbought_pct >= 0.20:
                score -= 1

    # Signal 2: Insider sentiment — net buying = forward bullish
    fund_stocks = fund_report.get("stocks", {})
    if fund_stocks:
        net_buying = sum(1 for d in fund_stocks.values()
                        if isinstance(d, dict) and
                        d.get("insider_sentiment") == "NET_BUYING")
        net_selling = sum(1 for d in fund_stocks.values()
                         if isinstance(d, dict) and
                         d.get("insider_sentiment") == "NET_SELLING")
        total = net_buying + net_selling
        if total > 0:
            buy_ratio = net_buying / total
            if buy_ratio >= 0.80:
                score += 2  # Overwhelming insider buying
            elif buy_ratio >= 0.60:
                score += 1
            elif buy_ratio <= 0.30:
                score -= 2  # Insiders fleeing
            elif buy_ratio <= 0.40:
                score -= 1

    # Signal 3: VIX trend — declining from peak = improving
    indicators = macro_report.get("indicators") or macro_report.get("macro_indicators", {})
    vix = indicators.get("vix")
    vix_trend = str(indicators.get("vix_trend", "")).upper()
    if vix is not None:
        if "DECLINING" in vix_trend or "FALLING" in vix_trend:
            score += 1
        elif "RISING" in vix_trend or "SPIKING" in vix_trend:
            score -= 1
        # Extreme VIX levels
        if isinstance(vix, (int, float)):
            if vix > 35:
                score -= 1  # Crisis level
            elif vix < 18:
                score += 1  # Calm

    # Signal 4: News catalyst polarity — more positive catalysts = improving
    breaking = news_report.get("breaking_news", [])
    if breaking:
        pos = sum(1 for n in breaking
                  if n.get("impact", "").endswith("POSITIVE"))
        neg = sum(1 for n in breaking
                  if n.get("impact", "").endswith("NEGATIVE"))
        if pos > neg + 1:
            score += 1
        elif neg > pos + 1:
            score -= 1

    # Signal 5: Credit direction — stable/improving vs deteriorating
    credit = str(indicators.get("credit_status", "")).upper()
    if "IMPROVING" in credit or "STABLE" in credit:
        score += 1
    elif "DETERIORATING" in credit or "WIDENING" in credit or "STRESS" in credit:
        score -= 1

    # Classify
    if score >= 4:
        momentum = "IMPROVING"
    elif score <= -4:
        momentum = "DETERIORATING"
    else:
        momentum = "STABLE"

    logger.info(
        "Regime momentum: score=%d → %s (rsi_breadth, insiders, vix, news, credit)",
        score, momentum,
    )
    return momentum


def compute_regime_transition(
    history_dir: Optional[str] = None,
    min_runs: int = 3,
) -> Dict[str, Any]:
    """
    CIO v23.3: Detect regime transitions by comparing short-term (3-run)
    vs long-term (7-run) moving averages of committee conviction scores.

    Returns dict with transition direction, scores, and trend classification.
    A rising short MA crossing above long MA = IMPROVING before the regime
    classifier would catch it.
    """
    import glob
    import json as _json
    from pathlib import Path

    hdir = Path(history_dir or os.path.expanduser("~/.weirdapps-trading/committee/history"))
    files = sorted(glob.glob(str(hdir / "concordance-*.json")))

    # Parse avg conviction from each run
    run_scores = []
    for fp in files:
        try:
            with open(fp) as f:
                data = _json.load(f)
            if isinstance(data, list):
                concs = data
            else:
                concs = data.get("concordance", [])
            if not concs:
                continue
            convictions = [c.get("conviction", 50) for c in concs if isinstance(c, dict)]
            if convictions:
                avg = sum(convictions) / len(convictions)
                date = data.get("date", Path(fp).stem.replace("concordance-", "")) if isinstance(data, dict) else Path(fp).stem.replace("concordance-", "")
                run_scores.append({"date": date, "avg_conviction": round(avg, 1)})
        except Exception:
            continue

    if len(run_scores) < min_runs:
        return {"transition": "INSUFFICIENT_DATA", "runs": len(run_scores)}

    scores = [r["avg_conviction"] for r in run_scores]
    short_window = min(3, len(scores))
    long_window = min(7, len(scores))

    short_ma = sum(scores[-short_window:]) / short_window
    long_ma = sum(scores[-long_window:]) / long_window

    # Direction: is short MA above or below long MA?
    spread = short_ma - long_ma
    if spread > 2.0:
        transition = "IMPROVING"
    elif spread < -2.0:
        transition = "DETERIORATING"
    else:
        transition = "STABLE"

    # Trend: is the short MA itself rising or falling?
    if len(scores) >= 2:
        recent_delta = scores[-1] - scores[-2]
    else:
        recent_delta = 0

    return {
        "transition": transition,
        "short_ma": round(short_ma, 1),
        "long_ma": round(long_ma, 1),
        "spread": round(spread, 1),
        "recent_delta": round(recent_delta, 1),
        "runs_analyzed": len(run_scores),
        "history": run_scores[-7:],
    }


def resolve_sector(ticker: str, sector_map: Dict[str, str]) -> str:
    """Resolve sector for a ticker using caller map, then built-in GICS fallback.

    CIO v8.0 F1: Eliminates 'Other' sector assignments by falling back
    to TICKER_GICS_MAP when the caller-provided sector_map has no entry.
    Also normalizes granular agent-reported sectors (e.g. 'Semiconductors')
    to standard GICS sectors (e.g. 'Technology').
    """
    # First try TICKER_GICS_MAP (highest confidence)
    gics = TICKER_GICS_MAP.get(ticker, "")
    if gics:
        return gics
    # Then try caller-provided sector_map
    sec = sector_map.get(ticker, "")
    if sec and sec != "Other":
        # Normalize agent-reported granular sectors to GICS standard
        return _SECTOR_NORMALIZE.get(sec, sec)
    return "Other"


def compute_sector_medians(
    portfolio_signals: Dict[str, Dict],
    sector_map: Dict[str, str],
) -> Tuple[Dict[str, float], float]:
    """Compute median EXRET per sector and universe median."""
    sector_exrets: Dict[str, List[float]] = {}
    all_exrets = []
    for ticker, sig in portfolio_signals.items():
        sector = resolve_sector(ticker, sector_map)
        exret = sig.get("exret", 0)
        if exret is None:
            exret = 0
        sector_exrets.setdefault(sector, []).append(exret)
        all_exrets.append(exret)

    sector_medians = {}
    for sec, vals in sector_exrets.items():
        s = sorted(vals)
        n = len(s)
        sector_medians[sec] = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    all_sorted = sorted(all_exrets)
    n = len(all_sorted)
    if n == 0:
        universe_median = 0
    elif n % 2:
        universe_median = all_sorted[n // 2]
    else:
        universe_median = (all_sorted[n // 2 - 1] + all_sorted[n // 2]) / 2
    return sector_medians, universe_median


def count_agent_votes(
    fund_score: float,
    tech_signal: str,
    tech_momentum: int,
    macro_fit: str,
    census_alignment: str,
    news_impact: str,
    risk_warning: bool,
    signal: str,
    fund_synthetic: bool = False,
    tech_synthetic: bool = False,
    regime: str = "",
    census_div_score: float = 0.0,
    rsi: float = 50.0,
    buy_pct: float = 50.0,
) -> Tuple[float, float]:
    """
    Count bull/bear weighted votes from agent views.

    CRITICAL: Neutral views split evenly between bull and bear.
    Only clear directional views count as bull or bear.
    This prevents inflation of bull_pct from neutral agents.

    CIO Legacy A2: Synthetic (fallback-generated) agent data receives
    a 0.5x discount because manufactured data from signal proxies is
    categorically less informative than real computed analysis.

    CIO Legacy A5: Tracks directional vs neutral weight internally.
    The directional_confidence ratio is stored as a module-level variable
    for the caller to access if needed.

    CIO v6.0 F5: When the Risk Manager finds no warning, its neutral vote
    is regime-sensitive. "No danger found" is mildly bullish in RISK_ON
    but merely non-negative in RISK_OFF.

    Returns:
        Tuple of (bull_weight, bear_weight, directional_confidence).
        CIO v7.0 P2: directional_confidence is now a proper return value
        instead of a module-level function attribute, making the function
        pure and thread-safe.
    """
    bull = 0.0
    bear = 0.0
    directional_weight = 0.0
    neutral_weight = 0.0

    # CIO Legacy A2: Synthetic discount — fallback data gets half weight
    fund_disc = 0.5 if fund_synthetic else 1.0
    tech_disc = 0.5 if tech_synthetic else 1.0

    # Fundamental (freshness 0.8x, synthetic discount)
    fund_weight = 0.8 * fund_disc
    if fund_score >= 70:
        bull += fund_weight
        directional_weight += fund_weight
    elif fund_score < 45:
        bear += fund_weight
        directional_weight += fund_weight
    else:
        bull += fund_weight / 2
        bear += fund_weight / 2
        neutral_weight += fund_weight

    # Technical (freshness 1.0x, synthetic discount)
    # CIO v14.0 V1: RSI-contextual tech vote. At RSI < 30 (deeply oversold),
    # AVOID/EXIT_SOON confirms the obvious decline — it carries no new bearish
    # information and selling here locks in worst-case exit. At RSI 30-40,
    # the stock is oversold and the tech AVOID is a weak signal at best.
    # Only at RSI > 40 does AVOID carry genuine directional weight.
    tech_weight = 1.0 * tech_disc
    if tech_signal == "ENTER_NOW":
        bull += tech_weight
        directional_weight += tech_weight
    elif tech_signal in ("AVOID", "EXIT_SOON"):
        if rsi < 30:
            # Deeply oversold — AVOID is stating the obvious, split neutral
            bull += tech_weight / 2
            bear += tech_weight / 2
            neutral_weight += tech_weight
        elif rsi < 40:
            # Oversold — weak bear, not full directional
            bull += tech_weight * 0.4
            bear += tech_weight * 0.6
            directional_weight += tech_weight
        else:
            bear += tech_weight
            directional_weight += tech_weight
    elif tech_signal == "WAIT_FOR_PULLBACK":
        bull += tech_weight * 0.6
        bear += tech_weight * 0.4
        directional_weight += tech_weight  # Leaning directional
    elif tech_momentum > 20:
        # CIO v18.0 B1: Regime-conditional MACD scoring. Backtest evidence
        # (Mar 16-25, 82 stocks) shows BULLISH MACD in RISK_OFF = -0.24% avg
        # (48% positive) while BEARISH MACD = +1.59% (63% positive). In bear
        # markets, BULLISH MACD means "hasn't cracked yet" — about to join the
        # selloff. Neutralize positive momentum signal in RISK_OFF.
        if regime == "RISK_OFF":
            bull += tech_weight / 2
            bear += tech_weight / 2
            neutral_weight += tech_weight
        else:
            bull += tech_weight * 0.7
            bear += tech_weight * 0.3
            directional_weight += tech_weight
    elif tech_momentum < -20:
        # CIO v18.0 B1: In RISK_OFF, bearish MACD is a counter-indicator —
        # stocks that already corrected tend to bounce. Neutralize rather than
        # count as bearish.
        if regime == "RISK_OFF":
            bull += tech_weight / 2
            bear += tech_weight / 2
            neutral_weight += tech_weight
        else:
            bull += tech_weight * 0.3
            bear += tech_weight * 0.7
            directional_weight += tech_weight
    else:
        bull += tech_weight / 2
        bear += tech_weight / 2
        neutral_weight += tech_weight

    # Macro (freshness 0.9x)
    if macro_fit == "FAVORABLE":
        bull += 0.9
        directional_weight += 0.9
    elif macro_fit == "UNFAVORABLE":
        bear += 0.9
        directional_weight += 0.9
    else:
        bull += 0.45
        bear += 0.45
        neutral_weight += 0.9

    # Census (freshness 0.85x)
    # CIO v12.0 M2: Magnitude-weighted census signal. A PI going from 0.5%
    # to 0.6% allocation is noise; from 2% to 5% is a conviction trade.
    # Scale the census weight by the absolute divergence score magnitude,
    # normalizing to [0.5, 1.0] range so low-magnitude signals still count
    # but high-magnitude signals get full weight.
    div_magnitude = min(abs(census_div_score) / 50.0, 1.0) if census_div_score else 0.5
    census_weight = 0.85 * (0.5 + 0.5 * div_magnitude)  # Range: 0.425 to 0.85
    if census_alignment == "ALIGNED":
        bull += census_weight
        directional_weight += census_weight
    elif census_alignment == "DIVERGENT":
        # Split proportionally to census_weight
        bull += census_weight * 0.41  # ~0.35 at full weight
        bear += census_weight * 0.59  # ~0.50 at full weight
        directional_weight += census_weight
    elif census_alignment == "CENSUS_DIV":
        bull += census_weight * 0.71  # ~0.60 at full weight
        bear += census_weight * 0.29  # ~0.25 at full weight
        directional_weight += census_weight
    else:
        bull += census_weight / 2
        bear += census_weight / 2
        neutral_weight += census_weight

    # News (freshness 1.0x)
    # CIO v12.0 M1: Asymmetric news weights. Behavioral finance research
    # (Kahneman & Tversky, prospect theory) establishes that negative news
    # has ~1.5x the market impact of positive news. Positive news tends to
    # be already priced in (travels faster); negative news creates persistent
    # underreaction. The asymmetry prevents conviction inflation from positive
    # catalysts while properly weighting downside risk from negative ones.
    if "HIGH_POSITIVE" in news_impact:
        bull += 0.85
        bear += 0.15
        directional_weight += 1.0
    elif "LOW_POSITIVE" in news_impact:
        bull += 0.65
        bear += 0.35
        directional_weight += 1.0
    elif "HIGH_NEGATIVE" in news_impact:
        bear += 1.3
        directional_weight += 1.3
    elif "LOW_NEGATIVE" in news_impact:
        bull += 0.25
        bear += 0.75
        directional_weight += 1.0
    else:
        bull += 0.5
        bear += 0.5
        neutral_weight += 1.0

    # Risk Manager (CIO Legacy B3: 1.2x for BUY assessment, 2.0x for SELL)
    # CIO v6.0 F5: Regime-sensitive neutral vote when no warning found
    # CIO v14.0 V3: When overwhelming consensus (buy_pct >= 90 AND fund_score >= 70),
    # the risk warning is contradicting 20+ investment bank analysts. The risk is
    # real but should not dominate the decision — reduce weight to 0.6x. This
    # prevents a single risk flag from suppressing the collective wisdom of the
    # sell-side, which has already verified the fundamental thesis independently.
    # fund_score >= 70 ensures we only override when fundamentals are at least decent;
    # the analyst consensus threshold (90%) is the primary gate.
    risk_mult = 2.0 if signal == "S" else 1.2
    if risk_warning and buy_pct >= 90 and fund_score >= 70:
        risk_mult = 0.6  # V3: defer to overwhelming consensus
    if risk_warning:
        bear += risk_mult
        directional_weight += risk_mult
    elif regime == "RISK_ON":
        # Absence of risk in calm markets is mildly bullish
        bull += 0.6
        bear += 0.4
        neutral_weight += 1.0
    elif regime == "RISK_OFF":
        # Absence of specific risk doesn't override systemic risk
        bull += 0.4
        bear += 0.6
        neutral_weight += 1.0
    else:
        bull += 0.5
        bear += 0.5
        neutral_weight += 1.0

    # CIO v15.0 W1: Consensus-anchored vote injection.
    # When buy_pct >= 75%, the sell-side analyst consensus is a strong independent
    # signal that the committee's neutral agents are simply not covering — NOT
    # contradicting. A stock with 95% BUY consensus from 20+ analysts should not
    # be suppressed to bull_pct=56% just because the macro/census/news agents
    # have no specific view. This injects a synthetic "analyst consensus" vote
    # proportional to consensus strength, treated as directional.
    if buy_pct >= 75 and signal == "B":
        # Scale from 0.0 at buy_pct=75 to 1.2 at buy_pct=100
        consensus_weight = (buy_pct - 75) / 25 * 1.2
        bull += consensus_weight
        directional_weight += consensus_weight
    elif buy_pct <= 25 and signal == "S":
        # Mirror for SELL: very low buy_pct strengthens bear case
        consensus_weight = (25 - buy_pct) / 25 * 1.2
        bear += consensus_weight
        directional_weight += consensus_weight

    # CIO Legacy A5: Directional confidence — ratio of directional
    # (non-neutral) weight to total weight. When 4 of 7 agents are neutral,
    # this will be low (~0.43), indicating the signal is diluted.
    # CIO v7.0 P2: Returned as third tuple element instead of function attribute.
    total_all = directional_weight + neutral_weight
    dir_confidence = directional_weight / total_all if total_all > 0 else 0.5

    return bull, bear, dir_confidence


def compute_portfolio_sharpe_impact(
    ticker: str,
    portfolio_signals: Dict[str, Dict],
    risk_report: Dict,
) -> int:
    """
    CIO v20.0 D6: Estimate if adding/keeping this stock improves portfolio Sharpe.

    Returns penalty (negative value) if the stock creates redundancy through
    high correlation with existing holdings.
    """
    correlations = risk_report.get("correlation_clusters", [])
    beta = portfolio_signals.get(ticker, {}).get("beta", 1.0)

    # Count highly correlated positions (>0.7)
    correlated_count = 0
    for cluster in correlations:
        if ticker in cluster.get("tickers", []):
            correlated_count = len(cluster.get("tickers", [])) - 1
            break

    # Penalize if adding creates redundancy
    if correlated_count >= 3:
        return -8  # Highly redundant
    elif correlated_count >= 2:
        return -4
    return 0


def compute_position_size(
    conviction: int,
    beta: float,
    correlated_positions: int,
    buy_pct: float,
) -> float:
    """
    CIO v20.0 D8: Kelly-inspired position sizing.

    Returns recommended position size as percentage of portfolio (1.0-5.0).
    """
    base_size = 3.0  # Conservative base (was 5%)

    # Scale by conviction
    if conviction >= 80:
        size = base_size * 1.5
    elif conviction >= 65:
        size = base_size * 1.2
    elif conviction >= 50:
        size = base_size
    else:
        size = base_size * 0.7

    # Reduce for high beta
    if beta > 1.5:
        size *= 0.8

    # Reduce for correlated positions
    size -= correlated_positions * 0.5

    # Reduce if consensus too high (contrarian)
    if buy_pct > 90:
        size *= 0.9

    return round(max(1.0, min(5.0, size)), 1)


def determine_base_conviction(
    bull_pct: float,
    signal: str,
    fund_score: float,
    excess_exret: float,
    bear_ratio: float,
    regime: str = "",
    regime_momentum: str = "STABLE",
) -> int:
    """
    Determine base conviction from agent consensus, anchored by signal.

    CIO v5.2: Uses continuous interpolation instead of 6-bucket discretization
    to break conviction clustering. The old system mapped 50-65% bull to the
    same base=60, losing differentiation. Now, 52% and 64% produce different
    bases (52 vs 62), which propagates through to final scores.

    CIO Legacy A1: Applies regime discount to prevent overconfidence in
    volatile markets. A conviction of 75 should not mean the same thing
    when VIX is at 12 vs 35.

    Key principles:
    - BUY signal: agents can upgrade but floor is 55
    - SELL signal: conviction reduced when agents disagree
    - HOLD signal: base capped at 70 to prevent easy escalation
    - RISK_OFF regime: 15% discount to agent base (before signal floors)
    - CAUTIOUS regime: 8% discount
    """
    # CIO Legacy B1: Sigmoid conviction mapping (replaces linear interpolation).
    # The sigmoid creates steeper differentiation near the 50% threshold (where
    # votes crossing majority carry the most information) and flattens at extremes
    # (where additional consensus is less informative). This is grounded in
    # information theory — near-50/50 votes are more surprising than 90/10 votes.
    x = (bull_pct - 50) / 8.33  # Maps 0→-6, 50→0, 100→+6
    sigmoid = 1 / (1 + math.exp(-x))
    agent_base = int(30 + sigmoid * 50)
    agent_base = max(30, min(80, agent_base))

    # CIO Legacy A1: Regime discount applied to agent_base BEFORE signal floors.
    # In RISK_OFF markets, even strong consensus deserves less confidence because
    # tail risks dominate and correlations spike. The discount is applied here
    # (not to final conviction) so that signal floors can still rescue genuinely
    # strong BUY-signal stocks.
    #
    # CIO v15.0 W3: For BUY signals, the regime discount is capped at 5% in
    # CAUTIOUS (was 8%) because the quantitative signal engine already validated
    # all tier-specific criteria under current market conditions. The full 8%
    # discount was compounding with sector concentration penalties to suppress
    # BUY-signal stocks by 15-18% total — far beyond the intended macro caution.
    # RISK_OFF retains the full 15% discount because it represents genuine crisis.
    #
    # CIO v19.0 R3: Variable regime discount based on regime momentum.
    # When RISK_OFF is IMPROVING (insiders buying, RSI capitulating, catalysts
    # turning positive), soften discount to 10% — the worst is likely behind us.
    # When DETERIORATING, harden to 20% — conditions actively worsening.
    if regime == "RISK_OFF":
        if regime_momentum == "IMPROVING":
            agent_base = int(agent_base * 0.90)  # Softened: 10% vs 15%
        elif regime_momentum == "DETERIORATING":
            agent_base = int(agent_base * 0.80)  # Hardened: 20% vs 15%
        else:
            agent_base = int(agent_base * 0.85)  # Default: 15%
    elif regime == "CAUTIOUS":
        discount = 0.95 if signal == "B" else 0.92
        agent_base = int(agent_base * discount)

    if signal == "B":
        base = max(agent_base, 55)
        # CIO v13.0 F4: BUY consensus premium — when the sigmoid agent_base
        # exceeds 55 (strong bull consensus), let it contribute. Previously
        # max(agent_base, 55) flattened everything to 55 for typical stocks.
        # Now add half the excess for differentiation.
        if agent_base > 55:
            base = 55 + (agent_base - 55) // 2 + (agent_base - 55) % 2
        # BUY-side quality bonus (CIO C1) — ONLY for BUY signals
        if fund_score >= 80:
            base = max(base + 10, 60)
        elif fund_score >= 65:
            base = max(base + 5, 55)
        # CIO v15.0 W4: Consensus premium — high buy_pct deserves a base boost
        # independent of agent votes. The sell-side consensus is the single most
        # validated input (20+ independent analyst teams), and its strength
        # should differentiate stocks in the base conviction, not just in bonuses.
        # This addresses the conviction compression where 95% and 60% buy_pct
        # stocks end up within 3 points of each other.
        if bull_pct >= 80:
            consensus_premium = min(8, int((bull_pct - 75) / 5))
            base += consensus_premium

        # Proportional excess EXRET bonus (CIO v5.2)
        # Instead of binary >=12 → +5, scale: 5→+1, 10→+3, 15→+4, 20+→+5
        # CIO Legacy B2: Extreme EXRET (>40) triggers a staleness penalty.
        # EXRET > 40% almost always indicates stale analyst targets (not updated
        # after a large drop), distressed/turnaround binary outcomes, or data
        # errors. E.g., MSTR at 171% EXRET is not a genuine expected return.
        # CIO v14.0 V4: Only apply when buy_pct < 80. When buy_pct >= 80 AND
        # excess_exret > 40, the high consensus VALIDATES the high target —
        # multiple analysts independently set high targets. NVDA at 50% EXRET
        # with 100% BUY consensus is genuine conviction, not stale data.
        if excess_exret > 40 and bull_pct < 65:
            base -= 3  # Staleness penalty for extreme EXRET with low consensus
        elif excess_exret >= 5:
            exret_bonus = min(5, int(excess_exret / 4))
            base += exret_bonus
    elif signal == "S":
        # CIO v12.0 R2: Continuous interpolation for SELL base conviction.
        # The previous discrete buckets created a dead zone in the 0.65-0.79
        # bear_ratio range where increasing consensus had zero effect on
        # conviction. Continuous mapping preserves differentiation.
        if bear_ratio >= 0.80:
            base = 85
        elif bear_ratio >= 0.40:
            # Linear interpolation: 0.40→50, 0.80→85
            base = int(50 + (bear_ratio - 0.40) * 87.5)
        else:
            base = 50
    else:
        # HOLD: agents determine, but capped at 70
        base = min(agent_base, 70)

    return base


def detect_contradictions(
    macro_fit: str,
    tech_signal: str,
    fund_score: float,
    risk_warning: bool,
    census_alignment: str,
    news_impact: str,
    **kwargs,
) -> Tuple[int, List[str]]:
    """
    Detect logical contradictions between agent views (CIO Legacy A3).

    Contradictory signals indicate unresolved uncertainty that should
    reduce conviction. The system currently counts votes independently
    but doesn't flag when agents disagree in ways that suggest the
    situation is genuinely uncertain (not just a matter of weighting).

    Returns (penalty_points, list_of_contradiction_descriptions).
    """
    contradictions: List[str] = []
    penalty = 0

    # Macro-Technical: regime says avoid but technicals say enter
    if macro_fit == "UNFAVORABLE" and tech_signal == "ENTER_NOW":
        contradictions.append("Macro UNFAVORABLE but Technical ENTER_NOW")
        penalty += 5

    # Fundamental-Risk: strong fundamentals but risk manager warns
    # CIO v14.0 V3: Skip when overwhelming consensus (buy_pct >= 90 AND
    # fund_score >= 70). When 20+ analysts and the fundamental model agree,
    # the risk warning reflects caution, not a genuine contradiction.
    if fund_score >= 70 and risk_warning:
        _bp = kwargs.get("buy_pct", 0)
        if _bp >= 90:
            # Overwhelming consensus — note but don't penalize
            contradictions.append(
                f"Fundamental score {fund_score:.0f} vs Risk warning (overridden by {_bp:.0f}% consensus)"
            )
        else:
            contradictions.append(
                f"Fundamental score {fund_score:.0f} but Risk Manager warns"
            )
            penalty += 3

    # Census-News: popular investors distributing despite positive news
    if census_alignment == "DIVERGENT" and "POSITIVE" in news_impact:
        contradictions.append("PIs distributing despite positive news")
        penalty += 3

    # Macro-News: macro bearish but news bullish (or vice versa)
    if macro_fit == "UNFAVORABLE" and "HIGH_POSITIVE" in news_impact:
        contradictions.append("Macro UNFAVORABLE but news HIGH_POSITIVE")
        penalty += 2

    return penalty, contradictions


def compute_signal_velocity(
    current_signal: str,
    previous_signal: Optional[str] = None,
    days_since_change: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Compute signal velocity bonus/penalty based on signal direction change
    (CIO Legacy B4).

    Academic research (Womack 1996, Post-Earnings Announcement Drift) shows
    that the direction and speed of signal changes carry predictive information
    independent of the signal level. A recent upgrade from SELL to HOLD is
    more informative than a stock that has been BUY for 6 months.

    Args:
        current_signal: Current signal character ("B", "H", "S", "I")
        previous_signal: Previous signal character (None = no history)
        days_since_change: Days since signal changed (None = unknown)

    Returns:
        (conviction_adjustment, velocity_label)
    """
    if previous_signal is None or days_since_change is None:
        return (0, "NO_HISTORY")

    upgrade_map = {"S": 0, "I": 1, "H": 2, "B": 3}
    current_rank = upgrade_map.get(current_signal, 1)
    previous_rank = upgrade_map.get(previous_signal, 1)
    delta = current_rank - previous_rank

    if delta > 0 and days_since_change <= 14:
        return (+5, "ACCELERATING")
    elif delta > 0 and days_since_change <= 30:
        return (+3, "IMPROVING")
    elif delta < 0 and days_since_change <= 14:
        return (-5, "DETERIORATING")
    elif delta < 0 and days_since_change <= 30:
        return (-3, "WEAKENING")
    elif delta == 0 and days_since_change > 90:
        return (-2, "STALE")
    return (0, "STABLE")


def get_earnings_surprise_adjustment(
    recent_surprise_pct: Optional[float] = None,
    consecutive_beats: int = 0,
    surprise_trajectory: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Conviction adjustment based on earnings surprise history (CIO Legacy B5).

    Post-Earnings Announcement Drift (PEAD) is one of the most robust
    anomalies in finance — stocks that beat estimates tend to continue
    outperforming for 60-90 days. Serial beaters (2+ consecutive) are
    even more predictive.

    CIO v12.0 M3: Added surprise_trajectory to catch "peak earnings"
    situations. A stock going from +2% to +8% to +15% beat (ACCELERATING)
    is qualitatively different from +15% to +8% to +2% (DECELERATING).
    Decelerating beats are a classic sell signal that the base logic misses.

    Args:
        recent_surprise_pct: Percentage beat/miss of most recent earnings.
            Positive = beat, negative = miss. None = no data.
        consecutive_beats: Number of consecutive earnings beats (0 if none).
        surprise_trajectory: Optional trajectory of surprise magnitude:
            "ACCELERATING" — beats are getting larger (bullish)
            "DECELERATING" — beats are shrinking (peak earnings warning)
            "STABLE" or None — no clear trend

    Returns:
        (conviction_adjustment, label)
    """
    if recent_surprise_pct is None:
        return (0, "NO_DATA")

    # CIO v20.0 D3: Enhanced PEAD bonus for serial beaters (+8 from +5)
    if recent_surprise_pct > 10 and consecutive_beats >= 2:
        adj, label = +8, "SERIAL_BEATER"
    elif recent_surprise_pct > 5:
        adj, label = +3, "BEAT"
    elif recent_surprise_pct < -10:
        adj, label = -5, "BIG_MISS"
    elif recent_surprise_pct < -5:
        adj, label = -3, "MISS"
    else:
        adj, label = 0, "IN_LINE"

    # CIO v12.0 M3: Trajectory modulation
    if surprise_trajectory == "ACCELERATING" and consecutive_beats >= 2:
        adj = min(adj + 2, 10)  # Updated cap from 7 to 10
        label = label + "_ACCEL"
    elif surprise_trajectory == "DECELERATING" and label in ("SERIAL_BEATER", "BEAT"):
        adj = max(adj - 2, 0)
        label = label + "_DECEL"

    return (adj, label)


def compute_dynamic_freshness(
    agent_timestamp: Optional[str] = None,
    committee_timestamp: Optional[str] = None,
) -> float:
    """
    Compute freshness multiplier based on actual data age (CIO Legacy A4).

    Static freshness weights assume all agents run at the same time. In
    practice, a committee run at market close has much fresher technical
    data than one at 6am. This function replaces static weights when
    timestamps are available.

    Args:
        agent_timestamp: ISO timestamp of when agent data was computed.
        committee_timestamp: ISO timestamp of committee run. Defaults to now.

    Returns:
        Freshness multiplier between 0.6 and 1.0.
    """
    if not agent_timestamp:
        return 1.0  # No timestamp → use static default

    from datetime import datetime

    try:
        # Handle both ISO format and date-only
        if "T" in agent_timestamp:
            agent_dt = datetime.fromisoformat(agent_timestamp.replace("Z", "+00:00").replace("+00:00", ""))
        else:
            agent_dt = datetime.strptime(agent_timestamp, "%Y-%m-%d")

        if committee_timestamp:
            if "T" in committee_timestamp:
                committee_dt = datetime.fromisoformat(committee_timestamp.replace("Z", "+00:00").replace("+00:00", ""))
            else:
                committee_dt = datetime.strptime(committee_timestamp, "%Y-%m-%d")
        else:
            committee_dt = datetime.now()

        age_hours = (committee_dt - agent_dt).total_seconds() / 3600
    except (ValueError, TypeError):
        return 1.0  # Parse error → use static default

    if age_hours <= 1:
        return 1.0
    elif age_hours <= 4:
        return 0.95
    elif age_hours <= 12:
        return 0.85
    elif age_hours <= 24:
        return 0.75
    return 0.6


def compute_adjustments(
    signal: str,
    fund_score: float,
    tech_signal: str,
    tech_momentum: int,
    rsi: float,
    macro_fit: str,
    census_alignment: str,
    div_score: int,
    census_ts: str,
    news_impact: str,
    risk_warning: bool,
    buy_pct: int,
    excess_exret: float,
    beta: float,
    quality_trap: bool,
    sector: str,
    sector_rankings: Dict[str, Any],
    bull_count: int,
) -> Tuple[int, int, Dict[str, int]]:
    """
    Compute bonus and penalty adjustments.

    Returns (bonuses, penalties, waterfall) all capped per skill spec.
    The waterfall dict tracks each modifier's contribution for attribution.
    """
    bonuses = 0
    penalties = 0
    _w: Dict[str, int] = {}

    # Agent agreement bonus
    if bull_count >= 6:
        bonuses += 15
        _w["agent_consensus"] = 15
    elif bull_count >= 5:
        bonuses += 10
        _w["agent_consensus"] = 10

    # Consensus warning — tiered by excess EXRET
    # CIO v16.0 W6: Smoothed consensus cliff. Previously a 7-point jump at
    # excess_exret=0 penalized stocks marginally below sector median (e.g.
    # -0.9%) the same as those far below (-25%). Added intermediate tier
    # for excess_exret in [-10, 0) to smooth the penalty gradient.
    if buy_pct > 90:
        if excess_exret >= 12:
            penalties += 5
            _w["consensus_crowded"] = -5
        elif excess_exret >= 0:
            penalties += 8
            _w["consensus_crowded"] = -8
        elif excess_exret >= -10:
            penalties += 11
            _w["consensus_crowded"] = -11
        else:
            penalties += 15
            _w["consensus_crowded"] = -15

    # Census alignment
    if -20 <= div_score <= 20:
        bonuses += 5
        _w["census_alignment"] = 5
    elif div_score < -20:
        bonuses += 8
        _w["census_alignment"] = 8

    # Census time-series
    if census_ts == "strong_accumulation":
        bonuses += 3
        _w["census_accumulation"] = 3
    elif census_ts in ("strong_distribution", "distribution"):
        penalties += 5
        _w["census_distribution"] = -5

    # News catalyst
    if "HIGH_POSITIVE" in news_impact:
        bonuses += 5
        _w["news_catalyst_pos"] = 5
    if "NEGATIVE" in news_impact:
        penalties += 5
        _w["news_catalyst_neg"] = -5

    # Technical overbought / oversold
    # CIO v18.0 B2: RSI contrarian weighting from backtest evidence.
    # RSI < 25 = +2.70% avg, 87% positive (strongest single predictor).
    # RSI > 75 = -4.12%, 25% positive. Previously only penalized overbought
    # with -5 and had no oversold bonus. Now: oversold bonus +10, overbought
    # penalty increased to -8.
    if rsi < 30:
        bonuses += 10
        _w["rsi_oversold"] = 10
    elif rsi > 70:
        penalties += 8
        _w["rsi_overbought"] = -8

    # Tech disagreement penalty (CIO v5.2): when BUY signal but tech says AVOID/EXIT
    # CIO v14.0 V2: Scale by RSI context. At RSI < 35, the tech AVOID is confirming
    # an oversold condition — it's NOT a genuine disagreement with the BUY signal.
    # The full -8 penalty should only apply when RSI > 50 (tech disagreeing at
    # a neutral-to-overbought level is genuinely informative).
    if signal == "B" and tech_signal in ("AVOID", "EXIT_SOON"):
        if rsi < 35:
            penalties += 2   # V2: oversold — tech confirms obvious, minimal penalty
            _w["tech_disagree"] = -2
        elif rsi < 50:
            penalties += 5   # V2: transitional — moderate penalty
            _w["tech_disagree"] = -5
        else:
            penalties += 8   # Full disagreement at neutral/overbought RSI
            _w["tech_disagree"] = -8
    elif signal == "B" and tech_momentum < -30:
        penalties += 5
        _w["tech_momentum_neg"] = -5

    # Macro sector-specific
    if macro_fit == "UNFAVORABLE":
        if sector in ("Financials", "Consumer Discretionary"):
            penalties += 10
            _w["macro_sector"] = -10
        else:
            penalties += 5
            _w["macro_sector"] = -5
    elif macro_fit == "FAVORABLE":
        bonuses += 5
        _w["macro_sector"] = 5

    # High beta
    if beta > 2.0:
        penalties += 5
        _w["high_beta"] = -5

    # Quality trap
    if quality_trap:
        penalties += 5
        _w["quality_trap"] = -5

    # Sector rotation
    etf = SECTOR_ETF_MAP.get(sector, "")
    if etf and etf in sector_rankings:
        rank = sector_rankings[etf].get("rank", 6)
        ret_1m = sector_rankings[etf].get("return_1m", 0)
        if rank <= 3 and ret_1m > 0:
            bonuses += 5
            _w["sector_rotation"] = 5
        elif rank >= 9 and ret_1m < -3:
            penalties += 5
            _w["sector_rotation"] = -5

    # SELL-specific: tech/macro disagreement reduces sell conviction
    if signal == "S":
        if tech_signal in ("ENTER_NOW", "WAIT_FOR_PULLBACK"):
            penalties += 5
            _w["sell_tech_disagree"] = -5
        if macro_fit == "FAVORABLE":
            penalties += 3
            _w["sell_macro_disagree"] = -3
        if census_alignment == "CENSUS_DIV" and div_score < -30:
            penalties += 5
            _w["sell_census_disagree"] = -5

    # CIO v20.0 D5: Regional calibration (apply before cap)
    # Note: This is applied in synthesize_stock where ticker is available

    # Cap adjustments
    _pre_cap_b = bonuses
    _pre_cap_p = penalties
    bonuses = min(bonuses, 20)
    penalties = min(penalties, 25)
    if _pre_cap_b > 20:
        _w["base_bonus_cap"] = _pre_cap_b - 20  # Positive = bonuses absorbed by cap
    if _pre_cap_p > 25:
        _w["base_penalty_cap"] = _pre_cap_p - 25

    return bonuses, penalties, _w


def apply_conviction_floors(
    conviction: int,
    signal: str,
    excess_exret: float,
    pef: float,
    pet: float,
    bull_count: int,
    fund_score: float,
    buy_pct: int,
) -> int:
    """Apply conviction floors to prevent unreasonable suppression.

    CIO v13.0 F1: Graduated floor system replaces the blunt 55 floor.
    Previously, three overlapping conditions ALL mapped to floor=55, causing
    42% of stocks to cluster at exactly 55. Now the floor scales from 43 to
    55 based on how many quality criteria are met, preserving differentiation
    within the floor-touched population while still protecting quality BUY stocks.
    """
    # BUY signal unconditional floor
    if signal == "B":
        conviction = max(conviction, 40)

    # CIO v13.0 F1: Graduated quality floor for BUY signals.
    # Count quality criteria met, scale floor from 43 (1 criterion) to 55 (5).
    if signal == "B":
        quality_hits = 0
        if excess_exret > 20 and pef > 0 and pet > 0 and pef < pet:
            quality_hits += 1
        if bull_count >= 5:
            quality_hits += 2  # Strong agreement is worth more
        elif bull_count >= 4:
            quality_hits += 1
        if fund_score >= 80:
            quality_hits += 2  # Strong fundamentals worth more
        elif fund_score >= 70:
            quality_hits += 1
        if buy_pct >= 80:
            quality_hits += 1
        elif buy_pct >= 70:
            quality_hits += 1

        # Floor scales: 0 hits = 40 (base), 1 = 43, 2 = 46, 3 = 49, 4 = 52, 5+ = 55
        if quality_hits > 0:
            graduated_floor = min(55, 40 + quality_hits * 3)
            conviction = max(conviction, graduated_floor)

    return max(0, min(100, conviction))


def determine_action(
    conviction: int,
    signal: str,
    tech_signal: str,
    risk_warning: bool,
) -> str:
    """Determine action from conviction and signal with signal-aware thresholds.

    Returns one of exactly 5 actions: BUY, ADD, HOLD, TRIM, SELL.
    - BUY is only assigned later for new opportunities (not in portfolio).
    - ADD = increase existing position (high-conviction BUY-signal stocks).
    - SELL = close position (high-conviction SELL-signal stocks).
    - TRIM = reduce position (lower-conviction SELL-signal or weak HOLD).
    """
    if signal == "S":
        if conviction >= 60:
            return "SELL"
        return "TRIM"
    elif signal == "B":
        if conviction >= 55:
            return "ADD"
        return "HOLD"
    else:  # HOLD signal
        if conviction >= 70:
            return "ADD"
        elif conviction >= 35:
            return "HOLD"
        return "TRIM"


def recalculate_trim_conviction(
    tech_signal: str,
    macro_fit: str,
    risk_warning: bool,
    beta: float,
    rsi: float,
    fund_score: float,
    census_alignment: str,
) -> int:
    """Recalculate conviction for TRIM actions to reflect trim confidence.

    CIO v13.0 F4: Apply diminishing returns to trim factors so conviction
    doesn't inflate unchecked. Each successive factor contributes less,
    preventing TRIM conviction from systematically exceeding BUY conviction.
    """
    trim_conv = 50

    # Collect positive trim factors with diminishing contribution
    trim_factors = []
    if tech_signal in ("EXIT_SOON", "AVOID"):
        trim_factors.append(12)  # Primary signal (reduced from 15)
    if macro_fit == "UNFAVORABLE":
        trim_factors.append(8)   # Reduced from 10
    if risk_warning:
        trim_factors.append(7)   # Reduced from 10
    if beta > 1.5:
        trim_factors.append(4)
    if rsi > 70:
        trim_factors.append(4)

    # Diminishing returns: sort largest first, each successive factor at 70%
    trim_factors.sort(reverse=True)
    weight = 1.0
    for factor in trim_factors:
        trim_conv += int(factor * weight)
        weight *= 0.7

    # Mitigating factors (unchanged)
    if fund_score >= 80:
        trim_conv -= 10
    if census_alignment == "ALIGNED":
        trim_conv -= 5

    return min(trim_conv, 80)


def classify_hold_tier(conviction: int) -> str:
    """Classify HOLD stocks into sub-tiers."""
    if conviction >= 55:
        return "STRONG"
    elif conviction >= 40:
        return "STANDARD"
    return "WEAK"


def synthesize_stock(
    ticker: str,
    sig_data: Dict[str, Any],
    fund_data: Dict[str, Any],
    tech_data: Dict[str, Any],
    macro_fit: str,
    census_alignment: str,
    div_score: int,
    census_ts_trend: str,
    news_impact: str,
    risk_warning: bool,
    sector: str,
    sector_median_exret: float,
    sector_rankings: Dict[str, Any],
    position_limit: float,
    regime: str = "",
    previous_signal: Optional[str] = None,
    days_since_signal_change: Optional[int] = None,
    earnings_surprise_pct: Optional[float] = None,
    consecutive_earnings_beats: int = 0,
    kill_thesis_triggered: bool = False,
    regime_momentum: str = "STABLE",
    fg_score: Optional[float] = None,
    earnings_days_away: Optional[int] = None,
    is_opportunity: bool = False,
    fx_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Synthesize a single stock through the full conviction scoring pipeline.

    This is the core function that replaces the ad-hoc /tmp scripts.
    """
    def _num(v, default=0):
        if v is None:
            return default
        try:
            return float(str(v).replace('%', '').replace(',', ''))
        except (ValueError, TypeError):
            return default

    signal = sig_data["signal"]
    exret = _num(sig_data.get("exret"), 0)
    buy_pct = _num(sig_data.get("buy_pct"), 0)
    beta = _num(sig_data.get("beta"), 0)
    pet = _num(sig_data.get("pet"), 0)
    pef = _num(sig_data.get("pef"), 0)

    # Fund score with fallback (guard against explicit None values)
    fund_score = fund_data.get("fundamental_score", 50)
    if fund_score is None:
        fund_score = 50
    quality_trap = fund_data.get("quality_trap_warning", False)
    pe_traj = fund_data.get("pe_trajectory", "stable")
    if pe_traj == "stable" and pet > 0 and pef > 0:
        if pef < pet * 0.8:
            pe_traj = "strong_improvement"
        elif pef < pet:
            pe_traj = "improving"
        elif pef > pet * 1.1:
            pe_traj = "deteriorating"

    fund_view = "BUY" if fund_score >= 70 else ("HOLD" if fund_score >= 50 else "SELL")

    # Tech data
    tech_mom = tech_data.get("momentum_score", 0)
    tech_signal = tech_data.get("timing_signal", "HOLD")
    rsi = tech_data.get("rsi", 50)
    macd_sig = tech_data.get("macd_signal", "NEUTRAL")

    # CIO v22.0 E1: Enhanced technical indicators from v3.0 analysis
    adx = tech_data.get("adx")
    divergence = tech_data.get("divergence")
    stop_losses = tech_data.get("stop_losses")

    # CIO v22.0 E2: Piotroski F-Score from enhanced fundamental analysis
    piotroski = fund_data.get("piotroski")
    piotroski_score = piotroski.get("f_score", 5) if isinstance(piotroski, dict) else None

    # Sector-relative EXRET
    excess_exret = exret - sector_median_exret

    # CIO Legacy A2: Track synthetic agent data
    fund_synthetic = fund_data.get("synthetic", False) or not fund_data.get("fundamental_score")
    tech_synthetic = tech_data.get("synthetic", False)

    # Step 1: Count agent votes
    bull_weight, bear_weight, dir_confidence = count_agent_votes(
        fund_score, tech_signal, tech_mom, macro_fit,
        census_alignment, news_impact, risk_warning, signal,
        fund_synthetic=fund_synthetic,
        tech_synthetic=tech_synthetic,
        regime=regime,
        census_div_score=float(div_score) if div_score else 0.0,
        rsi=rsi,
        buy_pct=float(buy_pct),
    )

    total_weight = bull_weight + bear_weight
    bull_pct = bull_weight / total_weight * 100 if total_weight > 0 else 50
    bear_ratio = bear_weight / total_weight if total_weight > 0 else 0

    # Step 2: Determine base conviction
    base = determine_base_conviction(
        bull_pct, signal, fund_score, excess_exret, bear_ratio,
        regime=regime,
        regime_momentum=regime_momentum,
    )

    # Step 3: Count directional agreement
    bull_count = sum(1 for x in [
        fund_score >= 70,
        tech_signal == "ENTER_NOW" or tech_mom > 20,
        macro_fit == "FAVORABLE",
        census_alignment == "ALIGNED",
        "POSITIVE" in news_impact,
        not risk_warning,
    ] if x)

    # Step 4: Compute adjustments
    bonuses, penalties, _w_base = compute_adjustments(
        signal, fund_score, tech_signal, tech_mom, rsi,
        macro_fit, census_alignment, div_score, census_ts_trend,
        news_impact, risk_warning, buy_pct, excess_exret,
        beta, quality_trap, sector, sector_rankings, bull_count,
    )

    # ── CIO v25.0: Conviction Attribution Waterfall ──────────────────────
    # Track every modifier's contribution for transparent audit trail.
    # Only non-zero deltas are recorded. Positive = bonus, negative = penalty.
    # Merge base modifiers from compute_adjustments into the waterfall.
    _w = dict(_w_base)
    _adj_bonuses = bonuses   # Snapshot base adjustments before individual modifiers
    _adj_penalties = penalties

    # Step 4b: Signal quality penalties — separate cap from base penalties.
    # CIO v6.0 H5: Base penalties (from compute_adjustments) are capped at 25.
    # Signal quality indicators (contradiction, velocity, earnings, directional
    # confidence) were previously absorbed into the same 25-point cap, meaning
    # they had ZERO effect when base penalties were saturated. This hid
    # meaningful risk information. Now, signal quality penalties get their own
    # cap of 10, allowing a combined maximum of 35 (25 base + 10 quality).
    signal_quality_penalty = 0

    # Contradiction penalty (CIO Legacy A3)
    contradiction_penalty, contradictions = detect_contradictions(
        macro_fit, tech_signal, fund_score, risk_warning,
        census_alignment, news_impact,
        buy_pct=buy_pct,  # CIO v14.0 V3: consensus override
    )
    signal_quality_penalty += contradiction_penalty
    if contradiction_penalty:
        _w["contradiction"] = -contradiction_penalty

    # Step 4c: Signal velocity (CIO Legacy B4)
    velocity_adj, velocity_label = compute_signal_velocity(
        signal, previous_signal, days_since_signal_change,
    )
    if velocity_adj > 0:
        bonuses = min(bonuses + velocity_adj, 20)
        _w["signal_velocity"] = velocity_adj
    elif velocity_adj < 0:
        signal_quality_penalty += abs(velocity_adj)
        _w["signal_velocity"] = velocity_adj

    # Step 4d: Earnings surprise (CIO Legacy B5)
    earnings_adj, earnings_label = get_earnings_surprise_adjustment(
        earnings_surprise_pct, consecutive_earnings_beats,
    )
    if earnings_adj > 0:
        bonuses = min(bonuses + earnings_adj, 20)
        _w["earnings_surprise"] = earnings_adj
    elif earnings_adj < 0:
        signal_quality_penalty += abs(earnings_adj)
        _w["earnings_surprise"] = earnings_adj

    # CIO v20.0 D5: Regional calibration
    # European stocks have 38.6% hit rate, HK has 66.4%
    ticker_str = str(ticker)
    is_european = any(ticker_str.endswith(s) for s in (
        '.DE', '.L', '.PA', '.AS', '.MI', '.MC', '.BR', '.OL', '.ST', '.HE', '.CO'
    ))
    is_hk = ticker_str.endswith('.HK')
    if is_european and signal == "B":
        penalties += 5  # European BUY discount
        _w["regional_EU"] = -5
    elif is_hk and signal == "B":
        bonuses = min(bonuses + 3, 20)  # HK premium
        _w["regional_HK"] = 3

    # CIO v20.0 D4: Stale analyst target detection
    # Type A = general analyst consensus (>6 months old), no momentum change
    am = sig_data.get("am", 0)  # analyst momentum from signal data
    analyst_type = sig_data.get("analyst_type", "A")
    if analyst_type == "A" and abs(am) < 1 and buy_pct > 70:
        penalties += 5  # Stale targets
        _w["stale_targets"] = -5

    # CIO v20.0 D2: Census contrarian indicator
    # When F&G > 80 (EXTREME_GREED), reduce all new BUY conviction
    if fg_score is not None and signal == "B":
        if fg_score > 80:
            fg_penalty = int((fg_score - 80) * 0.75)  # -15 at F&G=100
            penalties += fg_penalty
            _w["census_fear_greed"] = -fg_penalty
        elif fg_score < 20:
            fg_bonus = int((20 - fg_score) * 0.75)  # +15 at F&G=0
            bonuses = min(bonuses + fg_bonus, 20)
            _w["census_fear_greed"] = fg_bonus

    # CIO v20.0 D3: Earnings proximity guard
    if earnings_days_away is not None and signal == "B":
        if earnings_days_away <= 3:
            penalties += 15  # Earnings imminent critical
            _w["earnings_proximity"] = -15
        elif earnings_days_away <= 7:
            penalties += 12  # Near-binary event
            _w["earnings_proximity"] = -12

    # CIO v20.0 D7: Macro regime threshold tightening for new BUYs
    if regime == "RISK_OFF" and signal == "B" and is_opportunity:
        # In RISK_OFF, new positions need higher bar
        if excess_exret < 20:  # Normally 6-15% depending on tier
            penalties += 10
            _w["macro_regime_risk_off"] = -10

    # CIO v22.0 E3: ADX trend strength adjustment.
    # ADX > 30 with matching entry signal = trend confirmed, boost confidence.
    # ADX < 15 with extreme momentum = ranging market, signals unreliable.
    if adx is not None:
        if adx >= 30 and tech_signal in ("ENTER_NOW",):
            bonuses = min(bonuses + 5, 20)  # Strong trend confirms entry
            _w["adx_trend_confirm"] = 5
        elif adx >= 30 and tech_signal in ("AVOID", "EXIT_SOON"):
            penalties += 3  # Strong downtrend confirms exit
            _w["adx_downtrend"] = -3
        elif adx < 15 and abs(tech_mom) > 30:
            # Ranging market — extreme momentum is unreliable
            signal_quality_penalty += 3
            _w["adx_ranging"] = -3

    # CIO v22.0 E4: RSI divergence — highest-value reversal signal.
    # Bullish divergence (price lower low, RSI higher low) in oversold
    # territory is one of the strongest quantitative buy signals.
    # Bearish divergence in overbought is a strong sell warning.
    if divergence == "bullish" and rsi < 35:
        bonuses = min(bonuses + 8, 20)  # Bullish reversal signal
        _w["rsi_divergence_bullish"] = 8
    elif divergence == "bearish" and rsi > 65:
        penalties += 5  # Bearish reversal warning
        _w["rsi_divergence_bearish"] = -5

    # CIO v22.0 E5: Piotroski quality gate — independent 9-point quality metric.
    # F >= 7 independently validates fundamental thesis.
    # F <= 3 is a quality concern even if earnings quality score looks OK.
    if piotroski_score is not None:
        if piotroski_score >= 7 and signal == "B":
            bonuses = min(bonuses + 3, 20)  # Independent quality confirmation
            _w["piotroski_quality"] = 3
        elif piotroski_score <= 3:
            penalties += 3  # Weak quality fundamentals
            _w["piotroski_weak"] = -3

    # ── CIO v21.0 R1-R5: Codified conviction modifiers ────────────────────
    # Previously these were prompt-only instructions in AGENT.md. Codifying
    # them ensures deterministic, testable application every run.

    # R1: Relative strength vs SPY confirmation.
    # Stocks outperforming SPY over 3 months deserve a confidence boost;
    # underperformers with BUY signals face headwind — trend is not their friend.
    rs_vs_spy = tech_data.get("relative_strength_vs_spy")
    if rs_vs_spy is not None and isinstance(rs_vs_spy, (int, float)):
        if rs_vs_spy > 1.0 and signal == "B":
            bonuses = min(bonuses + 3, 20)  # Outperforming SPY — momentum confirmation
            _w["rel_strength_spy"] = 3
        elif rs_vs_spy < 0.80 and signal == "B":
            penalties += 8  # Severe underperformer — swimming against the tide
            _w["rel_strength_spy"] = -8
        elif rs_vs_spy < 0.95 and signal == "B":
            penalties += 5  # Underperforming SPY — headwind for BUY thesis
            _w["rel_strength_spy"] = -5

    # R2: Revenue growth trajectory.
    # Accelerating revenue growth is the single strongest fundamental signal
    # for equity re-rating. Declining growth even with good earnings is a trap.
    rev_growth = fund_data.get("revenue_growth", {})
    rev_class = rev_growth.get("classification", "") if isinstance(rev_growth, dict) else ""
    if rev_class and signal == "B":
        if rev_class.upper() in ("ACCELERATING", "STRONG_GROWTH"):
            bonuses = min(bonuses + 5, 20)
            _w["revenue_growth"] = 5
        elif rev_class.upper() in ("DECLINING", "DETERIORATING", "NEGATIVE"):
            penalties += 5
            _w["revenue_growth"] = -5

    # R3: Dividend cut risk — yield trap detection.
    # High dividend yield with deteriorating free cash flow is a classic trap.
    # Only penalize when FCF does NOT support the dividend payout.
    div_sustainability = fund_data.get("dividend_sustainability", {})
    if isinstance(div_sustainability, dict):
        div_y = div_sustainability.get("dividend_yield", 0) or 0
        fcf_y = div_sustainability.get("fcf_yield", 0) or 0
        if div_y > 4.0 and fcf_y < div_y * 0.5 and signal in ("B", "H"):
            penalties += 5  # Yield trap — dividend exceeds FCF capacity
            _w["dividend_yield_trap"] = -5

    # R4: ATR-based entry timing — high volatility stocks need pullback entries.
    # ATR% > 5% means daily swings are large enough that entering at market
    # price risks immediate 5%+ drawdown. Wait for pullback to limit-order entry.
    # NOTE: Applied after entry_timing initialization below (see "R4 late override").
    atr_pct = tech_data.get("atr_pct")
    _r4_override = (
        atr_pct is not None
        and isinstance(atr_pct, (int, float))
        and atr_pct > 5.0
        and signal == "B"
        and tech_signal == "ENTER_NOW"
    )

    # R5: Liquidity-adjusted position sizing — applied after compute_position_size().
    avg_volume = tech_data.get("volume", {})
    vol_avg = avg_volume.get("avg_volume") if isinstance(avg_volume, dict) else None
    current_price = sig_data.get("price", 0) or tech_data.get("price", 0)
    _r5_illiquid = False
    if vol_avg and current_price:
        try:
            daily_dollar_volume = float(vol_avg) * float(current_price)
            _r5_illiquid = daily_dollar_volume < 5_000_000
        except (ValueError, TypeError):
            pass
    # ── End CIO v21.0 R1-R5 ─────────────────────────────────────────────

    # ── CIO v23.0 S1-S2: Quick-win conviction modifiers ────────────────

    # S1: Short interest as conviction modifier.
    # High SI + BUY + bullish tech = squeeze potential (+3).
    # High SI + deteriorating fundamentals = confirms bear thesis (-3).
    short_interest = _num(sig_data.get("short_interest"), 0)
    if short_interest > 10:
        if signal == "B" and tech_signal in ("ENTER_NOW", "WAIT_FOR_PULLBACK"):
            bonuses = min(bonuses + 3, 20)  # Short squeeze potential
            _w["short_squeeze"] = 3
        elif fund_score < 50 or quality_trap:
            penalties += 3  # SI confirms fundamental weakness
            _w["short_interest_weakness"] = -3

    # S2: Target price dispersion as signal quality indicator.
    # Wide analyst disagreement = uncertain thesis. Narrow consensus = confidence.
    # CSV has no high/low targets, so we derive implied CV from buy_pct and
    # num_targets (same approach as opportunity_scanner.compute_target_dispersion).
    num_targets = _num(sig_data.get("num_targets"), 0)
    target_dispersion = None
    if num_targets >= 3 and buy_pct > 0:
        implied_cv = max(0.05, (100 - buy_pct) / 200)  # 0.05 to 0.50
        target_dispersion = round(implied_cv * 100, 1)  # as percentage
        if implied_cv > 0.30:
            signal_quality_penalty += 2  # Analysts disagree widely
            _w["target_dispersion_wide"] = -2
        elif implied_cv < 0.10 and signal in ("B", "H"):
            bonuses = min(bonuses + 2, 20)  # Strong analyst consensus
            _w["target_consensus"] = 2

    # ── End CIO v23.0 S1-S2 ────────────────────────────────────────────

    # ── CIO v23.1 R6: Currency risk modifier ────────────────────────────
    # Home currency EUR. Penalize when EUR strengthens against stock's currency
    # (EUR-denominated positions lose value when converted back to EUR).
    # Crypto is exempt — no traditional FX risk framework applies.
    currency_zone = sig_data.get("currency_zone") or _infer_currency_zone(ticker)
    fx_impact = "NEUTRAL"
    fx_pairs = (fx_data or {}).get("pairs", {})

    if currency_zone == "EUR":
        fx_impact = "HOME"  # No FX risk for home currency
    elif currency_zone == "CRYPTO":
        fx_impact = "EXEMPT"  # Crypto exempt from traditional FX analysis
    elif currency_zone == "USD":
        eur_usd = fx_pairs.get("EUR/USD", {})
        eur_chg = eur_usd.get("change_1m")
        if eur_chg is not None:
            if eur_chg > 5:
                penalties += 5  # Strong EUR headwind for USD holdings
                fx_impact = "STRONG_HEADWIND"
                _w["currency_risk_USD"] = -5
            elif eur_chg > 2:
                penalties += 3  # Moderate EUR headwind
                fx_impact = "HEADWIND"
                _w["currency_risk_USD"] = -3
            elif eur_chg < -3:
                fx_impact = "TAILWIND"  # EUR weakening = USD holdings gain
            elif eur_chg < -1:
                fx_impact = "MILD_TAILWIND"
    elif currency_zone in ("GBP", "HKD", "JPY", "CHF"):
        pair_key = f"{currency_zone}/EUR"
        pair_data = fx_pairs.get(pair_key, {})
        pair_chg = pair_data.get("change_1m")
        if pair_chg is not None:
            if pair_chg < -3:
                penalties += 4  # Foreign currency weakening vs EUR — double exposure
                fx_impact = "HEADWIND"
                _w[f"currency_risk_{currency_zone}"] = -4
            elif pair_chg < -1:
                penalties += 2
                fx_impact = "MILD_HEADWIND"
                _w[f"currency_risk_{currency_zone}"] = -2
            elif pair_chg > 2:
                fx_impact = "TAILWIND"

    # ── End CIO v23.1 R6 ────────────────────────────────────────────────

    # ── CIO v23.3: Multi-timeframe confluence ─────────────────────────
    confluence_score = _num(tech_data.get("confluence_score"), 0)
    if confluence_score >= 5 and signal == "B":
        bonuses = min(bonuses + 5, 20)  # Strong alignment across timeframes
        _w["confluence"] = 5
    elif confluence_score <= -3:
        signal_quality_penalty += 3  # Timeframes conflicting
        _w["confluence_conflict"] = -3

    # ── CIO v23.3: Volume confirmation ────────────────────────────────
    obv_trend = tech_data.get("obv_trend")
    rel_volume = _num(tech_data.get("relative_volume"), 0)
    if rel_volume > 1.5 and obv_trend == "RISING" and signal == "B":
        bonuses = min(bonuses + 3, 20)  # Volume confirming price
        _w["volume_confirm"] = 3
    elif rel_volume > 1.5 and obv_trend == "FALLING" and signal == "S":
        bonuses = min(bonuses + 2, 20)  # Volume confirming sell
        _w["volume_confirm_sell"] = 2

    # ── CIO v23.3: EPS revision tracking ─────────────────────────────
    eps_revisions = fund_data.get("eps_revisions") or {}
    eps_class = eps_revisions.get("classification", "")
    if eps_class == "REVISIONS_UP" and signal == "B":
        bonuses = min(bonuses + 3, 20)
        _w["eps_revisions_up"] = 3
    elif eps_class == "REVISIONS_DOWN":
        penalties += 3
        _w["eps_revisions_down"] = -3

    # ── CIO v25.0: IV rank × earnings interaction ───────────────────────
    # Previously flat +3 penalty (v23.4). Now scales by IV rank magnitude:
    # IV rank measures where current implied volatility sits relative to its
    # 52-week range. At IV>80, options are pricing in extreme moves — entering
    # within 7 days of earnings at these levels means buying peak uncertainty.
    # The penalty scales: 3 at IV=80, 6 at IV=90, 9 at IV=100.
    iv_rank = _num(tech_data.get("iv_rank"), None)
    if iv_rank is not None:
        if iv_rank > 80 and earnings_days_away is not None and earnings_days_away <= 7:
            iv_earn_penalty = 3 + int((iv_rank - 80) * 0.3)  # 3-9 scaled
            penalties += iv_earn_penalty
            _w["iv_x_earnings"] = -iv_earn_penalty
        elif iv_rank < 20 and signal == "B":
            bonuses = min(bonuses + 2, 20)  # Low IV = cheap entry
            _w["iv_low_entry"] = 2

    # ── CIO v23.4: Volatility regime sizing — applied after compute_position_size().
    vol_regime = tech_data.get("volatility_regime")

    # ── CIO v23.4: FCF quality modifier ──────────────────────────────
    fcf_quality = fund_data.get("fcf_quality") or {}
    if fcf_quality.get("concern") == "EARNINGS_QUALITY_CONCERN" and signal == "B":
        penalties += 3
        _w["fcf_quality_concern"] = -3
    elif fcf_quality.get("classification") == "STRONG" and signal == "B":
        bonuses = min(bonuses + 2, 20)
        _w["fcf_quality_strong"] = 2

    # ── CIO v23.4: Debt quality modifier ─────────────────────────────
    debt_quality = fund_data.get("debt_quality") or {}
    if debt_quality.get("risk") == "HIGH_RISK" and signal == "B":
        penalties += 4
        _w["debt_high_risk"] = -4

    # CIO Legacy A5: Directional confidence penalty
    # CIO v7.0 P2: dir_confidence is now from the return value (see Step 1)
    # CIO v11.0 L6: Skip when both agents are synthetic. Low directional
    # confidence from lack of coverage is fundamentally different from low
    # confidence from agent disagreement. Penalizing stocks that agents
    # simply don't cover (HK, EU, ME tickers) adds noise, not signal.
    if dir_confidence < 0.4 and not (fund_synthetic and tech_synthetic):
        signal_quality_penalty += 3
        _w["low_dir_confidence"] = -3

    # Cap signal quality penalties separately from base penalties
    _pre_sq_cap = signal_quality_penalty
    signal_quality_penalty = min(signal_quality_penalty, 10)
    penalties = penalties + signal_quality_penalty

    # CIO v25.0: Record signal quality cap effect in waterfall
    if _pre_sq_cap > 10:
        _w["signal_quality_cap"] = _pre_sq_cap - 10  # Positive = penalties absorbed by cap

    # CIO v14.0 V5: Penalty proportionality cap. After 13 review iterations,
    # penalty sources now include: base (25), quality (10), sector conc (6-15),
    # kill thesis (15). Total can reach 50+ points against max 20 bonus.
    # This 2.5:1 asymmetry systematically suppresses conviction. Cap total
    # effective penalties at 60% of base to ensure the base signal still
    # carries through. For base=68 with +13 bonuses, max penalty = 41,
    # giving min conviction = 68+13-41 = 40 (still meaningful differentiation).
    _pre_cap_penalties = penalties
    max_penalty = int(base * 0.60)
    penalties = min(penalties, max_penalty)
    if _pre_cap_penalties > max_penalty:
        _w["proportionality_cap"] = _pre_cap_penalties - max_penalty

    conviction = base + bonuses - penalties

    # Step 4e: Kill thesis penalty (CIO v6.0 E1)
    # When a previously logged kill thesis has triggered, apply -15 penalty
    # that BYPASSES the normal penalty cap. A triggered kill thesis represents
    # a specific, pre-identified failure mode — it is categorically different
    # from generic agent disagreement and deserves uncapped treatment.
    if kill_thesis_triggered:
        conviction -= 15
        _w["kill_thesis"] = -15

    # Step 5: Apply floors
    # CIO v12.0 R1: When kill thesis is triggered, only apply the unconditional
    # BUY floor (40), NOT quality floors (55). Quality floors exist to prevent
    # penalty stacking from suppressing quant-verified BUY stocks, but a
    # triggered kill thesis represents a *specific, pre-identified failure mode*
    # that should override quality metrics. Without this guard, a stock with
    # triggered kill thesis gets quality-floored back to 55 and recommended as
    # ADD — directly contradicting the purpose of kill thesis monitoring.
    _pre_floor = conviction
    if kill_thesis_triggered and signal == "B":
        conviction = max(conviction, 40)
        conviction = min(conviction, 100)
    else:
        conviction = apply_conviction_floors(
            conviction, signal, excess_exret, pef, pet,
            bull_count, fund_score, buy_pct,
        )
    if conviction > _pre_floor:
        _w["floor_applied"] = conviction - _pre_floor

    # Step 5b: RISK_OFF conviction cap (CIO v18.0 B3 + v19.0 R2)
    # Backtest evidence (Mar 16-25): conv 80-85 = -6.30% avg in RISK_OFF.
    # In bear markets, systemic risk makes all high-conviction calls unreliable.
    #
    # CIO v19.0 R2: Variable cap based on regime momentum.
    # IMPROVING (capitulation signals, insiders buying): cap at 82 — allow
    #   adding to oversold quality names as the regime transitions.
    # STABLE: cap at 75 — original backtest-calibrated cap.
    # DETERIORATING: cap at 70 — conditions actively worsening, be more cautious.
    if regime == "RISK_OFF" and signal != "S":
        _pre_riskoff = conviction
        if regime_momentum == "IMPROVING":
            conviction = min(conviction, 82)
        elif regime_momentum == "DETERIORATING":
            conviction = min(conviction, 70)
        else:
            conviction = min(conviction, 75)
        if conviction < _pre_riskoff:
            _w["risk_off_cap"] = conviction - _pre_riskoff

    # Step 6: Determine action
    action = determine_action(conviction, signal, tech_signal, risk_warning)

    # Risk manager override: downgrade to HOLD when tech is bearish + risk warns
    if signal == "H" and tech_signal in ("AVOID", "EXIT_SOON") and risk_warning:
        if action not in ("SELL", "TRIM"):
            action = "HOLD"

    # Trim escalation: HOLD-signal stocks that are overbought or have
    # bearish tech + risk warnings should be TRIM, not stuck at HOLD.
    # This ensures stocks like PANW (RSI=93) get proper trim treatment.
    #
    # CIO v7.0 P1: RSI floor guard — NEVER trim stocks with RSI < 30.
    # At RSI < 30 the stock is deeply oversold; trimming crystallizes losses
    # at the worst possible moment. E.g., SLB at RSI=16 should not be trimmed
    # even when tech=AVOID + risk_warning — the damage is already done.
    if action == "HOLD" and signal == "H" and rsi >= 30:
        if rsi > 80 and tech_signal in ("AVOID", "EXIT_SOON"):
            action = "TRIM"
        elif risk_warning and tech_signal == "AVOID":
            action = "TRIM"

    # TRIM conviction recalculation — replace residual buy-case score with
    # trim confidence (how confident we are the position should be reduced)
    if action == "TRIM":
        conviction = recalculate_trim_conviction(
            tech_signal, macro_fit, risk_warning, beta, rsi,
            fund_score, census_alignment,
        )
        # CIO v8.0 F3: Cap TRIM conviction for low-data stocks.
        # When both fundamental and technical data are synthetic (estimated
        # from signal proxies), the TRIM thesis is built on manufactured data
        # — cap at 55 to prevent high-confidence trims without real analysis.
        if fund_synthetic and tech_synthetic:
            conviction = min(conviction, 55)
        # INCONCLUSIVE signal = insufficient analyst coverage for any thesis
        if signal == "I":
            conviction = min(conviction, 50)

    # HOLD tiering
    hold_tier = ""
    if action == "HOLD":
        hold_tier = classify_hold_tier(conviction)

    # CIO v20.0 D9: Entry timing signal
    entry_timing = tech_signal
    entry_note = ""
    if signal == "B" and tech_signal == "WAIT_FOR_PULLBACK":
        entry_note = f"Signal confirmed but wait for RSI pullback from {rsi:.0f}"
    elif signal == "B" and tech_signal == "ENTER_NOW":
        entry_note = f"Strong entry point, RSI {rsi:.0f}"
    elif signal == "B" and tech_signal in ("AVOID", "EXIT_SOON"):
        entry_note = f"Caution: tech says {tech_signal} despite BUY signal"

    # CIO v21.0 R4 late override: ATR% > 5% forces WAIT_FOR_PULLBACK
    # Applied after D9 so it overrides the default entry_timing assignment.
    if _r4_override:
        entry_timing = "WAIT_FOR_PULLBACK"
        entry_note = f"ATR% {atr_pct:.1f}% — use limit order, avoid chasing"

    # CIO v20.0 D8: Position sizing recommendation
    # Compute correlated count from beta (proxy until risk_report available)
    correlated_count = 0  # Will be updated in build_concordance
    position_size_pct = compute_position_size(conviction, beta, correlated_count, buy_pct)

    # CIO v23.4: Volatility regime sizing (post-base)
    if vol_regime == "LOW_VOL":
        position_size_pct = round(min(position_size_pct * 1.20, 5.0), 2)
    elif vol_regime in ("HIGH_VOL", "EXTREME"):
        position_size_pct = round(position_size_pct * 0.80, 2)

    # CIO v21.0 R5: Liquidity-adjusted sizing (post-base)
    if _r5_illiquid:
        position_size_pct = round(position_size_pct * 0.6, 1)

    return {
        "ticker": ticker,
        "signal": signal,
        "sector": sector,
        "fund_score": round(fund_score, 1),
        "fund_view": fund_view,
        "pe_trajectory": pe_traj,
        "quality_trap": quality_trap,
        "tech_momentum": tech_mom,
        "tech_signal": tech_signal,
        "rsi": round(rsi, 1),
        "macd": macd_sig,
        "macro_fit": macro_fit,
        "census": census_alignment,
        "div_score": div_score,
        "census_ts": census_ts_trend,
        "news_impact": news_impact,
        "risk_warning": risk_warning,
        "exret": exret,
        "excess_exret": round(excess_exret, 1),
        "buy_pct": buy_pct,
        "beta": beta,
        "bull_weight": round(bull_weight, 2),
        "bear_weight": round(bear_weight, 2),
        "bull_pct": round(bull_pct, 1),
        "base": base,
        "bonuses": bonuses,
        "penalties": penalties,
        "conviction": conviction,
        "action": action,
        "hold_tier": hold_tier,
        "max_pct": position_limit,
        "fund_synthetic": fund_synthetic,
        "tech_synthetic": tech_synthetic,
        "contradictions": contradictions,
        "signal_velocity": velocity_label,
        "earnings_surprise": earnings_label,
        "directional_confidence": round(dir_confidence, 2),
        "pef": pef,   # CIO v11.0: stored for post-penalty floor reapplication
        "pet": pet,
        "price": sig_data.get("price", 0),  # CIO v17.0: for evaluate_recent() returns
        "entry_timing": entry_timing,  # CIO v20.0 D9
        "entry_note": entry_note,  # CIO v20.0 D9
        "position_size_pct": position_size_pct,  # CIO v20.0 D8
        "days_held": 0,  # CIO v20.0 D1: will be updated in build_concordance
        "adx": adx,  # CIO v22.0 E1
        "adx_trend": tech_data.get("adx_trend"),  # CIO v22.0 E1
        "divergence": divergence,  # CIO v22.0 E4
        "piotroski_score": piotroski_score,  # CIO v22.0 E5
        "stop_losses": stop_losses,  # CIO v22.0 E1
        "rs_vs_spy": rs_vs_spy,  # CIO v21.0 R1
        "revenue_growth_class": rev_class,  # CIO v21.0 R2
        "atr_pct": atr_pct,  # CIO v21.0 R4
        "short_interest_pct": short_interest if short_interest > 0 else None,  # CIO v23.0 S1
        "target_dispersion": target_dispersion,  # CIO v23.0 S2
        "currency_zone": currency_zone,  # CIO v23.1 R6
        "fx_impact": fx_impact,  # CIO v23.1 R6
        "confluence_score": confluence_score,  # CIO v23.3
        "obv_trend": obv_trend,  # CIO v23.3
        "relative_volume": rel_volume if rel_volume else None,  # CIO v23.3
        "eps_revisions": eps_class if eps_class else None,  # CIO v23.3
        "iv_rank": iv_rank,  # CIO v23.4
        "volatility_regime": vol_regime,  # CIO v23.4
        "fcf_classification": fcf_quality.get("classification") if fcf_quality else None,  # CIO v23.4
        "debt_risk": debt_quality.get("risk") if debt_quality else None,  # CIO v23.4
        "conviction_waterfall": _w,  # CIO v25.0: full modifier attribution
    }


# Action priority for sorting (5 canonical actions only)
ACTION_ORDER = {
    "SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4,
}


def _build_agent_lookups(
    fund_report: Dict,
    tech_report: Dict,
    macro_report: Dict,
    census_report: Dict,
    news_report: Dict,
    risk_report: Dict,
) -> Dict[str, Any]:
    """Extract lookup maps from agent reports (shared by portfolio and opportunity paths)."""
    macro_impl = macro_report.get("portfolio_implications", {})
    census_divs = census_report.get("divergences", {})
    port_news = news_report.get("portfolio_news", {})
    risk_warns = {w["ticker"] for w in risk_report.get("consensus_warnings", [])}
    risk_limits = risk_report.get("position_limits", {})
    sector_rankings = macro_report.get("sector_rankings", {})

    div_map: Dict[str, Tuple[str, int]] = {}
    for item in census_divs.get("consensus_aligned", []):
        div_map[item["ticker"]] = ("ALIGNED", item.get("divergence_score", 0))
    for item in census_divs.get("signal_divergences", []):
        div_map[item["ticker"]] = ("DIVERGENT", item.get("divergence_score", 0))
    for item in census_divs.get("census_divergences", []):
        div_map[item["ticker"]] = ("CENSUS_DIV", item.get("divergence_score", 0))

    return {
        "macro_impl": macro_impl,
        "div_map": div_map,
        "port_news": port_news,
        "risk_warns": risk_warns,
        "risk_limits": risk_limits,
        "sector_rankings": sector_rankings,
        "fx_data": risk_report.get("fx_data", {}),  # CIO v23.1 R6
    }


def _resolve_macro_fit(
    macro_impl: Dict,
    ticker: str,
    sector: str = "",
    regime: str = "",
) -> str:
    """
    Normalize macro fit string to FAVORABLE/UNFAVORABLE/NEUTRAL.

    CIO Review F2: When macro agent doesn't cover a stock, derive fit
    from sector + regime mapping instead of defaulting to NEUTRAL.
    """
    mf = macro_impl.get(ticker, {})
    v = mf.get("macro_fit", "") if isinstance(mf, dict) else str(mf)
    v = v.lower()
    if "unfavorable" in v or "negative" in v:
        return "UNFAVORABLE"
    if "favorable" in v or "positive" in v:
        return "FAVORABLE"
    if v and v != "neutral":
        return "NEUTRAL"

    # F2: Sector-regime derivation when agent provides no assessment
    if regime and sector:
        sector_lower = sector.lower()
        if regime == "RISK_OFF":
            if any(kw in sector_lower for kw in [
                "health", "pharma", "medical", "utilit", "staple",
                "defense", "commodit", "gold",
            ]):
                return "FAVORABLE"
            if any(kw in sector_lower for kw in [
                "tech", "software", "semi", "consumer elec",
                "e-commerce", "social", "entertainment",
                "fintech", "banking", "financial",
            ]):
                return "UNFAVORABLE"
        elif regime == "RISK_ON":
            if any(kw in sector_lower for kw in [
                "tech", "software", "semi", "consumer",
                "e-commerce", "social", "entertainment",
            ]):
                return "FAVORABLE"
            if any(kw in sector_lower for kw in [
                "utilit", "staple",
            ]):
                return "NEUTRAL"

    return "NEUTRAL"


def _resolve_news_impact(port_news: Dict, ticker: str) -> str:
    """
    Determine aggregate news impact for a ticker.

    CIO v6.0 H1: When conflicting high-impact news exists (both HIGH_POSITIVE
    and HIGH_NEGATIVE), resolve to MIXED rather than favoring positive.
    The previous implementation's priority order created a bullish bias —
    a stock with 1 HIGH_POSITIVE and 3 HIGH_NEGATIVE items would still
    resolve to HIGH_POSITIVE.
    """
    items = port_news.get(ticker, [])
    if not items:
        return "NEUTRAL"
    impacts = [i.get("impact", "NEUTRAL") for i in items]

    has_high_pos = "HIGH_POSITIVE" in impacts
    has_high_neg = "HIGH_NEGATIVE" in impacts

    # Conflicting high-impact signals cancel out — conservative treatment
    if has_high_pos and has_high_neg:
        return "MIXED"

    for level in ("HIGH_NEGATIVE", "HIGH_POSITIVE", "LOW_NEGATIVE", "LOW_POSITIVE"):
        if level in impacts:
            return level
    return "NEUTRAL"


def _fallback_technical(sig_data: Dict) -> Dict:
    """
    Generate synthetic technical view from signal CSV data (CIO v5.2).

    When the technical agent doesn't cover a stock, use PP (price performance),
    52W (52-week high %), and beta to derive a directional view instead of
    defaulting everything to momentum=0, timing=HOLD.
    """
    def _num(v, default=0):
        if v is None:
            return default
        try:
            return float(str(v).replace('%', '').replace(',', ''))
        except (ValueError, TypeError):
            return default

    pp = _num(sig_data.get("pp"), 0)
    w52 = _num(sig_data.get("52w"), 100)

    # Derive momentum from price performance
    if pp > 15:
        momentum = 35
    elif pp > 5:
        momentum = 15
    elif pp > -5:
        momentum = 0
    elif pp > -15:
        momentum = -20
    else:
        momentum = -40

    # Derive timing from 52W (distance from high)
    if w52 >= 95:
        timing = "WAIT_FOR_PULLBACK"
        rsi_est = 68
    elif w52 >= 80:
        timing = "HOLD"
        rsi_est = 55
    elif w52 >= 65:
        timing = "HOLD"
        rsi_est = 45
    elif w52 >= 50:
        timing = "AVOID"
        rsi_est = 38
    else:
        timing = "AVOID"
        rsi_est = 30

    return {
        "momentum_score": momentum,
        "timing_signal": timing,
        "rsi": rsi_est,
        "macd_signal": "BULLISH" if pp > 0 else "BEARISH",
        "synthetic": True,
    }


def _fallback_fundamental(sig_data: Dict) -> Dict:
    """Generate synthetic fundamental score from signal data."""
    def _num(v, default=0):
        if v is None:
            return default
        try:
            return float(str(v).replace('%', '').replace(',', ''))
        except (ValueError, TypeError):
            return default

    fs = 50
    exret = _num(sig_data.get("exret"), 0)
    bp = _num(sig_data.get("buy_pct"), 0)
    pet = _num(sig_data.get("pet"), 0)
    pef = _num(sig_data.get("pef"), 0)
    if exret > 25:
        fs += 15
    elif exret > 15:
        fs += 10
    elif exret > 5:
        fs += 5
    if bp > 80:
        fs += 10
    elif bp > 60:
        fs += 5
    if pet > 0 and pef > 0 and pef < pet:
        fs += 10
    return {"fundamental_score": min(fs, 90)}


def _synthesize_with_lookups(
    ticker: str,
    sig_data: Dict,
    lookups: Dict[str, Any],
    fund_report: Dict,
    tech_report: Dict,
    sector: str,
    sec_median: float,
    census_ts_map: Dict[str, str],
    regime: str = "",
    kill_thesis_triggered: bool = False,
    previous_signal: Optional[str] = None,
    days_since_signal_change: Optional[int] = None,
    regime_momentum: str = "STABLE",
    fg_score: Optional[float] = None,
    news_report: Optional[Dict] = None,
    is_opportunity: bool = False,
) -> Dict[str, Any]:
    """Run synthesize_stock using pre-built agent lookups."""
    # CIO v6.0 E3: Log warnings when agents have stocks section but ticker
    # is missing vs when entire stocks section is absent, to distinguish
    # "agent didn't cover this stock" from "agent produced malformed output."
    fund_stocks = fund_report.get("stocks", {})
    fund_data = fund_stocks.get(ticker, {})
    if not fund_data:
        if fund_stocks:
            # Agent has data for other stocks but not this one — expected gap
            logger.debug("Fundamental agent did not cover %s (using fallback)", ticker)
        elif fund_report:
            # Agent produced a report but no 'stocks' key — likely malformed
            logger.warning(
                "Fundamental agent report missing 'stocks' key — "
                "possible schema issue (using fallback for %s)", ticker,
            )
        fund_data = _fallback_fundamental(sig_data)

    tech_stocks = tech_report.get("stocks", {})
    tech_data = tech_stocks.get(ticker, {})
    if not tech_data:
        if tech_stocks:
            logger.debug("Technical agent did not cover %s (using fallback)", ticker)
        elif tech_report:
            logger.warning(
                "Technical agent report missing 'stocks' key — "
                "possible schema issue (using fallback for %s)", ticker,
            )
        tech_data = _fallback_technical(sig_data)
    macro_fit = _resolve_macro_fit(
        lookups["macro_impl"], ticker, sector=sector, regime=regime,
    )
    cen_align, div_score = lookups["div_map"].get(ticker, ("NEUTRAL", 0))
    news_impact = _resolve_news_impact(lookups["port_news"], ticker)
    rl = lookups["risk_limits"].get(ticker, {})
    pos_limit = rl.get("max_pct", 5.0) if isinstance(rl, dict) else 5.0
    ts_trend = census_ts_map.get(ticker, "stable")

    # CIO v11.0 L3: Extract earnings surprise from fundamental agent data
    # CIO v13.0 F3: Fallback estimation from signal CSV earnings growth (EG)
    # when fundamental agent doesn't provide earnings surprise data.
    # EG (earnings growth %) is a proxy: high growth often correlates with
    # positive surprises, though this is noisier than actual surprise data.
    earnings_surprise = fund_data.get("earnings_surprise_pct")
    consecutive_beats = fund_data.get("consecutive_earnings_beats", 0)
    if earnings_surprise is None:
        eg = sig_data.get("eg") or sig_data.get("EG")
        if eg is not None:
            try:
                eg_val = float(eg)
                # Conservative mapping: only flag strong signals
                if eg_val > 25:
                    earnings_surprise = min(eg_val * 0.3, 15.0)  # Damped proxy
                    consecutive_beats = 2  # Assume continuation
                elif eg_val < -20:
                    earnings_surprise = max(eg_val * 0.3, -15.0)
                    consecutive_beats = -2
            except (ValueError, TypeError):
                pass

    # CIO v23.0: Deterministic earnings date from fundamental report (yfinance calendar).
    # Falls back to news agent web search if fundamental report has no date.
    earnings_days_away = None
    if fund_data.get("next_earnings_date"):
        try:
            from datetime import datetime as _dt
            ed = _dt.strptime(fund_data["next_earnings_date"], "%Y-%m-%d").date()
            today = _dt.now().date()
            delta = (ed - today).days
            if 0 <= delta <= 30:
                earnings_days_away = delta
        except (ValueError, TypeError):
            pass
    # CIO v20.0 D3: Fallback — extract from news report if fundamental had no date
    if earnings_days_away is None and news_report:
        earnings_cal = news_report.get("earnings_calendar", {}).get("next_2_weeks", [])
        ticker_earnings = [e for e in earnings_cal if e.get("ticker") == ticker]
        if ticker_earnings:
            earnings_days_away = ticker_earnings[0].get("days_away", 14)

    return synthesize_stock(
        ticker=ticker,
        sig_data=sig_data,
        fund_data=fund_data,
        tech_data=tech_data,
        macro_fit=macro_fit,
        census_alignment=cen_align,
        div_score=div_score,
        census_ts_trend=ts_trend,
        news_impact=news_impact,
        risk_warning=(ticker in lookups["risk_warns"]),
        sector=sector,
        sector_median_exret=sec_median,
        sector_rankings=lookups["sector_rankings"],
        position_limit=pos_limit,
        regime=regime,
        kill_thesis_triggered=kill_thesis_triggered,
        previous_signal=previous_signal,
        days_since_signal_change=days_since_signal_change,
        earnings_surprise_pct=earnings_surprise,
        consecutive_earnings_beats=consecutive_beats,
        regime_momentum=regime_momentum,
        fg_score=fg_score,
        earnings_days_away=earnings_days_away,
        is_opportunity=is_opportunity,
        fx_data=lookups.get("fx_data"),  # CIO v23.1 R6
    )


def apply_opportunity_gate(
    entry: Dict[str, Any],
    fund_report: Dict,
    tech_report: Dict,
    macro_fit: str,
    census_alignment: str,
    portfolio_sectors: Dict[str, int],
    regime: str = "",
) -> Dict[str, Any]:
    """
    Apply CIO C2 validation gate to new opportunities.

    New stocks not in the portfolio get conviction discounts unless
    confirmed by multiple non-scanner agents. Hard cap at 75 on
    first appearance.

    Also applies sector gap bonus (+10) if the opportunity fills
    a sector underrepresented in the portfolio.
    """
    ticker = entry["ticker"]
    sector = entry["sector"]

    # Count non-scanner confirmations
    confirmations = 0
    fund_data = fund_report.get("stocks", {}).get(ticker, {})
    if fund_data.get("fundamental_score", 0) >= 70:
        confirmations += 1
    tech_data = tech_report.get("stocks", {}).get(ticker, {})
    if tech_data.get("timing_signal") == "ENTER_NOW":
        confirmations += 1
    if macro_fit == "FAVORABLE":
        confirmations += 1
    if census_alignment == "ALIGNED":
        confirmations += 1

    # CIO v8.0 F6: Regime-aware graduated discount.
    # In RISK_OFF, regime adjustment already penalized conviction by 15%.
    # Applying full opportunity discount double-counts the caution,
    # making all opportunities unreachable.
    if regime == "RISK_OFF":
        discount_scale = 0.5  # Halve opportunity discount
    elif regime == "CAUTIOUS":
        discount_scale = 0.7
    else:
        discount_scale = 1.0

    if confirmations == 0:
        entry["conviction"] -= int(15 * discount_scale)
    elif confirmations == 1:
        entry["conviction"] -= int(10 * discount_scale)
    # 2+ confirmations: no discount

    # Sector gap bonus
    if sector not in portfolio_sectors or portfolio_sectors.get(sector, 0) == 0:
        entry["conviction"] += 10
        entry["fills_sector_gap"] = True
    else:
        entry["fills_sector_gap"] = False

    # Hard cap at 75 on first appearance
    entry["conviction"] = min(entry["conviction"], 75)
    entry["conviction"] = max(entry["conviction"], 0)

    # Determine action for new opportunities using conviction threshold.
    # CIO v15.0 W5: Reduced threshold from 55 to 50. Opportunities already face
    # three systematic discounts (opportunity discount -10/-15, regime discount,
    # sector concentration penalty) that don't apply to existing holdings. Requiring
    # 55 after all these discounts was equivalent to requiring ~70 raw conviction,
    # which blocked 70% of candidates. At 50, opportunities with genuine committee
    # support pass through while truly weak candidates (conv < 50) remain filtered.
    if entry["conviction"] >= 50:
        entry["action"] = "BUY"
    else:
        # Below BUY threshold: WATCH candidate.
        # CIO v23.3: Generate re-entry trigger for watchlist
        entry["action"] = "HOLD"
        triggers = []
        rsi = tech_data.get("rsi")
        if rsi and rsi > 40:
            triggers.append(f"RSI drops below 35 (currently {rsi:.0f})")
        fund_score = fund_data.get("fundamental_score", 0)
        if fund_score < 70:
            triggers.append("Fundamental score improves above 70")
        if confirmations < 2:
            triggers.append(f"Gains additional agent confirmation ({confirmations}/4 currently)")
        if entry.get("conviction", 0) >= 40:
            triggers.append("Conviction improves by 10+ pts (near threshold)")
        if not triggers:
            triggers.append("Overall conditions improve")
        entry["watch_trigger"] = "; ".join(triggers[:3])

    entry["is_opportunity"] = True
    entry["confirmations"] = confirmations

    return entry


def detect_sector_gaps(
    portfolio_sectors: Dict[str, int],
    sector_rankings: Dict[str, Any],
    portfolio_weights: Optional[Dict[str, float]] = None,
    min_meaningful_exposure: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Identify sectors underweight in the portfolio that are leading in rotation.

    CIO Legacy C1: Uses exposure-weighted detection instead of binary
    presence/absence. Having a single micro-cap in a leading sector
    (e.g., 0.6% portfolio weight in Energy) is functionally the same
    as having zero exposure.

    Args:
        portfolio_sectors: Dict of sector -> stock count
        sector_rankings: Dict of ETF -> {rank, return_1m, ...}
        portfolio_weights: Optional dict of sector -> % of portfolio.
            When provided, uses min_meaningful_exposure threshold.
            When None, falls back to count==0 detection.
        min_meaningful_exposure: Minimum sector weight (%) to not be a gap.

    Returns list of {sector, portfolio_exposure, performance_1m, rank, urgency}.
    """
    gaps = []
    for etf, data in sector_rankings.items():
        # Reverse-map ETF to sector
        sector = None
        for sec, etf_code in SECTOR_ETF_MAP.items():
            if etf_code == etf:
                sector = sec
                break
        if not sector:
            continue

        count = portfolio_sectors.get(sector, 0)
        ret_1m = data.get("return_1m", 0)
        rank = data.get("rank", 11)

        # CIO Legacy C1: Use exposure weight if available, else count
        if portfolio_weights is not None:
            exposure = portfolio_weights.get(sector, 0.0)
            is_gap = exposure < min_meaningful_exposure
        else:
            exposure = 0.0 if count == 0 else float(count)
            is_gap = count == 0

        if is_gap and rank <= 5 and ret_1m > 0:
            urgency = "HIGH" if rank <= 3 and (portfolio_weights is None or exposure < 1.0) else "MEDIUM"
            gaps.append({
                "sector": sector,
                "portfolio_exposure": round(exposure, 1),
                "performance_1m": ret_1m,
                "rank": rank,
                "urgency": urgency,
            })

    gaps.sort(key=lambda g: g["rank"])
    return gaps


def build_concordance(
    portfolio_signals: Dict[str, Dict],
    fund_report: Dict,
    tech_report: Dict,
    macro_report: Dict,
    census_report: Dict,
    news_report: Dict,
    risk_report: Dict,
    sector_map: Dict[str, str],
    census_ts_map: Optional[Dict[str, str]] = None,
    opportunity_signals: Optional[Dict[str, Dict]] = None,
    opportunity_sector_map: Optional[Dict[str, str]] = None,
    triggered_kill_theses: Optional[Dict[str, bool]] = None,
    previous_concordance: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the full concordance matrix from all 7 agent reports.

    This is the main entry point for CIO synthesis.

    Args:
        portfolio_signals: Dict of ticker -> {signal, exret, buy_pct, beta, pet, pef}
        fund_report: Fundamental agent JSON output
        tech_report: Technical agent JSON output
        macro_report: Macro agent JSON output
        census_report: Census agent JSON output
        news_report: News agent JSON output
        risk_report: Risk agent JSON output
        sector_map: Dict of ticker -> GICS sector name
        census_ts_map: Optional dict of ticker -> trend classification
        opportunity_signals: Optional dict of ticker -> signal data for new candidates
        opportunity_sector_map: Optional dict of ticker -> sector for opportunities
        previous_concordance: Optional list/dict of previous concordance for
            signal velocity computation (CIO v11.0 L2)

    Returns:
        Sorted list of concordance entries (dicts).
    """
    if census_ts_map is None:
        census_ts_map = {}
    if opportunity_signals is None:
        opportunity_signals = {}
    if opportunity_sector_map is None:
        opportunity_sector_map = {}
    if triggered_kill_theses is None:
        triggered_kill_theses = {}

    # CIO v11.0 L2 + v13.0 F2: Build previous signal map for signal velocity.
    # v13.0 F2: Fixed data flow — when previous concordance is a bare list
    # (no date field), infer the date from the concordance file mtime or
    # default to 7 days (typical committee cadence). Previously, the missing
    # date caused ALL stocks to get NO_HISTORY, making velocity dead code.
    prev_signal_map: Dict[str, Tuple[str, Optional[int]]] = {}
    if previous_concordance:
        _prev_list = previous_concordance
        _prev_date_str = None
        if isinstance(previous_concordance, dict):
            if "stocks" in previous_concordance:
                _prev_list = [dict(v, ticker=k) for k, v in previous_concordance["stocks"].items()]
            _prev_date_str = previous_concordance.get("date")
            if not _prev_date_str:
                # Try concordance list entries for date
                concordance_entries = previous_concordance.get("concordance", [])
                if concordance_entries and isinstance(concordance_entries, list):
                    _prev_list = concordance_entries
        if isinstance(_prev_list, list):
            for _pe in _prev_list:
                _t = _pe.get("ticker", "")
                _ps = _pe.get("signal", "")
                if _t and _ps:
                    # Compute days since previous committee
                    _days = None
                    _d = _prev_date_str or _pe.get("date")
                    if _d:
                        try:
                            from datetime import datetime as _dt
                            _days = (_dt.now() - _dt.strptime(str(_d)[:10], "%Y-%m-%d")).days
                        except (ValueError, TypeError):
                            pass
                    # CIO v13.0 F2: When no date is available at all, use
                    # default of 7 days (typical weekly committee cadence).
                    # This is better than NO_HISTORY because it still enables
                    # upgrade/downgrade detection — only the recency bonus/
                    # penalty may be slightly off.
                    if _days is None:
                        _days = 7
                    prev_signal_map[_t] = (_ps, _days)

    # CIO v26.1: Normalize agent report structures before processing.
    fund_report, tech_report, macro_report, census_report, news_report, risk_report = (
        normalize_agent_reports(
            fund_report, tech_report, macro_report,
            census_report, news_report, risk_report,
        )
    )

    lookups = _build_agent_lookups(
        fund_report, tech_report, macro_report,
        census_report, news_report, risk_report,
    )

    # Compute sector medians from portfolio
    sector_medians, universe_median = compute_sector_medians(
        portfolio_signals, sector_map,
    )

    # Portfolio sector counts (for gap detection)
    portfolio_sectors: Dict[str, int] = {}
    for ticker in portfolio_signals:
        sec = resolve_sector(ticker, sector_map)
        portfolio_sectors[sec] = portfolio_sectors.get(sec, 0) + 1

    concordance = []

    # Extract regime for sector-regime macro fit derivation (CIO Review F2)
    # Macro agent writes regime as a dict: {classification: "RISK_OFF", ...}
    regime_data = macro_report.get("regime", "")
    regime = regime_data.get("classification") if isinstance(regime_data, dict) else regime_data
    if not regime:
        es = macro_report.get("executive_summary", {})
        regime = es.get("regime", "")

    # CIO v19.0 R1: Compute forward-looking regime momentum overlay
    regime_momentum = compute_regime_momentum(
        macro_report, tech_report, fund_report, news_report,
    )
    logger.info("Regime momentum overlay: %s (regime=%s)", regime_momentum, regime)

    # CIO v23.3: Regime transition model from concordance history
    regime_transition = compute_regime_transition()
    logger.info("Regime transition: %s (spread=%.1f)",
                regime_transition.get("transition", "N/A"),
                regime_transition.get("spread", 0))

    # CIO v7.0 P3: Risk warning dilution detection.
    # When >40% of portfolio stocks trigger risk warnings, the warnings are
    # systemic (macro-driven) rather than stock-specific. Track this so the
    # report can surface it as a portfolio-level concern instead of per-stock noise.
    risk_warns_set = lookups["risk_warns"]
    warned_count = sum(1 for t in portfolio_signals if t in risk_warns_set)
    total_count = len(portfolio_signals)
    risk_diluted = total_count > 0 and (warned_count / total_count) > 0.40

    if risk_diluted:
        logger.info(
            "Risk warning dilution detected: %d/%d (%.0f%%) stocks warned — "
            "treat as systemic risk, not stock-specific",
            warned_count, total_count, warned_count / total_count * 100,
        )

    # CIO v20.0 D2: Extract Fear & Greed score for contrarian indicator
    # Census agent writes fear_greed.current (not sentiment.fg_broad)
    fg_score = (
        census_report.get("fear_greed", {}).get("current")
        or census_report.get("sentiment", {}).get("fg_broad")
        or census_report.get("fg_broad")
    )

    # Process portfolio stocks
    for ticker, sig_data in portfolio_signals.items():
        sector = resolve_sector(ticker, sector_map)
        sec_median = sector_medians.get(sector, universe_median)
        # CIO v11.0 L2: Extract previous signal for velocity computation
        _prev_sig, _prev_days = prev_signal_map.get(ticker, (None, None))
        entry = _synthesize_with_lookups(
            ticker, sig_data, lookups, fund_report, tech_report,
            sector, sec_median, census_ts_map, regime=regime,
            kill_thesis_triggered=triggered_kill_theses.get(ticker, False),
            previous_signal=_prev_sig,
            days_since_signal_change=_prev_days,
            regime_momentum=regime_momentum,
            fg_score=fg_score,
            news_report=news_report,
            is_opportunity=False,
        )
        entry["is_opportunity"] = False
        entry["kill_thesis_triggered"] = triggered_kill_theses.get(ticker, False)
        entry["risk_diluted"] = risk_diluted

        # CIO v20.0 D1: Track days_held for conviction decay
        if _prev_sig == entry["signal"]:
            entry["days_held"] = _prev_days if _prev_days else 0

        concordance.append(entry)

    # Process new opportunities (CIO C2 gate)
    # CIO v8.0 F8: Normalize signal format from opportunity scanner.
    # Scanner uses full words ("BUY"/"HOLD") vs portfolio's single chars ("B"/"H").
    _SIG_NORM = {"BUY": "B", "HOLD": "H", "SELL": "S", "INCONCLUSIVE": "I"}
    for ticker, sig_data in opportunity_signals.items():
        if ticker in portfolio_signals:
            continue  # Already in portfolio
        raw_sig = sig_data.get("signal", "H")
        sig_data["signal"] = _SIG_NORM.get(raw_sig, raw_sig)
        sector = resolve_sector(ticker, opportunity_sector_map)
        sec_median = sector_medians.get(sector, universe_median)

        # CIO Review F1: Inject opportunity score for dual-synthetic stocks
        # When both fundamental and technical agents don't cover a stock,
        # use the Opportunity Scanner's own score to differentiate
        has_fund = bool(fund_report.get("stocks", {}).get(ticker))
        has_tech = bool(tech_report.get("stocks", {}).get(ticker))
        opp_score = sig_data.get("opportunity_score", 0)

        entry = _synthesize_with_lookups(
            ticker, sig_data, lookups, fund_report, tech_report,
            sector, sec_median, census_ts_map, regime=regime,
            regime_momentum=regime_momentum,
            fg_score=fg_score,
            news_report=news_report,
            is_opportunity=True,
        )

        # F1: Boost conviction for dual-synthetic stocks using opp_score.
        # CIO v6.0 fix: Apply as a conviction DELTA rather than a base override,
        # because bonuses/penalties were already computed for the original base.
        # Overriding the base while keeping original adjustments creates an
        # inconsistency. Instead, compute the delta and add it to conviction.
        if not has_fund and not has_tech and opp_score > 0:
            opp_base = max(55, min(70, int(opp_score * 0.8)))
            if opp_base > entry["base"]:
                delta = opp_base - entry["base"]
                entry["conviction"] = min(100, entry["conviction"] + delta)
                entry["base"] = opp_base
                entry["opp_score_injected"] = True
        # Apply C2 validation gate
        macro_fit = _resolve_macro_fit(lookups["macro_impl"], ticker)
        cen_align = lookups["div_map"].get(ticker, ("NEUTRAL", 0))[0]
        entry = apply_opportunity_gate(
            entry, fund_report, tech_report, macro_fit,
            cen_align, portfolio_sectors, regime=regime,
        )
        concordance.append(entry)

    # CIO v20.0 D1: Conviction decay — signals lose conviction over time
    # Apply time-weighted decay for stocks held in same signal state >14 days
    for entry in concordance:
        days_held = entry.get("days_held", 0)
        signal = entry.get("signal", "H")
        if days_held > 14 and signal in ("B", "H"):
            decay_factor = max(0.85, 0.98 ** (days_held / 7))
            old_conviction = entry["conviction"]
            entry["conviction"] = int(entry["conviction"] * decay_factor)
            entry["conviction_decay_days"] = days_held
            entry["conviction_decay_factor"] = round(decay_factor, 3)
            # CIO v25.0: Track decay in waterfall
            wf = entry.get("conviction_waterfall", {})
            wf["conviction_decay"] = entry["conviction"] - old_conviction
            entry["conviction_waterfall"] = wf
            logger.debug(
                "%s: conviction decay applied (days=%d, factor=%.3f, %d→%d)",
                entry["ticker"], days_held, decay_factor, old_conviction, entry["conviction"]
            )

    # CIO v20.0 D6: Portfolio Sharpe impact — penalize redundant positions
    # Compute correlation penalties based on risk report clusters
    for entry in concordance:
        ticker = entry["ticker"]
        sharpe_penalty = compute_portfolio_sharpe_impact(
            ticker, portfolio_signals, risk_report
        )
        if sharpe_penalty < 0:
            entry["conviction"] += sharpe_penalty  # sharpe_penalty is negative
            entry["portfolio_redundancy_penalty"] = abs(sharpe_penalty)
            # CIO v25.0: Track redundancy in waterfall
            wf = entry.get("conviction_waterfall", {})
            wf["portfolio_redundancy"] = sharpe_penalty
            entry["conviction_waterfall"] = wf
            # Update position_size_pct with actual correlated count
            correlations = risk_report.get("correlation_clusters", [])
            correlated_count = 0
            for cluster in correlations:
                if ticker in cluster.get("tickers", []):
                    correlated_count = len(cluster.get("tickers", [])) - 1
                    break
            entry["position_size_pct"] = compute_position_size(
                entry["conviction"],
                entry.get("beta", 1.0),
                correlated_count,
                entry.get("buy_pct", 0),
            )

    # CIO v7.0 P4 + v11.0 L1/L5/L7: Sector concentration penalty.
    # When 3+ stocks share the same sector in the concordance, apply a small
    # conviction penalty (-2 per stock beyond 2) to discourage over-concentration
    # at the scoring level. Only penalise BUY/ADD actions — we don't want to
    # artificially reduce conviction on HOLD/TRIM/SELL (those are risk decisions).
    #
    # CIO v11.0 L5: Reduced cap from 10 to 6 for existing holdings. Concentration
    # risk is a portfolio-level concern better handled by conviction_sizer (which
    # has cluster sizing and sector constraints). A 10-point penalty was pushing
    # 11 BUY-signal stocks to the 45-floor, destroying ranking signal.
    #
    # CIO v11.0 L7: Conviction floors are re-applied AFTER sector penalty to
    # preserve the design principle that BUY-signal quality floors are inviolable.
    # CIO v14.0 V6: Count sectors by action (BUY/ADD only) for concentration
    # penalty, not total population. Technology may have 10 stocks in the portfolio
    # but only 3 with BUY/ADD action — the other 7 are HOLD and don't represent
    # concentration risk in the ACTION space. Using total count inflated the penalty
    # and suppressed stocks that weren't actually adding concentration.
    sector_counts: Dict[str, int] = {}
    for entry in concordance:
        if entry.get("action") in ("BUY", "ADD") or entry.get("signal") == "B":
            sec = entry.get("sector", "Other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    for entry in concordance:
        if entry.get("action") not in ("BUY", "ADD"):
            continue
        sec = entry.get("sector", "Other")
        count = sector_counts.get(sec, 1)
        if count > 2:
            raw_penalty = (count - 2) * 2  # -2 per extra stock beyond 2
            if entry.get("is_opportunity"):
                penalty = min(raw_penalty, 15)  # cap at 15 for new entries
            else:
                penalty = min(raw_penalty, 6)  # v11.0 L5: cap at 6 for existing
            # CIO v15.0 W2: The sector concentration penalty must not breach the
            # conviction that was already floor-protected by apply_conviction_floors().
            # The pre-penalty conviction IS the minimum the quality system guarantees.
            # Sector concentration is a PORTFOLIO-LEVEL concern that should influence
            # position sizing (via conviction_sizer cluster sizing), not destroy
            # per-stock conviction integrity. Cap the effective penalty so conviction
            # stays at or above the stock's pre-penalty floor.
            pre_penalty_conv = entry["conviction"]
            floor = 45 if entry.get("signal") == "B" else 30
            # For BUY signals with strong fundamentals, preserve more of the conviction
            if entry.get("signal") == "B" and entry.get("buy_pct", 0) >= 80:
                floor = max(floor, 52)  # High-consensus BUY floor during concentration
            entry["conviction"] = max(floor, pre_penalty_conv - penalty)
            entry["sector_concentration_penalty"] = penalty
            # CIO v25.0: Track sector concentration in waterfall
            wf = entry.get("conviction_waterfall", {})
            actual_delta = entry["conviction"] - pre_penalty_conv
            if actual_delta != 0:
                wf["sector_concentration"] = actual_delta
            entry["conviction_waterfall"] = wf

    # CIO v11.0 L7: Re-apply conviction floors after sector concentration penalty.
    # The quality BUY floor (55 for strong BUY combos) should be immune to sector
    # penalty — the quant system verified BUY criteria regardless of concentration.
    for entry in concordance:
        if entry.get("sector_concentration_penalty", 0) > 0:
            entry["conviction"] = apply_conviction_floors(
                entry["conviction"],
                entry.get("signal", "H"),
                entry.get("excess_exret", 0),
                entry.get("pef", 0) if "pef" in entry else 0,
                entry.get("pet", 0) if "pet" in entry else 0,
                # Recount bull agents — we don't store bull_count but can
                # approximate from stored fields
                sum(1 for x in [
                    entry.get("fund_score", 0) >= 70,
                    entry.get("tech_signal") in ("ENTER_NOW",) or entry.get("tech_momentum", 0) > 20,
                    entry.get("macro_fit") == "FAVORABLE",
                    entry.get("census") == "ALIGNED",
                    "POSITIVE" in (entry.get("news_impact") or ""),
                    not entry.get("risk_warning", False),
                ] if x),
                entry.get("fund_score", 50),
                entry.get("buy_pct", 0),
            )

    # CIO v11.0 L1: Re-evaluate actions for ALL stocks after sector concentration
    # penalty. Previously only opportunities were re-evaluated, leaving existing
    # holdings with action=ADD but conviction < 55 (23% of portfolio affected).
    for entry in concordance:
        if entry.get("sector_concentration_penalty", 0) > 0:
            if entry.get("is_opportunity"):
                # Opportunities use BUY/HOLD (not ADD/TRIM) — re-apply gate logic
                entry["action"] = "BUY" if entry["conviction"] >= 55 else "HOLD"
            else:
                entry["action"] = determine_action(
                    entry["conviction"], entry.get("signal", "H"),
                    entry.get("tech_signal", "HOLD"),
                    entry.get("risk_warning", False),
                )

    # Break conviction ties with composite quality score (CIO v5.2, v6.0 normalized)
    # This prevents 11 stocks at identical conviction=65.
    # CIO v6.0: Normalize each component to 0-100 before weighting so that no
    # single factor dominates due to scale differences.
    for entry in concordance:
        # Excess EXRET: clip to [-30, 30], normalize to 0-100
        raw_exret = max(-30, min(30, entry.get("excess_exret", 0)))
        norm_exret = (raw_exret + 30) / 60 * 100
        # Fund score: already 0-100
        norm_fund = min(100, max(0, entry.get("fund_score", 50)))
        # Beta risk: lower beta = better. Clip to [0.3, 3.0], invert and normalize
        raw_beta = max(0.3, min(3.0, entry.get("beta", 1.0)))
        norm_beta = (3.0 - raw_beta) / 2.7 * 100
        # Bull pct: already 0-100
        norm_bull = min(100, max(0, entry.get("bull_pct", 50)))
        tiebreak = (
            norm_exret * 0.4
            + norm_fund * 0.3
            + norm_beta * 0.1
            + norm_bull * 0.2
        )
        entry["tiebreak"] = round(tiebreak, 2)

        # CIO Legacy C2: Capital efficiency score for within-group ranking.
        # Answers "if I have $10K to deploy, which stock in this action group
        # should get the marginal dollar?" by combining expected risk-adjusted
        # return with conviction confidence.
        beta_val = max(entry.get("beta", 1.0), 0.3)
        exret_val = entry.get("exret", 0)
        conviction_val = entry.get("conviction", 50)
        risk_adj_return = exret_val / beta_val
        confidence = conviction_val / 100.0
        entry["capital_efficiency"] = round(risk_adj_return * confidence, 2)

    # CIO v23.4: Tax-loss harvesting (Q4 only: Oct-Dec)
    from datetime import datetime as _dt_cls
    current_month = _dt_cls.now().month
    if current_month >= 10:  # Q4
        # Identify correlation clusters for substitute suggestions
        corr_clusters = risk_report.get("correlation_clusters", [])
        cluster_map: Dict[str, list] = {}
        for cl in corr_clusters:
            tickers = cl.get("tickers", cl.get("stocks", []))
            for t in tickers:
                cluster_map[t] = [x for x in tickers if x != t]

        for entry in concordance:
            if not entry.get("is_opportunity") and entry.get("action") in ("HOLD", "TRIM", "SELL"):
                price = entry.get("price", 0)
                exret = entry.get("exret", 0)
                # Negative EXRET as proxy for unrealized loss
                if exret < -10:
                    entry["tax_harvest_candidate"] = True
                    substitutes = cluster_map.get(entry["ticker"], [])
                    if substitutes:
                        entry["tax_substitute"] = substitutes[0]
                    else:
                        entry["tax_substitute"] = None

    # Sort by action priority, then conviction desc, then tiebreak desc
    concordance.sort(
        key=lambda x: (ACTION_ORDER.get(x["action"], 9), -x["conviction"], -x["tiebreak"])
    )

    return concordance


def build_agent_memory(
    prev_concordance: List[Dict[str, Any]],
    current_signals: Dict[str, Dict],
) -> Dict[str, str]:
    """
    Build per-agent feedback strings from previous concordance + current prices.

    CIO v6.0 R1: Each agent sees how its own previous assessments performed,
    creating a per-agent, per-stock feedback loop for calibration over time.

    Args:
        prev_concordance: List of concordance entries from the last committee run.
        current_signals: Dict of ticker -> signal data (must include 'price').

    Returns:
        Dict mapping agent name to a formatted feedback string.
    """
    if not prev_concordance:
        return {}

    # Track per-agent assessments vs outcomes
    agent_records: Dict[str, List[str]] = {
        "fundamental": [],
        "technical": [],
        "macro": [],
        "census": [],
        "news": [],
        "opportunity": [],
        "risk": [],
    }

    for entry in prev_concordance:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue

        prev_conv = entry.get("conviction", 50)
        prev_action = entry.get("action", "HOLD")
        prev_price = entry.get("price", 0)
        fund_score = entry.get("fund_score", 0)
        tech_signal = entry.get("tech_signal", "")
        macro_fit = entry.get("macro_fit", "")
        census_align = entry.get("census_alignment", "")
        news_impact = entry.get("news_impact", "")
        risk_warning = entry.get("risk_warning", False)

        # Get current price for return calculation
        curr_data = current_signals.get(ticker, {})
        curr_price = curr_data.get("price", 0)
        if not prev_price or not curr_price or prev_price <= 0:
            continue

        ret_pct = round((curr_price - prev_price) / prev_price * 100, 1)
        direction = "up" if ret_pct > 0 else "down" if ret_pct < 0 else "flat"

        # Fundamental feedback
        if fund_score:
            accuracy = ""
            if fund_score >= 70 and ret_pct < -5:
                accuracy = "TOO OPTIMISTIC"
            elif fund_score < 45 and ret_pct > 5:
                accuracy = "TOO PESSIMISTIC"
            elif (fund_score >= 70 and ret_pct > 0) or (fund_score < 45 and ret_pct < 0):
                accuracy = "CORRECT"
            else:
                accuracy = "NEUTRAL"
            agent_records["fundamental"].append(
                f"- {ticker}: score={fund_score:.0f}. Since then: {direction} {abs(ret_pct):.1f}%. [{accuracy}]"
            )

        # Technical feedback
        if tech_signal:
            accuracy = ""
            if tech_signal == "ENTER_NOW" and ret_pct < -5:
                accuracy = "WRONG"
            elif tech_signal in ("AVOID", "EXIT_SOON") and ret_pct > 5:
                accuracy = "WRONG"
            elif tech_signal == "ENTER_NOW" and ret_pct > 0:
                accuracy = "CORRECT"
            elif tech_signal in ("AVOID", "EXIT_SOON") and ret_pct < 0:
                accuracy = "CORRECT"
            else:
                accuracy = "NEUTRAL"
            agent_records["technical"].append(
                f"- {ticker}: signal={tech_signal}. Since then: {direction} {abs(ret_pct):.1f}%. [{accuracy}]"
            )

        # Macro feedback
        if macro_fit and macro_fit != "NEUTRAL":
            accuracy = ""
            if macro_fit == "FAVORABLE" and ret_pct < -5:
                accuracy = "WRONG"
            elif macro_fit == "UNFAVORABLE" and ret_pct > 5:
                accuracy = "WRONG"
            elif macro_fit == "FAVORABLE" and ret_pct > 0:
                accuracy = "CORRECT"
            elif macro_fit == "UNFAVORABLE" and ret_pct < 0:
                accuracy = "CORRECT"
            else:
                accuracy = "NEUTRAL"
            agent_records["macro"].append(
                f"- {ticker}: macro_fit={macro_fit}. Since then: {direction} {abs(ret_pct):.1f}%. [{accuracy}]"
            )

        # Census feedback
        if census_align and census_align != "NEUTRAL":
            agent_records["census"].append(
                f"- {ticker}: alignment={census_align}. Since then: {direction} {abs(ret_pct):.1f}%."
            )

        # News feedback
        if news_impact and news_impact not in ("NEUTRAL", "MIXED"):
            agent_records["news"].append(
                f"- {ticker}: impact={news_impact}. Since then: {direction} {abs(ret_pct):.1f}%."
            )

        # Risk feedback
        if risk_warning:
            accuracy = "VALIDATED" if ret_pct < 0 else "FALSE ALARM"
            agent_records["risk"].append(
                f"- {ticker}: WARNING issued. Since then: {direction} {abs(ret_pct):.1f}%. [{accuracy}]"
            )

        # Opportunity feedback (for BUY recommendations on new stocks)
        if prev_action == "BUY":
            accuracy = "GOOD PICK" if ret_pct > 0 else "POOR PICK"
            agent_records["opportunity"].append(
                f"- {ticker}: recommended BUY (conv={prev_conv}). Since then: {direction} {abs(ret_pct):.1f}%. [{accuracy}]"
            )

    # Format output strings, limit to top 10 most informative entries per agent
    result: Dict[str, str] = {}
    for agent, records in agent_records.items():
        if not records:
            continue
        # Sort by absolute accuracy signal — put WRONG/TOO OPTIMISTIC first
        priority_order = {"WRONG": 0, "TOO OPTIMISTIC": 1, "TOO PESSIMISTIC": 2,
                          "FALSE ALARM": 3, "POOR PICK": 4, "CORRECT": 5,
                          "VALIDATED": 6, "GOOD PICK": 7, "NEUTRAL": 8}
        records.sort(key=lambda r: priority_order.get(
            r.split("[")[-1].rstrip("]") if "[" in r else "NEUTRAL", 8
        ))
        result[agent] = "\n".join(records[:10])

    return result


def compute_changes(
    current: List[Dict[str, Any]],
    previous,
) -> List[Dict[str, Any]]:
    """
    Compare current and previous concordance to detect changes.

    Previous can be either:
    - A list of dicts (each with 'ticker' key)
    - A dict of ticker -> data (from concordance.json 'stocks' key)
    """
    if isinstance(previous, dict):
        # Handle concordance.json wrappers: {"stocks": {...}} or {"concordance": [...]}
        if "stocks" in previous:
            previous = previous["stocks"]
        elif "concordance" in previous:
            previous = previous["concordance"]
        if isinstance(previous, dict):
            prev_map = {k: dict(v, ticker=k) for k, v in previous.items()}
        else:
            prev_map = {c["ticker"]: c for c in previous}
    else:
        prev_map = {c["ticker"]: c for c in previous}
    changes = []

    for c in current:
        prev = prev_map.get(c["ticker"])
        if prev:
            if prev.get("action") != c["action"] or abs(prev.get("conviction", 0) - c["conviction"]) > 5:
                changes.append({
                    "ticker": c["ticker"],
                    "prev_action": prev.get("action"),
                    "prev_conviction": prev.get("conviction"),
                    "curr_action": c["action"],
                    "curr_conviction": c["conviction"],
                    "delta": c["conviction"] - prev.get("conviction", 0),
                    "type": "UPGRADE" if c["conviction"] > prev.get("conviction", 0) else "DOWNGRADE",
                })
        else:
            changes.append({
                "ticker": c["ticker"],
                "prev_action": None,
                "prev_conviction": None,
                "curr_action": c["action"],
                "curr_conviction": c["conviction"],
                "delta": 0,
                "type": "NEW",
            })

    return changes


def generate_synthesis_output(
    concordance: List[Dict[str, Any]],
    macro_report: Dict,
    census_report: Dict,
    news_report: Dict,
    risk_report: Dict,
    changes: List[Dict[str, Any]],
    sector_gaps: List[Dict[str, Any]],
    census_ts_map: Optional[Dict[str, str]] = None,
    opportunity_report: Optional[Dict] = None,
    performance_data: Optional[Dict] = None,
    tech_report: Optional[Dict] = None,
    fund_report: Optional[Dict] = None,
    regime_transition: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate complete synthesis JSON output from concordance and agent reports.

    This function produces the flat-key structure that the HTML generator
    expects, so the HTML can be generated deterministically from synthesis.json
    without needing to re-read individual agent reports.
    """
    from collections import Counter

    # CIO v26.1: Normalize report structures (idempotent, safe to re-apply)
    _normalize_breaking_news(news_report)
    _normalize_portfolio_news(news_report)
    _normalize_economic_events(news_report)
    _normalize_sector_rankings(macro_report)
    _normalize_census_divergences(census_report)
    _normalize_census_sentiment(census_report)
    if tech_report:
        _normalize_tech_stocks(tech_report)
    if fund_report:
        _normalize_fund_stocks(fund_report)

    action_dist = Counter(e["action"] for e in concordance)

    # Extract risk metrics — try multiple field name conventions
    pr = risk_report.get("portfolio_risk", {})
    var_95 = pr.get("var_95_annual") or pr.get("var_95_daily") or pr.get("var_95") or 0
    max_dd = pr.get("max_drawdown_1y") or pr.get("max_drawdown") or 0
    port_beta = pr.get("portfolio_beta_vs_spy") or pr.get("portfolio_beta") or 0
    risk_score = pr.get("risk_score") or risk_report.get("executive_summary", {}).get("overall_risk_rating_score", 50) or 50

    # Macro — try nested paths for agent report compatibility.
    # Macro agent writes: macro_indicators.us_10y_yield, regime.classification
    indicators = macro_report.get("macro_indicators") or macro_report.get("indicators", {})
    es = macro_report.get("executive_summary", {})
    regime_data = macro_report.get("regime", {})
    regime = (
        (regime_data.get("classification") if isinstance(regime_data, dict) else regime_data)
        or (es.get("regime") if isinstance(es, dict) else "")
        or "CAUTIOUS"
    )
    macro_score = macro_report.get("macro_score") or (es.get("macro_score") if isinstance(es, dict) else 0) or 0
    rotation = (
        macro_report.get("sector_rotation_view", {}).get("rationale", "").split(".")[0]
        if isinstance(macro_report.get("sector_rotation_view"), dict)
        else macro_report.get("sector_rotation_phase")
        or macro_report.get("rotation_phase")
        or "Unknown"
    )

    # Census — normalise agent output to flat keys for HTML generator.
    # Census agent writes: fear_greed.current, cash_trends.mean_cash_pct
    # Older formats may use: sentiment.top100.fear_greed_index, etc.
    sentiment = {}
    # Primary path: census agent top-level keys
    fg_data = census_report.get("fear_greed", {})
    cash_data = census_report.get("cash_trends", {})
    fg_val = fg_data.get("current") or fg_data.get("value")
    cash_val = cash_data.get("mean_cash_pct") or cash_data.get("avg_cash_pct")

    # Fallback: nested sentiment structure (older report formats)
    sentiment_raw = census_report.get("sentiment", {})
    t100 = sentiment_raw.get("top100", {})
    b1500 = sentiment_raw.get("broad_1500", sentiment_raw.get("broad", {}))

    sentiment["fg_top100"] = (
        fg_val
        or t100.get("fear_greed_index")
        or sentiment_raw.get("fg_top100")
        or 0
    )
    sentiment["fg_broad"] = (
        fg_val  # Census agent reports one F&G value, use for both
        or b1500.get("fear_greed_index")
        or sentiment_raw.get("fg_broad")
        or 0
    )
    sentiment["cash_top100"] = (
        cash_val
        or t100.get("avg_cash_pct")
        or sentiment_raw.get("cash_top100")
        or 0
    )
    sentiment["cash_broad"] = (
        cash_val
        or b1500.get("avg_cash_pct")
        or sentiment_raw.get("cash_broad")
        or 0
    )
    sentiment["trend_summary"] = (
        cash_data.get("assessment")
        or fg_data.get("contrarian_signal")
        or sentiment_raw.get("trend_summary")
        or ""
    )
    # Preserve extra keys from any format
    for k, v in sentiment_raw.items():
        if k not in ("top100", "broad_1500", "broad") and k not in sentiment:
            sentiment[k] = v

    # CIO v19.0: Compute regime momentum for synthesis output
    regime_momentum = compute_regime_momentum(
        macro_report,
        tech_report or {},
        fund_report or {},
        news_report,
    )

    # CIO v23.3: Build watchlist from gated opportunities
    watchlist = [
        {"ticker": e["ticker"], "conviction": e["conviction"],
         "watch_trigger": e.get("watch_trigger", ""),
         "sector": e.get("sector", "")}
        for e in concordance
        if e.get("is_opportunity") and e.get("action") == "HOLD" and e.get("watch_trigger")
    ]

    output = {
        "version": "v26.1",
        "concordance": concordance,
        "action_distribution": dict(action_dist),
        "changes": changes,
        "sector_gaps": sector_gaps,
        # Flat keys for HTML generator compatibility
        "regime": regime,
        "regime_momentum": regime_momentum,
        "macro_score": macro_score,
        "rotation_phase": rotation,
        "risk_score": risk_score,
        "var_95": round(var_95 * 100, 2) if abs(var_95) < 1 else round(var_95, 2),
        "max_drawdown": round(max_dd * 100, 1) if abs(max_dd) < 1 else round(max_dd, 1),
        "portfolio_beta": round(port_beta, 2),
        "portfolio_risk": pr,
        # Macro
        "indicators": indicators,
        "sector_rankings": macro_report.get("sector_rankings", {}),
        "macro_risks": macro_report.get("key_risks", []),
        # Census
        "fg_top100": sentiment.get("fg_top100", 0),
        "fg_broad": sentiment.get("fg_broad", 0),
        "census_sentiment": sentiment,
        "top_holdings_top100": (
            census_report.get("portfolio_pi_overlap", {}).get("most_popular_holdings")
            or census_report.get("top_holdings_top100", [])
        ),
        "missing_popular": (
            census_report.get("missing_popular", {}).get("stocks_not_in_portfolio_but_popular")
            or census_report.get("portfolio_alignment", {}).get("missing_popular", [])
        ),
        "census_time_series": census_ts_map or {},
        # News
        "breaking_news": news_report.get("breaking_news", []),
        "earnings_calendar": news_report.get("earnings_calendar", {}),
        "economic_events": news_report.get("economic_events", []),
        # Risk
        # CIO v7.0 P3: Risk warning dilution flag
        "risk_diluted": any(e.get("risk_diluted", False) for e in concordance),
        "correlation_clusters": risk_report.get("correlation_clusters", []),
        "concentration": risk_report.get("concentration", {}),
        "stress_scenarios": risk_report.get("stress_scenarios", {}),
        "position_limits": risk_report.get("position_limits", {}),
        "hard_constraints": risk_report.get("hard_constraints", []),
        # Opportunities
        "top_opportunities": (opportunity_report or {}).get("top_opportunities", []),
        # CIO v17.0: Performance feedback loop data
        "performance": performance_data or {},
        # Flat keys for HTML generator compatibility
        "scorecard": (performance_data or {}).get("scorecard", {}),
        "calibration_report": (performance_data or {}).get("calibration", {}),
        # CIO v23.3: Signal quality & timing
        "regime_transition": regime_transition or {},
        "watchlist": watchlist,
    }

    return output


def enrich_with_position_sizes(
    concordance: List[Dict[str, Any]],
    regime: str = "normal",
    portfolio_value: float = 450000.0,
    base_position_size: float = 2500.0,
) -> List[Dict[str, Any]]:
    """
    CIO v13.0 S2: Enrich BUY/ADD concordance entries with suggested position sizes.

    Uses the conviction_sizer module to compute dollar amounts based on
    conviction scores and regime. This transforms the committee output from
    "what to do" to "what to do and how much."

    Only enriches BUY and ADD actions. HOLD/TRIM/SELL don't need new sizing.
    """
    try:
        from trade_modules.conviction_sizer import calculate_conviction_size
    except ImportError:
        logger.warning("conviction_sizer not available — skipping position sizing")
        return concordance

    # Map regime names from committee format to sizer format
    regime_map = {
        "RISK_OFF": "high",
        "CAUTIOUS": "elevated",
        "NEUTRAL": "normal",
        "RISK_ON": "low",
        "": "normal",
    }
    sizer_regime = regime_map.get(regime, "normal")

    # Tier multipliers (from config.yaml defaults)
    tier_multipliers = {
        "MEGA": 5.0, "LARGE": 4.0, "MID": 3.0, "SMALL": 2.0, "MICRO": 1.0,
    }

    for entry in concordance:
        if entry.get("action") not in ("BUY", "ADD"):
            continue

        conviction = entry.get("conviction", 50)
        # Infer tier from market cap if available, default MID
        tier = "MID"
        cap = entry.get("market_cap", "")
        if isinstance(cap, str):
            for t in tier_multipliers:
                if t.lower() in cap.lower():
                    tier = t
                    break

        tier_mult = tier_multipliers.get(tier, 3.0)

        result = calculate_conviction_size(
            base_position_size=base_position_size,
            tier_multiplier=tier_mult,
            conviction_score=conviction,
            regime=sizer_regime,
            portfolio_value=portfolio_value,
            max_position_pct=5.0,
            tier=tier,
        )

        entry["suggested_size_usd"] = round(result.get("position_size", 0), 0)
        entry["size_pct"] = round(
            result.get("position_size", 0) / portfolio_value * 100, 2
        ) if portfolio_value > 0 else 0

    return concordance


def save_concordance(
    concordance: List[Dict[str, Any]],
    output_path: str,
    date_str: Optional[str] = None,
) -> None:
    """
    Save concordance with date wrapper for signal velocity support.

    CIO v13.0 F2: Previously, concordance was saved as a bare list,
    causing the velocity computation to never find a date and returning
    NO_HISTORY for all stocks. Now saves as {"date": "...", "concordance": [...]}.
    """
    import json
    from datetime import datetime

    dated_concordance = {
        "date": date_str or datetime.now().strftime("%Y-%m-%d"),
        "concordance": concordance,
    }
    with open(output_path, "w") as f:
        json.dump(dated_concordance, f, indent=2)

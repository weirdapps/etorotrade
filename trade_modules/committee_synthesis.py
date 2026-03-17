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
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    tech_weight = 1.0 * tech_disc
    if tech_signal == "ENTER_NOW":
        bull += tech_weight
        directional_weight += tech_weight
    elif tech_signal in ("AVOID", "EXIT_SOON"):
        bear += tech_weight
        directional_weight += tech_weight
    elif tech_signal == "WAIT_FOR_PULLBACK":
        bull += tech_weight * 0.6
        bear += tech_weight * 0.4
        directional_weight += tech_weight  # Leaning directional
    elif tech_momentum > 20:
        bull += tech_weight * 0.7
        bear += tech_weight * 0.3
        directional_weight += tech_weight
    elif tech_momentum < -20:
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
    if census_alignment == "ALIGNED":
        bull += 0.85
        directional_weight += 0.85
    elif census_alignment == "DIVERGENT":
        bull += 0.35
        bear += 0.50
        directional_weight += 0.85
    elif census_alignment == "CENSUS_DIV":
        bull += 0.6
        bear += 0.25
        directional_weight += 0.85
    else:
        bull += 0.425
        bear += 0.425
        neutral_weight += 0.85

    # News (freshness 1.0x)
    if "HIGH_POSITIVE" in news_impact:
        bull += 1.0
        directional_weight += 1.0
    elif "LOW_POSITIVE" in news_impact:
        bull += 0.7
        bear += 0.3
        directional_weight += 1.0
    elif "HIGH_NEGATIVE" in news_impact:
        bear += 1.0
        directional_weight += 1.0
    elif "LOW_NEGATIVE" in news_impact:
        bull += 0.3
        bear += 0.7
        directional_weight += 1.0
    else:
        bull += 0.5
        bear += 0.5
        neutral_weight += 1.0

    # Risk Manager (CIO Legacy B3: 1.2x for BUY assessment, 2.0x for SELL)
    # CIO v6.0 F5: Regime-sensitive neutral vote when no warning found
    risk_mult = 2.0 if signal == "S" else 1.2
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

    # CIO Legacy A5: Directional confidence — ratio of directional
    # (non-neutral) weight to total weight. When 4 of 7 agents are neutral,
    # this will be low (~0.43), indicating the signal is diluted.
    # CIO v7.0 P2: Returned as third tuple element instead of function attribute.
    total_all = directional_weight + neutral_weight
    dir_confidence = directional_weight / total_all if total_all > 0 else 0.5

    return bull, bear, dir_confidence


def determine_base_conviction(
    bull_pct: float,
    signal: str,
    fund_score: float,
    excess_exret: float,
    bear_ratio: float,
    regime: str = "",
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
    if regime == "RISK_OFF":
        agent_base = int(agent_base * 0.85)
    elif regime == "CAUTIOUS":
        agent_base = int(agent_base * 0.92)

    if signal == "B":
        base = max(agent_base, 55)
        # BUY-side quality bonus (CIO C1) — ONLY for BUY signals
        if fund_score >= 80:
            base = max(base + 10, 60)
        elif fund_score >= 65:
            base = max(base + 5, 55)
        # Proportional excess EXRET bonus (CIO v5.2)
        # Instead of binary >=12 → +5, scale: 5→+1, 10→+3, 15→+4, 20+→+5
        # CIO Legacy B2: Extreme EXRET (>40) triggers a staleness penalty.
        # EXRET > 40% almost always indicates stale analyst targets (not updated
        # after a large drop), distressed/turnaround binary outcomes, or data
        # errors. E.g., MSTR at 171% EXRET is not a genuine expected return.
        if excess_exret > 40:
            base -= 3  # Staleness penalty for extreme EXRET
        elif excess_exret >= 5:
            exret_bonus = min(5, int(excess_exret / 4))
            base += exret_bonus
    elif signal == "S":
        if bull_pct > 55:
            base = 50  # Agents mostly disagree with SELL
        elif bull_pct > 40:
            base = 60
        else:
            base = 70
        if bear_ratio >= 0.80:
            base = 85
        elif bear_ratio >= 0.65:
            base = max(base, 70)
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
    if fund_score >= 80 and risk_warning:
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
) -> Tuple[int, str]:
    """
    Conviction adjustment based on earnings surprise history (CIO Legacy B5).

    Post-Earnings Announcement Drift (PEAD) is one of the most robust
    anomalies in finance — stocks that beat estimates tend to continue
    outperforming for 60-90 days. Serial beaters (2+ consecutive) are
    even more predictive.

    Args:
        recent_surprise_pct: Percentage beat/miss of most recent earnings.
            Positive = beat, negative = miss. None = no data.
        consecutive_beats: Number of consecutive earnings beats (0 if none).

    Returns:
        (conviction_adjustment, label)
    """
    if recent_surprise_pct is None:
        return (0, "NO_DATA")

    if recent_surprise_pct > 10 and consecutive_beats >= 2:
        return (+5, "SERIAL_BEATER")
    elif recent_surprise_pct > 5:
        return (+3, "BEAT")
    elif recent_surprise_pct < -10:
        return (-5, "BIG_MISS")
    elif recent_surprise_pct < -5:
        return (-3, "MISS")
    return (0, "IN_LINE")


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
) -> Tuple[int, int]:
    """
    Compute bonus and penalty adjustments.

    Returns (bonuses, penalties) both capped per skill spec.
    """
    bonuses = 0
    penalties = 0

    # Agent agreement bonus
    if bull_count >= 6:
        bonuses += 15
    elif bull_count >= 5:
        bonuses += 10

    # Consensus warning — tiered by excess EXRET
    if buy_pct > 90:
        if excess_exret >= 12:
            penalties += 5
        elif excess_exret >= 0:
            penalties += 8
        else:
            penalties += 15

    # Census alignment
    if -20 <= div_score <= 20:
        bonuses += 5
    elif div_score < -20:
        bonuses += 8

    # Census time-series
    if census_ts == "strong_accumulation":
        bonuses += 3
    elif census_ts in ("strong_distribution", "distribution"):
        penalties += 5

    # News catalyst
    if "HIGH_POSITIVE" in news_impact:
        bonuses += 5
    if "NEGATIVE" in news_impact:
        penalties += 5

    # Technical overbought
    if rsi > 70:
        penalties += 5

    # Tech disagreement penalty (CIO v5.2): when BUY signal but tech says AVOID/EXIT
    if signal == "B" and tech_signal in ("AVOID", "EXIT_SOON"):
        penalties += 8
    elif signal == "B" and tech_momentum < -30:
        penalties += 5

    # Macro sector-specific
    if macro_fit == "UNFAVORABLE":
        if sector in ("Financials", "Consumer Discretionary"):
            penalties += 10
        else:
            penalties += 5
    elif macro_fit == "FAVORABLE":
        bonuses += 5

    # High beta
    if beta > 2.0:
        penalties += 5

    # Quality trap
    if quality_trap:
        penalties += 5

    # Sector rotation
    etf = SECTOR_ETF_MAP.get(sector, "")
    if etf and etf in sector_rankings:
        rank = sector_rankings[etf].get("rank", 6)
        ret_1m = sector_rankings[etf].get("return_1m", 0)
        if rank <= 3 and ret_1m > 0:
            bonuses += 5
        elif rank >= 9 and ret_1m < -3:
            penalties += 5

    # SELL-specific: tech/macro disagreement reduces sell conviction
    if signal == "S":
        if tech_signal in ("ENTER_NOW", "WAIT_FOR_PULLBACK"):
            penalties += 5
        if macro_fit == "FAVORABLE":
            penalties += 3
        if census_alignment == "CENSUS_DIV" and div_score < -30:
            penalties += 5

    # Cap adjustments
    bonuses = min(bonuses, 20)
    penalties = min(penalties, 25)

    return bonuses, penalties


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
    """Apply conviction floors to prevent unreasonable suppression."""
    # BUY signal unconditional floor
    if signal == "B":
        conviction = max(conviction, 40)

    # CIO v8.0 F9: Stronger floors for quality BUY names.
    # Penalty stacking (regime + consensus + macro + risk) can push MSFT,
    # BAC, SCHW from base=65 to conviction=50, producing HOLD when the
    # quant system verified all BUY criteria. Raise floors to 55 for
    # quality combinations so they remain actionable.
    if signal == "B" and excess_exret > 20 and pef > 0 and pet > 0 and pef < pet:
        conviction = max(conviction, 55)
    if signal == "B" and bull_count >= 4:
        conviction = max(conviction, 55)
    if signal == "B" and fund_score >= 70 and buy_pct >= 70:
        conviction = max(conviction, 55)

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
    """Recalculate conviction for TRIM actions to reflect trim confidence."""
    trim_conv = 50
    if tech_signal in ("EXIT_SOON", "AVOID"):
        trim_conv += 15
    if macro_fit == "UNFAVORABLE":
        trim_conv += 10
    if risk_warning:
        trim_conv += 10
    if beta > 1.5:
        trim_conv += 5
    if rsi > 70:
        trim_conv += 5
    if fund_score >= 80:
        trim_conv -= 10
    if census_alignment == "ALIGNED":
        trim_conv -= 5
    return min(trim_conv, 85)


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
) -> Dict[str, Any]:
    """
    Synthesize a single stock through the full conviction scoring pipeline.

    This is the core function that replaces the ad-hoc /tmp scripts.
    """
    signal = sig_data["signal"]
    exret = sig_data.get("exret", 0)
    buy_pct = sig_data.get("buy_pct", 0)
    beta = sig_data.get("beta", 0)
    pet = sig_data.get("pet", 0)
    pef = sig_data.get("pef", 0)

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
    )

    total_weight = bull_weight + bear_weight
    bull_pct = bull_weight / total_weight * 100 if total_weight > 0 else 50
    bear_ratio = bear_weight / total_weight if total_weight > 0 else 0

    # Step 2: Determine base conviction
    base = determine_base_conviction(
        bull_pct, signal, fund_score, excess_exret, bear_ratio,
        regime=regime,
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
    bonuses, penalties = compute_adjustments(
        signal, fund_score, tech_signal, tech_mom, rsi,
        macro_fit, census_alignment, div_score, census_ts_trend,
        news_impact, risk_warning, buy_pct, excess_exret,
        beta, quality_trap, sector, sector_rankings, bull_count,
    )

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
    )
    signal_quality_penalty += contradiction_penalty

    # Step 4c: Signal velocity (CIO Legacy B4)
    velocity_adj, velocity_label = compute_signal_velocity(
        signal, previous_signal, days_since_signal_change,
    )
    if velocity_adj > 0:
        bonuses = min(bonuses + velocity_adj, 20)
    elif velocity_adj < 0:
        signal_quality_penalty += abs(velocity_adj)

    # Step 4d: Earnings surprise (CIO Legacy B5)
    earnings_adj, earnings_label = get_earnings_surprise_adjustment(
        earnings_surprise_pct, consecutive_earnings_beats,
    )
    if earnings_adj > 0:
        bonuses = min(bonuses + earnings_adj, 20)
    elif earnings_adj < 0:
        signal_quality_penalty += abs(earnings_adj)

    # CIO Legacy A5: Directional confidence penalty
    # CIO v7.0 P2: dir_confidence is now from the return value (see Step 1)
    if dir_confidence < 0.4:
        signal_quality_penalty += 3

    # Cap signal quality penalties separately from base penalties
    signal_quality_penalty = min(signal_quality_penalty, 10)
    penalties = penalties + signal_quality_penalty

    conviction = base + bonuses - penalties

    # Step 4e: Kill thesis penalty (CIO v6.0 E1)
    # When a previously logged kill thesis has triggered, apply -15 penalty
    # that BYPASSES the normal penalty cap. A triggered kill thesis represents
    # a specific, pre-identified failure mode — it is categorically different
    # from generic agent disagreement and deserves uncapped treatment.
    if kill_thesis_triggered:
        conviction -= 15

    # Step 5: Apply floors
    conviction = apply_conviction_floors(
        conviction, signal, excess_exret, pef, pet,
        bull_count, fund_score, buy_pct,
    )

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
    pp = sig_data.get("pp", 0)
    w52 = sig_data.get("52w", 100)

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
    fs = 50
    exret = sig_data.get("exret", 0)
    bp = sig_data.get("buy_pct", 0)
    pet = sig_data.get("pet", 0)
    pef = sig_data.get("pef", 0)
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
    # Opportunities use conviction-based action mapping (per spec: conv > 55 = BUY)
    # instead of signal-dependent thresholds, because the signal character (B/H/S)
    # already influenced the base conviction. A HOLD-signal stock at conv=60 means
    # the committee found enough merit despite the signal system's caution.
    if entry["conviction"] >= 55:
        entry["action"] = "BUY"
    else:
        # Below BUY threshold: HOLD (watch candidate).
        # Opportunities never get SELL/TRIM — can't reduce
        # a position you don't hold. Low conviction simply means
        # "not compelling enough to buy."
        entry["action"] = "HOLD"

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
    regime = macro_report.get("regime", "")
    if not regime:
        es = macro_report.get("executive_summary", {})
        regime = es.get("regime", "")

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

    # Process portfolio stocks
    for ticker, sig_data in portfolio_signals.items():
        sector = resolve_sector(ticker, sector_map)
        sec_median = sector_medians.get(sector, universe_median)
        entry = _synthesize_with_lookups(
            ticker, sig_data, lookups, fund_report, tech_report,
            sector, sec_median, census_ts_map, regime=regime,
            kill_thesis_triggered=triggered_kill_theses.get(ticker, False),
        )
        entry["is_opportunity"] = False
        entry["kill_thesis_triggered"] = triggered_kill_theses.get(ticker, False)
        entry["risk_diluted"] = risk_diluted
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

    # CIO v7.0 P4: Sector concentration penalty.
    # When 3+ stocks share the same sector in the concordance, apply a small
    # conviction penalty (-2 per stock beyond 2) to discourage over-concentration
    # at the scoring level. Only penalise BUY/ADD actions — we don't want to
    # artificially reduce conviction on HOLD/TRIM/SELL (those are risk decisions).
    sector_counts: Dict[str, int] = {}
    for entry in concordance:
        sec = entry.get("sector", "Other")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    for entry in concordance:
        if entry.get("action") not in ("BUY", "ADD"):
            continue
        sec = entry.get("sector", "Other")
        count = sector_counts.get(sec, 1)
        if count > 2:
            penalty = (count - 2) * 2  # -2 per extra stock beyond 2
            # CIO v8.0 F2: Respect signal-aware conviction floors.
            # BUY-signal stocks already passed quant criteria — sector
            # concentration should nudge, not override. Floor at 45 for
            # BUY signals (preserves actionability), 30 for HOLD.
            floor = 45 if entry.get("signal") == "B" else 30
            entry["conviction"] = max(floor, entry["conviction"] - penalty)
            entry["sector_concentration_penalty"] = penalty

    # CIO v8.0: Re-evaluate opportunity actions after sector concentration penalty.
    # The gate set BUY for conv >= 55, but concentration penalty may have dropped
    # conviction below 55. Re-check and downgrade to HOLD if needed.
    for entry in concordance:
        if entry.get("is_opportunity") and entry["action"] == "BUY":
            if entry["conviction"] < 55:
                entry["action"] = "HOLD"

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
        # Handle concordance.json format: {"date": "...", "stocks": {ticker: data}}
        if "stocks" in previous:
            previous = previous["stocks"]
        prev_map = {k: dict(v, ticker=k) for k, v in previous.items()}
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
) -> Dict[str, Any]:
    """
    Generate complete synthesis JSON output from concordance and agent reports.

    This function produces the flat-key structure that the HTML generator
    expects, so the HTML can be generated deterministically from synthesis.json
    without needing to re-read individual agent reports.
    """
    from collections import Counter

    action_dist = Counter(e["action"] for e in concordance)

    # Extract risk metrics — try multiple field name conventions
    pr = risk_report.get("portfolio_risk", {})
    var_95 = pr.get("var_95_daily") or pr.get("var_95") or 0
    max_dd = pr.get("max_drawdown_1y") or pr.get("max_drawdown") or 0
    port_beta = pr.get("portfolio_beta", 0)
    risk_score = pr.get("risk_score") or 50

    # Macro — try nested paths for agent report compatibility
    indicators = macro_report.get("indicators") or macro_report.get("macro_indicators", {})
    es = macro_report.get("executive_summary", {})
    regime = macro_report.get("regime") or (es.get("regime") if isinstance(es, dict) else "") or "CAUTIOUS"
    macro_score = macro_report.get("macro_score") or (es.get("macro_score") if isinstance(es, dict) else 0) or 0
    rotation = macro_report.get("rotation_phase", "Unknown")

    # Census
    sentiment = census_report.get("sentiment", {})

    output = {
        "version": "v7.0_legacy_review",
        "concordance": concordance,
        "action_distribution": dict(action_dist),
        "changes": changes,
        "sector_gaps": sector_gaps,
        # Flat keys for HTML generator compatibility
        "regime": regime,
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
        "top_holdings_top100": census_report.get("top_holdings_top100", []),
        "missing_popular": census_report.get("portfolio_alignment", {}).get("missing_popular", []),
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
    }

    return output

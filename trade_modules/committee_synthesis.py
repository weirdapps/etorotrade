"""
Committee Synthesis Engine

CIO Review v5: Codifies the CIO synthesis logic that was previously
implemented ad-hoc in /tmp scripts during committee runs. This module
provides a deterministic, testable, versioned implementation of:

1. Agent vote counting with freshness weights
2. Signal-aware base conviction determination
3. BUY-side quality bonuses and conviction floors
4. Penalty/bonus adjustments with caps
5. Action assignment with signal-aware thresholds
6. Concordance matrix construction

Key design principles (from v3 live-run bug fixes):
- Neutral agent views are TRUE NEUTRAL (split evenly, do not lean bullish)
- BUY-side quality bonuses only apply to BUY-signal stocks
- BUY signal sets a conviction floor (the quant system verified criteria)
- SELL conviction is reduced when tech/macro agents disagree
- HOLD signal caps base to prevent easy escalation to ADD
"""

import logging
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


def compute_sector_medians(
    portfolio_signals: Dict[str, Dict],
    sector_map: Dict[str, str],
) -> Tuple[Dict[str, float], float]:
    """Compute median EXRET per sector and universe median."""
    sector_exrets: Dict[str, List[float]] = {}
    all_exrets = []
    for ticker, sig in portfolio_signals.items():
        sector = sector_map.get(ticker, "Other")
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
    universe_median = all_sorted[n // 2] if n else 0
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
) -> Tuple[float, float]:
    """
    Count bull/bear weighted votes from agent views.

    CRITICAL: Neutral views split evenly between bull and bear.
    Only clear directional views count as bull or bear.
    This prevents inflation of bull_pct from neutral agents.
    """
    bull = 0.0
    bear = 0.0

    # Fundamental (freshness 0.8x)
    if fund_score >= 70:
        bull += 0.8
    elif fund_score < 45:
        bear += 0.8
    else:
        bull += 0.4
        bear += 0.4

    # Technical (freshness 1.0x)
    if tech_signal == "ENTER_NOW":
        bull += 1.0
    elif tech_signal in ("AVOID", "EXIT_SOON"):
        bear += 1.0
    elif tech_signal == "WAIT_FOR_PULLBACK":
        bull += 0.6
        bear += 0.4
    elif tech_momentum > 20:
        bull += 0.7
        bear += 0.3
    elif tech_momentum < -20:
        bull += 0.3
        bear += 0.7
    else:
        bull += 0.5
        bear += 0.5

    # Macro (freshness 0.9x)
    if macro_fit == "FAVORABLE":
        bull += 0.9
    elif macro_fit == "UNFAVORABLE":
        bear += 0.9
    else:
        bull += 0.45
        bear += 0.45

    # Census (freshness 0.85x)
    if census_alignment == "ALIGNED":
        bull += 0.85
    elif census_alignment == "DIVERGENT":
        bull += 0.35
        bear += 0.50
    elif census_alignment == "CENSUS_DIV":
        # PIs disagree with signal — they're bullish
        bull += 0.6
        bear += 0.25
    else:
        bull += 0.425
        bear += 0.425

    # News (freshness 1.0x)
    if "HIGH_POSITIVE" in news_impact:
        bull += 1.0
    elif "LOW_POSITIVE" in news_impact:
        bull += 0.7
        bear += 0.3
    elif "HIGH_NEGATIVE" in news_impact:
        bear += 1.0
    elif "LOW_NEGATIVE" in news_impact:
        bull += 0.3
        bear += 0.7
    else:
        bull += 0.5
        bear += 0.5

    # Risk Manager (1.5x for BUY assessment, 2.0x for SELL)
    risk_mult = 2.0 if signal == "S" else 1.5
    if risk_warning:
        bear += risk_mult
    else:
        # No warning is absence of bad news, NOT a bullish vote
        bull += 0.5
        bear += 0.5

    return bull, bear


def determine_base_conviction(
    bull_pct: float,
    signal: str,
    fund_score: float,
    excess_exret: float,
    bear_ratio: float,
) -> int:
    """
    Determine base conviction from agent consensus, anchored by signal.

    CIO v5.2: Uses continuous interpolation instead of 6-bucket discretization
    to break conviction clustering. The old system mapped 50-65% bull to the
    same base=60, losing differentiation. Now, 52% and 64% produce different
    bases (52 vs 62), which propagates through to final scores.

    Key principles:
    - BUY signal: agents can upgrade but floor is 55
    - SELL signal: conviction reduced when agents disagree
    - HOLD signal: base capped at 70 to prevent easy escalation
    """
    # Continuous agent base: linear interpolation from 30 (at 0% bull) to 80 (at 100% bull)
    # This replaces the 6-bucket system that caused 11 stocks at identical conviction
    agent_base = int(30 + (bull_pct / 100) * 50)
    agent_base = max(30, min(80, agent_base))

    if signal == "B":
        base = max(agent_base, 55)
        # BUY-side quality bonus (CIO C1) — ONLY for BUY signals
        if fund_score >= 80:
            base = max(base + 10, 60)
        elif fund_score >= 65:
            base = max(base + 5, 55)
        # Proportional excess EXRET bonus (CIO v5.2)
        # Instead of binary >=12 → +5, scale: 5→+1, 10→+3, 15→+4, 20+→+5
        if excess_exret >= 5:
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

    # Stronger floors for quality combinations
    if signal == "B" and excess_exret > 20 and pef > 0 and pet > 0 and pef < pet:
        conviction = max(conviction, 50)
    if signal == "B" and bull_count >= 4:
        conviction = max(conviction, 50)
    if signal == "B" and fund_score >= 70 and buy_pct >= 70:
        conviction = max(conviction, 50)

    return max(0, min(100, conviction))


def determine_action(
    conviction: int,
    signal: str,
    tech_signal: str,
    risk_warning: bool,
) -> str:
    """Determine action from conviction and signal with signal-aware thresholds."""
    if signal == "S":
        if conviction >= 60:
            return "SELL"
        return "REDUCE"
    elif signal == "B":
        if conviction >= 75:
            return "BUY"
        elif conviction >= 55:
            return "ADD"
        return "HOLD"
    else:  # HOLD signal
        if conviction >= 70:
            return "ADD"
        elif conviction >= 50:
            return "HOLD"
        elif conviction >= 35:
            return "WEAK HOLD"
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

    # Fund score with fallback
    fund_score = fund_data.get("fundamental_score", 50)
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

    # Step 1: Count agent votes
    bull_weight, bear_weight = count_agent_votes(
        fund_score, tech_signal, tech_mom, macro_fit,
        census_alignment, news_impact, risk_warning, signal,
    )

    total_weight = bull_weight + bear_weight
    bull_pct = bull_weight / total_weight * 100 if total_weight > 0 else 50
    bear_ratio = bear_weight / total_weight if total_weight > 0 else 0

    # Step 2: Determine base conviction
    base = determine_base_conviction(
        bull_pct, signal, fund_score, excess_exret, bear_ratio,
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

    conviction = base + bonuses - penalties

    # Step 5: Apply floors
    conviction = apply_conviction_floors(
        conviction, signal, excess_exret, pef, pet,
        bull_count, fund_score, buy_pct,
    )

    # Step 6: Determine action
    action = determine_action(conviction, signal, tech_signal, risk_warning)

    # Risk manager override: downgrade to WEAK HOLD when tech is bearish + risk warns
    if signal == "H" and tech_signal in ("AVOID", "EXIT_SOON") and risk_warning:
        if action not in ("SELL", "REDUCE", "TRIM"):
            action = "WEAK HOLD"

    # Trim escalation: HOLD-signal stocks that are severely overbought or have
    # bearish tech + risk warnings should be TRIM, not stuck at WEAK HOLD.
    # This ensures stocks like PANW (RSI=93) get proper trim treatment.
    if action == "WEAK HOLD" and signal == "H":
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
    }


# Action priority for sorting
ACTION_ORDER = {
    "SELL": 0, "REDUCE": 1, "TRIM": 2, "WEAK HOLD": 3,
    "BUY NEW": 4, "BUY": 5, "ADD": 6, "HOLD": 7, "STRONG HOLD": 8,
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
    if "positive" in v or "favorable" in v:
        return "FAVORABLE"
    if "negative" in v or "unfavorable" in v:
        return "UNFAVORABLE"
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
    """Determine aggregate news impact for a ticker."""
    items = port_news.get(ticker, [])
    if not items:
        return "NEUTRAL"
    impacts = [i.get("impact", "NEUTRAL") for i in items]
    for level in ("HIGH_POSITIVE", "HIGH_NEGATIVE", "LOW_POSITIVE", "LOW_NEGATIVE"):
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
) -> Dict[str, Any]:
    """Run synthesize_stock using pre-built agent lookups."""
    fund_data = fund_report.get("stocks", {}).get(ticker, {})
    if not fund_data:
        fund_data = _fallback_fundamental(sig_data)

    tech_data = tech_report.get("stocks", {}).get(ticker, {})
    if not tech_data:
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
    )


def apply_opportunity_gate(
    entry: Dict[str, Any],
    fund_report: Dict,
    tech_report: Dict,
    macro_fit: str,
    census_alignment: str,
    portfolio_sectors: Dict[str, int],
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

    # Graduated discount
    if confirmations == 0:
        entry["conviction"] -= 15
    elif confirmations == 1:
        entry["conviction"] -= 10
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
    # Opportunities use conviction-based action mapping (per spec: conv > 55 = BUY NEW)
    # instead of signal-dependent thresholds, because the signal character (B/H/S)
    # already influenced the base conviction. A HOLD-signal stock at conv=60 means
    # the committee found enough merit despite the signal system's caution.
    if entry["conviction"] >= 55:
        entry["action"] = "BUY NEW"
    else:
        # Below BUY NEW threshold: HOLD (watch candidate).
        # Opportunities never get SELL/REDUCE/TRIM — can't reduce
        # a position you don't hold. Low conviction simply means
        # "not compelling enough to buy."
        entry["action"] = "HOLD"

    entry["is_opportunity"] = True
    entry["confirmations"] = confirmations

    return entry


def detect_sector_gaps(
    portfolio_sectors: Dict[str, int],
    sector_rankings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Identify sectors missing from the portfolio that are leading in rotation.

    Returns list of {sector, portfolio_exposure, performance_1m, urgency}.
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

        if count == 0 and rank <= 5 and ret_1m > 0:
            urgency = "HIGH" if rank <= 3 else "MEDIUM"
            gaps.append({
                "sector": sector,
                "portfolio_exposure": 0,
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
        sec = sector_map.get(ticker, "Other")
        portfolio_sectors[sec] = portfolio_sectors.get(sec, 0) + 1

    concordance = []

    # Extract regime for sector-regime macro fit derivation (CIO Review F2)
    regime = macro_report.get("regime", "")
    if not regime:
        es = macro_report.get("executive_summary", {})
        regime = es.get("regime", "")

    # Process portfolio stocks
    for ticker, sig_data in portfolio_signals.items():
        sector = sector_map.get(ticker, "Other")
        sec_median = sector_medians.get(sector, universe_median)
        entry = _synthesize_with_lookups(
            ticker, sig_data, lookups, fund_report, tech_report,
            sector, sec_median, census_ts_map, regime=regime,
        )
        entry["is_opportunity"] = False
        concordance.append(entry)

    # Process new opportunities (CIO C2 gate)
    for ticker, sig_data in opportunity_signals.items():
        if ticker in portfolio_signals:
            continue  # Already in portfolio
        sector = opportunity_sector_map.get(ticker, "Other")
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

        # F1: Override base conviction for dual-synthetic stocks using opp_score
        if not has_fund and not has_tech and opp_score > 0:
            opp_base = max(55, min(70, int(opp_score * 0.8)))
            if opp_base > entry["base"]:
                entry["base"] = opp_base
                entry["conviction"] = max(
                    entry["conviction"],
                    opp_base + entry.get("bonuses", 0) - entry.get("penalties", 0),
                )
                entry["opp_score_injected"] = True
        # Apply C2 validation gate
        macro_fit = _resolve_macro_fit(lookups["macro_impl"], ticker)
        cen_align = lookups["div_map"].get(ticker, ("NEUTRAL", 0))[0]
        entry = apply_opportunity_gate(
            entry, fund_report, tech_report, macro_fit,
            cen_align, portfolio_sectors,
        )
        concordance.append(entry)

    # Break conviction ties with composite quality score (CIO v5.2)
    # This prevents 11 stocks at identical conviction=65
    for entry in concordance:
        tiebreak = (
            entry.get("excess_exret", 0) * 0.4
            + entry.get("fund_score", 50) * 0.3
            + (100 - entry.get("beta", 1.0) * 20) * 0.1
            + entry.get("bull_pct", 50) * 0.2
        )
        entry["tiebreak"] = round(tiebreak, 2)

    # Sort by action priority, then conviction desc, then tiebreak desc
    concordance.sort(
        key=lambda x: (ACTION_ORDER.get(x["action"], 9), -x["conviction"], -x["tiebreak"])
    )

    return concordance


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
        "version": "v5.2_continuous_scoring",
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
        "correlation_clusters": risk_report.get("correlation_clusters", []),
        "concentration": risk_report.get("concentration", {}),
        "stress_scenarios": risk_report.get("stress_scenarios", {}),
        "position_limits": risk_report.get("position_limits", {}),
        "hard_constraints": risk_report.get("hard_constraints", []),
        # Opportunities
        "top_opportunities": (opportunity_report or {}).get("top_opportunities", []),
    }

    return output

"""
Committee Report Quality Assurance.

Validates synthesis + agent report data before HTML generation.
Returns a list of gaps/warnings that the CIO can attempt to fix
or flag in the report header.

Three-stage design:
  Stage 0 (normalize): Fix known agent output format mismatches in-place
  Stage 1 (pre-HTML):  Validate synthesis dict + agent reports for completeness
  Stage 2 (post-HTML): Scan generated HTML for rendering gaps (N/A, dashes, empty)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap severity
# ---------------------------------------------------------------------------
CRITICAL = "CRITICAL"  # Data completely missing, section broken
WARNING = "WARNING"    # Data partially missing, degrades quality
INFO = "INFO"          # Minor gap, cosmetic


def normalize_agent_reports(
    macro: Dict[str, Any],
    census: Dict[str, Any],
    news: Dict[str, Any],
    risk: Dict[str, Any],
    opps: Dict[str, Any],
) -> List[str]:
    """
    Stage 0: Fix known agent output format mismatches in-place.

    LLM agents write varying key names and structures. This normalizes
    them to what committee_synthesis.py expects. Mutates dicts in-place.
    Returns list of fixes applied (for logging).
    """
    fixes = []

    # ── Macro: portfolio_implications ──
    # Synthesis reads: macro["portfolio_implications"][ticker].get("fit")
    # Agents may write: macro["stock_macro_fit"][ticker] = "NEUTRAL" (string)
    if "portfolio_implications" not in macro and "stock_macro_fit" in macro:
        smf = macro["stock_macro_fit"]
        if isinstance(smf, dict):
            pi = {}
            for tkr, val in smf.items():
                if isinstance(val, str):
                    pi[tkr] = {"fit": val, "rationale": ""}
                elif isinstance(val, dict):
                    pi[tkr] = val
            macro["portfolio_implications"] = pi
            fixes.append("macro: stock_macro_fit → portfolio_implications")

    # ── Macro: indicators nested → flat ──
    # Synthesis now handles this (v28.0), but defensive fallback here too.
    indicators = macro.get("macro_indicators") or macro.get("indicators", {})
    _yc = indicators.get("yield_curve", {})
    if isinstance(_yc, dict) and _yc:
        indicators.setdefault("us_10y_yield", _yc.get("10y", 0))
        indicators.setdefault("yield_curve_10y_2y", _yc.get("spread_2_10", 0))
        fixes.append("macro: flattened yield_curve nested keys")
    _cur = indicators.get("currency", {})
    if isinstance(_cur, dict) and _cur:
        indicators.setdefault("dxy", _cur.get("dxy", 0))
        indicators.setdefault("eur_usd", _cur.get("eur_usd", 0))
        fixes.append("macro: flattened currency nested keys")
    if indicators.get("oil_brent") and not indicators.get("brent_crude"):
        indicators["brent_crude"] = indicators["oil_brent"]
        fixes.append("macro: oil_brent → brent_crude alias")
    # Write back under both keys so synthesis finds it
    if indicators:
        macro["indicators"] = indicators
        macro["macro_indicators"] = indicators

    # ── Macro: regime as string vs dict ──
    regime = macro.get("regime")
    if isinstance(regime, str) and regime:
        macro["regime"] = {"classification": regime}
        fixes.append(f"macro: regime string '{regime}' → dict")

    # ── Census: missing_popular as list vs dict ──
    # Synthesis reads: census["missing_popular"].get("stocks_not_in_portfolio_but_popular")
    mp = census.get("missing_popular")
    if isinstance(mp, list):
        census["missing_popular"] = {"stocks_not_in_portfolio_but_popular": mp}
        fixes.append("census: missing_popular list → dict")

    # ── Census: per_stock / stocks key ──
    # Synthesis reads census["stocks"] or census["per_stock"]
    if not census.get("stocks") and census.get("per_stock"):
        census["stocks"] = census["per_stock"]
        fixes.append("census: per_stock → stocks alias")
    elif not census.get("per_stock") and census.get("stocks"):
        census["per_stock"] = census["stocks"]

    # ── News: breaking_news may be top-level or nested ──
    if not news.get("breaking_news") and news.get("market_news"):
        news["breaking_news"] = news["market_news"]
        fixes.append("news: market_news → breaking_news alias")

    # ── News: portfolio_news as list vs dict ──
    pn = news.get("portfolio_news")
    if isinstance(pn, list):
        by_ticker = {}
        for item in pn:
            tkr = item.get("ticker", "")
            if tkr:
                by_ticker[tkr] = item
        news["portfolio_news"] = by_ticker
        fixes.append("news: portfolio_news list → dict by ticker")

    # ── Risk: consensus_warnings may be a dict instead of list ──
    cw = risk.get("consensus_warnings")
    if isinstance(cw, dict):
        risk["consensus_warnings"] = [{"ticker": k, **v} if isinstance(v, dict) else {"ticker": k, "warning": v}
                                       for k, v in cw.items()]
        fixes.append("risk: consensus_warnings dict → list")

    # ── Opportunities: top_opportunities key variants ──
    if not opps.get("top_opportunities"):
        for alt in ("opportunities", "screened_stocks", "results"):
            if opps.get(alt):
                opps["top_opportunities"] = opps[alt]
                fixes.append(f"opps: {alt} → top_opportunities alias")
                break

    if fixes:
        logger.info("Normalized %d agent report format issues: %s", len(fixes), "; ".join(fixes))

    return fixes


def validate_pre_html(
    synthesis: Dict[str, Any],
    fund: Dict[str, Any],
    tech: Dict[str, Any],
    macro: Dict[str, Any],
    census: Dict[str, Any],
    news: Dict[str, Any],
    opps: Dict[str, Any],
    risk: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Pass 1: Validate data completeness before HTML generation.

    Returns list of {severity, section, field, message} dicts.
    """
    gaps = []

    def gap(severity, section, field, msg):
        gaps.append({"severity": severity, "section": section, "field": field, "message": msg})

    concordance = synthesis.get("concordance", [])
    indicators = synthesis.get("indicators", {})

    # ── Executive Summary ──
    if not concordance:
        gap(CRITICAL, "Executive Summary", "concordance", "No concordance data — report will be empty")
    regime = synthesis.get("regime", "")
    if not regime:
        gap(WARNING, "Executive Summary", "regime", "No regime classification")
    p_beta = synthesis.get("portfolio_beta", 0)
    if not p_beta or abs(p_beta - 1.0) < 1e-9:
        gap(WARNING, "Executive Summary", "portfolio_beta",
            f"Portfolio beta is {p_beta} — may be default, not computed")

    # ── Macro Indicators ──
    for key, label in [("vix", "VIX"), ("us_10y_yield", "10Y Yield"), ("eur_usd", "EUR/USD")]:
        v = indicators.get(key)
        if v is None or v == 0:
            gap(WARNING, "Macro", key, f"{label} is missing or zero")
    if not synthesis.get("sector_rankings"):
        gap(WARNING, "Macro", "sector_rankings", "No sector rankings data")

    # ── Fundamental ──
    fund_stocks = fund.get("stocks", {})
    if not fund_stocks:
        gap(CRITICAL, "Fundamental", "stocks", "No fundamental stock data")
    else:
        no_score = [t for t, d in fund_stocks.items() if not d.get("fundamental_score")]
        if no_score:
            gap(WARNING, "Fundamental", "fundamental_score",
                f"{len(no_score)} stocks missing fundamental_score: {', '.join(no_score[:5])}")
        no_km = [t for t, d in fund_stocks.items()
                 if not d.get("key_metrics") and not d.get("piotroski")]
        if no_km:
            gap(WARNING, "Fundamental", "key_metrics",
                f"{len(no_km)} stocks missing key_metrics: {', '.join(no_km[:5])}")

    # ── Technical ──
    tech_stocks = tech.get("stocks", {})
    if not tech_stocks:
        gap(CRITICAL, "Technical", "stocks", "No technical stock data")
    else:
        no_rsi = [t for t, d in tech_stocks.items() if d.get("rsi") is None]
        if no_rsi:
            gap(WARNING, "Technical", "rsi",
                f"{len(no_rsi)} stocks missing RSI: {', '.join(no_rsi[:5])}")
        no_trend = [t for t, d in tech_stocks.items()
                    if not d.get("trend") and not d.get("adx_trend")]
        if no_trend:
            gap(INFO, "Technical", "trend",
                f"{len(no_trend)} stocks missing trend data")

    # ── News ──
    breaking = synthesis.get("breaking_news", [])
    if not breaking:
        gap(WARNING, "News", "breaking_news",
            "No breaking news items — news section will be empty")
    ec = synthesis.get("earnings_calendar", {})
    if not ec.get("next_2_weeks"):
        gap(INFO, "News", "earnings_calendar", "No upcoming earnings data")

    # ── Census ──
    crowd = synthesis.get("crowd_sentiment", "")
    if not crowd or crowd == "NEUTRAL":
        # NEUTRAL is valid but worth noting
        pass
    if not census.get("stocks") and not census.get("per_stock"):
        gap(INFO, "Census", "per_stock", "No per-stock census data")

    # ── Opportunities ──
    opp_list = opps.get("top_opportunities", [])
    if not opp_list:
        gap(INFO, "Opportunities", "top_opportunities", "No new opportunities found")
    else:
        no_why = [o.get("ticker", "?") for o in opp_list
                  if not o.get("why_compelling") and not o.get("rationale")]
        if no_why:
            gap(WARNING, "Opportunities", "why_compelling",
                f"{len(no_why)} opportunities missing rationale: {', '.join(no_why[:5])}")

    # ── Risk ──
    pr = risk.get("portfolio_risk", {})
    if not pr:
        gap(WARNING, "Risk", "portfolio_risk", "No portfolio risk data")
    if not risk.get("correlation_clusters") and not risk.get("crisis_correlation_clusters"):
        gap(INFO, "Risk", "correlation_clusters", "No correlation cluster data")

    # ── Concordance quality ──
    no_waterfall = [e["ticker"] for e in concordance if not e.get("conviction_waterfall")]
    if no_waterfall and len(no_waterfall) > len(concordance) * 0.5:
        gap(WARNING, "Concordance", "conviction_waterfall",
            f"{len(no_waterfall)}/{len(concordance)} entries missing conviction waterfall")

    no_action = [e["ticker"] for e in concordance if not e.get("action")]
    if no_action:
        gap(CRITICAL, "Concordance", "action",
            f"{len(no_action)} entries missing action assignment")

    return gaps


def validate_post_html(html: str) -> List[Dict[str, str]]:
    """
    Pass 2: Scan generated HTML for rendering gaps.

    Looks for patterns that indicate missing data made it into the HTML.
    """
    gaps = []

    def gap(severity, section, field, msg):
        gaps.append({"severity": severity, "section": section, "field": field, "message": msg})

    # Count N/A cells (excluding legitimate ones like non-equity PE)
    na_cells = len(re.findall(r'>N/A</span>', html))
    if na_cells > 20:
        gap(WARNING, "HTML", "N/A cells",
            f"{na_cells} N/A cells found — may indicate missing data flows")

    # Count empty cells
    empty_cells = len(re.findall(r'color:#64748b;">\s*</span></td>', html))
    if empty_cells > 5:
        gap(WARNING, "HTML", "empty cells",
            f"{empty_cells} empty cells found")

    # Check for "?" trend values
    q_marks = len(re.findall(r'>\?</span>', html))
    if q_marks > 3:
        gap(WARNING, "HTML", "question marks",
            f"{q_marks} '?' values found — trend or signal data not mapped")

    # Check for 0% EXRET that might be wrong
    zero_exret = len(re.findall(r'>0%</td>', html))
    if zero_exret > 5:
        gap(INFO, "HTML", "zero EXRET",
            f"{zero_exret} cells show 0% — may be missing expected return data")

    # Check key sections exist
    for section in ["Executive Summary", "Macro &amp; Market Context",
                    "News &amp; Events", "Technical Analysis",
                    "Fundamental Deep Dive", "Sentiment &amp; Census"]:
        if section not in html:
            gap(CRITICAL, "HTML", section, f"Section '{section}' not found in HTML")

    # Check section has content (not just header)
    for section_name, min_size in [("News &amp; Events", 500),
                                    ("Fundamental Deep Dive", 1000),
                                    ("Technical Analysis", 800)]:
        idx = html.find(section_name)
        if idx >= 0:
            # Find the section content between this header and the next section
            next_section = html.find('<h2', idx + len(section_name))
            if next_section < 0:
                next_section = len(html)
            section_html = html[idx:next_section]
            if len(section_html) < min_size:
                gap(WARNING, "HTML", section_name,
                    f"Section '{section_name}' seems too short ({len(section_html)} chars)")

    return gaps


def run_qa(
    synthesis: Dict[str, Any],
    fund: Dict[str, Any],
    tech: Dict[str, Any],
    macro: Dict[str, Any],
    census: Dict[str, Any],
    news: Dict[str, Any],
    opps: Dict[str, Any],
    risk: Dict[str, Any],
    html: str = "",
    normalize: bool = True,
) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Full QA pass. Returns (passed, gaps).

    If normalize=True (default), runs Stage 0 normalization on agent
    reports in-place BEFORE validation. This auto-fixes known format
    mismatches so they don't show up as gaps.

    passed = True if no CRITICAL gaps found.
    gaps = combined list from all stages.
    """
    gaps = []

    # Stage 0: Normalize agent reports (mutates in-place)
    if normalize:
        fixes = normalize_agent_reports(macro, census, news, risk, opps)
        for fix_msg in fixes:
            gaps.append({"severity": INFO, "section": "Normalize", "field": "auto-fix", "message": fix_msg})

    # Stage 1: Pre-HTML validation
    gaps.extend(validate_pre_html(synthesis, fund, tech, macro, census, news, opps, risk))

    # Stage 2: Post-HTML validation
    if html:
        gaps.extend(validate_post_html(html))

    criticals = [g for g in gaps if g["severity"] == CRITICAL]
    warnings = [g for g in gaps if g["severity"] == WARNING]
    infos = [g for g in gaps if g["severity"] == INFO]

    logger.info("QA result: %d critical, %d warning, %d info (%d auto-fixed)",
                len(criticals), len(warnings), len(infos), len(fixes) if normalize else 0)

    passed = len(criticals) == 0
    return passed, gaps


def format_qa_report(gaps: List[Dict[str, str]]) -> str:
    """Format QA gaps as a human-readable summary."""
    if not gaps:
        return "QA PASSED: No data gaps detected."

    criticals = [g for g in gaps if g["severity"] == CRITICAL]
    warnings = [g for g in gaps if g["severity"] == WARNING]
    auto_fixes = [g for g in gaps if g["section"] == "Normalize"]
    infos = [g for g in gaps if g["severity"] == INFO and g["section"] != "Normalize"]

    lines = []
    if auto_fixes:
        lines.append(f"AUTO-FIXED ({len(auto_fixes)}):")
        for g in auto_fixes:
            lines.append(f"  {g['message']}")
    if criticals:
        lines.append(f"CRITICAL ({len(criticals)}):")
        for g in criticals:
            lines.append(f"  [{g['section']}] {g['field']}: {g['message']}")
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        for g in warnings:
            lines.append(f"  [{g['section']}] {g['field']}: {g['message']}")
    if infos:
        lines.append(f"INFO ({len(infos)}):")
        for g in infos:
            lines.append(f"  [{g['section']}] {g['field']}: {g['message']}")

    status = "FAILED" if criticals else "PASSED with warnings" if warnings else "PASSED"
    lines.insert(0, f"QA {status}: {len(auto_fixes)} auto-fixed, {len(criticals)} critical, {len(warnings)} warnings, {len(infos)} info")
    return "\n".join(lines)

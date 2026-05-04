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

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap severity
# ---------------------------------------------------------------------------
CRITICAL = "CRITICAL"  # Data completely missing, section broken
WARNING = "WARNING"  # Data partially missing, degrades quality
INFO = "INFO"  # Minor gap, cosmetic


def normalize_agent_reports(
    macro: dict[str, Any],
    census: dict[str, Any],
    news: dict[str, Any],
    risk: dict[str, Any],
    opps: dict[str, Any],
) -> list[str]:
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
    # or macro["stocks"][ticker].macro_fit (BUG 2 fix)
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

    # BUG 2 fix: Agent writes per-stock macro_fit in macro["stocks"][ticker]
    # but not in portfolio_implications. Extract macro_fit from stocks.
    if "stocks" in macro:
        pi = macro.setdefault("portfolio_implications", {})
        stocks = macro["stocks"]
        if isinstance(stocks, dict):
            added_count = 0
            for tkr, data in stocks.items():
                if tkr in pi:
                    continue  # portfolio_implications takes precedence
                if isinstance(data, dict):
                    fit = data.get("macro_fit") or data.get("fit")
                    if fit:
                        pi[tkr] = {
                            "fit": fit,
                            "rationale": data.get("notes", data.get("rationale", "")),
                        }
                        added_count += 1
            if added_count > 0:
                fixes.append(f"macro: extracted macro_fit from stocks for {added_count} tickers")

    # ── Macro: indicators key aliases ──
    # Agents may write "key_indicators" instead of "indicators".
    if "key_indicators" in macro and "indicators" not in macro:
        macro["indicators"] = macro.pop("key_indicators")
        fixes.append("macro: key_indicators → indicators")

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
        risk["consensus_warnings"] = [
            {"ticker": k, **v} if isinstance(v, dict) else {"ticker": k, "warning": v}
            for k, v in cw.items()
        ]
        fixes.append("risk: consensus_warnings dict → list")

    # ── Risk: top_warnings → consensus_warnings alias ──
    if not risk.get("consensus_warnings") and risk.get("top_warnings"):
        risk["consensus_warnings"] = risk["top_warnings"]
        fixes.append("risk: top_warnings → consensus_warnings alias")

    # ── Risk: risk_warnings_by_stock → consensus_warnings (v33.0) ──
    rwbs = risk.get("risk_warnings_by_stock", {})
    if rwbs and isinstance(rwbs, dict):
        existing = risk.setdefault("consensus_warnings", [])
        existing_tickers = {w.get("ticker") for w in existing if isinstance(w, dict)}
        added = 0
        for tkr, data in rwbs.items():
            if tkr in existing_tickers:
                continue
            if isinstance(data, dict):
                existing.append(
                    {
                        "ticker": tkr,
                        "severity": data.get("severity", "MODERATE"),
                        "reason": str(data.get("warning", data.get("reason", "")))[:200],
                    }
                )
            elif isinstance(data, str):
                existing.append({"ticker": tkr, "severity": "MODERATE", "reason": data[:200]})
            elif isinstance(data, list):
                existing.append(
                    {
                        "ticker": tkr,
                        "severity": "MODERATE",
                        "reason": "; ".join(str(w) for w in data)[:200],
                    }
                )
            added += 1
        if added:
            fixes.append(
                f"risk: ingested {added} risk_warnings_by_stock entries into consensus_warnings"
            )

    # ── News: synthesise breaking_news from key_themes/sector_news ──
    if not news.get("breaking_news"):
        items = []
        for theme in news.get("key_themes", []):
            if isinstance(theme, str) and theme.strip():
                items.append(
                    {
                        "headline": theme.strip(),
                        "impact": "NEUTRAL",
                        "affected_tickers": [],
                    }
                )
            elif isinstance(theme, dict):
                hl = theme.get("theme", theme.get("title", theme.get("headline", "")))
                if hl:
                    items.append(
                        {
                            "headline": hl,
                            "impact": theme.get("sentiment", theme.get("impact", "NEUTRAL")),
                            "affected_tickers": theme.get(
                                "affected_tickers", theme.get("tickers", [])
                            ),
                        }
                    )
        if items:
            news["breaking_news"] = items
            fixes.append(f"news: synthesised breaking_news from key_themes ({len(items)} items)")

    # ── Macro: risk_indicators → indicators (v34.0) ──
    # Agents may write VIX/credit_spreads under "risk_indicators" instead of "indicators".
    if "risk_indicators" in macro and not macro.get("indicators"):
        ri = macro["risk_indicators"]
        if isinstance(ri, dict):
            macro["indicators"] = ri
            macro["macro_indicators"] = ri
            fixes.append("macro: risk_indicators → indicators alias")
    elif "risk_indicators" in macro and macro.get("indicators"):
        ri = macro["risk_indicators"]
        ind = macro["indicators"]
        if isinstance(ri, dict) and isinstance(ind, dict):
            merged = 0
            for k, v in ri.items():
                if k not in ind:
                    ind[k] = v
                    merged += 1
            if merged:
                fixes.append(f"macro: merged {merged} fields from risk_indicators into indicators")

    # ── Macro: extract numeric indicators from narrative context (v34.0) ──
    # Agents often embed EUR/USD, 10Y yield as context strings, not numeric fields.
    ind = macro.get("indicators", macro.get("macro_indicators", {}))
    if isinstance(ind, dict):
        ri = macro.get("risk_indicators", {})
        if not ind.get("eur_usd"):
            fx = macro.get("fx_impact", {})
            for src_text in [str(fx), str(ri.get("usd_context", ""))]:
                m = re.search(r"EUR/USD\s*(?:at\s+)?(\d+\.\d+)", src_text)
                if m:
                    ind["eur_usd"] = float(m.group(1))
                    fixes.append(f"macro: extracted eur_usd={ind['eur_usd']} from context")
                    break
        if not ind.get("us_10y_yield"):
            for src_text in [str(ri.get("yield_curve_context", "")), str(ri)]:
                m = re.search(
                    r"10[- ]?year\s+(?:Treasury\s+)?(?:near|at|~)?\s*(\d+\.?\d*)\s*%",
                    src_text,
                    re.IGNORECASE,
                )
                if m:
                    ind["us_10y_yield"] = float(m.group(1))
                    fixes.append(
                        f"macro: extracted us_10y_yield={ind['us_10y_yield']} from context"
                    )
                    break
        macro["indicators"] = ind
        macro["macro_indicators"] = ind

    # ── Risk: derive stress_scenarios from tail_risks (v34.0) ──
    if not risk.get("stress_scenarios"):
        tail = risk.get("tail_risks", [])
        if isinstance(tail, list) and tail:
            impact_map = {"CATASTROPHIC": -25, "HIGH": -15, "MEDIUM": -10, "LOW": -5}
            scenarios = []
            for tr in tail[:3]:
                if isinstance(tr, dict):
                    imp = str(tr.get("impact", "MEDIUM")).upper()
                    scenarios.append(
                        {
                            "name": tr.get("risk", tr.get("name", "Unknown")),
                            "portfolio_impact_pct": impact_map.get(imp, -10),
                            "probability": tr.get("probability", "MEDIUM"),
                        }
                    )
                elif isinstance(tr, str):
                    scenarios.append(
                        {"name": tr, "portfolio_impact_pct": -10, "probability": "MEDIUM"}
                    )
            if scenarios:
                risk["stress_scenarios"] = scenarios
                fixes.append(
                    f"risk: derived stress_scenarios from tail_risks ({len(scenarios)} scenarios)"
                )

    # ── Macro: derive portfolio_implications from sector_rankings (v34.0) ──
    # When agent provides sector_rankings with outlook but no portfolio_implications,
    # build per-sector fit so _resolve_macro_fit can find data.
    if not macro.get("portfolio_implications") and macro.get("sector_rankings"):
        sr = macro["sector_rankings"]
        if isinstance(sr, dict):
            pi: dict[str, Any] = {}
            outlook_to_fit = {
                "OVERWEIGHT": "FAVORABLE",
                "FAVORABLE": "FAVORABLE",
                "UNDERWEIGHT": "UNFAVORABLE",
                "UNFAVORABLE": "UNFAVORABLE",
                "EQUAL": "NEUTRAL",
                "NEUTRAL": "NEUTRAL",
            }
            for _sector_name, data in sr.items():
                if isinstance(data, dict):
                    outlook = data.get("recommendation", data.get("outlook", "NEUTRAL"))
                    fit = outlook_to_fit.get(str(outlook).upper(), "NEUTRAL")
                    data["fit"] = fit
            rank_order = {
                "OVERWEIGHT": 0,
                "FAVORABLE": 0,
                "EQUAL": 1,
                "NEUTRAL": 1,
                "UNDERWEIGHT": 2,
                "UNFAVORABLE": 2,
            }
            sorted_sectors = sorted(
                sr.items(),
                key=lambda x: rank_order.get(
                    str(x[1].get("recommendation", x[1].get("outlook", "NEUTRAL"))).upper(), 1
                )
                if isinstance(x[1], dict)
                else 1,
            )
            for i, (_sname, sdata) in enumerate(sorted_sectors, 1):
                if isinstance(sdata, dict) and "rank" not in sdata:
                    sdata["rank"] = i
            fixes.append("macro: enriched sector_rankings with fit from outlook")

    # ── Census: derive alignment from trend/interest_level (v34.0) ──
    # Agents write trend=ACCUMULATING/DISTRIBUTING/STABLE and interest_level=HIGH/MODERATE/LOW/MINIMAL
    # but synthesis reads alignment/sentiment/signal. Map trend→alignment.
    census_stocks = census.get("stocks", census.get("per_stock", {}))
    if isinstance(census_stocks, dict):
        mapped = 0
        for tkr, data in census_stocks.items():
            if not isinstance(data, dict):
                continue
            if data.get("alignment") or data.get("sentiment") or data.get("signal"):
                continue
            trend = str(data.get("trend", "")).upper()
            interest = str(data.get("interest_level", "")).upper()
            if trend in ("ACCUMULATING", "STRONG_ACCUMULATION", "ACCUMULATION"):
                data["alignment"] = "ACCUMULATING"
                mapped += 1
            elif trend in ("DISTRIBUTING", "STRONG_DISTRIBUTION", "DISTRIBUTION"):
                data["alignment"] = "DISTRIBUTING"
                mapped += 1
            elif interest in ("HIGH", "MODERATE") and trend == "STABLE":
                data["alignment"] = "CONSENSUS_ALIGNED"
                mapped += 1
        if mapped:
            fixes.append(f"census: derived alignment from trend for {mapped} stocks")

    # ── News: stock_news → portfolio_news with impact derivation (v34.0) ──
    # Agents write stock_news[ticker].news_sentiment/impact_magnitude,
    # but synthesis reads portfolio_news[ticker].impact.
    if not news.get("portfolio_news") and news.get("stock_news"):
        sn = news["stock_news"]
        if isinstance(sn, dict):
            pn: dict[str, Any] = {}
            for tkr, data in sn.items():
                if not isinstance(data, dict):
                    continue
                sentiment = str(data.get("news_sentiment", "NEUTRAL")).upper()
                magnitude = str(data.get("impact_magnitude", "LOW")).upper()
                if sentiment in ("POSITIVE", "VERY_POSITIVE"):
                    impact = "HIGH_POSITIVE" if magnitude == "HIGH" else "LOW_POSITIVE"
                elif sentiment in ("NEGATIVE", "VERY_NEGATIVE"):
                    impact = "HIGH_NEGATIVE" if magnitude == "HIGH" else "LOW_NEGATIVE"
                elif sentiment == "MIXED":
                    impact = "LOW_POSITIVE" if magnitude == "HIGH" else "NEUTRAL"
                else:
                    impact = "NEUTRAL"
                pn[tkr] = [
                    {
                        "impact": impact,
                        "headline": "; ".join(data.get("recent_headlines", [])[:2])[:200],
                        "catalyst": data.get("catalyst_type", ""),
                    }
                ]
            news["portfolio_news"] = pn
            fixes.append(f"news: derived portfolio_news from stock_news ({len(pn)} tickers)")

    # ── Risk: derive risk_warning from risk_level (v34.0) ──
    # Agents write risk_level=HIGH/EXTREME but not risk_warning boolean.
    risk_stocks = risk.get("stocks", {})
    if isinstance(risk_stocks, dict):
        warned = 0
        for tkr, data in risk_stocks.items():
            if not isinstance(data, dict):
                continue
            if "risk_warning" not in data:
                level = str(data.get("risk_level", "")).upper()
                data["risk_warning"] = level in ("HIGH", "EXTREME")
                if data["risk_warning"]:
                    warned += 1
        if warned:
            fixes.append(f"risk: derived risk_warning from risk_level for {warned} stocks")

    # ── Risk: string consensus_warnings/risk_warnings_by_stock → empty list (v34.0) ──
    for key in ("consensus_warnings", "risk_warnings_by_stock"):
        if isinstance(risk.get(key), str):
            risk[key] = []
            fixes.append(f"risk: {key} was string, reset to empty list")

    # ── Opportunities: top_opportunities key variants ──
    if not opps.get("top_opportunities"):
        for alt in ("opportunities", "screened_stocks", "results"):
            if opps.get(alt):
                opps["top_opportunities"] = opps[alt]
                fixes.append(f"opps: {alt} → top_opportunities alias")
                break

    if fixes:
        from yahoofinance.utils.log_safety import safe_for_log

        logger.info(
            "Normalized %d agent report format issues: %s",
            len(fixes),
            safe_for_log("; ".join(fixes), max_len=500),
        )

    return fixes


def validate_pre_html(
    synthesis: dict[str, Any],
    fund: dict[str, Any],
    tech: dict[str, Any],
    macro: dict[str, Any],
    census: dict[str, Any],
    news: dict[str, Any],
    opps: dict[str, Any],
    risk: dict[str, Any],
) -> list[dict[str, str]]:
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
        gap(
            CRITICAL,
            "Executive Summary",
            "concordance",
            "No concordance data — report will be empty",
        )
    regime = synthesis.get("regime", "")
    if not regime:
        gap(WARNING, "Executive Summary", "regime", "No regime classification")
    p_beta = synthesis.get("portfolio_beta", 0)
    if not p_beta or abs(p_beta - 1.0) < 1e-9:
        gap(
            WARNING,
            "Executive Summary",
            "portfolio_beta",
            f"Portfolio beta is {p_beta} — may be default, not computed",
        )

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
            gap(
                WARNING,
                "Fundamental",
                "fundamental_score",
                f"{len(no_score)} stocks missing fundamental_score: {', '.join(no_score[:5])}",
            )
        no_km = [
            t for t, d in fund_stocks.items() if not d.get("key_metrics") and not d.get("piotroski")
        ]
        if no_km:
            gap(
                WARNING,
                "Fundamental",
                "key_metrics",
                f"{len(no_km)} stocks missing key_metrics: {', '.join(no_km[:5])}",
            )

    # ── Technical ──
    tech_stocks = tech.get("stocks", {})
    if not tech_stocks:
        gap(CRITICAL, "Technical", "stocks", "No technical stock data")
    else:
        no_rsi = [t for t, d in tech_stocks.items() if d.get("rsi") is None]
        if no_rsi:
            gap(
                WARNING,
                "Technical",
                "rsi",
                f"{len(no_rsi)} stocks missing RSI: {', '.join(no_rsi[:5])}",
            )
        no_trend = [
            t for t, d in tech_stocks.items() if not d.get("trend") and not d.get("adx_trend")
        ]
        if no_trend:
            gap(INFO, "Technical", "trend", f"{len(no_trend)} stocks missing trend data")
        # v35.0: Verify normalizer produced timing_signal and momentum_score
        no_timing = [
            t for t, d in tech_stocks.items() if isinstance(d, dict) and not d.get("timing_signal")
        ]
        if no_timing and len(no_timing) == len(tech_stocks):
            gap(
                CRITICAL,
                "Technical",
                "timing_signal",
                "No stocks have timing_signal — normalizer did not run or "
                "agent used an unrecognized field name (check for timing/technical_signal/entry_timing)",
            )
        no_mom = [
            t for t, d in tech_stocks.items() if isinstance(d, dict) and not d.get("momentum_score")
        ]
        if no_mom and len(no_mom) == len(tech_stocks):
            gap(
                CRITICAL,
                "Technical",
                "momentum_score",
                "No stocks have momentum_score — normalizer did not run or "
                "agent used an unrecognized field name (check for momentum/momentum_pct)",
            )
        no_macd = [
            t for t, d in tech_stocks.items() if isinstance(d, dict) and not d.get("macd_signal")
        ]
        if no_macd and len(no_macd) == len(tech_stocks):
            gap(
                WARNING,
                "Technical",
                "macd_signal",
                "No stocks have macd_signal — check for macd field name variant",
            )

    # ── News ──
    breaking = synthesis.get("breaking_news", [])
    if not breaking:
        gap(WARNING, "News", "breaking_news", "No breaking news items — news section will be empty")
    ec = synthesis.get("earnings_calendar", {})
    ec_list = ec if isinstance(ec, list) else ec.get("next_2_weeks", [])
    if not ec_list:
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
        no_why = [
            o.get("ticker", "?")
            for o in opp_list
            if not o.get("why_compelling") and not o.get("rationale")
        ]
        if no_why:
            gap(
                WARNING,
                "Opportunities",
                "why_compelling",
                f"{len(no_why)} opportunities missing rationale: {', '.join(no_why[:5])}",
            )

    # ── Risk ──
    pr = risk.get("portfolio_risk", {})
    if not pr:
        gap(WARNING, "Risk", "portfolio_risk", "No portfolio risk data")
    if not risk.get("correlation_clusters") and not risk.get("crisis_correlation_clusters"):
        gap(INFO, "Risk", "correlation_clusters", "No correlation cluster data")
    if not risk.get("stress_scenarios") and not risk.get("tail_risks"):
        gap(
            WARNING,
            "Risk",
            "stress_scenarios",
            "No stress scenarios or tail risks — stress test section will be empty",
        )

    # ── Concordance quality ──
    no_waterfall = [e["ticker"] for e in concordance if not e.get("conviction_waterfall")]
    if no_waterfall and len(no_waterfall) > len(concordance) * 0.5:
        gap(
            WARNING,
            "Concordance",
            "conviction_waterfall",
            f"{len(no_waterfall)}/{len(concordance)} entries missing conviction waterfall",
        )

    no_action = [e["ticker"] for e in concordance if not e.get("action")]
    if no_action:
        gap(
            CRITICAL, "Concordance", "action", f"{len(no_action)} entries missing action assignment"
        )

    # ── Cross-validation: agent data provided vs synthesis consumed (v33.0) ──
    pn = news.get("portfolio_news", {})
    if isinstance(pn, dict) and len(pn) >= 5:
        news_used = sum(1 for e in concordance if e.get("news_impact", "NEUTRAL") != "NEUTRAL")
        if news_used == 0:
            gap(
                CRITICAL,
                "News",
                "news_impact_flow",
                f"News agent provided data for {len(pn)} tickers but "
                f"0 concordance entries have non-NEUTRAL news_impact",
            )

    rw_provided = set()
    for w in risk.get("consensus_warnings", []):
        if isinstance(w, dict) and w.get("ticker"):
            rw_provided.add(w["ticker"])
    for tkr in risk.get("risk_warnings_by_stock", {}):
        rw_provided.add(tkr)
    if len(rw_provided) >= 3:
        rw_used = sum(1 for e in concordance if e.get("risk_warning"))
        if rw_used == 0:
            gap(
                CRITICAL,
                "Risk",
                "risk_warning_flow",
                f"Risk agent provided warnings for {len(rw_provided)} tickers "
                f"but 0 concordance entries have risk_warning=True",
            )

    return gaps


# ---------------------------------------------------------------------------
# Stage 1b: Synthesis output completeness (schema-agnostic)
# ---------------------------------------------------------------------------

# Fields that MUST be populated for a usable report.
# Each entry: (dotted_path, description, is_critical)
# "critical" means the report is broken without it; "warning" means degraded.
_SYNTH_REQUIRED = [
    ("regime", "Market regime classification", True),
    ("risk_score", "Portfolio risk score", True),
    ("indicators.vix", "VIX index value", True),
    ("concordance", "Concordance matrix", True),
]
_SYNTH_IMPORTANT = [
    ("indicators.us_10y_yield", "10-Year Treasury yield"),
    ("indicators.eur_usd", "EUR/USD exchange rate"),
    ("indicators.dxy", "Dollar index"),
    ("portfolio_beta", "Portfolio beta"),
    ("sector_rankings", "Sector rankings"),
    ("breaking_news", "Breaking news items"),
    ("stress_scenarios", "Stress scenarios"),
    ("macro_score", "Macro score"),
    ("var_95", "Value at Risk (95%)"),
    ("macro_label", "Macro regime label"),
]


def _get_nested(d: dict, path: str):
    """Resolve dotted path like 'indicators.vix' on a dict."""
    for key in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def validate_synthesis_completeness(
    synthesis: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Stage 1b: Validate that synthesis output has non-zero values for
    critical display fields.

    This catches problems regardless of which agent field name caused them,
    because it validates the OUTPUT not the INPUT.

    Returns list of {severity, section, field, message} dicts.
    A synthesis that fails 3+ required fields is tagged CRITICAL with
    a single umbrella gap so the repair loop can trigger.
    """
    gaps = []

    def gap(severity, section, field, msg):
        gaps.append({"severity": severity, "section": section, "field": field, "message": msg})

    failed_critical = 0

    for path, label, is_crit in _SYNTH_REQUIRED:
        val = _get_nested(synthesis, path)
        if val is None or val == 0 or val == "" or val == []:
            sev = CRITICAL if is_crit else WARNING
            gap(sev, "Synthesis", path, f"{label} is missing/zero in synthesis output")
            if is_crit:
                failed_critical += 1

    failed_important = 0
    for path, label in _SYNTH_IMPORTANT:
        val = _get_nested(synthesis, path)
        if val is None or val == 0 or val == "" or val == [] or val == {}:
            gap(WARNING, "Synthesis", path, f"{label} is missing/zero in synthesis output")
            failed_important += 1

    # If many fields are zeroed simultaneously, it's a systemic extraction failure
    if failed_important >= 4:
        gap(
            CRITICAL,
            "Synthesis",
            "systemic",
            f"{failed_critical} critical + {failed_important} important fields "
            f"missing — likely agent→synthesis field name mismatch",
        )

    # Concordance quality: check for uniformity (all same action = likely broken)
    concordance = synthesis.get("concordance", [])
    if len(concordance) >= 5:
        actions = {e.get("action") for e in concordance}
        if len(actions) == 1:
            gap(
                CRITICAL,
                "Synthesis",
                "concordance_uniform",
                f"All {len(concordance)} stocks have action={actions.pop()} "
                f"— scoring is likely broken",
            )

        tech_signals = {e.get("tech_signal") for e in concordance}
        if tech_signals == {"HOLD"} or tech_signals == {"NEUTRAL"}:
            gap(
                WARNING,
                "Synthesis",
                "tech_uniform",
                "All stocks have same tech signal — tech normalizer may have failed",
            )

        macro_fits = {e.get("macro_fit") for e in concordance}
        if macro_fits == {"NEUTRAL"}:
            gap(
                WARNING,
                "Synthesis",
                "macro_uniform",
                "All stocks have macro_fit=NEUTRAL — macro normalizer may have failed",
            )

        # v33.0: Check for broken signal channels
        news_impacts = {e.get("news_impact") for e in concordance}
        if news_impacts == {"NEUTRAL"} and len(concordance) >= 10:
            gap(
                CRITICAL,
                "Synthesis",
                "news_impact_uniform",
                f"All {len(concordance)} stocks have news_impact=NEUTRAL "
                f"— news impact extraction likely broken",
            )

        risk_flags = {e.get("risk_warning") for e in concordance}
        if risk_flags == {False} and len(concordance) >= 15:
            gap(
                WARNING,
                "Synthesis",
                "risk_warning_uniform",
                f"All {len(concordance)} stocks have risk_warning=False "
                f"— risk warning extraction may have failed",
            )

        census_aligns = {e.get("census") for e in concordance}
        if census_aligns <= {"NEUTRAL", "MISSING", None, ""} and len(concordance) >= 10:
            gap(
                CRITICAL,
                "Synthesis",
                "census_uniform",
                f"All {len(concordance)} stocks have census=NEUTRAL/MISSING "
                f"— census alignment data not flowing to concordance",
            )

        # v35.0: Check entry_timing uniformity (TIMING column in grid)
        entry_timings = {e.get("entry_timing") for e in concordance}
        if entry_timings <= {"HOLD", None, ""} and len(concordance) >= 10:
            gap(
                CRITICAL,
                "Synthesis",
                "entry_timing_uniform",
                f"All {len(concordance)} stocks have entry_timing=HOLD "
                f"— tech timing→timing_signal normalizer likely failed",
            )

        # v35.0: Check tech_momentum all zero (momentum column)
        momentums = [e.get("tech_momentum", 0) for e in concordance]
        if all(m == 0 for m in momentums) and len(concordance) >= 10:
            gap(
                CRITICAL,
                "Synthesis",
                "tech_momentum_zero",
                f"All {len(concordance)} stocks have tech_momentum=0 "
                f"— momentum→momentum_score normalizer likely failed",
            )

    return gaps


def validate_post_html(html: str) -> list[dict[str, str]]:
    """
    Pass 2: Scan generated HTML for rendering gaps.

    Looks for patterns that indicate missing data made it into the HTML.
    """
    gaps = []

    def gap(severity, section, field, msg):
        gaps.append({"severity": severity, "section": section, "field": field, "message": msg})

    # Count N/A cells (excluding legitimate ones like non-equity PE)
    na_cells = len(re.findall(r">N/A</span>", html))
    if na_cells > 20:
        gap(
            WARNING,
            "HTML",
            "N/A cells",
            f"{na_cells} N/A cells found — may indicate missing data flows",
        )

    # Count empty cells
    empty_cells = len(re.findall(r'color:#64748b;">\s*</span></td>', html))
    if empty_cells > 5:
        gap(WARNING, "HTML", "empty cells", f"{empty_cells} empty cells found")

    # Check for "?" trend values
    q_marks = len(re.findall(r">\?</span>", html))
    if q_marks > 3:
        gap(
            WARNING,
            "HTML",
            "question marks",
            f"{q_marks} '?' values found — trend or signal data not mapped",
        )

    # Check for 0% EXRET that might be wrong
    zero_exret = len(re.findall(r">0%</td>", html))
    if zero_exret > 5:
        gap(
            INFO,
            "HTML",
            "zero EXRET",
            f"{zero_exret} cells show 0% — may be missing expected return data",
        )

    # Check key sections exist
    for section in [
        "Executive Summary",
        "Macro &amp; Market Context",
        "News &amp; Events",
        "Technical Analysis",
        "Fundamental Deep Dive",
        "Sentiment &amp; Census",
    ]:
        if section not in html:
            gap(CRITICAL, "HTML", section, f"Section '{section}' not found in HTML")

    # Check section has content (not just header)
    for section_name, min_size in [
        ("News &amp; Events", 500),
        ("Fundamental Deep Dive", 1000),
        ("Technical Analysis", 800),
        ("Macro &amp; Market Context", 400),
        ("Sentiment &amp; Census", 350),
    ]:
        idx = html.find(section_name)
        if idx >= 0:
            # Find the section content between this header and the next section
            next_section = html.find("<h2", idx + len(section_name))
            if next_section < 0:
                next_section = len(html)
            section_html = html[idx:next_section]
            if len(section_html) < min_size:
                gap(
                    WARNING,
                    "HTML",
                    section_name,
                    f"Section '{section_name}' seems too short ({len(section_html)} chars)",
                )

    return gaps


def run_qa(
    synthesis: dict[str, Any],
    fund: dict[str, Any],
    tech: dict[str, Any],
    macro: dict[str, Any],
    census: dict[str, Any],
    news: dict[str, Any],
    opps: dict[str, Any],
    risk: dict[str, Any],
    html: str = "",
    normalize: bool = True,
) -> tuple[bool, list[dict[str, str]]]:
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
            gaps.append(
                {"severity": INFO, "section": "Normalize", "field": "auto-fix", "message": fix_msg}
            )

    # Stage 1a: Pre-HTML validation (agent report completeness)
    gaps.extend(validate_pre_html(synthesis, fund, tech, macro, census, news, opps, risk))

    # Stage 1b: Synthesis output completeness (schema-agnostic)
    gaps.extend(validate_synthesis_completeness(synthesis))

    # Stage 2: Post-HTML validation
    if html:
        gaps.extend(validate_post_html(html))

    criticals = [g for g in gaps if g["severity"] == CRITICAL]
    warnings = [g for g in gaps if g["severity"] == WARNING]
    infos = [g for g in gaps if g["severity"] == INFO]

    logger.info(
        "QA result: %d critical, %d warning, %d info (%d auto-fixed)",
        len(criticals),
        len(warnings),
        len(infos),
        len(fixes) if normalize else 0,
    )

    passed = len(criticals) == 0
    return passed, gaps


def format_qa_report(gaps: list[dict[str, str]]) -> str:
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
    lines.insert(
        0,
        f"QA {status}: {len(auto_fixes)} auto-fixed, {len(criticals)} critical, {len(warnings)} warnings, {len(infos)} info",
    )
    return "\n".join(lines)

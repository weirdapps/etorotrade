"""
Committee HTML Report Generator

CIO v6.0 R3: Moved from /tmp/generate_committee_html.py into the versioned,
tested codebase. Generates the 12-section Investment Committee HTML report
from synthesis.json and individual agent reports.

Usage:
    from trade_modules.committee_html import generate_report_html, generate_report_from_files

    # From pre-loaded dicts:
    html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)

    # From files on disk (original /tmp behavior):
    output_path = generate_report_from_files()
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPORTS_DIR = Path(os.path.expanduser("~/.weirdapps-trading/committee/reports"))
OUTPUT_DIR = Path(os.path.expanduser("~/.weirdapps-trading/committee"))

def load_json(path):
    with open(path) as f:
        return json.load(f)

def sf(v, default=0):
    try:
        return float(str(v).replace('%','').replace(',','').replace('--',''))
    except:
        return default

def action_color(act):
    m = {"SELL":"#dc2626","TRIM":"#d97706","BUY":"#059669","ADD":"#059669","HOLD":"#6366f1"}
    return m.get(act, "#64748b")

def action_bg(act):
    m = {"SELL":"#fef2f2","TRIM":"#fffbeb","BUY":"#ecfdf5","ADD":"#ecfdf5","HOLD":"#ffffff"}
    return m.get(act, "#ffffff")

def action_border(act):
    m = {"SELL":"#fecaca","TRIM":"#fde68a","BUY":"#a7f3d0","ADD":"#a7f3d0","HOLD":"#e2e8f0"}
    return m.get(act, "#e2e8f0")

def sentiment_color(val):
    if val in ("BUY","BULLISH","ENTER_NOW","FAVORABLE","ALIGNED","POSITIVE","OK","STRONG"):
        return "#059669"
    if val in ("SELL","BEARISH","AVOID","EXIT_SOON","UNFAVORABLE","DIVERGENT","NEGATIVE","EXIT","TRIM"):
        return "#dc2626"
    if val in ("WAIT_FOR_PULLBACK","WARN","NEUTRAL_BEARISH","MODERATE_DIVERGENT"):
        return "#d97706"
    return "#64748b"

def conv_color(c):
    if c >= 70: return "#059669"
    if c >= 50: return "#d97706"
    return "#94a3b8"

def conv_bar(c, delta=None):
    col = conv_color(c)
    da = ""
    if delta is not None and delta != 0:
        ar = "&#9650;" if delta > 0 else "&#9660;"
        dc = "#059669" if delta > 0 else "#dc2626"
        da = f' <span style="color:{dc};font-size:9px;">{ar}{abs(delta)}</span>'
    return (f'<div style="font-weight:800;font-size:13px;color:#0f172a;">{c}{da}</div>'
            f'<div style="height:5px;background:#e2e8f0;border-radius:3px;margin-top:3px;'
            f'width:60px;display:inline-block;">'
            f'<div style="height:100%;border-radius:3px;background:{col};'
            f'width:{min(c,100)}%;"></div></div>')

def signal_badge(sig):
    c = {"B":"#059669","H":"#d97706","S":"#dc2626","I":"#94a3b8"}
    return f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{c.get(sig,"#94a3b8")};">{sig}</span>'

def dot(color):
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"></span>'

def agent_badge(name, view, cbg, ctxt, cbrd):
    return f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:{cbg};color:{ctxt};border:1px solid {cbrd};">{name}: {view}</span>'

def e(text):
    return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def abbr(text):
    """Designed abbreviations for agent views — readable, not truncated."""
    MAP = {
        "ENTER_NOW": "ENTER", "EXIT_SOON": "EXIT", "WAIT_FOR_PULLBACK": "WAIT",
        "FAVORABLE": "FAVOR", "UNFAVORABLE": "UNFAV",
        "ALIGNED": "ALIGN", "DIVERGENT": "DIVERG", "MODERATE_DIVERGENT": "M.DIV",
        "HIGH_NEGATIVE": "H.NEG", "HIGH_POSITIVE": "H.POS",
        "NEUTRAL": "NEUT", "POSITIVE": "POS", "NEGATIVE": "NEG",
    }
    return MAP.get(str(text), str(text))

# Map fine-grained agent sectors to GICS standard sectors
_GICS_MAP = {
    "Technology": "Technology", "Software": "Technology", "Semiconductors": "Technology",
    "Cybersecurity": "Technology", "Networking": "Technology", "Consumer Electronics": "Technology",
    "Financials": "Financials", "Banking": "Financials", "Insurance": "Financials",
    "Financial Services": "Financials", "Fintech": "Financials",
    "Healthcare": "Healthcare", "Pharmaceuticals": "Healthcare",
    "Consumer Discretionary": "Consumer Disc.", "Consumer Cyclical": "Consumer Disc.",
    "Automotive": "Consumer Disc.", "E-commerce": "Consumer Disc.",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy", "Energy Services": "Energy",
    "Industrials": "Industrials", "Defense": "Industrials",
    "Materials": "Materials", "Commodities": "Materials",
    "Communication Services": "Comm. Services", "Social Media": "Comm. Services",
    "Telecommunications": "Comm. Services",
    "Real Estate": "Real Estate", "Utilities": "Utilities",
    "Cryptocurrency": "Crypto/Alt", "Bitcoin Proxy": "Crypto/Alt",
}

def gics_sector(raw):
    """Map fine-grained sector to GICS standard."""
    return _GICS_MAP.get(raw, raw if raw else "Other")

def conv_color_action(conv, action):
    """Conviction color that respects action direction.
    For SELL/TRIM: higher conviction = more urgently bearish (red/amber).
    For BUY/ADD: higher conviction = more bullish (green).
    """
    if action in ("SELL", "TRIM"):
        if conv >= 70: return "#dc2626"  # high confidence to exit = red
        if conv >= 50: return "#d97706"  # moderate trim confidence = amber
        return "#94a3b8"
    return conv_color(conv)

# Normalize legacy action names to current 5-action system
_ACTION_MIGRATION = {
    "IMMEDIATE SELL": "SELL", "REDUCE": "TRIM",
    "WEAK HOLD": "HOLD", "STRONG HOLD": "HOLD",
    "BUY NEW": "BUY", "WATCH": "HOLD",
}

def normalize_action(act):
    return _ACTION_MIGRATION.get(act, act)

def _stock_kill_thesis(act, tkr, sec, rsi, exret, beta, buy_pct, macro_fit, tech_sig, fund_view):
    """Generate a stock-specific kill thesis using actual data points."""
    if act == "SELL":
        parts = [f"Wrong if {tkr} reverses"]
        if rsi < 35:
            parts[0] += f" — deeply oversold at RSI {rsi:.0f}"
        if exret > 20:
            parts.append(f"Abandoning {exret:.0f}% EXRET with {buy_pct:.0f}% analyst BUY consensus")
        if beta > 1.5:
            parts.append(f"but beta {beta:.1f} amplifies any adverse move")
        return ". ".join(parts) + "."
    if act == "TRIM":
        if rsi < 35:
            return (f"Trimming {tkr} at RSI {rsi:.0f} (oversold) risks selling near the bottom. "
                    f"EXRET {exret:.0f}%, {buy_pct:.0f}% BUY consensus — strong reversal case.")
        parts = []
        if "Crypto" in sec:
            parts.append(f"Wrong if crypto rally resumes — {tkr} has high beta to digital assets")
        elif tech_sig == "EXIT_SOON" and rsi > 60:
            parts.append(f"Wrong if RSI {rsi:.0f} consolidates rather than reverses")
        elif macro_fit == "UNFAVORABLE":
            parts.append(f"Wrong if {sec} macro headwinds ease — {fund_view} fundamental view with {exret:.0f}% EXRET argues to hold")
        else:
            parts.append(f"Wrong if {sec} rotation reverses or {buy_pct:.0f}% analyst consensus proves right")
        if exret > 30:
            parts.append(f"Exceptional EXRET {exret:.0f}% means large upside if trim is premature")
        return ". ".join(parts) + "." if parts else f"Monitor {tkr} for reversal."
    if act == "BUY":
        parts = []
        if sec == "Materials" and exret == 0:
            parts.append(f"Fails if dollar strengthens or inflation expectations reverse")
        elif beta > 1.5:
            parts.append(f"Fails if broad selloff deepens — beta {beta:.1f} amplifies drawdown")
        elif macro_fit == "UNFAVORABLE":
            parts.append(f"Buying into macro headwind for {sec}; fails if regime worsens")
        else:
            parts.append(f"Fails if {exret:.0f}% EXRET proves stale — watch for estimate cuts")
        if rsi > 60:
            parts.append(f"RSI {rsi:.0f} not ideal entry; pullback to ~45 would improve risk/reward")
        return ". ".join(parts) + "."
    if act == "ADD":
        parts = []
        if rsi < 35:
            parts.append(f"Adding at RSI {rsi:.0f} (oversold) but could go lower if selling accelerates")
        elif tech_sig == "AVOID":
            parts.append(f"Adding against technical AVOID signal — fails if downtrend continues")
        else:
            parts.append(f"Wrong if {exret:.0f}% EXRET target is cut or if earnings disappoint")
        if buy_pct > 90:
            parts.append(f"Consensus crowded at {buy_pct:.0f}% BUY — any miss gets punished hard")
        return ". ".join(parts) + "."
    return f"Monitor {tkr} for catalysts."


def generate_report_html(
    synth: Dict[str, Any],
    fund: Dict[str, Any],
    tech: Dict[str, Any],
    macro: Dict[str, Any],
    census: Dict[str, Any],
    news: Dict[str, Any],
    opps: Dict[str, Any],
    risk: Dict[str, Any],
    date_str: Optional[str] = None,
) -> str:
    """
    Generate the Investment Committee HTML report from data dicts.

    Args:
        synth: Synthesis output (concordance, regime, indicators, etc.)
        fund: Fundamental agent report
        tech: Technical agent report
        macro: Macro agent report (unused in HTML, included for completeness)
        census: Census agent report
        news: News agent report
        opps: Opportunity scanner report
        risk: Risk manager report
        date_str: Override date string (default: today)

    Returns:
        Complete HTML string for the report.
    """
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    try:
        today_long = datetime.strptime(today, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        today_long = today

    concordance = synth.get("concordance", [])
    regime = synth.get("regime", "CAUTIOUS")
    macro_score = synth.get("macro_score", 0)
    rotation = synth.get("rotation_phase", "UNKNOWN")
    risk_score = synth.get("risk_score", 50)
    # Use nested portfolio_risk for properly scaled VaR (decimal, not pct*100)
    pr_synth = synth.get("portfolio_risk", {})
    var_95_raw = pr_synth.get("var_95", synth.get("var_95", 0))
    # If VaR magnitude > 10, it was likely multiplied by 100 incorrectly
    var_95 = var_95_raw if abs(var_95_raw) <= 10 else var_95_raw / 100
    max_dd = pr_synth.get("max_drawdown", synth.get("max_drawdown", 0))
    p_beta = pr_synth.get("portfolio_beta", synth.get("portfolio_beta", 1.0))
    fg_top100 = synth.get("fg_top100", 50)
    fg_broad = synth.get("fg_broad", 50)
    changes = synth.get("changes", [])
    sector_gaps = synth.get("sector_gaps", [])
    top_opps = synth.get("top_opportunities", [])
    clusters = synth.get("correlation_clusters", [])
    concentration = synth.get("concentration", {})
    stress = synth.get("stress_scenarios", {})
    indicators = synth.get("indicators", {})
    sector_rankings = synth.get("sector_rankings", {})

    # Enhanced lookups
    delta_map = {c.get("ticker"): c.get("delta", 0) for c in changes}
    _actionable = {"SELL", "TRIM", "BUY", "ADD"}
    actionable_tickers = [entry.get("ticker") for entry in concordance
                          if entry.get("action") in _actionable]
    signal_date = synth.get("signal_date", today)
    census_date = synth.get("census_date", today)
    sector_counts = {}
    for entry in concordance:
        sec = gics_sector(entry.get("sector", ""))
        if sec:
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    action_order = {"SELL":0,"TRIM":1,"BUY":2,"ADD":3,"HOLD":4}
    concordance.sort(key=lambda x: (action_order.get(x.get("action","HOLD"),4), -x.get("conviction",0)))

    sells = sum(1 for c in concordance if c.get("action") == "SELL")
    buys = sum(1 for c in concordance if c.get("action") in ("BUY","ADD"))
    trims = sum(1 for c in concordance if c.get("action") == "TRIM")

    if regime == "RISK_ON":
        verdict = "RISK-ON"; vbg,vbrd,vval,vacc = "#ecfdf5","#a7f3d0","#059669","#059669"
    elif regime == "RISK_OFF":
        verdict = "DEFENSIVE"; vbg,vbrd,vval,vacc = "#fef2f2","#fecaca","#dc2626","#dc2626"
    else:
        verdict = "CAUTIOUS"; vbg,vbrd,vval,vacc = "#fffbeb","#fde68a","#d97706","#d97706"

    risk_color = "#dc2626" if risk_score >= 70 else "#d97706" if risk_score >= 40 else "#059669"

    sell_tickers = [c["ticker"] for c in concordance if c.get("action") == "SELL"]
    buy_tickers = [c["ticker"] for c in concordance if c.get("action") in ("BUY","ADD") and c.get("conviction",0)>=70][:3]
    new_tickers = [c["ticker"] for c in concordance if c.get("action") == "BUY"]

    # CIO-quality narrative synthesis
    vix_val = indicators.get('vix', 0)
    y10 = indicators.get('yield_10y', 0)
    # Count leading/lagging sectors
    leading = [etf for etf, d in sector_rankings.items() if d.get("return_1m", 0) > 2]
    lagging = [etf for etf, d in sector_rankings.items() if d.get("return_1m", 0) < -2]
    parts = []
    if regime == "RISK_OFF":
        parts.append(f"We are in a {rotation.replace('_', ' ').lower()} environment with deteriorating breadth")
        if lagging:
            parts[-1] += f" — {len(lagging)} of {len(sector_rankings)} sectors negative"
        parts[-1] += ". Defensive positioning warranted."
    elif regime == "RISK_ON":
        parts.append(f"Constructive macro backdrop. {len(leading)} sectors leading with broad participation.")
    else:
        parts.append(f"{rotation.replace('_', ' ')} phase with mixed signals (macro score {macro_score}).")
        if vix_val > 20:
            parts[-1] = parts[-1][:-1] + f", VIX elevated at {vix_val:.0f}."
    if sells:
        parts.append(f"Highest-priority action: exit {', '.join(sell_tickers)} ({sells} SELL signal{'s' if sells > 1 else ''}).")
    if trims:
        parts.append(f"Trim {trims} positions where technical momentum has broken down or macro headwinds apply.")
    top_adds = [c["ticker"] for c in concordance if c.get("action") in ("BUY", "ADD") and c.get("conviction", 0) >= 70][:3]
    if top_adds:
        parts.append(f"Highest-conviction additions: {', '.join(top_adds)}.")
    elif new_tickers:
        parts.append(f"New opportunities: {', '.join(new_tickers[:4])}.")
    narrative = " ".join(parts)

    # Disagreements — capture multiple types of agent conflict
    disagreements = []
    for entry in concordance:
        tkr = entry.get("ticker", "")
        fv = entry.get("fund_view", "")
        ts = entry.get("tech_signal", "")
        mf = entry.get("macro_fit", "")
        rw = entry.get("risk_warning", False)
        ce = entry.get("census", "")
        conv = entry.get("conviction", 0)
        act = entry.get("action", "")
        fs = entry.get("fund_score", 0)
        rsi = entry.get("rsi", 0)
        ex = entry.get("exret", 0)
        mom = entry.get("tech_momentum", 0)
        sec = gics_sector(entry.get("sector", ""))

        # Type 1: Fundamental vs Technical
        if (fv == "BUY" and ts in ("AVOID", "EXIT_SOON")) or (fv == "SELL" and ts == "ENTER_NOW"):
            dis_narr = (f"Fundamental analyst sees {tkr} at {fs:.0f}/100, "
                        f"EXRET {ex:.1f}%, signal {fv}. "
                        f"Technical strategist warns: RSI {rsi:.0f}, "
                        f"momentum {mom:+d}, timing signal {ts}.")
            if mf == "UNFAVORABLE":
                dis_narr += f" Macro fit UNFAVORABLE for {sec}."
            disagreements.append({"ticker": tkr, "headline": f"Fund {fv} ({fs:.0f}) vs Tech {ts} (RSI {rsi:.0f})", "fund_view": fv, "tech_signal": ts, "narrative": dis_narr, "resolution": act, "conviction": conv, "accent": action_color(act)})
        # Type 2: Strong fundamentals vs Risk Manager warning
        elif fv == "BUY" and fs >= 75 and rw and act in ("TRIM", "HOLD"):
            dis_narr = (f"Fundamental analyst rates {tkr} strongly ({fs:.0f}/100, "
                        f"EXRET {ex:.1f}%), but Risk Manager flags concerns. "
                        f"Technical: RSI {rsi:.0f}, signal {ts}. "
                        f"Macro: {mf}. Result: {act} despite strong fundamentals.")
            disagreements.append({"ticker": tkr, "headline": f"Fund BUY ({fs:.0f}) vs Risk WARN", "fund_view": fv, "tech_signal": ts, "narrative": dis_narr, "resolution": act, "conviction": conv, "accent": action_color(act)})
        # Type 3: Macro headwind on BUY-signal stock
        elif fv == "BUY" and mf == "UNFAVORABLE" and act in ("ADD", "HOLD"):
            dis_narr = (f"{tkr} has strong signal ({fv}, EXRET {ex:.1f}%) but "
                        f"macro fit is UNFAVORABLE for {sec}. "
                        f"Technical: RSI {rsi:.0f}, {ts}. "
                        f"Committee proceeded with {act} at reduced conviction ({conv}).")
            disagreements.append({"ticker": tkr, "headline": f"Signal BUY vs Macro UNFAVORABLE ({sec})", "fund_view": fv, "tech_signal": ts, "narrative": dis_narr, "resolution": act, "conviction": conv, "accent": "#d97706"})
        if len(disagreements) >= 6:
            break
    # Fallback: split votes
    if len(disagreements) < 3:
        for entry in concordance:
            bp = entry.get("bull_pct", 50)
            if 35 < bp < 65 and entry.get("action") not in ("HOLD",):
                tkr = entry.get("ticker", "")
                act = entry.get("action", "")
                disagreements.append({"ticker": tkr, "headline": f"Split: {entry.get('bull_weight',0):.1f} bull vs {entry.get('bear_weight',0):.1f} bear", "fund_view": entry.get("fund_view", ""), "tech_signal": entry.get("tech_signal", ""), "narrative": f"Committee split on {tkr}. Fund: {entry.get('fund_view','')} ({entry.get('fund_score',0):.0f}), Tech: {entry.get('tech_signal','')} (RSI {entry.get('rsi',0):.0f}), Macro: {entry.get('macro_fit','')}. Bull weight {entry.get('bull_weight',0):.1f} vs bear {entry.get('bear_weight',0):.1f}.", "resolution": act, "conviction": entry.get("conviction", 0), "accent": action_color(act)})
                if len(disagreements) >= 6:
                    break

    h = []
    h.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>')
    h.append('<body style="margin:0;padding:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,\'Helvetica Neue\',Arial,sans-serif;color:#334155;line-height:1.6;-webkit-font-smoothing:antialiased;">')
    h.append('<div style="max-width:960px;margin:0 auto;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.1);">')

    # HEADER
    h.append(f'<div style="background-color:#0f172a;padding:36px 40px 32px 40px;"><table style="width:100%;"><tr><td style="vertical-align:middle;width:50px;"><div style="width:36px;height:36px;border-radius:8px;background:rgba(255,255,255,0.15);text-align:center;line-height:36px;font-size:18px;color:#fff;font-weight:800;letter-spacing:-1px;">IC</div></td><td style="vertical-align:middle;"><h1 style="margin:0;font-size:26px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Investment Committee</h1><p style="margin:2px 0 0 0;font-size:13px;color:#94a3b8;">{today_long} &middot; 7 Specialist Agents + CIO Synthesis v7.0</p></td></tr></table><div style="margin-top:16px;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Signals: {e(signal_date)}</span> <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Census: {e(census_date)}</span> <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Macro: {rotation.replace("_"," ")}</span></div></div>')

    # S1: EXECUTIVE SUMMARY
    h.append(f'<div style="padding:32px 40px;border-bottom:2px solid #cbd5e1;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Executive Summary</h2>')
    h.append(f'<table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;"><tr>')
    h.append(f'<td style="width:33%;background:{vbg};border:1px solid {vbrd};border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{vval};margin-bottom:6px;">CIO Verdict</div><div style="font-size:28px;font-weight:800;color:{vval};">{verdict}</div><div style="font-size:11px;color:#64748b;margin-top:4px;">{sells} sell, {trims} trim, {buys} buy/add</div></td>')
    h.append(f'<td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Macro Regime</div><div style="font-size:28px;font-weight:800;color:{vval};">{regime}</div><div style="font-size:11px;color:#64748b;margin-top:4px;">Score: {macro_score} &middot; {rotation.replace("_"," ")}</div></td>')
    h.append(f'<td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Portfolio Risk</div><div style="font-size:28px;font-weight:800;color:{risk_color};">{risk_score}/100</div><div style="font-size:11px;color:#64748b;margin-top:4px;">Beta {p_beta:.2f} &middot; VaR {var_95:.1f}%</div></td>')
    h.append('</tr></table>')
    h.append(f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid {vacc};border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:20px;"><div style="font-size:13px;color:#334155;line-height:1.6;">{e(narrative)}</div></div>')
    # CIO v7.0 P3: Risk dilution banner
    if synth.get("risk_diluted"):
        warned = sum(1 for c in concordance if c.get("risk_warning"))
        h.append(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:12px 20px;margin-bottom:16px;"><span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#92400e;">Risk Warning Dilution</span><span style="font-size:12px;color:#78350f;margin-left:8px;">{warned}/{len(concordance)} stocks flagged ({warned*100//max(len(concordance),1)}%) — treat as systemic market risk, not stock-specific. Individual risk warnings are discounted.</span></div>')

    actionable_entries = [entry for entry in concordance if entry.get("action") in _actionable]
    if actionable_entries:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:8px;">Priority Actions</div><div>')
        for entry in actionable_entries[:12]:
            act = entry.get("action", "HOLD")
            tkr = entry.get("ticker", "")
            conv = entry.get("conviction", 0)
            ac = action_color(act)
            h.append(f'<span style="display:inline-block;padding:5px 12px;margin:3px 4px 3px 0;border-radius:6px;font-size:11px;font-weight:700;color:#fff;background:{ac};letter-spacing:0.2px;">{act} {e(tkr)} ({conv})</span>')
        h.append('</div>')
    h.append('</div>')

    # S2: MACRO ENVIRONMENT
    dxy_raw = sf(indicators.get('dxy', 0))
    dxy_str = f"{dxy_raw:.1f}" if 80 <= dxy_raw <= 120 else "N/A"
    macro_ind = [("10Y Yield",f"{indicators.get('yield_10y',0):.2f}%",indicators.get('yield_curve_status','NORMAL')),("Yield Curve",f"{indicators.get('yield_curve_spread',0):.0f}bps","POSITIVE" if sf(indicators.get('yield_curve_spread',0))>0 else "INVERTED"),("VIX",f"{indicators.get('vix',0):.1f}","ELEVATED" if sf(indicators.get('vix',0))>20 else "NORMAL"),("Dollar (DXY)",dxy_str,indicators.get('dollar_trend','STABLE') if dxy_str != "N/A" else "DATA ERROR"),("Macro Score",f"{macro_score}",regime)]
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Macro Environment</h2><table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:10px 12px;text-align:left;font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Indicator</th><th style="padding:10px 12px;text-align:center;font-size:11px;font-weight:700;color:#64748b;">Value</th><th style="padding:10px 12px;text-align:center;font-size:11px;font-weight:700;color:#64748b;">Signal</th><th style="padding:10px 12px;text-align:left;font-size:11px;font-weight:700;color:#64748b;">Status</th></tr>')
    for i,(n,v,st) in enumerate(macro_ind):
        bg = "#f8fafc" if i%2==1 else "#fff"
        dc = "#dc2626" if any(x in st for x in ("INVERTED","ELEVATED","RISK_OFF")) else "#059669" if any(x in st for x in ("POSITIVE","NORMAL","RISK_ON")) else "#d97706"
        h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:10px 12px;font-weight:600;color:#1e293b;">{e(n)}</td><td style="padding:10px 12px;text-align:center;font-weight:700;font-family:monospace;">{e(v)}</td><td style="padding:10px 12px;text-align:center;">{dot(dc)}</td><td style="padding:10px 12px;color:#334155;">{e(st)}</td></tr>')
    h.append('</table>')
    # Sector rotation bar
    h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Sector Rotation (1M)</div><table style="width:100%;border-collapse:separate;border-spacing:4px 0;font-size:11px;"><tr>')
    for etf,data in sorted(sector_rankings.items(), key=lambda x:x[1].get("return_1m",0), reverse=True):
        ret = data.get("return_1m",0)
        sbg = "#ecfdf5" if ret>2 else "#fef2f2" if ret<-2 else "#f8fafc"
        sbrd = "#a7f3d0" if ret>2 else "#fecaca" if ret<-2 else "#e2e8f0"
        stxt = "#059669" if ret>2 else "#dc2626" if ret<-2 else "#334155"
        h.append(f'<td style="padding:6px 4px;text-align:center;background:{sbg};border:1px solid {sbrd};border-radius:4px;"><div style="font-weight:700;color:#334155;font-size:10px;">{e(etf)}</div><div style="font-size:12px;font-weight:800;color:{stxt};">{ret:+.1f}%</div></td>')
    h.append('</tr></table></div>')

    # S3: STOCK ANALYSIS GRID — enhanced with EXRET, SIZE, abbreviations, conviction delta
    hdr = lambda bg, col, txt: f'<th style="padding:8px 4px;text-align:center;background:{bg};color:{col};font-size:9px;font-weight:700;border:1px solid #1e293b;">{txt}</th>'
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Stock Analysis Grid</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Concordance matrix: 7 specialist views per stock.</p><div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap;"><tr>')
    h.append(hdr("#0f172a","#fff","STOCK") + hdr("#0f172a","#fff","SIG"))
    h.append(hdr("#1e293b","#93c5fd","FUND") + hdr("#1e293b","#c4b5fd","TECH") + hdr("#1e293b","#fcd34d","MACRO") + hdr("#1e293b","#6ee7b7","CENS") + hdr("#1e293b","#fca5a5","NEWS") + hdr("#1e293b","#d6d3d1","RISK"))
    h.append(hdr("#0f172a","#fbbf24","EXRET") + hdr("#0f172a","#fff","VERDICT") + hdr("#0f172a","#fff","CONV") + hdr("#0f172a","#fff","SIZE"))
    h.append('</tr>')
    cg = None
    for entry in concordance:
        act = entry.get("action","HOLD")
        if act in ("SELL","TRIM") and cg!="SELL":
            cg="SELL"; h.append('<tr><td colspan="12" style="padding:4px 8px;background:#fef2f2;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#991b1b;border:1px solid #fecaca;">SELL / TRIM</td></tr>')
        elif act in ("BUY","ADD") and cg!="BUY":
            cg="BUY"; h.append('<tr><td colspan="12" style="padding:4px 8px;background:#ecfdf5;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#065f46;border:1px solid #a7f3d0;">BUY / ADD</td></tr>')
        elif act == "HOLD" and cg!="HOLD":
            cg="HOLD"; h.append('<tr><td colspan="12" style="padding:4px 8px;background:#f1f5f9;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#475569;border:1px solid #cbd5e1;">HOLD</td></tr>')
        tkr=entry.get("ticker",""); sig=entry.get("signal","?"); fs=entry.get("fund_score",0); fv=entry.get("fund_view","?"); ts=entry.get("tech_signal","?"); rsi=entry.get("rsi",0); mf=entry.get("macro_fit","?"); ce=entry.get("census","?"); ni=entry.get("news_impact","?"); rw="WARN" if entry.get("risk_warning") else "OK"; conv=entry.get("conviction",0); sm="*" if entry.get("fund_synthetic") else ""; rb=action_bg(act)
        entry["sector"] = gics_sector(entry.get("sector", ""))  # normalize for downstream
        ex = entry.get("exret", 0); mp = entry.get("max_pct", 0)
        delta = delta_map.get(tkr)
        exc = "#059669" if ex > 5 else "#dc2626" if ex < 0 else "#334155"
        sz_str = f"{mp:.0f}%" if mp else "-"
        td = 'style="padding:6px 4px;text-align:center;border:1px solid #e2e8f0;'
        h.append(f'<tr style="background:{rb};"><td {td}font-family:monospace;font-weight:700;font-size:11px;color:#0f172a;text-align:left;padding-left:8px;">{e(tkr)}</td><td {td}">{signal_badge(sig)}</td><td {td}color:{sentiment_color(fv)};font-weight:600;font-size:10px;">{abbr(fv)}({fs:.0f}){sm}</td><td {td}color:{sentiment_color(ts)};font-weight:600;font-size:10px;">{abbr(ts)}({rsi:.0f})</td><td {td}color:{sentiment_color(mf)};font-weight:600;font-size:10px;">{abbr(mf)}</td><td {td}color:{sentiment_color(ce)};font-weight:600;font-size:10px;">{abbr(ce)}</td><td {td}color:{sentiment_color(ni)};font-weight:600;font-size:10px;">{abbr(ni)}</td><td {td}color:{sentiment_color(rw)};font-weight:600;font-size:10px;">{rw}</td><td {td}font-family:monospace;font-weight:700;color:{exc};">{ex:.1f}%</td><td {td}"><span style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:700;color:#fff;background:{action_color(act)};">{act}</span></td><td {td}">{conv_bar(conv, delta)}</td><td {td}font-family:monospace;font-size:10px;font-weight:600;color:#64748b;">{sz_str}</td></tr>')
    h.append('</table></div><div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;font-size:10px;color:#64748b;">* = Synthetic score &middot; ENTER = Enter now &middot; EXIT = Exit soon &middot; WAIT = Wait for pullback &middot; FAVOR/UNFAV = Macro fit &middot; ALIGN = Census aligned &middot; &#9650;&#9660; = Conviction change</div></div>')

    # S4: DISAGREEMENTS
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Where We Disagreed</h2><p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">Structured disagreement is the committee\'s edge.</p>')
    for d in disagreements[:6]:
        fb = agent_badge("Fund",d["fund_view"],"#dbeafe","#1e40af","#93c5fd")
        tb = agent_badge("Tech",d["tech_signal"],"#ede9fe","#5b21b6","#c4b5fd")
        h.append(f'<div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid {d["accent"]};border-radius:0 8px 8px 0;padding:20px 24px;margin-bottom:16px;"><div style="margin-bottom:10px;"><span style="font-family:monospace;font-weight:800;font-size:14px;color:#0f172a;">{e(d["ticker"])}</span> <span style="font-size:13px;font-weight:600;color:#334155;margin-left:8px;">{e(d["headline"])}</span></div><div style="margin-bottom:12px;">{fb}{tb}</div><div style="font-size:13px;color:#334155;line-height:1.6;margin-bottom:12px;">{e(d["narrative"])}</div><div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;"><span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Resolution:</span> <span style="font-size:13px;font-weight:700;color:{action_color(d["resolution"])};">{d["resolution"]} (conv. {d["conviction"]})</span></div></div>')
    if not disagreements:
        h.append('<div style="padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;color:#64748b;font-size:13px;">No significant disagreements. All agents broadly aligned.</div>')
    h.append('</div>')

    # S5: FUNDAMENTAL DEEP DIVE — actionable stocks only
    fund_stocks = fund.get("stocks",{})
    actionable_fund = [(tkr, fund_stocks[tkr]) for tkr in actionable_tickers if tkr in fund_stocks]
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Fundamental Deep Dive</h2>')
    if actionable_fund:
        h.append(f'<p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Detail for {len(actionable_fund)} positions with pending actions.</p>')
        h.append('<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Score</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">PE Traj</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">EXRET</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Insider</th><th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Notes</th></tr>')
        for i,(tkr,data) in enumerate(actionable_fund):
            bg="#f8fafc" if i%2==1 else "#fff"; sc=sf(data.get("fundamental_score",0)); pt=str(data.get("pe_trajectory","?"))[:20]; ex=sf(data.get("exret",0)); ins=data.get("insider_sentiment","NEUTRAL"); notes=str(data.get("notes","") or "")[:60]
            ic = "#059669" if ins=="NET_BUYING" else "#dc2626" if ins=="NET_SELLING" else "#64748b"
            ex_str = f"{ex:.1f}%" if ex else "N/A"
            h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:8px 10px;font-family:monospace;font-weight:700;">{e(tkr)}</td><td style="padding:8px 10px;text-align:center;font-weight:800;color:{conv_color(int(sc))};">{sc:.0f}</td><td style="padding:8px 10px;text-align:center;font-size:11px;">{e(pt)}</td><td style="padding:8px 10px;text-align:center;">{ex_str}</td><td style="padding:8px 10px;text-align:center;color:{ic};font-weight:600;font-size:10px;">{e(ins)}</td><td style="padding:8px 10px;color:#334155;font-size:11px;">{e(notes)}</td></tr>')
        h.append('</table>')
    else:
        h.append('<p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">All positions on hold &mdash; no fundamental flags.</p>')
    qt = fund.get("quality_traps",[])
    if qt:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin:16px 0 10px 0;">Quality Traps</div>')
        for qt_item in qt:
            tkr = qt_item.get('ticker','?') if isinstance(qt_item,dict) else qt_item; d=fund_stocks.get(tkr,{}); h.append(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:12px;"><span style="font-family:monospace;font-weight:700;color:#991b1b;">{e(tkr)}</span> &mdash; {e(d.get("notes","Quality trap"))}</div>')
    h.append('</div>')

    # S6: TECHNICAL
    tech_stocks = tech.get("stocks",{})
    bearish_macd = sum(1 for d in tech_stocks.values() if d.get("macd_signal")=="BEARISH")
    total_tech = len(tech_stocks)
    avg_rsi = sum(d.get("rsi",50) for d in tech_stocks.values())/max(total_tech,1)
    below50 = sum(1 for d in tech_stocks.values() if not d.get("above_sma50",True))
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Technical Analysis</h2>')
    if bearish_macd > total_tech*0.5:
        h.append(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:0 6px 6px 0;padding:14px 20px;margin-bottom:20px;"><div style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#d97706;">MARKET-WIDE CONTEXT</div><div style="font-size:13px;color:#334155;margin-top:6px;">{bearish_macd}/{total_tech} stocks BEARISH MACD. {below50} below SMA50. Avg RSI: {avg_rsi:.0f}.</div></div>')
    actionable_tech = [(tkr, tech_stocks[tkr]) for tkr in actionable_tickers if tkr in tech_stocks]
    if actionable_tech:
        h.append(f'<p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Detail for {len(actionable_tech)} positions with pending actions.</p>')
        h.append('<table style="width:100%;border-collapse:collapse;font-size:11px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px 6px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">RSI</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">MACD</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">BB%</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Trend</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Mom</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Signal</th></tr>')
        for i,(tkr,d) in enumerate(actionable_tech):
            bg="#f8fafc" if i%2==1 else "#fff"; rsi=d.get("rsi",50); macd=d.get("macd_signal","?"); bb=d.get("bb_position",0.5); trend=d.get("trend","?"); mom=d.get("momentum_score",0); sig=d.get("timing_signal","?")
            rc = "#dc2626" if rsi>70 else "#059669" if rsi<30 else "#334155"
            h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:7px 8px;font-family:monospace;font-weight:700;">{e(tkr)}</td><td style="padding:7px 6px;text-align:center;font-weight:700;color:{rc};">{rsi:.0f}</td><td style="padding:7px 6px;text-align:center;color:{sentiment_color(macd)};font-weight:600;">{macd[:4]}</td><td style="padding:7px 6px;text-align:center;">{bb:.2f}</td><td style="padding:7px 6px;text-align:center;font-size:10px;">{e(trend[:12])}</td><td style="padding:7px 6px;text-align:center;font-weight:700;color:{"#059669" if mom>0 else "#dc2626" if mom<0 else "#64748b"};">{mom:+d}</td><td style="padding:7px 6px;text-align:center;"><span style="padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{sentiment_color(sig)};">{e(sig[:8])}</span></td></tr>')
        h.append('</table>')
    else:
        h.append('<p style="color:#64748b;font-size:12px;margin:0;">All positions on hold &mdash; no technical flags.</p>')
    oversold = [t for t,d in tech_stocks.items() if d.get("rsi",50)<30]
    overbought = [t for t,d in tech_stocks.items() if d.get("rsi",50)>70]
    if oversold: h.append(f'<div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:6px;padding:10px 14px;margin-top:12px;font-size:12px;"><span style="font-weight:700;color:#065f46;">Oversold (RSI&lt;30):</span> {", ".join(oversold)}</div>')
    if overbought: h.append(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px 14px;margin-top:8px;font-size:12px;"><span style="font-weight:700;color:#991b1b;">Overbought (RSI&gt;70):</span> {", ".join(overbought)}</div>')
    h.append('</div>')

    # S7: CENSUS
    fg_label = lambda v: "EXTREME GREED" if v>=75 else "GREED" if v>=55 else "NEUTRAL" if v>=45 else "FEAR" if v>=25 else "EXTREME FEAR"
    fg_color = lambda v: "#dc2626" if v>=75 else "#d97706" if v>=55 else "#64748b" if v>=45 else "#2563eb" if v>=25 else "#6366f1"
    cash100 = census.get("sentiment",{}).get("cash_top100",0)
    h.append(f'<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Census &amp; Sentiment</h2><table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;"><tr><td style="width:25%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;">F&amp;G Top100</div><div style="font-size:24px;font-weight:800;color:{fg_color(fg_top100)};">{fg_top100}</div><div style="font-size:11px;color:#64748b;">{fg_label(fg_top100)}</div></td><td style="width:25%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;">F&amp;G Broad</div><div style="font-size:24px;font-weight:800;color:{fg_color(fg_broad)};">{fg_broad}</div><div style="font-size:11px;color:#64748b;">{fg_label(fg_broad)}</div></td><td style="width:25%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;">Cash Top100</div><div style="font-size:24px;font-weight:800;color:#334155;">{cash100:.1f}%</div><div style="font-size:11px;color:#64748b;">{"Defensive" if cash100>15 else "Deploying" if cash100<8 else "Normal"}</div></td><td style="width:25%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;">Stocks</div><div style="font-size:24px;font-weight:800;color:#334155;">{len(concordance)}</div><div style="font-size:11px;color:#64748b;">{len([c for c in concordance if not c.get("is_opportunity")])} port + {len([c for c in concordance if c.get("is_opportunity")])} new</div></td></tr></table>')
    # Missing popular
    missing = synth.get("missing_popular",[])
    if missing:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#d97706;margin-bottom:8px;">Popular NOT in portfolio</div><div style="margin-bottom:16px;">')
        for m in missing[:10]:
            tkr = m if isinstance(m,str) else m.get("symbol", m.get("ticker","?"))
            h.append(f'<span style="display:inline-block;padding:4px 10px;margin:2px 4px;border-radius:12px;font-size:11px;font-weight:600;background:#fffbeb;color:#92400e;border:1px solid #fde68a;">{e(tkr)}</span>')
        h.append('</div>')
    h.append('</div>')

    # S8: NEWS
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">News &amp; Events</h2>')
    for item in news.get("breaking_news",[])[:5]:
        hl=item.get("headline",""); imp=item.get("impact",item.get("score","NEUTRAL"))
        if "NEGATIVE" in str(imp): nbg,nbrd,nlc = "#fef2f2","#dc2626","#991b1b"
        elif "POSITIVE" in str(imp): nbg,nbrd,nlc = "#ecfdf5","#059669","#065f46"
        else: nbg,nbrd,nlc = "#f8fafc","#94a3b8","#334155"
        h.append(f'<div style="background:{nbg};border-left:4px solid {nbrd};border-radius:0 6px 6px 0;padding:12px 16px;margin-bottom:10px;"><div style="font-size:13px;font-weight:700;color:{nlc};">{e(hl)}</div><div style="font-size:11px;color:#64748b;margin-top:4px;">{e(imp)}</div></div>')
    pn = news.get("portfolio_news",{})
    notable = [(t,items[0]) for t,items in pn.items() if items and any(x in str(items[0].get("impact",items[0].get("score",""))) for x in ("POSITIVE","NEGATIVE"))]
    if notable:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:16px 0 10px 0;">Portfolio News</div>')
        for tkr,item in notable[:6]:
            imp=item.get("impact",item.get("score","NEUTRAL"))
            ic="#dc2626" if "NEGATIVE" in str(imp) else "#059669"
            h.append(f'<div style="font-size:12px;margin-bottom:6px;"><span style="font-family:monospace;font-weight:700;">{e(tkr)}</span> <span style="color:{ic};font-weight:600;">[{e(imp)}]</span> {e(item.get("headline",""))}</div>')
    h.append('</div>')

    # S9: RISK DASHBOARD
    pr = risk.get("portfolio_risk", {})
    h.append(f'<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Risk Dashboard</h2><table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;"><tr><td style="width:50%;vertical-align:top;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Portfolio Metrics</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">VaR (95%, 1d)</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{var_95:.1f}%</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Max Drawdown (1Y)</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;color:#dc2626;">{max_dd:.1f}%</td></tr><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Beta</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{p_beta:.2f}</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Sortino</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{pr.get("sortino_ratio",0):.2f}</td></tr><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Risk Score</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;color:{risk_color};">{risk_score}/100</td></tr></table></td>')
    total_stocks = len(concordance) or 1
    h.append('<td style="width:50%;vertical-align:top;">')
    if sector_counts:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Sector Exposure</div><table style="width:100%;border-collapse:collapse;font-size:12px;">')
        for i, (sec, cnt) in enumerate(sorted(sector_counts.items(), key=lambda x: -x[1])):
            pct = cnt * 100 / total_stocks
            bg = "#f8fafc" if i % 2 == 1 else "#fff"
            bar_w = min(pct, 100)
            h.append(f'<tr style="background:{bg};"><td style="padding:4px 8px;font-weight:600;font-size:11px;">{e(sec)}</td><td style="padding:4px 8px;text-align:right;font-weight:700;font-family:monospace;font-size:11px;">{cnt} ({pct:.0f}%)</td><td style="padding:4px 8px;width:60px;"><div style="height:4px;background:#e2e8f0;border-radius:2px;"><div style="height:100%;border-radius:2px;background:#6366f1;width:{bar_w}%;"></div></div></td></tr>')
        h.append('</table>')
    else:
        geo = concentration.get("geography",{}).get("concentration",{})
        h.append(f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Concentration</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Top Region</td><td style="padding:8px 12px;text-align:right;font-weight:700;">{e(geo.get("top_region","US"))} ({geo.get("top_region_pct",0):.0f}%)</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Stocks Analyzed</td><td style="padding:8px 12px;text-align:right;font-weight:700;">{len(concordance)}</td></tr></table>')
    h.append('</td></tr></table>')
    # Sector gaps
    if sector_gaps:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#d97706;margin:4px 0 8px 0;">Sector Gaps</div>')
        for gap in sector_gaps[:4]:
            sec = gap.get("sector", gap.get("etf", "?"))
            assessment = gap.get("gap", gap.get("assessment", gap.get("note", "")))
            h.append(f'<div style="padding:6px 12px;margin-bottom:4px;background:#fffbeb;border:1px solid #fde68a;border-radius:4px;font-size:11px;"><span style="font-weight:700;">{e(str(sec))}</span>: {e(str(assessment)[:80])}</div>')
    # Clusters
    if clusters:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin-bottom:10px;">Correlation Clusters</div>')
        for cl in clusters[:4]:
            stks = cl.get("stocks",cl.get("tickers",[])); ac = cl.get("avg_correlation",cl.get("average_correlation",0)); rn = cl.get("risk",cl.get("note",""))
            h.append(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:12px;"><span style="font-weight:700;color:#991b1b;">{", ".join(stks[:6])}</span> (corr: {ac:.2f})<br/><span style="color:#64748b;font-size:11px;">{e(str(rn)[:100])}</span></div>')
    # Stress
    if stress:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:20px 0 10px 0;">Stress Scenarios</div><table style="width:100%;border-collapse:separate;border-spacing:8px 0;"><tr>')
        for title,keys,bg,brd,lc,vc in [("CRASH -10%",["market_crash_10pct"],"#fef2f2","#fecaca","#991b1b","#dc2626"),("RATE +100bps",["rate_shock_100bps"],"#fffbeb","#fde68a","#92400e","#d97706"),("VIX TO 40",["vix_spike_to_40","vix_spike_40","vix_spike"],"#fef2f2","#fecaca","#991b1b","#dc2626")]:
            key = next((k for k in keys if k in stress), keys[0])
            sd = stress.get(key,{}); imp = sd.get("portfolio_impact_pct",sd.get("estimated_portfolio_impact_pct","?"))
            h.append(f'<td style="width:33%;padding:12px;background:{bg};border:1px solid {brd};border-radius:6px;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{lc};">{title}</div><div style="font-size:20px;font-weight:800;color:{vc};">{imp}%</div></td>')
        h.append('</tr></table>')
    h.append('</div>')

    # S10: OPPORTUNITIES — exclude stocks already in portfolio concordance
    portfolio_tickers = {c.get("ticker") for c in concordance if not c.get("is_opportunity")}
    opp_tops_raw = synth.get("top_opportunities", opps.get("top_opportunities", []))
    opp_tops = [o for o in opp_tops_raw if o.get("ticker") not in portfolio_tickers]
    h.append(f'<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">New Opportunities</h2><p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">{len(opp_tops)} candidates ranked.</p><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">#</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Sector</th><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Score</th><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">EXRET</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Thesis</th></tr>')
    for i,o in enumerate(opp_tops[:10]):
        bg = "#ecfdf5" if i<3 else "#f8fafc" if i%2==0 else "#fff"
        sc = o.get("scores",{}).get("final_score",o.get("opportunity_score",0)); ex = o.get("fundamentals",{}).get("exret",o.get("exret",""))
        thesis_raw = o.get("why_compelling", o.get("thesis", ""))
        if isinstance(thesis_raw, list):
            thesis_str = " | ".join(str(t) for t in thesis_raw)
        else:
            thesis_str = str(thesis_raw)
        h.append(f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;"><td style="padding:8px;text-align:center;font-weight:700;color:#059669;">{i+1}</td><td style="padding:8px;font-family:monospace;font-weight:700;">{e(o.get("ticker",""))}</td><td style="padding:8px;">{e(gics_sector(o.get("sector","")))}</td><td style="padding:8px;text-align:center;font-weight:700;color:{conv_color(int(sf(sc)))};">{sf(sc):.0f}</td><td style="padding:8px;text-align:center;">{e(str(ex))}</td><td style="padding:8px;font-size:11px;color:#334155;">{e(thesis_str[:100])}</td></tr>')
    h.append('</table></div>')

    # S11: ACTION ITEMS — split into Immediate Actions + Monitoring
    immediate = [entry for entry in concordance if entry.get("action") in _actionable]
    monitoring = [entry for entry in concordance if entry.get("action") not in _actionable]
    h.append('<div style="padding:32px 40px;border-bottom:2px solid #0f172a;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Action Items</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Priority-ranked by committee consensus.</p>')
    # Position limits from risk agent
    pos_limits = synth.get("position_limits", risk.get("position_limits", {}))
    if immediate:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin-bottom:10px;">Immediate Actions</div>')
        h.append('<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #1e293b;"><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:20px;">#</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:55px;">ACTION</th><th style="padding:10px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:60px;">STOCK</th><th style="padding:10px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;">DETAIL</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:32px;">CONV</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:45px;">SIZE</th><th style="padding:10px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;">KILL THESIS</th></tr>')
        for i, entry in enumerate(immediate):
            act = entry.get("action", "HOLD")
            tkr = entry.get("ticker", "")
            conv = entry.get("conviction", 0)
            sig = entry.get("signal", "?")
            ex = entry.get("exret", 0)
            sec = gics_sector(entry.get("sector", ""))
            rsi = entry.get("rsi", 0)
            ts = entry.get("tech_signal", "")
            mf = entry.get("macro_fit", "")
            # Use risk manager position limits
            pl = pos_limits.get(tkr, {})
            mp = pl.get("max_pct", entry.get("max_pct", 5.0))
            # Build informative detail
            if act == "SELL":
                sz = "Exit"
                dt = f"{sec} | EXRET {ex:.0f}% | RSI {rsi:.0f} | {ts}"
            elif act == "TRIM":
                sz = "Reduce"
                dt = f"{sec} | RSI {rsi:.0f} | {ts} | Macro: {abbr(mf)}"
            else:
                sz = f"{mp:.1f}%"
                dt = f"{sec} | EXRET {ex:.0f}% | RSI {rsi:.0f} | {ts}"
            # Generate stock-specific kill thesis from context
            kill = entry.get("kill_thesis", "")
            if not kill:
                beta = entry.get("beta", 1.0)
                bp = entry.get("buy_pct", 50)
                fv = entry.get("fund_view", "?")
                kill = _stock_kill_thesis(act, tkr, sec, rsi, ex, beta, bp, mf, ts, fv)
            ac, ab, abrd = action_color(act), action_bg(act), action_border(act)
            cc = conv_color_action(conv, act)
            h.append(f'<tr style="background:{ab};border-bottom:1px solid {abrd};"><td style="padding:8px 6px;text-align:center;font-weight:800;">{i+1}</td><td style="padding:8px 6px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:{ac};">{act}</span></td><td style="padding:8px 6px;font-family:monospace;font-weight:700;font-size:12px;">{e(tkr)}</td><td style="padding:8px 6px;color:#334155;font-size:11px;">{e(dt)}</td><td style="padding:8px 6px;text-align:center;font-weight:800;font-size:13px;color:{cc};">{conv}</td><td style="padding:8px 6px;text-align:center;font-family:monospace;font-size:10px;font-weight:600;">{sz}</td><td style="padding:6px;color:#64748b;font-style:italic;font-size:10px;line-height:1.3;">{e(kill[:120])}</td></tr>')
        h.append('</table>')
    if monitoring:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:20px 0 8px 0;">Monitoring</div>')
        h.append('<table style="width:100%;border-collapse:collapse;font-size:11px;"><tr style="border-bottom:1px solid #e2e8f0;"><th style="padding:6px 8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">STOCK</th><th style="padding:6px 8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">CONV</th><th style="padding:6px 8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">WATCH FOR</th></tr>')
        for i, entry in enumerate(monitoring):
            tkr = entry.get("ticker", "")
            conv = entry.get("conviction", 0)
            rsi = entry.get("rsi", 0)
            ts = entry.get("tech_signal", "")
            mf = entry.get("macro_fit", "")
            sig = entry.get("signal", "?")
            ex = entry.get("exret", 0)
            # Generate monitoring context
            notes = []
            if sig == "B" and conv < 55:
                notes.append(f"BUY signal but low conviction ({conv})")
            if rsi > 65:
                notes.append(f"RSI {rsi:.0f} approaching overbought")
            elif rsi < 35:
                notes.append(f"RSI {rsi:.0f} oversold — watch for reversal")
            if mf == "UNFAVORABLE":
                notes.append("macro headwind")
            if not notes:
                notes.append(f"EXRET {ex:.0f}% | RSI {rsi:.0f} | {abbr(ts)}")
            watch = "; ".join(notes)
            bg = "#f8fafc" if i % 2 == 1 else "#fff"
            h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:5px 8px;font-family:monospace;font-weight:700;">{e(tkr)}</td><td style="padding:5px 8px;text-align:center;font-weight:700;color:{conv_color(conv)};">{conv}</td><td style="padding:5px 8px;color:#64748b;">{e(watch)}</td></tr>')
        h.append('</table>')
    h.append('</div>')

    # S12: CHANGES
    # Normalize old action names and filter out pure migration noise
    real_changes = []
    for c in changes:
        prev_raw = c.get("prev_action", "?")
        curr = c.get("curr_action", "?")
        prev_norm = normalize_action(prev_raw)
        delta = c.get("delta", 0)
        # Skip if the change is just a nomenclature migration (same normalized action, small delta)
        if prev_norm == curr and abs(delta) <= 5:
            continue
        # Re-classify type based on normalized actions
        if prev_norm != curr:
            # Real action change
            act_rank = {"SELL": 0, "TRIM": 1, "HOLD": 2, "ADD": 3, "BUY": 4}
            if act_rank.get(curr, 2) > act_rank.get(prev_norm, 2):
                ct = "UPGRADE"
            elif act_rank.get(curr, 2) < act_rank.get(prev_norm, 2):
                ct = "DOWNGRADE"
            else:
                ct = "UPGRADE" if delta > 0 else "DOWNGRADE" if delta < 0 else "STABLE"
        elif c.get("type") == "NEW":
            ct = "NEW"
        else:
            # Same action, significant conviction change
            ct = "UPGRADE" if delta > 0 else "DOWNGRADE"
        real_changes.append({**c, "type": ct, "prev_display": prev_raw, "prev_norm": prev_norm})
    sig_changes = [c for c in real_changes if abs(c.get("delta", 0)) > 5 or c.get("prev_norm") != c.get("curr_action")]
    if sig_changes:
        h.append(f'<div style="padding:32px 40px;border-bottom:2px solid #cbd5e1;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Changes Since Last Committee</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">{len(sig_changes)} significant changes.</p><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #1e293b;"><th style="padding:8px;text-align:left;background:#f1f5f9;font-size:10px;font-weight:700;">STOCK</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">PREVIOUS</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">CURRENT</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">DELTA</th><th style="padding:8px;text-align:left;background:#f1f5f9;font-size:10px;font-weight:700;">TYPE</th></tr>')
        for c in sig_changes[:20]:
            ct = c.get("type", "")
            if ct == "DOWNGRADE":
                rbg, rbrd, ar, tc = "#fef2f2", "#fecaca", "&#9660;", "#991b1b"
            elif ct == "UPGRADE":
                rbg, rbrd, ar, tc = "#ecfdf5", "#a7f3d0", "&#9650;", "#065f46"
            elif ct == "NEW":
                rbg, rbrd, ar, tc = "#eff6ff", "#bfdbfe", "&#9733;", "#2563eb"
            else:
                rbg, rbrd, ar, tc = "#fff", "#e2e8f0", "&middot;", "#64748b"
            d = c.get("delta", 0)
            pn = c.get("prev_norm", normalize_action(c.get("prev_display", "?")))
            h.append(f'<tr style="background:{rbg};border-bottom:1px solid {rbrd};"><td style="padding:8px;font-family:monospace;font-weight:700;">{e(c.get("ticker",""))}</td><td style="padding:8px;text-align:center;">{e(pn)} ({c.get("prev_conviction",0)})</td><td style="padding:8px;text-align:center;">{e(c.get("curr_action","?"))} ({c.get("curr_conviction",0)})</td><td style="padding:8px;text-align:center;font-weight:700;color:{tc};">{d:+d}</td><td style="padding:8px;color:{tc};font-weight:600;">{ar} {e(ct)}</td></tr>')
        h.append('</table></div>')

    # FOOTER
    h.append(f'<div style="padding:24px 40px 28px 40px;background:#f8fafc;border-top:1px solid #e2e8f0;"><table style="width:100%;"><tr><td style="font-size:10px;color:#94a3b8;line-height:1.6;"><div style="font-weight:600;color:#64748b;margin-bottom:2px;">Investment Committee Report v6.0</div>Generated {today_long} &middot; 7 Specialist Agents (Claude Sonnet) + CIO Synthesis (Claude Opus)<br/>Data: etorotrade signals, eToro census (1,500 PIs), yfinance market data, web search</td><td style="text-align:right;font-size:10px;color:#cbd5e0;vertical-align:bottom;">For informational purposes only.<br/>Not financial advice.</td></tr></table></div>')
    h.append('</div></body></html>')

    return "\n".join(h)


def generate_report_from_files(
    reports_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate the HTML report from files on disk (original /tmp behavior).

    Args:
        reports_dir: Directory containing agent JSON reports.
        output_dir: Directory to write the HTML output.

    Returns:
        Path to the generated HTML file.
    """
    rd = reports_dir or REPORTS_DIR
    od = output_dir or OUTPUT_DIR

    synth = load_json(rd / "synthesis.json")
    fund = load_json(rd / "fundamental.json")
    tech = load_json(rd / "technical.json")
    macro = load_json(rd / "macro.json")
    census = load_json(rd / "census.json")
    news = load_json(rd / "news.json")
    opps = load_json(rd / "opportunities.json")
    risk = load_json(rd / "risk.json")

    html_str = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = od / f"{today}.html"
    od.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_str)

    return str(output_path)


if __name__ == "__main__":
    path = generate_report_from_files()
    print(f"HTML report: {path}")

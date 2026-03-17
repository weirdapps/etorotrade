"""
Committee HTML Report Generator

CIO v10.0: Consistent design system across all sections. No size constraint.
Every section uses the same visual language: section headers, tables, cards, and badges.

Usage:
    from trade_modules.committee_html import generate_report_html, generate_report_from_files

    # From pre-loaded dicts:
    html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)

    # From files on disk:
    output_path = generate_report_from_files()
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPORTS_DIR = Path(os.path.expanduser("~/.weirdapps-trading/committee/reports"))
OUTPUT_DIR = Path(os.path.expanduser("~/.weirdapps-trading/committee"))

# ═══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — shared primitives used by every section
# ═══════════════════════════════════════════════════════════════════════

# --- Color tokens ---
_C = {
    "bg_page": "#f8fafc", "bg_white": "#ffffff", "bg_alt": "#f1f5f9",
    "text_dark": "#0f172a", "text_body": "#334155", "text_mid": "#1e293b",
    "text_muted": "#64748b", "text_light": "#94a3b8",
    "border": "#e2e8f0", "border_heavy": "#cbd5e1",
    "bull": "#059669", "bull_text": "#065f46", "bull_bg": "#ecfdf5", "bull_border": "#a7f3d0",
    "bear": "#dc2626", "bear_text": "#991b1b", "bear_bg": "#fef2f2", "bear_border": "#fecaca",
    "warn": "#d97706", "warn_text": "#92400e", "warn_bg": "#fffbeb", "warn_border": "#fde68a",
    "info": "#2563eb", "info_bg": "#eff6ff", "info_border": "#bfdbfe",
    "hold": "#6366f1",
}

# --- Spacing tokens ---
_SECTION = "padding:32px 40px;border-bottom:1px solid #e2e8f0;"
_SECTION_H2 = "margin:0 0 16px 0;font-size:18px;font-weight:700;color:#0f172a;letter-spacing:-0.3px;"
_SECTION_SUB = "margin:0 0 14px 0;font-size:12px;color:#64748b;"
_TABLE = "width:100%;border-collapse:collapse;font-size:12px;"
_TH = "padding:8px 10px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.3px;text-transform:uppercase;color:#64748b;border-bottom:2px solid #e2e8f0;"
_TD = "padding:8px 10px;"
_MONO = "font-family:'SF Mono',Consolas,monospace;"
_LABEL = "font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;"

# --- Helper functions ---

def load_json(path):
    with open(path) as f:
        return json.load(f)

def sf(v, default=0):
    try:
        return float(str(v).replace('%', '').replace(',', '').replace('--', ''))
    except (ValueError, TypeError):
        return default

def e(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def action_color(act):
    return {"SELL": _C["bear"], "TRIM": _C["warn"], "BUY": _C["bull"], "ADD": _C["bull"], "HOLD": _C["hold"]}.get(act, _C["text_muted"])

def action_bg(act):
    return {"SELL": _C["bear_bg"], "TRIM": _C["warn_bg"], "BUY": _C["bull_bg"], "ADD": _C["bull_bg"], "HOLD": _C["bg_white"]}.get(act, _C["bg_white"])

def action_border(act):
    return {"SELL": _C["bear_border"], "TRIM": _C["warn_border"], "BUY": _C["bull_border"], "ADD": _C["bull_border"], "HOLD": _C["border"]}.get(act, _C["border"])

def sentiment_color(val):
    if val in ("BUY", "BULLISH", "ENTER_NOW", "FAVORABLE", "ALIGNED", "POSITIVE", "OK", "STRONG"):
        return _C["bull"]
    if val in ("SELL", "BEARISH", "AVOID", "EXIT_SOON", "UNFAVORABLE", "DIVERGENT", "NEGATIVE", "EXIT", "TRIM"):
        return _C["bear"]
    if val in ("WAIT_FOR_PULLBACK", "WARN", "NEUTRAL_BEARISH", "MODERATE_DIVERGENT"):
        return _C["warn"]
    return _C["text_muted"]

def conv_color(c):
    if c >= 70: return _C["bull"]
    if c >= 50: return _C["warn"]
    return _C["text_light"]

def conv_color_action(conv, action):
    if action in ("SELL", "TRIM"):
        if conv >= 70: return _C["bear"]
        if conv >= 50: return _C["warn"]
        return _C["text_light"]
    return conv_color(conv)

def badge(text, bg, color):
    """Consistent badge/pill element."""
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;color:{color};background:{bg};">{text}</span>'

def signal_badge(sig):
    c = {"B": _C["bull"], "H": _C["warn"], "S": _C["bear"], "I": _C["text_light"]}
    return badge(sig, c.get(sig, _C["text_light"]), "#fff")

def dot(color):
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};vertical-align:middle;"></span>'

def conv_display(c, delta=None):
    """Consistent conviction display: colored number + optional delta arrow."""
    col = conv_color(c)
    da = ""
    if delta is not None and delta != 0:
        ar = "&#9650;" if delta > 0 else "&#9660;"
        dc = _C["bull"] if delta > 0 else _C["bear"]
        da = f' <span style="color:{dc};font-size:9px;">{ar}{abs(delta)}</span>'
    return f'<span style="font-weight:800;font-size:13px;color:{col};">{c}</span>{da}'

def agent_badge(name, view, cbg, ctxt, cbrd):
    return f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:{cbg};color:{ctxt};border:1px solid {cbrd};">{name}: {view}</span>'

def abbr(text):
    """Designed abbreviations for agent views."""
    MAP = {
        "ENTER_NOW": "ENTER", "EXIT_SOON": "EXIT", "WAIT_FOR_PULLBACK": "WAIT",
        "FAVORABLE": "FAVOR", "UNFAVORABLE": "UNFAV",
        "ALIGNED": "ALIGN", "DIVERGENT": "DIVERG", "MODERATE_DIVERGENT": "M.DIV",
        "HIGH_NEGATIVE": "H.NEG", "HIGH_POSITIVE": "H.POS",
        "NEUTRAL": "NEUT", "POSITIVE": "POS", "NEGATIVE": "NEG",
    }
    return MAP.get(str(text), str(text))

# --- Sector normalization ---
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
    return _GICS_MAP.get(raw, raw if raw else "Other")

_ACTION_MIGRATION = {
    "IMMEDIATE SELL": "SELL", "REDUCE": "TRIM",
    "WEAK HOLD": "HOLD", "STRONG HOLD": "HOLD",
    "BUY NEW": "BUY", "WATCH": "HOLD",
}

def normalize_action(act):
    return _ACTION_MIGRATION.get(act, act)

# --- Kill thesis generation ---

def _stock_kill_thesis(act, tkr, sec, rsi, exret, beta, buy_pct, macro_fit, tech_sig, fund_view):
    """Generate a stock-specific kill thesis using actual data points."""
    if act == "SELL":
        parts = []
        if rsi < 35:
            parts.append(f"Selling at RSI {rsi:.0f} risks locking in worst-case exit")
        elif exret > 20 and buy_pct > 70:
            parts.append(f"Abandoning {exret:.0f}% EXRET with {buy_pct:.0f}% BUY consensus")
        elif "Crypto" in sec or "Digital" in sec:
            parts.append(f"Crypto sentiment can flip fast; {tkr} could rally 20%+ on any positive catalyst")
        else:
            parts.append(f"Thesis fails if {sec} sector reverses or {tkr} reports above expectations")
        if beta > 1.5:
            parts.append(f"Beta {beta:.1f} means sharp recovery if timing is wrong")
        return ". ".join(parts) + "."
    if act == "TRIM":
        parts = []
        if rsi < 35:
            parts.append(f"Trimming at RSI {rsi:.0f} (oversold) risks selling near the bottom")
            parts.append(f"EXRET {exret:.0f}% and {buy_pct:.0f}% BUY consensus suggest patience")
        elif "Crypto" in sec or "Digital" in sec:
            parts.append(f"{tkr} has high beta to digital assets; any crypto catalyst sparks outsized move")
        elif tech_sig == "EXIT_SOON" and rsi > 70:
            parts.append(f"RSI {rsi:.0f} overbought but strong trends can persist; premature exit loses momentum")
        elif tech_sig == "EXIT_SOON":
            parts.append(f"Tech says EXIT at RSI {rsi:.0f} but fund view {fund_view} ({exret:.0f}% EXRET) favors holding")
        elif macro_fit == "UNFAVORABLE":
            parts.append(f"Macro headwinds for {sec} could ease; {fund_view} view with {exret:.0f}% EXRET argues patience")
        else:
            parts.append(f"Trimming risks missing {exret:.0f}% EXRET if {buy_pct:.0f}% analyst consensus proves correct")
        if exret > 30:
            parts.append(f"Exceptional {exret:.0f}% upside means large opportunity cost if premature")
        return ". ".join(parts) + "."
    if act == "BUY":
        parts = []
        if sec == "Materials" and exret == 0:
            parts.append(f"Fails if dollar strengthens or inflation expectations reverse")
        elif beta > 1.5:
            parts.append(f"Beta {beta:.1f} amplifies drawdowns; broad selloff could erase entry")
        elif macro_fit == "UNFAVORABLE":
            parts.append(f"Buying into {sec} macro headwind; fails if regime worsens further")
        else:
            parts.append(f"Fails if {exret:.0f}% EXRET proves stale -- watch for estimate revisions")
        if rsi > 60:
            parts.append(f"RSI {rsi:.0f} not ideal entry; pullback to ~45 would improve risk/reward")
        return ". ".join(parts) + "."
    if act == "ADD":
        parts = []
        if rsi < 35:
            parts.append(f"Adding at RSI {rsi:.0f} (oversold) but momentum could deteriorate further")
        elif tech_sig == "AVOID":
            parts.append(f"Adding against technical AVOID signal -- fails if downtrend accelerates")
        else:
            parts.append(f"Fails if {exret:.0f}% EXRET target is cut or earnings disappoint")
        if buy_pct > 90:
            parts.append(f"Consensus crowded at {buy_pct:.0f}% BUY -- any miss gets punished hard")
        return ". ".join(parts) + "."
    return f"Monitor {tkr} for catalysts."


# ═══════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _section_open(title, subtitle=None, border="1px solid #e2e8f0"):
    """Consistent section wrapper opening."""
    parts = [f'<div style="padding:32px 40px;border-bottom:{border};">']
    parts.append(f'<h2 style="{_SECTION_H2}">{title}</h2>')
    if subtitle:
        parts.append(f'<p style="{_SECTION_SUB}">{subtitle}</p>')
    return "".join(parts)


def _section_close():
    return '</div>'


def _table_open(columns, col_styles=None):
    """Consistent table with header row.
    columns: list of (label, align) tuples.
    col_styles: optional dict mapping index to extra style.
    """
    parts = [f'<table style="{_TABLE}"><tr>']
    for i, (label, align) in enumerate(columns):
        extra = col_styles.get(i, "") if col_styles else ""
        parts.append(f'<th style="{_TH}text-align:{align};{extra}">{label}</th>')
    parts.append('</tr>')
    return "".join(parts)


def _table_row(cells, bg=None, border_color=None):
    """Consistent table row.
    cells: list of (content, align, extra_style) tuples.
    """
    bg_str = bg or _C["bg_white"]
    brd = border_color or "#f1f5f9"
    parts = [f'<tr style="background:{bg_str};">']
    for content, align, extra in cells:
        parts.append(f'<td style="{_TD}text-align:{align};border-bottom:1px solid {brd};{extra}">{content}</td>')
    parts.append('</tr>')
    return "".join(parts)


def _card(content, accent_color, bg=None):
    """Consistent card with left accent border."""
    card_bg = bg or _C["bg_white"]
    return (f'<div style="background:{card_bg};border:1px solid {_C["border"]};'
            f'border-left:4px solid {accent_color};border-radius:0 8px 8px 0;'
            f'padding:16px 20px;margin-bottom:12px;">{content}</div>')


def _group_separator(label, bg, text_color, border_color):
    """Consistent group separator row in tables."""
    return (f'<tr><td colspan="9" style="padding:4px 10px;background:{bg};'
            f'font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;'
            f'color:{text_color};border:1px solid {border_color};">{label}</td></tr>')


def _kpi_card(label, value, subtitle, value_color="#334155", bg=None):
    """Consistent KPI stat card."""
    card_bg = bg or _C["bg_alt"]
    return (f'<td style="width:33%;background:{card_bg};border:1px solid {_C["border"]};'
            f'border-radius:8px;padding:16px 20px;text-align:center;vertical-align:top;">'
            f'<div style="{_LABEL}margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{value_color};">{value}</div>'
            f'<div style="font-size:11px;color:{_C["text_muted"]};margin-top:2px;">{subtitle}</div></td>')


# ═══════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════

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
    Generate the Investment Committee HTML report.

    CIO v10.0 — 8 sections with consistent design system:
    S1: Executive Summary
    S2: Macro & Market Context
    S3: Stock Analysis Grid (all stocks, grouped by action)
    S4: Where We Disagreed
    S5: Sentiment & Census
    S6: News & Events
    S7: Risk Dashboard
    S8: Action Items (tiered with kill theses)
    Epilogue: Changes Since Last Committee
    """
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    try:
        today_long = datetime.strptime(today, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        today_long = today

    # --- Extract data ---
    concordance = synth.get("concordance", [])
    regime = synth.get("regime", "CAUTIOUS")
    macro_score = synth.get("macro_score", 0)
    rotation = synth.get("rotation_phase", "UNKNOWN")
    risk_score = synth.get("risk_score", 50)
    pr_synth = synth.get("portfolio_risk", {})
    var_95_raw = pr_synth.get("var_95", synth.get("var_95", 0))
    var_95 = var_95_raw if abs(var_95_raw) <= 10 else var_95_raw / 100
    max_dd_raw = pr_synth.get("max_drawdown", synth.get("max_drawdown", 0))
    max_dd = max_dd_raw * 100 if 0 < abs(max_dd_raw) < 1 else max_dd_raw
    p_beta = pr_synth.get("portfolio_beta", synth.get("portfolio_beta", 1.0))
    fg_top100 = synth.get("fg_top100", 50)
    fg_broad = synth.get("fg_broad", 50)
    changes = synth.get("changes", [])
    sector_gaps = synth.get("sector_gaps", [])
    clusters = synth.get("correlation_clusters", [])
    stress = synth.get("stress_scenarios", {})
    indicators = synth.get("indicators", {})
    sector_rankings = synth.get("sector_rankings", {})
    signal_date = synth.get("signal_date", today)
    census_date = synth.get("census_date", today)

    delta_map = {c.get("ticker"): c.get("delta", 0) for c in changes}

    # Normalize sectors
    for entry in concordance:
        entry["sector"] = gics_sector(entry.get("sector", ""))

    sector_counts = {}
    for entry in concordance:
        sec = entry.get("sector", "Other")
        if sec:
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # Sort: SELL > TRIM > BUY > ADD > HOLD, then by conviction desc
    action_order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    concordance.sort(key=lambda x: (action_order.get(x.get("action", "HOLD"), 4), -x.get("conviction", 0)))

    sells = sum(1 for c in concordance if c.get("action") == "SELL")
    buys = sum(1 for c in concordance if c.get("action") in ("BUY", "ADD"))
    trims = sum(1 for c in concordance if c.get("action") == "TRIM")
    holds = sum(1 for c in concordance if c.get("action") == "HOLD")

    # Verdict
    if regime == "RISK_ON":
        verdict, vbg, vbrd, vval = "RISK-ON", _C["bull_bg"], _C["bull_border"], _C["bull"]
    elif regime == "RISK_OFF":
        verdict, vbg, vbrd, vval = "DEFENSIVE", _C["bear_bg"], _C["bear_border"], _C["bear"]
    else:
        verdict, vbg, vbrd, vval = "CAUTIOUS", _C["warn_bg"], _C["warn_border"], _C["warn"]
    risk_color = _C["bear"] if risk_score >= 70 else _C["warn"] if risk_score >= 40 else _C["bull"]

    # CIO narrative
    sell_tickers = [c["ticker"] for c in concordance if c.get("action") == "SELL"]
    vix_val = indicators.get('vix', 0)
    leading = [etf for etf, d in sector_rankings.items() if d.get("return_1m", 0) > 2]
    lagging = [etf for etf, d in sector_rankings.items() if d.get("return_1m", 0) < -2]
    parts = []
    if regime == "RISK_OFF":
        parts.append(f"We are in a {rotation.replace('_', ' ').lower()} environment with deteriorating breadth")
        if lagging:
            parts[-1] += f" -- {len(lagging)} of {len(sector_rankings)} sectors negative"
        parts[-1] += ". Defensive positioning warranted."
    elif regime == "RISK_ON":
        parts.append(f"Constructive macro backdrop. {len(leading)} sectors leading with broad participation.")
    else:
        parts.append(f"{rotation.replace('_', ' ')} phase with mixed signals (macro score {macro_score}).")
        if vix_val > 20:
            parts[-1] = parts[-1][:-1] + f", VIX elevated at {vix_val:.0f}."
    if sells:
        parts.append(f"Highest-priority action: exit {', '.join(sell_tickers)} ({sells} SELL).")
    if trims:
        parts.append(f"Trim {trims} positions where technical momentum has broken down or macro headwinds apply.")
    top_adds = [c["ticker"] for c in concordance if c.get("action") in ("BUY", "ADD") and c.get("conviction", 0) >= 70][:3]
    if top_adds:
        parts.append(f"Highest-conviction additions: {', '.join(top_adds)}.")
    narrative = " ".join(parts)

    # --- Disagreement detection ---
    disagreements = _detect_disagreements(concordance)

    # --- Build HTML ---
    h = []
    h.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1.0"></head>')
    h.append(f'<body style="margin:0;padding:0;background:{_C["bg_page"]};'
             f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,'
             f'\'Helvetica Neue\',Arial,sans-serif;color:{_C["text_body"]};'
             f'line-height:1.6;-webkit-font-smoothing:antialiased;">')
    h.append(f'<div style="max-width:960px;margin:0 auto;background:{_C["bg_white"]};'
             f'box-shadow:0 1px 3px rgba(0,0,0,0.1);">')

    # ── HEADER ──
    h.append(f'<div style="background-color:{_C["text_dark"]};padding:32px 40px 28px 40px;">'
             f'<table style="width:100%;"><tr>'
             f'<td style="vertical-align:middle;width:50px;">'
             f'<div style="width:36px;height:36px;border-radius:8px;background:rgba(255,255,255,0.15);'
             f'text-align:center;line-height:36px;font-size:18px;color:#fff;font-weight:800;">IC</div></td>'
             f'<td style="vertical-align:middle;">'
             f'<h1 style="margin:0;font-size:24px;font-weight:700;color:#fff;letter-spacing:-0.5px;">'
             f'Investment Committee</h1>'
             f'<p style="margin:2px 0 0 0;font-size:12px;color:{_C["text_light"]};">'
             f'{today_long} &middot; Signals: {e(signal_date)} &middot; Census: {e(census_date)}</p>'
             f'</td></tr></table></div>')

    # ── S1: EXECUTIVE SUMMARY ──
    h.append(_section_open("Executive Summary", border="2px solid " + _C["border_heavy"]))
    # KPI cards
    h.append('<table style="width:100%;border-collapse:separate;border-spacing:10px 0;margin-bottom:20px;"><tr>')
    h.append(_kpi_card("CIO Verdict", verdict, f"{sells}S {trims}T {buys}B/A {holds}H", vval, vbg))
    h.append(_kpi_card("Macro Regime", regime, f"Score {macro_score} &middot; {rotation.replace('_', ' ')}", vval))
    h.append(_kpi_card("Portfolio Risk", f"{risk_score}/100", f"Beta {p_beta:.2f} &middot; VaR {var_95:.1f}%", risk_color))
    h.append('</tr></table>')
    # Narrative
    h.append(f'<div style="background:{_C["bg_page"]};border:1px solid {_C["border"]};'
             f'border-left:4px solid {vval};border-radius:0 8px 8px 0;'
             f'padding:14px 18px;margin-bottom:16px;font-size:13px;color:{_C["text_body"]};'
             f'line-height:1.6;">{e(narrative)}</div>')
    # Risk dilution warning
    if synth.get("risk_diluted"):
        warned = sum(1 for c in concordance if c.get("risk_warning"))
        h.append(f'<div style="background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                 f'border-left:4px solid {_C["warn"]};border-radius:0 8px 8px 0;'
                 f'padding:10px 18px;margin-bottom:16px;font-size:11px;color:#78350f;">'
                 f'<b>Risk Warning Dilution:</b> {warned}/{len(concordance)} stocks flagged '
                 f'({warned * 100 // max(len(concordance), 1)}%) -- systemic, not stock-specific.</div>')
    # Stress scenarios
    if stress:
        h.append('<table style="width:100%;border-collapse:separate;border-spacing:8px 0;margin-bottom:16px;"><tr>')
        scenarios = [
            ("CRASH -10%", ["market_crash_10pct"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
            ("RATE +100bps", ["rate_shock_100bps"], _C["warn_bg"], _C["warn_border"], _C["warn_text"], _C["warn"]),
            ("VIX TO 40", ["vix_spike_to_40", "vix_spike_40", "vix_spike"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
        ]
        for title, keys, bg, brd, lc, vc in scenarios:
            key = next((k for k in keys if k in stress), keys[0])
            sd = stress.get(key, {})
            imp = sd.get("portfolio_impact_pct", sd.get("estimated_portfolio_impact_pct", "?"))
            h.append(f'<td style="width:33%;padding:10px;background:{bg};border:1px solid {brd};'
                     f'border-radius:6px;text-align:center;">'
                     f'<div style="{_LABEL}color:{lc};">{title}</div>'
                     f'<div style="font-size:18px;font-weight:800;color:{vc};margin-top:4px;">{imp}%</div></td>')
        h.append('</tr></table>')
    # Priority actions
    urgent = [en for en in concordance if en.get("action") in ("SELL", "TRIM")]
    top_buys = [en for en in concordance if en.get("action") in ("BUY", "ADD") and en.get("conviction", 0) >= 65][:5]
    if urgent or top_buys:
        h.append(f'<div style="{_LABEL}margin-bottom:8px;">Priority Actions</div><div>')
        for en in urgent + top_buys:
            act, tkr, conv = en.get("action", "HOLD"), en.get("ticker", ""), en.get("conviction", 0)
            h.append(f'<span style="display:inline-block;padding:4px 10px;margin:2px 3px 2px 0;'
                     f'border-radius:5px;font-size:11px;font-weight:700;color:#fff;'
                     f'background:{action_color(act)};">{act} {e(tkr)} ({conv})</span>')
        h.append('</div>')
    h.append(_section_close())

    # ── S2: MACRO & MARKET CONTEXT ──
    h.append(_section_open("Macro &amp; Market Context"))
    dxy_raw = sf(indicators.get('dxy', 0))
    dxy_str = f"{dxy_raw:.1f}" if 80 <= dxy_raw <= 120 else "N/A"
    macro_ind = [
        ("10Y Yield", f"{indicators.get('yield_10y', 0):.2f}%", indicators.get('yield_curve_status', 'NORMAL')),
        ("Yield Curve", f"{indicators.get('yield_curve_spread', 0):.0f}bps",
         "POSITIVE" if sf(indicators.get('yield_curve_spread', 0)) > 0 else "INVERTED"),
        ("VIX", f"{indicators.get('vix', 0):.1f}",
         "ELEVATED" if sf(indicators.get('vix', 0)) > 20 else "NORMAL"),
        ("Dollar (DXY)", dxy_str,
         indicators.get('dollar_trend', 'STABLE') if dxy_str != "N/A" else "N/A"),
    ]
    h.append(_table_open([("Indicator", "left"), ("Value", "center"), ("Signal", "center")]))
    for i, (n, v, st) in enumerate(macro_ind):
        bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
        dc = (_C["bear"] if any(x in str(st) for x in ("INVERTED", "ELEVATED", "RISK_OFF"))
              else _C["bull"] if any(x in str(st) for x in ("POSITIVE", "NORMAL", "RISK_ON"))
              else _C["warn"])
        h.append(_table_row([
            (f'<span style="font-weight:600;color:{_C["text_mid"]};">{e(n)}</span>', "left", ""),
            (f'<span style="{_MONO}font-weight:700;">{e(v)}</span>', "center", ""),
            (f'{dot(dc)} <span style="font-size:11px;color:{_C["text_muted"]};">{e(st)}</span>', "center", ""),
        ], bg=bg))
    h.append('</table>')

    # Market-wide technical context
    tech_stocks = tech.get("stocks", {})
    bearish_macd = sum(1 for d in tech_stocks.values() if d.get("macd_signal") == "BEARISH")
    total_tech = len(tech_stocks)
    if total_tech > 0:
        avg_rsi = sum(d.get("rsi", 50) for d in tech_stocks.values()) / total_tech
        below50 = sum(1 for d in tech_stocks.values() if not d.get("above_sma50", True))
        if bearish_macd > total_tech * 0.4:
            h.append(f'<div style="background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                     f'border-left:4px solid {_C["warn"]};border-radius:0 8px 8px 0;'
                     f'padding:12px 18px;margin:16px 0;font-size:12px;color:{_C["text_body"]};">'
                     f'<b style="color:{_C["warn"]};">BROAD MARKET SIGNAL:</b> '
                     f'{bearish_macd}/{total_tech} stocks BEARISH MACD. '
                     f'{below50} below SMA50. Avg RSI: {avg_rsi:.0f}.</div>')

    # Sector rotation
    if sector_rankings:
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Sector Rotation (1M)</div>')
        h.append('<table style="width:100%;border-collapse:separate;border-spacing:4px 0;font-size:10px;"><tr>')
        for etf, data in sorted(sector_rankings.items(), key=lambda x: x[1].get("return_1m", 0), reverse=True):
            ret = data.get("return_1m", 0)
            sbg = _C["bull_bg"] if ret > 2 else _C["bear_bg"] if ret < -2 else _C["bg_page"]
            stxt = _C["bull"] if ret > 2 else _C["bear"] if ret < -2 else _C["text_body"]
            h.append(f'<td style="padding:6px 4px;text-align:center;background:{sbg};border-radius:4px;">'
                     f'<div style="font-weight:700;color:{_C["text_body"]};font-size:9px;">{e(etf)}</div>'
                     f'<div style="font-weight:800;color:{stxt};">{ret:+.1f}%</div></td>')
        h.append('</tr></table>')
    h.append(_section_close())

    # ── S3: STOCK ANALYSIS GRID ──
    # All stocks in one grid, grouped by action type
    sell_trim_list = [en for en in concordance if en.get("action") in ("SELL", "TRIM")]
    buy_add_list = [en for en in concordance if en.get("action") in ("BUY", "ADD")]
    hold_list = [en for en in concordance if en.get("action") == "HOLD"]

    h.append(_section_open("Stock Analysis Grid",
                           f"{len(concordance)} stocks: {sells} sell, {trims} trim, {buys} buy/add, {holds} hold."))
    h.append('<div style="overflow-x:auto;">')
    # Grid header
    grid_hdr = lambda bg, col, txt: (f'<th style="padding:8px 6px;text-align:center;background:{bg};'
                                     f'color:{col};font-size:9px;font-weight:700;letter-spacing:0.3px;'
                                     f'border:1px solid #1e293b;">{txt}</th>')
    h.append(f'<table style="width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap;"><tr>')
    h.append(grid_hdr(_C["text_dark"], "#fff", "STOCK") + grid_hdr(_C["text_dark"], "#fff", "SIG"))
    h.append(grid_hdr("#1e293b", "#93c5fd", "FUND") + grid_hdr("#1e293b", "#c4b5fd", "TECH"))
    h.append(grid_hdr("#1e293b", "#fcd34d", "MACRO") + grid_hdr("#1e293b", "#6ee7b7", "CENS"))
    h.append(grid_hdr(_C["text_dark"], "#fbbf24", "EXR") + grid_hdr(_C["text_dark"], "#fff", "ACT"))
    h.append(grid_hdr(_C["text_dark"], "#fff", "CONV"))
    h.append('</tr>')

    def _grid_row(entry):
        act = entry.get("action", "HOLD")
        tkr = entry.get("ticker", "")
        sig = entry.get("signal", "?")
        fs = entry.get("fund_score", 0)
        fv = entry.get("fund_view", "?")
        ts = entry.get("tech_signal", "?")
        rsi = entry.get("rsi", 0)
        mf = entry.get("macro_fit", "?")
        ce = entry.get("census", "?")
        conv = entry.get("conviction", 0)
        sm = "*" if entry.get("fund_synthetic") else ""
        ex = entry.get("exret", 0)
        delta = delta_map.get(tkr)
        exc = _C["bull"] if ex > 5 else _C["bear"] if ex < 0 else _C["text_body"]
        rb = action_bg(act)
        p = f'style="padding:6px 4px;text-align:center;border:1px solid {_C["border"]};'
        sv = "font-weight:600;font-size:10px;"
        return (f'<tr style="background:{rb};">'
                f'<td {p}{_MONO}font-weight:700;font-size:11px;color:{_C["text_dark"]};text-align:left;padding-left:8px;">{e(tkr)}</td>'
                f'<td {p}">{signal_badge(sig)}</td>'
                f'<td {p}color:{sentiment_color(fv)};{sv}">{abbr(fv)}({fs:.0f}){sm}</td>'
                f'<td {p}color:{sentiment_color(ts)};{sv}">{abbr(ts)}({rsi:.0f})</td>'
                f'<td {p}color:{sentiment_color(mf)};{sv}">{abbr(mf)}</td>'
                f'<td {p}color:{sentiment_color(ce)};{sv}">{abbr(ce)}</td>'
                f'<td {p}{_MONO}font-weight:700;color:{exc};">{ex:.0f}%</td>'
                f'<td {p}">{badge(act, action_color(act), "#fff")}</td>'
                f'<td {p}">{conv_display(conv, delta)}</td></tr>')

    if sell_trim_list:
        h.append(_group_separator("SELL / TRIM", _C["bear_bg"], _C["bear_text"], _C["bear_border"]))
        for en in sell_trim_list:
            h.append(_grid_row(en))
    if buy_add_list:
        h.append(_group_separator("BUY / ADD", _C["bull_bg"], _C["bull_text"], _C["bull_border"]))
        for en in buy_add_list:
            h.append(_grid_row(en))
    if hold_list:
        h.append(_group_separator("HOLD / MONITOR", _C["bg_alt"], "#475569", _C["border_heavy"]))
        for en in hold_list:
            h.append(_grid_row(en))
    h.append('</table></div>')
    # Legend
    h.append(f'<div style="margin-top:10px;padding:8px 12px;background:{_C["bg_page"]};'
             f'border:1px solid {_C["border"]};border-radius:6px;font-size:9px;color:{_C["text_muted"]};">'
             f'* = Synthetic &middot; ENTER/EXIT/WAIT = Tech timing &middot; '
             f'FAVOR/UNFAV = Macro fit &middot; ALIGN/DIVERG = Census &middot; '
             f'&#9650;&#9660; = Conviction change from prior committee</div>')
    h.append(_section_close())

    # ── S4: WHERE WE DISAGREED ──
    h.append(_section_open("Where We Disagreed",
                           "Structured disagreement is the committee's edge."))
    for d in disagreements[:6]:
        fb = agent_badge("Fund", d["fund_view"], "#dbeafe", "#1e40af", "#93c5fd")
        tb = agent_badge("Tech", d["tech_signal"], "#ede9fe", "#5b21b6", "#c4b5fd")
        inner = (f'<div style="margin-bottom:8px;">'
                 f'<span style="{_MONO}font-weight:800;font-size:14px;color:{_C["text_dark"]};">{e(d["ticker"])}</span>'
                 f' <span style="font-size:12px;font-weight:600;color:{_C["text_body"]};margin-left:8px;">'
                 f'{e(d["headline"])}</span></div>'
                 f'<div style="margin-bottom:10px;">{fb}{tb}</div>'
                 f'<div style="font-size:12px;color:{_C["text_body"]};line-height:1.6;margin-bottom:12px;">'
                 f'{e(d["narrative"])}</div>'
                 f'<div style="background:{_C["bg_page"]};border:1px solid {_C["border"]};'
                 f'border-radius:6px;padding:10px 14px;">'
                 f'<span style="{_LABEL}">Resolution:</span> '
                 f'<span style="font-size:12px;font-weight:700;color:{action_color(d["resolution"])};">'
                 f'{d["resolution"]} (conv. {d["conviction"]})</span></div>')
        h.append(_card(inner, d["accent"]))
    if not disagreements:
        h.append(f'<div style="padding:14px;background:{_C["bg_page"]};border:1px solid {_C["border"]};'
                 f'border-radius:6px;color:{_C["text_muted"]};font-size:12px;">All agents broadly aligned.</div>')
    h.append(_section_close())

    # ── S5: SENTIMENT & CENSUS ──
    fg_label = lambda v: ("EXTREME GREED" if v >= 75 else "GREED" if v >= 55 else "NEUTRAL" if v >= 45
                          else "FEAR" if v >= 25 else "EXTREME FEAR")
    fg_color = lambda v: (_C["bear"] if v >= 75 else _C["warn"] if v >= 55 else _C["text_muted"] if v >= 45
                          else _C["info"] if v >= 25 else _C["hold"])
    cash100 = census.get("sentiment", {}).get("cash_top100", 0)
    cash_label = "Defensive" if cash100 > 15 else "Deploying" if cash100 < 8 else "Normal"
    h.append(_section_open("Sentiment &amp; Census"))
    h.append('<table style="width:100%;border-collapse:separate;border-spacing:10px 0;margin-bottom:16px;"><tr>')
    h.append(_kpi_card("F&amp;G Top 100", str(fg_top100), fg_label(fg_top100), fg_color(fg_top100)))
    h.append(_kpi_card("F&amp;G Broad", str(fg_broad), fg_label(fg_broad), fg_color(fg_broad)))
    h.append(_kpi_card("Cash Top 100", f"{cash100:.1f}%", cash_label))
    h.append('</tr></table>')
    missing = synth.get("missing_popular", [])
    if missing:
        h.append(f'<div style="{_LABEL}color:{_C["warn"]};margin-bottom:8px;">Popular NOT in Portfolio</div><div>')
        for m in missing[:10]:
            tkr = m if isinstance(m, str) else m.get("symbol", m.get("ticker", "?"))
            h.append(f'<span style="display:inline-block;padding:3px 10px;margin:2px 3px;border-radius:12px;'
                     f'font-size:10px;font-weight:600;background:{_C["warn_bg"]};color:{_C["warn_text"]};'
                     f'border:1px solid {_C["warn_border"]};">{e(tkr)}</span>')
        h.append('</div>')
    h.append(_section_close())

    # ── S6: NEWS & EVENTS ──
    h.append(_section_open("News &amp; Events"))
    for item in news.get("breaking_news", [])[:5]:
        hl = item.get("headline", "")
        imp = item.get("impact", item.get("score", "NEUTRAL"))
        if "NEGATIVE" in str(imp):
            nbg, nbrd, nlc = _C["bear_bg"], _C["bear"], _C["bear_text"]
        elif "POSITIVE" in str(imp):
            nbg, nbrd, nlc = _C["bull_bg"], _C["bull"], _C["bull_text"]
        else:
            nbg, nbrd, nlc = _C["bg_page"], _C["text_light"], _C["text_body"]
        h.append(f'<div style="background:{nbg};border-left:3px solid {nbrd};border-radius:0 4px 4px 0;'
                 f'padding:10px 14px;margin-bottom:8px;font-size:12px;font-weight:600;color:{nlc};">'
                 f'{e(hl)} <span style="font-weight:400;color:{_C["text_muted"]};font-size:10px;">'
                 f'({e(imp)})</span></div>')
    # Portfolio-specific news
    pn = news.get("portfolio_news", {})
    notable = [(t, items[0]) for t, items in pn.items()
               if items and any(x in str(items[0].get("impact", items[0].get("score", "")))
                                for x in ("POSITIVE", "NEGATIVE"))]
    if notable:
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Portfolio News</div>')
        h.append(_table_open([("Stock", "left"), ("Impact", "center"), ("Headline", "left")]))
        for tkr, item in notable[:6]:
            imp = item.get("impact", item.get("score", "NEUTRAL"))
            ic = _C["bear"] if "NEGATIVE" in str(imp) else _C["bull"]
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(tkr)}</span>', "left", ""),
                (f'<span style="color:{ic};font-weight:600;font-size:10px;">{e(imp)}</span>', "center", ""),
                (e(item.get("headline", "")[:80]), "left", f"color:{_C['text_body']};"),
            ]))
        h.append('</table>')
    h.append(_section_close())

    # ── S7: RISK DASHBOARD ──
    pr = risk.get("portfolio_risk", {})
    h.append(_section_open("Risk Dashboard"))
    h.append(f'<table style="width:100%;border-collapse:separate;border-spacing:16px 0;"><tr>'
             f'<td style="width:50%;vertical-align:top;">')
    # Risk metrics table
    risk_metrics = [
        ("VaR (95%)", f"{var_95:.1f}%", _C["text_body"]),
        ("Max Drawdown", f"{max_dd:.1f}%", _C["bear"]),
        ("Portfolio Beta", f"{p_beta:.2f}", _C["text_body"]),
        ("Sortino Ratio", f"{pr.get('sortino_ratio', 0):.2f}", _C["text_body"]),
    ]
    h.append(f'<table style="{_TABLE}">')
    for i, (label, val, vc) in enumerate(risk_metrics):
        bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
        h.append(f'<tr style="background:{bg};"><td style="padding:8px 10px;font-weight:600;">{label}</td>'
                 f'<td style="padding:8px 10px;text-align:right;{_MONO}font-weight:700;color:{vc};">{val}</td></tr>')
    h.append('</table>')
    h.append(f'</td><td style="width:50%;vertical-align:top;">')
    # Sector exposure
    total_stocks = len(concordance) or 1
    if sector_counts:
        h.append(f'<div style="{_LABEL}margin-bottom:8px;">Sector Exposure</div>')
        h.append(f'<table style="{_TABLE}">')
        for i, (sec, cnt) in enumerate(sorted(sector_counts.items(), key=lambda x: -x[1])):
            pct = cnt * 100 / total_stocks
            bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            h.append(f'<tr style="background:{bg};">'
                     f'<td style="padding:4px 8px;font-weight:600;font-size:11px;">{e(sec)}</td>'
                     f'<td style="padding:4px 8px;text-align:right;{_MONO}font-weight:700;font-size:11px;">'
                     f'{cnt} ({pct:.0f}%)</td>'
                     f'<td style="padding:4px 8px;width:60px;">'
                     f'<div style="height:4px;background:{_C["border"]};border-radius:2px;">'
                     f'<div style="height:100%;border-radius:2px;background:{_C["hold"]};'
                     f'width:{min(pct, 100):.0f}%;"></div></div></td></tr>')
        h.append('</table>')
    h.append('</td></tr></table>')
    # Correlation clusters
    if clusters:
        h.append(f'<div style="{_LABEL}color:{_C["bear"]};margin:16px 0 8px 0;">Correlation Clusters</div>')
        for cl in clusters[:4]:
            stks = cl.get("stocks", cl.get("tickers", []))
            ac = cl.get("avg_correlation", cl.get("average_correlation", 0))
            inner = (f'<span style="font-weight:700;color:{_C["bear_text"]};">{", ".join(stks[:5])}</span>'
                     f' <span style="color:{_C["text_muted"]};">(r={ac:.2f})</span>')
            h.append(_card(inner, _C["bear"], _C["bear_bg"]))
    # Sector gaps
    if sector_gaps:
        h.append(f'<div style="{_LABEL}color:{_C["warn"]};margin:16px 0 8px 0;">Sector Gaps</div>')
        for gap in sector_gaps[:4]:
            sec = gap.get("sector", gap.get("etf", "?"))
            assessment = gap.get("gap", gap.get("assessment", gap.get("note", "")))
            inner = f'<b>{e(str(sec))}</b>: {e(str(assessment)[:100])}'
            h.append(_card(inner, _C["warn"], _C["warn_bg"]))
    h.append(_section_close())

    # ── S8: ACTION ITEMS ──
    pos_limits = synth.get("position_limits", risk.get("position_limits", {}))
    urgent_items = [en for en in concordance if en.get("action") in ("SELL", "TRIM")]
    deploy_items = sorted([en for en in concordance if en.get("action") in ("BUY", "ADD")],
                          key=lambda x: -x.get("conviction", 0))
    monitor_items = sorted([en for en in concordance if en.get("action") == "HOLD"],
                           key=lambda x: -x.get("conviction", 0))

    h.append(_section_open("Action Items",
                           "Tiered by urgency. Kill thesis: what makes each trade fail.",
                           border="2px solid " + _C["text_dark"]))

    # Urgent: SELL/TRIM — cards with kill thesis
    if urgent_items:
        h.append(f'<div style="{_LABEL}color:{_C["bear_text"]};margin-bottom:10px;">'
                 f'Urgent: Exit / Reduce ({len(urgent_items)})</div>')
        for en in urgent_items:
            act = en.get("action", "HOLD")
            tkr = en.get("ticker", "")
            conv = en.get("conviction", 0)
            ex = en.get("exret", 0)
            rsi = en.get("rsi", 0)
            sec = en.get("sector", "")
            ts = en.get("tech_signal", "")
            mf = en.get("macro_fit", "")
            beta = en.get("beta", 1.0)
            bp = en.get("buy_pct", 50)
            fv = en.get("fund_view", "?")
            kill = en.get("kill_thesis") or _stock_kill_thesis(act, tkr, sec, rsi, ex, beta, bp, mf, ts, fv)
            cc = conv_color_action(conv, act)
            inner = (f'<table style="width:100%;"><tr><td>'
                     f'{badge(act, action_color(act), "#fff")} '
                     f'<span style="{_MONO}font-weight:800;font-size:14px;margin-left:8px;">{e(tkr)}</span> '
                     f'<span style="font-size:11px;color:{_C["text_muted"]};margin-left:8px;">'
                     f'{sec} | RSI {rsi:.0f} | {abbr(ts)} | {abbr(mf)}</span></td>'
                     f'<td style="text-align:right;">{conv_display(conv)}</td></tr></table>'
                     f'<div style="font-size:11px;color:{_C["text_muted"]};font-style:italic;'
                     f'margin-top:8px;line-height:1.5;">{e(kill)}</div>')
            h.append(_card(inner, action_color(act), action_bg(act)))

    # Deploy: BUY/ADD — consistent table
    if deploy_items:
        h.append(f'<div style="{_LABEL}color:{_C["bull_text"]};margin:20px 0 10px 0;">'
                 f'Deploy: Buy / Add ({len(deploy_items)})</div>')
        h.append(_table_open([("ACT", "center"), ("STOCK", "left"), ("SECTOR", "left"),
                              ("EXRET", "center"), ("RSI", "center"), ("TECH", "center"),
                              ("CONV", "center"), ("SIZE", "center")],
                             {0: "width:50px;", 6: "width:40px;", 7: "width:45px;"}))
        for en in deploy_items:
            act = en.get("action", "BUY")
            tkr = en.get("ticker", "")
            conv = en.get("conviction", 0)
            ex = en.get("exret", 0)
            rsi = en.get("rsi", 0)
            sec = en.get("sector", "")
            ts = en.get("tech_signal", "")
            pl = pos_limits.get(tkr, {})
            mp = pl.get("max_pct", en.get("max_pct", 5.0))
            h.append(_table_row([
                (badge(act, action_color(act), "#fff"), "center", ""),
                (f'<span style="{_MONO}font-weight:700;">{e(tkr)}</span>', "left", ""),
                (f'<span style="font-size:11px;">{e(sec)}</span>', "left", ""),
                (f'<span style="{_MONO}font-weight:700;color:{_C["bull"] if ex > 5 else _C["text_body"]};">'
                 f'{ex:.0f}%</span>', "center", ""),
                (f'<span style="{_MONO}">{rsi:.0f}</span>', "center", ""),
                (f'<span style="color:{sentiment_color(ts)};font-weight:600;font-size:10px;">'
                 f'{abbr(ts)}</span>', "center", ""),
                (conv_display(conv), "center", ""),
                (f'<span style="{_MONO}font-size:11px;color:{_C["text_muted"]};">{mp:.0f}%</span>', "center", ""),
            ], bg=_C["bull_bg"], border_color=_C["bull_border"]))
        h.append('</table>')

    # Monitor: HOLD — same table structure
    if monitor_items:
        h.append(f'<div style="{_LABEL}margin:20px 0 10px 0;">Monitor ({len(monitor_items)})</div>')
        h.append(_table_open([("STOCK", "left"), ("SIG", "center"), ("FUND", "center"),
                              ("TECH", "center"), ("EXRET", "center"), ("CONV", "center"),
                              ("WATCH FOR", "left")],
                             {5: "width:40px;"}))
        for i, en in enumerate(monitor_items):
            tkr = en.get("ticker", "")
            sig = en.get("signal", "?")
            fv = en.get("fund_view", "?")
            fs = en.get("fund_score", 0)
            ts = en.get("tech_signal", "?")
            rsi = en.get("rsi", 0)
            conv = en.get("conviction", 0)
            ex = en.get("exret", 0)
            mf = en.get("macro_fit", "?")
            # Context-aware watch note
            notes = []
            if sig == "B" and conv < 55:
                notes.append(f"BUY signal, low conviction")
            if rsi > 65:
                notes.append(f"RSI {rsi:.0f} nearing overbought")
            elif rsi < 35:
                notes.append(f"RSI {rsi:.0f} oversold")
            if mf == "UNFAVORABLE":
                notes.append("macro headwind")
            if not notes:
                notes.append(f"{abbr(mf)}")
            bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(tkr)}</span>', "left", ""),
                (signal_badge(sig), "center", ""),
                (f'<span style="color:{sentiment_color(fv)};font-weight:600;font-size:10px;">'
                 f'{abbr(fv)}({fs:.0f})</span>', "center", ""),
                (f'<span style="color:{sentiment_color(ts)};font-weight:600;font-size:10px;">'
                 f'{abbr(ts)}({rsi:.0f})</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;">{ex:.0f}%</span>', "center", ""),
                (conv_display(conv, delta_map.get(tkr)), "center", ""),
                (f'<span style="font-size:10px;color:{_C["text_muted"]};">{"; ".join(notes)}</span>', "left", ""),
            ], bg=bg))
        h.append('</table>')
    h.append(_section_close())

    # ── EPILOGUE: CHANGES SINCE LAST COMMITTEE ──
    real_changes = []
    for c in changes:
        prev_raw = c.get("prev_action", "?")
        curr = c.get("curr_action", "?")
        prev_norm = normalize_action(prev_raw)
        delta = c.get("delta", 0)
        if prev_norm == curr and abs(delta) <= 5:
            continue
        if prev_norm != curr:
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
            ct = "UPGRADE" if delta > 0 else "DOWNGRADE"
        real_changes.append({**c, "type": ct, "prev_norm": prev_norm})
    sig_changes = [c for c in real_changes
                   if abs(c.get("delta", 0)) > 5 or c.get("prev_norm") != c.get("curr_action")]
    if sig_changes:
        h.append(_section_open("Changes Since Last Committee",
                               f"{len(sig_changes)} significant changes.",
                               border="1px solid " + _C["border_heavy"]))
        h.append(_table_open([("Stock", "left"), ("Previous", "center"),
                              ("Current", "center"), ("&#9651;", "center"), ("Type", "left")]))
        for c in sig_changes[:15]:
            ct = c.get("type", "")
            if ct == "DOWNGRADE":
                rbg, rbrd, ar, tc = _C["bear_bg"], _C["bear_border"], "&#9660;", _C["bear_text"]
            elif ct == "UPGRADE":
                rbg, rbrd, ar, tc = _C["bull_bg"], _C["bull_border"], "&#9650;", _C["bull_text"]
            elif ct == "NEW":
                rbg, rbrd, ar, tc = _C["info_bg"], _C["info_border"], "&#9733;", _C["info"]
            else:
                rbg, rbrd, ar, tc = _C["bg_white"], _C["border"], "&middot;", _C["text_muted"]
            d = c.get("delta", 0)
            pn = c.get("prev_norm", normalize_action(c.get("prev_action", "?")))
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(c.get("ticker", ""))}</span>', "left", ""),
                (f'{e(pn)} ({c.get("prev_conviction", 0)})', "center", "font-size:11px;"),
                (f'{e(c.get("curr_action", "?"))} ({c.get("curr_conviction", 0)})', "center", "font-size:11px;"),
                (f'<span style="font-weight:700;color:{tc};">{d:+d}</span>', "center", ""),
                (f'{ar} <span style="color:{tc};font-weight:600;">{e(ct)}</span>', "left", ""),
            ], bg=rbg, border_color=rbrd))
        h.append('</table>')
        h.append(_section_close())

    # ── FOOTER ──
    h.append(f'<div style="padding:24px 40px;background:{_C["bg_page"]};border-top:1px solid {_C["border"]};">'
             f'<table style="width:100%;"><tr>'
             f'<td style="font-size:10px;color:{_C["text_light"]};line-height:1.5;">'
             f'<b style="color:{_C["text_muted"]};">Investment Committee v10.0</b><br/>'
             f'{today_long} &middot; 7 Agents (Sonnet) + CIO (Opus)</td>'
             f'<td style="text-align:right;font-size:9px;color:{_C["text_light"]};">Not financial advice.</td>'
             f'</tr></table></div>')
    h.append('</div></body></html>')

    return "\n".join(h)


# ═══════════════════════════════════════════════════════════════════════
# DISAGREEMENT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _detect_disagreements(concordance: List[Dict]) -> List[Dict]:
    """Detect genuine disagreements among specialist agents."""
    disagreements = []
    for entry in concordance:
        tkr = entry.get("ticker", "")
        sig = entry.get("signal", "?")
        fv = entry.get("fund_view", "")
        ts = entry.get("tech_signal", "")
        mf = entry.get("macro_fit", "")
        rw = entry.get("risk_warning", False)
        conv = entry.get("conviction", 0)
        act = entry.get("action", "")
        fs = entry.get("fund_score", 0)
        rsi = entry.get("rsi", 0)
        ex = entry.get("exret", 0)
        mom = entry.get("tech_momentum", 0)
        sec = entry.get("sector", "")

        if (fv == "BUY" and ts in ("AVOID", "EXIT_SOON")) or (fv == "SELL" and ts == "ENTER_NOW"):
            narr = (f"Fundamental analyst sees {tkr} at {fs:.0f}/100, EXRET {ex:.1f}%, signal {fv}. "
                    f"Technical strategist warns: RSI {rsi:.0f}, momentum {mom:+d}, timing signal {ts}.")
            if mf == "UNFAVORABLE":
                narr += f" Macro fit UNFAVORABLE for {sec}."
            disagreements.append({
                "ticker": tkr, "headline": f"Fund {fv} ({fs:.0f}) vs Tech {ts} (RSI {rsi:.0f})",
                "fund_view": fv, "tech_signal": ts, "narrative": narr,
                "resolution": act, "conviction": conv, "accent": action_color(act),
            })
        elif fv == "BUY" and fs >= 75 and rw and act in ("TRIM", "HOLD"):
            narr = (f"Fundamental analyst rates {tkr} strongly ({fs:.0f}/100, EXRET {ex:.1f}%), "
                    f"but Risk Manager flags concerns. Technical: RSI {rsi:.0f}, signal {ts}. "
                    f"Macro: {mf}. Result: {act} despite strong fundamentals.")
            disagreements.append({
                "ticker": tkr, "headline": f"Fund BUY ({fs:.0f}) vs Risk WARN",
                "fund_view": fv, "tech_signal": ts, "narrative": narr,
                "resolution": act, "conviction": conv, "accent": action_color(act),
            })
        elif fv == "BUY" and mf == "UNFAVORABLE" and act in ("ADD", "HOLD"):
            narr = (f"{tkr} has strong signal ({fv}, EXRET {ex:.1f}%) but macro fit is UNFAVORABLE "
                    f"for {sec}. Technical: RSI {rsi:.0f}, {ts}. Committee proceeded with {act} "
                    f"at reduced conviction ({conv}).")
            disagreements.append({
                "ticker": tkr, "headline": f"Signal BUY vs Macro UNFAVORABLE ({sec})",
                "fund_view": fv, "tech_signal": ts, "narrative": narr,
                "resolution": act, "conviction": conv, "accent": _C["warn"],
            })
        elif (sig == "H" and act in ("ADD", "BUY")) or (sig == "B" and act in ("HOLD", "TRIM")):
            override_dir = "upgraded" if act in ("ADD", "BUY") else "downgraded"
            narr = (f"Signal system rates {tkr} as {'HOLD' if sig == 'H' else 'BUY'} "
                    f"but committee {override_dir} to {act} (conviction {conv}). "
                    f"Fund: {fv} ({fs:.0f}), Tech: {ts} (RSI {rsi:.0f}), EXRET {ex:.1f}%, Macro: {mf}.")
            if sig == "H" and act in ("ADD", "BUY"):
                narr += f" Committee sees quality (fund {fs:.0f}) and agent consensus to override HOLD signal."
            elif sig == "B" and act in ("HOLD", "TRIM"):
                narr += (" Penalty stacking (regime, technicals, risk) reduced conviction "
                         "despite passing quant BUY criteria.")
            disagreements.append({
                "ticker": tkr, "headline": f"Signal {'H' if sig == 'H' else 'B'} overridden to {act}",
                "fund_view": fv, "tech_signal": ts, "narrative": narr,
                "resolution": act, "conviction": conv, "accent": _C["info"],
            })
        if len(disagreements) >= 6:
            break

    # Fallback: split votes
    if len(disagreements) < 3:
        for entry in concordance:
            bp = entry.get("bull_pct", 50)
            if 35 < bp < 65 and entry.get("action") not in ("HOLD",):
                tkr = entry.get("ticker", "")
                act = entry.get("action", "")
                disagreements.append({
                    "ticker": tkr,
                    "headline": (f"Split: {entry.get('bull_weight', 0):.1f} bull vs "
                                 f"{entry.get('bear_weight', 0):.1f} bear"),
                    "fund_view": entry.get("fund_view", ""),
                    "tech_signal": entry.get("tech_signal", ""),
                    "narrative": (f"Committee split on {tkr}. Fund: {entry.get('fund_view', '')} "
                                 f"({entry.get('fund_score', 0):.0f}), Tech: {entry.get('tech_signal', '')} "
                                 f"(RSI {entry.get('rsi', 0):.0f}), Macro: {entry.get('macro_fit', '')}. "
                                 f"Bull weight {entry.get('bull_weight', 0):.1f} vs "
                                 f"bear {entry.get('bear_weight', 0):.1f}."),
                    "resolution": act,
                    "conviction": entry.get("conviction", 0),
                    "accent": action_color(act),
                })
                if len(disagreements) >= 6:
                    break

    return disagreements


# ═══════════════════════════════════════════════════════════════════════
# FILE-BASED ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def generate_report_from_files(
    reports_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """Generate the HTML report from files on disk."""
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
    print(f"Report generated: {path}")

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
    if act is None:
        return None
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


def _group_separator(label, bg, text_color, border_color, colspan=11):
    """Consistent group separator row in tables."""
    return (f'<tr><td colspan="{colspan}" style="padding:4px 10px;background:{bg};'
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
    name_map: Optional[Dict[str, str]] = None,
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

    # --- Build name map from opps data if not provided ---
    _names: Dict[str, str] = dict(name_map or {})
    for opp in (opps or {}).get("top_opportunities", []):
        t = opp.get("ticker", "")
        if t and t not in _names and opp.get("name"):
            _names[t] = opp["name"]

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
            imp_str = f"{float(imp):.1f}" if isinstance(imp, (int, float)) else str(imp)
            h.append(f'<td style="width:33%;padding:10px;background:{bg};border:1px solid {brd};'
                     f'border-radius:6px;text-align:center;">'
                     f'<div style="{_LABEL}color:{lc};">{title}</div>'
                     f'<div style="font-size:18px;font-weight:800;color:{vc};margin-top:4px;">{imp_str}%</div></td>')
        h.append('</tr></table>')
    # Priority actions — pill format
    urgent = [en for en in concordance if en.get("action") in ("SELL", "TRIM")]
    top_buys = [en for en in concordance if en.get("action") in ("BUY", "ADD") and en.get("conviction", 0) >= 65][:5]
    if urgent or top_buys:
        h.append(f'<div style="{_LABEL}margin-bottom:8px;">Priority Actions</div>')
        h.append('<div style="line-height:2.4;">')
        for en in urgent + top_buys:
            act = en.get("action", "HOLD")
            tkr = en.get("ticker", "")
            ac = action_color(act)
            h.append(f'<span style="display:inline-block;padding:4px 12px;margin:2px 4px;border-radius:6px;'
                     f'font-size:11px;font-weight:700;background:{action_bg(act)};'
                     f'border:1px solid {action_border(act)};">'
                     f'<span style="color:{ac};">{act}</span> '
                     f'<span style="{_MONO}color:{_C["text_dark"]};">{e(tkr)}</span></span>')
        h.append('</div>')
    h.append(_section_close())

    # ── TRACK RECORD (CIO v17.0: Performance feedback loop) ──
    perf = synth.get("performance", {})
    if perf and perf.get("status") == "complete" and perf.get("total_evaluated", 0) > 0:
        h.append(_section_open("Track Record",
                               f"Performance since {e(perf.get('prev_committee_date', '?'))}",
                               border="1px solid " + _C["border"]))
        perf_actions = perf.get("actions", {})
        # KPI row: hit rates by action type
        perf_kpis = []
        for act_name in ("ADD", "HOLD", "TRIM", "SELL"):
            ad = perf_actions.get(act_name, {})
            if not ad:
                continue
            hr = ad.get("hit_rate", 0)
            cnt = ad.get("count", 0)
            avg_r = ad.get("avg_return", 0)
            if act_name in ("ADD", "BUY"):
                hr_col = _C["bull"] if hr >= 50 else _C["bear"]
            elif act_name in ("SELL", "TRIM"):
                hr_col = _C["bull"] if hr >= 50 else _C["bear"]
            else:
                hr_col = _C["hold"]
            perf_kpis.append((act_name, hr, cnt, avg_r, hr_col))

        if perf_kpis:
            h.append('<table style="width:100%;border-collapse:separate;border-spacing:8px 0;margin-bottom:16px;"><tr>')
            for act_name, hr, cnt, avg_r, hr_col in perf_kpis:
                ret_col = _C["bull"] if avg_r > 0 else _C["bear"] if avg_r < 0 else _C["text_muted"]
                h.append(f'<td style="padding:10px;background:{_C["bg_page"]};border:1px solid {_C["border"]};'
                         f'border-radius:6px;text-align:center;">'
                         f'<div style="{_LABEL}">{act_name} (n={cnt})</div>'
                         f'<div style="font-size:18px;font-weight:800;color:{hr_col};margin-top:4px;">{hr:.0f}%</div>'
                         f'<div style="font-size:10px;color:{ret_col};margin-top:2px;">avg {avg_r:+.1f}%</div></td>')
            h.append('</tr></table>')

        # Per-stock detail table (top 5 best + worst)
        all_details = []
        for act_name, ad in perf_actions.items():
            best = ad.get("best")
            worst = ad.get("worst")
            if best and isinstance(best, dict):
                all_details.append(best)
            if worst and isinstance(worst, dict) and worst != best:
                all_details.append(worst)
        if all_details:
            all_details.sort(key=lambda d: d.get("return_pct", 0), reverse=True)
            # Show top 3 winners and bottom 3 losers
            winners = [d for d in all_details if d.get("return_pct", 0) > 0][:3]
            losers = [d for d in all_details if d.get("return_pct", 0) < 0][-3:]
            show = winners + losers
            if show:
                h.append(_table_open([("Stock", "left"), ("Action", "center"),
                                      ("Conv", "center"), ("Return", "right")]))
                for d in show:
                    ret = d.get("return_pct", 0)
                    rc = _C["bull"] if ret > 0 else _C["bear"]
                    rbg = _C["bull_bg"] if ret > 2 else _C["bear_bg"] if ret < -2 else _C["bg_white"]
                    h.append(_table_row([
                        (f'<span style="{_MONO}font-weight:700;">{e(d.get("ticker", ""))}</span>', "left", ""),
                        (e(str(d.get("action", "?"))), "center", ""),
                        (str(d.get("conviction", "")), "center", _MONO),
                        (f'<span style="font-weight:700;color:{rc};">{ret:+.1f}%</span>', "right", ""),
                    ], bg=rbg))
                h.append('</table>')

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
            (f'<table style="margin:0 auto;border-collapse:collapse;"><tr>'
             f'<td style="padding:0 6px 0 0;vertical-align:middle;">{dot(dc)}</td>'
             f'<td style="padding:0;vertical-align:middle;font-size:11px;color:{_C["text_muted"]};'
             f'text-align:left;white-space:nowrap;">{e(st)}</td></tr></table>', "center", ""),
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
    # All stocks in one grid, grouped by 5 action types
    sell_list = [en for en in concordance if en.get("action") == "SELL"]
    trim_list = [en for en in concordance if en.get("action") == "TRIM"]
    buy_list = [en for en in concordance if en.get("action") == "BUY"]
    add_list = [en for en in concordance if en.get("action") == "ADD"]
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
    h.append(grid_hdr("#1e293b", "#fca5a5", "NEWS") + grid_hdr("#1e293b", "#d6d3d1", "RISK"))
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
        ni = entry.get("news_impact", "NEUTRAL")
        rw = entry.get("risk_warning", False)
        conv = entry.get("conviction", 0)
        sm = "*" if entry.get("fund_synthetic") else ""
        ex = entry.get("exret", 0)
        delta = delta_map.get(tkr)
        exc = _C["bull"] if ex > 5 else _C["bear"] if ex < 0 else _C["text_body"]
        rb = action_bg(act)
        p = f'style="padding:6px 4px;text-align:center;border:1px solid {_C["border"]};'
        sv = "font-weight:600;font-size:10px;"
        # NEWS cell
        ni_abbr = {"HIGH_POSITIVE": "H.POS", "HIGH_NEGATIVE": "H.NEG", "LOW_POSITIVE": "L.POS",
                   "LOW_NEGATIVE": "L.NEG", "POSITIVE": "POS", "NEGATIVE": "NEG",
                   "MIXED": "MIX", "NEUTRAL": "NEUT"}.get(ni, abbr(ni))
        # RISK cell
        risk_label = "WARN" if rw else "OK"
        risk_col = _C["bear"] if rw else _C["bull"]
        return (f'<tr style="background:{rb};">'
                f'<td {p}{_MONO}font-weight:700;font-size:11px;color:{_C["text_dark"]};text-align:left;padding-left:8px;">{e(tkr)}</td>'
                f'<td {p}">{signal_badge(sig)}</td>'
                f'<td {p}color:{sentiment_color(fv)};{sv}">{abbr(fv)}({fs:.0f}){sm}</td>'
                f'<td {p}color:{sentiment_color(ts)};{sv}">{abbr(ts)}({rsi:.0f})</td>'
                f'<td {p}color:{sentiment_color(mf)};{sv}">{abbr(mf)}</td>'
                f'<td {p}color:{sentiment_color(ce)};{sv}">{abbr(ce)}</td>'
                f'<td {p}color:{sentiment_color(ni)};{sv}">{ni_abbr}</td>'
                f'<td {p}color:{risk_col};{sv}">{risk_label}</td>'
                f'<td {p}{_MONO}font-weight:700;color:{exc};">{ex:.0f}%</td>'
                f'<td {p}">{badge(act, action_color(act), "#fff")}</td>'
                f'<td {p}">{conv_display(conv, delta)}</td></tr>')

    if sell_list:
        h.append(_group_separator(f"SELL ({len(sell_list)})", _C["bear_bg"], _C["bear_text"], _C["bear_border"]))
        for en in sell_list:
            h.append(_grid_row(en))
    if trim_list:
        h.append(_group_separator(f"TRIM ({len(trim_list)})", _C["warn_bg"], _C["warn_text"], _C["warn_border"]))
        for en in trim_list:
            h.append(_grid_row(en))
    if buy_list:
        h.append(_group_separator(f"BUY ({len(buy_list)})", _C["bull_bg"], _C["bull_text"], _C["bull_border"]))
        for en in buy_list:
            h.append(_grid_row(en))
    if add_list:
        h.append(_group_separator(f"ADD ({len(add_list)})", "#ecfdf5", "#0d9488", "#99f6e4"))
        for en in add_list:
            h.append(_grid_row(en))
    if hold_list:
        h.append(_group_separator(f"HOLD ({len(hold_list)})", _C["bg_alt"], "#475569", _C["border_heavy"]))
        for en in hold_list:
            h.append(_grid_row(en))
    h.append('</table></div>')
    # Legend
    h.append(f'<div style="margin-top:10px;padding:8px 12px;background:{_C["bg_page"]};'
             f'border:1px solid {_C["border"]};border-radius:6px;font-size:9px;color:{_C["text_muted"]};">'
             f'* = Synthetic &middot; ENTER/EXIT/WAIT = Tech timing &middot; '
             f'FAVOR/UNFAV = Macro fit &middot; ALIGN/DIVERG = Census &middot; '
             f'&#9650;&#9660; = Conviction change &middot; '
             f'ACCEL/DETER = Signal velocity &middot; SERIA/BEAT/MISS = Earnings surprise</div>')
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

    # ── S5: FUNDAMENTAL DEEP DIVE ──
    fund_stocks = fund.get("stocks", {})
    quality_traps = fund.get("quality_traps", [])
    if fund_stocks:
        h.append(_section_open("Fundamental Deep Dive",
                               f"{len(fund_stocks)} stocks analyzed. Top scores and quality trap warnings."))
        # Top fundamental scores table
        h.append(f'<div style="{_LABEL}margin-bottom:8px;">Top Fundamental Scores</div>')
        h.append(_table_open([("Stock", "left"), ("Score", "center"), ("Quality", "center"),
                              ("PE T&#8594;F", "center"), ("EXRET", "center"),
                              ("Insider", "center"), ("Key Insight", "left")]))
        sorted_fund = sorted(fund_stocks.items(), key=lambda x: -x[1].get("fundamental_score", 0))
        for i, (tkr, fd) in enumerate(sorted_fund[:12]):
            fs = fd.get("fundamental_score", 0)
            eq = fd.get("earnings_quality", "?")
            pet = fd.get("pe_trailing", fd.get("pet", 0))
            pef = fd.get("pe_forward", fd.get("pef", 0))
            ex = fd.get("exret", fd.get("upside_pct", 0))
            ins = fd.get("insider_sentiment", "N/A")
            note = fd.get("notes", "")[:50]
            fs_col = _C["bull"] if fs >= 80 else _C["warn"] if fs >= 60 else _C["text_muted"]
            ins_col = _C["bull"] if "BUY" in str(ins) else _C["bear"] if "SELL" in str(ins) else _C["text_muted"]
            bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            pe_str = f"{pet:.0f}&#8594;{pef:.0f}" if pet and pef else "N/A"
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(tkr)}</span>', "left", ""),
                (f'<span style="font-weight:800;color:{fs_col};">{fs:.0f}</span>', "center", ""),
                (f'<span style="font-size:10px;">{e(eq)}</span>', "center", ""),
                (f'<span style="{_MONO}font-size:10px;">{pe_str}</span>', "center", ""),
                (f'{ex:.0f}%', "center", f"{_MONO}font-weight:600;"),
                (f'<span style="color:{ins_col};font-size:10px;">{e(str(ins)[:12])}</span>', "center", ""),
                (f'<span style="font-size:10px;color:{_C["text_muted"]};">{e(note)}</span>', "left", ""),
            ], bg=bg))
        h.append('</table>')

        # Quality traps
        trap_stocks = [t for t, d in fund_stocks.items() if d.get("quality_trap_warning")]
        if trap_stocks:
            h.append(f'<div style="{_LABEL}color:{_C["bear"]};margin:16px 0 8px 0;">'
                     f'Quality Traps ({len(trap_stocks)})</div>')
            for tkr in trap_stocks[:5]:
                fd = fund_stocks[tkr]
                reasons = fd.get("quality_trap_reasons", [])
                reason_text = "; ".join(reasons[:2]) if reasons else "Metrics diverge from fundamentals"
                inner = (f'<span style="{_MONO}font-weight:700;color:{_C["bear_text"]};">{e(tkr)}</span>'
                         f' <span style="font-size:11px;color:{_C["text_body"]};">{e(reason_text)}</span>')
                h.append(_card(inner, _C["bear"], _C["bear_bg"]))
        h.append(_section_close())

    # ── S6: TECHNICAL ANALYSIS ──
    if tech_stocks:
        h.append(_section_open("Technical Analysis",
                               f"{total_tech} stocks. Avg RSI: {avg_rsi:.0f}. "
                               f"{bearish_macd}/{total_tech} bearish MACD."))
        # Table of technical signals
        h.append(_table_open([("Stock", "left"), ("RSI", "center"), ("MACD", "center"),
                              ("BB%", "center"), ("Trend", "center"), ("Mom", "center"),
                              ("Signal", "center")]))
        sorted_tech = sorted(tech_stocks.items(),
                             key=lambda x: x[1].get("momentum_score", 0))
        for i, (tkr, td) in enumerate(sorted_tech[:20]):
            rsi_v = td.get("rsi", 50)
            macd = td.get("macd_signal", "?")
            bb = td.get("bb_position", 0.5)
            trend = td.get("trend", "?")
            mom = td.get("momentum_score", 0)
            sig = td.get("timing_signal", "?")
            rsi_col = _C["bull"] if rsi_v < 30 else _C["bear"] if rsi_v > 70 else _C["text_body"]
            macd_col = _C["bull"] if macd == "BULLISH" else _C["bear"]
            mom_col = _C["bull"] if mom > 20 else _C["bear"] if mom < -20 else _C["text_muted"]
            sig_col = sentiment_color(sig)
            bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            trend_short = {"STRONG_DOWNTREND": "STR DN", "WEAK_DOWNTREND": "WK DN",
                           "CONSOLIDATION": "CONSOL", "WEAK_UPTREND": "WK UP",
                           "STRONG_UPTREND": "STR UP"}.get(trend, trend[:6])
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(tkr)}</span>', "left", ""),
                (f'<span style="{_MONO}font-weight:700;color:{rsi_col};">{rsi_v:.0f}</span>', "center", ""),
                (f'<span style="color:{macd_col};font-size:10px;font-weight:600;">{macd[:4]}</span>', "center", ""),
                (f'<span style="{_MONO}font-size:10px;">{bb:.2f}</span>', "center", ""),
                (f'<span style="font-size:10px;">{trend_short}</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;color:{mom_col};">{int(mom):+d}</span>', "center", ""),
                (f'<span style="color:{sig_col};font-weight:600;font-size:10px;">{abbr(sig)}</span>', "center", ""),
            ], bg=bg))
        h.append('</table>')
        # Oversold / Overbought callouts
        oversold = [t for t, d in tech_stocks.items() if d.get("rsi", 50) < 30]
        overbought = [t for t, d in tech_stocks.items() if d.get("rsi", 50) > 70]
        if oversold:
            pills = " ".join(f'<span style="display:inline-block;padding:2px 8px;margin:2px;border-radius:4px;'
                             f'{_MONO}font-size:10px;font-weight:700;background:{_C["bull_bg"]};'
                             f'color:{_C["bull_text"]};border:1px solid {_C["bull_border"]};">{e(t)}</span>'
                             for t in oversold)
            h.append(f'<div style="margin-top:12px;"><span style="{_LABEL}color:{_C["bull"]};">Oversold (RSI&lt;30): </span>{pills}</div>')
        if overbought:
            pills = " ".join(f'<span style="display:inline-block;padding:2px 8px;margin:2px;border-radius:4px;'
                             f'{_MONO}font-size:10px;font-weight:700;background:{_C["bear_bg"]};'
                             f'color:{_C["bear_text"]};border:1px solid {_C["bear_border"]};">{e(t)}</span>'
                             for t in overbought)
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">Overbought (RSI&gt;70): </span>{pills}</div>')
        h.append(_section_close())

    # ── S7: SENTIMENT & CENSUS ──
    fg_label = lambda v: ("EXTREME GREED" if v >= 75 else "GREED" if v >= 55 else "NEUTRAL" if v >= 45
                          else "FEAR" if v >= 25 else "EXTREME FEAR")
    fg_color = lambda v: (_C["bear"] if v >= 75 else _C["warn"] if v >= 55 else _C["text_muted"] if v >= 45
                          else _C["info"] if v >= 25 else _C["hold"])
    cash100 = census.get("sentiment", {}).get("cash_top100") or 0
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
        affected_sectors = item.get("affected_sectors", [])
        affected_tickers = item.get("affected_tickers", [])
        source = item.get("source", "")
        if "NEGATIVE" in str(imp):
            nbg, nbrd, nlc = _C["bear_bg"], _C["bear"], _C["bear_text"]
            imp_color = _C["bear"]
        elif "POSITIVE" in str(imp):
            nbg, nbrd, nlc = _C["bull_bg"], _C["bull"], _C["bull_text"]
            imp_color = _C["bull"]
        else:
            nbg, nbrd, nlc = _C["bg_page"], _C["text_light"], _C["text_body"]
            imp_color = _C["text_muted"]
        # Build structured detail line
        detail_cells = []
        detail_cells.append(f'<span style="color:{imp_color};font-weight:600;">{e(imp)}</span>')
        if affected_sectors:
            detail_cells.append(e(", ".join(affected_sectors[:4])))
        if affected_tickers:
            tkr_pills = " ".join(
                f'<span style="display:inline-block;padding:1px 5px;border-radius:3px;'
                f'background:{_C["bg_alt"]};{_MONO}font-size:9px;font-weight:600;">{e(t)}</span>'
                for t in affected_tickers[:5])
            detail_cells.append(tkr_pills)
        if source:
            detail_cells.append(f'<span style="font-style:italic;">{e(source)}</span>')
        detail_line = (f'<div style="font-size:10px;color:{_C["text_muted"]};margin-top:5px;'
                       f'line-height:1.6;">'
                       + " &middot; ".join(detail_cells) + '</div>')
        h.append(f'<div style="background:{nbg};border-left:3px solid {nbrd};border-radius:0 4px 4px 0;'
                 f'padding:10px 14px;margin-bottom:8px;">'
                 f'<div style="font-size:12px;font-weight:600;color:{nlc};">'
                 f'{e(hl)}</div>{detail_line}</div>')
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
    # Earnings calendar
    earnings_cal = synth.get("earnings_calendar", {})
    earnings_items = earnings_cal.get("next_2_weeks", [])
    if earnings_items:
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Upcoming Earnings</div>')
        h.append(_table_open([("Stock", "left"), ("Date", "center"), ("Days", "center"), ("Risk", "center")]))
        for i, ear in enumerate(sorted(earnings_items, key=lambda x: x.get("days_away", 999))):
            days = ear.get("days_away", 0)
            rl = ear.get("risk_level", "LOW")
            rl_col = _C["bear"] if rl == "HIGH" else _C["warn"] if rl == "MEDIUM" else _C["text_muted"]
            rbg = _C["bear_bg"] if days <= 7 else _C["warn_bg"] if days <= 14 else _C["bg_white"]
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(ear.get("ticker", ""))}</span>', "left", ""),
                (e(ear.get("expected_date", "?")), "center", f"font-size:11px;{_MONO}"),
                (f'<span style="font-weight:700;color:{rl_col};">{days}d</span>', "center", ""),
                (badge(rl, rl_col, "#fff"), "center", ""),
            ], bg=rbg))
        h.append('</table>')
    # Economic events
    econ_events = synth.get("economic_events", [])
    if econ_events:
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Economic Calendar</div>')
        for ev in econ_events[:5]:
            imp = ev.get("importance", "LOW")
            if imp == "CRITICAL":
                ebg, ebrd, elc = _C["bear_bg"], _C["bear"], _C["bear_text"]
            elif imp == "HIGH":
                ebg, ebrd, elc = _C["warn_bg"], _C["warn"], _C["warn_text"]
            else:
                ebg, ebrd, elc = _C["bg_page"], _C["text_light"], _C["text_body"]
            h.append(f'<div style="background:{ebg};border-left:3px solid {ebrd};'
                     f'border-radius:0 4px 4px 0;padding:8px 14px;margin-bottom:6px;">'
                     f'<span style="{_MONO}font-size:10px;color:{_C["text_muted"]};">{e(ev.get("date", ""))}</span> '
                     f'<span style="font-size:12px;font-weight:600;color:{elc};">{e(ev.get("event", "")[:80])}</span>'
                     f'<div style="font-size:10px;color:{_C["text_muted"]};margin-top:2px;">'
                     f'{e(ev.get("sector_impact", "")[:80])}</div></div>')
    h.append(_section_close())

    # ── S9: NEW OPPORTUNITIES ──
    opp_list = opps.get("top_opportunities", [])
    opp_stats = opps.get("screening_stats", {})
    if opp_list:
        h.append(_section_open("New Opportunities",
                               f"Screened {opp_stats.get('universe_size', '?')} stocks. "
                               f"{opp_stats.get('unique_candidates', '?')} passed filters."))
        h.append(_table_open([("#", "center"), ("Stock", "left"), ("Sector", "left"),
                              ("Score", "center"), ("EXRET", "center"), ("PE F", "center"),
                              ("%BUY", "center"), ("Why Compelling", "left")]))
        for i, opp in enumerate(opp_list[:10]):
            score = opp.get("opportunity_score", 0)
            sc_col = _C["bull"] if score >= 60 else _C["warn"] if score >= 40 else _C["text_muted"]
            bg = _C["bull_bg"] if i < 3 else _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            exr = str(opp.get("exret", "0")).replace("%", "")
            pef = opp.get("pe_forward", 0)
            bp = opp.get("buy_pct", 0)
            why = opp.get("why_compelling", "")[:60]
            h.append(_table_row([
                (f'<span style="font-weight:700;color:{_C["bull"]};">{i + 1}</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;">{e(opp.get("ticker", ""))}</span>', "left", ""),
                (f'<span style="font-size:10px;">{e(opp.get("sector", "")[:15])}</span>', "left", ""),
                (f'<span style="{_MONO}font-weight:700;color:{sc_col};">{score:.0f}</span>', "center", ""),
                (f'{exr}%', "center", f"{_MONO}font-weight:600;"),
                (f'{pef:.1f}' if pef else "N/A", "center", _MONO),
                (f'{bp:.0f}%', "center", ""),
                (f'<span style="font-size:10px;color:{_C["text_muted"]};">{e(why)}</span>', "left", ""),
            ], bg=bg))
        h.append('</table>')

        # Sector gaps from synthesis
        if sector_gaps:
            h.append(f'<div style="{_LABEL}color:{_C["warn"]};margin:16px 0 8px 0;">'
                     f'Portfolio Sector Gaps</div>')
            for gap in sector_gaps[:4]:
                sec = gap.get("sector", gap.get("etf", "?"))
                assessment = gap.get("gap", gap.get("assessment", gap.get("note", "")))
                inner = f'<b>{e(str(sec))}</b>: {e(str(assessment)[:100])}'
                h.append(_card(inner, _C["warn"], _C["warn_bg"]))

        h.append(_section_close())

    # ── S10: RISK DASHBOARD ──
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
    # Correlation clusters — handle both grouped and pairwise formats
    # Prefer risk agent's grouped clusters; fall back to synthesis pairwise data
    risk_clusters = risk.get("correlation_clusters", [])
    display_clusters = []
    if risk_clusters and isinstance(risk_clusters, list) and risk_clusters[0].get("stocks"):
        # Grouped format from risk agent: {stocks: [...], avg_correlation: X, risk: "..."}
        display_clusters = risk_clusters
    elif clusters:
        # Pairwise format from synthesis: {stock1, stock2, correlation}
        # Group pairwise into clusters by finding connected components
        from collections import defaultdict
        adj = defaultdict(list)
        pair_corr = {}
        for cl in clusters:
            s1, s2 = cl.get("stock1", ""), cl.get("stock2", "")
            corr = cl.get("correlation", cl.get("avg_correlation", 0))
            if s1 and s2 and corr > 0.7:
                adj[s1].append(s2)
                adj[s2].append(s1)
                pair_corr[(s1, s2)] = corr
                pair_corr[(s2, s1)] = corr
        visited = set()
        for node in adj:
            if node in visited:
                continue
            # BFS to find cluster
            cluster_stocks = []
            queue = [node]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                cluster_stocks.append(n)
                for nb in adj[n]:
                    if nb not in visited:
                        queue.append(nb)
            if len(cluster_stocks) >= 2:
                # Compute average correlation
                corrs = []
                for i_c in range(len(cluster_stocks)):
                    for j_c in range(i_c + 1, len(cluster_stocks)):
                        key = (cluster_stocks[i_c], cluster_stocks[j_c])
                        if key in pair_corr:
                            corrs.append(pair_corr[key])
                avg_c = sum(corrs) / len(corrs) if corrs else 0
                display_clusters.append({
                    "stocks": cluster_stocks[:5],
                    "avg_correlation": avg_c,
                    "risk": f"{len(cluster_stocks)} correlated positions — false diversification risk",
                })
        display_clusters.sort(key=lambda x: -x.get("avg_correlation", 0))

    if display_clusters:
        h.append(f'<div style="{_LABEL}color:{_C["bear"]};margin:16px 0 8px 0;">Correlation Clusters</div>')
        h.append(f'<table style="{_TABLE}">')
        h.append(f'<tr style="border-bottom:2px solid {_C["border"]};">'
                 f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Cluster</th>'
                 f'<th style="padding:6px 10px;text-align:center;{_LABEL}width:60px;">Corr</th>'
                 f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Risk Note</th></tr>')
        for i, cl in enumerate(display_clusters[:5]):
            stks = cl.get("stocks", cl.get("tickers", []))
            ac = cl.get("avg_correlation", cl.get("average_correlation", 0))
            rn = cl.get("risk", "Correlated positions")
            bg = _C["bear_bg"] if i % 2 == 0 else _C["bg_white"]
            bar_w = min(ac * 100, 100)
            pills = " ".join(
                f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
                f'{_MONO}font-size:10px;font-weight:700;background:{_C["bg_alt"]};">{e(s)}</span>'
                for s in stks[:5])
            h.append(f'<tr style="background:{bg};">'
                     f'<td style="padding:6px 10px;">{pills}</td>'
                     f'<td style="padding:6px 10px;text-align:center;">'
                     f'<span style="{_MONO}font-weight:700;color:{_C["bear"]};">{ac:.2f}</span>'
                     f'<div style="height:3px;background:{_C["border"]};border-radius:2px;margin-top:3px;">'
                     f'<div style="height:100%;border-radius:2px;background:{_C["bear"]};'
                     f'width:{bar_w:.0f}%;"></div></div></td>'
                     f'<td style="padding:6px 10px;font-size:11px;color:{_C["text_muted"]};">'
                     f'{e(str(rn)[:80])}</td></tr>')
        h.append('</table>')
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
    sell_items = [en for en in concordance if en.get("action") == "SELL"]
    trim_items = [en for en in concordance if en.get("action") == "TRIM"]
    # CIO v12.0 P2: Sort BUY/ADD by conviction, then capital efficiency as
    # tiebreaker. CE answers "which stock deserves the marginal dollar?"
    buy_items = sorted([en for en in concordance if en.get("action") == "BUY"],
                       key=lambda x: (-x.get("conviction", 0), -x.get("capital_efficiency", 0)))
    add_items = sorted([en for en in concordance if en.get("action") == "ADD"],
                       key=lambda x: (-x.get("conviction", 0), -x.get("capital_efficiency", 0)))
    monitor_items = sorted([en for en in concordance if en.get("action") == "HOLD"],
                           key=lambda x: -x.get("conviction", 0))

    h.append(_section_open("Action Items",
                           "Tiered by urgency. Kill thesis: what makes each trade fail.",
                           border="2px solid " + _C["text_dark"]))

    # Helper for action item cards
    def _action_card(en, label_color):
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
        mp = en.get("max_pct", pos_limits.get(tkr, {}).get("max_pct", 5.0))
        ce = en.get("capital_efficiency", 0)
        kill = en.get("kill_thesis") or _stock_kill_thesis(act, tkr, sec, rsi, ex, beta, bp, mf, ts, fv)
        # CIO v11.0: Surface velocity and earnings indicators when populated
        vel = en.get("signal_velocity", "NO_HISTORY")
        earn = en.get("earnings_surprise", "NO_DATA")
        extra_tags = ""
        if vel not in ("NO_HISTORY", "STABLE", ""):
            vc = _C["bull"] if vel in ("ACCELERATING", "IMPROVING") else _C["bear"]
            extra_tags += f' | <span style="color:{vc};">{vel[:5]}</span>'
        if earn not in ("NO_DATA", "IN_LINE", ""):
            ec = _C["bull"] if earn in ("SERIAL_BEATER", "BEAT") else _C["bear"]
            extra_tags += f' | <span style="color:{ec};">{earn[:6]}</span>'
        inner = (f'<table style="width:100%;"><tr><td>'
                 f'{badge(act, action_color(act), "#fff")} '
                 f'<span style="{_MONO}font-weight:800;font-size:14px;margin-left:8px;">{e(tkr)}</span> '
                 f'<span style="font-size:11px;color:{_C["text_muted"]};margin-left:8px;">'
                 f'{sec} | RSI {rsi:.0f} | {abbr(ts)} | {abbr(mf)}'
                 f'{" | Max " + str(int(mp)) + "%" if act in ("BUY","ADD") else ""}'
                 f'{" | $" + str(int(en.get("suggested_size_usd", 0))) if en.get("suggested_size_usd") and act in ("BUY","ADD") else ""}'
                 f'{" | CE " + f"{ce:.1f}" if ce and act in ("BUY","ADD") else ""}'
                 f'{extra_tags}</span></td>'
                 f'<td style="text-align:right;">{conv_display(conv)}</td></tr></table>'
                 f'<div style="font-size:11px;color:{_C["text_muted"]};font-style:italic;'
                 f'margin-top:8px;line-height:1.5;">{e(kill)}</div>')
        h.append(_card(inner, action_color(act), action_bg(act)))

    # SELL
    if sell_items:
        h.append(f'<div style="{_LABEL}color:{_C["bear_text"]};margin-bottom:10px;">'
                 f'Sell ({len(sell_items)})</div>')
        for en in sell_items:
            _action_card(en, _C["bear_text"])

    # TRIM
    if trim_items:
        h.append(f'<div style="{_LABEL}color:{_C["warn_text"]};margin:20px 0 10px 0;">'
                 f'Trim ({len(trim_items)})</div>')
        for en in trim_items:
            _action_card(en, _C["warn_text"])

    # BUY
    if buy_items:
        h.append(f'<div style="{_LABEL}color:{_C["bull_text"]};margin:20px 0 10px 0;">'
                 f'Buy ({len(buy_items)})</div>')
        for en in buy_items:
            _action_card(en, _C["bull_text"])

    # ADD
    if add_items:
        h.append(f'<div style="{_LABEL}color:#0d9488;margin:20px 0 10px 0;">'
                 f'Add ({len(add_items)})</div>')
        for en in add_items:
            _action_card(en, "#0d9488")

    # Monitor: HOLD — compact pill list
    if monitor_items:
        h.append(f'<div style="{_LABEL}margin:20px 0 10px 0;">Monitor ({len(monitor_items)})</div>')
        h.append('<div style="line-height:2.2;">')
        for en in monitor_items:
            tkr = en.get("ticker", "")
            conv = en.get("conviction", 0)
            cc = conv_color(conv)
            h.append(f'<span style="display:inline-block;padding:3px 10px;margin:2px 3px;border-radius:4px;'
                     f'{_MONO}font-size:11px;font-weight:700;background:{_C["bg_alt"]};'
                     f'border:1px solid {_C["border"]};">{e(tkr)}'
                     f'<span style="color:{cc};font-size:10px;margin-left:4px;">{conv}</span></span>')
        h.append('</div>')
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
            prev_conv = c.get("prev_conviction")
            prev_label = f'{e(pn or "NEW")} ({prev_conv if prev_conv is not None else "—"})'
            h.append(_table_row([
                (f'<span style="{_MONO}font-weight:700;">{e(c.get("ticker", ""))}</span>', "left", ""),
                (prev_label, "center", "font-size:11px;"),
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
             f'<b style="color:{_C["text_muted"]};">Investment Committee v17.0</b><br/>'
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
                    f"Technical strategist warns: RSI {rsi:.0f}, momentum {mom:+.0f}, timing signal {ts}.")
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

    # Build name map from portfolio + etoro CSVs
    name_map: Dict[str, str] = {}
    for csv_name in ("portfolio.csv", "etoro.csv"):
        csv_path = Path.home() / "SourceCode" / "etorotrade" / "yahoofinance" / "output" / csv_name
        if csv_path.exists():
            try:
                import csv as csv_mod
                with open(csv_path) as cf:
                    for row in csv_mod.DictReader(cf):
                        t = row.get("TKR", "")
                        n = row.get("NAME", "")
                        if t and n and t not in name_map:
                            name_map[t] = n
            except Exception:
                pass

    html_str = generate_report_html(synth, fund, tech, macro, census, news, opps, risk,
                                    name_map=name_map)

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = od / f"{today}.html"
    od.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_str)

    # Archive concordance for backtesting (dated copy)
    _archive_concordance(synth, od, today)

    return str(output_path)


def _archive_concordance(
    synth: Dict[str, Any],
    output_dir: Path,
    date_str: str,
) -> None:
    """Save a dated concordance snapshot for backtesting history."""
    concordance = synth.get("concordance", [])
    if not concordance:
        return
    archive_dir = output_dir / "history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"concordance-{date_str}.json"
    archive_data = {
        "date": date_str,
        "version": "v10.0",
        "regime": synth.get("regime", "UNKNOWN"),
        "concordance": [
            {
                "ticker": en.get("ticker", ""),
                "signal": en.get("signal", "?"),
                "action": en.get("action", "HOLD"),
                "conviction": en.get("conviction", 0),
                "fund_score": en.get("fund_score", 0),
                "fund_view": en.get("fund_view", "?"),
                "tech_signal": en.get("tech_signal", "?"),
                "rsi": en.get("rsi", 0),
                "macro_fit": en.get("macro_fit", "?"),
                "census": en.get("census", "?"),
                "exret": en.get("exret", 0),
                "excess_exret": en.get("excess_exret", 0),
                "beta": en.get("beta", 1.0),
                "sector": en.get("sector", ""),
                "bull_pct": en.get("bull_pct", 50),
                "bull_weight": en.get("bull_weight", 2.5),
                "bear_weight": en.get("bear_weight", 2.5),
                "bonuses": en.get("bonuses", 0),
                "penalties": en.get("penalties", 0),
                "fund_synthetic": en.get("fund_synthetic", False),
                "tech_synthetic": en.get("tech_synthetic", False),
                "is_opportunity": en.get("is_opportunity", False),
                "price": en.get("price", 0),
            }
            for en in concordance
        ],
    }
    try:
        with open(archive_path, "w") as f:
            json.dump(archive_data, f, indent=2)
    except OSError:
        pass


if __name__ == "__main__":
    path = generate_report_from_files()
    print(f"Report generated: {path}")

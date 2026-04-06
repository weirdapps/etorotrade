"""
Committee HTML Report Generator

CIO v24.0: Three-Act report structure with ~85 codified conviction modifiers across 14+ series.
18 report sections (full) / 9 sections (daily digest).
Includes: conviction decay, entry timing, position sizing, census contrarian, regional calibration,
Monte Carlo VaR (portfolio + per-stock), factor exposure decomposition, correlation regime tracking,
RRG sector rotation, multi-timeframe confluence, IV rank, volatility regime, FCF/debt quality,
EPS revisions, ADX trend strength, RSI divergence, Piotroski F-Score, ATR-based stop-losses,
currency risk (EUR home), liquidity scoring, self-calibration infrastructure, benchmark comparison.
Mobile-responsive CSS for email readability.

Three-Act Structure:
  ACT I: DECISIONS — Executive Summary, Action Items, Currency Exposure
  ACT II: ANALYSIS — Macro, Stock Grid, Disagreements, Portfolio Risk, Portfolio Construction
  ACT III: DEEP CONTEXT — Fundamentals, Technicals, Sentiment, News, Opportunities (full only)
  EPILOGUE — Watchlist, Changes, Track Record, Report Card, Regime Transition

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
    "accent_blue": "#3b82f6",
}

# --- Spacing tokens ---
_SECTION = "padding:32px 40px;border-bottom:1px solid #e2e8f0;"
_SECTION_H2 = "margin:0 0 16px 0;font-size:18px;font-weight:700;color:#0f172a;letter-spacing:-0.3px;"
_SECTION_H2_GRADIENT = "margin:0 0 16px 0;font-size:18px;font-weight:700;color:#ffffff;letter-spacing:-0.3px;background:linear-gradient(135deg, #0f172a 0%, #1e293b 100%);padding:12px 20px;border-radius:6px;"
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

def _clean_name(raw):
    """Clean up verbose yfinance names: strip suffixes like 'Inc.', 'Corp.', 'PLC ORD 5P'."""
    import re
    if not raw:
        return raw
    # Remove share class / listing suffixes
    raw = re.sub(r'\s+ORD\s+\d+P$', '', raw, flags=re.I)
    raw = re.sub(r'\s+CLASS\s+[A-C]$', '', raw, flags=re.I)
    raw = re.sub(r'\s+CL\s+[A-C]$', '', raw, flags=re.I)
    # Remove common suffixes for brevity
    for suffix in [', Inc.', ' Inc.', ', Inc', ' Inc', ' Corporation', ' Corp.',
                   ' Corp', ' Holdings', ' Group', ' Ltd.', ' Ltd', ' PLC',
                   ' plc', ' SE', ' S.A.', ' N.V.', '.com']:
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
            break
    return raw.strip()

def _tn(ticker, names, max_name=22):
    """Ticker with company name: 'Name (TKR)' or just 'TKR' if name unknown."""
    n = _clean_name(names.get(ticker, ""))
    if not n:
        return ticker
    if len(n) > max_name:
        n = n[:max_name - 1] + "\u2026"
    return f"{n} ({ticker})"

def _tn_cell(ticker, names):
    """Two-line HTML for table cells: company name above ticker."""
    n = _clean_name(names.get(ticker, ""))
    if not n:
        return f'<span style="{_MONO}font-weight:700;font-size:11px;">{e(ticker)}</span>'
    if len(n) > 20:
        n = n[:19] + "\u2026"
    return (f'<div style="font-size:9px;color:{_C["text_muted"]};line-height:1.2;'
            f'white-space:nowrap;">{e(n)}</div>'
            f'<div style="{_MONO}font-weight:700;font-size:11px;color:{_C["text_dark"]};">'
            f'{e(ticker)}</div>')

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
    "Consumer Defensive": "Consumer Staples", "Consumer Defense": "Consumer Staples",
    "Basic Materials": "Materials", "Gold": "Materials", "Precious Metals": "Materials",
    "Gold Mining": "Materials", "Copper Mining": "Materials", "Mining": "Materials",
    "Financial Servi": "Financials", "Financial Services": "Financials",
    "Financial Infrastructure": "Financials", "FinTech": "Financials",
    "Consumer Discreti": "Consumer Disc.", "Gaming": "Consumer Disc.",
    "Telecom": "Comm. Services",
    "Pharma": "Healthcare", "Pharma/GLP-1": "Healthcare", "Life Sciences": "Healthcare",
    "Medical Devices": "Healthcare",
    "Aerospace": "Industrials", "Tech Services": "Industrials",
    "Enterprise Software": "Technology", "Tech/Cloud": "Technology",
    "AI/Defense": "Technology", "Tech/Consumer": "Technology",
    "Tech/Advertising": "Technology", "Tech/Social": "Technology",
    "Tech/Entertainment": "Comm. Services", "Tech/Gaming": "Comm. Services",
    "Brokerage": "Financials", "Payments": "Financials", "Financial Data": "Financials",
    "Beverages": "Consumer Staples", "Retail": "Consumer Staples",
    "Entertainment": "Comm. Services",
    "Gold ETF": "Materials", "Bitcoin Proxy": "Crypto/Alt",
    "ETF/Greece": "ETF",
    "Unknown": "Other", "": "Other",
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
        elif beta > 1.5 and macro_fit == "UNFAVORABLE":
            parts.append(f"Beta {beta:.1f} + unfavorable macro for {sec} = amplified downside risk")
        elif beta > 1.5:
            parts.append(f"Beta {beta:.1f} amplifies drawdowns; broad selloff could erase entry")
        elif macro_fit == "UNFAVORABLE":
            parts.append(f"Buying into {sec} macro headwind; fails if regime worsens further")
        elif buy_pct < 60:
            parts.append(f"Only {buy_pct:.0f}% analyst BUY consensus -- weak conviction risks thesis collapse on any negative catalyst")
        elif exret > 40:
            parts.append(f"Exceptional {exret:.0f}% EXRET likely stale; fails if estimates are cut or sector de-rates")
        elif "Technology" in sec or "Semicon" in sec:
            parts.append(f"Fails if AI/tech capex cycle peaks or {tkr} guides below on next earnings")
        elif "Financial" in sec or "Bank" in sec:
            parts.append(f"Fails if credit cycle turns; watch NIM compression or loan loss provision spike at {tkr}")
        elif "Healthcare" in sec or "Pharma" in sec:
            parts.append(f"Fails if pipeline setback, regulatory block, or pricing pressure hits {tkr}")
        elif "Energy" in sec:
            parts.append(f"Fails if oil/commodity prices reverse sharply or demand outlook deteriorates")
        else:
            parts.append(f"Fails if {exret:.0f}% EXRET target is cut, {sec} sector de-rates, or earnings disappoint")
        if rsi > 60:
            parts.append(f"RSI {rsi:.0f} not ideal entry; pullback to ~45 would improve risk/reward")
        elif rsi < 25 and tech_sig in ("AVOID", "EXIT_SOON"):
            parts.append(f"RSI {rsi:.0f} oversold but technicals say {tech_sig} -- momentum trap risk")
        return ". ".join(parts) + "."
    if act == "ADD":
        parts = []
        if rsi < 35:
            parts.append(f"Adding at RSI {rsi:.0f} (oversold) but momentum could deteriorate further")
        elif tech_sig == "AVOID":
            parts.append(f"Adding against technical AVOID signal -- fails if downtrend accelerates")
        elif buy_pct > 90:
            parts.append(f"Consensus crowded at {buy_pct:.0f}% BUY -- any earnings miss or guidance cut gets punished severely")
        elif "Technology" in sec:
            parts.append(f"Adding to {tkr} in {sec}; fails if AI capex cycle peaks or valuation compresses")
        else:
            parts.append(f"Adding to {tkr}; fails if {exret:.0f}% EXRET target is cut or {sec} sector rotates out")
        if buy_pct > 90 and "Consensus" not in parts[0]:
            parts.append(f"Crowded consensus at {buy_pct:.0f}% BUY -- any miss gets punished hard")
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


def _group_separator(label, bg, text_color, border_color, colspan=12):
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


def _act_header(label, subtitle):
    """Clean act separator with left accent bar — colors signal structure, not decoration."""
    return (f'<div style="background:{_C["bg_alt"]};'
            f'padding:14px 40px;border-left:5px solid {_C["accent_blue"]};">'
            f'<div style="color:{_C["text_dark"]};font-size:10px;font-weight:800;'
            f'letter-spacing:2px;text-transform:uppercase;">{label}</div>'
            f'<div style="color:{_C["text_muted"]};font-size:11px;margin-top:2px;">{subtitle}</div>'
            f'</div>')


def _pill_list(items, bg, txt, border, mono=True):
    """Render a list of ticker pills."""
    if not items:
        return ""
    font = _MONO if mono else ""
    pills = " ".join(
        f'<span style="display:inline-block;padding:2px 8px;margin:2px;border-radius:4px;'
        f'{font}font-size:10px;font-weight:700;background:{bg};color:{txt};'
        f'border:1px solid {border};">{e(str(item))}</span>'
        for item in items
    )
    return f'<div style="line-height:2.2;">{pills}</div>'


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
    mode: str = "full",
) -> str:
    """
    Generate the Investment Committee HTML report.

    CIO v24.0 — Three-Act Structure with consistent design system:

    ACT I: DECISIONS ("What to Do")
      S1: Executive Summary
      S2: Action Items (tiered with kill theses, position sizing)
      S3: Currency Exposure (FX pairs, zone breakdown, impact)

    ACT II: ANALYSIS ("Market Intelligence")
      S4: Macro & Market Context
      S5: Stock Analysis Grid (all stocks, grouped by action)
      S6: Where We Disagreed
      S7: Portfolio Risk (metrics, VaR, drawdown, stress)
      S8: Portfolio Construction (sectors, factors, clusters, regime, liquidity)

    ACT III: DEEP CONTEXT ("Research & Analysis", full mode only)
      S9: Fundamental Deep Dive
      S10: Technical Analysis
      S11: Sentiment & Census
      S12: News & Events
      S13: New Opportunities (+ sector gaps)

    EPILOGUE:
      S14: Watchlist
      S15: Changes Since Last Committee
      S16: Track Record
      S17: Signal Report Card
      S18: Regime Transition
    """
    daily = mode == "daily"
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    try:
        today_long = datetime.strptime(today, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        today_long = today

    # CIO v26.1: Normalize agent report structures at boundary.
    # Agents (LLMs) produce varying JSON schemas — lists vs dicts, flat vs nested.
    if fund and isinstance(fund.get("stocks"), list):
        fund["stocks"] = {
            item.get("ticker", item.get("symbol", f"UNK_{i}")): item
            for i, item in enumerate(fund["stocks"]) if isinstance(item, dict)
        }
    if news and isinstance(news.get("breaking_news"), dict):
        _sev = {"CRITICAL": "HIGH_NEGATIVE", "HIGH": "LOW_NEGATIVE"}
        _flat = []
        for _items in news["breaking_news"].values():
            if isinstance(_items, list):
                for _it in _items:
                    if isinstance(_it, dict):
                        _it.setdefault("headline", _it.get("title", _it.get("event", "")))
                        _it.setdefault("impact", _sev.get(str(_it.get("severity", "")).upper(), "NEUTRAL"))
                        _it.setdefault("affected_tickers", _it.get("tickers", []))
                        _flat.append(_it)
        news["breaking_news"] = _flat
    sr = synth.get("sector_rankings", {})
    if isinstance(sr, list):
        synth["sector_rankings"] = {
            item.get("etf", item.get("sector", "")): {
                "return_1m": item.get("1m_return", item.get("return_1m", item.get("change_1m", 0))),
                "return_3m": item.get("3m_return", item.get("return_3m", item.get("change_3m", 0))),
                "rank": item.get("rank", 6),
            }
            for item in sr if isinstance(item, dict)
        }

    # --- Build name map from all available data sources ---
    _names: Dict[str, str] = dict(name_map or {})
    for opp in (opps or {}).get("top_opportunities", []):
        t = opp.get("ticker", "")
        if t and t not in _names and opp.get("name"):
            _names[t] = opp["name"]
    # Populate from fund report (stock_analyses or stocks may carry names)
    for t, fd in (fund or {}).get("stocks", (fund or {}).get("stock_analyses", {})).items():
        if t and t not in _names:
            n = fd.get("name") or fd.get("company_name") or fd.get("company") or ""
            if n:
                _names[t] = n
    # Populate from tech report
    for t, td in (tech or {}).get("stocks", (tech or {}).get("stock_analyses", {})).items():
        if t and t not in _names:
            n = td.get("name") or td.get("company_name") or ""
            if n:
                _names[t] = n
    # Populate from concordance entries (some may carry name)
    for entry in (synth or {}).get("concordance", []):
        t = entry.get("ticker", "")
        if t and t not in _names:
            n = entry.get("name") or entry.get("company_name") or ""
            if n:
                _names[t] = n

    # --- Extract data ---
    concordance = synth.get("concordance", [])
    regime = synth.get("regime", "CAUTIOUS")
    macro_score = synth.get("macro_score", 0)
    rotation = synth.get("rotation_phase", "UNKNOWN")
    risk_score = synth.get("risk_score", 50)
    pr_synth = synth.get("portfolio_risk", {})
    # VaR: prefer flat synth key (already normalized to %) over raw portfolio_risk dict
    var_95_raw = synth.get("var_95") or pr_synth.get("var_95_annual") or pr_synth.get("var_95") or 0
    # If value is a small decimal (e.g. -0.2374), convert to percentage (-23.74)
    var_95 = var_95_raw * 100 if 0 < abs(var_95_raw) < 1 else var_95_raw
    max_dd_raw = synth.get("max_drawdown") or pr_synth.get("max_drawdown") or 0
    max_dd = max_dd_raw * 100 if 0 < abs(max_dd_raw) < 1 else max_dd_raw
    p_beta = synth.get("portfolio_beta") or pr_synth.get("portfolio_beta_vs_spy") or pr_synth.get("portfolio_beta") or 1.0
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
    _vix_raw = indicators.get('vix', 0)
    vix_val = _vix_raw.get("current", 0) if isinstance(_vix_raw, dict) else (_vix_raw if isinstance(_vix_raw, (int, float)) else 0)
    leading = [etf for etf, d in sector_rankings.items() if isinstance(d, dict) and d.get("return_1m", 0) > 2]
    lagging = [etf for etf, d in sector_rankings.items() if isinstance(d, dict) and d.get("return_1m", 0) < -2]
    parts = []
    if regime == "RISK_OFF":
        rot_label = rotation.replace('_', ' ').lower()
        article = "an" if rot_label[0] in "aeiou" else "a"
        parts.append(f"We are in {article} {rot_label} environment with deteriorating breadth")
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
        parts.append(f"Highest-priority action: exit {', '.join(sell_tickers)} ({sells} SELL).")
    if trims:
        parts.append(f"Trim {trims} positions where technical momentum has broken down or macro headwinds apply.")
    top_adds = [c["ticker"] for c in concordance if c.get("action") in ("BUY", "ADD") and c.get("conviction", 0) >= 70][:3]
    if top_adds:
        parts.append(f"Highest-conviction additions: {', '.join(top_adds)}.")
    narrative = " ".join(parts)

    # --- Disagreement detection ---
    disagreements = _detect_disagreements(concordance)

    # --- Pre-compute tech data for reuse ---
    tech_stocks = tech.get("stocks", {})
    total_tech = len(tech_stocks)
    avg_rsi = 50
    bearish_macd = 0
    below50 = 0
    if total_tech > 0:
        avg_rsi = sum(d.get("rsi", 50) for d in tech_stocks.values()) / total_tech
        bearish_macd = sum(1 for d in tech_stocks.values() if d.get("macd_signal") == "BEARISH")
        below50 = sum(1 for d in tech_stocks.values() if not d.get("above_sma50", True))

    # --- Pre-compute risk data for reuse ---
    pr = risk.get("portfolio_risk", {})
    sharpe = synth.get("sharpe_ratio", pr.get("sharpe_ratio_1y", 0)) or 0
    cvar = synth.get("cvar_95", pr.get("cvar_95_daily", 0)) or 0
    mc = pr.get("monte_carlo_var", {})
    dd = pr.get("drawdown_analysis", {})
    mc_var = mc.get("var_pct", 0) if mc else 0
    mc_cvar = mc.get("cvar_pct", 0) if mc else 0
    calmar = dd.get("calmar_ratio") if dd else None
    calmar = calmar if calmar is not None else 0
    curr_dd = dd.get("current_drawdown_pct", 0) if dd else 0
    pvar = risk.get("portfolio_var") or {}
    pvar_val = pvar.get("portfolio_var_pct")
    pcvar_val = pvar.get("portfolio_cvar_pct")
    div_benefit = pvar.get("diversification_benefit_pct")
    fexp = risk.get("factor_exposure") or {}
    corr_regime = risk.get("correlation_regime") or {}
    liq_scores = synth.get("liquidity_scores", {})
    fx_data = risk.get("fx_data", {})
    fx_pairs = fx_data.get("pairs", {})
    dxy_data = fx_data.get("dxy", {})

    # --- Action Items data (reused in both daily and full) ---
    pos_limits = synth.get("position_limits", risk.get("position_limits", {}))
    sell_items = [en for en in concordance if en.get("action") == "SELL"]
    trim_items = [en for en in concordance if en.get("action") == "TRIM"]
    buy_items = sorted([en for en in concordance if en.get("action") == "BUY"],
                       key=lambda x: (-x.get("conviction", 0), -x.get("capital_efficiency", 0)))
    add_items = sorted([en for en in concordance if en.get("action") == "ADD"],
                       key=lambda x: (-x.get("conviction", 0), -x.get("capital_efficiency", 0)))
    monitor_items = sorted([en for en in concordance if en.get("action") == "HOLD"],
                           key=lambda x: -x.get("conviction", 0))

    # --- Build HTML ---
    h = []
    h.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
             '<style>'
             '@media screen and (max-width:768px){'
             'body{font-size:13px !important;}'
             'div[style*="max-width:960px"]{max-width:100% !important;}'
             'div[style*="padding:32px 40px"]{padding:16px 12px !important;}'
             'div[style*="padding:28px 40px"]{padding:14px 12px !important;}'
             'table{font-size:11px !important;}'
             'td,th{padding:4px 6px !important;}'
             'div[style*="overflow-x:auto"]{overflow-x:scroll !important;-webkit-overflow-scrolling:touch;}'
             'table[style*="border-spacing:16px"]{border-spacing:0 !important;}'
             'table[style*="border-spacing:16px"] td{display:block !important;width:100% !important;}'
             '}'
             '@media screen and (max-width:480px){'
             'div[style*="padding:14px 40px"]{padding:10px 8px !important;}'
             'div[style*="display:flex"]{flex-direction:column !important;}'
             'div[style*="flex:1"]{width:100% !important;}'
             '}'
             '</style></head>')
    h.append(f'<body style="margin:0;padding:0;background:{_C["bg_page"]};'
             f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,'
             f'\'Helvetica Neue\',Arial,sans-serif;color:{_C["text_body"]};'
             f'line-height:1.6;-webkit-font-smoothing:antialiased;">')
    h.append(f'<div style="max-width:960px;margin:0 auto;background:{_C["bg_white"]};'
             f'box-shadow:0 1px 3px rgba(0,0,0,0.1);">')

    # ── HEADER ── (white, accent color = regime)
    regime_accent = (_C["bear"] if regime in ("RISK_OFF", "BEARISH", "CRISIS")
                     else _C["bull"] if regime in ("RISK_ON", "BULLISH")
                     else _C["warn"] if regime in ("CAUTIOUS", "TRANSITIONAL")
                     else _C["info"])
    mode_badge = (' <span style="display:inline-block;padding:2px 10px;border-radius:4px;'
                  f'font-size:10px;font-weight:700;background:{_C["info_bg"]};'
                  f'color:{_C["info"]};margin-left:8px;vertical-align:middle;">Daily Digest</span>'
                  if daily else
                  ' <span style="display:inline-block;padding:2px 10px;border-radius:4px;'
                  f'font-size:10px;font-weight:700;background:{_C["bg_alt"]};'
                  f'color:{_C["text_muted"]};margin-left:8px;vertical-align:middle;">Weekly Deep Dive</span>')
    h.append(f'<div style="background:{_C["bg_white"]};padding:32px 40px 28px 40px;'
             f'border-bottom:3px solid {regime_accent};">'
             f'<table style="width:100%;"><tr>'
             f'<td style="vertical-align:middle;width:50px;">'
             f'<div style="width:36px;height:36px;border-radius:8px;background:{regime_accent};'
             f'text-align:center;line-height:36px;font-size:18px;color:#fff;font-weight:800;">IC</div></td>'
             f'<td style="vertical-align:middle;">'
             f'<h1 style="margin:0;font-size:24px;font-weight:700;color:{_C["text_dark"]};letter-spacing:-0.5px;">'
             f'Investment Committee{mode_badge}</h1>'
             f'<p style="margin:2px 0 0 0;font-size:12px;color:{_C["text_muted"]};">'
             f'{today_long} &middot; Signals: {e(signal_date)} &middot; Census: {e(census_date)}</p>'
             f'</td></tr></table></div>')

    # ══════════════════════════════════════════════════════════════════
    # ACT I: DECISIONS
    # ══════════════════════════════════════════════════════════════════
    h.append(_act_header("Decisions", "What to do with your portfolio right now"))

    # ── S1: EXECUTIVE SUMMARY ──
    h.append(_section_open("Executive Summary",
                           "Your portfolio at a glance &mdash; key metrics and the committee's overall assessment.",
                           border="2px solid " + _C["border_heavy"]))
    # KPI cards
    h.append('<table style="width:100%;border-collapse:separate;border-spacing:10px 0;margin-bottom:20px;"><tr>')
    h.append(_kpi_card("CIO Verdict", verdict, f"{sells} Sell &middot; {trims} Trim &middot; {buys} Buy/Add &middot; {holds} Hold", vval, vbg))
    h.append(_kpi_card("Macro Regime", regime, f"Score {macro_score} &middot; {rotation.replace('_', ' ')}", vval))
    fg_exec = fg_top100 if fg_top100 else fg_broad
    fg_exec_color = (_C["bear"] if fg_exec >= 75 else _C["warn"] if fg_exec > 55
                     else _C["text_muted"] if fg_exec >= 45 else _C["info"] if fg_exec >= 25 else _C["hold"])
    fg_exec_label = ("EXTREME GREED" if fg_exec >= 75 else "GREED" if fg_exec > 55
                     else "NEUTRAL" if fg_exec >= 45 else "FEAR" if fg_exec >= 25 else "EXTREME FEAR")
    h.append(_kpi_card("Fear &amp; Greed", str(fg_exec), fg_exec_label, fg_exec_color))
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
    # Stress scenarios — compact row
    if stress:
        scenarios = [
            ("CRASH -10%", ["market_crash_spy_minus_10pct", "market_crash_10pct"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
            ("RATE +100bps", ["rate_shock_plus_100bps", "rate_shock_100bps"], _C["warn_bg"], _C["warn_border"], _C["warn_text"], _C["warn"]),
            ("VIX TO 40", ["vix_spike_to_40", "vix_spike_40", "vix_spike"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
        ]
        h.append('<div style="display:flex;gap:8px;margin-bottom:16px;">')
        for title, keys, bg, brd, lc, vc in scenarios:
            key = next((k for k in keys if k in stress), keys[0])
            sd = stress.get(key, {})
            ei = sd.get("estimated_impact", {})
            imp = (sd.get("portfolio_impact_pct")
                   or sd.get("portfolio_expected_loss_pct")
                   or sd.get("portfolio_expected_impact_pct")
                   or sd.get("estimated_portfolio_impact_pct")
                   or (ei.get("portfolio_drop_pct") if isinstance(ei, dict) else None)
                   or "?")
            imp_str = f"{float(imp):.1f}" if isinstance(imp, (int, float)) else str(imp)
            h.append(f'<div style="flex:1;padding:8px 12px;background:{bg};border:1px solid {brd};'
                     f'border-radius:6px;text-align:center;">'
                     f'<div style="font-size:9px;font-weight:700;letter-spacing:0.5px;'
                     f'text-transform:uppercase;color:{lc};">{title}</div>'
                     f'<div style="font-size:16px;font-weight:800;color:{vc};margin-top:2px;">{imp_str}%</div></div>')
        h.append('</div>')
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
                     f'<span style="color:{_C["text_dark"]};">{e(_tn(tkr, _names))}</span></span>')
        h.append('</div>')
    h.append(_section_close())

    # ── READING GUIDE (compact, educational) ──
    h.append(f'<div style="padding:16px 40px;background:{_C["bg_page"]};border-bottom:1px solid {_C["border"]};">'
             f'<div style="font-size:11px;font-weight:700;color:{_C["text_dark"]};margin-bottom:8px;'
             f'letter-spacing:0.3px;">HOW TO READ THIS REPORT</div>'
             f'<div style="font-size:11px;color:{_C["text_body"]};line-height:1.7;">'
             f'This report is produced by <b>7 AI analysts</b> examining your portfolio '
             f'from different angles &mdash; fundamentals, technicals, macro, census (crowd behavior), '
             f'news, opportunities, and risk. Each stock receives a <b>conviction score (0&ndash;100)</b>: '
             f'the higher the number, the more confident the committee is in its recommendation.'
             f'<div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;">'
             f'<span style="font-size:10px;"><span style="color:{_C["bear"]};font-weight:700;">SELL</span> = exit position</span>'
             f'<span style="font-size:10px;"><span style="color:{_C["warn"]};font-weight:700;">TRIM</span> = reduce size</span>'
             f'<span style="font-size:10px;"><span style="color:{_C["hold"]};font-weight:700;">HOLD</span> = keep, no action</span>'
             f'<span style="font-size:10px;"><span style="color:{_C["bull"]};font-weight:700;">ADD</span> = increase position</span>'
             f'<span style="font-size:10px;"><span style="color:{_C["bull"]};font-weight:700;">BUY</span> = new position</span>'
             f'</div>'
             f'<div style="margin-top:6px;font-size:10px;color:{_C["text_muted"]};">'
             f'Colors throughout: '
             f'<span style="color:{_C["bull"]};">&#9679; green = positive/bullish</span> &middot; '
             f'<span style="color:{_C["bear"]};">&#9679; red = negative/bearish</span> &middot; '
             f'<span style="color:{_C["warn"]};">&#9679; amber = caution</span> &middot; '
             f'<span style="color:{_C["hold"]};">&#9679; purple = neutral/hold</span>'
             f'</div></div></div>')

    # ── S2: ACTION ITEMS ──
    h.append(_section_open("Action Items",
                           "What to do now: stocks to buy, sell, or watch, ranked by urgency. "
                           "Kill thesis = what would make this trade wrong.",
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
        pos_size_pct = en.get("position_size_pct", 0)
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
        # Position size with color coding
        if pos_size_pct and act in ("BUY", "ADD"):
            psc = _C["bull"] if pos_size_pct >= 4.0 else _C["text_muted"] if pos_size_pct < 2.0 else _C["text_body"]
            ps_weight = "font-weight:700;" if pos_size_pct >= 4.0 else ""
            extra_tags += f' | <span style="color:{psc};{ps_weight}">Size {pos_size_pct:.1f}%</span>'
        # Stop-loss levels (CIO v22.0)
        sl = en.get("stop_losses", {})
        sl_html = ""
        if sl and act in ("BUY", "ADD", "SELL", "TRIM"):
            chandelier = sl.get("chandelier_stop") or sl.get("chandelier")
            tight = sl.get("tight_stop") or sl.get("tight")
            sma200_sl = sl.get("sma200_stop") or sl.get("sma200")
            sl_parts = []
            if chandelier and chandelier > 0:
                sl_parts.append(f'Chan ${chandelier:.2f}')
            if tight and tight > 0:
                sl_parts.append(f'Tight ${tight:.2f}')
            if sma200_sl and sma200_sl > 0:
                sl_parts.append(f'SMA200 ${sma200_sl:.2f}')
            if sl_parts:
                sl_html = (f'<div style="font-size:10px;color:{_C["bear"]};margin-top:4px;'
                           f'{_MONO}">&#9660; Stops: {" | ".join(sl_parts)}</div>')
        # CIO v25.0: Conviction waterfall — compact modifier attribution
        wf = en.get("conviction_waterfall", {})
        wf_html = ""
        if wf and act in ("BUY", "ADD", "SELL", "TRIM"):
            wf_parts = []
            for k, v in sorted(wf.items(), key=lambda x: abs(x[1]), reverse=True):
                if k.startswith("_"):
                    continue
                color = _C["bull"] if v > 0 else _C["bear"]
                sign = "+" if v > 0 else ""
                label = k.replace("_", " ")
                wf_parts.append(f'<span style="color:{color};">{label} {sign}{v}</span>')
            if wf_parts:
                wf_html = (f'<div style="font-size:10px;color:{_C["text_muted"]};margin-top:4px;'
                           f'{_MONO}">Factors: {" &middot; ".join(wf_parts)}</div>')
        # Entry timing-based left border color
        timing = en.get("entry_timing", "")
        timing_border_colors = {
            "ENTER_NOW": _C["bull"],
            "WAIT_FOR_PULLBACK": _C["warn"],
            "EXIT_SOON": _C["bear"],
            "AVOID": _C["bear"],
        }
        card_border = timing_border_colors.get(timing, action_color(act))
        name_label = _clean_name(_names.get(tkr, ""))
        name_html = (f'<span style="font-size:12px;color:{_C["text_body"]};margin-left:8px;">'
                     f'{e(name_label)}</span> ' if name_label else "")
        inner = (f'<table style="width:100%;"><tr><td>'
                 f'{badge(act, action_color(act), "#fff")} '
                 f'<span style="{_MONO}font-weight:800;font-size:14px;margin-left:8px;">{e(tkr)}</span> '
                 f'{name_html}'
                 f'<span style="font-size:11px;color:{_C["text_muted"]};margin-left:8px;">'
                 f'{sec} | RSI {rsi:.0f} | {abbr(ts)} | {abbr(mf)}'
                 f'{" | Max " + str(int(mp)) + "%" if act in ("BUY","ADD") else ""}'
                 f'{" | $" + str(int(en.get("suggested_size_usd", 0))) if en.get("suggested_size_usd") and act in ("BUY","ADD") else ""}'
                 f'{" | CE " + f"{ce:.1f}" if ce and act in ("BUY","ADD") else ""}'
                 f'{extra_tags}</span></td>'
                 f'<td style="text-align:right;">{conv_display(conv)}</td></tr></table>'
                 f'{sl_html}'
                 f'<div style="font-size:11px;color:{_C["text_muted"]};font-style:italic;'
                 f'margin-top:8px;line-height:1.5;">{e(kill)}</div>'
                 f'{wf_html}')
        h.append(_card(inner, card_border, action_bg(act)))

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

    # Monitor: HOLD — compact pill list (skip in daily)
    if monitor_items and not daily:
        h.append(f'<div style="{_LABEL}margin:20px 0 10px 0;">Monitor ({len(monitor_items)})</div>')
        h.append('<div style="line-height:2.2;">')
        for en in monitor_items:
            tkr = en.get("ticker", "")
            conv = en.get("conviction", 0)
            cc = conv_color(conv)
            h.append(f'<span style="display:inline-block;padding:3px 10px;margin:2px 3px;border-radius:4px;'
                     f'{_MONO}font-size:11px;font-weight:700;background:{_C["bg_alt"]};'
                     f'border:1px solid {_C["border"]};">{e(_tn(tkr, _names))}'
                     f'<span style="color:{cc};font-size:10px;margin-left:4px;">{conv}</span></span>')
        h.append('</div>')
    h.append(_section_close())

    # ── S3: CURRENCY EXPOSURE ──
    if fx_pairs:
        if daily:
            # Daily: compact card
            eur_usd = fx_pairs.get("EUR/USD", {})
            rate = eur_usd.get("rate", 0)
            chg = eur_usd.get("change_1m_pct", eur_usd.get("change_1m", 0)) or 0
            trend = eur_usd.get("trend", "Stable")
            usd_count = sum(1 for c in concordance if (c.get("currency_zone") or "USD") == "USD")
            h.append(f'<div style="padding:32px 40px;border-bottom:1px solid {_C["border"]};">')
            h.append(f'<h2 style="{_SECTION_H2}">Currency Snapshot</h2>')
            h.append(_card(
                f'<b>EUR/USD</b> {rate:.4f} ({chg:+.2f}% 1M) — {e(str(trend))}<br>'
                f'<span style="font-size:11px;color:{_C["text_muted"]};">'
                f'{usd_count}/{len(concordance)} positions USD-denominated</span>',
                _C["info"], _C["info_bg"]
            ))
            h.append(_section_close())
        else:
            # Full: complete FX table
            h.append(_section_open("Currency Exposure",
                                   "How exchange rate movements affect your returns (you invest in EUR, but many stocks trade in USD, GBP, etc.)",
                                   border=f"1px solid {_C['border_heavy']}"))
            # FX Rates table
            h.append(_table_open([("Pair", "left"), ("Rate", "center"),
                                  ("1M Chg", "center"), ("3M Chg", "center"), ("Trend", "center")]))
            for pair_name, pair_info in fx_pairs.items():
                chg1 = pair_info.get("change_1m")
                chg3 = pair_info.get("change_3m")
                trend = pair_info.get("trend", "")
                chg1_str = f"{chg1:+.2f}%" if chg1 is not None else "—"
                chg3_str = f"{chg3:+.2f}%" if chg3 is not None else "—"
                chg1_col = _C["bear"] if chg1 and chg1 > 2 and pair_name == "EUR/USD" else _C["bull"] if chg1 and chg1 < -2 and pair_name == "EUR/USD" else _C["text_muted"]
                trend_label = trend.replace("_", " ").replace("EUR ", "").title()
                trend_col = _C["bear"] if "HEADWIND" in trend else _C["bull"] if "TAILWIND" in trend else _C["text_muted"]
                h.append(_table_row([
                    (f'<span style="{_MONO}font-weight:700;">{e(pair_name)}</span>', "left", ""),
                    (f'{pair_info.get("rate", "—")}', "center", ""),
                    (f'<span style="color:{chg1_col};font-weight:600;">{chg1_str}</span>', "center", ""),
                    (f'{chg3_str}', "center", f"color:{_C['text_muted']};"),
                    (f'<span style="color:{trend_col};font-weight:600;">{e(trend_label)}</span>', "center", ""),
                ]))
            # DXY row
            if dxy_data:
                dxy_chg1 = dxy_data.get("change_1m")
                h.append(_table_row([
                    (f'<span style="{_MONO}font-weight:700;">DXY (Dollar Index)</span>', "left", ""),
                    (f'{dxy_data.get("value", "—")}', "center", ""),
                    (f'{dxy_chg1:+.2f}%' if dxy_chg1 is not None else "—", "center", ""),
                    (f'{dxy_data.get("change_3m", 0):+.2f}%' if dxy_data.get("change_3m") is not None else "—", "center", f"color:{_C['text_muted']};"),
                    ("", "center", ""),
                ]))
            h.append('</table>')

            # Per-zone exposure summary
            zones = {}
            for item in concordance:
                z = item.get("currency_zone", "USD")
                zones.setdefault(z, []).append(item)

            h.append(f'<div style="margin-top:12px;padding:12px;background:{_C["bg_page"]};border-radius:6px;">')
            h.append(f'<div style="font-weight:700;font-size:12px;color:{_C["text_dark"]};margin-bottom:8px;">Portfolio Currency Breakdown</div>')
            for zone in ["USD", "EUR", "GBP", "HKD", "JPY", "CHF", "CRYPTO"]:
                items = zones.get(zone, [])
                if not items:
                    continue
                tickers = ", ".join(_tn(i["ticker"], _names) for i in items[:8])
                if len(items) > 8:
                    tickers += f" +{len(items)-8} more"
                fx_note = ""
                if zone == "EUR":
                    fx_note = '<span style="color:' + _C["bull"] + ';">Home currency — no FX risk</span>'
                elif zone == "CRYPTO":
                    fx_note = '<span style="color:' + _C["text_muted"] + ';">FX exempt</span>'
                else:
                    pair = fx_pairs.get(f"EUR/{zone}", fx_pairs.get(f"{zone}/EUR", {}))
                    impact = pair.get("trend", "STABLE")
                    if "HEADWIND" in impact:
                        fx_note = f'<span style="color:{_C["bear"]};">FX headwind</span>'
                    elif "TAILWIND" in impact:
                        fx_note = f'<span style="color:{_C["bull"]};">FX tailwind</span>'
                    else:
                        fx_note = f'<span style="color:{_C["text_muted"]};">Stable</span>'
                h.append(f'<div style="font-size:11px;margin:4px 0;color:{_C["text_muted"]};">'
                         f'<b>{zone}</b> ({len(items)}): {e(tickers)} — {fx_note}</div>')
            h.append('</div>')
            h.append(_section_close())

    # ══════════════════════════════════════════════════════════════════
    # ACT II: ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    h.append(_act_header("Market Intelligence", "The bigger picture: economy, markets, and how your portfolio fits"))

    # ── S4: MACRO & MARKET CONTEXT ──
    if daily:
        h.append(_section_open("Macro Snapshot"))
    else:
        h.append(_section_open("Macro &amp; Market Context"))
    # Macro indicators — try multiple key conventions (agent output varies).
    # Macro agent writes: us_10y_yield, yield_curve_10y_2y, vix, dxy, eur_usd
    t10y = sf(indicators.get('us_10y_yield') or indicators.get('treasury_10y') or indicators.get('yield_10y') or 0)
    yc_spread = sf(indicators.get('yield_curve_10y_2y') or indicators.get('yield_curve') or indicators.get('yield_curve_spread') or 0)
    yc_bps = yc_spread * 100 if abs(yc_spread) < 5 else yc_spread  # 0.53 -> 53bps
    vix_ind = sf(indicators.get('vix') or 0)
    dxy_raw = sf(indicators.get('dxy') or indicators.get('dollar_index') or 0)
    dxy_str = f"{dxy_raw:.1f}" if 80 <= dxy_raw <= 120 else "N/A"
    eur_usd = sf(indicators.get('eur_usd') or 0)
    brent = sf(indicators.get('brent_crude') or indicators.get('brent') or 0)
    macro_ind = [
        ("10Y Yield", f"{t10y:.2f}%",
         "NORMAL" if t10y < 4.5 else "ELEVATED" if t10y < 5.0 else "HIGH"),
        ("Yield Curve", f"{yc_bps:.0f}bps",
         "POSITIVE" if yc_bps > 0 else "INVERTED"),
        ("VIX", f"{vix_ind:.1f}",
         "ELEVATED" if vix_ind > 20 else "NORMAL"),
        ("Dollar (DXY)", dxy_str,
         "STRONG" if dxy_raw > 105 else "STABLE" if dxy_raw >= 95 else "WEAK" if dxy_str != "N/A" else "N/A"),
    ]
    if eur_usd > 0:
        macro_ind.append(("EUR/USD", f"{eur_usd:.4f}",
                          "EUR STRONG" if eur_usd > 1.12 else "NEUTRAL" if eur_usd > 1.05 else "USD STRONG"))
    if brent > 0:
        macro_ind.append(("Brent Crude", f"${brent:.2f}",
                          "CRISIS" if brent > 100 else "ELEVATED" if brent > 80 else "NORMAL"))
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
    if total_tech > 0:
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
        for etf, data in sorted(sector_rankings.items(), key=lambda x: (x[1].get("return_1m", 0) if isinstance(x[1], dict) else -x[1]), reverse=True):
            if not isinstance(data, dict):
                continue  # skip int-only rankings from macro agent
            ret = data.get("return_1m", 0)
            sbg = _C["bull_bg"] if ret > 2 else _C["bear_bg"] if ret < -2 else _C["bg_page"]
            stxt = _C["bull"] if ret > 2 else _C["bear"] if ret < -2 else _C["text_body"]
            h.append(f'<td style="padding:6px 4px;text-align:center;background:{sbg};border-radius:4px;">'
                     f'<div style="font-weight:700;color:{_C["text_body"]};font-size:9px;">{e(etf)}</div>'
                     f'<div style="font-weight:800;color:{stxt};">{ret:+.1f}%</div></td>')
        h.append('</tr></table>')

    # CIO v23.5: Relative Rotation Graph (sector ETFs vs SPY) — full only
    sector_rotation = tech.get("sector_rotation", [])
    if sector_rotation and not daily:
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Sector Rotation (RRG)</div>')
        quadrant_colors = {
            'LEADING': _C["bull"], 'WEAKENING': _C["warn"],
            'LAGGING': _C["bear"], 'IMPROVING': _C["info"],
        }
        h.append(f'<table style="{_TABLE}">')
        h.append(f'<tr style="border-bottom:2px solid {_C["border"]};">'
                 f'<th style="padding:4px 8px;text-align:left;{_LABEL}font-size:10px;">Sector</th>'
                 f'<th style="padding:4px 8px;text-align:center;{_LABEL}font-size:10px;">RS Ratio</th>'
                 f'<th style="padding:4px 8px;text-align:center;{_LABEL}font-size:10px;">RS Mom</th>'
                 f'<th style="padding:4px 8px;text-align:center;{_LABEL}font-size:10px;">Quadrant</th></tr>')
        for i, sr in enumerate(sector_rotation):
            bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
            qc = quadrant_colors.get(sr.get("quadrant", ""), _C["text_muted"])
            h.append(f'<tr style="background:{bg};">'
                     f'<td style="padding:4px 8px;font-weight:600;font-size:10px;">'
                     f'{e(sr.get("sector", ""))}</td>'
                     f'<td style="padding:4px 8px;text-align:center;{_MONO}font-size:10px;">'
                     f'{sr.get("rs_ratio", 100):.1f}</td>'
                     f'<td style="padding:4px 8px;text-align:center;{_MONO}font-size:10px;">'
                     f'{sr.get("rs_momentum", 100):.1f}</td>'
                     f'<td style="padding:4px 8px;text-align:center;">'
                     f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
                     f'background:{qc};color:#fff;font-size:9px;font-weight:700;">'
                     f'{e(sr.get("quadrant", ""))}</span></td></tr>')
        h.append('</table>')

    h.append(_section_close())

    # ── S5: STOCK ANALYSIS GRID ──
    sell_list = [en for en in concordance if en.get("action") == "SELL"]
    trim_list = [en for en in concordance if en.get("action") == "TRIM"]
    buy_list = [en for en in concordance if en.get("action") == "BUY"]
    add_list = [en for en in concordance if en.get("action") == "ADD"]
    hold_list = [en for en in concordance if en.get("action") == "HOLD"]

    h.append(_section_open("Stock Analysis Grid",
                           f"{len(concordance)} stocks scored by all 7 analysts. "
                           f"Each column shows one analyst's view &mdash; green means bullish, red means bearish. "
                           f"SIG = overall signal (B=Buy, H=Hold, S=Sell). "
                           f"CONV = conviction score (0-100, higher = more confident)."))
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
    h.append(grid_hdr(_C["text_dark"], "#fff", "CONV") + grid_hdr(_C["text_dark"], "#9ca3af", "TIMING"))
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
        # TIMING cell
        timing = entry.get("entry_timing", "")
        timing_col = {"ENTER_NOW": _C["bull"], "WAIT_FOR_PULLBACK": _C["warn"],
                      "AVOID": _C["bear"], "EXIT_SOON": _C["bear"],
                      "buy": _C["bull"], "accumulate": _C["bull"],
                      "hold": _C["text_muted"], "HOLD": _C["text_muted"],
                      "reduce": _C["warn"], "sell": _C["bear"],
                      "strong_sell": _C["bear"]}.get(timing, _C["text_muted"])
        timing_abbr = {"ENTER_NOW": "ENTER", "WAIT_FOR_PULLBACK": "WAIT",
                       "AVOID": "AVOID", "EXIT_SOON": "EXIT",
                       "buy": "BUY", "accumulate": "ACCUM",
                       "hold": "HOLD", "HOLD": "HOLD",
                       "reduce": "REDUC", "sell": "SELL",
                       "strong_sell": "STR.SL"}.get(timing, timing[:6] if timing else "—")
        # Regional indicator
        region_suffix = ""
        _region_map = {".DE": "EU", ".PA": "EU", ".BR": "EU", ".L": "GB",
                       ".OL": "NO", ".CO": "DK", ".HK": "HK", ".AE": "AE", ".T": "JP"}
        for _sfx, _lbl in _region_map.items():
            if tkr.endswith(_sfx):
                region_suffix = f'<span style="font-size:9px;color:{_C["text_light"]};margin-left:2px;">({_lbl})</span>'
                break
        # Conviction decay
        decay_days = entry.get("conviction_decay_days", 0)
        decay_factor = entry.get("conviction_decay_factor", 1.0)
        conv_html = conv_display(conv, delta)
        if decay_days > 0:
            conv_html += f'<span style="font-size:10px;color:#94a3b8;margin-left:3px;">↓{decay_factor:.1f}x ({decay_days}d)</span>'
        return (f'<tr style="background:{rb};">'
                f'<td {p}text-align:left;padding-left:8px;">{_tn_cell(tkr, _names)}{region_suffix}</td>'
                f'<td {p}">{signal_badge(sig)}</td>'
                f'<td {p}color:{sentiment_color(fv)};{sv}">{abbr(fv)}({fs:.0f}){sm}</td>'
                f'<td {p}color:{sentiment_color(ts)};{sv}">{abbr(ts)}({rsi:.0f})</td>'
                f'<td {p}color:{sentiment_color(mf)};{sv}">{abbr(mf)}</td>'
                f'<td {p}color:{sentiment_color(ce)};{sv}">{abbr(ce)}</td>'
                f'<td {p}color:{sentiment_color(ni)};{sv}">{ni_abbr}</td>'
                f'<td {p}color:{risk_col};{sv}">{risk_label}</td>'
                f'<td {p}{_MONO}font-weight:700;color:{exc};">{ex:.0f}%</td>'
                f'<td {p}">{badge(act, action_color(act), "#fff")}</td>'
                f'<td {p}">{conv_html}</td>'
                f'<td {p}color:{timing_col};{sv}">{timing_abbr}</td></tr>')

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
    if daily:
        # Daily: actionable rows shown above; HOLD as pills
        if hold_list:
            h.append(_group_separator(f"HOLD ({len(hold_list)} — monitoring)", _C["bg_alt"], "#475569", _C["border_heavy"]))
    else:
        if hold_list:
            h.append(_group_separator(f"HOLD ({len(hold_list)})", _C["bg_alt"], "#475569", _C["border_heavy"]))
            for en in hold_list:
                h.append(_grid_row(en))
    h.append('</table></div>')
    # Legend
    h.append(f'<div style="margin-top:10px;padding:10px 12px;background:{_C["bg_page"]};'
             f'border:1px solid {_C["border"]};border-radius:6px;font-size:10px;color:{_C["text_muted"]};line-height:1.6;">'
             f'<b style="color:{_C["text_dark"]};">Reading the Grid:</b> '
             f'<b>FUND</b> = company financials (BUY/SELL/HOLD + score) &middot; '
             f'<b>TECH</b> = price momentum (ENTER/WAIT/EXIT + RSI) &middot; '
             f'<b>MACRO</b> = economic fit (FAVOR/UNFAV) &middot; '
             f'<b>CENS</b> = crowd alignment (ALIGN/DIVERG) &middot; '
             f'<b>NEWS</b> = recent news impact &middot; '
             f'<b>RISK</b> = risk manager flag &middot; '
             f'<b>EXR</b> = expected return above market (%) &middot; '
             f'<b>TIMING</b> = when to act (ENTER now vs WAIT for a better price)<br/>'
             f'* = Synthetic data (estimated, not reported) &middot; '
             f'&#9650;&#9660; = Conviction change vs last committee &middot; '
             f'↓ = Conviction decay (signal aging)</div>')
    # Entry Timing Summary
    enter_now = [en for en in concordance if en.get("entry_timing") == "ENTER_NOW"]
    wait_pullback = [en for en in concordance if en.get("entry_timing") == "WAIT_FOR_PULLBACK"]
    if enter_now or wait_pullback:
        h.append(f'<div style="margin-top:16px;"><div style="{_LABEL}margin-bottom:8px;">Entry Timing Summary</div>')
        h.append('<div style="display:flex;gap:10px;">')
        if enter_now:
            enter_tickers = [_tn(en.get("ticker", ""), _names) for en in enter_now[:10]]
            h.append(f'<div style="flex:1;background:{_C["bull_bg"]};border:1px solid {_C["bull_border"]};'
                     f'border-radius:6px;padding:10px;">'
                     f'<div style="font-size:10px;font-weight:700;color:{_C["bull_text"]};margin-bottom:6px;">ENTER NOW ({len(enter_now)})</div>'
                     f'{_pill_list(enter_tickers, _C["bg_white"], _C["text_dark"], _C["border"])}'
                     f'</div>')
        if wait_pullback:
            wait_tickers = [_tn(en.get("ticker", ""), _names) for en in wait_pullback[:10]]
            h.append(f'<div style="flex:1;background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                     f'border-radius:6px;padding:10px;">'
                     f'<div style="font-size:10px;font-weight:700;color:{_C["warn_text"]};margin-bottom:6px;">WAIT FOR PULLBACK ({len(wait_pullback)})</div>'
                     f'{_pill_list(wait_tickers, _C["bg_white"], _C["text_dark"], _C["border"])}'
                     f'</div>')
        h.append('</div></div>')
    h.append(_section_close())

    # ── S6: WHERE WE DISAGREED ──
    h.append(_section_open("Where We Disagreed",
                           "When analysts disagree, it often reveals the most important insights. "
                           "These stocks had conflicting views that required committee judgment."))
    for d in disagreements[:6]:
        fb = agent_badge("Fund", d["fund_view"], "#dbeafe", "#1e40af", "#93c5fd")
        tb = agent_badge("Tech", d["tech_signal"], "#ede9fe", "#5b21b6", "#c4b5fd")
        d_name = _clean_name(_names.get(d["ticker"], ""))
        d_name_html = (f' <span style="font-size:11px;color:{_C["text_muted"]};">'
                       f'{e(d_name)}</span>') if d_name else ""
        inner = (f'<div style="margin-bottom:8px;">'
                 f'<span style="{_MONO}font-weight:800;font-size:14px;color:{_C["text_dark"]};">{e(d["ticker"])}</span>'
                 f'{d_name_html}'
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

    # ── S7: PORTFOLIO RISK ──
    if daily:
        # Daily: condensed Risk Alerts
        h.append(_section_open("Risk Alerts",
                               "Key risk flags &mdash; VaR shows maximum expected loss, factor tilts show portfolio imbalances."))
        # VaR + CVaR compact
        h.append('<div style="display:flex;gap:8px;margin-bottom:12px;">')
        pvar_display = f"{pvar_val:.1f}%" if pvar_val is not None else f"{mc_var:.1f}%"
        pcvar_display = f"{pcvar_val:.1f}%" if pcvar_val is not None else f"{mc_cvar:.1f}%"
        var_col = _C["bear"] if (pvar_val or mc_var) and (pvar_val or mc_var) < -8 else _C["warn"]
        h.append(f'<div style="flex:1;padding:10px;background:{_C["bg_page"]};border:1px solid {_C["border"]};'
                 f'border-radius:6px;text-align:center;">'
                 f'<div style="{_LABEL}">Port VaR 21d</div>'
                 f'<div style="font-size:18px;font-weight:800;color:{var_col};margin-top:2px;">{pvar_display}</div></div>')
        h.append(f'<div style="flex:1;padding:10px;background:{_C["bg_page"]};border:1px solid {_C["border"]};'
                 f'border-radius:6px;text-align:center;">'
                 f'<div style="{_LABEL}">Port CVaR 21d</div>'
                 f'<div style="font-size:18px;font-weight:800;color:{_C["bear"]};margin-top:2px;">{pcvar_display}</div></div>')
        h.append('</div>')
        # Factor tilt alerts
        if fexp.get("tilts"):
            for tilt in fexp["tilts"]:
                h.append(f'<div style="background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                         f'padding:6px 12px;margin:6px 0;border-radius:4px;font-size:11px;'
                         f'color:{_C["warn_text"]};">&#9888; {e(tilt)}</div>')
        # Correlation regime if not STABLE
        if corr_regime.get("warning"):
            regime_label = corr_regime.get("regime", "UNKNOWN")
            baseline = corr_regime.get("baseline_avg_correlation", 0)
            recent = corr_regime.get("recent_avg_correlation", 0)
            h.append(f'<div style="background:{_C["bear_bg"]};border:1px solid {_C["bear_border"]};'
                     f'border-left:4px solid {_C["bear"]};border-radius:0 6px 6px 0;'
                     f'padding:10px 14px;margin:8px 0;font-size:12px;color:{_C["bear_text"]};">'
                     f'<b>&#9888; Correlation Regime: {e(regime_label)}</b> &mdash; '
                     f'Recent 30d ({recent:.3f}) vs 1Y ({baseline:.3f})</div>')
        h.append(_section_close())
    else:
        # Full: Portfolio Risk section (metrics, VaR, stress)
        h.append(_section_open("Portfolio Risk",
                               "How much could you lose? Risk metrics, stress tests, and worst-case scenarios."))
        # Risk metrics table
        risk_metrics = [
            ("Sharpe (1Y)", f"{sharpe:.2f}", _C["bull"] if sharpe > 0.5 else _C["warn"] if sharpe > 0 else _C["bear"]),
            ("Sortino Ratio", f"{pr.get('sortino_ratio', 0):.2f}", _C["text_body"]),
            ("Max Drawdown", f"{max_dd:.1f}%", _C["bear"]),
            ("Current DD", f"{curr_dd:.1f}%", _C["bear"] if curr_dd < -5 else _C["bull"] if curr_dd == 0 else _C["warn"]),
            ("Calmar Ratio", f"{calmar:.2f}", _C["bull"] if calmar > 1 else _C["warn"] if calmar > 0.5 else _C["bear"]),
            ("Portfolio Beta", f"{p_beta:.2f}", _C["text_body"]),
        ]
        h.append(f'<table style="{_TABLE}">')
        for i, (label, val, vc) in enumerate(risk_metrics):
            bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
            h.append(f'<tr style="background:{bg};"><td style="padding:8px 10px;font-weight:600;">{label}</td>'
                     f'<td style="padding:8px 10px;text-align:right;{_MONO}font-weight:700;color:{vc};">{val}</td></tr>')
        h.append('</table>')

        # Portfolio VaR section
        var_metrics = [
            ("VaR (95%)", f"{var_95:.1f}%", _C["text_body"]),
            ("CVaR (95%)", f"{cvar*100:.1f}%" if abs(cvar) < 1 else f"{cvar:.1f}%", _C["bear"]),
            ("MC VaR (21d)", f"{mc_var:.1f}%", _C["bear"] if mc_var < -5 else _C["text_body"]),
            ("MC CVaR (21d)", f"{mc_cvar:.1f}%", _C["bear"]),
        ]
        if pvar_val is not None:
            var_metrics.append(("Port. VaR (21d)", f"{pvar_val:.1f}%",
                                _C["bear"] if pvar_val < -8 else _C["warn"] if pvar_val < -5 else _C["text_body"]))
        if pcvar_val is not None:
            var_metrics.append(("Port. CVaR (21d)", f"{pcvar_val:.1f}%", _C["bear"]))
        if div_benefit is not None:
            var_metrics.append(("Diversification", f"{div_benefit:.0f}%",
                                _C["bull"] if div_benefit > 20 else _C["warn"] if div_benefit > 10 else _C["bear"]))
        h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Portfolio VaR</div>')
        h.append(f'<table style="{_TABLE}">')
        for i, (label, val, vc) in enumerate(var_metrics):
            bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
            h.append(f'<tr style="background:{bg};"><td style="padding:8px 10px;font-weight:600;">{label}</td>'
                     f'<td style="padding:8px 10px;text-align:right;{_MONO}font-weight:700;color:{vc};">{val}</td></tr>')
        h.append('</table>')

        # Stress scenarios compact row (same as exec summary)
        if stress:
            h.append(f'<div style="{_LABEL}margin:16px 0 8px 0;">Stress Scenarios</div>')
            scenarios = [
                ("CRASH -10%", ["market_crash_spy_minus_10pct", "market_crash_10pct"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
                ("RATE +100bps", ["rate_shock_plus_100bps", "rate_shock_100bps"], _C["warn_bg"], _C["warn_border"], _C["warn_text"], _C["warn"]),
                ("VIX TO 40", ["vix_spike_to_40", "vix_spike_40", "vix_spike"], _C["bear_bg"], _C["bear_border"], _C["bear_text"], _C["bear"]),
            ]
            h.append('<div style="display:flex;gap:8px;margin-bottom:12px;">')
            for title, keys, bg, brd, lc, vc in scenarios:
                key = next((k for k in keys if k in stress), keys[0])
                sd = stress.get(key, {})
                ei = sd.get("estimated_impact", {})
                imp = (sd.get("portfolio_impact_pct")
                       or sd.get("portfolio_expected_loss_pct")
                       or sd.get("portfolio_expected_impact_pct")
                       or sd.get("estimated_portfolio_impact_pct")
                       or (ei.get("portfolio_drop_pct") if isinstance(ei, dict) else None)
                       or "?")
                imp_str = f"{float(imp):.1f}" if isinstance(imp, (int, float)) else str(imp)
                h.append(f'<div style="flex:1;padding:8px 12px;background:{bg};border:1px solid {brd};'
                         f'border-radius:6px;text-align:center;">'
                         f'<div style="font-size:9px;font-weight:700;letter-spacing:0.5px;'
                         f'text-transform:uppercase;color:{lc};">{title}</div>'
                         f'<div style="font-size:16px;font-weight:800;color:{vc};margin-top:2px;">{imp_str}%</div></div>')
            h.append('</div>')
        h.append(_section_close())

        # ── S8: PORTFOLIO CONSTRUCTION ──
        h.append(_section_open("Portfolio Construction",
                               "Is your portfolio well-balanced? Sector concentration, style tilts, and diversification quality."))

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

        # Factor exposure decomposition
        if fexp.get("market"):
            h.append(f'<div style="{_LABEL}color:{_C["info"]};margin:16px 0 8px 0;">Factor Exposure</div>')
            h.append(f'<table style="{_TABLE}">')
            h.append(f'<tr style="border-bottom:2px solid {_C["border"]};">'
                     f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Factor</th>'
                     f'<th style="padding:6px 10px;text-align:center;{_LABEL}width:80px;">Value</th>'
                     f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Detail</th></tr>')
            factor_rows = [
                ("Market (Beta)", fexp.get("market", {}).get("avg_beta"),
                 f'High \u03B2: {fexp.get("market", {}).get("high_beta_count", 0)}, '
                 f'Low \u03B2: {fexp.get("market", {}).get("low_beta_count", 0)}'),
                ("Size", fexp.get("size", {}).get("mega_pct"),
                 f'Mega: {fexp.get("size", {}).get("mega_pct", 0):.0f}%, '
                 f'Sm/Mid: {fexp.get("size", {}).get("small_mid_pct", 0):.0f}%'),
                ("Value (PE)", fexp.get("value", {}).get("avg_pe"),
                 f'Value (<15): {fexp.get("value", {}).get("value_stocks", 0)}, '
                 f'Growth (>30): {fexp.get("value", {}).get("growth_stocks", 0)}'),
                ("Momentum (RS)", fexp.get("momentum", {}).get("avg_rs"),
                 f'Outperforming SPY: {fexp.get("momentum", {}).get("outperforming_spy_pct") or 0:.0f}%'),
                ("Quality (ROE)", fexp.get("quality", {}).get("avg_roe"),
                 None),
            ]
            for i, (lbl, val, detail) in enumerate(factor_rows):
                if val is None:
                    continue
                avg_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                detail_str = f'<span style="color:{_C["text_muted"]};font-size:10px;">{e(detail)}</span>' if detail else ""
                bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
                h.append(f'<tr style="background:{bg};">'
                         f'<td style="padding:6px 10px;font-weight:600;font-size:11px;">{lbl}</td>'
                         f'<td style="padding:6px 10px;text-align:center;{_MONO}font-weight:700;font-size:11px;">'
                         f'{avg_str}</td>'
                         f'<td style="padding:6px 10px;">{detail_str}</td></tr>')
            h.append('</table>')
            if fexp.get("tilts"):
                for tilt in fexp["tilts"]:
                    h.append(f'<div style="background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                             f'padding:6px 12px;margin:6px 0;border-radius:4px;font-size:11px;'
                             f'color:{_C["warn_text"]};">&#9888; {e(tilt)}</div>')

        # Correlation clusters — handle both grouped and pairwise formats
        risk_clusters = risk.get("correlation_clusters", [])
        display_clusters = []
        if risk_clusters and isinstance(risk_clusters, list) and risk_clusters[0].get("stocks"):
            display_clusters = risk_clusters
        elif clusters:
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
                pills = _pill_list(stks[:5], _C["bg_alt"], _C["text_dark"], _C["border"])
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

        # Correlation regime shift warning
        if corr_regime.get("warning"):
            regime_label = corr_regime.get("regime", "UNKNOWN")
            baseline = corr_regime.get("baseline_avg_correlation", 0)
            recent = corr_regime.get("recent_avg_correlation", 0)
            h.append(f'<div style="background:{_C["bear_bg"]};border:1px solid {_C["bear_border"]};'
                     f'border-left:4px solid {_C["bear"]};border-radius:0 6px 6px 0;'
                     f'padding:10px 14px;margin:12px 0;font-size:12px;color:{_C["bear_text"]};">'
                     f'<b>&#9888; Correlation Regime: {e(regime_label)}</b> &mdash; '
                     f'Recent 30d avg correlation ({recent:.3f}) vs 1Y baseline ({baseline:.3f}). '
                     f'Diversification benefit may be evaporating.</div>')
        elif corr_regime.get("regime"):
            h.append(f'<div style="background:{_C["info_bg"]};border:1px solid {_C["info_border"]};'
                     f'padding:6px 12px;margin:8px 0;border-radius:4px;font-size:11px;'
                     f'color:{_C["text_body"]};">'
                     f'Correlation Regime: <b>{e(corr_regime["regime"])}</b> '
                     f'(30d: {corr_regime.get("recent_avg_correlation", 0):.3f} vs '
                     f'1Y: {corr_regime.get("baseline_avg_correlation", 0):.3f})</div>')

        # Liquidity alerts
        low_liq = [(t, s) for t, s in liq_scores.items()
                   if isinstance(s, dict) and s.get("label") in ("LOW", "ILLIQUID")]
        if low_liq:
            h.append(f'<div style="background:{_C["warn_bg"]};border:1px solid {_C["warn_border"]};'
                     f'border-left:4px solid {_C["warn"]};border-radius:0 6px 6px 0;'
                     f'padding:10px 14px;margin:12px 0;font-size:12px;color:{_C["warn_text"]};">'
                     f'<b>&#9888; Low Liquidity:</b> ')
            parts = []
            for tkr, info in low_liq[:5]:
                vol = info.get("avg_daily_volume", 0)
                parts.append(f'{e(_tn(tkr, _names))} (vol {vol:,.0f})')
            h.append(", ".join(parts))
            h.append(' &mdash; position sizes reduced 40%</div>')

        # Sector gaps (canonical location, NOT duplicated in Opportunities)
        if sector_gaps:
            h.append(f'<div style="{_LABEL}color:{_C["warn"]};margin:16px 0 8px 0;">Sector Gaps</div>')
            for gap in sector_gaps[:4]:
                sec = gap.get("sector", gap.get("etf", "?"))
                assessment = gap.get("gap", gap.get("assessment", gap.get("note", "")))
                if not assessment:
                    recs = gap.get("recommended_stocks", [])
                    urg = gap.get("urgency", "")
                    gparts = []
                    if gap.get("portfolio_exposure"):
                        gparts.append(f"{gap['portfolio_exposure']} exposure")
                    if recs:
                        gparts.append(", ".join(str(r).split(" (")[0] for r in recs[:3]))
                    if urg:
                        gparts.append(f"[{urg}]")
                    assessment = " — ".join(gparts) if gparts else "Gap identified"
                inner = f'<b>{e(str(sec))}</b>: {e(str(assessment)[:100])}'
                h.append(_card(inner, _C["warn"], _C["warn_bg"]))

        h.append(_section_close())

    # ══════════════════════════════════════════════════════════════════
    # ACT III: DEEP CONTEXT (full mode only)
    # ══════════════════════════════════════════════════════════════════
    fund_stocks = fund.get("stocks", {})
    if not fund_stocks:
        fund_stocks = fund.get("stock_analyses", {})

    if not daily:
        h.append(_act_header("Research &amp; Analysis", "Detailed data behind the recommendations &mdash; for those who want to dig deeper"))

    # ── S9: FUNDAMENTAL DEEP DIVE ──
    quality_traps = fund.get("quality_traps", [])
    if fund_stocks and not daily:
        h.append(_section_open("Fundamental Deep Dive",
                               f"{len(fund_stocks)} stocks analyzed. Company financials: earnings quality, "
                               f"revenue growth, valuation, and financial health (Piotroski score 0-9, higher = healthier)."))
        # Top fundamental scores table
        h.append(f'<div style="{_LABEL}margin-bottom:8px;">Top Fundamental Scores</div>')
        h.append(_table_open([("Stock", "left"), ("Score", "center"), ("Piotroski", "center"),
                              ("Rev Growth", "center"), ("Quality", "center"),
                              ("PE T&#8594;F", "center"), ("EXRET", "center"),
                              ("Insider", "center"), ("Key Insight", "left")]))
        sorted_fund = sorted(fund_stocks.items(), key=lambda x: -x[1].get("fundamental_score", 0))
        for i, (tkr, fd) in enumerate(sorted_fund[:12]):
            fs = fd.get("fundamental_score", 0)
            eq_raw = fd.get("earnings_quality", "?")
            # earnings_quality may be a dict with {score, pe_trajectory, upside_pct, ...}
            if isinstance(eq_raw, dict):
                eq_traj = eq_raw.get("pe_trajectory", "")
                traj_icons = {"IMPROVING": "&#9650;", "DETERIORATING": "&#9660;", "STABLE": "&#9632;"}
                traj_colors = {"IMPROVING": _C["bull"], "DETERIORATING": _C["bear"], "STABLE": _C["text_muted"]}
                traj_icon = traj_icons.get(eq_traj, "")
                traj_color = traj_colors.get(eq_traj, _C["text_muted"])
                eq = f'<span style="color:{traj_color};">{traj_icon} {eq_traj[:3]}</span>' if eq_traj else "?"
                pet = fd.get("pe_trailing", fd.get("pet", 0)) or eq_raw.get("pet", 0)
                pef = fd.get("pe_forward", fd.get("pef", 0)) or eq_raw.get("pef", 0)
                ex = fd.get("exret", fd.get("upside_pct", 0)) or eq_raw.get("upside_pct", 0)
                note = eq_raw.get("notes", fd.get("notes", ""))[:120]
            else:
                eq = str(eq_raw)
                pet = fd.get("pe_trailing", fd.get("pet", 0))
                pef = fd.get("pe_forward", fd.get("pef", 0))
                ex = fd.get("exret", fd.get("upside_pct", 0))
                note = fd.get("notes", "")[:120]
            ins = fd.get("insider_sentiment", "N/A")
            if str(ins).upper() in ("UNKNOWN", "NONE", ""):
                ins = "N/A"
            if fs <= 10:
                fs_col = _C["bull"] if fs >= 6 else _C["warn"] if fs >= 4 else _C["text_muted"]
            else:
                fs_col = _C["bull"] if fs >= 80 else _C["warn"] if fs >= 60 else _C["text_muted"]
            ins_col = _C["bull"] if "BUY" in str(ins) else _C["bear"] if "SELL" in str(ins) else _C["text_muted"]
            bg = _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            pet = sf(pet, 0)
            pef = sf(pef, 0)
            ex = sf(ex, 0)
            import re as _re
            pe_match = _re.search(r'PE \w+ \(([\d.]+)(?:->|[\u2192\u2192]+)([\d.]+)\)', note) if note else None
            if pe_match:
                pet = sf(pe_match.group(1), 0)
                pef = sf(pe_match.group(2), 0)
            pe_str = f"{pet:.0f}&#8594;{pef:.0f}" if pet and pef else "N/A"
            pio = fd.get("piotroski", {})
            pio_score = pio.get("f_score", 0) if isinstance(pio, dict) else 0
            pio_col = _C["bull"] if pio_score >= 7 else _C["bear"] if pio_score <= 3 else _C["text_muted"]
            pio_str = f"{pio_score}/9" if pio_score else "-"
            rev_g = fd.get("revenue_growth", {})
            rev_cls = rev_g.get("classification", "") if isinstance(rev_g, dict) else ""
            rev_short = {"ACCELERATING": "ACC", "STRONG_GROWTH": "STRG", "STABLE": "STBL",
                         "DECLINING": "DEC", "DETERIORATING": "DET", "NEGATIVE": "NEG",
                         "POSITIVE": "POS", "N/A": "-"}.get(rev_cls.upper(), rev_cls[:4] if rev_cls else "-")
            rev_col = _C["bull"] if rev_cls.upper() in ("ACCELERATING", "STRONG_GROWTH") else _C["bear"] if rev_cls.upper() in ("DECLINING", "DETERIORATING", "NEGATIVE") else _C["text_muted"]
            h.append(_table_row([
                (_tn_cell(tkr, _names), "left", ""),
                (f'<span style="font-weight:800;color:{fs_col};">{fs:.0f}</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;color:{pio_col};">{pio_str}</span>', "center", ""),
                (f'<span style="color:{rev_col};font-size:10px;font-weight:600;">{rev_short}</span>', "center", ""),
                (f'<span style="font-size:10px;">{eq}</span>', "center", ""),
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
                if not reasons:
                    eq = fd.get("earnings_quality", {})
                    fs = fd.get("fundamental_score", 0)
                    sig = fd.get("signal", "")
                    trap_parts = []
                    if isinstance(eq, dict):
                        eq_score = eq.get("score", 0)
                        if eq_score <= 2:
                            trap_parts.append(f"Low EQ score ({eq_score}/5)")
                        am = eq.get("analyst_momentum", 0)
                        if am and sf(am) < 0:
                            trap_parts.append(f"Analysts souring (AM {am})")
                        traj = eq.get("pe_trajectory", "")
                        if traj == "DETERIORATING":
                            trap_parts.append("PE deteriorating")
                    if sig == "B" and fs < 60:
                        trap_parts.append(f"BUY signal but score only {fs:.0f}")
                    reasons = trap_parts
                reason_text = "; ".join(reasons[:2]) if reasons else "Metrics diverge from fundamentals"
                inner = (f'<span style="{_MONO}font-weight:700;color:{_C["bear_text"]};">{e(_tn(tkr, _names))}</span>'
                         f' <span style="font-size:11px;color:{_C["text_body"]};">{e(reason_text)}</span>')
                h.append(_card(inner, _C["bear"], _C["bear_bg"]))
        # CIO v23.4: FCF quality concerns
        weak_fcf = [(t, d.get("fcf_quality", {})) for t, d in fund_stocks.items()
                     if isinstance(d.get("fcf_quality"), dict) and d["fcf_quality"].get("concern") == "EARNINGS_QUALITY_CONCERN"]
        if weak_fcf:
            fcf_items = [f'{e(t)} ({fcf.get("conversion_ratio", 0):.2f}x)' for t, fcf in weak_fcf]
            h.append(f'<div style="margin-top:12px;"><span style="{_LABEL}color:{_C["warn"]};">Weak FCF Conversion: </span>'
                     f'{_pill_list(fcf_items, _C["warn_bg"], _C["warn_text"], _C["warn_border"], mono=False)}</div>')
        # CIO v23.4: High debt risk
        high_debt = [(t, d.get("debt_quality", {})) for t, d in fund_stocks.items()
                      if isinstance(d.get("debt_quality"), dict) and d["debt_quality"].get("risk") == "HIGH_RISK"]
        if high_debt:
            debt_tickers = [t for t, _ in high_debt]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">High Debt Risk: </span>'
                     f'{_pill_list(debt_tickers, _C["bear_bg"], _C["bear_text"], _C["bear_border"])}</div>')
        # CIO v23.3: EPS revision highlights
        eps_up = [(t, d.get("eps_revisions", {})) for t, d in fund_stocks.items()
                   if isinstance(d.get("eps_revisions"), dict) and d["eps_revisions"].get("classification") == "REVISIONS_UP"]
        eps_down = [(t, d.get("eps_revisions", {})) for t, d in fund_stocks.items()
                     if isinstance(d.get("eps_revisions"), dict) and d["eps_revisions"].get("classification") == "REVISIONS_DOWN"]
        if eps_up:
            eps_up_items = [f'{e(t)} +{eps.get("growth_pct", 0):.0f}%' for t, eps in eps_up]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bull"]};">EPS Revisions Up: </span>'
                     f'{_pill_list(eps_up_items, _C["bull_bg"], _C["bull_text"], _C["bull_border"], mono=False)}</div>')
        if eps_down:
            eps_down_items = [f'{e(t)} {eps.get("growth_pct", 0):+.0f}%' for t, eps in eps_down]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">EPS Revisions Down: </span>'
                     f'{_pill_list(eps_down_items, _C["bear_bg"], _C["bear_text"], _C["bear_border"], mono=False)}</div>')
        h.append(_section_close())

    # ── S10: TECHNICAL ANALYSIS ──
    if tech_stocks and not daily:
        h.append(_section_open("Technical Analysis",
                               f"{total_tech} stocks. Price momentum and trend signals &mdash; "
                               f"RSI measures momentum (below 30 = oversold opportunity, above 70 = overheated). "
                               f"ADX measures trend strength (above 30 = strong trend)."))
        # Table of technical signals
        h.append(_table_open([("Stock", "left"), ("RSI", "center"), ("MACD", "center"),
                              ("BB%", "center"), ("ADX", "center"), ("Div", "center"),
                              ("RS/SPY", "center"), ("ATR%", "center"),
                              ("Trend", "center"), ("Mom", "center"),
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
                           "STRONG_UPTREND": "STR UP",
                           "strong_downtrend": "STR DN", "weak_downtrend": "WK DN",
                           "consolidation": "CONSOL", "weak_uptrend": "WK UP",
                           "strong_uptrend": "STR UP",
                           "pullback_in_uptrend": "PULLBK", "recovery": "RECOV",
                           "strong_downtrend_oversold": "DN OVS",
                           "mixed": "MIXED"}.get(trend, trend[:7])
            adx_v = td.get("adx")
            adx_str = f"{adx_v:.0f}" if adx_v is not None else "-"
            adx_col = _C["bull"] if adx_v and adx_v >= 30 else _C["bear"] if adx_v and adx_v < 15 else _C["text_muted"]
            div_v = td.get("divergence", "")
            div_str = {"bullish": "BULL", "bearish": "BEAR"}.get(div_v, "-")
            div_col = _C["bull"] if div_v == "bullish" else _C["bear"] if div_v == "bearish" else _C["text_muted"]
            rs_v = td.get("relative_strength_vs_spy")
            rs_str = f"{rs_v:.2f}" if rs_v is not None else "-"
            rs_col = _C["bull"] if rs_v and rs_v > 1.0 else _C["bear"] if rs_v and rs_v < 0.95 else _C["text_muted"]
            atr_p = td.get("atr_pct")
            atr_str = f"{atr_p:.1f}%" if atr_p is not None else "-"
            atr_col = _C["bear"] if atr_p and atr_p > 5.0 else _C["warn"] if atr_p and atr_p > 3.0 else _C["text_muted"]
            h.append(_table_row([
                (_tn_cell(tkr, _names), "left", ""),
                (f'<span style="{_MONO}font-weight:700;color:{rsi_col};">{rsi_v:.0f}</span>', "center", ""),
                (f'<span style="color:{macd_col};font-size:10px;font-weight:600;">{macd[:4]}</span>', "center", ""),
                (f'<span style="{_MONO}font-size:10px;">{bb:.2f}</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;color:{adx_col};">{adx_str}</span>', "center", ""),
                (f'<span style="color:{div_col};font-size:10px;font-weight:600;">{div_str}</span>', "center", ""),
                (f'<span style="{_MONO}font-size:10px;color:{rs_col};">{rs_str}</span>', "center", ""),
                (f'<span style="{_MONO}font-size:10px;color:{atr_col};">{atr_str}</span>', "center", ""),
                (f'<span style="font-size:10px;">{trend_short}</span>', "center", ""),
                (f'<span style="{_MONO}font-weight:700;color:{mom_col};">{int(mom):+d}</span>', "center", ""),
                (f'<span style="color:{sig_col};font-weight:600;font-size:10px;">{abbr(sig)}</span>', "center", ""),
            ], bg=bg))
        h.append('</table>')
        # Oversold / Overbought callouts
        oversold = [t for t, d in tech_stocks.items() if d.get("rsi", 50) < 30]
        overbought = [t for t, d in tech_stocks.items() if d.get("rsi", 50) > 70]
        if oversold:
            h.append(f'<div style="margin-top:12px;"><span style="{_LABEL}color:{_C["bull"]};">Oversold (RSI&lt;30): </span>'
                     f'{_pill_list(oversold, _C["bull_bg"], _C["bull_text"], _C["bull_border"])}</div>')
        if overbought:
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">Overbought (RSI&gt;70): </span>'
                     f'{_pill_list(overbought, _C["bear_bg"], _C["bear_text"], _C["bear_border"])}</div>')
        # Divergence alerts (CIO v22.0 E4)
        bull_divs = [t for t, d in tech_stocks.items() if d.get("divergence") == "bullish"]
        bear_divs = [t for t, d in tech_stocks.items() if d.get("divergence") == "bearish"]
        if bull_divs:
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bull"]};">Bullish Divergence: </span>'
                     f'{_pill_list(bull_divs, _C["bull_bg"], _C["bull_text"], _C["bull_border"])}</div>')
        if bear_divs:
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">Bearish Divergence: </span>'
                     f'{_pill_list(bear_divs, _C["bear_bg"], _C["bear_text"], _C["bear_border"])}</div>')
        # CIO v23.3: Multi-timeframe confluence highlights
        strong_conf = [(t, d.get("confluence_score", 0)) for t, d in tech_stocks.items() if d.get("confluence_score", 0) >= 5]
        weak_conf = [(t, d.get("confluence_score", 0)) for t, d in tech_stocks.items() if d.get("confluence_score", 0) <= -3]
        if strong_conf:
            conf_items = [f'{e(t)} ({s:+d})' for t, s in strong_conf]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bull"]};">Strong Confluence (D+W): </span>'
                     f'{_pill_list(conf_items, _C["bull_bg"], _C["bull_text"], _C["bull_border"], mono=False)}</div>')
        if weak_conf:
            wconf_items = [f'{e(t)} ({s:+d})' for t, s in weak_conf]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">Conflicting Timeframes: </span>'
                     f'{_pill_list(wconf_items, _C["bear_bg"], _C["bear_text"], _C["bear_border"], mono=False)}</div>')
        # CIO v23.4: High IV rank alerts
        high_iv = [(t, d.get("iv_rank", 0)) for t, d in tech_stocks.items() if d.get("iv_rank") and d["iv_rank"] > 70]
        if high_iv:
            iv_items = [f'{e(t)} IV{iv:.0f}' for t, iv in high_iv]
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["warn"]};">High IV Rank (&gt;70): </span>'
                     f'{_pill_list(iv_items, _C["warn_bg"], _C["warn_text"], _C["warn_border"], mono=False)}</div>')
        # CIO v23.4: Volatility regime extremes
        extreme_vol = [t for t, d in tech_stocks.items() if d.get("volatility_regime") in ("HIGH_VOL", "EXTREME")]
        if extreme_vol:
            h.append(f'<div style="margin-top:8px;"><span style="{_LABEL}color:{_C["bear"]};">High Volatility: </span>'
                     f'{_pill_list(extreme_vol, _C["bear_bg"], _C["bear_text"], _C["bear_border"])}</div>')
        h.append(_section_close())

    # ── S11: SENTIMENT & CENSUS ──
    fg_label = lambda v: ("EXTREME GREED" if v >= 75 else "GREED" if v > 55 else "NEUTRAL" if v >= 45
                          else "FEAR" if v >= 25 else "EXTREME FEAR")
    fg_color = lambda v: (_C["bear"] if v >= 75 else _C["warn"] if v > 55 else _C["text_muted"] if v >= 45
                          else _C["info"] if v >= 25 else _C["hold"])
    cash100 = (synth.get("census_sentiment", {}).get("cash_top100")
               or census.get("cash_trends", {}).get("mean_cash_pct")
               or census.get("sentiment", {}).get("top100", {}).get("avg_cash_pct")
               or census.get("sentiment", {}).get("cash_top100")
               or 0)
    cash_label = "Defensive" if cash100 > 15 else "Deploying" if cash100 < 8 else "Normal"
    if not daily:
        h.append(_section_open("Sentiment &amp; Census",
                               "What other investors are doing. Fear &amp; Greed (0-100) measures crowd emotion: "
                               "below 25 = extreme fear (often a buying opportunity), above 75 = extreme greed (caution)."))
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
                         f'border:1px solid {_C["warn_border"]};">{e(_tn(tkr, _names))}</span>')
            h.append('</div>')
        h.append(_section_close())

    # ── S12: NEWS & EVENTS ──
    _skip_news = daily
    if not _skip_news:
        h.append(_section_open("News &amp; Events",
                               "Breaking news and upcoming events that could move your stocks. "
                               "Earnings reports are the biggest single-day movers for most stocks."))
    for item in (news.get("breaking_news", [])[:5] if not _skip_news else []):
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
            import re as _re_src
            domain = _re_src.sub(r'^https?://(www\.)?', '', source).split('/')[0]
            detail_cells.append(f'<span style="font-style:italic;">{e(domain)}</span>')
        detail_line = (f'<div style="font-size:10px;color:{_C["text_muted"]};margin-top:5px;'
                       f'line-height:1.6;">'
                       + " &middot; ".join(detail_cells) + '</div>')
        h.append(f'<div style="background:{nbg};border-left:3px solid {nbrd};border-radius:0 4px 4px 0;'
                 f'padding:10px 14px;margin-bottom:8px;">'
                 f'<div style="font-size:12px;font-weight:600;color:{nlc};">'
                 f'{e(hl)}</div>{detail_line}</div>')
    # Portfolio-specific news
    pn_raw = news.get("portfolio_news", {}) if not _skip_news else {}
    # Normalize: accept both dict {ticker: [items]} and list [{ticker, news_items}]
    if isinstance(pn_raw, list):
        pn = {}
        for entry in pn_raw:
            tkr = entry.get("ticker", "")
            items = entry.get("news_items", [entry] if "headline" in entry else [])
            if tkr and items:
                pn[tkr] = items
    else:
        pn = pn_raw
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
                (_tn_cell(tkr, _names), "left", ""),
                (f'<span style="color:{ic};font-weight:600;font-size:10px;">{e(imp)}</span>', "center", ""),
                (e(item.get("headline", "")[:80]), "left", f"color:{_C['text_body']};"),
            ]))
        h.append('</table>')
    # Earnings calendar
    earnings_cal = synth.get("earnings_calendar", {}) if not _skip_news else {}
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
                (_tn_cell(ear.get("ticker", ""), _names), "left", ""),
                (e(ear.get("expected_date", "?")), "center", f"font-size:11px;{_MONO}"),
                (f'<span style="font-weight:700;color:{rl_col};">{days}d</span>', "center", ""),
                (badge(rl, rl_col, "#fff"), "center", ""),
            ], bg=rbg))
        h.append('</table>')
    # Economic events
    econ_events = synth.get("economic_events", []) if not _skip_news else []
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
    if not _skip_news:
        h.append(_section_close())

    # ── S13: NEW OPPORTUNITIES ──
    opp_list = opps.get("top_opportunities", [])
    opp_stats = opps.get("screening_stats", {})
    if opp_list:
        if daily:
            # Daily: top 5 + sector gaps
            passed = opp_stats.get("unique_candidates") or len(opp_list)
            h.append(_section_open("New Opportunities",
                                   f"Top {min(5, len(opp_list))} from {passed} candidates."))
        else:
            passed = opp_stats.get("unique_candidates") or len(opp_list)
            h.append(_section_open("New Opportunities",
                                   f"Screened {opp_stats.get('universe_size', len(opp_list))} stocks. "
                                   f"{passed} passed filters."))
        display_count = 5 if daily else 10
        h.append(_table_open([("#", "center"), ("Stock", "left"), ("Sector", "left"),
                              ("Score", "center"), ("EXRET", "center"), ("PE F", "center"),
                              ("%BUY", "center"), ("Why Compelling", "left")]))
        for i, opp in enumerate(opp_list[:display_count]):
            score = opp.get("opportunity_score", 0)
            sc_col = _C["bull"] if score >= 60 else _C["warn"] if score >= 40 else _C["text_muted"]
            bg = _C["bull_bg"] if i < 3 else _C["bg_page"] if i % 2 == 1 else _C["bg_white"]
            exr = str(opp.get("exret", "0")).replace("%", "")
            pef = opp.get("pe_forward", 0)
            bp = opp.get("buy_pct", 0)
            why = opp.get("why_compelling", "")[:150]
            opp_sec = gics_sector(opp.get("sector", ""))
            h.append(_table_row([
                (f'<span style="font-weight:700;color:{_C["bull"]};">{i + 1}</span>', "center", ""),
                (_tn_cell(opp.get("ticker", ""), _names), "left", ""),
                (f'<span style="font-size:10px;">{e(opp_sec)}</span>', "left", ""),
                (f'<span style="{_MONO}font-weight:700;color:{sc_col};">{score:.0f}</span>', "center", ""),
                (f'{exr}%', "center", f"{_MONO}font-weight:600;"),
                (f'{pef:.1f}' if pef else "N/A", "center", _MONO),
                (f'{bp:.0f}%', "center", ""),
                (f'<span style="font-size:10px;color:{_C["text_muted"]};">{e(why)}</span>', "left", ""),
            ], bg=bg))
        h.append('</table>')

        # Sector gaps in daily mode (since they're not in Portfolio Construction for daily)
        if daily and sector_gaps:
            h.append(f'<div style="{_LABEL}color:{_C["warn"]};margin:16px 0 8px 0;">'
                     f'Portfolio Sector Gaps</div>')
            for gap in sector_gaps[:4]:
                sec = gap.get("sector", gap.get("etf", "?"))
                assessment = gap.get("gap", gap.get("assessment", gap.get("note", "")))
                if not assessment:
                    recs = gap.get("recommended_stocks", [])
                    urg = gap.get("urgency", "")
                    gparts = []
                    if gap.get("portfolio_exposure"):
                        gparts.append(f"{gap['portfolio_exposure']} exposure")
                    if recs:
                        gparts.append(", ".join(str(r).split(" (")[0] for r in recs[:3]))
                    if urg:
                        gparts.append(f"[{urg}]")
                    assessment = " — ".join(gparts) if gparts else "Gap identified"
                inner = f'<b>{e(str(sec))}</b>: {e(str(assessment)[:100])}'
                h.append(_card(inner, _C["warn"], _C["warn_bg"]))

        h.append(_section_close())

    # ══════════════════════════════════════════════════════════════════
    # EPILOGUE
    # ══════════════════════════════════════════════════════════════════

    # ── S14: WATCHLIST (CIO v23.3) ──
    watchlist = synth.get("watchlist", [])
    if watchlist and not daily:
        h.append(_section_open("Watchlist", "Opportunities below BUY threshold — re-entry triggers"))
        h.append(f'<table style="{_TABLE}">')
        h.append(f'<tr style="border-bottom:2px solid {_C["border"]};">'
                 f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Ticker</th>'
                 f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Sector</th>'
                 f'<th style="padding:6px 10px;text-align:center;{_LABEL}width:60px;">Conv</th>'
                 f'<th style="padding:6px 10px;text-align:left;{_LABEL}">Re-entry Trigger</th></tr>')
        for i, w in enumerate(watchlist[:10]):
            bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
            conv = w.get("conviction", 0)
            cc = _C["warn"] if conv >= 40 else _C["bear"]
            h.append(f'<tr style="background:{bg};">'
                     f'<td style="padding:6px 10px;">{_tn_cell(w.get("ticker", ""), _names)}</td>'
                     f'<td style="padding:6px 10px;font-size:11px;color:{_C["text_muted"]};">'
                     f'{e(w.get("sector", ""))}</td>'
                     f'<td style="padding:6px 10px;text-align:center;{_MONO}font-weight:700;color:{cc};">'
                     f'{conv}</td>'
                     f'<td style="padding:6px 10px;font-size:11px;">{e(w.get("watch_trigger", ""))}</td></tr>')
        h.append('</table>')
        h.append(_section_close())

    # ── S15: CHANGES SINCE LAST COMMITTEE ──
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
        display_limit = 5 if daily else 15
        h.append(_section_open("Changes Since Last Committee",
                               f"{len(sig_changes)} significant changes.",
                               border="1px solid " + _C["border_heavy"]))
        h.append(_table_open([("Stock", "left"), ("Previous", "center"),
                              ("Current", "center"), ("&#9651;", "center"), ("Type", "left")]))
        for c in sig_changes[:display_limit]:
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
                (_tn_cell(c.get("ticker", ""), _names), "left", ""),
                (prev_label, "center", "font-size:11px;"),
                (f'{e(c.get("curr_action", "?"))} ({c.get("curr_conviction", 0)})', "center", "font-size:11px;"),
                (f'<span style="font-weight:700;color:{tc};">{d:+d}</span>', "center", ""),
                (f'{ar} <span style="color:{tc};font-weight:600;">{e(ct)}</span>', "left", ""),
            ], bg=rbg, border_color=rbrd))
        h.append('</table>')
        h.append(_section_close())

    # ── S16: TRACK RECORD (moved to epilogue) ──
    perf = synth.get("performance", {})
    if perf and perf.get("status") == "complete" and perf.get("total_evaluated", 0) > 0 and not daily:
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

    # ── S17: SIGNAL REPORT CARD ──
    perf_data = synth.get("performance_data", synth.get("report_card", {})) or {}
    _scorecard_data = synth.get("scorecard", {}) or synth.get("performance", {}).get("scorecard", {})
    _calibration_data = synth.get("calibration_report", {}) or synth.get("performance", {}).get("calibration", {})
    _has_s17_content = (perf_data.get("actions") or _scorecard_data.get("buy_recommendations", {}).get("total", 0) > 0
                        or any(e.get("conviction_waterfall") for e in concordance))
    if _has_s17_content and not daily:
        h.append(_section_open("Signal Report Card", "Performance feedback on previous committee actions"))
        if perf_data.get("actions"):
            h.append(_table_open([("Action Type", "left"), ("Count", "center"), ("Avg Conviction", "center"),
                                  ("Avg Return", "center"), ("Hit Rate", "center")]))
            for act_type in ("BUY", "ADD", "HOLD", "TRIM", "SELL"):
                ad = perf_data.get("actions", {}).get(act_type, {})
                if not ad:
                    continue
                cnt = ad.get("count", 0)
                if cnt == 0:
                    continue
                avg_conv = ad.get("avg_conviction", 0)
                avg_ret = ad.get("avg_return", 0)
                hit_rate = ad.get("hit_rate", 0)
                hr_col = _C["bull"] if hit_rate >= 60 else _C["warn"] if hit_rate >= 40 else _C["bear"]
                ret_col = _C["bull"] if avg_ret > 0 else _C["bear"] if avg_ret < 0 else _C["text_muted"]
                h.append(_table_row([
                    (f'<span style="font-weight:700;">{act_type}</span>', "left", ""),
                    (str(cnt), "center", f"{_MONO}font-weight:600;"),
                    (f'{avg_conv:.0f}', "center", f"{_MONO}font-weight:600;"),
                    (f'<span style="color:{ret_col};font-weight:700;">{avg_ret:+.1f}%</span>', "center", ""),
                    (f'<span style="color:{hr_col};font-weight:700;">{hit_rate:.0f}%</span>', "center", ""),
                ]))
            h.append('</table>')

        # CIO v23.6: Benchmark comparison
        benchmark = synth.get("benchmark_comparison") or perf_data.get("benchmark_comparison", {})
        if benchmark.get("sufficient_data"):
            port_ret = benchmark.get("portfolio_return", 0)
            spy_ret = benchmark.get("spy_return", 0)
            excess = benchmark.get("excess_return", 0)
            info_r = benchmark.get("information_ratio", 0)
            bc = _C["bull"] if excess > 0 else _C["bear"]
            h.append(f'<div style="background:{_C["bg_alt"]};border:1px solid {_C["border"]};'
                     f'border-radius:6px;padding:12px 16px;margin:12px 0;font-size:12px;">'
                     f'<b>vs SPY Benchmark</b> ({benchmark.get("period_months", 3)}M) &mdash; '
                     f'Portfolio: <span style="color:{bc};font-weight:700;">{port_ret:+.1f}%</span> | '
                     f'SPY: {spy_ret:+.1f}% | '
                     f'Excess: <span style="color:{bc};font-weight:700;">{excess:+.1f}%</span> | '
                     f'IR: {info_r:.2f} | '
                     f'TE: {benchmark.get("tracking_error", 0):.1f}%</div>')

        # ── Scorecard: Historical T+7/T+30 Hit Rates ──
        scorecard = _scorecard_data
        if scorecard and scorecard.get("buy_recommendations", {}).get("total", 0) > 0:
            buy_sc = scorecard.get("buy_recommendations", {})
            sell_sc = scorecard.get("sell_recommendations", {})
            hold_sc = scorecard.get("hold_recommendations", {})
            h.append(f'<div style="margin-top:16px;"><b style="font-size:12px;">Historical Performance</b>'
                     f'<span style="font-size:10px;color:{_C["text_muted"]};margin-left:8px;">'
                     f'T+7 and T+30 hit rates ({scorecard.get("period_months", 3)}M window)</span></div>')
            h.append(_table_open([("Type", "left"), ("Count", "center"),
                                  ("Hit T+7", "center"), ("Hit T+30", "center"),
                                  ("Avg T+30", "right")]))
            for label, sc_data in [("BUY/ADD", buy_sc), ("SELL/TRIM", sell_sc), ("HOLD", hold_sc)]:
                total = sc_data.get("total", 0)
                if total == 0:
                    continue
                hr7 = sc_data.get("hit_rate_7d", sc_data.get("avg_return_7d"))
                hr30 = sc_data.get("hit_rate_30d", sc_data.get("validated_30d"))
                avg30 = sc_data.get("avg_return_30d", sc_data.get("avg_avoided_loss", 0))
                hr7_s = f'{hr7:.0f}%' if hr7 is not None else "—"
                hr30_s = f'{hr30:.0f}%' if hr30 is not None else "—"
                avg30_s = f'{avg30:+.1f}%' if avg30 is not None else "—"
                hr30_col = _C["bull"] if (hr30 or 0) >= 50 else _C["bear"] if hr30 is not None else _C["text_muted"]
                avg_col = _C["bull"] if (avg30 or 0) > 0 else _C["bear"] if (avg30 or 0) < 0 else _C["text_muted"]
                h.append(_table_row([
                    (f'<span style="font-weight:700;">{label}</span>', "left", ""),
                    (str(total), "center", f"{_MONO}font-weight:600;"),
                    (hr7_s, "center", _MONO),
                    (f'<span style="color:{hr30_col};font-weight:700;">{hr30_s}</span>', "center", ""),
                    (f'<span style="color:{avg_col};font-weight:700;">{avg30_s}</span>', "right", ""),
                ]))
            h.append('</table>')

            # Conviction calibration insight
            conv_cal = scorecard.get("conviction_calibration", {})
            if conv_cal.get("sufficient_data"):
                predictive = conv_cal.get("conviction_predictive", False)
                spread = conv_cal.get("conviction_spread", 0)
                icon = "&#10004;" if predictive else "&#10008;"
                cal_col = _C["bull"] if predictive else _C["warn"]
                h.append(f'<div style="font-size:11px;color:{_C["text_muted"]};margin:8px 0 0 0;">'
                         f'<span style="color:{cal_col};">{icon}</span> '
                         f'Conviction calibration: high-conviction stocks '
                         f'{"outperform" if predictive else "underperform"} low-conviction '
                         f'by {spread:+.1f}pp at T+30</div>')

        # Calibration report (advisory, from weekly backtest)
        calibration = _calibration_data
        if calibration and calibration.get("sufficient_data") and calibration.get("modifiers"):
            mods = calibration["modifiers"]
            keep = [(k, v) for k, v in mods.items() if v.get("recommendation") == "KEEP"]
            remove = [(k, v) for k, v in mods.items() if v.get("recommendation") == "REMOVE"]
            adjust = [(k, v) for k, v in mods.items() if v.get("recommendation") == "ADJUST"]

            h.append(f'<div style="margin-top:16px;"><b style="font-size:12px;">Parameter Health Check</b>'
                     f'<span style="font-size:10px;color:{_C["text_muted"]};margin-left:8px;">'
                     f'Modifier effectiveness from calibration</span></div>')

            if remove:
                h.append(f'<div style="background:{_C["bear_bg"]};border:1px solid {_C["bear_border"]};'
                         f'border-radius:6px;padding:8px 12px;margin:8px 0;font-size:11px;">'
                         f'<b style="color:{_C["bear"]};">Remove ({len(remove)})</b>: ')
                h.append(', '.join(f'{k.replace("_"," ")} ({v.get("return_delta",0):+.1f}pp)'
                                  for k, v in remove))
                h.append('</div>')

            if adjust:
                h.append(f'<div style="background:{_C["info_bg"]};border:1px solid {_C["info_border"]};'
                         f'border-radius:6px;padding:8px 12px;margin:8px 0;font-size:11px;">'
                         f'<b>Adjust ({len(adjust)})</b>: ')
                h.append(', '.join(f'{k.replace("_"," ")}' for k, v in adjust))
                h.append('</div>')

            if keep:
                h.append(f'<div style="background:{_C["bull_bg"]};border:1px solid {_C["bull_border"]};'
                         f'border-radius:6px;padding:8px 12px;margin:8px 0;font-size:11px;">'
                         f'<b style="color:{_C["bull"]};">Keep ({len(keep)})</b>: ')
                h.append(', '.join(f'{k.replace("_"," ")} ({v.get("return_delta",0):+.1f}pp)'
                                  for k, v in keep))
                h.append('</div>')

        # CIO v25.0: Waterfall aggregation — which modifiers fire most across portfolio
        wf_agg = {}
        wf_counts = {}
        for entry in concordance:
            wf = entry.get("conviction_waterfall", {})
            for k, v in wf.items():
                if k.startswith("_"):
                    continue
                wf_agg[k] = wf_agg.get(k, 0) + v
                wf_counts[k] = wf_counts.get(k, 0) + 1
        if wf_agg:
            h.append(f'<div style="margin-top:16px;"><b style="font-size:12px;">Modifier Attribution</b>'
                     f'<span style="font-size:10px;color:{_C["text_muted"]};margin-left:8px;">'
                     f'Net impact across {len(concordance)} stocks</span></div>')
            h.append(_table_open([("Modifier", "left"), ("Fires", "center"),
                                  ("Avg", "center"), ("Net", "center")]))
            for k, net in sorted(wf_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:15]:
                cnt = wf_counts[k]
                avg = net / cnt if cnt else 0
                net_col = _C["bull"] if net > 0 else _C["bear"] if net < 0 else _C["text_muted"]
                label = k.replace("_", " ")
                h.append(_table_row([
                    (f'<span style="{_MONO}font-size:11px;">{e(label)}</span>', "left", ""),
                    (str(cnt), "center", f"{_MONO}font-weight:600;"),
                    (f'{avg:+.1f}', "center", f"{_MONO}"),
                    (f'<span style="color:{net_col};font-weight:700;">{net:+.0f}</span>', "center", ""),
                ]))
            h.append('</table>')

        h.append(_section_close())

    # ── S18: REGIME TRANSITION (CIO v23.3) ──
    regime_trans = synth.get("regime_transition", {})
    if regime_trans.get("transition") and regime_trans["transition"] != "INSUFFICIENT_DATA" and not daily:
        h.append(_section_open("Regime Transition",
                               "Are things getting better or worse? Tracks how average conviction "
                               "scores are trending over multiple committee runs."))
        trans = regime_trans["transition"]
        if trans == "IMPROVING":
            tc, tbg, tborder = _C["bull"], _C["bull_bg"], _C["bull_border"]
        elif trans == "DETERIORATING":
            tc, tbg, tborder = _C["bear"], _C["bear_bg"], _C["bear_border"]
        else:
            tc, tbg, tborder = _C["text_body"], _C["info_bg"], _C["info_border"]
        history = regime_trans.get("history", [])
        h.append(f'<div style="background:{tbg};border:1px solid {tborder};'
                 f'border-left:4px solid {tc};border-radius:0 6px 6px 0;'
                 f'padding:10px 14px;margin:0 0 16px 0;font-size:12px;">'
                 f'<b style="color:{tc};">Regime Transition: {e(trans)}</b> '
                 f'&mdash; 3-run MA: {regime_trans.get("short_ma", 0):.1f} vs '
                 f'7-run MA: {regime_trans.get("long_ma", 0):.1f} '
                 f'(spread: {regime_trans.get("spread", 0):+.1f})')
        if history:
            pts = " &rarr; ".join(f'{h_entry.get("avg_conviction", 0):.0f}' for h_entry in history[-5:])
            h.append(f'<br/><span style="color:{_C["text_muted"]};font-size:11px;">'
                     f'Conviction trend: {pts}</span>')
        h.append('</div>')
        h.append(_section_close())

    # ── GLOSSARY (casual investor reference) ──
    if not daily:
        _gloss = [
            ("RSI", "Relative Strength Index &mdash; momentum indicator from 0-100. Below 30 = oversold (potential buying opportunity). Above 70 = overbought (may be overheated)."),
            ("MACD", "Moving Average Convergence/Divergence &mdash; trend-following signal. Bullish = upward momentum, Bearish = downward."),
            ("EXRET", "Expected Return &mdash; how much a stock is expected to return above the market average, based on analyst price targets."),
            ("VaR", "Value at Risk &mdash; the maximum expected loss over a period at 95% confidence. A VaR of -8% means there's only a 5% chance you'd lose more than 8%."),
            ("CVaR", "Conditional VaR &mdash; the average loss in the worst 5% of scenarios. Worse than VaR, shows the tail risk."),
            ("Sharpe Ratio", "Risk-adjusted return &mdash; how much return you get per unit of risk. Above 1.0 = good, above 2.0 = excellent, below 0 = losing money."),
            ("Beta", "Market sensitivity &mdash; 1.0 means the stock moves with the market. Above 1.5 = more volatile than market. Below 0.8 = more stable."),
            ("Piotroski F-Score", "Financial health score from 0-9 based on 9 accounting tests. 7+ = financially strong, 3 or below = red flag."),
            ("ADX", "Average Directional Index &mdash; measures trend strength (not direction). Above 30 = strong trend, below 15 = no clear trend."),
            ("F&amp;G", "Fear &amp; Greed Index &mdash; market sentiment from 0-100. Extreme fear (below 25) often = buying opportunity. Extreme greed (above 75) = caution."),
            ("Conviction Score", "Committee confidence from 0-100. Combines all 7 analysts' views, adjusted for market conditions, risk, and data quality."),
            ("Kill Thesis", "The specific scenario that would prove the trade recommendation wrong &mdash; a built-in reality check for every recommendation."),
            ("Entry Timing", "ENTER NOW = favorable technical setup. WAIT FOR PULLBACK = good stock but wait for a lower price. AVOID/EXIT = unfavorable."),
            ("ATR", "Average True Range &mdash; daily price volatility in percentage terms. Higher ATR = more volatile stock."),
        ]
        h.append(f'<div style="padding:24px 40px;border-bottom:1px solid {_C["border"]};">')
        h.append(f'<h2 style="{_SECTION_H2}">Glossary</h2>')
        h.append(f'<p style="font-size:11px;color:{_C["text_muted"]};margin:0 0 12px 0;">'
                 f'Key terms used in this report, explained in plain language.</p>')
        h.append(f'<table style="{_TABLE}">')
        for i, (term, defn) in enumerate(_gloss):
            bg = _C["bg_page"] if i % 2 == 0 else _C["bg_white"]
            h.append(f'<tr style="background:{bg};">'
                     f'<td style="padding:6px 10px;font-weight:700;{_MONO}font-size:11px;'
                     f'color:{_C["text_dark"]};white-space:nowrap;vertical-align:top;width:120px;">{term}</td>'
                     f'<td style="padding:6px 10px;font-size:11px;color:{_C["text_body"]};'
                     f'line-height:1.5;">{defn}</td></tr>')
        h.append('</table></div>')

    # ── FOOTER ──
    h.append(f'<div style="background:{_C["bg_page"]};padding:20px 40px;border-top:2px solid {_C["border"]};">'
             f'<table style="width:100%;"><tr>'
             f'<td style="vertical-align:top;">'
             f'<div style="font-size:12px;font-weight:700;color:{_C["text_dark"]};">Investment Committee v26.1</div>'
             f'<div style="font-size:10px;color:{_C["text_muted"]};margin-top:2px;">'
             f'{today_long} &middot; 7 Agents (Sonnet) + CIO (Opus)</div>'
             f'<div style="font-size:10px;color:{_C["text_light"]};margin-top:2px;">'
             f'Signals: {e(signal_date)} &middot; Census: {e(census_date)}</div></td>'
             f'<td style="text-align:right;vertical-align:top;">'
             f'<div style="font-size:10px;color:{_C["text_light"]};font-style:italic;">Not financial advice.</div>'
             f'</td></tr></table></div>')
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
    mode: str = "full",
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

    # Build name map — CSV names are truncated (10 char), so enrich with yfinance
    name_map: Dict[str, str] = {}
    # 1. Seed from CSV (truncated but always available)
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
    # 2. Collect ALL tickers that will appear in the report
    all_tickers = set()
    for en in synth.get("concordance", []):
        if en.get("ticker"):
            all_tickers.add(en["ticker"])
    for opp in opps.get("top_opportunities", []):
        if opp.get("ticker"):
            all_tickers.add(opp["ticker"])
    for t in fund.get("stocks", fund.get("stock_analyses", {})):
        all_tickers.add(t)
    for t in tech.get("stocks", tech.get("stock_analyses", {})):
        all_tickers.add(t)
    # 3. Resolve full names via yfinance for tickers with truncated or missing names
    tickers_needing_names = [
        t for t in all_tickers
        if t and (t not in name_map or len(name_map.get(t, "")) <= 10)
    ]
    if tickers_needing_names:
        try:
            import yfinance as yf
            # Process in batches of 20 to avoid yfinance rate limits
            for batch_start in range(0, min(len(tickers_needing_names), 80), 20):
                batch = tickers_needing_names[batch_start:batch_start + 20]
                try:
                    data = yf.Tickers(" ".join(batch))
                    for tkr in batch:
                        try:
                            info = data.tickers[tkr].info
                            short = info.get("shortName") or info.get("longName") or ""
                            existing = name_map.get(tkr, "")
                            # Prefer yfinance name: it's typically fuller and properly cased
                            if short and (not existing or existing.endswith(".")
                                         or len(short) >= len(existing)):
                                name_map[tkr] = short
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    html_str = generate_report_html(synth, fund, tech, macro, census, news, opps, risk,
                                    name_map=name_map, mode=mode)

    today = datetime.now().strftime("%Y-%m-%d")
    suffix = "-daily" if mode == "daily" else ""
    output_path = od / f"{today}{suffix}.html"
    od.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_str)

    # Archive concordance for backtesting (dated copy, skip for daily digest)
    if mode != "daily":
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
        "version": synth.get("version", "v25.0"),
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
                "conviction_waterfall": en.get("conviction_waterfall"),  # CIO v25.0
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

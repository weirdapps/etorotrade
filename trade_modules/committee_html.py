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
    m = {"SELL":"#dc2626","IMMEDIATE SELL":"#dc2626","REDUCE":"#dc2626","TRIM":"#d97706","WEAK HOLD":"#d97706","BUY":"#059669","ADD":"#059669","BUY NEW":"#0d9488","HOLD":"#6366f1","STRONG HOLD":"#6366f1","WATCH":"#2563eb"}
    return m.get(act, "#64748b")

def action_bg(act):
    m = {"SELL":"#fef2f2","IMMEDIATE SELL":"#fef2f2","REDUCE":"#fef2f2","TRIM":"#fffbeb","WEAK HOLD":"#fffbeb","BUY":"#ecfdf5","ADD":"#ecfdf5","BUY NEW":"#ecfdf5","HOLD":"#ffffff","STRONG HOLD":"#f0fdf4","WATCH":"#eff6ff"}
    return m.get(act, "#ffffff")

def action_border(act):
    m = {"SELL":"#fecaca","IMMEDIATE SELL":"#fecaca","REDUCE":"#fecaca","TRIM":"#fde68a","WEAK HOLD":"#fde68a","BUY":"#a7f3d0","ADD":"#a7f3d0","BUY NEW":"#a7f3d0","HOLD":"#e2e8f0","STRONG HOLD":"#bbf7d0","WATCH":"#bfdbfe"}
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

def conv_bar(c):
    col = conv_color(c)
    return f'<div style="font-weight:800;font-size:13px;color:#0f172a;">{c}</div><div style="height:3px;background:#e2e8f0;border-radius:2px;margin-top:3px;width:40px;display:inline-block;"><div style="height:100%;border-radius:2px;background:{col};width:{min(c,100)}%;"></div></div>'

def signal_badge(sig):
    c = {"B":"#059669","H":"#d97706","S":"#dc2626","I":"#94a3b8"}
    return f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{c.get(sig,"#94a3b8")};">{sig}</span>'

def dot(color):
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"></span>'

def agent_badge(name, view, cbg, ctxt, cbrd):
    return f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:{cbg};color:{ctxt};border:1px solid {cbrd};">{name}: {view}</span>'

def e(text):
    return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

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
    var_95 = synth.get("var_95", 0)
    max_dd = synth.get("max_drawdown", 0)
    p_beta = synth.get("portfolio_beta", 1.0)
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

    action_order = {"IMMEDIATE SELL":0,"SELL":1,"REDUCE":2,"TRIM":3,"BUY NEW":4,"BUY":5,"ADD":6,"STRONG HOLD":7,"HOLD":8,"WEAK HOLD":9,"WATCH":10}
    concordance.sort(key=lambda x: (action_order.get(x.get("action","HOLD"),9), -x.get("conviction",0)))

    sells = sum(1 for c in concordance if c.get("action") in ("SELL","IMMEDIATE SELL","REDUCE"))
    buys = sum(1 for c in concordance if c.get("action") in ("BUY","ADD","BUY NEW"))
    trims = sum(1 for c in concordance if c.get("action") == "TRIM")

    if regime == "RISK_ON":
        verdict = "RISK-ON"; vbg,vbrd,vval,vacc = "#ecfdf5","#a7f3d0","#059669","#059669"
    elif regime == "RISK_OFF":
        verdict = "DEFENSIVE"; vbg,vbrd,vval,vacc = "#fef2f2","#fecaca","#dc2626","#dc2626"
    else:
        verdict = "CAUTIOUS"; vbg,vbrd,vval,vacc = "#fffbeb","#fde68a","#d97706","#d97706"

    risk_color = "#dc2626" if risk_score >= 70 else "#d97706" if risk_score >= 40 else "#059669"

    sell_tickers = [c["ticker"] for c in concordance if c.get("action") in ("SELL","REDUCE","IMMEDIATE SELL")]
    buy_tickers = [c["ticker"] for c in concordance if c.get("action") in ("BUY","ADD") and c.get("conviction",0)>=70][:3]
    new_tickers = [c["ticker"] for c in concordance if c.get("action")=="BUY NEW"]

    narrative = f"Market regime is {regime} (score {macro_score}) in {rotation.replace('_',' ').lower()} phase. "
    narrative += f"VIX at {indicators.get('vix',0):.1f}, 10Y yield {indicators.get('yield_10y',0):.2f}%. "
    if sells: narrative += f"Reduce {len(sell_tickers)} positions ({', '.join(sell_tickers)}). "
    if buy_tickers: narrative += f"Top buys: {', '.join(buy_tickers)}. "
    if new_tickers: narrative += f"New opportunities: {', '.join(new_tickers)}. "

    # Disagreements
    disagreements = []
    for entry in concordance:
        fv,ts = entry.get("fund_view",""), entry.get("tech_signal","")
        if (fv == "BUY" and ts in ("AVOID","EXIT_SOON")) or (fv == "SELL" and ts == "ENTER_NOW"):
            tkr = entry.get("ticker","")
            disagreements.append({"ticker":tkr,"headline":f"Fund {fv} ({entry.get('fund_score',0):.0f}) vs Tech {ts} (RSI {entry.get('rsi',0):.0f})","fund_view":fv,"tech_signal":ts,"narrative":f"Fundamental scores {tkr} at {entry.get('fund_score',0):.0f}/100 ({fv}), EXRET {entry.get('exret',0):.1f}%. Technical: RSI {entry.get('rsi',0):.0f}, momentum {entry.get('tech_momentum',0)}, signal {ts}. Macro fit: {entry.get('macro_fit','?')}.","resolution":entry.get("action",""),"conviction":entry.get("conviction",0),"accent":action_color(entry.get("action",""))})
    if not disagreements:
        for entry in concordance:
            bp = entry.get("bull_pct",50)
            if 35 < bp < 65 and entry.get("action") not in ("HOLD",):
                tkr = entry.get("ticker","")
                disagreements.append({"ticker":tkr,"headline":f"Split: {entry.get('bull_weight',0):.1f} bull vs {entry.get('bear_weight',0):.1f} bear","fund_view":entry.get("fund_view",""),"tech_signal":entry.get("tech_signal",""),"narrative":f"Committee split on {tkr}. Fund: {entry.get('fund_view','')} ({entry.get('fund_score',0):.0f}), Tech: {entry.get('tech_signal','')} (RSI {entry.get('rsi',0):.0f}), Macro: {entry.get('macro_fit','')}. Bull weight {entry.get('bull_weight',0):.1f} vs bear {entry.get('bear_weight',0):.1f}.","resolution":entry.get("action",""),"conviction":entry.get("conviction",0),"accent":action_color(entry.get("action",""))})
                if len(disagreements)>=6: break

    h = []
    h.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>')
    h.append('<body style="margin:0;padding:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,\'Helvetica Neue\',Arial,sans-serif;color:#334155;line-height:1.6;-webkit-font-smoothing:antialiased;">')
    h.append('<div style="max-width:960px;margin:0 auto;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.1);">')

    # HEADER
    h.append(f'<div style="background-color:#0f172a;padding:36px 40px 32px 40px;"><table style="width:100%;"><tr><td style="vertical-align:middle;width:50px;"><div style="width:36px;height:36px;border-radius:8px;background:rgba(255,255,255,0.15);text-align:center;line-height:36px;font-size:18px;color:#fff;font-weight:800;letter-spacing:-1px;">IC</div></td><td style="vertical-align:middle;"><h1 style="margin:0;font-size:26px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Investment Committee</h1><p style="margin:2px 0 0 0;font-size:13px;color:#94a3b8;">{today_long} &middot; 7 Specialist Agents + CIO Synthesis v5.4</p></td></tr></table><div style="margin-top:16px;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Signals: 2026-03-16</span> <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Census: 2026-03-16</span> <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Macro: {rotation.replace("_"," ")}</span></div></div>')

    # S1: EXECUTIVE SUMMARY
    h.append(f'<div style="padding:32px 40px;border-bottom:2px solid #cbd5e1;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Executive Summary</h2>')
    h.append(f'<table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;"><tr>')
    h.append(f'<td style="width:33%;background:{vbg};border:1px solid {vbrd};border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{vval};margin-bottom:6px;">CIO Verdict</div><div style="font-size:28px;font-weight:800;color:{vval};">{verdict}</div><div style="font-size:11px;color:#64748b;margin-top:4px;">{sells} reduce, {trims} trim, {buys} buy/add</div></td>')
    h.append(f'<td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Macro Regime</div><div style="font-size:28px;font-weight:800;color:{vval};">{regime}</div><div style="font-size:11px;color:#64748b;margin-top:4px;">Score: {macro_score} &middot; {rotation.replace("_"," ")}</div></td>')
    h.append(f'<td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Portfolio Risk</div><div style="font-size:28px;font-weight:800;color:{risk_color};">{risk_score}/100</div><div style="font-size:11px;color:#64748b;margin-top:4px;">Beta {p_beta:.2f} &middot; VaR {var_95:.2f}%</div></td>')
    h.append('</tr></table>')
    h.append(f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid {vacc};border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:20px;"><div style="font-size:13px;color:#334155;line-height:1.6;">{e(narrative)}</div></div>')
    h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Priority Actions</div><table style="width:100%;border-collapse:separate;border-spacing:0 4px;font-size:13px;">')
    for entry in concordance[:8]:
        act,tkr,conv = entry.get("action","HOLD"),entry.get("ticker",""),entry.get("conviction",0)
        detail = f"Sig:{entry.get('signal','?')} EXRET:{entry.get('exret',0):.1f}% {entry.get('sector','')}"
        ac,ab,abr = action_color(act),action_bg(act),action_border(act)
        h.append(f'<tr><td style="padding:8px 12px;border:1px solid {abr};background:{ab};border-radius:6px 0 0 6px;width:80px;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:{ac};">{act}</span></td><td style="padding:8px 12px;border:1px solid {abr};background:{ab};font-weight:700;font-family:monospace;width:80px;">{e(tkr)}</td><td style="padding:8px 12px;border:1px solid {abr};background:{ab};color:#334155;font-size:12px;">{e(detail)}</td><td style="padding:8px 12px;border:1px solid {abr};background:{ab};text-align:center;font-weight:700;border-radius:0 6px 6px 0;width:40px;color:{conv_color(conv)};">{conv}</td></tr>')
    h.append('</table></div>')

    # S2: MACRO ENVIRONMENT
    macro_ind = [("10Y Yield",f"{indicators.get('yield_10y',0):.2f}%",indicators.get('yield_curve_status','NORMAL')),("Yield Curve",f"{indicators.get('yield_curve_spread',0):.0f}bps","POSITIVE" if sf(indicators.get('yield_curve_spread',0))>0 else "INVERTED"),("VIX",f"{indicators.get('vix',0):.1f}","ELEVATED" if sf(indicators.get('vix',0))>20 else "NORMAL"),("Dollar (DXY)",f"{indicators.get('dxy',0):.1f}",indicators.get('dollar_trend','STABLE')),("Macro Score",f"{macro_score}",regime)]
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

    # S3: STOCK ANALYSIS GRID
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Stock Analysis Grid</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Concordance matrix: 7 specialist views per stock.</p><div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap;"><tr><th style="padding:8px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;border:1px solid #1e293b;">STOCK</th><th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;border:1px solid #1e293b;">SIG</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#93c5fd;font-size:10px;font-weight:600;border:1px solid #334155;">FUND</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#c4b5fd;font-size:10px;font-weight:600;border:1px solid #334155;">TECH</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#fcd34d;font-size:10px;font-weight:600;border:1px solid #334155;">MACRO</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#6ee7b7;font-size:10px;font-weight:600;border:1px solid #334155;">CENSUS</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#fca5a5;font-size:10px;font-weight:600;border:1px solid #334155;">NEWS</th><th style="padding:8px 6px;text-align:center;background:#1e293b;color:#d6d3d1;font-size:10px;font-weight:600;border:1px solid #334155;">RISK</th><th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;border:1px solid #1e293b;">VERDICT</th><th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;border:1px solid #1e293b;">CONV</th></tr>')
    cg = None
    for entry in concordance:
        act = entry.get("action","HOLD")
        if act in ("SELL","IMMEDIATE SELL","REDUCE","TRIM") and cg!="SELL":
            cg="SELL"; h.append('<tr><td colspan="10" style="padding:4px 8px;background:#fef2f2;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#991b1b;border:1px solid #fecaca;">SELL / REDUCE / TRIM</td></tr>')
        elif act in ("BUY","ADD","BUY NEW") and cg!="BUY":
            cg="BUY"; h.append('<tr><td colspan="10" style="padding:4px 8px;background:#ecfdf5;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#065f46;border:1px solid #a7f3d0;">BUY / ADD / NEW</td></tr>')
        elif act in ("HOLD","STRONG HOLD","WEAK HOLD","WATCH") and cg!="HOLD":
            cg="HOLD"; h.append('<tr><td colspan="10" style="padding:4px 8px;background:#f1f5f9;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#475569;border:1px solid #cbd5e1;">HOLD / MONITOR</td></tr>')
        tkr=entry.get("ticker",""); sig=entry.get("signal","?"); fs=entry.get("fund_score",0); fv=entry.get("fund_view","?"); ts=entry.get("tech_signal","?"); rsi=entry.get("rsi",0); mf=entry.get("macro_fit","?"); ce=entry.get("census","?"); ni=entry.get("news_impact","?"); rw="WARN" if entry.get("risk_warning") else "OK"; conv=entry.get("conviction",0); sm="*" if entry.get("fund_synthetic") else ""; rb=action_bg(act)
        h.append(f'<tr style="background:{rb};"><td style="padding:7px 8px;border:1px solid #e2e8f0;font-family:monospace;font-weight:700;font-size:12px;color:#0f172a;">{e(tkr)}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;">{signal_badge(sig)}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(fv)};font-weight:600;font-size:10px;">{fv}({fs:.0f}){sm}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(ts)};font-weight:600;font-size:10px;">{ts[:6]}({rsi:.0f})</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(mf)};font-weight:600;font-size:10px;">{mf[:5]}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(ce)};font-weight:600;font-size:10px;">{ce[:5]}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(ni)};font-weight:600;font-size:10px;">{ni[:5]}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{sentiment_color(rw)};font-weight:600;font-size:10px;">{rw}</td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;"><span style="display:inline-block;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:{action_color(act)};">{act}</span></td><td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;">{conv_bar(conv)}</td></tr>')
    h.append('</table></div><div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;font-size:10px;color:#64748b;">* = Synthetic fundamental score &middot; WAIT = Wait for pullback &middot; ENTER = Enter now &middot; AVOID = Avoid &middot; FAVOR = Macro favorable &middot; ALIGN = Census aligned</div></div>')

    # S4: DISAGREEMENTS
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Where We Disagreed</h2><p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">Structured disagreement is the committee\'s edge.</p>')
    for d in disagreements[:6]:
        fb = agent_badge("Fund",d["fund_view"],"#dbeafe","#1e40af","#93c5fd")
        tb = agent_badge("Tech",d["tech_signal"],"#ede9fe","#5b21b6","#c4b5fd")
        h.append(f'<div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid {d["accent"]};border-radius:0 8px 8px 0;padding:20px 24px;margin-bottom:16px;"><div style="margin-bottom:10px;"><span style="font-family:monospace;font-weight:800;font-size:14px;color:#0f172a;">{e(d["ticker"])}</span> <span style="font-size:13px;font-weight:600;color:#334155;margin-left:8px;">{e(d["headline"])}</span></div><div style="margin-bottom:12px;">{fb}{tb}</div><div style="font-size:13px;color:#334155;line-height:1.6;margin-bottom:12px;">{e(d["narrative"])}</div><div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;"><span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Resolution:</span> <span style="font-size:13px;font-weight:700;color:{action_color(d["resolution"])};">{d["resolution"]} (conv. {d["conviction"]})</span></div></div>')
    if not disagreements:
        h.append('<div style="padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;color:#64748b;font-size:13px;">No significant disagreements. All agents broadly aligned.</div>')
    h.append('</div>')

    # S5: FUNDAMENTAL DEEP DIVE
    fund_stocks = fund.get("stocks",{})
    sorted_fund = sorted(fund_stocks.items(), key=lambda x:x[1].get("fundamental_score",0), reverse=True)
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Fundamental Deep Dive</h2><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Score</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">PE Traj</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">EXRET</th><th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Insider</th><th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Notes</th></tr>')
    for i,(tkr,data) in enumerate(sorted_fund[:12]):
        bg="#f8fafc" if i%2==1 else "#fff"; sc=data.get("fundamental_score",0); pt=data.get("pe_trajectory","?")[:20]; ex=data.get("exret",0); ins=data.get("insider_sentiment","NEUTRAL"); notes=data.get("notes","")[:60]
        ic = "#059669" if ins=="NET_BUYING" else "#dc2626" if ins=="NET_SELLING" else "#64748b"
        h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:8px 10px;font-family:monospace;font-weight:700;">{e(tkr)}</td><td style="padding:8px 10px;text-align:center;font-weight:800;color:{conv_color(int(sc))};">{sc:.0f}</td><td style="padding:8px 10px;text-align:center;font-size:11px;">{e(pt)}</td><td style="padding:8px 10px;text-align:center;">{ex}%</td><td style="padding:8px 10px;text-align:center;color:{ic};font-weight:600;font-size:10px;">{e(ins)}</td><td style="padding:8px 10px;color:#334155;font-size:11px;">{e(notes)}</td></tr>')
    h.append('</table>')
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
    h.append('<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Technical Analysis</h2>')
    if bearish_macd > total_tech*0.5:
        h.append(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:0 6px 6px 0;padding:14px 20px;margin-bottom:20px;"><div style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#d97706;">MARKET-WIDE CONTEXT</div><div style="font-size:13px;color:#334155;margin-top:6px;">{bearish_macd}/{total_tech} stocks BEARISH MACD. {below50} below SMA50. Avg RSI: {avg_rsi:.0f}.</div></div>')
    h.append('<table style="width:100%;border-collapse:collapse;font-size:11px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px 6px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">RSI</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">MACD</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">BB%</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Trend</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Mom</th><th style="padding:8px 6px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Signal</th></tr>')
    for i,(tkr,d) in enumerate(sorted(tech_stocks.items(),key=lambda x:x[1].get("momentum_score",0),reverse=True)[:15]):
        bg="#f8fafc" if i%2==1 else "#fff"; rsi=d.get("rsi",50); macd=d.get("macd_signal","?"); bb=d.get("bb_position",0.5); trend=d.get("trend","?"); mom=d.get("momentum_score",0); sig=d.get("timing_signal","?")
        rc = "#dc2626" if rsi>70 else "#059669" if rsi<30 else "#334155"
        h.append(f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;"><td style="padding:7px 8px;font-family:monospace;font-weight:700;">{e(tkr)}</td><td style="padding:7px 6px;text-align:center;font-weight:700;color:{rc};">{rsi:.0f}</td><td style="padding:7px 6px;text-align:center;color:{sentiment_color(macd)};font-weight:600;">{macd[:4]}</td><td style="padding:7px 6px;text-align:center;">{bb:.2f}</td><td style="padding:7px 6px;text-align:center;font-size:10px;">{e(trend[:12])}</td><td style="padding:7px 6px;text-align:center;font-weight:700;color:{"#059669" if mom>0 else "#dc2626" if mom<0 else "#64748b"};">{mom:+d}</td><td style="padding:7px 6px;text-align:center;"><span style="padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{sentiment_color(sig)};">{e(sig[:8])}</span></td></tr>')
    h.append('</table>')
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
            tkr = m if isinstance(m,str) else m.get("ticker","?")
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
    pr = risk.get("portfolio_risk",{})
    h.append(f'<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Risk Dashboard</h2><table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;"><tr><td style="width:50%;vertical-align:top;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Portfolio Metrics</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">VaR (95%)</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{var_95:.2f}%</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Max Drawdown</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;color:#dc2626;">{max_dd:.1f}%</td></tr><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Beta</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{p_beta:.2f}</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Sortino</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;">{pr.get("sortino_ratio",0):.2f}</td></tr><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Risk Score</td><td style="padding:8px 12px;text-align:right;font-weight:700;font-family:monospace;color:{risk_color};">{risk_score}/100</td></tr></table></td>')
    geo = concentration.get("geography",{}).get("concentration",{})
    h.append(f'<td style="width:50%;vertical-align:top;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Concentration</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f8fafc;"><td style="padding:8px 12px;font-weight:600;">Top Region</td><td style="padding:8px 12px;text-align:right;font-weight:700;">{e(geo.get("top_region","US"))} ({geo.get("top_region_pct",0):.0f}%)</td></tr><tr><td style="padding:8px 12px;font-weight:600;">Stocks Analyzed</td><td style="padding:8px 12px;text-align:right;font-weight:700;">{len(concordance)}</td></tr></table></td></tr></table>')
    # Clusters
    if clusters:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin-bottom:10px;">Correlation Clusters</div>')
        for cl in clusters[:4]:
            stks = cl.get("stocks",cl.get("tickers",[])); ac = cl.get("avg_correlation",cl.get("average_correlation",0)); rn = cl.get("risk",cl.get("note",""))
            h.append(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:12px;"><span style="font-weight:700;color:#991b1b;">{", ".join(stks[:6])}</span> (corr: {ac:.2f})<br/><span style="color:#64748b;font-size:11px;">{e(str(rn)[:100])}</span></div>')
    # Stress
    if stress:
        h.append('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:20px 0 10px 0;">Stress Scenarios</div><table style="width:100%;border-collapse:separate;border-spacing:8px 0;"><tr>')
        for title,key,bg,brd,lc,vc in [("CRASH -10%","market_crash_10pct","#fef2f2","#fecaca","#991b1b","#dc2626"),("RATE +100bps","rate_shock_100bps","#fffbeb","#fde68a","#92400e","#d97706"),("VIX TO 40","vix_spike_40","#fef2f2","#fecaca","#991b1b","#dc2626")]:
            sd = stress.get(key,{}); imp = sd.get("portfolio_impact_pct",sd.get("estimated_portfolio_impact_pct","?"))
            h.append(f'<td style="width:33%;padding:12px;background:{bg};border:1px solid {brd};border-radius:6px;"><div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{lc};">{title}</div><div style="font-size:20px;font-weight:800;color:{vc};">{imp}%</div></td>')
        h.append('</tr></table>')
    h.append('</div>')

    # S10: OPPORTUNITIES
    opp_tops = opps.get("top_opportunities",[])
    h.append(f'<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">New Opportunities</h2><p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">{len(opp_tops)} candidates ranked.</p><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #e2e8f0;"><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">#</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Stock</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Sector</th><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">Score</th><th style="padding:8px;text-align:center;font-size:10px;font-weight:700;color:#64748b;">EXRET</th><th style="padding:8px;text-align:left;font-size:10px;font-weight:700;color:#64748b;">Thesis</th></tr>')
    for i,o in enumerate(opp_tops[:10]):
        bg = "#ecfdf5" if i<3 else "#f8fafc" if i%2==0 else "#fff"
        sc = o.get("scores",{}).get("final_score",o.get("opportunity_score",0)); ex = o.get("fundamentals",{}).get("exret",o.get("exret",""))
        h.append(f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;"><td style="padding:8px;text-align:center;font-weight:700;color:#059669;">{i+1}</td><td style="padding:8px;font-family:monospace;font-weight:700;">{e(o.get("ticker",""))}</td><td style="padding:8px;">{e(o.get("sector",""))}</td><td style="padding:8px;text-align:center;font-weight:700;color:{conv_color(int(sf(sc)))};">{sf(sc):.0f}</td><td style="padding:8px;text-align:center;">{e(str(ex))}</td><td style="padding:8px;font-size:11px;color:#334155;">{e(str(o.get("why_compelling",o.get("thesis","")))[:80])}</td></tr>')
    h.append('</table></div>')

    # S11: ACTION ITEMS
    h.append('<div style="padding:32px 40px;border-bottom:2px solid #0f172a;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Action Items</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Priority-ranked by committee consensus.</p><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #1e293b;"><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:25px;">#</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:70px;">ACTION</th><th style="padding:10px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:65px;">STOCK</th><th style="padding:10px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;">DETAIL</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:35px;">CONV</th><th style="padding:10px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;width:50px;">SIZE</th></tr>')
    for i,entry in enumerate(concordance):
        act=entry.get("action","HOLD"); tkr=entry.get("ticker",""); conv=entry.get("conviction",0); sig=entry.get("signal","?"); ex=entry.get("exret",0); sec=entry.get("sector",""); mp=entry.get("max_pct",5.0); rsi=entry.get("rsi",0)
        if act in ("SELL","REDUCE","IMMEDIATE SELL"): sz="Exit"; dt=f"Sig:{sig} EXRET:{ex:.1f}% RSI:{rsi:.0f}"
        elif act=="TRIM": sz="Reduce"; dt=f"Sig:{sig} RSI:{rsi:.0f} {sec}"
        elif act in ("BUY","ADD","BUY NEW"): sz=f"Max {mp:.1f}%"; dt=f"Sig:{sig} EXRET:{ex:.1f}% {sec}"
        else: sz="-"; dt=f"Sig:{sig} EXRET:{ex:.1f}%"
        ac,ab,abr = action_color(act),action_bg(act),action_border(act)
        h.append(f'<tr style="background:{ab};border-bottom:1px solid {abr};"><td style="padding:10px 8px;text-align:center;font-weight:800;">{i+1}</td><td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:{ac};">{act}</span></td><td style="padding:10px 8px;font-family:monospace;font-weight:700;font-size:13px;">{e(tkr)}</td><td style="padding:10px 8px;color:#334155;font-size:11px;">{e(dt)}</td><td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:{conv_color(conv)};">{conv}</td><td style="padding:10px 8px;text-align:center;font-family:monospace;font-size:11px;font-weight:600;">{sz}</td></tr>')
    h.append('</table></div>')

    # S12: CHANGES
    sig_changes = [c for c in changes if abs(c.get("delta",0))>5 or c.get("prev_action")!=c.get("curr_action")]
    if sig_changes:
        h.append(f'<div style="padding:32px 40px;border-bottom:2px solid #cbd5e1;"><h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Changes Since Last Committee</h2><p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">{len(sig_changes)} significant changes.</p><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="border-bottom:2px solid #1e293b;"><th style="padding:8px;text-align:left;background:#f1f5f9;font-size:10px;font-weight:700;">STOCK</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">PREVIOUS</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">CURRENT</th><th style="padding:8px;text-align:center;background:#f1f5f9;font-size:10px;font-weight:700;">DELTA</th><th style="padding:8px;text-align:left;background:#f1f5f9;font-size:10px;font-weight:700;">TYPE</th></tr>')
        for c in sig_changes[:20]:
            ct=c.get("type","")
            if ct=="DOWNGRADE": rbg,rbrd,ar,tc = "#fef2f2","#fecaca","&#9660;","#991b1b"
            elif ct=="UPGRADE": rbg,rbrd,ar,tc = "#ecfdf5","#a7f3d0","&#9650;","#065f46"
            elif ct=="NEW": rbg,rbrd,ar,tc = "#eff6ff","#bfdbfe","&#9733;","#2563eb"
            else: rbg,rbrd,ar,tc = "#fff","#e2e8f0","&middot;","#64748b"
            d=c.get("delta",0)
            h.append(f'<tr style="background:{rbg};border-bottom:1px solid {rbrd};"><td style="padding:8px;font-family:monospace;font-weight:700;">{e(c.get("ticker",""))}</td><td style="padding:8px;text-align:center;">{e(c.get("prev_action","?"))} ({c.get("prev_conviction",0)})</td><td style="padding:8px;text-align:center;">{e(c.get("curr_action","?"))} ({c.get("curr_conviction",0)})</td><td style="padding:8px;text-align:center;font-weight:700;color:{tc};">{d:+d}</td><td style="padding:8px;color:{tc};font-weight:600;">{ar} {e(ct)}</td></tr>')
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

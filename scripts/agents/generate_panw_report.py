#!/usr/bin/env python3
"""Generate PANW Investment Committee HTML report."""

import json
import os

OUTPUT = os.path.expanduser("~/.weirdapps-trading/committee/2026-03-06.html")
REPORTS = os.path.expanduser("~/.weirdapps-trading/committee/reports")


def load_report(name):
    with open(os.path.join(REPORTS, f"{name}.json")) as f:
        return json.load(f)


def generate():
    load_report("fundamental")
    load_report("technical")
    load_report("macro")
    load_report("census")
    load_report("news")
    load_report("opportunities")
    load_report("risk")

    html = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Investment Committee - PANW Deep Dive</title></head>
<body style="margin:0;padding:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;color:#334155;line-height:1.6;-webkit-font-smoothing:antialiased;">
<div style="max-width:960px;margin:0 auto;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.1);">""")

    # HEADER
    html.append("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#334155 100%);padding:36px 40px 32px 40px;">
  <div style="display:flex;align-items:center;margin-bottom:4px;">
    <div style="width:36px;height:36px;border-radius:8px;background:rgba(255,255,255,0.15);display:inline-flex;align-items:center;justify-content:center;margin-right:14px;font-size:18px;color:#fff;font-weight:800;letter-spacing:-1px;">IC</div>
    <div>
      <h1 style="margin:0;font-size:26px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Investment Committee: PANW Deep Dive</h1>
      <p style="margin:2px 0 0 0;font-size:13px;color:#94a3b8;">March 6, 2026 &middot; 7 Specialist Agents + CIO Synthesis</p>
    </div>
  </div>
  <div style="margin-top:16px;display:flex;gap:12px;">
    <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Signals: Mar 5, 2026</span>
    <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Census: Mar 5, 2026</span>
    <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:0.3px;background:rgba(255,255,255,0.12);color:#94a3b8;">Focus: Cybersecurity Sector</span>
  </div>
</div>""")

    # SECTION 1: EXECUTIVE SUMMARY
    html.append("""
<div style="padding:32px 40px;border-bottom:2px solid #cbd5e1;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;letter-spacing:-0.3px;">Executive Summary</h2>
  <table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin-bottom:24px;" cellpadding="0" cellspacing="0">
  <tr>
    <td style="width:33%;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:20px 24px;text-align:center;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#92400e;margin-bottom:6px;">CIO Verdict</div>
      <div style="font-size:28px;font-weight:800;color:#d97706;letter-spacing:-0.5px;">CAUTIOUS BUY</div>
      <div style="font-size:11px;color:#92400e;margin-top:4px;">3 bullish vs 4 cautious agents</div>
    </td>
    <td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Macro Regime</div>
      <div style="font-size:28px;font-weight:800;color:#d97706;letter-spacing:-0.5px;">LATE CYCLE</div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;">Score: +5 &middot; Defensive rotation</div>
    </td>
    <td style="width:33%;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:20px 24px;text-align:center;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Portfolio Risk</div>
      <div style="font-size:28px;font-weight:800;color:#d97706;letter-spacing:-0.5px;">MOD-HIGH</div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;">Max position: 3.5% &middot; Tech 60%+</div>
    </td>
  </tr>
  </table>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:20px;">
    <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">PANW down 27% on acquisition costs, not operational weakness</div>
    <div style="font-size:13px;color:#334155;line-height:1.6;">Palo Alto Networks is a "fallen angel" &mdash; stock crashed on Q3 profit guidance miss from $28B integration costs (CyberArk $25B + Chronosphere $3.35B), while business metrics remain strong: Revenue +15%, NGS ARR +33%, 110 net new platformizations. Analysts 95% bullish with $216 avg target (+32%). Committee split 3-4: Fundamental, News, and Macro bullish; Technical, Census, Opportunity Scanner, and Risk Manager cautious. Start 2% position, scale to 3.5% on confirmation.</div>
  </div>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Priority Actions</div>
  <table style="width:100%;border-collapse:separate;border-spacing:0 6px;font-size:13px;">
    <tr>
      <td style="padding:10px 14px;border:1px solid #a7f3d0;background:#ecfdf5;border-radius:6px 0 0 6px;width:70px;">
        <span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.3px;color:#fff;background:#059669;">BUY</span>
      </td>
      <td style="padding:10px 14px;border:1px solid #a7f3d0;background:#ecfdf5;font-weight:700;font-family:'SF Mono',Consolas,monospace;letter-spacing:0.3px;width:60px;">PANW</td>
      <td style="padding:10px 14px;border:1px solid #a7f3d0;background:#ecfdf5;color:#334155;">Initiate 2% position. Fallen angel: PE 83&rarr;37, institutions accumulating, 95% analyst BUY. Scale to 3.5% if $155 holds.</td>
      <td style="padding:10px 14px;border:1px solid #a7f3d0;background:#ecfdf5;text-align:center;font-weight:700;border-radius:0 6px 6px 0;width:50px;">65</td>
    </tr>
    <tr>
      <td style="padding:10px 14px;border:1px solid #bfdbfe;background:#eff6ff;border-radius:6px 0 0 6px;width:70px;">
        <span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.3px;color:#fff;background:#2563eb;">WATCH</span>
      </td>
      <td style="padding:10px 14px;border:1px solid #bfdbfe;background:#eff6ff;font-weight:700;font-family:'SF Mono',Consolas,monospace;letter-spacing:0.3px;width:60px;">OKTA</td>
      <td style="padding:10px 14px;border:1px solid #bfdbfe;background:#eff6ff;color:#334155;">Highest opportunity score (73.8). PE compression 66&rarr;20, new CEO with equity grant. Wait for next earnings confirmation.</td>
      <td style="padding:10px 14px;border:1px solid #bfdbfe;background:#eff6ff;text-align:center;font-weight:700;border-radius:0 6px 6px 0;width:50px;">50</td>
    </tr>
    <tr>
      <td style="padding:10px 14px;border:1px solid #fecaca;background:#fef2f2;border-radius:6px 0 0 6px;width:70px;">
        <span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.3px;color:#fff;background:#dc2626;">AVOID</span>
      </td>
      <td style="padding:10px 14px;border:1px solid #fecaca;background:#fef2f2;font-weight:700;font-family:'SF Mono',Consolas,monospace;letter-spacing:0.3px;width:60px;">FTNT</td>
      <td style="padding:10px 14px;border:1px solid #fecaca;background:#fef2f2;color:#334155;">Massive insider selling at current prices ($42M CTO+CEO), institutional exodus, only 31% BUY, negative earnings growth.</td>
      <td style="padding:10px 14px;border:1px solid #fecaca;background:#fef2f2;text-align:center;font-weight:700;border-radius:0 6px 6px 0;width:50px;">70</td>
    </tr>
  </table>
</div>""")

    # SECTION 2: MACRO ENVIRONMENT
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Macro Environment</h2>
  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px;">
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:10px 12px;text-align:left;font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Indicator</th>
      <th style="padding:10px 12px;text-align:center;font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Value</th>
      <th style="padding:10px 12px;text-align:center;font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Signal</th>
      <th style="padding:10px 12px;text-align:left;font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Interpretation</th>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">10Y Treasury</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;">4.15%</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#d97706;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Stable; no multiple expansion catalyst</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Yield Curve</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;color:#059669;">+56bp</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#059669;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Normal slope, no recession signal</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">VIX</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;color:#dc2626;">25.8</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc2626;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Elevated; pressures growth stocks</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Dollar Index</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;">99.34</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#6366f1;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Neutral; minimal FX headwind</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Fed Rate</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;">3.50-3.75%</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#6366f1;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Hold expected Mar 17/18; mid-90s% odds</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">CIBR ETF</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;color:#dc2626;">-12.3% (3M)</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc2626;"></span></td>
      <td style="padding:10px 12px;color:#334155;">Sector underperforming despite strong fundamentals</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:10px 12px;font-weight:600;color:#1e293b;">Cyber Spending</td>
      <td style="padding:10px 12px;text-align:center;font-weight:700;font-family:'SF Mono',Consolas,monospace;color:#059669;">+12.5% YoY</td>
      <td style="padding:10px 12px;text-align:center;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#059669;"></span></td>
      <td style="padding:10px 12px;color:#334155;">$240B globally; up from 4% prior year</td>
    </tr>
  </table>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Sector Rotation (1M Performance)</div>
  <table style="width:100%;border-collapse:separate;border-spacing:4px 0;font-size:12px;">
    <tr>
      <td style="padding:8px 4px;text-align:center;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:4px;">
        <div style="font-weight:700;color:#065f46;">Energy</div><div style="font-size:13px;font-weight:800;color:#059669;">+9.3%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:4px;">
        <div style="font-weight:700;color:#065f46;">Utilities</div><div style="font-size:13px;font-weight:800;color:#059669;">+8.5%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:4px;">
        <div style="font-weight:700;color:#065f46;">Real Est</div><div style="font-size:13px;font-weight:800;color:#059669;">+6.2%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;">
        <div style="font-weight:700;color:#334155;">Indust</div><div style="font-size:13px;font-weight:800;color:#334155;">+1.8%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;">
        <div style="font-weight:700;color:#991b1b;">Tech</div><div style="font-size:13px;font-weight:800;color:#dc2626;">-1.3%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;">
        <div style="font-weight:700;color:#991b1b;">ConDisc</div><div style="font-size:13px;font-weight:800;color:#dc2626;">-3.7%</div>
      </td>
      <td style="padding:8px 4px;text-align:center;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;">
        <div style="font-weight:700;color:#991b1b;">Financ</div><div style="font-size:13px;font-weight:800;color:#dc2626;">-4.3%</div>
      </td>
    </tr>
  </table>
  <div style="font-size:11px;color:#94a3b8;margin-top:8px;font-style:italic;">Late cycle rotation: defensives leading while growth and Financials lag. Cybersecurity underperforming broader tech (-12.3% vs -1.3%).</div>
</div>""")

    # SECTION 3: STOCK ANALYSIS GRID
    grid_rows = [
        (
            "BUY",
            "#ecfdf5",
            "#a7f3d0",
            "#065f46",
            [
                (
                    "PANW",
                    "B",
                    "#059669",
                    "BUY (92)",
                    "#059669",
                    "WAIT (-5)",
                    "#d97706",
                    "FAVOR",
                    "#059669",
                    "DIV 4%",
                    "#dc2626",
                    "POS",
                    "#059669",
                    "#2 (65.7)",
                    "#d97706",
                    "WARN 3.5%",
                    "#d97706",
                    "BUY",
                    "#059669",
                    65,
                    "#d97706",
                ),
            ],
        ),
        (
            "WATCH",
            "#eff6ff",
            "#bfdbfe",
            "#1e40af",
            [
                (
                    "OKTA",
                    "H",
                    "#d97706",
                    "HOLD (72)",
                    "#059669",
                    "WAIT (-15)",
                    "#d97706",
                    "FAVOR",
                    "#059669",
                    "DIV 0%",
                    "#dc2626",
                    "POS +11%",
                    "#059669",
                    "#1 (73.8)",
                    "#059669",
                    "OK 2.5%",
                    "#059669",
                    "WATCH",
                    "#2563eb",
                    50,
                    "#94a3b8",
                ),
                (
                    "ZS",
                    "H",
                    "#d97706",
                    "HOLD (88)",
                    "#059669",
                    "AVOID (-25)",
                    "#dc2626",
                    "FAVOR",
                    "#059669",
                    "DIV 1%",
                    "#dc2626",
                    "POS",
                    "#059669",
                    "#3 (55.4)",
                    "#d97706",
                    "AVOID",
                    "#dc2626",
                    "WATCH",
                    "#2563eb",
                    40,
                    "#94a3b8",
                ),
            ],
        ),
        (
            "HOLD / MONITOR",
            "#f1f5f9",
            "#cbd5e1",
            "#475569",
            [
                (
                    "CRWD",
                    "H",
                    "#d97706",
                    "TRAP (68)",
                    "#d97706",
                    "WAIT (-20)",
                    "#d97706",
                    "FAVOR",
                    "#059669",
                    "ALIGN 10%",
                    "#059669",
                    "POS",
                    "#059669",
                    "#4 (39.2)",
                    "#d97706",
                    "BEST R/R",
                    "#059669",
                    "HOLD",
                    "#6366f1",
                    45,
                    "#94a3b8",
                ),
                (
                    "NET",
                    "H",
                    "#d97706",
                    "TRAP (58)",
                    "#d97706",
                    "WAIT (15)",
                    "#059669",
                    "FAVOR",
                    "#059669",
                    "NEUT 4%",
                    "#64748b",
                    "POS",
                    "#059669",
                    "#5 (29.1)",
                    "#64748b",
                    "WARN",
                    "#d97706",
                    "HOLD",
                    "#6366f1",
                    35,
                    "#94a3b8",
                ),
            ],
        ),
        (
            "AVOID",
            "#fef2f2",
            "#fecaca",
            "#991b1b",
            [
                (
                    "FTNT",
                    "S",
                    "#dc2626",
                    "SELL (35)",
                    "#dc2626",
                    "WAIT (15)",
                    "#059669",
                    "FAVOR",
                    "#059669",
                    "DIV 3%",
                    "#dc2626",
                    "POS",
                    "#059669",
                    "#6 (16.9)",
                    "#dc2626",
                    "AVOID",
                    "#dc2626",
                    "AVOID",
                    "#dc2626",
                    70,
                    "#d97706",
                ),
                (
                    "CYBR",
                    "S",
                    "#dc2626",
                    "SELL (42)",
                    "#dc2626",
                    "AVOID (-35)",
                    "#dc2626",
                    "FAVOR",
                    "#059669",
                    "DIV 0%",
                    "#dc2626",
                    "NEUT",
                    "#64748b",
                    "#7 (13.6)",
                    "#dc2626",
                    "AVOID",
                    "#dc2626",
                    "AVOID",
                    "#dc2626",
                    75,
                    "#d97706",
                ),
            ],
        ),
    ]

    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Stock Analysis Grid</h2>
  <p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Concordance matrix: all 7 specialist views per stock. Sorted by CIO action priority.</p>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap;">
    <tr>
      <th style="padding:8px 6px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;border:1px solid #1e293b;">STOCK</th>
      <th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;border:1px solid #1e293b;">SIG</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#93c5fd;font-size:10px;font-weight:600;border:1px solid #334155;">FUND</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#c4b5fd;font-size:10px;font-weight:600;border:1px solid #334155;">TECH</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#fcd34d;font-size:10px;font-weight:600;border:1px solid #334155;">MACRO</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#6ee7b7;font-size:10px;font-weight:600;border:1px solid #334155;">CENSUS</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#fca5a5;font-size:10px;font-weight:600;border:1px solid #334155;">NEWS</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#7dd3fc;font-size:10px;font-weight:600;border:1px solid #334155;">OPPTY</th>
      <th style="padding:8px 6px;text-align:center;background:#1e293b;color:#d6d3d1;font-size:10px;font-weight:600;border:1px solid #334155;">RISK</th>
      <th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;border:1px solid #1e293b;">VERDICT</th>
      <th style="padding:8px 6px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;border:1px solid #1e293b;">CONV</th>
    </tr>""")

    alt = False
    for group_name, group_bg, group_border, group_color, stocks in grid_rows:
        html.append(
            f'    <tr><td colspan="11" style="padding:4px 8px;background:{group_bg};font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{group_color};border:1px solid {group_border};">{group_name}</td></tr>'
        )
        for s in stocks:
            (
                ticker,
                sig,
                sig_color,
                fund_v,
                fund_c,
                tech_v,
                tech_c,
                macro_v,
                macro_c,
                cens_v,
                cens_c,
                news_v,
                news_c,
                oppty_v,
                oppty_c,
                risk_v,
                risk_c,
                verdict,
                verdict_c,
                conv,
                bar_c,
            ) = s
            row_bg = (
                group_bg if group_name in ("BUY", "AVOID") else ("#f8fafc" if alt else "#ffffff")
            )
            alt = not alt
            html.append(f"""    <tr style="background:{row_bg};">
      <td style="padding:7px 8px;border:1px solid #e2e8f0;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:12px;letter-spacing:0.3px;color:#0f172a;">{ticker}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;"><span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{sig_color};">{sig}</span></td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{fund_c};font-weight:600;">{fund_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{tech_c};font-weight:600;">{tech_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{macro_c};font-weight:600;">{macro_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{cens_c};font-weight:600;">{cens_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{news_c};font-weight:600;">{news_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{oppty_c};font-weight:600;">{oppty_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;color:{risk_c};font-weight:600;">{risk_v}</td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;"><span style="display:inline-block;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:{verdict_c};">{verdict}</span></td>
      <td style="padding:7px 6px;text-align:center;border:1px solid #e2e8f0;"><div style="font-weight:800;font-size:13px;color:#0f172a;">{conv}</div><div style="height:3px;background:#e2e8f0;border-radius:2px;margin-top:3px;width:40px;display:inline-block;"><div style="height:100%;border-radius:2px;background:{bar_c};width:{conv}%;"></div></div></td>
    </tr>""")

    html.append("""  </table>
  </div>
  <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;font-size:10px;color:#64748b;line-height:1.8;">
    <span style="font-weight:700;">Legend:</span> DC = Death Cross &middot; TRAP = Quality Trap Warning &middot; DIV = Census Divergent &middot; ALIGN = Census Aligned &middot; FAVOR = Macro Favorable &middot; R/R = Risk/Reward
  </div>
</div>""")

    # SECTION 4: WHERE WE DISAGREED
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Where We Disagreed</h2>
  <p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">Structured disagreement is the committee's edge.</p>

  <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:20px 24px;margin-bottom:16px;">
    <div style="margin-bottom:10px;">
      <span style="font-family:'SF Mono',Consolas,monospace;font-weight:800;font-size:14px;color:#0f172a;">PANW</span>
      <span style="font-size:13px;font-weight:600;color:#334155;margin-left:8px;">Fundamental BUY (92) vs Technical WAIT (-5) vs Risk WARN</span>
    </div>
    <div style="margin-bottom:12px;">
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;">Fund: BUY 92</span>
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;">News: STRONG BUY</span>
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;">Tech: WAIT -5</span>
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#f5f5f4;color:#44403c;border:1px solid #d6d3d1;">Risk: WARN 3.5%</span>
    </div>
    <div style="font-size:13px;color:#334155;line-height:1.6;margin-bottom:12px;">
      <strong>Bulls (Fund+News+Macro):</strong> Fallen angel &mdash; 27% crash on $28B acquisition costs, not weakness. Rev +15%, NGS ARR +33%, PE 83&rarr;37, institutions accumulating (Norges +133%).
      <br/><strong>Bears (Tech+Census+Risk):</strong> Death cross, below both SMAs. Only 4% PI conviction. 95% consensus = crowded. VaR -4.35%. Portfolio already 60%+ tech.
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;">
      <span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Resolution:</span>
      <span style="font-size:13px;font-weight:700;color:#d97706;margin-left:6px;">CAUTIOUS BUY (conv. 65)</span>
      <div style="font-size:12px;color:#475569;margin-top:4px;">Fundamentals win but Risk Manager's sizing discipline applies. Start 2%, scale to 3.5% max. Selloff is acquisition-driven (temporary) not operational (permanent).</div>
    </div>
  </div>

  <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;border-radius:0 8px 8px 0;padding:20px 24px;margin-bottom:16px;">
    <div style="margin-bottom:10px;">
      <span style="font-family:'SF Mono',Consolas,monospace;font-weight:800;font-size:14px;color:#0f172a;">OKTA vs PANW</span>
      <span style="font-size:13px;font-weight:600;color:#334155;margin-left:8px;">Opportunity Scanner ranks OKTA #1 over PANW #2</span>
    </div>
    <div style="margin-bottom:12px;">
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#e0f2fe;color:#075985;border:1px solid #7dd3fc;">Oppty: OKTA 73.8</span>
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;">Fund: PANW 92 vs OKTA 72</span>
    </div>
    <div style="font-size:13px;color:#334155;line-height:1.6;margin-bottom:12px;">
      OKTA scores higher on PE compression (66&rarr;20), FCF yield (7.5%), and lower risk. But PANW has 95% analyst BUY (vs 67%), better margins, and is the sector leader. OKTA is a turnaround story with higher execution risk.
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;">
      <span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Resolution:</span>
      <span style="font-size:13px;font-weight:700;color:#2563eb;margin-left:6px;">BUY PANW now, WATCH OKTA for confirmation</span>
      <div style="font-size:12px;color:#475569;margin-top:4px;">Prefer proven execution over turnaround potential. Add OKTA if next earnings confirm trajectory.</div>
    </div>
  </div>

  <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #059669;border-radius:0 8px 8px 0;padding:20px 24px;margin-bottom:16px;">
    <div style="margin-bottom:10px;">
      <span style="font-family:'SF Mono',Consolas,monospace;font-weight:800;font-size:14px;color:#0f172a;">PANW</span>
      <span style="font-size:13px;font-weight:600;color:#334155;margin-left:8px;">Census (4% retail) vs Institutions (Norges +133%)</span>
    </div>
    <div style="margin-bottom:12px;">
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;">Census: LOW 4%</span>
      <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin-right:4px;background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;">Institutions: BUYING</span>
    </div>
    <div style="font-size:13px;color:#334155;line-height:1.6;margin-bottom:12px;">
      Only 4% of eToro Top 100 PIs hold PANW while institutions aggressively accumulate (Norges +133%, BofA +11%, Blackrock +7.4%). Retail selling what institutions buy &mdash; historically bullish.
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;">
      <span style="font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Resolution:</span>
      <span style="font-size:13px;font-weight:700;color:#059669;margin-left:6px;">Follow institutions over retail (bullish)</span>
      <div style="font-size:12px;color:#475569;margin-top:4px;">Norges Bank (sovereign wealth) +133% is a strong conviction signal. Census divergence is contrarian indicator, not warning.</div>
    </div>
  </div>
</div>""")

    # SECTION 5: FUNDAMENTAL DEEP DIVE
    fund_rows = [
        (
            "PANW",
            92,
            "#059669",
            "83&rarr;37",
            "38.4%",
            "95%",
            "Sold higher",
            "#d97706",
            "Fallen angel; institutions accumulating",
            "#ecfdf5",
        ),
        (
            "ZS",
            88,
            "#059669",
            "--&rarr;32",
            "76.5%",
            "89%",
            "Neutral",
            "#059669",
            "Hidden gem; T.Rowe +43%, insiders holding",
            "#ffffff",
        ),
        (
            "OKTA",
            72,
            "#d97706",
            "66&rarr;20",
            "35.1%",
            "67%",
            "Neutral",
            "#059669",
            "Turnaround play; new CEO, best D/E",
            "#f8fafc",
        ),
        (
            "CRWD",
            68,
            "#d97706",
            "--&rarr;77",
            "31.2%",
            "76%",
            "Heavy sell",
            "#dc2626",
            "Quality trap: high val, neg ROE, insiders selling",
            "#ffffff",
        ),
        (
            "NET",
            58,
            "#d97706",
            "--&rarr;119",
            "19.3%",
            "56%",
            "Heavy sell",
            "#dc2626",
            "AI play but extreme val, growth funds exiting",
            "#f8fafc",
        ),
        (
            "CYBR",
            42,
            "#dc2626",
            "--&rarr;--",
            "4.5%",
            "32%",
            "Unknown",
            "#64748b",
            "Activist/M&amp;A speculation, not fundamentals",
            "#fef2f2",
        ),
        (
            "FTNT",
            35,
            "#dc2626",
            "33&rarr;24",
            "4.2%",
            "31%",
            "Massive sell",
            "#dc2626",
            "$42M insider selling at current prices",
            "#fef2f2",
        ),
    ]

    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Fundamental Deep Dive</h2>
  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Peer Ranking by Fundamental Score</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:24px;">
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Stock</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Score</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">PE T&rarr;F</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">EXRET</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">%BUY</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Insider</th>
      <th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Key Insight</th>
    </tr>""")

    for ticker, score, score_c, pe, exret, buy_pct, insider, insider_c, insight, bg in fund_rows:
        html.append(f"""    <tr style="border-bottom:1px solid #f1f5f9;background:{bg};">
      <td style="padding:8px 10px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#0f172a;">{ticker}</td>
      <td style="padding:8px 10px;text-align:center;font-weight:800;color:{score_c};">{score}</td>
      <td style="padding:8px 10px;text-align:center;font-family:monospace;">{pe}</td>
      <td style="padding:8px 10px;text-align:center;">{exret}</td>
      <td style="padding:8px 10px;text-align:center;">{buy_pct}</td>
      <td style="padding:8px 10px;text-align:center;color:{insider_c};font-weight:600;">{insider}</td>
      <td style="padding:8px 10px;color:#334155;">{insight}</td>
    </tr>""")

    # Quality traps
    traps = [
        ("CRWD", "MOD", "#d97706", "High valuation (77 FPE), neg ROE, heavy CEO selling ($20M)"),
        (
            "FTNT",
            "SEVERE",
            "#dc2626",
            "$42M insider selling at current prices, institutional exodus, -7 analyst momentum",
        ),
        ("NET", "MOD-HI", "#d97706", "119 FPE, neg ROE, D/E 241x, CEO/President selling $40M+"),
        (
            "CYBR",
            "HIGH",
            "#dc2626",
            "Zero insider ownership (0.03%), activist pattern, only 32% BUY",
        ),
    ]

    html.append("""  </table>
  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin-bottom:10px;">Quality Traps Identified</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <tr style="border-bottom:2px solid #fecaca;">
      <th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#991b1b;">Stock</th>
      <th style="padding:8px 10px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#991b1b;">Severity</th>
      <th style="padding:8px 10px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#991b1b;">Trap Characteristics</th>
    </tr>""")
    for ticker, sev, sev_c, desc in traps:
        html.append(f"""    <tr style="background:#fef2f2;border-bottom:1px solid #fecaca;">
      <td style="padding:8px 10px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#991b1b;">{ticker}</td>
      <td style="padding:8px 10px;text-align:center;"><span style="display:inline-block;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{sev_c};">{sev}</span></td>
      <td style="padding:8px 10px;color:#334155;">{desc}</td>
    </tr>""")
    html.append("  </table>\n</div>")

    # SECTION 6: TECHNICAL ANALYSIS
    tech_rows = [
        (
            "PANW",
            "$163",
            "50.3",
            "BULL",
            "#059669",
            "&#10007; 173",
            "#dc2626",
            "&#10007; 191",
            "#dc2626",
            "-26%",
            -5,
            "#d97706",
            "CONSOL",
            "WAIT",
            "#d97706",
            "#fffbeb",
        ),
        (
            "CRWD",
            "$426",
            "53.5",
            "BULL",
            "#059669",
            "&#10007; 434",
            "#dc2626",
            "&#10007; 469",
            "#dc2626",
            "-24%",
            -20,
            "#dc2626",
            "WK DN",
            "WAIT",
            "#d97706",
            "#ffffff",
        ),
        (
            "FTNT",
            "$84",
            "50.3",
            "BULL",
            "#059669",
            "&#10003; 80",
            "#059669",
            "&#10007; 87",
            "#dc2626",
            "-22%",
            15,
            "#059669",
            "WK UP",
            "WAIT",
            "#d97706",
            "#f8fafc",
        ),
        (
            "NET",
            "$192",
            "54.0",
            "BULL",
            "#059669",
            "&#10003; 185",
            "#059669",
            "&#10007; 198",
            "#dc2626",
            "-24%",
            15,
            "#059669",
            "WK UP",
            "WAIT",
            "#d97706",
            "#ffffff",
        ),
        (
            "ZS",
            "$162",
            "45.6",
            "BULL",
            "#059669",
            "&#10007; 194",
            "#dc2626",
            "&#10007; 264",
            "#dc2626",
            "-52%",
            -25,
            "#dc2626",
            "WK DN",
            "AVOID",
            "#dc2626",
            "#f8fafc",
        ),
        (
            "OKTA",
            "$80",
            "43.6",
            "BULL",
            "#059669",
            "&#10007; 85",
            "#dc2626",
            "&#10007; 91",
            "#dc2626",
            "-37%",
            -15,
            "#dc2626",
            "WK DN",
            "WAIT",
            "#d97706",
            "#ffffff",
        ),
        (
            "CYBR",
            "$409",
            "30.6",
            "BEAR",
            "#dc2626",
            "&#10007; 444",
            "#dc2626",
            "&#10007; 436",
            "#dc2626",
            "-22%",
            -35,
            "#dc2626",
            "WK DN",
            "AVOID",
            "#dc2626",
            "#fef2f2",
        ),
    ]

    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Technical Analysis</h2>
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px;">
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Stock</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Price</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">RSI</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">MACD</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">SMA50</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">SMA200</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">52W%</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Mom</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Trend</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Signal</th>
    </tr>""")

    for (
        ticker,
        price,
        rsi,
        macd,
        macd_c,
        sma50,
        sma50_c,
        sma200,
        sma200_c,
        w52,
        mom,
        mom_c,
        trend,
        signal,
        sig_c,
        bg,
    ) in tech_rows:
        rsi_style = "color:#dc2626;font-weight:600;" if float(rsi) < 35 else ""
        html.append(f"""    <tr style="border-bottom:1px solid #f1f5f9;background:{bg};">
      <td style="padding:8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#0f172a;">{ticker}</td>
      <td style="padding:8px;text-align:center;font-family:monospace;">{price}</td>
      <td style="padding:8px;text-align:center;{rsi_style}">{rsi}</td>
      <td style="padding:8px;text-align:center;color:{macd_c};font-weight:600;">{macd}</td>
      <td style="padding:8px;text-align:center;color:{sma50_c};">{sma50}</td>
      <td style="padding:8px;text-align:center;color:{sma200_c};">{sma200}</td>
      <td style="padding:8px;text-align:center;">{w52}</td>
      <td style="padding:8px;text-align:center;font-weight:700;color:{mom_c};">{mom}</td>
      <td style="padding:8px;text-align:center;">{trend}</td>
      <td style="padding:8px;text-align:center;"><span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:{sig_c};">{signal}</span></td>
    </tr>""")

    html.append("""  </table>
  <div style="font-size:11px;color:#94a3b8;font-style:italic;">All 7 stocks show death cross except CYBR. Entire sector technically weak despite emerging MACD bullish crossovers. No ENTER_NOW signals.</div>
</div>""")

    # SECTION 7: CENSUS & SENTIMENT
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Census &amp; Sentiment</h2>
  <table style="width:100%;border-collapse:separate;border-spacing:16px 0;" cellpadding="0" cellspacing="0">
  <tr>
    <td style="width:50%;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Sentiment Indicators</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">F&amp;G Top 100</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#d97706;">60 (GREED)</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">F&amp;G Broad</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#64748b;">54 (NEUTRAL)</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Cash Top 100</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;">10.4%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Cash Broad</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;">12.9%</td>
        </tr>
      </table>
    </td>
    <td style="width:50%;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Cybersecurity PI Holdings</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-family:monospace;font-weight:700;">CRWD</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#059669;">10% Top100</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-family:monospace;font-weight:700;">PANW</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">4% Top100</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-family:monospace;font-weight:700;">NET</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">4% Top100</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-family:monospace;font-weight:700;">FTNT</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">3% Top100</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-family:monospace;font-weight:700;">ZS/OKTA/CYBR</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">0-1% Top100</td>
        </tr>
      </table>
    </td>
  </tr>
  </table>
  <div style="margin-top:16px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:14px 18px;font-size:12px;color:#92400e;line-height:1.6;">
    <strong>Key Insight:</strong> Cybersecurity is unpopular with retail PIs &mdash; CRWD at 10% is highest, PANW at only 4%. This is contrarian: institutions (Norges, Blackrock, T.Rowe) are accumulating while retail ignores the sector. Low retail sentiment + institutional buying = potential smart-money divergence play.
  </div>
</div>""")

    # SECTION 8: NEWS & EVENTS
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">News &amp; Event Risk</h2>

  <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;">
    <div style="font-size:12px;font-weight:700;color:#991b1b;">Feb 17-25: PANW crashes 27% on Q3 profit guidance shock</div>
    <div style="font-size:12px;color:#334155;margin-top:4px;">EPS guidance cut to $3.65-3.70 from $3.80-3.90 due to $28B acquisition integration costs. Business strong: Rev +15%, NGS ARR +33%, 110 platformizations (record). Analyst avg target $216 (+32%).</div>
  </div>

  <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;">
    <div style="font-size:12px;font-weight:700;color:#991b1b;">Feb 20: Anthropic Claude Code Security triggers 5-9% sector selloff</div>
    <div style="font-size:12px;color:#334155;margin-top:4px;">AI vulnerability scanning tool feared as disruption. Analysts: "Workflow tool, not platform replacement." CRWD CEO: complementary, not competitive. OVERREACTION.</div>
  </div>

  <div style="background:#ecfdf5;border-left:4px solid #059669;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;">
    <div style="font-size:12px;font-weight:700;color:#065f46;">Feb 2026: Record cyberattacks validate sector spending</div>
    <div style="font-size:12px;color:#334155;margin-top:4px;">Conduent (25M records), Odido (6M), Panera (5.1M). 680 ransomware victims across 72 countries. New ransomware families. Validates cybersecurity investment.</div>
  </div>

  <div style="background:#ecfdf5;border-left:4px solid #059669;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;">
    <div style="font-size:12px;font-weight:700;color:#065f46;">Mar 3-4: CRWD blowout + OKTA beat confirm sector strength</div>
    <div style="font-size:12px;color:#334155;margin-top:4px;">CRWD: First to $5B ARR, EPS $1.12 vs $0.74 est. OKTA: EPS $0.90 vs $0.85, stock +11%. Both beat despite sector selloff. Validates fundamentals over sentiment.</div>
  </div>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:20px 0 10px 0;">Earnings Calendar</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Stock</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Date</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Days</th>
      <th style="padding:8px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Status</th>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:8px;font-family:monospace;font-weight:700;">FTNT</td>
      <td style="padding:8px;text-align:center;">Apr 30</td>
      <td style="padding:8px;text-align:center;">55</td>
      <td style="padding:8px;">Q1 2026</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
      <td style="padding:8px;font-family:monospace;font-weight:700;">CRWD</td>
      <td style="padding:8px;text-align:center;">May 3</td>
      <td style="padding:8px;text-align:center;">58</td>
      <td style="padding:8px;">Q1 FY27 (est)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:8px;font-family:monospace;font-weight:700;">PANW</td>
      <td style="padding:8px;text-align:center;">May 21</td>
      <td style="padding:8px;text-align:center;font-weight:700;color:#d97706;">76</td>
      <td style="padding:8px;">Q3 FY26 &mdash; integration progress key</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;background:#f8fafc;">
      <td style="padding:8px;font-family:monospace;font-weight:700;">OKTA</td>
      <td style="padding:8px;text-align:center;">May 27</td>
      <td style="padding:8px;text-align:center;">82</td>
      <td style="padding:8px;">Q1 FY27</td>
    </tr>
  </table>
</div>""")

    # SECTION 9: RISK DASHBOARD
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 20px 0;font-size:18px;font-weight:700;color:#0f172a;">Risk Dashboard</h2>
  <table style="width:100%;border-collapse:separate;border-spacing:16px 0;" cellpadding="0" cellspacing="0">
  <tr>
    <td style="width:50%;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">PANW Risk Metrics</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Volatility (1Y)</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">36.6%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Max Drawdown (1Y)</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">-36.0%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">VaR 95% Daily</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">-4.35%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Sharpe (1Y)</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">-0.08</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Tech Basket Corr</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#d97706;">0.53</td>
        </tr>
      </table>
    </td>
    <td style="width:50%;vertical-align:top;">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Concentration &amp; Limits</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Portfolio Tech %</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">~60%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">With PANW</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">~63-65%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Max Position</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#d97706;">3.5%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Initial Size</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;">2.0%</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:8px 0;font-weight:600;color:#1e293b;">Stop Loss</td>
          <td style="padding:8px 0;text-align:right;font-weight:700;color:#dc2626;">-15%</td>
        </tr>
      </table>
    </td>
  </tr>
  </table>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#dc2626;margin:20px 0 10px 0;">Correlation Clusters</div>
  <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px 18px;margin-bottom:12px;">
    <div style="font-size:12px;font-weight:700;color:#991b1b;margin-bottom:4px;">Cybersecurity Cluster (68-74% correlated)</div>
    <div style="font-size:12px;color:#334155;">PANW &harr; CRWD (0.74), CYBR (0.70), ZS (0.68), FTNT (0.63). Highly redundant &mdash; adding multiple cybersecurity names provides minimal diversification benefit.</div>
  </div>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin:20px 0 10px 0;">Hard Constraints</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:6px 8px;width:30px;text-align:center;color:#d97706;">&#9888;</td>
      <td style="padding:6px 8px;color:#334155;">Max single position: 5.0% (PANW limited to 3.5% due to penalties)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:6px 8px;width:30px;text-align:center;color:#d97706;">&#9888;</td>
      <td style="padding:6px 8px;color:#334155;">Max sector: 65% tech (currently ~60%, PANW pushes to ~63%)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:6px 8px;width:30px;text-align:center;color:#dc2626;">&#9888;</td>
      <td style="padding:6px 8px;color:#334155;">Cannot add PANW + CRWD without trimming existing tech (violates 65% limit)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:6px 8px;width:30px;text-align:center;color:#059669;">&#10003;</td>
      <td style="padding:6px 8px;color:#334155;">Mandatory -15% stop loss for all new tech positions given concentration</td>
    </tr>
  </table>
</div>""")

    # SECTION 10: NEW OPPORTUNITIES
    html.append("""
<div style="padding:32px 40px;border-bottom:1px solid #e2e8f0;">
  <h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">New Opportunities</h2>
  <p style="color:#64748b;font-size:12px;margin:0 0 20px 0;">Screened from 7-stock cybersecurity universe. Portfolio has ZERO cybersecurity exposure.</p>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:10px;">Top Ranked Candidates</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:24px;">
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">#</th>
      <th style="padding:8px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Stock</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Score</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">EXRET</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">PE F</th>
      <th style="padding:8px;text-align:center;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Signal</th>
      <th style="padding:8px;text-align:left;font-size:10px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:#64748b;">Why Compelling</th>
    </tr>
    <tr style="background:#ecfdf5;border-bottom:1px solid #a7f3d0;">
      <td style="padding:8px;text-align:center;font-weight:700;color:#059669;">1</td>
      <td style="padding:8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#0f172a;">OKTA</td>
      <td style="padding:8px;text-align:center;font-weight:800;color:#059669;">73.8</td>
      <td style="padding:8px;text-align:center;">35.1%</td>
      <td style="padding:8px;text-align:center;">19.7</td>
      <td style="padding:8px;text-align:center;"><span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:#d97706;">H</span></td>
      <td style="padding:8px;color:#334155;">Best PE compression (66&rarr;20), 7.5% FCF yield, new CEO turnaround</td>
    </tr>
    <tr style="background:#ecfdf5;border-bottom:1px solid #a7f3d0;">
      <td style="padding:8px;text-align:center;font-weight:700;color:#059669;">2</td>
      <td style="padding:8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#0f172a;">PANW</td>
      <td style="padding:8px;text-align:center;font-weight:800;color:#059669;">65.7</td>
      <td style="padding:8px;text-align:center;">38.4%</td>
      <td style="padding:8px;text-align:center;">37.4</td>
      <td style="padding:8px;text-align:center;"><span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:#059669;">B</span></td>
      <td style="padding:8px;color:#334155;">Only BUY signal; 95% analyst conviction; sector leader</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0;">
      <td style="padding:8px;text-align:center;font-weight:700;color:#d97706;">3</td>
      <td style="padding:8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;color:#0f172a;">ZS</td>
      <td style="padding:8px;text-align:center;font-weight:800;color:#d97706;">55.4</td>
      <td style="padding:8px;text-align:center;">76.5%</td>
      <td style="padding:8px;text-align:center;">32.0</td>
      <td style="padding:8px;text-align:center;"><span style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:#d97706;">H</span></td>
      <td style="padding:8px;color:#334155;">Highest upside (86%) but -50% EPS growth; contrarian play</td>
    </tr>
  </table>

  <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#d97706;margin-bottom:10px;">Sector Gap</div>
  <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:14px 18px;">
    <div style="font-size:12px;font-weight:700;color:#92400e;margin-bottom:4px;">&#9888; CRITICAL: Portfolio has ZERO cybersecurity exposure</div>
    <div style="font-size:12px;color:#334155;">Despite cybersecurity spending growing 12.5% YoY ($240B globally), rising threat landscape (680 ransomware victims in Feb alone), and AI security market at $35.4B, the portfolio has no exposure to this essential sector. PANW is the recommended entry point as sector leader with strongest analyst conviction.</div>
  </div>
</div>""")

    # SECTION 11: ACTION ITEMS
    html.append("""
<div style="padding:32px 40px;border-bottom:2px solid #0f172a;">
  <h2 style="margin:0 0 4px 0;font-size:18px;font-weight:700;color:#0f172a;">Action Items</h2>
  <p style="color:#64748b;font-size:12px;margin:0 0 16px 0;">Priority-ranked. Conviction reflects committee agreement strength and risk adjustment.</p>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <tr style="border-bottom:2px solid #1e293b;">
      <th style="padding:10px 8px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;width:30px;">#</th>
      <th style="padding:10px 8px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;width:70px;">ACTION</th>
      <th style="padding:10px 8px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;">STOCK</th>
      <th style="padding:10px 8px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;">DETAIL</th>
      <th style="padding:10px 8px;text-align:center;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;width:45px;">CONV</th>
      <th style="padding:10px 8px;text-align:left;background:#0f172a;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;">KEY DISSENT</th>
    </tr>
    <tr style="background:#ecfdf5;border-bottom:1px solid #a7f3d0;">
      <td style="padding:10px 8px;text-align:center;font-weight:800;color:#0f172a;">1</td>
      <td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:#059669;letter-spacing:0.3px;">BUY NEW</span></td>
      <td style="padding:10px 8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:13px;color:#0f172a;">PANW</td>
      <td style="padding:10px 8px;color:#334155;">Initiate 2% position. Fallen angel with PE 83&rarr;37, institutions accumulating. Scale to 3.5% if $155 holds. Stop at -15%.</td>
      <td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:#0f172a;">65</td>
      <td style="padding:10px 8px;color:#64748b;font-style:italic;font-size:11px;">Tech: death cross, below SMAs. Risk: 95% consensus = crowded. Census: only 4% PI holders.</td>
    </tr>
    <tr style="background:#eff6ff;border-bottom:1px solid #bfdbfe;">
      <td style="padding:10px 8px;text-align:center;font-weight:800;color:#0f172a;">2</td>
      <td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:#2563eb;letter-spacing:0.3px;">WATCH</span></td>
      <td style="padding:10px 8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:13px;color:#0f172a;">OKTA</td>
      <td style="padding:10px 8px;color:#334155;">Highest opportunity score (73.8). Wait for next earnings to confirm turnaround. PE 66&rarr;20, FCF 7.5%.</td>
      <td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:#0f172a;">50</td>
      <td style="padding:10px 8px;color:#64748b;font-style:italic;font-size:11px;">Fund: only 72 score. Tech: WAIT signal. Turnaround risk with new CEO.</td>
    </tr>
    <tr style="background:#eff6ff;border-bottom:1px solid #bfdbfe;">
      <td style="padding:10px 8px;text-align:center;font-weight:800;color:#0f172a;">3</td>
      <td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:#2563eb;letter-spacing:0.3px;">WATCH</span></td>
      <td style="padding:10px 8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:13px;color:#0f172a;">ZS</td>
      <td style="padding:10px 8px;color:#334155;">Hidden gem: 86% upside, T.Rowe +43%, insiders not selling. But -50% EPS growth and -52% from high.</td>
      <td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:#0f172a;">40</td>
      <td style="padding:10px 8px;color:#64748b;font-style:italic;font-size:11px;">Tech: AVOID signal. Risk: max DD -57%, broken fundamentals.</td>
    </tr>
    <tr style="background:#fef2f2;border-bottom:1px solid #fecaca;">
      <td style="padding:10px 8px;text-align:center;font-weight:800;color:#0f172a;">4</td>
      <td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:#dc2626;letter-spacing:0.3px;">AVOID</span></td>
      <td style="padding:10px 8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:13px;color:#0f172a;">FTNT</td>
      <td style="padding:10px 8px;color:#334155;">Massive insider selling ($42M at current prices), institutional exodus, only 31% BUY, -9.9% EPS growth.</td>
      <td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:#0f172a;">70</td>
      <td style="padding:10px 8px;color:#64748b;font-style:italic;font-size:11px;">Tech: only stock above SMA50 (weak uptrend). Macro: sector tailwinds apply.</td>
    </tr>
    <tr style="background:#fef2f2;border-bottom:1px solid #fecaca;">
      <td style="padding:10px 8px;text-align:center;font-weight:800;color:#0f172a;">5</td>
      <td style="padding:10px 8px;text-align:center;"><span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;color:#fff;background:#dc2626;letter-spacing:0.3px;">AVOID</span></td>
      <td style="padding:10px 8px;font-family:'SF Mono',Consolas,monospace;font-weight:700;font-size:13px;color:#0f172a;">CYBR</td>
      <td style="padding:10px 8px;color:#334155;">Now part of PANW ($25B acquisition). Activist/M&amp;A speculation, zero insider ownership, only 32% BUY.</td>
      <td style="padding:10px 8px;text-align:center;font-weight:800;font-size:14px;color:#0f172a;">75</td>
      <td style="padding:10px 8px;color:#64748b;font-style:italic;font-size:11px;">Hedge fund accumulation may reflect M&amp;A arb, not fundamentals.</td>
    </tr>
  </table>
</div>""")

    # FOOTER
    html.append("""
<div style="padding:24px 40px 28px 40px;background:#f8fafc;border-top:1px solid #e2e8f0;">
  <table style="width:100%;">
    <tr>
      <td style="font-size:10px;color:#94a3b8;line-height:1.6;">
        <div style="font-weight:600;color:#64748b;margin-bottom:2px;">Investment Committee Report</div>
        Generated March 6, 2026 &middot; 7 Specialist Agents (Claude Sonnet) + CIO Synthesis (Claude Opus)<br/>
        Data: etorotrade signals, eToro census (1,500 PIs), yfinance market data, web search
      </td>
      <td style="text-align:right;font-size:10px;color:#cbd5e0;vertical-align:bottom;">
        For informational purposes only.<br/>Not financial advice.
      </td>
    </tr>
  </table>
</div>

</div>
</body>
</html>""")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(html))

    print(f"Report written to {OUTPUT}")
    print(f"File size: {os.path.getsize(OUTPUT):,} bytes")


if __name__ == "__main__":
    generate()

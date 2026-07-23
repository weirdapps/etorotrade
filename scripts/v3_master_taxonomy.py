"""Render the v3 MASTER factor taxonomy — the single decision table.

Six dimensions (size / value+quality / growth / momentum / risk / analysts). Every metric we
have access to, with role (active / gate / sizing / watch / discard), weight (active only),
best-available 10yr evidence (from v3_fundamentals_stats.json), and rationale. Plus the locked
gates + the sizing rule. Pure rendering — no backtest.

    .venv/bin/python scripts/v3_master_taxonomy.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

STATS = json.load(open(os.path.expanduser("~/.weirdapps-trading/v3_fundamentals_stats.json"))).get(
    "factors", {}
)

# metric -> 10yr stats key (None = no 10yr: 5mo/live-only or gate with no scored backtest)
SK = {
    "roa": "roa",
    "roa/roe": "roa",
    "pe_trailing": "earnings_yield",
    "gp_assets": "gp_assets",
    "net_issuance": "net_issuance",
    "fcf": "fcf_yield",
    "value_ps": "sales_yield",
    "value_pb": "book_to_price",
    "roe": "roe",
    "op_margin": "op_margin",
    "ev_ebitda": "ev_ebitda",
    "gross_margin": "gross_margin",
    "accruals": "accruals",
    "current_ratio": "current_ratio",
    "de": "de",
    "earn_growth": "earn_growth",
    "earn_stability": "earn_stability",
    "sue": "sue",
    "rev_growth": "rev_growth",
    "pct_52w_high": "pct_52w_high",
    "mom_12_1": "mom_12_1",
    "price_perf": "price_perf",
    "residual_mom": "residual_mom",
    "reversal_1m": "reversal_1m",
    "realized_vol": "realized_vol",
    "beta": "beta",
    "div_yield": "div_yield",
    "asset_growth": "asset_growth",
}
NO10 = {
    "earn_trajectory",
    "trajectory_spread",
    "analyst_mom",
    "upside",
    "buy_pct",
    "target_dispersion",
    "analyst_count",
    "pe_forward",
    "expected_return",
    "short_interest",
    "piotroski",
}

# (dimension, metric, role, weight_pct, rationale). role: active | active* | gate | sizing | watch | discard
TAX = [
    # ---- SIZE ----
    (
        "SIZE",
        "market_cap",
        "sizing",
        None,
        "SIZING input (bigger/steadier → larger cap) + PROMINENT display of what size we hold. Size premium itself weak — not a scored bet.",
    ),
    (
        "SIZE",
        "adv_usd",
        "gate",
        None,
        "Tradeability: ADV(20d) ≥ $3M AND target position ≤ 10-15% of one day's ADV (full exit clears).",
    ),
    (
        "SIZE",
        "amihud",
        "watch",
        None,
        "Illiquidity/price-impact — UNTESTED (needs volume ingest). Finer ADV; sizing/cost, not alpha.",
    ),
    # ---- VALUE (incl QUALITY) ----
    (
        "VALUE (incl quality)",
        "roe",
        "active",
        13,
        "leverage-clean profitability level. NOTE: the engine scores ROE for ALL names; the sector-conditional ROA/ROE split is documented but NOT implemented (ROE-only today).",
    ),
    (
        "VALUE (incl quality)",
        "gp_assets",
        "active",
        13,
        "t2.98 — the cleanest diversifier (ρ<0.18 to every factor). Novy-Marx quality anchor.",
    ),
    (
        "VALUE (incl quality)",
        "net_issuance",
        "active",
        9,
        "t2.95 — buyback(+)/dilution(−) capital-allocation quality (Pontiff-Woodgate). Bumped above P/S per evidence-order.",
    ),
    (
        "VALUE (incl quality)",
        "fcf",
        "active",
        7,
        "t2.80 — distinct cash-quality channel (ρ 0.26-0.45 vs profitability level).",
    ),
    (
        "VALUE (incl quality)",
        "value_ps",
        "active",
        7,
        "P/S (sales-yield), sector-conditional DEFAULT — trap-immune value lead; trimmed to 7 (t1.07 ≪ net_issuance).",
    ),
    (
        "VALUE (incl quality)",
        "pe_forward",
        "active",
        6,
        "OWNER CALL: full scored value factor REPLACING pe_trailing. EARN-IN — 5mo/live only, no 10yr PIT; caveat: analyst-optimism bias (worst for growth names).",
    ),
    (
        "VALUE (incl quality)",
        "value_pb",
        "active",
        3,
        "P/B, sector-FENCED — financials/REITs only (NAV / regulatory-capital proxy).",
    ),
    (
        "VALUE (incl quality)",
        "div_yield",
        "sizing",
        None,
        "SIZING input — quality/stability lens (moderate yield up-sizes; >6% = value-trap flag, no up-size). Not scored.",
    ),
    (
        "VALUE (incl quality)",
        "earn_trajectory",
        "gate",
        None,
        "Value-trap GATE: fwd P/E > 1.10× trailing → ineligible for new BUY; held trap → SELL.",
    ),
    (
        "VALUE (incl quality)",
        "current_ratio",
        "gate",
        None,
        "Loose solvency: <1.0 flag, <0.8 + rising leverage → ineligible. SUPPRESSED for financials.",
    ),
    (
        "VALUE (incl quality)",
        "de",
        "gate",
        None,
        "Loose leverage screen on extreme non-financial gearing. SUPPRESSED for financials.",
    ),
    (
        "VALUE (incl quality)",
        "piotroski",
        "gate",
        None,
        "delta-F (0-5) value-trap discriminator within the value sleeve (earn-in). Components, NOT the aggregate score.",
    ),
    (
        "VALUE (incl quality)",
        "pe_trailing",
        "discard",
        None,
        "Replaced by pe_forward (owner call). t≈0 standalone this regime.",
    ),
    (
        "VALUE (incl quality)",
        "op_margin",
        "discard",
        None,
        "ρ0.68 w/ roa — ~0 incremental (confirmed). Double-counts profitability.",
    ),
    (
        "VALUE (incl quality)",
        "ev_ebitda",
        "discard",
        None,
        "ρ0.71 w/ pe, t≈0 — and the wrong metric for banks (P/B is the financials value axis).",
    ),
    ("VALUE (incl quality)", "gross_margin", "discard", None, "t≈0 — superseded by gp_assets."),
    ("VALUE (incl quality)", "accruals", "discard", None, "t−1.9, no clean alpha (Sloan)."),
    # ---- GROWTH (earnings) ----
    (
        "GROWTH",
        "earn_growth",
        "active",
        8,
        "t2.67 — the scored earnings/growth slot (persistent, fits monthly cadence).",
    ),
    (
        "GROWTH",
        "sue",
        "active",
        6,
        "t2.02 — PEAD. Bumped to 6 (owner call). Note: event-trigger — monthly cadence captures only part of the drift.",
    ),
    (
        "GROWTH",
        "trajectory_spread",
        "active",
        5,
        "PET/PEF spread, earn-in (5mo/no-PIT) — the forward view scored as a SPREAD (was the strongest clean signal, t2.17).",
    ),
    (
        "GROWTH",
        "earn_stability",
        "active",
        4,
        "t2.36 — QMJ 'safety' (low earnings coefficient-of-variation). DISTINCT (ρ≤0.18 to all). Borderline/probation.",
    ),
    (
        "GROWTH",
        "rev_growth",
        "watch",
        None,
        "t1.66 sub-bar — margin-CONDITIONING variable (reward growth only WITH margin expansion; CGS-trap guard).",
    ),
    # ---- MOMENTUM ----
    (
        "MOMENTUM",
        "pct_52w_high",
        "active",
        14,
        "t3.77 — strongest single momentum (George-Hwang). The ONE momentum slot.",
    ),
    (
        "MOMENTUM",
        "residual_mom",
        "watch",
        None,
        "Tested t0.79 — no lift over pct_52w_high here, but conceptually distinct → WATCH (not discarded).",
    ),
    (
        "MOMENTUM",
        "reversal_1m",
        "watch",
        None,
        "Tested t0.05 — dead at monthly cadence; kept as an entry-timing overlay candidate → WATCH.",
    ),
    (
        "MOMENTUM",
        "mom_12_1",
        "discard",
        None,
        "ρ0.95 w/ price_perf, 0.66-0.75 w/ pct_52w_high — clone.",
    ),
    ("MOMENTUM", "price_perf", "discard", None, "ρ0.95 w/ mom_12_1 — outright clone."),
    # ---- RISK (sizing / gate, NOT alpha) ----
    (
        "RISK",
        "realized_vol",
        "sizing",
        None,
        "Inverse-vol SIZING (raw IC real but a sector+rate-duration collinearity artifact — not scored).",
    ),
    (
        "RISK",
        "beta",
        "sizing",
        None,
        "De-gross only (raw cross-sectional IC ≈ 0). Vasicek-shrunk. Never lever beta up.",
    ),
    (
        "RISK",
        "short_interest",
        "gate",
        None,
        "GRADED: <10% full · 10-20% SOFT size-down ×0.5 + flag · ≥20% HARD exclude.",
    ),
    (
        "RISK",
        "MAX",
        "discard",
        None,
        "Lottery-demand — subsumed by realized_vol (red-team reject).",
    ),
    # ---- EXPECTATIONS (analyst / forward-looking; yfinance, no PIT) ----
    (
        "EXPECTATIONS",
        "analyst_mom",
        "active",
        5,
        "EPS-revision momentum — bumped to 5 (owner call). EARN-IN: no 10yr PIT, KILL on 3mo negative live-IC. yfinance ~5mo git-panel + growing.",
    ),
    (
        "EXPECTATIONS",
        "analyst_count",
        "gate",
        None,
        "≥3 covering analysts required to APPLY any analyst overlay (kills single-analyst noise).",
    ),
    (
        "EXPECTATIONS",
        "upside",
        "gate",
        None,
        "LOOSE downside-guard: < −5% → size ×0.75; < −15% & ≥5 analysts → exclude. NEVER rewards high upside (anti-selection).",
    ),
    (
        "EXPECTATIONS",
        "buy_pct",
        "gate",
        None,
        "Confidence weight + sell-tail guard: < 25% (majority SELL) → size-down. NOT a ⅔ hard gate (that re-imports anti-selection).",
    ),
    (
        "EXPECTATIONS",
        "target_dispersion",
        "gate",
        None,
        "Analyst-confidence: high dispersion → down-weight analyst signals + small size haircut (×0.85).",
    ),
    (
        "EXPECTATIONS",
        "expected_return",
        "watch",
        None,
        "buy% × upside — derivative of the gate inputs; MONITORED, does NOT participate in any decision.",
    ),
]

ROLE_BADGE = {
    "active": ("#0a7d33", "#e5f6ea"),
    "active*": ("#0a7d33", "#e5f6ea"),
    "gate": ("#1d4ed8", "#e7edfd"),
    "sizing": ("#0e7490", "#e0f2f7"),
    "watch": ("#916516", "#f8f1df"),
    "discard": ("#6b7280", "#eef0f2"),
}
GATES = [
    ("ADV tradeability (HARD)", "ADV(20d) ≥ $3M AND target position ≤ 10-15% of one day's ADV."),
    (
        "Short interest (GRADED)",
        "SI/float < 10% full size · 10-20% soft ×0.5 + flag · ≥ 20% HARD exclude.",
    ),
    (
        "Value-trap earn_trajectory (HARD)",
        "fwd P/E > 1.10× trailing → ineligible new BUY; held → SELL.",
    ),
    (
        "Solvency current_ratio (LOOSE)",
        "< 1.0 flag · < 0.8 + rising leverage → ineligible. Suppressed for financials.",
    ),
    (
        "Leverage D/E (LOOSE)",
        "Extreme non-financial gearing → ineligible. Suppressed for financials.",
    ),
    (
        "Analyst coverage (HARD, signal-only)",
        "≥ 3 covering analysts required to apply any analyst overlay.",
    ),
    (
        "Upside downside-guard (SOFT)",
        "< −5% → size ×0.75 · < −15% & ≥5 analysts → exclude. Never rewards high upside.",
    ),
    (
        "Buy% sell-tail (SOFT)",
        "< 25% (majority SELL) → size-down. Confidence weight otherwise; NOT a ⅔ hard gate.",
    ),
]
SIZING = (
    "<b>Inputs: market_cap, div_yield, realized_vol, beta</b> (all role=sizing). "
    "Post-selection (alpha picks WHICH names, risk sets HOW MUCH). "
    "weight ∝ conviction × (1 / max(beta, 0.6)) × (1 + 0.5·div_yield) × size_tier, then capped. "
    "Beta de-gross only (Vasicek-shrunk); div_yield rewarded 1-4% (stability) but > 6% = value-trap flag (NO up-size); "
    "vol-floor so utilities aren't over-loaded; high-vol mega (NVDA-type) haircut to ~7%."
)
CAPS = [
    ("Mega > $200B", "10%"),
    ("Large $20-200B", "6%"),
    ("Mid $2-20B", "2%"),
    ("Small $1-2B", "0.5%"),
    ("Micro < $1B", "0.25%"),
]


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "—"


def _ev(metric):
    k = SK.get(metric)
    if k and k in STATS:
        s = STATS[k]
        return "10yr", "#0a7d33", s.get("ic_neu"), s.get("t_hac"), s.get("hit")
    return ("5mo/live" if metric in NO10 else "—"), "#916516", None, None, None


def _rows(dim):
    out = ""
    for d, m, role, wt, why in TAX:
        if d != dim:
            continue
        fg, bg = ROLE_BADGE.get(role, ("#6b7280", "#eef0f2"))
        win, wc, icn, t, hit = _ev(m)
        nc = "#0a7d33" if (icn or 0) > 0 else "#b91c1c"
        tw = "700" if abs(t or 0) >= 2 else "400"
        wtxt = (
            f"<b>{wt}%</b>"
            if isinstance(wt, (int, float))
            else ("<span style='color:#9aa1a9'>—</span>")
        )
        out += (
            f"<tr><td><code>{m}</code></td>"
            f"<td><span class='b' style='color:{fg};background:{bg}'>{role}</span></td>"
            f"<td>{wtxt}</td><td class='win' style='color:{wc}'>{win}</td>"
            f"<td style='color:{nc}'>{_f(icn)}</td><td style='font-weight:{tw}'>{_f(t, 2)}</td>"
            f"<td>{_f(hit, 2)}</td><td class='mdesc'>{why}</td></tr>"
        )
    return out


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_master_taxonomy.html")
    active = [(m, wt) for _, m, r, wt, _ in TAX if r == "active" and isinstance(wt, (int, float))]
    tot = sum(wt for _, wt in active)
    dims = ["SIZE", "VALUE (incl quality)", "GROWTH", "MOMENTUM", "RISK", "EXPECTATIONS"]
    blocks = ""
    for dim in dims:
        blocks += (
            f"<h2>{dim}</h2><table><thead><tr><th>Metric</th><th>Role</th><th>Weight</th>"
            "<th>window</th><th>IC β-neut</th><th>t (HAC)</th><th>hit</th><th>Rationale</th>"
            f"</tr></thead><tbody>{_rows(dim)}</tbody></table>"
        )
    gaterows = "".join(f"<tr><td><b>{g}</b></td><td class='mdesc'>{r}</td></tr>" for g, r in GATES)
    caprows = "".join(f"<tr><td>{t}</td><td><b>{c}</b></td></tr>" for t, c in CAPS)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 Master Factor Taxonomy</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:1060px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:14px}}
h2{{font-size:16px;margin:24px 0 6px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}}
th{{text-align:left;background:#f7f8fa;padding:6px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2;vertical-align:top}}
code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:11.5px}}
.mdesc{{color:#374151;font-size:11.5px;max-width:440px}}.win{{font-size:11px;text-align:center}}
.b{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:20px}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0}}.two{{display:flex;gap:24px;flex-wrap:wrap}}.two>div{{flex:1;min-width:320px}}</style></head><body>
<h1>v3 Master Factor Taxonomy — the decision table</h1>
<div class="sub">{stamp} UTC · live ~$1.1M book · 6 dimensions × every metric × role × weight × 10yr evidence.
Active (scored) Σ = <b>{tot}%</b>. <b>Weights frozen / earn-in — proposal for sign-off, not executed.</b><br>
<b>Effective weights:</b> per-metric numbers are the evidence-ranked TARGET; the live engine (combine.py) applies each CLUSTER weight with an EQUAL MEAN of its metrics, so cluster totals match but the per-metric split is approximate, not executed as shown.<br>
Roles: <span class="b" style="color:#0a7d33;background:#e5f6ea">active</span> scored ·
<span class="b" style="color:#0a7d33;background:#e5f6ea">active*</span> sector-substitute ·
<span class="b" style="color:#1d4ed8;background:#e7edfd">gate</span> eligibility screen ·
<span class="b" style="color:#0e7490;background:#e0f2f7">sizing</span> position-size input ·
<span class="b" style="color:#916516;background:#f8f1df">watch</span> shadow/monitor ·
<span class="b" style="color:#6b7280;background:#eef0f2">discard</span> dropped.
10yr = survivorship-clean Sharadar (114 mo); 5mo/live = yfinance git-panel, earn-in. Bold t = |t| ≥ 2.</div>
{blocks}
<h2>Gates (eligibility screens)</h2>
<table><thead><tr><th>Gate</th><th>Rule</th></tr></thead><tbody>{gaterows}</tbody></table>
<h2>Sizing rule &amp; caps</h2>
<div class="two"><div><p class="note">{SIZING}</p></div>
<div><table><thead><tr><th>Cap tier</th><th>Max single name</th></tr></thead><tbody>{caprows}</tbody></table>
<p class="note">+ portfolio guards: max sector ~25-30% · top-10 ≤ ~55-60% · long gross ≤ 100%.</p></div></div>
<h2>Sector recipes</h2>
<p class="note"><b>Profitability:</b> ROA + gp_assets + fcf (default) · Financials/REITs → <b>ROE</b> (gp_assets undefined, leverage is the business). ·
<b>Value:</b> P/S (default) · Financials/REITs → <b>P/B</b>. · Z-score EVERY factor WITHIN GICS sector. · Suppress current_ratio/D-E for financials.</p>
</body></html>"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"master taxonomy -> {out}  | active Σ={tot}%")


if __name__ == "__main__":
    main()

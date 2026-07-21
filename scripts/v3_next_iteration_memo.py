"""Render the v3 next-iteration decision memo (one self-contained HTML).

Combines three artifacts already on disk:
  - ~/.weirdapps-trading/v3_fundamentals_stats.json   (10yr survivorship-clean per-factor IC/t/hit)
  - ~/.weirdapps-trading/v3_redundancy.json           (time-avg cross-sectional correlations)
  - ~/.weirdapps-trading/v3_debate.json               (CIO synthesis + red-team from the 7-agent debate)

Sections: exec summary -> decision table (proposed weights vs evidence, post-debate) -> redundancy
evidence -> missing factors / shadow earn-in book -> analyst-data sourcing -> open risks ->
parameter benchmark by cluster. No backtest re-run — pure rendering.

    .venv/bin/python scripts/v3_next_iteration_memo.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

W = os.path.expanduser("~/.weirdapps-trading")
STATS = json.load(open(f"{W}/v3_fundamentals_stats.json")).get("factors", {})
RED = json.load(open(f"{W}/v3_redundancy.json"))
DEB = json.load(open(f"{W}/v3_debate.json"))
SYN = DEB["synth"]

# metric token -> 10yr fundamentals-stats key (evidence lookup)
STAT_KEY = {
    "gp_assets": "gp_assets",
    "roa": "roa",
    "roe": "roe",
    "op_margin": "op_margin",
    "fcf": "fcf_yield",
    "pct_52w_high": "pct_52w_high",
    "mom_12_1": "mom_12_1",
    "price_perf": "price_perf",
    "realized_vol": "realized_vol",
    "beta": "beta",
    "sue": "sue",
    "earn_growth": "earn_growth",
    "rev_growth": "rev_growth",
    "div_yield": "div_yield",
    "ev_ebitda": "ev_ebitda",
    "pe": "earnings_yield",
    "pb": "book_to_price",
    "value_composite": "sales_yield",
    "de": "de",
    "current_ratio": "current_ratio",
    "gross_margin": "gross_margin",
    "accruals": "accruals",
    "asset_growth": "asset_growth",
}
# current live per-metric weight (cluster model: cluster_wt / n_members); recipe = sector-conditional
CUR = {
    "roe": 8.0,
    "fcf": 8.0,
    "gp_assets": 8.0,
    "pe": 9.0,
    "ev_ebitda": 9.0,
    "mom_12_1": 9.0,
    "pct_52w_high": 9.0,
    "earn_growth": 1.5,
    "rev_growth": 1.5,
    "beta": 4.5,
    "realized_vol": 4.5,
    "sue": 10.0,
    "earn_trajectory": 10.0,
    "value_composite": 18.0,
}
NO10YR = {
    "earn_trajectory",
    "short_interest",
    "analyst_mom",
    "upside",
    "buy_pct",
    "pe_forward",
    "peg",
}
TIER_ORDER = ["participate", "sizing-only", "watch", "gate", "drop"]
TIER_BADGE = {
    "participate": ("#0a7d33", "#e5f6ea"),
    "sizing-only": ("#0e7490", "#e0f2f7"),
    "watch": ("#916516", "#f8f1df"),
    "gate": ("#1d4ed8", "#e7edfd"),
    "drop": ("#6b7280", "#eef0f2"),
    "add-now": ("#0a7d33", "#e5f6ea"),
    "shadow-earn-in": ("#7c3aed", "#f1e9fd"),
    "reject": ("#6b7280", "#eef0f2"),
}


def _tok(metric: str) -> str:
    return metric.split(" (")[0].split(" /")[0].strip()


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "—"


def _evidence(token: str):
    k = STAT_KEY.get(token)
    if k and k in STATS:
        s = STATS[k]
        return "10yr", s.get("ic_neu"), s.get("t_hac"), s.get("hit")
    win = "5mo·no10yr" if token in NO10YR else "—"
    return win, None, None, None


def _decision_table() -> str:
    rows_by_tier: dict[str, list] = {t: [] for t in TIER_ORDER}
    for fx in SYN["factor_set"]:
        rows_by_tier.setdefault(fx["tier"], []).append(fx)
    body = ""
    labels = {
        "participate": ("● PARTICIPATE — scored (Σ ≈ 100%)", "#e5f6ea"),
        "sizing-only": ("◆ SIZING-ONLY — risk input, not alpha (0 selection weight)", "#e0f2f7"),
        "watch": ("◐ WATCH / conditioning — 0 weight", "#f8f1df"),
        "gate": ("▣ GATES — eligibility screens, no weight", "#e7edfd"),
        "drop": ("○ DROP — redundant / no clean alpha", "#eef0f2"),
    }
    for tier in TIER_ORDER:
        items = rows_by_tier.get(tier, [])
        if not items:
            continue
        if tier == "participate":
            items = sorted(items, key=lambda x: -x["proposed_weight_pct"])
        lab, bg = labels[tier]
        body += f"<tr><td colspan='9' style='background:{bg};font-weight:700;padding:6px 9px'>{lab}</td></tr>"
        for fx in items:
            tok = _tok(fx["metric"])
            win, icn, t, hit = _evidence(tok)
            pw = fx["proposed_weight_pct"]
            cw = CUR.get(tok, 0.0)
            pwt = f"<b>{pw:.0f}%</b>" if pw else "<span style='color:#6b7280'>0</span>"
            cwt = f"{cw:.1f}%" if isinstance(cw, (int, float)) and cw else "—"
            if pw or cw:
                d = pw - (cw if isinstance(cw, (int, float)) else 0)
                dc = "#0a7d33" if d > 0.05 else ("#b91c1c" if d < -0.05 else "#6b7280")
                dtxt = f"<span style='color:{dc}'>{d:+.0f}</span>"
            else:
                dtxt = "—"
            wc = "#0a7d33" if win == "10yr" else "#916516"
            nc = "#0a7d33" if (icn or 0) > 0 else "#b91c1c"
            tw = "700" if abs(t or 0) >= 2 else "400"
            body += (
                f"<tr><td><code>{fx['metric']}</code></td><td>{pwt}</td><td style='color:#6b7280'>{cwt}</td>"
                f"<td>{dtxt}</td><td class='win' style='color:{wc}'>{win}</td>"
                f"<td style='color:{nc}'>{_f(icn)}</td><td style='font-weight:{tw}'>{_f(t, 2)}</td>"
                f"<td>{_f(hit, 2)}</td><td class='mdesc'>{fx['rationale']}</td></tr>"
            )
    return (
        "<h2>1 · Decision table — proposed weights vs evidence (post-debate)</h2>"
        '<p class="note">The de-duplicated set from the debate. <b>Proposed</b> = diversified equal-risk after '
        "collapsing redundant pairs; <b>Δ</b> = proposed − current. Judge each weight against its 10yr "
        "<b>t (HAC)</b>. Bold t = |t| ≥ 2. <b>sizing-only</b> = real risk signal used for inverse-vol sizing, "
        "not scored as alpha.</p>"
        "<table><thead><tr><th>Metric</th><th>Proposed</th><th>Current</th><th>Δ pp</th><th>window</th>"
        "<th>IC β-neut</th><th>t (HAC)</th><th>hit</th><th>Rationale</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _redundancy() -> str:
    g = RED["groups"]
    out = ""
    for name, label in [
        ("profitability", "Profitability"),
        ("momentum", "Momentum"),
        ("low_vol", "Low-vol"),
        ("value", "Value"),
        ("growth_income", "Growth / Income"),
    ]:
        if name not in g:
            continue
        blk = g[name]
        mem = blk["members"]
        head = "".join(f"<th>{m}</th>" for m in mem)
        rows = ""
        for a in mem:
            cells = ""
            for b in mem:
                v = blk["corr"].get(a, {}).get(b)
                hot = "#b91c1c" if (v or 0) >= 0.85 and a != b else "#1a1a1a"
                cells += f"<td style='color:{hot};text-align:center'>{_f(v, 2)}</td>"
            rows += f"<tr><td><code>{a}</code></td>{cells}</tr>"
        out += (
            f"<h3 style='margin-top:14px'>{label} — avg off-diag ρ {blk['avg_offdiag_corr']:.2f} · "
            f"combo IC {blk['combo_ic']:.4f} vs best single {blk['best_single']} "
            f"({blk['single_ic'][blk['best_single']]:.4f}) → uplift {blk['uplift_vs_best']:+.4f}</h3>"
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table>"
        )
    return (
        "<h2>2 · Why — redundancy evidence (time-avg cross-sectional ρ)</h2>"
        '<p class="note">ρ ≥ 0.85 (red) = the same bet. Combining redundant members does NOT lift IC — proof '
        "that piling on correlated metrics silently over-weights one bet without adding signal.</p>"
        + out
    )


def _missing() -> str:
    rows = ""
    for m in SYN["missing_to_add"]:
        fg, bg = TIER_BADGE.get(m["verdict"], ("#6b7280", "#eef0f2"))
        badge = f'<span class="b" style="color:{fg};background:{bg}">{m["verdict"]}</span>'
        rows += (
            f"<tr><td><b>{m['name']}</b> {badge}</td><td class='mdesc'>{m['data_need']}</td>"
            f"<td class='mdesc'>{m['rationale']}</td></tr>"
        )
    return (
        "<h2>3 · Missing factors — shadow earn-in book</h2>"
        '<p class="note">What the 5 lenses + red-team say we lack. <b>add-now</b> = ship (construction, not alpha); '
        "<b>shadow-earn-in</b> = compute at ZERO live weight, promote only when the live IC logger clears its "
        "net-of-cost hurdle; <b>reject</b> = not worth it. Almost all are <b>free</b> (Sharadar/price we already own).</p>"
        "<table><thead><tr><th>Factor</th><th>Data need</th><th>Rationale</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _sourcing() -> str:
    items = "".join(f"<li>{s}</li>" for s in SYN["data_sourcing"])
    return (
        "<h2>4 · Analyst / forward-data sourcing</h2>"
        '<p class="note"><b>Bottom line: do NOT buy an analyst-estimate vendor at $1.1M</b> — the fixed cost dwarfs '
        "the marginal, fast-decaying, crowded edge; and only estimate <b>revisions</b> work (levels — upside %, "
        "buy % — are dead, t −0.15). Use free TipRanks as a live-only shadow first.</p>"
        f"<ul class='src'>{items}</ul>"
    )


def _risks() -> str:
    items = "".join(f"<li>{r}</li>" for r in SYN["open_risks"])
    return (
        "<h2>5 · Open risks (read before trusting any weight)</h2>"
        '<p class="note">This is real money on ~1 regime of data. The honest caveats:</p>'
        f"<ul class='risk'>{items}</ul>"
    )


def _benchmark() -> str:
    clusters = {
        "Quality / profitability": [
            "gp_assets",
            "roa",
            "roe",
            "op_margin",
            "fcf_yield",
            "gross_margin",
            "accruals",
        ],
        "Value": ["sales_yield", "earnings_yield", "book_to_price", "ev_ebitda"],
        "Momentum": ["pct_52w_high", "mom_12_1", "price_perf"],
        "Low-vol": ["realized_vol", "beta"],
        "PEAD": ["sue"],
        "Growth / Income": ["earn_growth", "rev_growth", "div_yield"],
        "Investment": ["asset_growth"],
    }
    out = ""
    for cl, keys in clusters.items():
        rows = ""
        for k in keys:
            if k not in STATS:
                continue
            s = STATS[k]
            icn, t, hit = s.get("ic_neu"), s.get("t_hac"), s.get("hit")
            nc = "#0a7d33" if (icn or 0) > 0 else "#b91c1c"
            tw = "700" if abs(t or 0) >= 2 else "400"
            rows += (
                f"<tr><td><code>{k}</code></td><td style='color:{nc}'>{_f(icn)}</td>"
                f"<td style='font-weight:{tw}'>{_f(t, 2)}</td><td>{_f(hit, 2)}</td>"
                f"<td>{s.get('n_dates', '')}</td></tr>"
            )
        if rows:
            out += (
                f"<h3 style='margin-top:14px'>{cl}</h3>"
                "<table><thead><tr><th>Metric</th><th>IC β-neut</th><th>t (HAC)</th><th>hit</th>"
                f"<th>dates</th></tr></thead><tbody>{rows}</tbody></table>"
            )
    return (
        "<h2>6 · Parameter benchmark by cluster — 10yr survivorship-clean</h2>"
        '<p class="note">Every 10yr-testable parameter, grouped. Sharadar SF1+DAILY, 114 monthly cross-sections '
        "2017-2026, delisted-inclusive, β-neutral forward IC, HAC t. Analyst / forward metrics omitted (no 10yr "
        "data). Bold t = |t| ≥ 2.</p>" + out
    )


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_next_iteration_memo.html")
    changes = "".join(f"<li>{c}</li>" for c in SYN.get("key_changes", []))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>v3 Factor Model — Next-Iteration Decision Memo</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:1040px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:16px}}
h2{{font-size:17px;margin:26px 0 8px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
h3{{font-size:13.5px;margin:12px 0 5px;color:#1a1a1a}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}}
th{{text-align:left;background:#f7f8fa;padding:6px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2;vertical-align:top}}
code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:11.5px}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0}}.mdesc{{color:#374151;font-size:12px;max-width:430px}}
.win{{font-size:11px;text-align:center}}.b{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:20px;margin-left:4px}}
.exec{{background:#f7f8fa;border-left:3px solid #1a1a1a;padding:12px 16px;font-size:13px;line-height:1.55;border-radius:0 6px 6px 0}}
ul.src li,ul.risk li{{margin-bottom:7px;font-size:12.5px;line-height:1.5}}
ul.risk li{{color:#7a1f1f}}</style></head><body>
<h1>v3 Factor Model — Next-Iteration Decision Memo</h1>
<div class="sub">Generated {stamp} UTC · live ~$1.1M book · method: 10yr survivorship-clean quant (114 monthly
cross-sections) → redundancy/incremental-IC → 5 expert lenses → red-team → CIO synthesis (7 agents).
<b>Weights stay frozen / earn-in — this is a de-duplication + honesty pass, a proposal for discussion, not an executed change.</b></div>
<h2>0 · Executive summary</h2>
<div class="exec">{SYN["executive_summary"]}</div>
<h3 style="margin-top:14px">Key changes vs current</h3><ul class="src">{changes}</ul>
{_decision_table()}
{_redundancy()}
{_missing()}
{_sourcing()}
{_risks()}
{_benchmark()}
</body></html>"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"memo -> {out}")


if __name__ == "__main__":
    main()

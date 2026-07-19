"""v3 model pipeline — indicator participation map.

Deterministic, network-free: reads the LIVE combiner config (clusters, weights,
directions, and the P1 sector-conditional value recipes) and renders one HTML that
shows, for every raw indicator, how it flows into the decision:

  SCORED     -> feeds the weighted conviction (a live cluster member)
  GATE       -> exclude / screen (never scored)
  MONITORED  -> shadow-logged + backtested, ZERO conviction (context only)
  DISCARDED  -> removed from scoring (validated by the per-parameter backtest)

    .venv/bin/python scripts/v3_pipeline_map.py

Writes ~/Downloads/<UTCstamp>_v3_pipeline.html. The non-scored classification is the
documented FIX-NOW / redesign decision set; everything scored is read from code.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.combine import (  # noqa: E402
    CLUSTER_WEIGHTS,
    CLUSTERS,
    DIRECTION,
    VALUE_GROUP_RECIPES,
    VALUE_WEIGHT_MULT,
)

LABEL = {
    "pe_trailing": "P/E (trailing)",
    "ev_ebitda": "EV/EBITDA",
    "pb": "P/B (book)",
    "ps_sector": "P/S (sales)",
    "roe": "ROE",
    "fcf": "Free cash flow",
    "mom_12_1": "Momentum 12-1",
    "pct_52w_high": "% of 52-week high",
    "earn_growth": "Earnings growth",
    "rev_growth": "Revenue growth",
    "beta": "Beta",
    "realized_vol": "Realized vol",
    "analyst_mom": "Analyst revision",
    "div_yield": "Dividend yield",
    "upside": "Analyst upside",
    "buy_pct": "Buy %",
    "peg": "PEG",
    "price_perf": "Price performance",
    "pe_forward": "P/E (forward)",
    "short_interest": "Short interest",
    "de": "Debt / equity",
    "current_ratio": "Current ratio",
    "roa": "ROA",
    "gross_margin": "Gross margin",
    "op_margin": "Operating margin",
    "accruals": "Accruals",
    "target_dispersion": "Target dispersion",
}
_DIRTXT = {1: "high = good", -1: "low = good"}

CLUSTER_LABEL = {
    "value_z": "Value",
    "quality_z": "Quality",
    "momentum_z": "Momentum",
    "growth_z": "Growth",
    "lowvol_z": "Low volatility",
    "strength_z": "Strength",
}

# Curated non-scored classification (documented FIX-NOW / redesign decisions).
GATES = {
    "short_interest": "squeeze / hard-to-borrow EXCLUDE gate (SI &gt; 20%) — not scored",
    "de": "distress screen (with current ratio) — financial strength, not profitability",
    "current_ratio": "distress screen (with D/E) — liquidity, not profitability",
}
MONITORED = {  # shadow-logged + backtested, ZERO conviction
    "div_yield": "mildly positive in the backtest; candidate for a utilities tilt (shadow)",
    "upside": "contrarian / contaminated (upside↔52w-high ρ=−0.72) — context only",
    "buy_pct": "analyst buy-% LEVEL ≈ zero IC — context only",
    "peg": "growth-adjusted value — shadow, noisy",
    "price_perf": "≈ momentum (ρ 0.95) — shown as context, double-count avoided",
}
DISCARDED = {
    "pe_forward": "near-duplicate of trailing P/E (ρ 0.75) — removed",
    "target_dispersion": "live-fetched, non-point-in-time — removed",
    "accruals": "needs a point-in-time balance sheet — BUILD / shadow",
    "roa": "retired for the interim quality set (ROE + FCF)",
    "gross_margin": "per-sales margin; the Novy-Marx anchor is GP/assets — BUILD",
    "op_margin": "per-sales margin; operating-profitability/assets — BUILD",
}


def _scored_members() -> set[str]:
    members = {m for ms in CLUSTERS.values() for m in ms}
    members.update({m for r in VALUE_GROUP_RECIPES.values() for m in r})  # P1 value candidates
    return members


def _value_recipe_html() -> str:
    rows = ""
    grp_name = {
        "A": "A · conventional",
        "B": "B · financials / REITs",
        "C": "C · growth / intangible",
    }
    for g, recipe in VALUE_GROUP_RECIPES.items():
        metrics = ", ".join(f"{LABEL.get(m, m)}×{w:g}" for m, w in recipe.items())
        mult = VALUE_WEIGHT_MULT.get(g, 1.0)
        multtxt = "" if mult == 1.0 else f" &nbsp;<b>(cluster weight ×{mult:g})</b>"
        rows += f"<tr><td>{grp_name.get(g, g)}</td><td>{metrics}{multtxt}</td></tr>"
    return (
        "<table><thead><tr><th>Sector group</th><th>Value recipe (P1 sector-conditional)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _scored_html() -> str:
    rows = ""
    for c, w in CLUSTER_WEIGHTS.items():
        if c == "value_z":
            metrics = "<i>sector-conditional — see below</i>"
        else:
            metrics = ", ".join(
                f"{LABEL.get(m, m)} <span class='d'>({_DIRTXT[DIRECTION.get(m, 1)]})</span>"
                for m in CLUSTERS.get(c, [])
            )
        rows += (
            f"<tr><td><b>{CLUSTER_LABEL.get(c, c)}</b></td>"
            f"<td style='text-align:right'>{w:.0%}</td><td>{metrics}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Cluster (scored)</th><th>Weight</th><th>Metrics</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _class_html(title: str, sub: str, items: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{k}</code> {LABEL.get(k, k)}</td><td>{v}</td></tr>"
        for k, v in items.items()
    )
    return (
        f"<h2>{title}</h2><p class='note'>{sub}</p>"
        f"<table><thead><tr><th>Indicator</th><th>Why / how</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def build_html() -> str:
    scored = _scored_members()
    n_scored = len(scored)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 Pipeline Map</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:900px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:18px}}
h2{{font-size:17px;margin:24px 0 6px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:4px 0 8px}}
th{{text-align:left;background:#f7f8fa;padding:7px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2;vertical-align:top}}
code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.d{{color:#8a929c;font-size:11.5px}}.note{{color:#5b6470;font-size:12.5px;margin:4px 0}}
.leg span{{display:inline-block;margin-right:14px;font-size:12px}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}}</style></head><body>
<h1>v3 Model Pipeline — indicator participation map</h1>
<div class="sub">How every raw indicator flows into the decision, read live from the combiner config
(reflects the merged P1 sector-conditional value). {n_scored} indicators are scored; the rest are
gates, shadow-monitored, or discarded — and each non-scored factor is measured in the
per-parameter backtest to validate its status.</div>
<p class="leg"><span><span class="dot" style="background:#0a7d33"></span>Scored</span>
<span><span class="dot" style="background:#b45309"></span>Gate</span>
<span><span class="dot" style="background:#2563eb"></span>Monitored (shadow)</span>
<span><span class="dot" style="background:#9aa1a9"></span>Discarded</span></p>
<h2>Scored — feeds the weighted conviction</h2>
<p class="note">Each raw metric → van-der-Waerden rank-z → directional sign → within-sector demean →
cluster-z (mean of members) → equal-vol standardized → weighted conviction. Weights are frozen
(equal-risk core); estimation is not fit on the short panel.</p>
{_scored_html()}
<h3 style="font-size:14px;margin:10px 0 2px">Value is sector-conditional (P1)</h3>
<p class="note">The value cluster uses a different recipe per sector group — banks/REITs on book &amp;
cash-flow, tech on sales — instead of one uniform P/E+EV/EBITDA. Value is never dropped or sign-flipped.</p>
{_value_recipe_html()}
{_class_html("Gates — exclude / screen (never scored)", "Risk screens applied before scoring; they remove names, they do not tilt.", GATES)}
{_class_html("Monitored — shadow-logged, ZERO conviction", "Logged forward for IC and shown as context; they do NOT move conviction until they clear the multi-year gate.", MONITORED)}
{_class_html("Discarded — removed from scoring", "Dropped for redundancy, contamination, or non-point-in-time data; kept in the backtest to keep validating the discard.", DISCARDED)}
<p class="note">Every non-scored indicator above is still measured in the per-parameter backtest
(the third tracked file) — nothing is dropped without being watched.</p>
</body></html>"""


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_pipeline.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_html())
    print(f"pipeline map -> {out}")


if __name__ == "__main__":
    main()

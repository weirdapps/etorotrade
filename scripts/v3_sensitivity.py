# scripts/v3_sensitivity.py
"""Trading Model v3 — factor-weight 3-way SENSITIVITY runner.

Question this answers: the default v3 model is value-heavy (Value cluster
penalises high P/E, P/S, P/B), so it wants to SELL the owner's successful
mega-cap growth core (NVDA / GOOG / MSFT / AMZN / AVGO / TSM / META / AAPL).
Does BROADENING the factor mix beyond value — adding a Growth cluster and
shifting weight toward momentum + growth — keep that core ON MERIT, or does the
core only survive when the weights are tuned to it?

It assembles the universe + enriches features ONCE (reusing the v3_full_report
assembly), then re-scores the SAME feature panel under three cluster-weight
schemes and diffs each resulting book against the live account:

    value_heavy    (the current model, growth OFF)
    balanced       (growth 0.15, momentum up, value down)
    growth_forward (growth 0.20, momentum 0.30, value 0.10)

For each config: n_buy / n_add / n_trim / n_sell / n_hold, CORE SURVIVAL for the
mega-cap growth core, top-12 target names, post-gate portfolio vol + deployment,
and a style tilt (weight-avg trailing P/E and weight-avg 12-1 momentum of the
target book). Renders a light-theme, warm-palette comparison HTML to
``~/Downloads/<UTCstamp>_v3_sensitivity.html`` and prints a console comparison.

Run (VPS / network allowed):   .venv/bin/python scripts/v3_sensitivity.py

The pure helpers (summarize_book / style_tilt / render_comparison_html) are
unit-tested with synthetic inputs; the network assembly in main() runs on the
VPS only. No module-level ``yahoofinance.core.config`` import (pipeline imports
are confined to main()).
"""

from __future__ import annotations

import html as _html
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- the three factor-weight schemes over the six clusters (short names) ------
# value_heavy is the CURRENT model (growth 0). The other two dial value down and
# growth + momentum up, weighted by evidence-style priors, NOT fitted to the book.
CONFIGS: dict[str, dict[str, float]] = {
    "value_heavy": {
        "value": 0.275,
        "quality": 0.275,
        "momentum": 0.20,
        "growth": 0.0,
        "lowvol": 0.15,
        "strength": 0.10,
    },
    "balanced": {
        "value": 0.15,
        "quality": 0.25,
        "momentum": 0.25,
        "growth": 0.15,
        "lowvol": 0.12,
        "strength": 0.08,
    },
    "growth_forward": {
        "value": 0.10,
        "quality": 0.22,
        "momentum": 0.30,
        "growth": 0.20,
        "lowvol": 0.10,
        "strength": 0.08,
    },
}

# The owner's proven mega-cap growth core — the survival test set.
CORE: list[str] = ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "AVGO", "TSM", "META"]

_EPS = 1e-6
_KEPT_ACTIONS = frozenset({"BUY", "ADD", "HOLD"})  # in-target = kept (vs SELL/TRIM)
_ACTION_COUNT_KEY = {
    "BUY": "n_buy",
    "ADD": "n_add",
    "TRIM": "n_trim",
    "SELL": "n_sell",
    "HOLD": "n_hold",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested with synthetic data)
# ---------------------------------------------------------------------------


def _to_float(value) -> float:
    """Coerce to float; None / non-numeric / NaN -> float('nan')."""
    if value is None:
        return float("nan")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _lookup(scored, ticker, col):
    """Best-effort scored[ticker, col] lookup; None when unavailable."""
    try:
        if scored is not None and col in scored.columns and ticker in scored.index:
            val = scored.at[ticker, col]
            return None if val is None else val
    except (KeyError, AttributeError, ValueError):
        pass
    return None


def summarize_book(actions: list[dict], scored, core_list: list[str], top_n: int = 12) -> dict:
    """Summarize one config's action book: counts + core survival + top names.

    Args:
        actions: build_actions output (list of per-ticker action dicts).
        scored: compute_scores frame (used only as a name/conviction fallback;
            may be an empty DataFrame).
        core_list: tickers whose survival is tracked (the mega-cap growth core).
        top_n: how many top target names to return (by target weight).

    Returns:
        ``{n_buy, n_add, n_trim, n_sell, n_hold, core_survival, core_kept,
        core_total, top_names}`` where ``core_survival`` is one row per
        ``core_list`` ticker (in order) as ``{ticker, action, kept}`` (kept =
        action in BUY/ADD/HOLD; ABSENT when the name is in neither book), and
        ``top_names`` is ``[{ticker, name, target_pct, conviction}]``.
    """
    counts = {"n_buy": 0, "n_add": 0, "n_trim": 0, "n_sell": 0, "n_hold": 0}
    by_ticker: dict[str, dict] = {}
    for a in actions:
        key = _ACTION_COUNT_KEY.get(a.get("action"))
        if key:
            counts[key] += 1
        by_ticker[a.get("ticker")] = a

    core_survival: list[dict] = []
    kept = 0
    for t in core_list:
        a = by_ticker.get(t)
        action = a.get("action") if a else "ABSENT"
        is_kept = action in _KEPT_ACTIONS
        kept += int(is_kept)
        core_survival.append({"ticker": t, "action": action, "kept": is_kept})

    targeted = [a for a in actions if _to_float(a.get("target_pct")) > _EPS]
    targeted.sort(key=lambda a: _to_float(a.get("target_pct")), reverse=True)
    top_names: list[dict] = []
    for a in targeted[:top_n]:
        tkr = a.get("ticker")
        name = a.get("name") or _lookup(scored, tkr, "name") or tkr
        conv = a.get("conviction")
        conv = (
            _to_float(conv) if conv is not None else _to_float(_lookup(scored, tkr, "conviction"))
        )
        top_names.append(
            {
                "ticker": tkr,
                "name": name,
                "target_pct": _to_float(a.get("target_pct")),
                "conviction": conv,
            }
        )

    return {
        **counts,
        "core_survival": core_survival,
        "core_kept": kept,
        "core_total": len(core_list),
        "top_names": top_names,
    }


def style_tilt(actions: list[dict], scored) -> dict:
    """Weight-average trailing P/E and 12-1 momentum of the TARGET book.

    Weights are the target weights (target_pct); each average is taken over the
    target names that have a finite value for that metric (weights renormalized),
    so a missing P/E on one name does not poison the whole average. An empty book
    yields NaN for both.
    """
    targets = {
        a.get("ticker"): _to_float(a.get("target_pct"))
        for a in actions
        if _to_float(a.get("target_pct")) > _EPS
    }

    def _wavg(col: str) -> float:
        num = den = 0.0
        for tkr, w in targets.items():
            v = _to_float(_lookup(scored, tkr, col))
            if math.isnan(v):
                continue
            num += w * v
            den += w
        return num / den if den > 0 else float("nan")

    return {"wavg_pe": _wavg("pe_trailing"), "wavg_mom": _wavg("mom_12_1")}


# ---------------------------------------------------------------------------
# HTML comparison (light theme, warm palette, IBM Plex mono numerals)
# ---------------------------------------------------------------------------

_CLUSTER_ROWS = [
    ("value", "Value"),
    ("quality", "Quality"),
    ("momentum", "Momentum"),
    ("growth", "Growth"),
    ("lowvol", "Low-vol"),
    ("strength", "Strength"),
]


def _pct(x, digits: int = 0) -> str:
    """Format a fraction as a percent; 'n/a' when not finite."""
    f = _to_float(x)
    if math.isnan(f):
        return "n/a"
    return f"{f * 100:.{digits}f}%"


def _num(x, digits: int = 1) -> str:
    """Format a number; 'n/a' when not finite."""
    f = _to_float(x)
    if math.isnan(f):
        return "n/a"
    return f"{f:.{digits}f}"


def _esc(s) -> str:
    return _html.escape(str(s))


def _action_class(action: str) -> str:
    a = (action or "").upper()
    if a in ("BUY", "ADD", "HOLD"):
        return "keep"
    if a in ("TRIM",):
        return "trim"
    return "cut"  # SELL / ABSENT


def _weights_block(cfg: dict) -> str:
    rows = []
    for key, label in _CLUSTER_ROWS:
        w = _to_float(cfg.get(key, 0.0))
        bar = max(0.0, min(1.0, w / 0.30)) * 100  # 30% = full bar
        emph = ' style="font-weight:600;color:var(--accent);"' if key == "growth" and w > 0 else ""
        rows.append(
            f'<div class="wrow"><span class="wlab"{emph}>{_esc(label)}</span>'
            f'<span class="wtrack"><span class="wfill" style="width:{bar:.0f}%;"></span></span>'
            f'<span class="mono wval">{w * 100:.1f}</span></div>'
        )
    return '<div class="weights">' + "".join(rows) + "</div>"


def _top_names_block(top_names: list[dict]) -> str:
    if not top_names:
        return '<div class="muted">no names</div>'
    rows = []
    for t in top_names:
        rows.append(
            f'<div class="tn"><span class="tn-tk mono">{_esc(t["ticker"])}</span>'
            f'<span class="tn-nm">{_esc(t["name"])}</span>'
            f'<span class="tn-w mono">{_pct(t["target_pct"], 1)}</span>'
            f'<span class="tn-c mono">{_num(t["conviction"], 2)}</span></div>'
        )
    return '<div class="tns">' + "".join(rows) + "</div>"


def _config_card(col: dict) -> str:
    label = col["label"]
    shape = (
        f"BUY {col['n_buy']} &middot; ADD {col['n_add']} &middot; TRIM {col['n_trim']} "
        f"&middot; SELL {col['n_sell']} &middot; HOLD {col['n_hold']}"
    )
    kept = f"{col['core_kept']} / {col['core_total']}"
    return (
        f'<div class="card">'
        f'<div class="card-hd">{_esc(label)}</div>'
        f'<div class="sub">cluster weights (%)</div>'
        f"{_weights_block(col['weights'])}"
        f'<div class="metric"><span class="mlab">Core kept</span>'
        f'<span class="mono mbig">{kept}</span></div>'
        f'<div class="metric"><span class="mlab">Book shape</span>'
        f'<span class="mshape mono">{shape}</span></div>'
        f'<div class="metric"><span class="mlab">Deployment</span>'
        f'<span class="mono">{_pct(col["deployment"], 0)}</span></div>'
        f'<div class="metric"><span class="mlab">Post-gate vol</span>'
        f'<span class="mono">{_pct(col["port_vol"], 1)}</span></div>'
        f'<div class="metric"><span class="mlab">Wt-avg P/E</span>'
        f'<span class="mono">{_num(col["wavg_pe"], 1)}</span></div>'
        f'<div class="metric"><span class="mlab">Wt-avg 12-1 mom</span>'
        f'<span class="mono">{_pct(col["wavg_mom"], 1)}</span></div>'
        f'<div class="sub">top target names</div>'
        f"{_top_names_block(col['top_names'])}"
        f"</div>"
    )


def _core_matrix(cols: list[dict]) -> str:
    """Full-width core-survival matrix: core tickers x configs (action + kept)."""
    head = '<th class="cm-tk">Core name</th>' + "".join(
        f"<th>{_esc(c['label'])}</th>" for c in cols
    )
    # index each config's survival by ticker
    per_cfg = [{r["ticker"]: r for r in c["core_survival"]} for c in cols]
    body_rows = []
    for tkr in CORE:
        cells = [f'<td class="cm-tk mono">{_esc(tkr)}</td>']
        for surv in per_cfg:
            row = surv.get(tkr, {"action": "ABSENT", "kept": False})
            cls = _action_class(row["action"])
            mark = "kept" if row["kept"] else "out"
            cells.append(
                f'<td class="cm-{cls}"><span class="pill pill--{cls}">{_esc(row["action"])}</span>'
                f'<span class="cm-mark">{mark}</span></td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    kept_row = ['<td class="cm-tk">Kept count</td>']
    for c in cols:
        kept_row.append(f'<td class="mono cm-keptcount">{c["core_kept"]} / {c["core_total"]}</td>')
    body_rows.append('<tr class="cm-total">' + "".join(kept_row) + "</tr>")
    return (
        '<table class="cmatrix"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def render_comparison_html(cols: list[dict], meta: dict) -> str:
    """Render the 3-column factor-weight comparison as a standalone HTML doc.

    Args:
        cols: per-config summary dicts (column order preserved), each merging
            summarize_book + style_tilt output with ``label``, ``weights``,
            ``deployment`` and ``port_vol``.
        meta: ``{date, nav, universe, regime}`` masthead context.

    Returns:
        A complete ``<!DOCTYPE html>`` string (light theme, warm palette, IBM
        Plex Mono numerals). Contains no em-dash characters.
    """
    date = _esc(meta.get("date", ""))
    nav = meta.get("nav")
    nav_s = f"${nav:,.0f}" if isinstance(nav, (int, float)) else "n/a"
    universe = meta.get("universe", "n/a")
    regime = _esc(meta.get("regime", "n/a"))

    cards = "".join(_config_card(c) for c in cols)
    matrix = _core_matrix(cols)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v3 Factor-Weight Sensitivity</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --canvas:#ffffff;--warm:#faf9f7;--ink:#16181d;--ink2:#4a4e57;--muted:#9aa0a8;
  --line:#e7e6e2;--accent:#123b3a;--track:#efece6;--hover:#f4f2ee;
  --keep:#2d6a4f;--keep-bg:#e9f5ee;--trim:#96601a;--trim-bg:#f7efe0;
  --cut:#b3402f;--cut-bg:#fbeeec;
  --serif:'Fraunces',Georgia,serif;--sans:'IBM Plex Sans',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--warm);color:var(--ink2);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;padding:40px 32px 64px;}}
.mono{{font-family:var(--mono);font-variant-numeric:tabular-nums;}}
.wrap{{max-width:1160px;margin:0 auto;}}
.eyebrow{{font-size:10.5px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:var(--muted);}}
h1{{font-family:var(--serif);font-weight:400;font-size:44px;line-height:1.05;letter-spacing:-0.6px;
  color:var(--ink);margin:10px 0 6px;}}
.rule{{width:70px;height:3px;background:var(--accent);margin:18px 0 14px;}}
.stand{{font-family:var(--serif);font-style:italic;font-size:18px;line-height:1.5;color:var(--ink2);
  max-width:820px;margin:0 0 10px;}}
.meta{{display:flex;flex-wrap:wrap;gap:8px 22px;padding-top:14px;border-top:1px solid var(--line);
  font-size:12px;color:var(--ink2);}}
.meta b{{color:var(--ink);font-weight:600;}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:26px 0 30px;}}
.card{{background:var(--canvas);border:1px solid var(--line);border-radius:10px;padding:18px 18px 20px;}}
.card-hd{{font-family:var(--mono);font-size:15px;font-weight:500;color:var(--accent);
  letter-spacing:0.2px;padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:12px;}}
.sub{{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--muted);
  margin:14px 0 8px;}}
.weights{{display:flex;flex-direction:column;gap:5px;}}
.wrow{{display:grid;grid-template-columns:64px 1fr 34px;align-items:center;gap:8px;font-size:12px;}}
.wlab{{color:var(--ink2);}}
.wtrack{{height:6px;background:var(--track);border-radius:3px;overflow:hidden;}}
.wfill{{display:block;height:100%;background:var(--accent);border-radius:3px;}}
.wval{{text-align:right;font-size:11.5px;color:var(--ink2);}}
.metric{{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  padding:7px 0;border-bottom:1px dotted var(--line);}}
.mlab{{font-size:12px;color:var(--ink2);}}
.mbig{{font-size:17px;font-weight:500;color:var(--ink);}}
.mshape{{font-size:11px;color:var(--ink2);text-align:right;}}
.tns{{display:flex;flex-direction:column;gap:3px;}}
.tn{{display:grid;grid-template-columns:52px 1fr 44px 40px;align-items:baseline;gap:6px;
  font-size:12px;padding:2px 0;border-bottom:1px dotted var(--line);}}
.tn-tk{{color:var(--accent);font-weight:500;}}
.tn-nm{{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.tn-w{{text-align:right;color:var(--ink);}}
.tn-c{{text-align:right;color:var(--muted);}}
.muted{{color:var(--muted);font-size:12px;}}
.section-hd{{font-family:var(--serif);font-size:24px;color:var(--ink);margin:6px 0 4px;}}
.section-sub{{font-size:13px;color:var(--ink2);margin:0 0 16px;max-width:820px;}}
.cmatrix{{width:100%;border-collapse:collapse;background:var(--canvas);border:1px solid var(--line);
  border-radius:10px;overflow:hidden;font-size:13px;}}
.cmatrix th,.cmatrix td{{padding:11px 14px;text-align:center;border-bottom:1px solid var(--line);}}
.cmatrix thead th{{background:var(--warm);font-size:10.5px;font-weight:600;letter-spacing:0.6px;
  text-transform:uppercase;color:var(--muted);}}
.cmatrix .cm-tk{{text-align:left;color:var(--ink);font-weight:500;}}
.pill{{display:inline-block;padding:2px 9px;border-radius:999px;font-family:var(--mono);
  font-size:11px;font-weight:500;}}
.pill--keep{{background:var(--keep-bg);color:var(--keep);}}
.pill--trim{{background:var(--trim-bg);color:var(--trim);}}
.pill--cut{{background:var(--cut-bg);color:var(--cut);}}
.cm-mark{{display:block;font-size:9.5px;letter-spacing:0.5px;text-transform:uppercase;
  color:var(--muted);margin-top:3px;}}
.cm-total td{{background:var(--warm);font-weight:600;color:var(--ink);border-bottom:none;}}
.cm-keptcount{{font-size:14px;}}
.foot{{margin-top:30px;font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px;}}
</style></head>
<body><div class="wrap">
<div class="eyebrow">Trading Model v3 &middot; factor-weight sensitivity</div>
<h1>Does broadening beyond value keep the core on merit?</h1>
<div class="rule"></div>
<p class="stand">One feature panel, three cluster-weight schemes. The core survives on
merit only if it stays in-target as growth and momentum weight rises and value weight falls,
not just under the value-heavy default.</p>
<div class="meta"><span>Date <b>{date}</b></span><span>NAV <b class="mono">{nav_s}</b></span>
<span>Universe <b class="mono">{universe}</b></span><span>Regime <b>{regime}</b></span></div>

<div class="grid">{cards}</div>

<div class="section-hd">Core survival</div>
<p class="section-sub">Each mega-cap growth name, and whether each weighting scheme keeps it
(BUY / ADD / HOLD = in-target) or exits it (TRIM / SELL). ABSENT = not selected into the book.</p>
{matrix}

<div class="foot">Decision-support only; not a trade instruction. Weights are evidence-style priors,
NOT fitted to the book. Post-gate vol / deployment are after the v3 risk gate.</div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Live pipeline (assemble ONCE, loop configs) — VPS run, not unit-tested
# ---------------------------------------------------------------------------


def main() -> None:
    # Pipeline imports confined to main() (keeps the module import network-free
    # and free of any yahoofinance.core.config dependency).
    import pandas as pd  # noqa: PLC0415

    from scripts.v3_full_report import (  # noqa: PLC0415
        _read_tickers,
        load_account_json,
        resolve_current_weights,
    )
    from scripts.v3_portfolio import trend_regime  # noqa: PLC0415
    from trade_modules.v3.actions import build_actions  # noqa: PLC0415
    from trade_modules.v3.combine import compute_scores  # noqa: PLC0415
    from trade_modules.v3.conditioning import resolve_deployment  # noqa: PLC0415
    from trade_modules.v3.construct import build_portfolio  # noqa: PLC0415
    from trade_modules.v3.features import enrich_features  # noqa: PLC0415
    from trade_modules.v3.fetch import robust_fetch_prices  # noqa: PLC0415

    portfolio_csv = "yahoofinance/output/portfolio.csv"
    buy_csv = "yahoofinance/output/buy.csv"
    etoro_csv = "yahoofinance/output/etoro.csv"

    # --- Universe assembly (mirrors v3_full_report) ---
    port = _read_tickers(portfolio_csv)
    buy = _read_tickers(buy_csv)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + {len(set(buy) - port_set)} candidates)"
    )

    # --- Feature enrichment ONCE (shared across all three configs) ---
    feats = enrich_features(universe, etoro_csv, price_period="2y", accruals_fetch=lambda _t: {})
    print(
        f"features: {len(feats)} rows, priced={int(feats['mom_12_1'].notna().sum())}, "
        f"growth_earn={int(feats['earn_growth'].notna().sum())}, "
        f"growth_rev={int(feats.get('rev_growth', pd.Series(dtype=float)).notna().sum())}"
    )

    # --- Prices ONCE + market regime ---
    prices: pd.DataFrame = pd.DataFrame()
    spx_close: pd.Series = pd.Series(dtype=float)
    try:
        prices = robust_fetch_prices(universe, period="2y")
    except Exception as exc:  # noqa: BLE001
        print(f"warn: universe price fetch failed ({exc})", file=sys.stderr)
    try:
        spx_raw = robust_fetch_prices(["^GSPC"], period="2y")
        if spx_raw is not None and not spx_raw.empty:
            spx_close = spx_raw.iloc[:, 0]
    except Exception as exc:  # noqa: BLE001
        print(f"warn: ^GSPC fetch failed ({exc}); defaulting to neutral", file=sys.stderr)

    regime, _mult = trend_regime(spx_close)
    gross_target, _cond = resolve_deployment(regime, polymarket_signal=None)
    print(f"regime: {regime}  deployment: {gross_target:.0%}")

    # --- Live account (current weights + NAV) ONCE ---
    account_weights, nav, present = load_account_json()
    current_weights, approx = resolve_current_weights(port, account_weights, present)
    src = "live account" if (present and not approx) else "equal-split fallback"
    print(f"current weights: {src} ({len(current_weights)} names, nav={nav})")

    # --- Score / construct / diff per config ---
    cols: list[dict] = []
    for label, cfg in CONFIGS.items():
        scores = compute_scores(feats, sector_neutral=True, cluster_weights=cfg)
        scores["is_portfolio"] = scores.index.isin(port_set)
        result = build_portfolio(
            scores,
            prices,
            top_n=20,
            target_vol=0.12,
            name_cap=0.08,
            sector_cap=0.25,
            usd_bloc_cap=0.60,
            gross_target=gross_target,
        )
        actions = build_actions(result["weights"], current_weights, scores, nav=nav)
        col = summarize_book(actions, scores, CORE, top_n=12)
        col.update(style_tilt(actions, scores))
        col["label"] = label
        col["weights"] = cfg
        col["deployment"] = float(result.get("gross", 0.0))
        col["port_vol"] = float(
            result.get("diagnostics", {}).get("gate", {}).get("vol_after", float("nan"))
        )
        cols.append(col)

    # --- Console comparison ---
    print("\n=== v3 factor-weight sensitivity ===")
    hdr = f"{'metric':<18}" + "".join(f"{c['label']:>16}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("core kept", lambda c: f"{c['core_kept']}/{c['core_total']}"),
        ("n_buy", lambda c: c["n_buy"]),
        ("n_add", lambda c: c["n_add"]),
        ("n_trim", lambda c: c["n_trim"]),
        ("n_sell", lambda c: c["n_sell"]),
        ("n_hold", lambda c: c["n_hold"]),
        ("deployment", lambda c: _pct(c["deployment"], 0)),
        ("post-gate vol", lambda c: _pct(c["port_vol"], 1)),
        ("wt-avg P/E", lambda c: _num(c["wavg_pe"], 1)),
        ("wt-avg 12-1 mom", lambda c: _pct(c["wavg_mom"], 1)),
    ]
    for name, fn in rows:
        print(f"{name:<18}" + "".join(f"{fn(c)!s:>16}" for c in cols))
    print("\ncore survival:")
    for tkr in CORE:
        marks = []
        for c in cols:
            surv = {r["ticker"]: r for r in c["core_survival"]}.get(tkr, {"action": "ABSENT"})
            marks.append(surv["action"])
        print(f"  {tkr:<6}" + "".join(f"{m:>16}" for m in marks))

    # --- Render + write HTML ---
    now = datetime.now(timezone.utc)
    meta = {
        "date": now.strftime("%Y-%m-%d"),
        "nav": nav,
        "universe": len(universe),
        "regime": str(regime).upper(),
    }
    html = render_comparison_html(cols, meta)
    stamp = now.strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_sensitivity.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nsensitivity comparison -> {out}")


if __name__ == "__main__":
    main()

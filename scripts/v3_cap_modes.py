# scripts/v3_cap_modes.py
"""Trading Model v3 — three cap-scaled vol modes, compared on the live book.

Runs the pipeline ONCE (universe -> features -> compute_scores(balanced) ->
load account), then calls ``build_overlay`` under each of the three cap-scaling
risk modes with identical parameters:

  - cap_budget   : per-name risk-contribution allowance proportional to log(cap);
                   no hard vol ceiling (vol floats). Large caps kept, small
                   high-vol names trimmed.
  - cap_exempt   : sigmoid-graded exemption by log(cap); mega caps ~fully held,
                   small caps vol-managed to ``managed_vol_ceiling``.
  - cap_ordered  : hard vol ceiling, but trims smallest-cap first (mega caps
                   sticky until last).

Per mode collected: portfolio vol (post-gate), n_sell, n_buy, turnover,
deployment (gross), each mega-cap core name's weight + their SUM, and the
small-cap bucket weight (market cap < $10B). A "current" reference column shows
the live account.

Output:
  - Console comparison table (rows = core names + CORE TOTAL + small-cap bucket +
    summary stats; columns = current + the three modes)
  - Compact light-theme HTML -> ~/Downloads/<UTCstamp>_v3_cap_modes.html

Run on the VPS (network allowed):  .venv/bin/python scripts/v3_cap_modes.py
Import-clean check (no network):   .venv/bin/python -c "import scripts.v3_cap_modes"

No module-level ``yahoofinance.core.config`` import.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.v3_full_report import (  # noqa: E402
    _read_tickers,
    load_account_json,
    resolve_current_weights,
)
from scripts.v3_portfolio import trend_regime  # noqa: E402
from trade_modules.v3.combine import CLUSTER_WEIGHTS, compute_scores  # noqa: E402
from trade_modules.v3.conditioning import resolve_deployment  # noqa: E402
from trade_modules.v3.features import enrich_features  # noqa: E402
from trade_modules.v3.fetch import robust_fetch_prices  # noqa: E402
from trade_modules.v3.overlay import build_overlay  # noqa: E402
from trade_modules.v3.report import compute_regime  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"

# Derived from the engine SSOT (combine.METRIC_WEIGHTS -> CLUSTER_WEIGHTS) so this analysis
# tool never drifts from the live model (owner taxonomy 2026-07-23).
BALANCED_WEIGHTS: dict[str, float] = {c[:-2]: w for c, w in CLUSTER_WEIGHTS.items()}

MEGA_CORE: list[str] = ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "AVGO", "TSM", "META"]

# The three cap-scaling modes, ordered for display, with human labels.
MODES: list[str] = ["cap_budget", "cap_exempt", "cap_ordered"]
MODE_LABELS: dict[str, str] = {
    "cap_budget": "risk-budget",
    "cap_exempt": "graded-exempt",
    "cap_ordered": "cap-ordered",
}

# Risk-gate parameters shared across modes (mirror overlay_report.py).
_NAME_CAP = 0.08
_SECTOR_CAP = 0.25
_USD_BLOC_CAP = 0.60
# cap_ordered uses a hard ceiling; cap_exempt manages only the small-cap sleeve.
_ORDERED_CEILING = 0.25
_MANAGED_CEILING = 0.18
_SMALLCAP_THRESHOLD = 10e9  # $10B


# ---------------------------------------------------------------------------
# Per-mode result collection
# ---------------------------------------------------------------------------


def _smallcap_weight(weights: pd.Series, scores: pd.DataFrame) -> float:
    """Sum of held weight in names with market cap < $10B."""
    if "cap" not in scores.columns:
        return float("nan")
    total = 0.0
    for tkr, w in weights.items():
        if w <= 1e-12 or tkr not in scores.index:
            continue
        cap = scores.loc[tkr, "cap"]
        if pd.notna(cap) and float(cap) < _SMALLCAP_THRESHOLD:
            total += float(w)
    return total


def _run_mode(
    mode: str,
    scores: pd.DataFrame,
    current_weights: pd.Series,
    prices: pd.DataFrame,
    gross_target: float,
) -> dict:
    """Call build_overlay under one cap_mode and return a result dict."""
    overlay = build_overlay(
        scores,
        current_weights,
        prices,
        gross_target=gross_target,
        name_cap=_NAME_CAP,
        sector_cap=_SECTOR_CAP,
        usd_bloc_cap=_USD_BLOC_CAP,
        vol_ceiling=_ORDERED_CEILING,
        managed_vol_ceiling=_MANAGED_CEILING,
        core_list=MEGA_CORE,
        cap_mode=mode,
    )
    weights = overlay["weights"]
    diag = overlay["diagnostics"]
    gate = diag.get("gate") or {}

    deployment = float(weights.sum()) if len(weights) else 0.0
    core_w = {t: (float(weights[t]) if t in weights.index else 0.0) for t in MEGA_CORE}

    return {
        "mode": mode,
        "vol_after": gate.get("vol_after"),
        "n_sell": diag["n_sell"],
        "n_buy": diag["n_buy"],
        "turnover": diag["turnover"],
        "deployment": deployment,
        "core_weights": core_w,
        "core_total": sum(core_w.values()),
        "smallcap": _smallcap_weight(weights, scores),
    }


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def print_console_table(
    results: list[dict], current_weights: pd.Series, scores: pd.DataFrame
) -> None:
    col_w = 12
    row_label_w = 16

    def _cell(v, is_pct: bool = True) -> str:
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return "n/a".rjust(col_w)
        return (f"{v * 100:.1f}%" if is_pct else str(v)).rjust(col_w)

    labels = [MODE_LABELS[r["mode"]] for r in results]
    header_cells = ["current".rjust(col_w)] + [lbl.rjust(col_w) for lbl in labels]
    sep = "-" * (row_label_w + (col_w + 1) * (1 + len(results)))

    print()
    print("Cap-Scaled Vol Modes — Impact on the Live Book")
    print(sep)
    print(f"{'':>{row_label_w}}", "  ".join(header_cells))
    print(sep)

    for t in MEGA_CORE:
        cur = float(current_weights[t]) if t in current_weights.index else 0.0
        cells = [_cell(cur)] + [_cell(r["core_weights"][t]) for r in results]
        print(f"{t:>{row_label_w}}", "  ".join(cells))

    cur_core = sum(
        float(current_weights[t]) if t in current_weights.index else 0.0 for t in MEGA_CORE
    )
    print(sep[:8] + "-" * (len(sep) - 8))
    print(
        f"{'CORE TOTAL':>{row_label_w}}",
        "  ".join([_cell(cur_core)] + [_cell(r["core_total"]) for r in results]),
    )
    cur_small = _smallcap_weight(current_weights, scores)
    print(
        f"{'SMALL-CAP <$10B':>{row_label_w}}",
        "  ".join([_cell(cur_small)] + [_cell(r["smallcap"]) for r in results]),
    )
    print(sep)

    def _stat(label: str, key: str, is_pct: bool = True) -> None:
        cells = ["n/a".rjust(col_w)] + [_cell(r[key], is_pct) for r in results]
        print(f"{label:>{row_label_w}}", "  ".join(cells))

    _stat("portfolio vol", "vol_after")
    _stat("n_sell", "n_sell", is_pct=False)
    _stat("n_buy", "n_buy", is_pct=False)
    _stat("turnover", "turnover")
    _stat("deployment", "deployment")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------


_HTML_CSS = """
<style>
  :root {
    --bg: #fafaf8; --surface: #ffffff; --border: #e2e0da; --text: #1a1917;
    --muted: #6b6860; --accent: #b45309; --buy: #166534; --sell: #991b1b;
    --mono: 'IBM Plex Mono', 'Menlo', monospace;
    --sans: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; padding: 32px 40px; }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  table { border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  th, td { padding: 9px 14px; text-align: right; font-family: var(--mono); font-size: 13px; border-bottom: 1px solid var(--border); }
  th { background: #f4f2ee; font-weight: 600; color: var(--text); font-family: var(--sans); }
  th.name-col, td.name-col { text-align: left; font-family: var(--sans); }
  tr:last-child td { border-bottom: none; }
  tr.divider td { border-top: 2px solid var(--border); font-weight: 600; background: #f9f8f5; }
  tr.stat-row td { color: var(--muted); }
  tr.stat-row td.name-col { color: var(--text); }
  tr.stat-first td { border-top: 2px solid var(--border); }
  .low { color: var(--sell); }
  .high { color: var(--buy); }
  .current { color: var(--accent); }
  .footer { margin-top: 16px; font-size: 12px; color: var(--muted); }
</style>
"""


def _pct_cell(v, cls: str = "") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "<td>n/a</td>"
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{v * 100:.1f}%</td>"


def build_html(
    results: list[dict], current_weights: pd.Series, scores: pd.DataFrame, generated_utc: str
) -> str:
    labels = [MODE_LABELS[r["mode"]] for r in results]
    header_ths = "<th class='name-col'>Name</th><th class='current'>Current</th>"
    for lbl in labels:
        header_ths += f"<th>{lbl}</th>"

    rows_html = ""
    for t in MEGA_CORE:
        cur = float(current_weights[t]) if t in current_weights.index else 0.0
        cells = f'<td class="current">{cur * 100:.1f}%</td>'
        for r in results:
            w = r["core_weights"][t]
            cells += _pct_cell(w, "low" if (w < 0.02 and cur >= 0.02) else "")
        rows_html += f'<tr><td class="name-col">{t}</td>{cells}</tr>\n'

    cur_core = sum(
        float(current_weights[t]) if t in current_weights.index else 0.0 for t in MEGA_CORE
    )
    tc = f'<td class="current"><strong>{cur_core * 100:.1f}%</strong></td>'
    for r in results:
        tc += f"<td><strong>{r['core_total'] * 100:.1f}%</strong></td>"
    rows_html += f'<tr class="divider"><td class="name-col">CORE TOTAL</td>{tc}</tr>\n'

    cur_small = _smallcap_weight(current_weights, scores)
    sc = _pct_cell(cur_small, "current")
    for r in results:
        sc += _pct_cell(r["smallcap"])
    rows_html += f'<tr class="stat-row"><td class="name-col">Small-cap &lt;$10B</td>{sc}</tr>\n'

    def _stat(label: str, key: str, first: bool = False, is_pct: bool = True) -> str:
        cls = "stat-row stat-first" if first else "stat-row"
        cells = "<td>n/a</td>"
        for r in results:
            if is_pct:
                cells += _pct_cell(r[key])
            else:
                cells += f"<td>{r[key]}</td>"
        return f'<tr class="{cls}"><td class="name-col">{label}</td>{cells}</tr>\n'

    rows_html += _stat("Portfolio vol", "vol_after", first=True)
    rows_html += _stat("n_sell", "n_sell", is_pct=False)
    rows_html += _stat("n_buy", "n_buy", is_pct=False)
    rows_html += _stat("Turnover", "turnover")
    rows_html += _stat("Deployment", "deployment")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v3 Cap-Scaled Vol Modes</title>
{_HTML_CSS}
</head>
<body>
<h1>Cap-Scaled Vol Modes &mdash; Impact on the Live Book</h1>
<p class="subtitle">Generated {generated_utc} UTC &nbsp;&middot;&nbsp;
Balanced cluster weights &nbsp;&middot;&nbsp;
Modes: {" / ".join(labels)}</p>
<table>
  <thead><tr>{header_ths}</tr></thead>
  <tbody>
{rows_html}
  </tbody>
</table>
<p class="footer">
  cap_budget: allowance proportional to log(cap), vol floats.
  cap_exempt: sigmoid-graded exemption by cap; small-cap sleeve managed to {_MANAGED_CEILING:.0%}.
  cap_ordered: hard {_ORDERED_CEILING:.0%} vol ceiling, trims smallest-cap first.<br>
  Weights shown as % of NAV. "Current" (amber) = live account; "low" (red) = core name trimmed below 2%.
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + "
        f"{len(set(buy) - port_set)} candidates)"
    )

    feats = enrich_features(
        universe, ETORO_CSV, price_period="2y", accruals_fetch=lambda _tickers: {}
    )
    scores = compute_scores(feats, sector_neutral=True, cluster_weights=BALANCED_WEIGHTS)
    scores["is_portfolio"] = scores.index.isin(port_set)

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
    regime_label, _ = compute_regime(spx_close)
    print(f"regime: {regime_label}  gross_target: {gross_target:.0%}")

    account_weights, nav, present = load_account_json()
    current_weights, _approx = resolve_current_weights(port, account_weights, present)
    if present:
        print(f"current book: live account ({len(current_weights)} names, nav={nav})")
    else:
        print(f"current book: equal-split fallback ({len(current_weights)} names)")

    results: list[dict] = []
    for mode in MODES:
        print(f"running mode={mode} ...", end=" ", flush=True)
        r = _run_mode(mode, scores, current_weights, prices, gross_target)
        vol_str = f"{r['vol_after'] * 100:.1f}%" if r["vol_after"] is not None else "n/a"
        print(
            f"vol={vol_str}  sell={r['n_sell']}  buy={r['n_buy']}  "
            f"core_total={r['core_total'] * 100:.1f}%  turnover={r['turnover'] * 100:.1f}%"
        )
        results.append(r)

    print_console_table(results, current_weights, scores)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M")
    html = build_html(results, current_weights, scores, now.strftime("%Y-%m-%d %H:%M"))
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_cap_modes.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"cap-modes report -> {out}")


if __name__ == "__main__":
    main()

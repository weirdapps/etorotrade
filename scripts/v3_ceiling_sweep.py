# scripts/v3_ceiling_sweep.py
"""Trading Model v3 — vol-ceiling sweep for overlay (core weight vs vol trade-off).

Runs the pipeline ONCE (universe -> features -> compute_scores(balanced) ->
load account), then calls ``build_overlay`` for four ``vol_ceiling`` values
(0.18, 0.25, 0.35, 0.50) with identical conviction + core_list parameters.

Per ceiling collected: portfolio vol (post-gate), n_sell, n_buy, turnover,
deployment (gross weight), and the resulting weight of each mega-cap core name
plus their SUM.  The current account weight of each core name is shown in a
"current" reference column.

Output:
  - Console comparison table (rows = core names + CORE TOTAL + divider + summary
    stats; columns = current + 18% / 25% / 35% / 50%)
  - Compact light-theme HTML -> ~/Downloads/<UTCstamp>_v3_ceiling_sweep.html

Run on the VPS (network allowed):  .venv/bin/python scripts/v3_ceiling_sweep.py
Import-clean check (no network):   .venv/bin/python -c "import scripts.v3_ceiling_sweep"

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

# Ceilings to sweep (ordered low -> high). Override: V3_CEILINGS="0.25,0.35,0.5,0.7".
CEILINGS: list[float] = [
    float(x) for x in os.environ.get("V3_CEILINGS", "0.18,0.25,0.35,0.50").split(",")
]
# Optional cap-scaling mode applied at every ceiling (e.g. V3_CAP_MODE=cap_ordered).
CAP_MODE: str | None = os.environ.get("V3_CAP_MODE") or None

# Risk-gate parameters shared across all ceiling runs (mirrors overlay_report.py).
# Concentration caps are tunable via env (relaxing them lets the core run higher).
_NAME_CAP = float(os.environ.get("V3_NAME_CAP", "0.08"))
_SECTOR_CAP = float(os.environ.get("V3_SECTOR_CAP", "0.25"))
_USD_BLOC_CAP = float(os.environ.get("V3_USD_BLOC_CAP", "0.60"))


# ---------------------------------------------------------------------------
# Per-ceiling result collection
# ---------------------------------------------------------------------------


def _sweep_one(
    ceiling: float,
    scores: pd.DataFrame,
    current_weights: pd.Series,
    prices: pd.DataFrame,
    gross_target: float,
) -> dict:
    """Call build_overlay for a single ceiling and return a result dict."""
    overlay = build_overlay(
        scores,
        current_weights,
        prices,
        gross_target=gross_target,
        name_cap=_NAME_CAP,
        sector_cap=_SECTOR_CAP,
        usd_bloc_cap=_USD_BLOC_CAP,
        vol_ceiling=ceiling,
        core_list=MEGA_CORE,
        cap_mode=CAP_MODE,
    )
    weights = overlay["weights"]
    diag = overlay["diagnostics"]
    gate = diag.get("gate") or {}

    deployment = float(weights.sum()) if len(weights) else 0.0
    vol_after = gate.get("vol_after")

    core_w: dict[str, float] = {}
    for t in MEGA_CORE:
        w = float(weights[t]) if t in weights.index else 0.0
        core_w[t] = w

    return {
        "ceiling": ceiling,
        "vol_after": vol_after,
        "n_sell": diag["n_sell"],
        "n_buy": diag["n_buy"],
        "turnover": diag["turnover"],
        "deployment": deployment,
        "core_weights": core_w,
        "core_total": sum(core_w.values()),
    }


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def _fmt_pct(v, decimals: int = 1) -> str:
    if v is None:
        return "  n/a "
    return f"{v * 100:6.{decimals}f}%"


def _fmt_pct_narrow(v) -> str:
    """5-char field: '12.3%' or '  n/a'."""
    if v is None:
        return "  n/a"
    return f"{v * 100:5.1f}%"


def print_console_table(
    results: list[dict],
    current_weights: pd.Series,
) -> None:
    col_w = 10  # width of each data column

    def _cell(v, is_pct: bool = True) -> str:
        if v is None:
            return "n/a".rjust(col_w)
        if is_pct:
            return f"{v * 100:.1f}%".rjust(col_w)
        return str(v).rjust(col_w)

    ceilings_labels = [f"{int(r['ceiling'] * 100)}%" for r in results]
    header_cells = ["current".rjust(col_w)] + [lbl.rjust(col_w) for lbl in ceilings_labels]
    row_label_w = 16

    sep = "-" * (row_label_w + (col_w + 1) * (1 + len(results)))

    print()
    print("Vol-Ceiling Sweep — Core Weight vs Vol Trade-Off")
    print(sep)
    print(f"{'':>{row_label_w}}", "  ".join(header_cells))
    print(sep)

    # Core names
    for t in MEGA_CORE:
        cur_w = float(current_weights[t]) if t in current_weights.index else 0.0
        cells = [_cell(cur_w)] + [_cell(r["core_weights"][t]) for r in results]
        print(f"{t:>{row_label_w}}", "  ".join(cells))

    # Core total
    cur_core_total = sum(
        float(current_weights[t]) if t in current_weights.index else 0.0 for t in MEGA_CORE
    )
    cells = [_cell(cur_core_total)] + [_cell(r["core_total"]) for r in results]
    print(sep[:8] + "-" * (len(sep) - 8))
    print(f"{'CORE TOTAL':>{row_label_w}}", "  ".join(cells))
    print(sep)

    # Summary stats (no "current" value -> n/a)
    def _stat_row(label: str, key: str, is_pct: bool = True) -> None:
        cells = ["n/a".rjust(col_w)] + [_cell(r[key], is_pct) for r in results]
        print(f"{label:>{row_label_w}}", "  ".join(cells))

    def _stat_int_row(label: str, key: str) -> None:
        cells = ["n/a".rjust(col_w)] + [str(r[key]).rjust(col_w) for r in results]
        print(f"{label:>{row_label_w}}", "  ".join(cells))

    _stat_row("portfolio vol", "vol_after")
    _stat_int_row("n_sell", "n_sell")
    _stat_int_row("n_buy", "n_buy")
    _stat_row("turnover", "turnover")
    _stat_row("deployment", "deployment")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------


_HTML_CSS = """
<style>
  :root {
    --bg: #fafaf8;
    --surface: #ffffff;
    --border: #e2e0da;
    --text: #1a1917;
    --muted: #6b6860;
    --accent: #b45309;
    --buy: #166534;
    --sell: #991b1b;
    --mono: 'IBM Plex Mono', 'Menlo', monospace;
    --sans: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    padding: 32px 40px;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  table {
    border-collapse: collapse;
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  th, td {
    padding: 9px 14px;
    text-align: right;
    font-family: var(--mono);
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
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


def _pct_html(v, decimals: int = 1, low_thr: float = 0.0, high_thr: float = 1.1) -> str:
    if v is None:
        return "<td>n/a</td>"
    s = f"{v * 100:.{decimals}f}%"
    if v <= low_thr:
        return f'<td class="low">{s}</td>'
    if v >= high_thr:
        return f'<td class="high">{s}</td>'
    return f"<td>{s}</td>"


def _int_html(v) -> str:
    if v is None:
        return "<td>n/a</td>"
    return f"<td>{v}</td>"


def _na_cell() -> str:
    return '<td class="muted">—</td>'


def build_html(
    results: list[dict],
    current_weights: pd.Series,
    generated_utc: str,
) -> str:
    ceilings_labels = [f"{int(r['ceiling'] * 100)}%" for r in results]

    # Column headers
    header_ths = "<th>Name</th><th class='current'>Current</th>"
    for lbl in ceilings_labels:
        header_ths += f"<th>{lbl} ceiling</th>"

    rows_html = ""

    # Core name rows
    for t in MEGA_CORE:
        cur_w = float(current_weights[t]) if t in current_weights.index else 0.0
        cells = f'<td class="current">{cur_w * 100:.1f}%</td>'
        for r in results:
            w = r["core_weights"][t]
            # Flag a name trimmed to <2% at any ceiling
            if w < 0.02 and cur_w >= 0.02:
                cells += f'<td class="low">{w * 100:.1f}%</td>'
            else:
                cells += f"<td>{w * 100:.1f}%</td>"
        rows_html += f'<tr><td class="name-col">{t}</td>{cells}</tr>\n'

    # Core total divider row
    cur_total = sum(
        float(current_weights[t]) if t in current_weights.index else 0.0 for t in MEGA_CORE
    )
    total_cells = f'<td class="current"><strong>{cur_total * 100:.1f}%</strong></td>'
    for r in results:
        total_cells += f"<td><strong>{r['core_total'] * 100:.1f}%</strong></td>"
    rows_html += f'<tr class="divider"><td class="name-col">CORE TOTAL</td>{total_cells}</tr>\n'

    # Summary stat rows
    def _stat_row(label: str, cells_html: str, first: bool = False) -> str:
        cls = 'class="stat-row stat-first"' if first else 'class="stat-row"'
        return f'<tr {cls}><td class="name-col">{label}</td>{cells_html}</tr>\n'

    # portfolio vol
    vol_cells = _na_cell()
    for r in results:
        vol_cells += _pct_html(r["vol_after"], low_thr=0.0, high_thr=0.30)
    rows_html += _stat_row("Portfolio vol", vol_cells, first=True)

    # n_sell
    sell_cells = _na_cell()
    for r in results:
        sell_cells += _int_html(r["n_sell"])
    rows_html += _stat_row("n_sell", sell_cells)

    # n_buy
    buy_cells = _na_cell()
    for r in results:
        buy_cells += _int_html(r["n_buy"])
    rows_html += _stat_row("n_buy", buy_cells)

    # turnover
    turn_cells = _na_cell()
    for r in results:
        turn_cells += _pct_html(r["turnover"])
    rows_html += _stat_row("Turnover", turn_cells)

    # deployment
    dep_cells = _na_cell()
    for r in results:
        dep_cells += _pct_html(r["deployment"])
    rows_html += _stat_row("Deployment", dep_cells)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v3 Vol-Ceiling Sweep</title>
{_HTML_CSS}
</head>
<body>
<h1>Vol-Ceiling Sweep — Core Weight vs Vol Trade-Off</h1>
<p class="subtitle">Generated {generated_utc} UTC &nbsp;&middot;&nbsp;
Balanced cluster weights &nbsp;&middot;&nbsp;
Ceilings: {" / ".join(ceilings_labels)}</p>
<table>
  <thead><tr>{header_ths}</tr></thead>
  <tbody>
{rows_html}
  </tbody>
</table>
<p class="footer">
  Pipeline: universe &rarr; enrich_features &rarr; compute_scores(balanced)
  &rarr; build_overlay &times; {len(results)} ceilings.<br>
  "Low" (red) = core name trimmed below current weight; "Current" (amber) = live account.
  Weights shown as % of NAV.
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Universe assembly (mirrors v3_overlay_report.py) ---
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + "
        f"{len(set(buy) - port_set)} candidates)"
    )

    # --- Feature enrichment + scoring (BALANCED weights, single run) ---
    feats = enrich_features(
        universe, ETORO_CSV, price_period="2y", accruals_fetch=lambda _tickers: {}
    )
    scores = compute_scores(feats, sector_neutral=True, cluster_weights=BALANCED_WEIGHTS)
    scores["is_portfolio"] = scores.index.isin(port_set)

    # --- Prices + regime ---
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

    # --- Current book (live account) ---
    account_weights, nav, present = load_account_json()
    current_weights, approx = resolve_current_weights(port, account_weights, present)
    if present:
        print(f"current book: live account ({len(current_weights)} names, nav={nav})")
    else:
        print(f"current book: equal-split fallback ({len(current_weights)} names)")

    # --- Ceiling sweep ---
    results: list[dict] = []
    for ceiling in CEILINGS:
        print(f"sweeping ceiling={ceiling:.0%} ...", end=" ", flush=True)
        r = _sweep_one(ceiling, scores, current_weights, prices, gross_target)
        vol_str = f"{r['vol_after'] * 100:.1f}%" if r["vol_after"] is not None else "n/a"
        print(
            f"vol={vol_str}  sell={r['n_sell']}  buy={r['n_buy']}  "
            f"core_total={r['core_total'] * 100:.1f}%"
        )
        results.append(r)

    # --- Console table ---
    print_console_table(results, current_weights)

    # --- HTML output ---
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M")
    generated_utc = now.strftime("%Y-%m-%d %H:%M")
    html = build_html(results, current_weights, generated_utc)
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_ceiling_sweep.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"sweep report -> {out}")


if __name__ == "__main__":
    main()

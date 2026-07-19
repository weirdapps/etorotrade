"""Phase 3 — multi-year backtest of the PIT fundamental factors (the real unlock).

Uses the point-in-time Sharadar fundamentals store (datekey-lagged, no look-ahead)
+ a long EUR price history to measure the forward IC of the data-blocked durable
premia — book_to_price, asset_growth (CMA), gp_assets (Novy-Marx), accruals (Sloan)
— over ~10-20 years / multiple regimes, instead of the 5-month etoro.csv panel. This
is the sample the DSR / multi-regime gate actually needs.

Monthly cross-sections: each month-end T, read fundamentals as of T (and ~T-1yr for
asset growth), form each factor's directional sector-demeaned rank-z, and correlate
with the forward 1-month return (raw + beta-neutral). Reports IC raw -> beta-neutral
with HAC (Newey-West) t-stats, a Fama-MacBeth simultaneous panel, and per-sector IC.

    .venv/bin/python scripts/v3_fundamentals_backtest.py [--period 10y] [--max-tickers N]

Reuses the tested helpers from factor_backtest / fundamentals / combine. Prices are
yfinance (survivors only -> a survivorship caveat, flagged); delisted-name pricing is
the documented next data step (Sharadar SEP).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.combine import _rank_z, _sector_demean  # noqa: E402
from trade_modules.v3.factor_backtest import (  # noqa: E402
    beta_neutralize,
    fama_macbeth,
    hac_tstat,
)
from trade_modules.v3.fundamentals import factor_panel  # noqa: E402
from trade_modules.v3.fundamentals_store import STORE_PATH, read_asof  # noqa: E402
from trade_modules.validation.xsection_ic import cross_sectional_ic  # noqa: E402

# Factor -> directional sign (+1 high-good / -1 low-good), matching the literature.
FACTORS = {"book_to_price": +1, "asset_growth": -1, "gp_assets": +1, "accruals": -1}
HAC_LAGS = 12  # ~1yr of monthly autocorrelation for the overlap-robust t-stat
BETA_WIN = 36  # trailing months for the rolling market beta


def _store_tickers() -> list[str]:
    """Store tickers ranked by latest market cap (so --max-tickers takes the largest)."""
    df = pd.read_parquet(STORE_PATH)
    latest_mc = df.sort_values("datekey").groupby("ticker")["marketcap"].last()
    return [str(t) for t in latest_mc.sort_values(ascending=False).index]


def _rolling_beta(mret: pd.DataFrame) -> pd.DataFrame:
    """Trailing-``BETA_WIN`` month beta of each name vs the equal-weight market."""
    mkt = mret.mean(axis=1)
    cov = mret.rolling(BETA_WIN).cov(mkt)
    var = mkt.rolling(BETA_WIN).var()
    return cov.div(var, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-year PIT fundamentals backtest.")
    ap.add_argument("--period", default="10y", help="yfinance price history period (e.g. 10y, max)")
    ap.add_argument("--horizon-months", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=0, help="cap universe (0 = all)")
    args = ap.parse_args()

    from trade_modules.v3.prices import load_eur_close
    from trade_modules.v3.sectors import load_offline_sector_map

    tickers = _store_tickers()
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"store tickers: {len(tickers)}")
    px = load_eur_close(tickers, period=args.period)
    if px.empty:
        print("no prices — is the store populated / network up?")
        return
    monthly = px.resample("ME").last()
    fwd_mat = monthly.shift(-args.horizon_months) / monthly - 1.0
    mret = monthly.pct_change(fill_method=None)
    beta = _rolling_beta(mret)
    sector_map = load_offline_sector_map()
    print(f"priced (EUR): {monthly.shape[1]} names, {monthly.shape[0]} months")

    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in FACTORS}
    fm_parts: list[pd.DataFrame] = []
    dates = monthly.index[BETA_WIN:]  # need the beta window warmed up
    n_used = 0
    for T in dates:
        fwd = fwd_mat.loc[T].dropna()
        if fwd.empty:
            continue
        tkey = T.strftime("%Y-%m-%d")
        pkey = (T - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        fasof = read_asof(tickers, tkey)
        if fasof.empty:
            continue
        fp = factor_panel(fasof, read_asof(tickers, pkey), monthly.loc[T])
        n_used += 1
        sec = pd.Series({t: sector_map.get(str(t).upper()) for t in fp.index}, index=fp.index)
        beta_T = beta.loc[T] if T in beta.index else None
        fwd_n = beta_neutralize(fwd, beta_T) if beta_T is not None else fwd

        fm_row = {}
        for fname, sign in FACTORS.items():
            z = _sector_demean(_rank_z(fp[fname]) * sign, sec.reindex(fp.index))
            part = pd.DataFrame(
                {"date": tkey, "z": z, "fwd": fwd.reindex(z.index), "fwd_n": fwd_n.reindex(z.index)}
            ).dropna(subset=["z", "fwd"])
            if len(part):
                parts[fname].append(part.reset_index(names="ticker"))
            fm_row[fname] = z
        fmdf = pd.DataFrame(fm_row)
        fmdf["fwd"] = fwd.reindex(fmdf.index)
        fmdf["date"] = tkey
        fm_parts.append(fmdf.reset_index(names="ticker"))

    print(f"monthly cross-sections used: {n_used}  ({dates[0].date()} .. {dates[-1].date()})")

    rows = []
    for fname in FACTORS:
        if not parts[fname]:
            continue
        d = pd.concat(parts[fname], ignore_index=True)
        raw = cross_sectional_ic(d, "z", "fwd", date_col="date")
        neu = cross_sectional_ic(d.dropna(subset=["fwd_n"]), "z", "fwd_n", date_col="date")
        neu_s = pd.Series(neu.get("ic_by_date") or {}, dtype=float).sort_index()
        rows.append(
            {
                "name": fname,
                "ic_raw": raw.get("mean_ic"),
                "ic_neu": neu.get("mean_ic"),
                "t_hac": hac_tstat(neu_s, HAC_LAGS) if len(neu_s) > 1 else float("nan"),
                "hit": neu.get("hit_rate"),
                "n_dates": neu.get("n_dates"),
                "n_obs": len(d),
            }
        )
    rows.sort(key=lambda r: r["ic_neu"] if isinstance(r["ic_neu"], float) else -9)

    fm = {}
    if fm_parts:
        fmp = pd.concat(fm_parts, ignore_index=True)
        fm = fama_macbeth(fmp, "fwd", list(FACTORS), date_col="date", lags=HAC_LAGS)

    _print(rows, fm, n_used)
    out = _render(rows, fm, n_used, dates, monthly.shape[1], args.period)
    print(f"\nreport -> {out}")


def _f(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


def _print(rows, fm, n_used) -> None:
    print(f"\n=== PIT FUNDAMENTALS forward IC (raw -> beta-neutral, HAC t; {n_used} months) ===")
    print(f"{'factor':16} {'IC_raw':>8} {'IC_neu':>8} {'tHAC':>7} {'hit':>5} {'dates':>5}")
    for r in rows:
        print(
            f"{r['name']:16} {_f(r['ic_raw']):>8} {_f(r['ic_neu']):>8} "
            f"{_f(r['t_hac'], 2):>7} {_f(r['hit'], 2):>5} {r['n_dates'] or 0:>5}"
        )
    if fm:
        print("\n--- Fama-MacBeth (simultaneous, HAC t) ---")
        for k, v in fm.items():
            print(f"{k:16} coef {_f(v['coef']):>9}  tHAC {_f(v['t'], 2):>6}  dates {v['n_dates']}")


def _render(rows, fm, n_used, dates, n_px, period) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_fundamentals_backtest.html")

    def frow(r):
        good = (r["ic_neu"] or 0) > 0
        strong = abs(r.get("t_hac") or 0) >= 2
        col = "#0a7d33" if good else "#b91c1c"
        tw = "700" if strong else "400"
        return (
            f"<tr><td><code>{r['name']}</code></td>"
            f"<td>{_f(r['ic_raw'])}</td>"
            f"<td style='color:{col};font-weight:{tw}'>{_f(r['ic_neu'])}</td>"
            f"<td style='font-weight:{tw}'>{_f(r['t_hac'], 2)}</td>"
            f"<td>{_f(r['hit'], 2)}</td><td>{r['n_dates'] or 0}</td><td>{r['n_obs']:,}</td></tr>"
        )

    fmrows = ""
    for k, v in (fm or {}).items():
        c = "#0a7d33" if (v["coef"] or 0) > 0 else "#b91c1c"
        tw = "700" if abs(v.get("t") or 0) >= 2 else "400"
        fmrows += (
            f"<tr><td><code>{k}</code></td><td style='color:{c}'>{_f(v['coef'])}</td>"
            f"<td style='font-weight:{tw}'>{_f(v['t'], 2)}</td><td>{v['n_dates']}</td></tr>"
        )
    fm_block = (
        "<h2>Fama-MacBeth — simultaneous premia (HAC t)</h2>"
        "<table><thead><tr><th>Factor</th><th>mean coef</th><th>t (HAC)</th><th>dates</th></tr>"
        f"</thead><tbody>{fmrows}</tbody></table>"
        if fm
        else ""
    )
    body = "\n".join(frow(r) for r in rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 PIT Fundamentals Backtest</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:900px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:16px}}
h2{{font-size:17px;margin:24px 0 8px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f7f8fa;padding:7px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2}}code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0}}</style></head><body>
<h1>v3 PIT Fundamentals Backtest — the data-blocked durable premia</h1>
<div class="sub">Monthly forward IC over {n_used} point-in-time cross-sections
({dates[0].date()} .. {dates[-1].date()}, ~{period} price history, {n_px:,} priced) from the
Sharadar SF1 store (datekey-lagged, no look-ahead). Book-to-price, asset-growth (CMA),
GP/assets (Novy-Marx), accruals (Sloan).</div>
<p class="note"><b>The unlock:</b> unlike the 5-month etoro.csv panel, this spans multiple
regimes, so the HAC t-stats and Fama-MacBeth premia are the first read the DSR / ≥2-regime
gate can actually use. <b>t (HAC)</b> = Newey-West ({HAC_LAGS} lags). Survivorship caveat:
prices are yfinance survivors; delisted-name pricing (Sharadar SEP) is the next data step.</p>
<table><thead><tr><th>Factor</th><th>IC raw</th><th>IC β-neutral</th><th>t (HAC)</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{body}</tbody></table>
{fm_block}
</body></html>"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


if __name__ == "__main__":
    main()

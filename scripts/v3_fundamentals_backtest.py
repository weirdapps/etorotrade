"""Phase 3 — SURVIVORSHIP-CLEAN multi-year backtest of the PIT fundamental factors.

Sources (all delisted-inclusive, point-in-time):
- DAILY store (``v3_daily_monthly.parquet``): month-end marketcap (return proxy +
  size) and pb / pe / ps (PIT valuation ratios), for the full US common-stock
  universe INCLUDING the ~2,915 delisted names — this removes the survivorship bias.
- SF1 store (``v3_fundamentals_store.parquet``): assets/equity/gp/netinc/ncfo/eps.

Factors (directional sign): book_to_price=1/pb, earnings_yield=1/pe, sales_yield=1/ps
(value); gp_assets (Novy-Marx quality); asset_growth (CMA, −); accruals (Sloan, −);
sue (PEAD, seasonal-random-walk). Forward return = month-over-month marketcap growth.
Reports forward IC raw→beta-neutral with HAC (Newey-West) t + Fama-MacBeth.

    .venv/bin/python scripts/v3_fundamentals_backtest.py

Caveat: marketcap growth ≈ price return only while shares are stable (issuance/buyback
noise, mainly the asset-growth factor). Exact prices need SEP-full. Global rank-z
(no sector demean here — the offline sector map is thin; per-sector value was covered
by the etoro-panel backtest).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.combine import _rank_z  # noqa: E402
from trade_modules.v3.factor_backtest import (  # noqa: E402
    beta_neutralize,
    fama_macbeth,
    hac_tstat,
)
from trade_modules.v3.fundamentals import factor_panel  # noqa: E402
from trade_modules.validation.xsection_ic import cross_sectional_ic  # noqa: E402

DAILY_STORE = str(Path("~/.weirdapps-trading/v3_daily_monthly.parquet").expanduser())
SF1_STORE = str(Path("~/.weirdapps-trading/v3_fundamentals_store.parquet").expanduser())
HAC_LAGS = 12
BETA_WIN = 24
# factor -> directional sign (+1 high-good / -1 low-good)
SIGNS = {
    "book_to_price": +1,
    "earnings_yield": +1,
    "sales_yield": +1,
    "gp_assets": +1,
    "asset_growth": -1,
    "accruals": -1,
    "sue": +1,
}


def _matrix(daily: pd.DataFrame, col: str) -> pd.DataFrame:
    m = daily.pivot_table(index="date", columns="ticker", values=col, aggfunc="last")
    m.index = pd.to_datetime(m.index)
    return m.sort_index()


def _asof(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    sub = df[df["datekey"] <= as_of]
    return sub.sort_values("datekey").groupby("ticker").last() if not sub.empty else pd.DataFrame()


def _sue_table(sf1: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, datekey) seasonal-random-walk SUE from the EPS history."""
    out = []
    for tkr, g in sf1.sort_values("datekey").groupby("ticker"):
        eps = pd.to_numeric(g["eps"], errors="coerce")
        d4 = eps.diff(4)
        sue = d4 / d4.expanding(min_periods=2).std()
        out.append(
            pd.DataFrame({"ticker": tkr, "datekey": g["datekey"].to_numpy(), "sue": sue.to_numpy()})
        )
    return pd.concat(out, ignore_index=True).dropna(subset=["sue"]) if out else pd.DataFrame()


def main() -> None:
    if not Path(DAILY_STORE).exists():
        print(f"no DAILY store at {DAILY_STORE} — run v3_daily_update.py first")
        return
    daily = pd.read_parquet(DAILY_STORE)
    sf1 = pd.read_parquet(SF1_STORE)
    mc = _matrix(daily, "marketcap")
    pb, pe, ps = _matrix(daily, "pb"), _matrix(daily, "pe"), _matrix(daily, "ps")
    print(
        f"DAILY store: {mc.shape[1]} names, {mc.shape[0]} months ({mc.index[0].date()}..{mc.index[-1].date()})"
    )
    print(f"SF1 store: {sf1['ticker'].nunique()} names")

    fwd_mat = mc.shift(-1) / mc - 1.0
    mret = mc.pct_change(fill_method=None)
    mkt = mret.mean(axis=1)
    beta_mat = mret.rolling(BETA_WIN).cov(mkt).div(mkt.rolling(BETA_WIN).var(), axis=0)
    sue_tbl = _sue_table(sf1)

    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in SIGNS}
    fm_parts: list[pd.DataFrame] = []
    dates = mc.index[12:-1]
    n_used = 0
    for T in dates:
        fwd = fwd_mat.loc[T].dropna()
        if len(fwd) < 20:
            continue
        tkey = T.strftime("%Y-%m-%d")
        pkey = (T - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        fasof = _asof(sf1, tkey)
        if fasof.empty:
            continue
        n_used += 1
        # value factors straight from DAILY PIT ratios (guard ratio > 0)
        vals: dict[str, pd.Series] = {}
        for name, ratio in (("book_to_price", pb), ("earnings_yield", pe), ("sales_yield", ps)):
            r = ratio.loc[T]
            vals[name] = (1.0 / r).where(r > 0)
        # SF1 balance-sheet / cash-flow factors
        fp = factor_panel(fasof, _asof(sf1, pkey), mc.loc[T])
        for name in ("gp_assets", "asset_growth", "accruals"):
            vals[name] = fp[name]
        sue_asof = _asof(sue_tbl, tkey) if not sue_tbl.empty else pd.DataFrame()
        vals["sue"] = sue_asof["sue"] if "sue" in sue_asof.columns else pd.Series(dtype=float)

        beta_T = beta_mat.loc[T] if T in beta_mat.index else None
        fwd_n = (
            beta_neutralize(fwd, beta_T)
            if beta_T is not None
            else pd.Series(np.nan, index=fwd.index)
        )

        fm_row: dict[str, pd.Series] = {}
        for name, sign in SIGNS.items():
            z = _rank_z(vals[name]) * sign
            part = pd.DataFrame(
                {"date": tkey, "z": z, "fwd": fwd.reindex(z.index), "fwd_n": fwd_n.reindex(z.index)}
            ).dropna(subset=["z", "fwd"])
            if len(part):
                parts[name].append(part.reset_index(names="ticker"))
            fm_row[name] = z
        fmdf = pd.DataFrame(fm_row)
        fmdf["fwd"] = fwd.reindex(fmdf.index)
        fmdf["date"] = tkey
        fm_parts.append(fmdf.reset_index(names="ticker"))

    print(f"monthly cross-sections used: {n_used}  ({dates[0].date()}..{dates[-1].date()})")

    rows = []
    for name in SIGNS:
        if not parts[name]:
            continue
        d = pd.concat(parts[name], ignore_index=True)
        raw = cross_sectional_ic(d, "z", "fwd", date_col="date")
        neu = cross_sectional_ic(d.dropna(subset=["fwd_n"]), "z", "fwd_n", date_col="date")
        raw_s = pd.Series(raw.get("ic_by_date") or {}, dtype=float).sort_index()
        neu_s = pd.Series(neu.get("ic_by_date") or {}, dtype=float).sort_index()
        rows.append(
            {
                "name": name,
                "ic_raw": raw.get("mean_ic"),
                "t_raw": hac_tstat(raw_s, HAC_LAGS) if len(raw_s) > 1 else float("nan"),
                "ic_neu": neu.get("mean_ic"),
                "t_neu": hac_tstat(neu_s, HAC_LAGS) if len(neu_s) > 1 else float("nan"),
                "hit": neu.get("hit_rate"),
                "n_dates": raw.get("n_dates"),
                "n_obs": len(d),
            }
        )
    rows.sort(
        key=lambda r: (
            r["ic_neu"] if isinstance(r["ic_neu"], float) and r["ic_neu"] == r["ic_neu"] else -9
        )
    )

    fm = {}
    if fm_parts:
        fmp = pd.concat(fm_parts, ignore_index=True)
        fm = fama_macbeth(fmp, "fwd", list(SIGNS), date_col="date", lags=HAC_LAGS)

    _print(rows, fm, n_used, mc)
    out = _render(rows, fm, n_used, dates, mc.shape[1])
    print(f"\nreport -> {out}")


def _f(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


def _print(rows, fm, n_used, mc) -> None:
    print(f"\n=== SURVIVORSHIP-CLEAN PIT FUNDAMENTALS forward IC ({n_used} months, HAC t) ===")
    print(
        f"{'factor':16} {'IC_raw':>8} {'tR_HAC':>7} {'IC_neu':>8} {'tN_HAC':>7} {'hit':>5} {'dates':>5}"
    )
    for r in rows:
        print(
            f"{r['name']:16} {_f(r['ic_raw']):>8} {_f(r['t_raw'], 2):>7} {_f(r['ic_neu']):>8} "
            f"{_f(r['t_neu'], 2):>7} {_f(r['hit'], 2):>5} {r['n_dates'] or 0:>5}"
        )
    if fm:
        print("\n--- Fama-MacBeth (all factors jointly, HAC t) ---")
        for k, v in fm.items():
            print(f"{k:16} coef {_f(v['coef']):>9}  tHAC {_f(v['t'], 2):>6}")


def _render(rows, fm, n_used, dates, n_px) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_fundamentals_backtest.html")

    def frow(r):
        good = (r["ic_neu"] or 0) > 0
        strong = abs(r.get("t_neu") or 0) >= 2
        col = "#0a7d33" if good else "#b91c1c"
        tw = "700" if strong else "400"
        return (
            f"<tr><td><code>{r['name']}</code></td><td>{_f(r['ic_raw'])}</td>"
            f"<td>{_f(r['t_raw'], 2)}</td><td style='color:{col};font-weight:{tw}'>{_f(r['ic_neu'])}</td>"
            f"<td style='font-weight:{tw}'>{_f(r['t_neu'], 2)}</td>"
            f"<td>{_f(r['hit'], 2)}</td><td>{r['n_dates'] or 0}</td><td>{r['n_obs']:,}</td></tr>"
        )

    fmrows = ""
    for k, v in (fm or {}).items():
        c = "#0a7d33" if (v["coef"] or 0) > 0 else "#b91c1c"
        tw = "700" if abs(v.get("t") or 0) >= 2 else "400"
        fmrows += f"<tr><td><code>{k}</code></td><td style='color:{c}'>{_f(v['coef'])}</td><td style='font-weight:{tw}'>{_f(v['t'], 2)}</td></tr>"
    fm_block = (
        "<h2>Fama-MacBeth — all factors jointly (HAC t)</h2><table><thead><tr><th>Factor</th>"
        f"<th>mean coef</th><th>t (HAC)</th></tr></thead><tbody>{fmrows}</tbody></table>"
        if fm
        else ""
    )
    body = "\n".join(frow(r) for r in rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 Survivorship-Clean Fundamentals Backtest</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:900px;margin:0 auto;padding:30px}}
h1{{font-size:22px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:14px}}
h2{{font-size:17px;margin:22px 0 8px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f7f8fa;padding:7px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2}}code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0}}</style></head><body>
<h1>v3 Survivorship-Clean PIT Fundamentals Backtest</h1>
<div class="sub">{n_used} monthly cross-sections ({dates[0].date()}..{dates[-1].date()}), {n_px:,} US common
stocks <b>including delisted</b> (Sharadar DAILY, marketcap return proxy) × the SF1 point-in-time store.</div>
<p class="note"><b>t (HAC)</b> = Newey-West ({HAC_LAGS} lags). β-neutral strips the market/beta component.
Survivorship removed (delisted names included). Residual caveat: marketcap growth ≈ price return only
while share count is stable (issuance/buyback noise — mainly asset_growth). Global rank-z.</p>
<table><thead><tr><th>Factor</th><th>IC raw</th><th>t raw</th><th>IC β-neutral</th><th>t β-neut</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{body}</tbody></table>
{fm_block}
</body></html>"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


if __name__ == "__main__":
    main()

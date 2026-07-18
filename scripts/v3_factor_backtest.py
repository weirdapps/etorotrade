"""Panel-history per-factor alpha backtest.

Reconstructs the etoro.csv panel from its git history (a de-facto point-in-time
panel, committed daily), computes each factor's directional cross-sectional rank-z
per weekly snapshot, and measures forward IC (Spearman, at V3_SIGNAL_HORIZON) per
FACTOR and per CLUSTER — including the DISCARDED panel factors so the discard
decisions can be validated. Forward returns come from EUR-adjusted prices.

    .venv/bin/python scripts/v3_factor_backtest.py

Writes ~/Downloads/<UTCstamp>_v3_factor_backtest.html + prints a ranked table.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.combine import CLUSTERS, DIRECTION, _rank_z, _sector_demean  # noqa: E402
from trade_modules.v3.factor_backtest import (  # noqa: E402
    BACKTEST_PANEL_FACTORS,
    factor_zscore,
    forward_return_at,
    is_active,
)
from trade_modules.validation.xsection_ic import cross_sectional_ic  # noqa: E402

ETORO = "yahoofinance/output/etoro.csv"
HORIZON = 21  # V3_SIGNAL_HORIZON (trading days)


def git_snapshots(path: str, weekly: bool = True) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%H|%cI", "--", path], capture_output=True, text=True
    ).stdout
    snaps = [(c, d[:10]) for c, d in (ln.split("|") for ln in out.splitlines() if "|" in ln)]
    if weekly:  # keep one (the newest) commit per ISO week
        seen: set = set()
        keep = []
        for c, d in snaps:  # newest-first
            wk = datetime.fromisoformat(d).isocalendar()[:2]
            if wk not in seen:
                seen.add(wk)
                keep.append((c, d))
        snaps = keep
    return list(reversed(snaps))  # oldest-first


def load_panel_at(commit: str, path: str) -> pd.DataFrame | None:
    txt = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True).stdout
    if not txt.strip():
        return None
    df = pd.read_csv(io.StringIO(txt), na_values=["--"])
    if "TKR" not in df.columns:
        return None
    return df.dropna(subset=["TKR"]).drop_duplicates("TKR").set_index("TKR")


def main() -> None:
    from trade_modules.v3.prices import load_eur_close
    from trade_modules.v3.sectors import load_offline_sector_map
    from trade_modules.v3.universe import load_universe

    snaps = git_snapshots(ETORO)
    print(f"weekly snapshots: {len(snaps)}  ({snaps[0][1]} .. {snaps[-1][1]})")

    universe = set(load_universe(ETORO, min_factor_coverage=6))  # global coverage names
    print(f"backtest universe (global coverage ≥6/7): {len(universe)}")
    px = load_eur_close(sorted(universe), period="1y")
    print(f"priced (EUR): {px.shape[1]} names, {px.shape[0]} bars")
    sector_map = load_offline_sector_map()
    pidx = px.index

    factors = dict(BACKTEST_PANEL_FACTORS)  # feature-name set (panel) + derived below
    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in set(factors.values())}
    parts["mom_12_1"] = []
    parts["realized_vol"] = []
    cluster_parts: dict[str, list[pd.DataFrame]] = {c: [] for c in CLUSTERS}

    n_used = 0
    for commit, date in snaps:
        panel = load_panel_at(commit, ETORO)
        if panel is None:
            continue
        panel = panel[panel.index.astype(str).isin(universe)]
        if panel.empty:
            continue
        mask = pidx <= pd.Timestamp(date)
        if not mask.any():
            continue
        asof = pidx[mask].max()
        fwd = forward_return_at(px, asof, HORIZON)
        if fwd.empty:
            continue
        n_used += 1
        sec = pd.Series({t: sector_map.get(str(t).upper()) for t in panel.index}, index=panel.index)

        zcols: dict[str, pd.Series] = {}
        for col, feat in BACKTEST_PANEL_FACTORS.items():
            if col not in panel.columns:
                continue
            z = factor_zscore(panel[col], feat, sector=sec, sector_neutral=True)
            zcols[feat] = z
            part = pd.DataFrame({"date": date, "z": z, "fwd": fwd.reindex(z.index)}).dropna()
            if len(part):
                parts[feat].append(part.reset_index(names="ticker"))

        # price-derived momentum (12-1) + realized vol, PIT from the price matrix
        i = pidx.get_loc(asof)
        if i >= 252:
            mom = (px.iloc[i - 21] / px.iloc[i - 252]) - 1.0
            vol = px.iloc[i - 252 : i + 1].pct_change(fill_method=None).std() * (252**0.5)
            for feat, val in (("mom_12_1", mom), ("realized_vol", vol)):
                z = _rank_z(val) * DIRECTION[feat]
                z = _sector_demean(z, sec.reindex(z.index))
                zcols[feat] = z
                part = pd.DataFrame({"date": date, "z": z, "fwd": fwd.reindex(z.index)}).dropna()
                if len(part):
                    parts[feat].append(part.reset_index(names="ticker"))

        # per-cluster z = mean of member factor-z present this snapshot
        for cluster, members in CLUSTERS.items():
            present = [zcols[m] for m in members if m in zcols]
            if not present:
                continue
            cz = pd.concat(present, axis=1).mean(axis=1, skipna=True)
            part = pd.DataFrame({"date": date, "z": cz, "fwd": fwd.reindex(cz.index)}).dropna()
            if len(part):
                cluster_parts[cluster].append(part.reset_index(names="ticker"))

    print(f"snapshots used (with forward data): {n_used}")

    def ic_rows(store: dict, kind: str) -> list[dict]:
        rows = []
        for name, plist in store.items():
            if not plist:
                continue
            d = pd.concat(plist, ignore_index=True)
            ic = cross_sectional_ic(d, "z", "fwd", date_col="date")
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "active": is_active(name) if kind == "factor" else True,
                    "mean_ic": ic.get("mean_ic"),
                    "t_stat": ic.get("t_stat"),
                    "hit_rate": ic.get("hit_rate"),
                    "n_dates": ic.get("n_dates"),
                    "n_obs": len(d),
                }
            )
        return rows

    fac = sorted(ic_rows(parts, "factor"), key=lambda r: r["mean_ic"] or -9, reverse=True)
    clu = sorted(ic_rows(cluster_parts, "cluster"), key=lambda r: r["mean_ic"] or -9, reverse=True)

    _print(clu, "CLUSTERS")
    _print(fac, "FACTORS")
    out = _render(fac, clu, n_used, len(universe), px.shape[1], snaps)
    print(f"\nreport -> {out}")


def _print(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} (forward IC @ {HORIZON}td) ===")
    print(f"{'name':16} {'act':>3} {'meanIC':>8} {'t':>6} {'hit':>6} {'dates':>5} {'obs':>7}")
    for r in rows:
        act = "" if r["kind"] == "cluster" else ("A" if r["active"] else "·")
        print(
            f"{r['name']:16} {act:>3} {(_f(r['mean_ic'])):>8} {_f(r['t_stat'], 2):>6} "
            f"{_f(r['hit_rate'], 2):>6} {r['n_dates'] or 0:>5} {r['n_obs']:>7}"
        )


def _f(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


def _render(fac, clu, n_snap, n_uni, n_px, snaps) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_factor_backtest.html")

    def rowhtml(r):
        good = (r["mean_ic"] or 0) > 0
        strong = abs(r.get("t_stat") or 0) >= 2
        act = "active" if r["active"] else "discarded"
        badge = (
            f'<span class="b {"act" if r["active"] else "disc"}">{act}</span>'
            if r["kind"] == "factor"
            else ""
        )
        col = "#0a7d33" if good else "#b91c1c"
        tw = "700" if strong else "400"
        return (
            f"<tr><td><code>{r['name']}</code> {badge}</td>"
            f"<td style='color:{col};font-weight:{tw}'>{_f(r['mean_ic'])}</td>"
            f"<td style='font-weight:{tw}'>{_f(r['t_stat'], 2)}</td>"
            f"<td>{_f(r['hit_rate'], 2)}</td><td>{r['n_dates'] or 0}</td><td>{r['n_obs']:,}</td></tr>"
        )

    ch = "\n".join(rowhtml(r) for r in clu)
    fh = "\n".join(rowhtml(r) for r in fac)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 Factor Backtest</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:900px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:20px}}
h2{{font-size:17px;margin:26px 0 8px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f7f8fa;padding:7px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2}}code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.b{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:20px;margin-left:4px}}.act{{color:#0a7d33;background:#e5f6ea}}.disc{{color:#6b7280;background:#eef0f2}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0 2px}}</style></head><body>
<h1>v3 Panel-History Factor Backtest</h1>
<div class="sub">Forward IC (Spearman) at {HORIZON}td, over {n_snap} weekly point-in-time snapshots
({snaps[0][1]} .. {snaps[-1][1]}) reconstructed from etoro.csv git history · universe {n_uni:,} global coverage names, {n_px:,} priced (EUR).</div>
<p class="note"><b>How to read:</b> positive mean IC with |t|≥2 (bold) = evidence of alpha at this horizon on this (short) sample. <b>active</b> = in a live cluster; <b>discarded</b> = removed from scoring (shown to validate the discard). This is ~{n_snap} weekly dates — indicative, not the deflated-Sharpe gate.</p>
<h2>Clusters</h2>
<table><thead><tr><th>Cluster</th><th>mean IC</th><th>t-stat</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{ch}</tbody></table>
<h2>Factors (active + discarded)</h2>
<table><thead><tr><th>Factor</th><th>mean IC</th><th>t-stat</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{fh}</tbody></table>
<p class="note">Excludes ev_ebitda + rev_growth (yfinance .info is current-only, not point-in-time reconstructable). mom_12_1 / realized_vol reconstructed from adjusted prices.</p>
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    main()

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
    beta_neutralize,
    factor_zscore,
    forward_return_at,
    ic_by_sector,
    is_active,
)
from trade_modules.v3.features import _num  # noqa: E402
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


def _append_part(store: dict, key: str, date: str, z, fwd, fwd_n, sector) -> None:
    """Append (ticker, z, raw fwd, beta-neutral fwd, sector) rows to ``store[key]``."""
    part = pd.DataFrame(
        {
            "date": date,
            "z": z,
            "fwd": fwd.reindex(z.index),
            "fwd_n": fwd_n.reindex(z.index),
            "sector": pd.Series(sector).reindex(z.index).astype("object"),
        }
    ).dropna(subset=["z", "fwd"])
    if len(part):
        store[key].append(part.reset_index(names="ticker"))


# Valuation/quality metrics whose per-sector IC we compare (label shown in the report).
VAL_FEATS: list[tuple[str, str]] = [
    ("pe_trailing", "P/E"),
    ("ps_sector", "P/S"),
    ("peg", "PEG"),
    ("roe", "ROE"),
    ("fcf", "FCF"),
]


def _sector_matrix(parts: dict, feats: list[tuple[str, str]]) -> list[dict]:
    """Per-sector beta-neutral IC for each feature -> one row per sector (thin ones dropped)."""
    per_feat: dict[str, dict] = {}
    for feat, label in feats:
        plist = parts.get(feat) or []
        if not plist:
            continue
        d = pd.concat(plist, ignore_index=True)
        per_feat[label] = ic_by_sector(
            d, "z", "fwd_n", "sector", date_col="date", min_names_per_date=8
        )
    sectors = sorted({s for m in per_feat.values() for s in m})
    rows: list[dict] = []
    for s in sectors:
        row: dict = {"sector": s}
        n = 0
        for _feat, label in feats:
            cell = per_feat.get(label, {}).get(s)
            row[label] = cell["mean_ic"] if cell else None
            if cell:
                n = max(n, int(cell["n_obs"]))
        row["_n"] = n
        rows.append(row)
    return [r for r in rows if r["_n"] >= 24]  # drop sectors too thin to rank


def _print_sector(rows: list[dict], labels: list[str]) -> None:
    if not rows:
        print(f"\n=== PER-SECTOR valuation IC (β-neutral @ {HORIZON}td): insufficient data ===")
        return
    print(f"\n=== PER-SECTOR valuation IC (β-neutral @ {HORIZON}td) ===")
    print(f"{'sector':24}" + "".join(f"{lb:>7}" for lb in labels) + f"{'obs':>7}")
    for r in rows:
        print(
            f"{str(r['sector'])[:24]:24}"
            + "".join(f"{_f(r[lb], 3):>7}" for lb in labels)
            + f"{r['_n']:>7}"
        )


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
        # Market-neutral forward return: cross-sectional residual on beta (removes the
        # market-beta component that dominates a trending regime) — the honest alpha test.
        beta = _num(panel["B"]).reindex(fwd.index) if "B" in panel.columns else None
        fwd_n = beta_neutralize(fwd, beta) if beta is not None else fwd

        zcols: dict[str, pd.Series] = {}
        for col, feat in BACKTEST_PANEL_FACTORS.items():
            if col not in panel.columns:
                continue
            z = factor_zscore(panel[col], feat, sector=sec, sector_neutral=True)
            zcols[feat] = z
            _append_part(parts, feat, date, z, fwd, fwd_n, sec)

        # price-derived momentum (12-1) + realized vol, PIT from the price matrix
        i = pidx.get_loc(asof)
        if i >= 252:
            mom = (px.iloc[i - 21] / px.iloc[i - 252]) - 1.0
            vol = px.iloc[i - 252 : i + 1].pct_change(fill_method=None).std() * (252**0.5)
            for feat, val in (("mom_12_1", mom), ("realized_vol", vol)):
                z = _sector_demean(_rank_z(val) * DIRECTION[feat], sec.reindex(val.index))
                zcols[feat] = z
                _append_part(parts, feat, date, z, fwd, fwd_n, sec)

        # per-cluster z = mean of member factor-z present this snapshot
        for cluster, members in CLUSTERS.items():
            present = [zcols[m] for m in members if m in zcols]
            if present:
                cz = pd.concat(present, axis=1).mean(axis=1, skipna=True)
                _append_part(cluster_parts, cluster, date, cz, fwd, fwd_n, sec)

    print(f"snapshots used (with forward data): {n_used}")

    def ic_rows(store: dict, kind: str) -> list[dict]:
        rows = []
        for name, plist in store.items():
            if not plist:
                continue
            d = pd.concat(plist, ignore_index=True)
            raw = cross_sectional_ic(d, "z", "fwd", date_col="date")
            neu = cross_sectional_ic(d.dropna(subset=["fwd_n"]), "z", "fwd_n", date_col="date")
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "active": is_active(name) if kind == "factor" else True,
                    "ic_raw": raw.get("mean_ic"),
                    "t_raw": raw.get("t_stat"),
                    "ic_neu": neu.get("mean_ic"),
                    "t_neu": neu.get("t_stat"),
                    "hit_neu": neu.get("hit_rate"),
                    "n_dates": neu.get("n_dates"),
                    "n_obs": len(d),
                }
            )
        return rows

    def _key(r: dict) -> float:
        v = r["ic_neu"]
        return v if isinstance(v, float) and v == v else -9.0

    fac = sorted(ic_rows(parts, "factor"), key=_key, reverse=True)
    clu = sorted(ic_rows(cluster_parts, "cluster"), key=_key, reverse=True)
    sector_val = _sector_matrix(parts, VAL_FEATS)

    _print(clu, "CLUSTERS")
    _print(fac, "FACTORS")
    _print_sector(sector_val, [lb for _feat, lb in VAL_FEATS])
    out = _render(fac, clu, sector_val, n_used, len(universe), px.shape[1], snaps)
    print(f"\nreport -> {out}")


def _print(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} (forward IC @ {HORIZON}td: raw -> beta-neutral) ===")
    print(
        f"{'name':16} {'act':>3} {'IC_raw':>8} {'IC_neu':>8} {'t_neu':>6} {'hit':>5} {'dates':>5}"
    )
    for r in rows:
        act = "" if r["kind"] == "cluster" else ("A" if r["active"] else "·")
        print(
            f"{r['name']:16} {act:>3} {_f(r['ic_raw']):>8} {_f(r['ic_neu']):>8} "
            f"{_f(r['t_neu'], 2):>6} {_f(r['hit_neu'], 2):>5} {r['n_dates'] or 0:>5}"
        )


def _f(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


def _render(fac, clu, sector_val, n_snap, n_uni, n_px, snaps) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_factor_backtest.html")

    def rowhtml(r):
        good = (r["ic_neu"] or 0) > 0
        strong = abs(r.get("t_neu") or 0) >= 2
        act = "active" if r["active"] else "discarded"
        badge = (
            f'<span class="b {"act" if r["active"] else "disc"}">{act}</span>'
            if r["kind"] == "factor"
            else ""
        )
        col = "#0a7d33" if good else "#b91c1c"
        rawcol = "#0a7d33" if (r["ic_raw"] or 0) > 0 else "#b91c1c"
        tw = "700" if strong else "400"
        return (
            f"<tr><td><code>{r['name']}</code> {badge}</td>"
            f"<td style='color:{rawcol}'>{_f(r['ic_raw'])}</td>"
            f"<td style='color:{col};font-weight:{tw}'>{_f(r['ic_neu'])}</td>"
            f"<td style='font-weight:{tw}'>{_f(r['t_neu'], 2)}</td>"
            f"<td>{_f(r['hit_neu'], 2)}</td><td>{r['n_dates'] or 0}</td><td>{r['n_obs']:,}</td></tr>"
        )

    ch = "\n".join(rowhtml(r) for r in clu)
    fh = "\n".join(rowhtml(r) for r in fac)

    labels = [lb for _feat, lb in VAL_FEATS]

    def sechtml(r):
        tds = ""
        for lb in labels:
            v = r[lb]
            has = isinstance(v, (int, float)) and v == v
            c = "#0a7d33" if (has and v > 0) else ("#b91c1c" if has else "#9aa1a9")
            tds += f"<td style='color:{c}'>{_f(v, 3)}</td>"
        return f"<tr><td>{r['sector']}</td>{tds}<td>{r['_n']:,}</td></tr>"

    if sector_val:
        sh = "\n".join(sechtml(r) for r in sector_val)
        sec_th = "".join(f"<th>{lb}</th>" for lb in labels)
        sec_block = (
            "<h2>Per-sector valuation IC (β-neutral) — is the right metric sector-specific?</h2>"
            f"<table><thead><tr><th>Sector</th>{sec_th}<th>obs</th></tr></thead>"
            f"<tbody>{sh}</tbody></table>"
            '<p class="note">Within-sector forward IC (β-neutral, ≥8 names/date-slice) of each '
            "valuation/quality metric. A metric that predicts in one sector but not another is "
            "evidence valuation should be <b>sector-conditional</b> — e.g. banks are classically "
            "valued on <b>P/B</b> and REITs on <b>P/FFO / P-NAV</b>, neither of which is in this "
            "point-in-time panel (yfinance-.info-only) — both flagged as a BUILD item. Thin sectors "
            "(few names/date) omitted.</p>"
        )
    else:
        sec_block = (
            "<h2>Per-sector valuation IC</h2>"
            '<p class="note">Insufficient per-sector data at this sample size.</p>'
        )
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
<p class="note"><b>How to read:</b> <b>IC raw</b> = Spearman(factor-z, raw forward return); <b>IC β-neutral</b> = vs the beta-residualised return (per-date OLS residual on beta) — this removes the market-beta component that dominates a trending regime and is the <b>honest test of factor alpha</b>. Positive β-neutral IC with |t|≥2 (bold) = evidence of alpha on this (short) sample. <b>active</b> = in a live cluster; <b>discarded</b> = removed from scoring (shown to validate the discard). ~{n_snap} weekly dates — indicative, NOT the deflated-Sharpe gate.</p>
<h2>Clusters</h2>
<table><thead><tr><th>Cluster</th><th>IC raw</th><th>IC β-neutral</th><th>t (neu)</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{ch}</tbody></table>
<h2>Factors (active + discarded)</h2>
<table><thead><tr><th>Factor</th><th>IC raw</th><th>IC β-neutral</th><th>t (neu)</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{fh}</tbody></table>
<p class="note">β-neutral = cross-sectional residual of the forward return on beta (isolates factor alpha from the market/regime). Excludes ev_ebitda + rev_growth (yfinance .info is current-only, not point-in-time reconstructable). mom_12_1 / realized_vol reconstructed from adjusted prices.</p>
{sec_block}
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    main()

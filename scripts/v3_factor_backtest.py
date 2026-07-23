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

from trade_modules.v3.combine import (  # noqa: E402
    CLUSTER_WEIGHTS,
    CLUSTERS,
    DIRECTION,
    _rank_z,
    _sector_demean,
)
from trade_modules.v3.factor_backtest import (  # noqa: E402
    BACKTEST_PANEL_FACTORS,
    beta_neutralize,
    factor_zscore,
    fama_macbeth,
    forward_return_at,
    hac_tstat,
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
    # Keep the newest commit per ISO week (weekly) or per calendar day (daily).
    seen: set = set()
    keep = []
    for c, d in snaps:  # newest-first
        key = datetime.fromisoformat(d).isocalendar()[:2] if weekly else d
        if key not in seen:
            seen.add(key)
            keep.append((c, d))
    return list(reversed(keep))  # oldest-first


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

# P3: core clusters entered SIMULTANEOUSLY in the Fama-MacBeth regression (+ beta).
# Growth/strength are excluded (too sparse -> would shrink the joint sample).
_FM_CLUSTERS = ["value_z", "quality_z", "momentum_z", "lowvol_z"]


def _sector_matrix(parts: dict, feats: list[tuple[str, str]]) -> list[dict]:
    """Per-sector beta-neutral IC for each feature -> one row per sector (thin ones dropped)."""
    per_feat: dict[str, dict] = {}
    for feat, label in feats:
        plist = parts.get(feat) or []
        if not plist:
            continue
        d = pd.concat(plist, ignore_index=True)
        per_feat[label] = ic_by_sector(
            d, "z", "fwd_n", "sector", date_col="date", min_names_per_date=8, min_dates=3
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

    snaps = git_snapshots(ETORO, weekly=False)  # P3: full daily panel (~5x the dates)
    print(f"daily snapshots: {len(snaps)}  ({snaps[0][1]} .. {snaps[-1][1]})")

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
    parts["earn_trajectory"] = []
    cluster_parts: dict[str, list[pd.DataFrame]] = {c: [] for c in CLUSTERS}
    fm_parts: list[pd.DataFrame] = []  # P3: wide panel for Fama-MacBeth
    first_tickers: set = set()
    last_tickers: set = set()

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

        # earnings trajectory = forward/trailing P/E (PEF/PET, owner 2026-07-23): LOW =
        # forward cheaper = earnings expected to RISE; high = value-trap. DIRECTION -1
        # (smaller is better). The z is IDENTICAL to the old PET/PEF (reciprocal + sign
        # cancel in rank-z), so the measured IC/t is unchanged — this only matches the
        # engine's expression faithfully.
        if "PET" in panel.columns and "PEF" in panel.columns:
            pet, pef = _num(panel["PET"]), _num(panel["PEF"])
            traj = (pef / pet).where((pet > 0) & (pef > 0))
            ztraj = _sector_demean(
                _rank_z(traj) * DIRECTION["earn_trajectory"], sec.reindex(traj.index)
            )
            zcols["earn_trajectory"] = ztraj
            _append_part(parts, "earn_trajectory", date, ztraj, fwd, fwd_n, sec)

        # per-cluster z = mean of member factor-z present this snapshot
        cluster_cz: dict[str, pd.Series] = {}
        for cluster, members in CLUSTERS.items():
            present = [zcols[m] for m in members if m in zcols]
            if present:
                cz = pd.concat(present, axis=1).mean(axis=1, skipna=True)
                cluster_cz[cluster] = cz
                _append_part(cluster_parts, cluster, date, cz, fwd, fwd_n, sec)

        # P3: wide row for the Fama-MacBeth regression (core clusters + beta vs fwd).
        if beta is not None:
            fm = pd.DataFrame({c: cluster_cz[c] for c in _FM_CLUSTERS if c in cluster_cz})
            if "growth_z" in cluster_cz:  # include growth + trajectory for the spanning test
                fm["growth_z"] = cluster_cz["growth_z"]
            if "earn_trajectory" in zcols:
                fm["earn_trajectory"] = zcols["earn_trajectory"]
            if not fm.empty:
                fm["beta"] = beta.reindex(fm.index)
                fm["fwd"] = fwd.reindex(fm.index)
                fm["date"] = date
                fm_parts.append(fm.reset_index(names="ticker"))
        # P3: survivorship — first vs last snapshot ticker sets (churn = look-ahead flag).
        if not first_tickers:
            first_tickers = set(panel.index.astype(str))
        last_tickers = set(panel.index.astype(str))

    print(f"snapshots used (with forward data): {n_used}")

    def ic_rows(store: dict, kind: str) -> list[dict]:
        rows = []
        for name, plist in store.items():
            if not plist:
                continue
            d = pd.concat(plist, ignore_index=True)
            raw = cross_sectional_ic(d, "z", "fwd", date_col="date")
            neu = cross_sectional_ic(d.dropna(subset=["fwd_n"]), "z", "fwd_n", date_col="date")
            # P3: HAC (Newey-West) t on the date-ordered per-date IC series — the
            # naive t overstates significance under overlapping forward windows.
            raw_s = pd.Series(raw.get("ic_by_date") or {}, dtype=float).sort_index()
            neu_s = pd.Series(neu.get("ic_by_date") or {}, dtype=float).sort_index()
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "active": is_active(name) if kind == "factor" else True,
                    "ic_raw": raw.get("mean_ic"),
                    "t_raw": raw.get("t_stat"),
                    "ic_neu": neu.get("mean_ic"),
                    "t_neu": neu.get("t_stat"),
                    "t_neu_hac": hac_tstat(neu_s, HORIZON) if len(neu_s) > 1 else float("nan"),
                    "t_raw_hac": hac_tstat(raw_s, HORIZON) if len(raw_s) > 1 else float("nan"),
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

    # P3: Fama-MacBeth simultaneous premia (replaces sequential residualization) + churn.
    fm_result: dict = {}
    if fm_parts:
        fm_panel = pd.concat(fm_parts, ignore_index=True)
        fm_x = [
            c for c in (*_FM_CLUSTERS, "growth_z", "earn_trajectory") if c in fm_panel.columns
        ] + ["beta"]
        fm_result = fama_macbeth(fm_panel, "fwd", fm_x, date_col="date", lags=HORIZON)
    churn = (len(first_tickers - last_tickers) / len(first_tickers)) if first_tickers else 0.0

    _print(clu, "CLUSTERS")
    _print(fac, "FACTORS")
    _print_sector(sector_val, [lb for _feat, lb in VAL_FEATS])
    _print_fm(fm_result)
    out = _render(fac, clu, sector_val, fm_result, churn, n_used, len(universe), px.shape[1], snaps)
    print(f"\nreport -> {out}")


def _print(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} (forward IC @ {HORIZON}td: raw -> beta-neutral) ===")
    print(
        f"{'name':16} {'act':>3} {'IC_raw':>8} {'IC_neu':>8} "
        f"{'t_neu':>6} {'tHAC':>6} {'hit':>5} {'dates':>5}"
    )
    for r in rows:
        act = "" if r["kind"] == "cluster" else ("A" if r["active"] else "·")
        print(
            f"{r['name']:16} {act:>3} {_f(r['ic_raw']):>8} {_f(r['ic_neu']):>8} "
            f"{_f(r['t_neu'], 2):>6} {_f(r['t_neu_hac'], 2):>6} "
            f"{_f(r['hit_neu'], 2):>5} {r['n_dates'] or 0:>5}"
        )


def _print_fm(fm: dict) -> None:
    if not fm:
        print("\n=== FAMA-MACBETH: insufficient data ===")
        return
    print(f"\n=== FAMA-MACBETH (simultaneous premia, HAC t @ {HORIZON}td) ===")
    print(f"{'factor':14} {'coef':>10} {'tHAC':>7} {'dates':>6}")
    for k, v in fm.items():
        print(f"{k:14} {_f(v['coef']):>10} {_f(v['t'], 2):>7} {v['n_dates']:>6}")


def _f(v, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


# Human-readable cluster names + the per-cluster benchmark taxonomy. SCORED members come
# from combine.CLUSTERS (weight = cluster_weight / n_members, equal-mean); the rest are the
# monitored / gate / excluded metrics assigned to the cluster they conceptually belong to.
_CLUSTER_LABEL = {
    "value_z": "Value",
    "quality_z": "Quality",
    "momentum_z": "Momentum",
    "pead_z": "PEAD (earnings surprise)",
    "trajectory_z": "Trajectory (PEF/PET)",
    "lowvol_z": "Low-vol",
    "strength_z": "Strength (analyst revisions)",
    "growth_z": "Growth",
    "_income": "Income (context only)",
    "_investment": "Investment (excluded)",
}
# metric -> (cluster_key, status, note). status: monitor | exclude | gate | scored*.
_NONSCORED = {
    "pe_forward": (
        "value_z",
        "exclude",
        "level ρ0.75 w/ trailing (≈0 IC); its SPREAD is scored via Trajectory",
    ),
    "pb": ("value_z", "scored*", "sector-conditional: financials/REIT value recipe"),
    "ps_sector": ("value_z", "scored*", "sector-conditional: tech/healthcare value recipe"),
    "peg": ("value_z", "monitor", "growth-adjusted value; noisy"),
    "roa": ("quality_z", "exclude", "retired for the interim quality set"),
    "gross_margin": ("quality_z", "exclude", "superseded by GP/assets"),
    "op_margin": ("quality_z", "exclude", "operating-profitability/assets is a BUILD"),
    "current_ratio": ("quality_z", "gate", "distress screen (with D/E) — not scored"),
    "de": ("quality_z", "gate", "distress screen (with current ratio) — not scored"),
    "accruals": ("quality_z", "exclude", "Sloan accruals — no clean alpha"),
    "price_perf": ("momentum_z", "monitor", "≈ mom_12-1 (ρ0.95) — double-count avoided"),
    "short_interest": ("strength_z", "gate", "squeeze EXCLUDE gate (SI > 20%)"),
    "upside": ("strength_z", "monitor", "contaminated ρ−0.72 w/ 52w-high; zero β-alpha (t −0.15)"),
    "buy_pct": ("strength_z", "monitor", "analyst buy-% level ≈ zero IC"),
    "target_dispersion": ("strength_z", "exclude", "live-fetched, non-point-in-time"),
    "asset_growth": ("_investment", "exclude", "CMA / investment — no clean alpha (FM ≈ 0)"),
    "div_yield": ("_income", "monitor", "mildly positive; utilities-tilt candidate"),
}
_STATUS_BADGE = {
    "scored": ("#0a7d33", "#e5f6ea"),
    "scored*": ("#0a7d33", "#e5f6ea"),
    "monitor": ("#916516", "#f8f1df"),
    "gate": ("#1d4ed8", "#e7edfd"),
    "exclude": ("#6b7280", "#eef0f2"),
}


# Plain-language description per metric (owner wants to know what each metric IS).
_METRIC_DESC = {
    "pe_trailing": "price ÷ trailing-12m earnings — cheaper = higher earnings yield",
    "ev_ebitda": "enterprise value ÷ EBITDA — capital-structure-neutral cheapness",
    "pb": "price ÷ book value — Fama-French value anchor (banks / REITs)",
    "ps_sector": "price ÷ sales — value for unprofitable / growth names",
    "peg": "P/E ÷ earnings growth — growth-adjusted value",
    "pe_forward": "price ÷ next-12m expected earnings",
    "earn_trajectory": "forward-P/E ÷ trailing-P/E — <1 = earnings expected to RISE (value-trap guard)",
    "roe": "net income ÷ shareholders' equity — return on equity",
    "roa": "net income ÷ total assets — return on assets",
    "gross_margin": "gross profit ÷ revenue",
    "op_margin": "operating profit ÷ revenue",
    "fcf": "free-cash-flow yield — cash after capex ÷ price",
    "gp_assets": "gross profit ÷ total assets — Novy-Marx profitability (the quality anchor)",
    "current_ratio": "current assets ÷ current liabilities — short-term liquidity",
    "de": "total debt ÷ equity — leverage",
    "accruals": "(net income − operating cash flow) ÷ assets — earnings quality, Sloan (high = bad)",
    "mom_12_1": "12-month return skipping the last month — classic momentum",
    "pct_52w_high": "price ÷ 52-week high — proximity to the highs",
    "price_perf": "recent 12-month price performance",
    "sue": "standardized unexpected earnings — post-earnings-announcement drift (PEAD)",
    "beta": "sensitivity to the market — lower = more defensive",
    "realized_vol": "annualized volatility of daily returns — lower = calmer",
    "analyst_mom": "direction of analyst estimate revisions — up = upgrades",
    "upside": "analyst avg target ÷ price − 1 — implied upside",
    "buy_pct": "% of covering analysts rating Buy",
    "target_dispersion": "spread of analyst price targets — disagreement",
    "short_interest": "% of float sold short — squeeze / borrow risk",
    "earn_growth": "year-over-year earnings growth",
    "rev_growth": "year-over-year revenue growth",
    "asset_growth": "year-over-year total-asset growth — CMA investment factor (high = bad)",
    "div_yield": "dividend ÷ price",
}
# etoro-panel metric -> fundamentals-backtest factor name (the 10yr survivorship-clean stat).
_FUND_STAT = {
    "pe_trailing": "earnings_yield",
    "pb": "book_to_price",
    "ps_sector": "sales_yield",
    "gp_assets": "gp_assets",
    "sue": "sue",
    "asset_growth": "asset_growth",
    "accruals": "accruals",
    # profitability / margins / growth / leverage / cash-flow (derived + Sharadar precomputed)
    "roe": "roe",
    "roa": "roa",
    "gross_margin": "gross_margin",
    "op_margin": "op_margin",
    "earn_growth": "earn_growth",
    "rev_growth": "rev_growth",
    "fcf": "fcf_yield",
    "de": "de",
    "current_ratio": "current_ratio",
    "ev_ebitda": "ev_ebitda",
    # price-derived (monthly marketcap proxy over the 10yr Sharadar DAILY panel)
    "mom_12_1": "mom_12_1",
    "price_perf": "price_perf",
    "pct_52w_high": "pct_52w_high",
    "beta": "beta",
    "realized_vol": "realized_vol",
    "div_yield": "div_yield",
}
# Metrics for which NO 10yr backtest is possible — Sharadar has no analyst estimates,
# forward earnings, short-interest or dividend history. These legitimately stay on the
# ~5-mo etoro git panel (or have no backtest at all); it is NOT a coverage gap.
_NO_10YR = {
    "pe_forward",
    "earn_trajectory",
    "analyst_mom",
    "upside",
    "buy_pct",
    "target_dispersion",
    "peg",
    "short_interest",
}


_PROPOSAL_BADGE = {
    "keep": ("#0a7d33", "#e5f6ea"),  # green — scored + validated, hold
    "promote?": ("#1d4ed8", "#e7edfd"),  # blue — unscored but strong 10yr, candidate
    "earn-in": ("#0e7490", "#e0f2f7"),  # teal — no 10yr possible, accrue on 5mo
    "watch": ("#916516", "#f8f1df"),  # amber — weak / regime-dependent, monitor
    "drop": ("#6b7280", "#eef0f2"),  # grey — confirm exclude, no clean alpha
}
# My data-driven call per metric given the 10yr β-neutral evidence (proposes, does NOT
# impose — weights stay frozen / earn-in per governance). (verdict, one-line rationale).
_PROPOSAL = {
    # quality
    "roe": ("keep", "scored · 10yr t3.5 ✓"),
    "fcf": ("keep", "scored · 10yr t2.8 ✓"),
    "gp_assets": ("keep", "quality anchor · 10yr t3.0 ✓"),
    "roa": ("promote?", "10yr t4.0 (>roe, but ρ-high) — swap/blend with roe"),
    "op_margin": ("promote?", "10yr t2.8, additive — add to quality"),
    "gross_margin": ("drop", "10yr t≈0 — confirm exclude (gp_assets dominates)"),
    "current_ratio": ("keep", "distress gate — keep"),
    "de": ("keep", "leverage gate — keep"),
    "accruals": ("drop", "10yr t-1.9, no clean alpha — confirm exclude"),
    # value
    "pe_trailing": (
        "keep",
        "value regime-weak 10yr — keep at DISCOUNTED weight (durable premium, don't kill)",
    ),
    "ev_ebitda": ("keep", "value regime-weak 10yr — keep discounted vs pe_trailing"),
    "pb": ("keep", "sector recipe (banks / REIT) — hold"),
    "ps_sector": ("keep", "sector recipe (tech / health) — hold"),
    "pe_forward": ("drop", "level ≈0 IC; spread scored via trajectory — keep excluded"),
    "peg": ("drop", "noisy, no 10yr — keep monitor"),
    # momentum
    "mom_12_1": ("keep", "scored · 10yr t2.6 ✓"),
    "pct_52w_high": ("keep", "scored · 10yr t3.8 (top momentum) ✓"),
    "price_perf": ("drop", "ρ0.95 with mom_12-1 — keep monitor (double-count)"),
    # low-vol
    "beta": ("keep", "scored · 10yr t2.2 ✓ (was a broken 25-date read)"),
    "realized_vol": ("keep", "scored · 10yr t3.9 ✓"),
    # strength (analyst / forward — no 10yr possible)
    "analyst_mom": ("earn-in", "scored on 5mo; no 10yr possible — earn-in"),
    "short_interest": ("keep", "squeeze gate — keep"),
    "upside": ("drop", "5mo zero/neg β-alpha — keep monitor, do NOT gate"),
    "buy_pct": ("drop", "5mo ≈0 IC — keep monitor"),
    "target_dispersion": ("drop", "non-PIT, live-fetched — keep excluded"),
    # growth
    "earn_growth": ("keep", "scored · 10yr t2.7 ✓"),
    "rev_growth": ("watch", "10yr t1.7 borderline — monitor"),
    # pead
    "sue": ("keep", "scored · 10yr t2.0 ✓"),
    # trajectory (no 10yr possible)
    "earn_trajectory": ("earn-in", "value-trap gate + 5mo scored — earn-in"),
    # investment / income
    "asset_growth": ("drop", "CMA / investment, FM≈0 — keep excluded"),
    "div_yield": ("watch", "mildly positive; no 10yr — keep monitor"),
}


def _load_fund_stats() -> dict:
    """The 10yr survivorship-clean stats cached by scripts/v3_fundamentals_backtest.py."""
    import json  # noqa: PLC0415

    try:
        with open(os.path.expanduser("~/.weirdapps-trading/v3_fundamentals_stats.json")) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


# Proposed per-metric weights (RAW points — normalized to 100% in the decision table).
# Method: diversified equal-risk, NOT IC-proportional. Correlated profitability metrics share
# a capped sleeve; value is kept at a regime-discounted floor (its weak 10yr is a value-hostile
# regime, not death); analyst/forward sit on a small earn-in sleeve (unvalidatable on 10yr).
# A PROPOSAL to earn into per governance, not an in-sample optimum.
_PROPOSED_WT_RAW = {
    # profitability / quality (strongest 10yr; 5 correlated members → capped sleeve)
    "gp_assets": 6.0,
    "roa": 5.0,
    "roe": 5.0,
    "op_margin": 5.0,
    "fcf": 5.0,
    # momentum
    "pct_52w_high": 7.0,
    "mom_12_1": 6.0,
    # low-vol
    "realized_vol": 7.0,
    "beta": 6.0,
    # value (regime-discounted floor; sector-conditional recipe)
    "pe_trailing": 4.0,
    "pb": 4.0,
    "ps_sector": 4.0,
    "ev_ebitda": 3.0,
    # pead
    "sue": 6.0,
    # growth
    "earn_growth": 5.0,
    # analyst / forward — earn-in (no 10yr possible → kept small)
    "analyst_mom": 5.0,
    "earn_trajectory": 4.0,
}
# Active exclusion screens (no weight — kept as gates, not scored signals).
_GATES = {"current_ratio", "de", "short_interest"}
# metric -> cluster key (for the Cluster column)
_CLUSTER_OF: dict[str, str] = {}
for _cl, _mem in CLUSTERS.items():
    for _m in _mem:
        _CLUSTER_OF[_m] = _cl
for _m, (_cl, _st, _nt) in _NONSCORED.items():
    _CLUSTER_OF.setdefault(_m, _cl)


def _decision_table(fac: list[dict]) -> str:
    """The MAIN view: a flat, per-metric decision table. Proposed weight (diversified
    equal-risk) sorted ↓ for the participating set, then WATCH, GATES, DROP — each row with
    current weight, Δ, and the best-available 10yr / 5mo evidence, so the owner can judge
    whether each weight is right."""
    by = {r["name"]: r for r in fac}
    fund_f = _load_fund_stats().get("factors", {})
    cur: dict[str, float] = {}
    for cl, mem in CLUSTERS.items():
        w = CLUSTER_WEIGHTS.get(cl, 0.0)
        for m in mem:
            cur[m] = 100.0 * w / len(mem)
    tot = sum(_PROPOSED_WT_RAW.values()) or 1.0
    prop = {m: 100.0 * v / tot for m, v in _PROPOSED_WT_RAW.items()}

    def _stat(metric: str):
        fkey = _FUND_STAT.get(metric)
        if fkey and fkey in fund_f:
            s = fund_f[fkey]
            return "10yr", "#0a7d33", s["ic_neu"], s["t_hac"], s["hit"]
        if metric in by:
            r = by[metric]
            win = "5mo·no10yr" if metric in _NO_10YR else "5mo"
            return win, "#916516", r["ic_neu"], r["t_neu_hac"], r["hit_neu"]
        return "—", "#9aa1a9", None, None, None

    def _row(metric: str, pw, cw) -> str:
        cl = _CLUSTER_OF.get(metric, "")
        clabel = _CLUSTER_LABEL.get(cl, cl).split(" ")[0] if cl else ""
        win, wc, icn, t, hit = _stat(metric)
        nc = "#0a7d33" if (icn or 0) > 0 else "#b91c1c"
        tw = "700" if abs(t or 0) >= 2 else "400"
        if isinstance(pw, float) and isinstance(cw, (int, float)):
            d = pw - cw
            dc = "#0a7d33" if d > 0.05 else ("#b91c1c" if d < -0.05 else "#6b7280")
            dtxt = f"<span style='color:{dc}'>{d:+.1f}</span>"
        else:
            dtxt = "<span style='color:#9aa1a9'>—</span>"
        pwt = (
            f"<b>{pw:.1f}%</b>"
            if isinstance(pw, float)
            else f"<span style='color:#6b7280'>{pw}</span>"
        )
        cwt = (
            f"{cw:.1f}%"
            if isinstance(cw, (int, float))
            else f"<span style='color:#9aa1a9'>{cw}</span>"
        )
        why = _PROPOSAL.get(metric, ("", ""))[1]
        return (
            f"<tr><td><code>{metric}</code></td><td style='color:#5b6470'>{clabel}</td>"
            f"<td>{pwt}</td><td>{cwt}</td><td>{dtxt}</td>"
            f"<td class='win' style='color:{wc}'>{win}</td>"
            f"<td style='color:{nc}'>{_f(icn)}</td><td style='font-weight:{tw}'>{_f(t, 2)}</td>"
            f"<td>{_f(hit, 2)}</td><td class='mdesc'>{why}</td></tr>"
        )

    part = sorted(_PROPOSED_WT_RAW, key=lambda m: -prop[m])
    watch = sorted(
        m
        for m, (v, _) in _PROPOSAL.items()
        if v == "watch" and m not in _PROPOSED_WT_RAW and m not in _GATES
    )
    gates = sorted(_GATES)
    drop = sorted(m for m, (v, _) in _PROPOSAL.items() if v == "drop" and m not in _PROPOSED_WT_RAW)

    def _tier(label: str, bg: str) -> str:
        return (
            f"<tr><td colspan='10' style='background:{bg};font-weight:700;padding:6px 9px'>"
            f"{label}</td></tr>"
        )

    # current weight: cluster members carry their live weight; pb/ps_sector are a sector
    # recipe (no fixed number); everything else is currently unscored → 0%.
    _recipe = {"pb", "ps_sector"}

    def _cur_wt(m):
        if m in cur:
            return cur[m]
        return "recipe" if m in _recipe else 0.0

    prows = "".join(_row(m, prop[m], _cur_wt(m)) for m in part)
    wrows = "".join(_row(m, "—", _cur_wt(m)) for m in watch)
    grows = "".join(_row(m, "gate", _cur_wt(m)) for m in gates)
    drows = "".join(_row(m, "—", _cur_wt(m)) for m in drop)
    body = (
        _tier(f"● PARTICIPATE — scored · proposed weight ↓ · Σ = 100% ({len(part)})", "#e5f6ea")
        + prows
        + _tier(f"◐ WATCH — shadow, 0 weight ({len(watch)})", "#f8f1df")
        + wrows
        + _tier(f"▣ GATES — active exclusion screens, no weight ({len(gates)})", "#e7edfd")
        + grows
        + _tier(f"○ DROP — excluded, no clean alpha ({len(drop)})", "#eef0f2")
        + drows
    )
    return (
        "<h2>Decision table — proposed weights vs evidence</h2>"
        '<p class="note">The <b>main</b> view: every metric, flat. <b>Proposed weight</b> = '
        "diversified equal-risk (NOT IC-proportional) — correlated profitability metrics share a "
        "capped sleeve; value is kept at a <b>regime-discounted</b> floor (its weak 10yr is a "
        "value-hostile regime, not death); analyst / forward sit on a small <b>earn-in</b> sleeve "
        "(unvalidatable on 10yr). A proposal to <b>earn into</b> per governance, not an in-sample "
        "optimum. Judge each proposed weight against its 10yr <b>t (HAC)</b> + hit; <b>Δ pp</b> = "
        "proposed − current (— where current is a sector recipe). Bold t = |HAC t| ≥ 2.</p>"
        "<table><thead><tr><th>Metric</th><th>Cluster</th><th>Proposed</th><th>Current</th>"
        "<th>Δ pp</th><th>window</th><th>IC β-neut</th><th>t (HAC)</th><th>hit</th>"
        "<th>Rationale</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _cluster_benchmark(fac: list[dict]) -> str:
    """Cluster-organized parameter benchmark: per cluster, every metric with a plain-language
    description, its weight, and its BEST-available forward-IC stats — the 10yr survivorship-clean
    Sharadar backtest where we have it, else the ~5-mo etoro git panel."""
    by = {r["name"]: r for r in fac}
    fund = _load_fund_stats()
    fund_f = fund.get("factors", {})
    fund_win = str(fund.get("window", "10yr"))

    def _stat_cells(metric: str) -> str:
        fkey = _FUND_STAT.get(metric)
        if fkey and fkey in fund_f:
            s = fund_f[fkey]
            ic_raw, ic_neu, t, hit, win, wc = (
                s["ic_raw"],
                s["ic_neu"],
                s["t_hac"],
                s["hit"],
                "10yr",
                "#0a7d33",
            )
        elif metric in by:
            r = by[metric]
            ic_raw, ic_neu, t, hit, win, wc = (
                r["ic_raw"],
                r["ic_neu"],
                r["t_neu_hac"],
                r["hit_neu"],
                "5mo",
                "#916516",
            )
        else:
            reason = (
                "analyst / forward — no 10yr data possible"
                if metric in _NO_10YR
                else "no backtest data (.info-only)"
            )
            return f"<td class='win'>—</td><td colspan='4' style='color:#9aa1a9'>{reason}</td>"
        rc = "#0a7d33" if (ic_raw or 0) > 0 else "#b91c1c"
        nc = "#0a7d33" if (ic_neu or 0) > 0 else "#b91c1c"
        tw = "700" if abs(t or 0) >= 2 else "400"
        # A 5mo window on an analyst/forward metric is legitimate (no Sharadar 10yr), not a gap.
        win_cell = (
            f"<td class='win' style='color:{wc}'>5mo "
            "<span style='font-size:10px;color:#9aa1a9'>no 10yr</span></td>"
            if (win == "5mo" and metric in _NO_10YR)
            else f"<td class='win' style='color:{wc};font-weight:700'>{win}</td>"
        )
        return (
            win_cell
            + f"<td style='color:{rc}'>{_f(ic_raw)}</td><td style='color:{nc}'>{_f(ic_neu)}</td>"
            f"<td style='font-weight:{tw}'>{_f(t, 2)}</td><td>{_f(hit, 2)}</td>"
        )

    def _prop_cell(metric: str) -> str:
        verdict, why = _PROPOSAL.get(metric, ("", ""))
        if not verdict:
            return "<td></td>"
        fg, bg = _PROPOSAL_BADGE.get(verdict, ("#6b7280", "#eef0f2"))
        return (
            f'<td><span class="b" style="color:{fg};background:{bg}">{verdict}</span>'
            f'<div class="mdesc" style="margin-top:3px">{why}</div></td>'
        )

    def _mrow(metric: str, status: str, weight: str) -> str:
        fg, bg = _STATUS_BADGE.get(status, ("#6b7280", "#eef0f2"))
        badge = f'<span class="b" style="color:{fg};background:{bg}">{status}</span>'
        return (
            f"<tr><td><code>{metric}</code> {badge}</td>"
            f"<td class='mdesc'>{_METRIC_DESC.get(metric, '')}</td>"
            f"<td>{weight}</td>{_stat_cells(metric)}{_prop_cell(metric)}</tr>"
        )

    order = sorted(CLUSTER_WEIGHTS, key=lambda c: -CLUSTER_WEIGHTS[c]) + ["_income", "_investment"]
    blocks = []
    for cl in order:
        members = CLUSTERS.get(cl, [])
        w = CLUSTER_WEIGHTS.get(cl, 0.0)
        n = len(members) or 1
        rows = [
            _mrow(m, "scored", "recipe" if cl == "value_z" else f"{100 * w / n:.1f}%")
            for m in members
        ]
        for m, (mc, status, _note) in _NONSCORED.items():
            if mc == cl:
                rows.append(_mrow(m, status, "recipe" if status == "scored*" else "—"))
        if not rows:
            continue
        wtxt = f" · cluster weight <b>{w * 100:.0f}%</b>" if w else " · not a scored cluster"
        blocks.append(
            f"<h3>{_CLUSTER_LABEL.get(cl, cl)}{wtxt}</h3>"
            "<table><thead><tr><th>Metric</th><th>What it measures</th><th>weight</th>"
            "<th>window</th><th>IC raw</th><th>IC β-neut</th><th>t (HAC)</th><th>hit</th>"
            "<th>Proposal</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    fund_range = fund_win.replace("10yr ", "").strip("()") or "10yr Sharadar"
    return (
        "<h2>Parameter benchmark — by cluster</h2>"
        '<p class="note">Every parameter grouped by its cluster, with a plain-language description, '
        "its weight within conviction (cluster weight ÷ members; Value is sector-conditional → "
        "recipe-weighted), and its <b>best-available</b> forward-IC stats — "
        f"<b style='color:#0a7d33'>10yr</b> = survivorship-clean Sharadar ({fund_range}); "
        "<b style='color:#916516'>5mo</b> = the etoro git panel — now used <b>only</b> for "
        "analyst / forward metrics (analyst revisions, forward P/E, earnings trajectory, short "
        "interest), where Sharadar has no 10yr equivalent. Every durable-fundamental or "
        "price-based parameter (roe, roa, margins, growth, leverage, momentum, vol, beta …) is "
        "now on the 10yr window. "
        "<b>scored</b> = live in conviction · <b>scored*</b> = some sector recipes only · "
        "<b>monitor</b> = shadow, 0 weight · <b>gate</b> = exclusion screen · <b>exclude</b> = dropped. "
        "Bold t = |HAC t| ≥ 2.<br><b>Proposal</b> = my data-driven call given the 10yr evidence "
        "(proposes, does not impose — weights stay frozen / earn-in per governance): "
        "<b style='color:#0a7d33'>keep</b> = scored &amp; validated · "
        "<b style='color:#1d4ed8'>promote?</b> = unscored but strong 10yr, candidate · "
        "<b style='color:#0e7490'>earn-in</b> = no 10yr possible, accrue on 5mo · "
        "<b style='color:#916516'>watch</b> = weak / regime-dependent · "
        "<b style='color:#6b7280'>drop</b> = confirm exclude.</p>" + "".join(blocks)
    )


def _render(fac, clu, sector_val, fm_result, churn, n_snap, n_uni, n_px, snaps) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_factor_backtest.html")

    def rowhtml(r):
        good = (r["ic_neu"] or 0) > 0
        strong = abs(r.get("t_neu_hac") or 0) >= 2  # HAC = the honest significance
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
            f"<td style='color:{col}'>{_f(r['ic_neu'])}</td>"
            f"<td>{_f(r['t_neu'], 2)}</td>"
            f"<td style='font-weight:{tw}'>{_f(r['t_neu_hac'], 2)}</td>"
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

    def fm_html(fm):
        if not fm:
            return ""
        body = ""
        for k, v in fm.items():
            c = "#0a7d33" if (v["coef"] or 0) > 0 else "#b91c1c"
            tw = "700" if abs(v.get("t") or 0) >= 2 else "400"
            body += (
                f"<tr><td><code>{k}</code></td>"
                f"<td style='color:{c}'>{_f(v['coef'])}</td>"
                f"<td style='font-weight:{tw}'>{_f(v['t'], 2)}</td>"
                f"<td>{v['n_dates']}</td></tr>"
            )
        return (
            "<h2>Fama-MacBeth — simultaneous factor premia (HAC t)</h2>"
            "<table><thead><tr><th>Factor</th><th>mean coef</th><th>t (HAC)</th>"
            f"<th>dates</th></tr></thead><tbody>{body}</tbody></table>"
            '<p class="note">Per-date multivariate OLS of the forward return on the core '
            "cluster-z's + beta <b>simultaneously</b> — the correct alternative to sequential "
            "beta-residualization. coef = average per-date slope; t = Newey-West (overlap-robust). "
            "Growth/strength excluded (too sparse for the joint sample).</p>"
        )

    fm_block = fm_html(fm_result)
    pit_note = (
        f'<p class="note"><b>Rigor / PIT:</b> daily panel ({n_snap} usable dates); '
        f"<b>t (HAC)</b> = Newey-West with {HORIZON} lags — the naive t OVERSTATES significance "
        f"under {HORIZON}-day overlapping windows, so read the HAC column. Survivorship: "
        f"{churn:.0%} of the earliest snapshot's names are absent from the latest (ticker churn) "
        "— a look-ahead / survivorship flag for the git-reconstructed panel.</p>"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 Factor Backtest</title>
<style>body{{font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:900px;margin:0 auto;padding:30px}}
h1{{font-size:23px;margin:0 0 4px}}.sub{{color:#5b6470;font-size:13px;margin-bottom:20px}}
h2{{font-size:17px;margin:26px 0 8px;border-bottom:2px solid #1a1a1a;padding-bottom:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f7f8fa;padding:7px 9px;border-bottom:1px solid #e6e8ec}}
td{{padding:6px 9px;border-bottom:1px solid #eef0f2}}code{{background:#f2f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.b{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:20px;margin-left:4px}}.act{{color:#0a7d33;background:#e5f6ea}}.disc{{color:#6b7280;background:#eef0f2}}
.note{{color:#5b6470;font-size:12.5px;margin:6px 0 2px}}
h3{{font-size:14.5px;margin:18px 0 6px;color:#1a1a1a}}.mdesc{{color:#374151;font-size:12px;max-width:360px}}.win{{font-size:11px;text-align:center}}</style></head><body>
<h1>v3 Panel-History Factor Backtest</h1>
<div class="sub">Forward IC (Spearman) at {HORIZON}td, over {n_snap} daily point-in-time snapshots
({snaps[0][1]} .. {snaps[-1][1]}) reconstructed from etoro.csv git history · universe {n_uni:,} global coverage names, {n_px:,} priced (EUR).</div>
<p class="note"><b>How to read:</b> <b>IC raw</b> = Spearman(factor-z, raw forward return); <b>IC β-neutral</b> = vs the beta-residualised return (per-date OLS residual on beta) — this removes the market-beta component that dominates a trending regime and is the <b>honest test of factor alpha</b>. Positive β-neutral IC with |t|≥2 (bold) = evidence of alpha on this (short) sample. <b>active</b> = in a live cluster; <b>discarded</b> = removed from scoring (shown to validate the discard). ~{n_snap} daily dates — indicative, NOT the deflated-Sharpe gate.</p>
{pit_note}
{_decision_table(fac)}
{_cluster_benchmark(fac)}
<h2>Clusters</h2>
<table><thead><tr><th>Cluster</th><th>IC raw</th><th>IC β-neutral</th><th>t (naive)</th><th>t (HAC)</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{ch}</tbody></table>
<h2>Factors (active + discarded)</h2>
<table><thead><tr><th>Factor</th><th>IC raw</th><th>IC β-neutral</th><th>t (naive)</th><th>t (HAC)</th><th>hit</th><th>dates</th><th>obs</th></tr></thead><tbody>{fh}</tbody></table>
<p class="note">β-neutral = cross-sectional residual of the forward return on beta (isolates factor alpha from the market/regime). Excludes ev_ebitda + rev_growth (yfinance .info is current-only, not point-in-time reconstructable). mom_12_1 / realized_vol reconstructed from adjusted prices.</p>
{fm_block}
{sec_block}
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    main()

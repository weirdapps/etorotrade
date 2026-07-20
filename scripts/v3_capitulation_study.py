"""Study: does an 'analyst capitulation' conjunction predict UNDERperformance?

Tests option B — a capitulation gate that would block a NEW BUY when the analyst
consensus is uniformly bearish: ``upside < threshold`` AND ``analyst_mom < 0``.

For the gate to earn its place it must show that, AMONG would-be buys (top-conviction
names), capitulation names underperform the rest. We reconstruct the etoro.csv panel
from git history (PIT), compute a panel-native conviction proxy + beta-neutral forward
returns, and report — per upside threshold — the buy-zone (top 15% conviction) forward-
return difference (capitulation - rest) with a HAC t-stat, plus the pooled unconditional
difference. Research only; prints a table, writes nothing to production.

    .venv/bin/python scripts/v3_capitulation_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from v3_factor_backtest import git_snapshots, load_panel_at  # noqa: E402

from trade_modules.v3.combine import _rank_z  # noqa: E402
from trade_modules.v3.factor_backtest import (  # noqa: E402
    beta_neutralize,
    forward_return_at,
    hac_tstat,
)
from trade_modules.v3.features import _num  # noqa: E402
from trade_modules.v3.prices import load_eur_close  # noqa: E402
from trade_modules.v3.universe import load_universe  # noqa: E402

ETORO = "yahoofinance/output/etoro.csv"
HORIZON = 21
BUY_PCTILE = 0.85  # top 15% conviction = would-be buys (overlay buy_pctile)
THRESHOLDS = [-5.0, -10.0, -15.0]  # upside % floors to test for the capitulation flag
# Panel-native conviction proxy (value/quality/momentum-proxy/low-vol), directional.
CONV = {"PET": -1, "ROE": +1, "FCF": +1, "52W": +1, "EG": +1, "B": -1}


def _conviction(panel: pd.DataFrame) -> pd.Series:
    zs = [_rank_z(_num(panel[c])) * d for c, d in CONV.items() if c in panel.columns]
    return pd.concat(zs, axis=1).mean(axis=1, skipna=True) if zs else pd.Series(dtype=float)


def _mean(x: list[float]) -> float:
    s = pd.Series(x, dtype=float)
    return float(s.mean()) if len(s) else float("nan")


def main() -> None:
    snaps = git_snapshots(ETORO, weekly=False)
    universe = set(load_universe(ETORO, min_factor_coverage=6))
    px = load_eur_close(sorted(universe), period="1y")
    pidx = px.index
    print(f"snapshots: {len(snaps)}  universe: {len(universe)}  priced: {px.shape[1]}")

    per_date_diff: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    pooled_cap: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    pooled_non: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    n_flag_buyzone: dict[float, int] = dict.fromkeys(THRESHOLDS, 0)
    n_buyzone = 0
    n_dates = 0

    for commit, date in snaps:
        panel = load_panel_at(commit, ETORO)
        if panel is None:
            continue
        panel = panel[panel.index.astype(str).isin(universe)]
        if panel.empty or "UP%" not in panel.columns or "AM" not in panel.columns:
            continue
        mask = pidx <= pd.Timestamp(date)
        if not mask.any():
            continue
        fwd = forward_return_at(px, pidx[mask].max(), HORIZON)
        if fwd.empty:
            continue
        beta = _num(panel["B"]).reindex(fwd.index) if "B" in panel.columns else None
        fwd_n = beta_neutralize(fwd, beta) if beta is not None else fwd

        idx = panel.index
        fn = fwd_n.reindex(idx)
        up = _num(panel["UP%"]).reindex(idx)
        am = _num(panel["AM"]).reindex(idx)
        conv = _conviction(panel)
        if conv.dropna().empty:
            continue
        top = (conv >= conv.quantile(BUY_PCTILE)) & fn.notna()
        n_dates += 1
        n_buyzone += int(top.sum())

        for thr in THRESHOLDS:
            cap = (up < thr) & (am < 0)
            bz_cap = top & cap
            bz_non = top & ~cap
            n_flag_buyzone[thr] += int(bz_cap.sum())
            if bz_cap.sum() >= 1 and bz_non.sum() >= 1:
                per_date_diff[thr].append(float(fn[bz_cap].mean() - fn[bz_non].mean()))
            pooled_cap[thr].extend(fn[cap & fn.notna()].tolist())
            pooled_non[thr].extend(fn[(~cap) & fn.notna()].tolist())

    print(f"\ndates used: {n_dates}   avg buy-zone names/date: {n_buyzone / max(n_dates, 1):.0f}")
    print("\n=== CAPITULATION GATE STUDY (beta-neutral forward return @ 21td) ===")
    print("(a diff < 0 with |t| > 2 would justify the gate: capitulation names underperform)\n")
    hdr = (
        f"{'upside<':>8}{'BUY-ZONE diff':>15}{'HAC t':>8}{'dates':>7}{'flagged':>9}"
        f"{'POOLED cap':>12}{'pooled non':>12}{'n_cap':>7}"
    )
    print(hdr)
    for thr in THRESHOLDS:
        diffs = pd.Series(per_date_diff[thr], dtype=float)
        t = hac_tstat(diffs, HORIZON) if len(diffs) > 2 else float("nan")
        cap_m, non_m = _mean(pooled_cap[thr]), _mean(pooled_non[thr])
        print(
            f"{thr:>7.0f}%{_mean(per_date_diff[thr]) * 100:>14.2f}%{t:>8.2f}"
            f"{len(diffs):>7}{n_flag_buyzone[thr]:>9}"
            f"{cap_m * 100:>11.2f}%{non_m * 100:>11.2f}%{len(pooled_cap[thr]):>7}"
        )


if __name__ == "__main__":
    main()

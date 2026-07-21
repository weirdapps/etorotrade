"""Redundancy / incremental-IC analysis on the 10yr survivorship-clean z-panel.

Answers the owner's questions with numbers, not opinion:
  - Are gp_assets ≈ roa ≈ roe ≈ op_margin (profitability redundancy)?
  - Is mom_12_1 ≈ price_perf ≈ pct_52w_high (momentum redundancy)?
  - Is beta ≈ realized_vol (low-vol redundancy)?
  - Does each group add incremental IC beyond its best single member, or is combining
    just double-counting the same bet?
  - Is realized_vol a distinct alpha (survives controlling for the others) or a proxy?

Reads the signed-z panel dumped by scripts/v3_fundamentals_backtest.py
(~/.weirdapps-trading/v3_factor_zpanel.parquet: ticker × date × signed-z per factor + fwd).
All z are already sign-applied (high = good), so within a same-direction group a high
positive correlation = genuine redundancy.

    .venv/bin/python scripts/v3_factor_redundancy.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.validation.xsection_ic import cross_sectional_ic  # noqa: E402

ZP = os.path.expanduser("~/.weirdapps-trading/v3_factor_zpanel.parquet")
OUT = os.path.expanduser("~/.weirdapps-trading/v3_redundancy.json")

# The owner's groups, using the z-panel's fundamentals-backtest factor names.
GROUPS = {
    "profitability": ["gp_assets", "roa", "roe", "op_margin", "fcf_yield"],
    "momentum": ["mom_12_1", "price_perf", "pct_52w_high"],
    "low_vol": ["beta", "realized_vol"],
    "value": ["earnings_yield", "book_to_price", "sales_yield", "ev_ebitda"],
    "growth_income": ["earn_growth", "rev_growth", "div_yield"],
}


def _ic(df: pd.DataFrame, x: str) -> float:
    sub = df.dropna(subset=[x, "fwd"])
    if len(sub) < 50:
        return float("nan")
    r = cross_sectional_ic(sub, x, "fwd", date_col="date")
    return float(r.get("mean_ic") or float("nan"))


def _mean_corr(z: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Time-averaged cross-sectional correlation of the signed z's."""
    acc, n = None, 0
    for _, g in z.groupby("date"):
        c = g[factors].corr()
        if c.notna().to_numpy().any():
            acc = c if acc is None else acc.add(c, fill_value=0.0)
            n += 1
    return (acc / n) if acc is not None and n else pd.DataFrame(index=factors, columns=factors)


def main() -> None:
    if not Path(ZP).exists():
        print(f"no z-panel at {ZP} — run scripts/v3_fundamentals_backtest.py first")
        return
    z = pd.read_parquet(ZP)
    factors = [c for c in z.columns if c not in ("ticker", "fwd", "date")]
    print(f"z-panel: {len(z):,} rows, {z['date'].nunique()} dates, {len(factors)} factors\n")

    result: dict = {"groups": {}}
    for gname, members in GROUPS.items():
        mem = [m for m in members if m in factors]
        if len(mem) < 2:
            continue
        corr = _mean_corr(z, mem)
        singles = {m: _ic(z, m) for m in mem}
        z["_combo"] = z[mem].mean(axis=1, skipna=True)
        combo_ic = _ic(z, "_combo")
        best = max(singles, key=lambda m: singles[m] if singles[m] == singles[m] else -9)
        # incremental IC of the combo over the best single member
        uplift = combo_ic - singles[best]

        print(f"=== {gname.upper()} ===")
        print("mean cross-sectional correlation:")
        print(corr.round(2).to_string())
        print("single-factor IC:", {m: round(v, 4) for m, v in singles.items()})
        print(
            f"combo (equal-wt) IC {combo_ic:.4f} vs best single {best} {singles[best]:.4f} "
            f"→ uplift {uplift:+.4f}"
        )
        # off-diagonal average correlation (redundancy summary)
        offdiag = corr.where(~np.eye(len(mem), dtype=bool)).stack()
        print(f"avg off-diagonal corr: {offdiag.mean():.2f} (>0.7 = strongly redundant)\n")

        result["groups"][gname] = {
            "members": mem,
            "corr": {a: {b: _rnd(corr.loc[a, b]) for b in mem} for a in mem},
            "single_ic": {m: _rnd(v) for m, v in singles.items()},
            "combo_ic": _rnd(combo_ic),
            "best_single": best,
            "uplift_vs_best": _rnd(uplift),
            "avg_offdiag_corr": _rnd(offdiag.mean()),
        }

    # cross-group: does low_vol overlap momentum/quality? (realized_vol distinctness)
    key = [
        "realized_vol",
        "beta",
        "gp_assets",
        "roa",
        "pct_52w_high",
        "mom_12_1",
        "earnings_yield",
        "sue",
        "earn_growth",
        "div_yield",
        "net_issuance",
    ]
    key = [k for k in key if k in factors]
    gcorr = _mean_corr(z, key)
    print("=== CROSS-FACTOR correlation (distinctness check) ===")
    print(gcorr.round(2).to_string())
    result["cross_corr"] = {a: {b: _rnd(gcorr.loc[a, b]) for b in key} for a in key}

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved -> {OUT}")


def _rnd(v) -> float | None:
    try:
        f = float(v)
        return round(f, 4) if f == f else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()

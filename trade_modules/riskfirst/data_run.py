"""Data-connected shadow run — the price-history upgrade of shadow_run.

Pipeline:
  1. snapshot pre-screen the eligible universe (fast) -> top candidates + current book
  2. fetch 2y price history + sectors for that bounded candidate set (network)
  3. build EMPIRICAL shrunk covariance + TRUE 12-1 momentum + realized-vol low-vol
     (replacing the beta/52W proxies) and attach SECTOR labels (activates sector cap)
  4. construct the risk-first book, recommend vs current, run the edge gate

Still SHADOW: the FULL composite cannot be point-in-time backtested (no historical
fundamentals), so the edge gate stays FAIL until a forward track record accrues.
"""

from __future__ import annotations

import datetime
import json
import os

from .edgegate import gate_verdict
from .engine import composite_score, eligible_universe, recommend, select_and_construct
from .factors import quality, size, value
from .prices import (
    daily_returns,
    fetch_prices,
    fetch_sectors,
    price_lowvol_factor,
    price_momentum_factor,
    shrunk_cov,
)
from .shadow_run import DEFAULT_PORTFOLIO, DEFAULT_UNIVERSE, load_current_weights, load_universe

SNAPSHOT_FACTORS = [value.compute, quality.compute, size.compute]


def build_candidates(universe_path, portfolio_path, prescreen_n=40):
    df = eligible_universe(load_universe(universe_path))
    comp = composite_score(df, SNAPSHOT_FACTORS)
    top = list(comp.dropna().sort_values(ascending=False).index[:prescreen_n])
    current = load_current_weights(portfolio_path)
    cands = list(dict.fromkeys(top + list(current.index)))
    cand_df = df.loc[[c for c in cands if c in df.index]].copy()
    return cand_df, current


def run_data(
    universe_path: str = DEFAULT_UNIVERSE,
    portfolio_path: str = DEFAULT_PORTFOLIO,
    *,
    prescreen_n: int = 40,
    top_n: int = 20,
    period: str = "2y",
    name_cap: float = 0.08,
    usd_bloc_cap: float = 0.60,
    sector_cap: float = 0.25,
    target_vol: float = 0.12,
) -> dict:
    cand_df, current = build_candidates(universe_path, portfolio_path, prescreen_n)
    print(f"candidates: {len(cand_df)} (pre-screen {prescreen_n} + current book)")

    prices = fetch_prices(cand_df.index.tolist(), period=period)
    priced = [t for t in cand_df.index if t in prices.columns and prices[t].notna().sum() > 260]
    print(f"priced (>=260 obs): {len(priced)} / {len(cand_df)}")
    cand_df = cand_df.loc[priced]
    prices = prices[priced]

    sectors = fetch_sectors(priced)
    cand_df["SECTOR"] = [sectors.get(t) or "UNKNOWN" for t in cand_df.index]
    n_sect = len({s for s in cand_df["SECTOR"] if s != "UNKNOWN"})
    print(
        f"sectors resolved: {sum(1 for t in priced if sectors.get(t))} / {len(priced)} ({n_sect} distinct)"
    )

    factors = [
        value.compute,
        quality.compute,
        price_momentum_factor(prices),
        price_lowvol_factor(prices),
        size.compute,
    ]

    def cov_fn(selected):
        return shrunk_cov(daily_returns(prices[list(selected)]))

    built = select_and_construct(
        cand_df,
        factors,
        top_n=top_n,
        cov_fn=cov_fn,
        name_cap=name_cap,
        usd_bloc_cap=usd_bloc_cap,
        sector_cap=sector_cap,
        target_vol=target_vol,
    )
    target = built["weights"][built["weights"] > 1e-9]
    recs = recommend(target, current)
    verdict = gate_verdict(sr=0.0, n_obs=0, n_trials=1, var_sr=0.02, n_regimes=1)

    return {
        "mode": "SHADOW (price-history)",
        "selected": built["selected"],
        "target_weights": target,
        "gross": built["gross"],
        "cash": built["cash"],
        "usd_bloc": built["usd_bloc"],
        "sectors": {t: cand_df.loc[t, "SECTOR"] for t in target.index},
        "recommendations": recs,
        "edge_gate": verdict,
        "promotable": verdict["passed"],
    }


def main(argv=None) -> int:  # pragma: no cover - network integration entry point
    res = run_data()
    tw = res["target_weights"].sort_values(ascending=False)
    print("\n=== riskfirst DATA (price-history) shadow book ===")
    print(f"gross {res['gross']:.1%} | cash {res['cash']:.1%} | USD-bloc {res['usd_bloc']:.1%}")
    for t, w in tw.items():
        print(f"  {t:12s} {w:6.2%}  [{res['sectors'].get(t, '?')}]")
    print(
        f"\nEdge gate: {'PASS' if res['edge_gate']['passed'] else 'FAIL'} "
        f"(DSR {res['edge_gate']['dsr']:.3f}) | promotable: {res['promotable']}"
    )

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    path = os.path.expanduser(f"~/Downloads/{ts}_riskfirst_data_shadow.json")
    with open(path, "w") as f:
        json.dump(
            {
                "mode": res["mode"],
                "gross": res["gross"],
                "cash": res["cash"],
                "usd_bloc": res["usd_bloc"],
                "sectors": res["sectors"],
                "target_weights": res["target_weights"].round(4).to_dict(),
                "recommendations": res["recommendations"].to_dict(orient="records"),
                "edge_gate": res["edge_gate"],
                "promotable": res["promotable"],
            },
            f,
            indent=2,
        )
    print(f"JSON: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

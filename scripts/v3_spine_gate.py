# scripts/v3_spine_gate.py
"""Phase-1 decisive gate: does the price spine (12-1 momentum + low-vol) show
positive OOS cross-sectional IC + pass the DSR gate, on the USD sub-universe, in EUR?
Run: .venv/bin/python scripts/v3_spine_gate.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from trade_modules.v3.labels import forward_returns
from trade_modules.v3.prices import load_eur_close
from trade_modules.v3.spine import spine_scores
from trade_modules.v3.universe import load_universe
from trade_modules.v3.validate_spine import run_gate

ETORO_CSV = "yahoofinance/output/etoro.csv"
HORIZONS = [5, 21, 63]


def month_end_rebalances(index) -> list:
    s = pd.Series(index=index, data=1)
    return list(s.resample("ME").last().dropna().index.intersection(index))


def main() -> None:
    tickers = load_universe(ETORO_CSV)
    print(f"USD universe: {len(tickers)} names")
    eur = load_eur_close(tickers, period="2y")
    eur = eur.dropna(axis=1, thresh=int(0.6 * len(eur)))  # keep names with ≥60% history
    print(f"priced (EUR) names: {eur.shape[1]}, bars: {eur.shape[0]}")

    rebal = [d for d in month_end_rebalances(eur.index)]
    rebal = [d for d in rebal if eur.index.get_loc(d) >= 252]  # momentum warmup
    print(f"rebalance dates (warmed up): {len(rebal)}")

    scores = spine_scores(eur, rebal)
    fwd = forward_returns(eur, rebal, HORIZONS)
    verdict = run_gate(scores, fwd, HORIZONS, n_trials=2, min_obs=10)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_spine_gate.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "ic": verdict["ic"],
                "ic_decay": verdict["ic_decay"],
                "primary_ic_pass": verdict["primary_ic_pass"],
                "dsr_pass": verdict["dsr_pass"],
                "gate_pass": verdict["gate_pass"],
                "harness_overall": verdict["harness"].get("overall", {}),
            },
            fh,
            indent=2,
            default=str,
        )

    print("\n=== PRICE-SPINE GATE VERDICT ===")
    for h in HORIZONS:
        s = verdict["ic"][h]
        print(
            f"  h={h:>3}d  IC={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  hit={s['hit_rate']:.0%}  n={s['n']}"
        )
    print(
        f"  primary_ic_pass={verdict['primary_ic_pass']}  dsr_pass={verdict['dsr_pass']}  "
        f"GATE_PASS={verdict['gate_pass']}"
    )
    print(f"  report: {out}")


if __name__ == "__main__":
    main()

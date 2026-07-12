# scripts/v3_report.py
"""Trading Model v3 — Factor Report driver.

Universe = unique tickers from portfolio.csv + buy.csv. Enriches features,
computes cluster/conviction scores, classifies the market regime from a
fetched index series, renders the house-Swiss HTML report and writes it to
~/Downloads/<UTCstamp>_v3_factor_report.html.

Run (on the VPS / where network is allowed):
    .venv/bin/python scripts/v3_report.py

This performs LIVE network fetches (yfinance prices + .info). Do not run in
an environment where live fetches are prohibited.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.combine import compute_scores  # noqa: E402
from trade_modules.v3.features import enrich_features  # noqa: E402
from trade_modules.v3.fetch import robust_fetch_prices  # noqa: E402
from trade_modules.v3.report import compute_regime, render_report  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"


def _read_tickers(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, na_values=["--"])
        return df["TKR"].dropna().astype(str).tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not read {path}: {exc}", file=sys.stderr)
        return []


def _system_read(scores: pd.DataFrame, regime: str) -> str:
    port = scores[scores.get("is_portfolio", False) == True]  # noqa: E712
    adds = int((port["conviction"] > 0.5).sum())
    trims = int((port["conviction"] < -0.5).sum())
    cand = scores[scores.get("is_portfolio", True) == False]  # noqa: E712
    watch = int((cand["conviction"] > 0.5).sum())
    return (
        f"Regime {regime}: {adds} portfolio ADD / {trims} TRIM signals, "
        f"{watch} candidate(s) clearing the buy-watch bar. Shadow snapshot — "
        f"weigh full-cluster conviction, not single factors."
    )


def main() -> None:
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    n_port = len(port_set)
    n_cand = len(set(buy) - port_set)
    print(f"universe: {len(universe)} tickers ({n_port} portfolio + {n_cand} candidates)")

    feats = enrich_features(universe, ETORO_CSV, price_period="2y")
    priced = int(feats["mom_12_1"].notna().sum())
    enriched = int(feats["pb"].notna().sum())

    scores = compute_scores(feats, sector_neutral=True)
    scores["is_portfolio"] = scores.index.isin(port_set)

    # Market regime from a fetched index series (S&P 500).
    regime, regime_detail = "NEUTRAL", ""
    try:
        spx = robust_fetch_prices(["^GSPC"], period="2y")
        series = spx.iloc[:, 0] if spx is not None and not spx.empty else pd.Series(dtype=float)
        regime, regime_detail = compute_regime(series)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: regime fetch failed ({exc}); defaulting to NEUTRAL", file=sys.stderr)

    now = datetime.now(timezone.utc)
    meta = {
        "date": now.strftime("%Y-%m-%d"),
        "n_portfolio": n_port,
        "n_candidates": n_cand,
        "regime": regime,
        "regime_detail": regime_detail,
        "system_read": _system_read(scores, regime),
        "priced": priced,
        "enriched": enriched,
        "generated_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
    }

    html = render_report(scores, meta)
    stamp = now.strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_factor_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    top5 = scores.sort_values("conviction", ascending=False).head(5)
    print(out)
    print(f"universe size:      {len(universe)}")
    print(f"priced (mom/vol):   {priced}")
    print(f"enriched (.info):   {enriched}")
    print("top 5 by conviction:")
    for tkr, row in top5.iterrows():
        rank = row.get("rank")
        rank_s = "—" if pd.isna(rank) else int(rank)
        print(f"  {tkr:<8} conv {row['conviction']:+.2f}  rank {rank_s}")


if __name__ == "__main__":
    main()

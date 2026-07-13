"""Trading Model v3 — Portfolio Construction runner.

Universe = unique tickers from portfolio.csv + buy.csv.  Fetches prices,
enriches features, scores, detects the market regime from ^GSPC, constructs a
risk-first target book via build_portfolio, and writes a JSON artefact to
~/Downloads/<UTCstamp>_v3_portfolio.json.

Run (on the VPS / where network is allowed):
    .venv/bin/python scripts/v3_portfolio.py

Mirrors the universe-assembly and price-fetch sequence in v3_report.py exactly
(same CSV paths, same price_period="2y", same enrich_features / compute_scores
call). Does NOT refactor or modify v3_report.py.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.riskfirst.regime_overlay import REGIME_EXPOSURE  # noqa: E402
from trade_modules.v3.combine import compute_scores  # noqa: E402
from trade_modules.v3.conditioning import resolve_deployment  # noqa: E402
from trade_modules.v3.construct import build_portfolio  # noqa: E402
from trade_modules.v3.features import enrich_features  # noqa: E402
from trade_modules.v3.fetch import robust_fetch_prices  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"

# DEPLOYMENT_BY_REGIME is now the single source of truth in
# trade_modules.v3.conditioning — imported above.


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested with synthetic data)
# ---------------------------------------------------------------------------


def trend_regime(spx_close: pd.Series) -> tuple[str, float]:
    """Classify the market regime from a daily S&P 500 close series.

    Args:
        spx_close: Daily closing prices (dates as index, values as floats).
            Any length; a short series (<200 observations) falls back to
            "neutral".

    Returns:
        ``(regime, multiplier)`` where:
        - ``regime`` is one of ``"risk_on"``, ``"neutral"``, ``"risk_off"``.
        - ``multiplier`` is the matching :data:`REGIME_EXPOSURE` float.

    Rules:
        - ``risk_on``  if last close > 200-day SMA **and** 50-day SMA > 200-day SMA.
        - ``risk_off`` if last close < 200-day SMA.
        - ``neutral``  otherwise (last >= 200-day SMA but 50-day SMA <= 200-day SMA).
        - Short series (<200 obs) → ``neutral`` unconditionally.
    """
    s = pd.to_numeric(spx_close, errors="coerce").dropna()
    if len(s) < 200:
        return "neutral", float(REGIME_EXPOSURE["neutral"])

    sma200 = float(s.rolling(200).mean().iloc[-1])
    sma50 = float(s.rolling(50).mean().iloc[-1])
    last = float(s.iloc[-1])

    if last > sma200 and sma50 > sma200:
        regime = "risk_on"
    elif last < sma200:
        regime = "risk_off"
    else:
        regime = "neutral"

    return regime, float(REGIME_EXPOSURE[regime])


def build_target_rows(scored: pd.DataFrame, result: dict) -> list[dict]:
    """Build a sorted list of target-holding dicts from scored + build_portfolio output.

    Filters to names with ``result["weights"] > 0``, then emits one row per
    name with the keys required for the JSON artefact and console table.  Rows
    are sorted by ``target_pct`` descending (largest allocation first).

    Args:
        scored: Output of :func:`trade_modules.v3.combine.compute_scores` (with
            enriched features columns already merged in).
        result: Return value of :func:`trade_modules.v3.construct.build_portfolio`.

    Returns:
        List of dicts, each with keys:
        ``{ticker, name, sector, conviction, rank, target_pct, price,
        stop_loss, take_profit}``.
    """
    weights: pd.Series = result["weights"]
    invested = weights[weights > 0]
    if invested.empty:
        return []

    def _get(row, col):
        val = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    rows = []
    for ticker, w in invested.items():
        if ticker not in scored.index:
            continue
        row = scored.loc[ticker]
        rows.append(
            {
                "ticker": ticker,
                "name": _get(row, "name"),
                "sector": _get(row, "sector"),
                "conviction": _get(row, "conviction"),
                "rank": _get(row, "rank"),
                "target_pct": float(w),
                "price": _get(row, "price"),
                "stop_loss": _get(row, "stop_loss"),
                "take_profit": _get(row, "take_profit"),
            }
        )

    rows.sort(key=lambda r: r["target_pct"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# IO wrapper (smoke-tested by VPS run, not unit tests)
# ---------------------------------------------------------------------------


def _read_tickers(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, na_values=["--"])
        return df["TKR"].dropna().astype(str).tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not read {path}: {exc}", file=sys.stderr)
        return []


def main() -> None:
    # --- Universe assembly (mirrors v3_report.py exactly) ---
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + "
        f"{len(set(buy) - port_set)} candidates)"
    )

    # --- Feature enrichment (mirrors v3_report.py exactly) ---
    feats = enrich_features(
        universe, ETORO_CSV, price_period="2y", accruals_fetch=lambda _tickers: {}
    )
    priced = int(feats["mom_12_1"].notna().sum())

    # --- Scoring (mirrors v3_report.py exactly) ---
    scores = compute_scores(feats, sector_neutral=True)
    scores["is_portfolio"] = scores.index.isin(port_set)

    elig = scores.get("eligible", pd.Series(True, index=scores.index)).fillna(False).astype(bool)
    n_eligible = int(elig.sum())

    # --- Price fetch for covariance + regime (mirrors v3_report.py SPX fetch) ---
    prices: pd.DataFrame = pd.DataFrame()
    spx_close: pd.Series = pd.Series(dtype=float)
    try:
        prices = robust_fetch_prices(universe, period="2y")
    except Exception as exc:  # noqa: BLE001
        print(f"warn: universe price fetch failed ({exc})", file=sys.stderr)

    try:
        spx_raw = robust_fetch_prices(["^GSPC"], period="2y")
        if spx_raw is not None and not spx_raw.empty:
            spx_close = spx_raw.iloc[:, 0]
    except Exception as exc:  # noqa: BLE001
        print(f"warn: ^GSPC fetch failed ({exc}); defaulting to neutral", file=sys.stderr)

    regime, mult = trend_regime(spx_close)
    # Polymarket is a shadow/zero-conviction cross-repo signal (lives in trading-hub).
    # The seam is inert until wired and validated; max_pm_tilt=0 (default) guarantees
    # zero effect on deployment regardless of what polymarket_signal contains.
    gross_target, dial_diag = resolve_deployment(regime, polymarket_signal=None)
    print(f"regime: {regime}  multiplier: {mult:.2f}  deployment: {gross_target:.0%}")

    # --- Portfolio construction ---
    result = build_portfolio(
        scores,
        prices,
        top_n=20,
        target_vol=0.12,
        name_cap=0.08,
        sector_cap=0.25,
        usd_bloc_cap=0.60,
        gross_target=gross_target,
    )

    gross = result["gross"]
    cash = result["cash"]
    usd_bloc = result["usd_bloc"]
    diag = result["diagnostics"]
    sector_exp = result["sector_exposures"]

    # --- Console summary ---
    print()
    print(f"{'─' * 60}")
    print("PORTFOLIO DIAGNOSTICS")
    print(f"{'─' * 60}")
    print(f"  gross deployed:    {gross:.1%}")
    print(f"  cash:              {cash:.1%}")
    print(f"  USD-bloc:          {usd_bloc:.1%}")
    print(f"  CVaR-95 risk book: {diag['cvar_95_risk_book']:.1%}")
    print(f"  CVaR-95 deployed:  {diag['cvar_95_deployed']:.1%}")
    print(f"  net beta:          {diag['net_beta']:.2f}")
    print(f"  effective bets:    {diag['effective_bets']:.1f}")
    binding = diag["binding"]
    flags = [k for k, v in binding.items() if v]
    print(f"  binding flags:     {flags if flags else 'none'}")
    print()
    print("  Sector exposures:")
    for sec, w in sorted(sector_exp.items(), key=lambda x: -x[1]):
        print(f"    {sec:<22} {w:.1%}")

    holdings = build_target_rows(scores, result)
    print()
    print(f"{'─' * 60}")
    print(f"TARGET HOLDINGS  ({len(holdings)} names)")
    print(f"{'─' * 60}")
    hdr = (
        f"{'Ticker':<10} {'Pct':>6}  {'Conv':>6}  {'Rank':>5}  {'Sector':<18}  {'SL':>8}  {'TP':>8}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for h in holdings:
        sl = f"{h['stop_loss']:.2f}" if h["stop_loss"] is not None else "  n/a"
        tp = f"{h['take_profit']:.2f}" if h["take_profit"] is not None else "  n/a"
        conv = f"{h['conviction']:+.2f}" if h["conviction"] is not None else "  n/a"
        rank = str(int(h["rank"])) if h["rank"] is not None else "n/a"
        sector = str(h["sector"] or "")[:18]
        print(
            f"{h['ticker']:<10} {h['target_pct']:>6.1%}  {conv:>6}  {rank:>5}  {sector:<18}  {sl:>8}  {tp:>8}"
        )

    # --- JSON output ---
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M")

    def _serial(v):
        """JSON-serialise pandas NA / numpy types."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v

    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "universe_size": len(universe),
        "priced": priced,
        "eligible": n_eligible,
        "regime": regime,
        "multiplier": mult,
        "gross_target": gross_target,
        "conditioning": dial_diag,
        "diagnostics": {k: (_serial(v) if not isinstance(v, dict) else v) for k, v in diag.items()},
        "gross": gross,
        "cash": cash,
        "holdings": [{k: _serial(v) for k, v in h.items()} for h in holdings],
    }

    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_portfolio.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print(f"JSON written → {out}")


if __name__ == "__main__":
    main()

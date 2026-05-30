"""EUR/USD (and other-currency) exposure + hedge monitor for the EUR-home book.

MONITOR ONLY — recommends, never trades. Buckets holding weights by currency
(reusing fx_sizing.currency_for_ticker), reports the USD bloc (USD + USD-pegged
HKD), and the EURUSD hedge implied at several ratios — expressed as % of equity
(dollar equity is not available at committee-report time). Execution is a manual
EURUSD forex position on eToro once a ratio is chosen.
"""

from __future__ import annotations

from typing import Any

from trade_modules.fx_sizing import currency_for_ticker

USD_PEGGED = {"USD", "HKD"}  # HKD tracks USD -> same EUR/USD hedge bloc


def compute_currency_exposure(weights: dict[str, float]) -> dict[str, Any]:
    """Bucket holding weights (percent) by currency.

    Returns {by_currency: [{currency, weight_pct} sorted desc], usd_bloc_pct,
    non_eur_pct}. usd_bloc_pct = USD + HKD (USD-pegged).
    """
    buckets: dict[str, float] = {}
    for tkr, w in weights.items():
        ccy = currency_for_ticker(tkr)
        buckets[ccy] = buckets.get(ccy, 0.0) + float(w)
    by_currency = [
        {"currency": c, "weight_pct": round(v, 2)}
        for c, v in sorted(buckets.items(), key=lambda kv: -kv[1])
    ]
    usd_bloc = sum(v for c, v in buckets.items() if c in USD_PEGGED)
    non_eur = sum(v for c, v in buckets.items() if c != "EUR")
    return {
        "by_currency": by_currency,
        "usd_bloc_pct": round(usd_bloc, 2),
        "non_eur_pct": round(non_eur, 2),
    }


def hedge_pct_table(
    usd_bloc_pct: float, ratios: tuple[float, ...] = (0.0, 0.5, 1.0)
) -> list[dict[str, Any]]:
    """EURUSD hedge size at each ratio, as % of equity (= usd_bloc_pct * ratio)."""
    return [{"ratio": r, "hedge_pct_of_equity": round(usd_bloc_pct * r, 2)} for r in ratios]

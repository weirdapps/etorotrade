"""FX exposure for a EUR-based investor.

Policy (chosen 2026-07): UNHEDGED, but the net USD-bloc concentration is a
BINDING constraint in construction, and a hedge is only ADVISED when the book is
over-exposed AND the carry is favorable. The FX P&L diagnostic showed the EUR/USD
move swung EUR P&L by ~EUR12k, so currency concentration must be controlled even
though we don't mechanically hedge (carry cost; Campbell-Viceira).

HKD is USD-pegged (~7.78), so HKD names count inside the USD bloc.
"""

from __future__ import annotations

import numpy as np

_EUR_SUFFIXES = {
    ".DE",
    ".PA",
    ".MI",
    ".AS",
    ".MC",
    ".OL",
    ".ST",
    ".CO",
    ".HE",
    ".LS",
    ".BR",
    ".VI",
}
_SUFFIX_CCY = {".L": "GBP", ".HK": "HKD", ".T": "JPY", ".SW": "CHF", ".TO": "CAD", ".AX": "AUD"}

# Currencies that move with (or are pegged to) the US dollar for a EUR investor.
USD_BLOC = frozenset({"USD", "HKD"})


def currency_of(ticker: str) -> str:
    """Infer the listing currency from a ticker suffix (bare ticker -> USD)."""
    t = str(ticker).upper()
    if "." in t:
        suffix = "." + t.rsplit(".", 1)[-1]
        if suffix in _EUR_SUFFIXES:
            return "EUR"
        if suffix in _SUFFIX_CCY:
            return _SUFFIX_CCY[suffix]
    return "USD"


def bloc_exposure(weights: dict, bloc=USD_BLOC) -> float:
    """Total portfolio weight whose currency falls in the USD bloc (USD + HKD)."""
    return float(sum(w for tkr, w in weights.items() if currency_of(tkr) in bloc))


def cap_bloc(weights: np.ndarray, is_bloc: np.ndarray, cap: float) -> np.ndarray:
    """If the aggregate USD-bloc weight exceeds ``cap``, scale the bloc down to the
    cap and redistribute the freed weight to the non-bloc names proportionally."""
    w = np.asarray(weights, dtype=float).copy()
    is_bloc = np.asarray(is_bloc, dtype=bool)
    bloc_sum = float(w[is_bloc].sum())
    if bloc_sum <= cap + 1e-12 or bloc_sum <= 0:
        return w
    excess = bloc_sum - cap
    w[is_bloc] *= cap / bloc_sum
    nonbloc = (~is_bloc) & (w > 0)
    nonbloc_sum = float(w[nonbloc].sum())
    if nonbloc.any() and nonbloc_sum > 0:
        w[nonbloc] += excess * (w[nonbloc] / nonbloc_sum)
    return w


def hedge_advisory(bloc_exposure_pct: float, rate_diff_pct: float, cap: float = 0.4) -> dict:
    """Advisory only (never mandates a hedge).

    Args:
        bloc_exposure_pct: net USD-bloc weight (0-1).
        rate_diff_pct: USD policy rate minus EUR policy rate. Positive => hedging
            USD->EUR costs carry (the typical 2023-25 environment).
        cap: the concentration threshold above which FX risk is "material".
    """
    over_cap = bloc_exposure_pct > cap
    carry_costly = rate_diff_pct > 0
    hedge_recommended = bool(over_cap and not carry_costly)
    if not over_cap:
        note = "USD-bloc within cap — unhedged, monitor only."
    elif carry_costly:
        note = (
            "Over-exposed to USD bloc but hedging carries negative carry "
            f"(rate diff +{rate_diff_pct:.2f}pp) — stay UNHEDGED; reduce concentration instead."
        )
    else:
        note = "Over-exposed and carry favorable — consider a tactical partial USD hedge."
    return {
        "bloc_exposure": bloc_exposure_pct,
        "over_cap": over_cap,
        "rate_diff_pct": rate_diff_pct,
        "hedge_recommended": hedge_recommended,
        "note": note,
    }

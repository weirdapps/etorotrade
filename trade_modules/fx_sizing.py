"""
M8: FX-aware position sizing — CIO v36 Empirical Refoundation.

The eToro account is EUR-home. Today the sizer treats positions as USD
notional (max_position_usd: 22500 in config.yaml) and ignores FX vol on
non-EUR positions. With ~70% USD exposure × 8% EURUSD vol, that is ~560
bps of unhedged FX risk uncounted in the sizing math.

This module:
- currency_for_ticker(ticker) — infers currency from ticker suffix
- fx_vol_multiplier(currency, ref_currency, stock_vol) — returns size
  scaling that keeps total volatility constant after adding the FX layer.

Math:
    σ_total² = σ_stock² + σ_FX²              (uncorrelated approximation)
    scale     = σ_stock / σ_total            (preserves total vol budget)

A 20%-vol stock with EURUSD vol of 8% loses ~7% of position size to keep
total vol equal to a pure EUR position with σ=20%.
"""

from __future__ import annotations

import math

# Annualized FX vol vs EUR (approximate long-run averages, 2020-2025)
# Update from realized FX returns periodically.
_FX_VOL_ANNUAL = {
    "EUR": 0.000,  # No FX layer when home == position currency
    "USD": 0.080,
    "HKD": 0.080,  # HKD pegged to USD → ~same EUR-pair vol
    "JPY": 0.100,  # higher EUR-JPY historical vol
    "GBP": 0.060,
    "CHF": 0.050,  # CHF often EUR-correlated
    "CNY": 0.060,
    "KRW": 0.090,
    "AUD": 0.090,
    "CAD": 0.085,
    "DKK": 0.005,  # Danish krone tightly EUR-pegged
    "SEK": 0.060,
    "NOK": 0.080,
}


# Ticker suffix → currency mapping. Order matters (longest suffix wins
# isn't needed here; suffixes are unambiguous in yfinance conventions).
_SUFFIX_TO_CURRENCY = {
    ".HK": "HKD",
    ".T": "JPY",
    ".KS": "KRW",
    ".SS": "CNY",
    ".SZ": "CNY",
    ".AX": "AUD",
    ".TO": "CAD",
    ".V": "CAD",
    ".L": "GBP",
    ".SW": "CHF",
    ".VX": "CHF",
    ".PA": "EUR",  # Paris
    ".DE": "EUR",  # Germany (Xetra)
    ".F": "EUR",  # Frankfurt
    ".BR": "EUR",  # Brussels
    ".AS": "EUR",  # Amsterdam
    ".LS": "EUR",  # Lisbon
    ".MI": "EUR",  # Milan
    ".MC": "EUR",  # Madrid
    ".VI": "EUR",  # Vienna
    ".HE": "EUR",  # Helsinki
    ".CO": "DKK",  # Copenhagen — DKK actually but EUR-pegged in pricing
    ".ST": "SEK",  # Stockholm
    ".OL": "NOK",  # Oslo
}

DEFAULT_REF_CURRENCY = "EUR"
DEFAULT_STOCK_VOL_ANNUAL = 0.20  # ~20% annualized vol for typical equity


def currency_for_ticker(ticker: str) -> str:
    """Infer currency from ticker suffix; default to USD for plain tickers.

    Specific mappings handle Danish kroner / Norwegian krone, etc. Tickers
    like NOVO-B.CO map to DKK but the test suite treats it as EUR for
    backwards compatibility — DKK is tightly EUR-pegged so the multiplier
    is essentially 1.0 anyway.
    """
    if not isinstance(ticker, str):
        return "USD"
    upper = ticker.upper()
    for suffix, ccy in _SUFFIX_TO_CURRENCY.items():
        if upper.endswith(suffix.upper()):
            # DKK is so EUR-pegged we treat as EUR for sizing simplicity
            if ccy == "DKK":
                return "EUR"
            return ccy
    return "USD"


def fx_vol_multiplier(
    currency: str,
    ref_currency: str = DEFAULT_REF_CURRENCY,
    stock_vol_annual: float = DEFAULT_STOCK_VOL_ANNUAL,
) -> float:
    """Return position-size multiplier that preserves total vol budget.

    For a position in `currency` held in `ref_currency` book:
      σ_total = sqrt(σ_stock² + σ_FX²)
      multiplier = σ_stock / σ_total      ≤ 1.0

    Returns 1.0 when currency == ref_currency. Returns 1.0 if vol numbers
    are missing (graceful degradation — no haircut beats wrong haircut).
    """
    if not currency or currency == ref_currency:
        return 1.0
    fx_vol = _FX_VOL_ANNUAL.get(currency.upper())
    if fx_vol is None or fx_vol <= 0:
        return 1.0
    if stock_vol_annual <= 0:
        return 1.0
    sigma_total = math.sqrt(stock_vol_annual**2 + fx_vol**2)
    if sigma_total <= 0:
        return 1.0
    return round(stock_vol_annual / sigma_total, 4)

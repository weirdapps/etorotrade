"""BUILD ⑥c (2026-07-18, D25): EUR-denominated S&P 500 benchmark.

The owner's home currency is EUR, so the benchmark is the EUR return of holding SPY
= SPY's USD return combined with the EUR/USD move. Pure conversion + period-return
helpers plus a thin fetch wrapper (SPY + EURUSD via robust_fetch_prices) that
degrades to NaN when prices are unavailable.
"""

from __future__ import annotations

import pandas as pd

SPY_TICKER = "SPY"
EURUSD_TICKER = "EURUSD=X"  # yfinance forex: USD per 1 EUR


def to_eur(usd_close, eurusd):
    """Convert a USD price (series or scalar) to EUR: EUR = USD / (USD per EUR)."""
    return usd_close / eurusd


def period_return_pct(series) -> float:
    """Total return % over a price series: (last / first - 1) x 100. NaN if < 2
    clean points or a zero base."""
    s = pd.Series(series).dropna()
    if len(s) < 2 or float(s.iloc[0]) == 0.0:
        return float("nan")
    return (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0


def spy_eur_return_pct(spy_usd_close, eurusd_close) -> float:
    """EUR total return % of holding SPY over the series window (aligned on common
    dates)."""
    spy = pd.Series(spy_usd_close).dropna()
    fx = pd.Series(eurusd_close).dropna()
    idx = spy.index.intersection(fx.index)
    if len(idx) < 2:
        return float("nan")
    return period_return_pct(spy.loc[idx] / fx.loc[idx])


def fetch_spy_eur_return_pct(period: str = "1y", *, fetch=None) -> float:
    """Fetch SPY + EURUSD and return the EUR return % over ``period``. NaN on any
    failure (unreachable prices, missing columns) so callers degrade gracefully."""
    if fetch is None:
        from trade_modules.v3.fetch import robust_fetch_prices

        fetch = robust_fetch_prices
    try:
        df = fetch([SPY_TICKER, EURUSD_TICKER], period=period)
    except Exception:  # noqa: BLE001 - benchmark is best-effort; never break the report
        return float("nan")
    if df is None or df.empty or SPY_TICKER not in df.columns or EURUSD_TICKER not in df.columns:
        return float("nan")
    return spy_eur_return_pct(df[SPY_TICKER], df[EURUSD_TICKER])

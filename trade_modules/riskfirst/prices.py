"""riskfirst prices — price-series utilities for the signal engine.

All pure functions — no I/O except the fetch_* helpers (which call yfinance and
are tested with mocks, not in the pure unit suite).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore

# ---------------------------------------------------------------------------
# Core price utilities
# ---------------------------------------------------------------------------


def daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; drops the leading all-NaN row."""
    return price_df.pct_change().iloc[1:]


def momentum_12_1(
    prices: pd.Series,
    month: int = 21,
    year: int = 252,
) -> float:
    """Jegadeesh-Titman 12-1 momentum: return from ~12 months ago to ~1 month ago
    (the most recent month is SKIPPED to avoid short-term reversal). NaN if the
    series is shorter than a year+1 of observations.

    Args:
        prices: a daily closing-price series (DatetimeIndex preferred).
        month:  number of trading days in 1 month (default 21).
        year:   number of trading days in 12 months (default 252).

    Returns:
        float: (price_{-month} / price_{-year}) - 1, or NaN if insufficient history.
    """
    n = len(prices)
    min_required = year + 1  # need at least 253 observations
    if n < min_required:
        return float("nan")

    # Drop NaN tails
    valid = prices.dropna()
    if len(valid) < min_required:
        return float("nan")

    p_1m = float(valid.iloc[-month])  # price 1 month ago (skip last month)
    p_12m = float(valid.iloc[-year])  # price 12 months ago

    if p_12m <= 0:
        return float("nan")

    return p_1m / p_12m - 1.0


def realized_vol(prices: pd.Series, window: int = 252, ppy: int = 252) -> float:
    """Annualised realized volatility of daily log returns over the last window."""
    valid = prices.dropna()
    if len(valid) < window + 1:
        return float("nan")
    log_ret = np.log(valid.iloc[-window - 1 :] / valid.iloc[-window - 1 :].shift(1)).dropna()
    return float(log_ret.std(ddof=1) * np.sqrt(ppy))


# ---------------------------------------------------------------------------
# Factor factories (closures over a price DataFrame)
# ---------------------------------------------------------------------------


def price_momentum_factor(prices_df: pd.DataFrame) -> Callable:
    """Factory: a factor ``compute(df)`` returning z-scored true 12-1 momentum,
    computed from ``prices_df`` (dates x tickers). Higher = stronger momentum."""

    def compute(df: pd.DataFrame) -> pd.Series:
        tickers = df.index.tolist()
        vals = {}
        for t in tickers:
            if t in prices_df.columns:
                vals[t] = momentum_12_1(prices_df[t])
            else:
                vals[t] = float("nan")
        s = pd.Series(vals, index=df.index, dtype=float)
        return zscore(s)

    return compute


def price_lowvol_factor(prices_df: pd.DataFrame) -> Callable:
    """Factory: a factor ``compute(df)`` returning z-scored INVERSE realized vol
    (lower vol -> higher score, the low-vol anomaly)."""

    def compute(df: pd.DataFrame) -> pd.Series:
        tickers = df.index.tolist()
        vals = {}
        for t in tickers:
            if t in prices_df.columns:
                v = realized_vol(prices_df[t])
                vals[t] = -v if np.isfinite(v) else float("nan")
            else:
                vals[t] = float("nan")
        s = pd.Series(vals, index=df.index, dtype=float)
        return zscore(s)

    return compute


# ---------------------------------------------------------------------------
# Data-fetch helpers (impure — wrapped so tests mock them)
# ---------------------------------------------------------------------------


def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Daily adjusted closes for tickers as a (dates x tickers) frame."""
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("yfinance is required for fetch_prices") from exc
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return close.sort_index()


def fetch_sectors(tickers: list[str]) -> dict:
    """Best-effort {ticker: sector} via yfinance (one call per name)."""
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError:
        return {}
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = info.get("sector", "Unknown")
        except Exception:
            out[t] = "Unknown"
    return out


def fetch_earnings_dates(tickers: list[str]) -> dict:
    """Best-effort {ticker: 'YYYY-MM-DD'} next-earnings map via yfinance.
    Failures per ticker are swallowed (ticker omitted) — FAIL-OPEN: a transient
    fetch miss must NOT exclude a good name (earnings blackout is entry-timing,
    not a survival rail)."""
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError:
        return {}
    out = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            if cal is not None and not cal.empty:
                v = cal.get("Earnings Date")
                if v is not None:
                    dates = sorted(str(d)[:10] for d in v if pd.notna(d))
                    if dates:
                        out[t] = dates[0]
        except Exception:
            pass
    return out

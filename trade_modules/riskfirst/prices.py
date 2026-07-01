"""Price-history factor primitives + thin fetch glue.

Replaces the snapshot proxies used in the first-cut engine:
- true 12-1 (skip-month) momentum instead of 52W-proximity;
- realized volatility instead of beta (for the low-vol factor / sizing);
- an empirical shrunk covariance instead of the single-factor beta covariance.

The pure computations are unit-tested; ``fetch_prices`` / ``fetch_sectors`` are
network glue (yfinance) exercised at integration time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import zscore


def daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; drops the leading all-NaN row."""
    return price_df.pct_change(fill_method=None).dropna(how="all")


def momentum_12_1(prices: pd.Series, month: int = 21, year: int = 252) -> float:
    """Jegadeesh-Titman 12-1 momentum: return from ~12 months ago to ~1 month ago
    (the most recent month is SKIPPED to avoid short-term reversal). NaN if the
    series is shorter than a year+1 of observations."""
    prices = prices.dropna()
    if len(prices) < year + 1:
        return float("nan")
    p_1m = float(prices.iloc[-(month + 1)])
    p_12m = float(prices.iloc[-(year + 1)])
    if p_12m <= 0:
        return float("nan")
    return p_1m / p_12m - 1.0


def realized_vol(prices: pd.Series, window: int = 252, ppy: int = 252) -> float:
    """Annualised realized volatility of daily log returns over the last window."""
    prices = prices.dropna()
    logret = np.log(prices / prices.shift(1)).dropna()
    if len(logret) == 0:
        return float("nan")
    return float(logret.iloc[-window:].std(ddof=0) * np.sqrt(ppy))


def shrunk_cov(returns_df: pd.DataFrame, shrink: float = 0.2, annualize: int = 252) -> np.ndarray:
    """Sample covariance shrunk toward its diagonal (a transparent Ledoit-Wolf-style
    estimator): (1-d)*S + d*diag(S). Off-diagonals shrink by (1-d); diagonal intact.
    Reduces the estimation error that wrecks mean-variance/ERC on short samples."""
    R = returns_df.dropna(how="any")
    S = np.atleast_2d(np.cov(R.to_numpy(), rowvar=False)) * annualize
    target = np.diag(np.diag(S))
    return (1.0 - shrink) * S + shrink * target


def price_momentum_factor(prices_df: pd.DataFrame):
    """Factory: a factor ``compute(df)`` returning z-scored true 12-1 momentum,
    computed from ``prices_df`` (dates x tickers). Higher = stronger momentum."""

    def compute(df: pd.DataFrame) -> pd.Series:
        vals = {
            t: (momentum_12_1(prices_df[t]) if t in prices_df.columns else float("nan"))
            for t in df.index
        }
        return zscore(pd.Series(vals).reindex(df.index))

    return compute


def price_lowvol_factor(prices_df: pd.DataFrame):
    """Factory: a factor ``compute(df)`` returning z-scored INVERSE realized vol
    (lower vol -> higher score, the low-vol anomaly)."""

    def compute(df: pd.DataFrame) -> pd.Series:
        vals = {
            t: (realized_vol(prices_df[t]) if t in prices_df.columns else float("nan"))
            for t in df.index
        }
        return zscore(-pd.Series(vals).reindex(df.index))

    return compute


# --------------------------------------------------------------------------- #
# Network glue (yfinance) — exercised at integration time.
# --------------------------------------------------------------------------- #


def fetch_prices(tickers, period: str = "2y") -> pd.DataFrame:  # pragma: no cover
    """Daily adjusted closes for tickers as a (dates x tickers) frame."""
    import yfinance as yf

    data = yf.download(list(tickers), period=period, progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) or "Close" in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close.dropna(how="all")


def fetch_sectors(tickers) -> dict:  # pragma: no cover
    """Best-effort {ticker: sector} via yfinance (one call per name)."""
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            out[t] = info.get("sector")
        except Exception:
            out[t] = None
    return out

"""12-1 month momentum factor (Jegadeesh-Titman 1993).

Computes trailing 12-month return excluding the most recent month.
The skip-month avoids short-term reversal contamination and is the
standard academic construction for cross-sectional momentum.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_momentum_12_1m(
    price_now: float,
    price_1m_ago: float | None,
    price_12m_ago: float | None,
) -> float | None:
    """12-month return excluding last month (Jegadeesh-Titman skip-month).

    Returns percentage: +23.8 means the stock rose 23.8% over months 2-12
    (skipping the most recent month).

    Args:
        price_now: Current price (unused in pure skip-month, but needed
                   if price_1m_ago is unavailable -- falls back to 11-month)
        price_1m_ago: Price 1 month (~21 trading days) ago
        price_12m_ago: Price 12 months (~252 trading days) ago

    Returns:
        Momentum percentage, or None if insufficient data
    """
    if price_12m_ago is None or price_12m_ago <= 0:
        return None
    if price_1m_ago is not None and price_1m_ago > 0:
        return round(((price_1m_ago / price_12m_ago) - 1) * 100, 2)
    # Fallback: full 12-month if 1m price unavailable
    if price_now is not None and price_now > 0:
        return round(((price_now / price_12m_ago) - 1) * 100, 2)
    return None


def compute_momentum_from_series(
    prices: "np.ndarray | list[float]",
) -> float | None:
    """Compute 12-1m momentum from a daily price series.

    Expects at least 252 trading days of data. Uses the last price as
    "now", price at -21 days as "1m ago", and price at -252 as "12m ago".
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 252:
        return None
    price_now = float(arr[-1])
    price_1m = float(arr[-21]) if len(arr) >= 21 else None
    price_12m = float(arr[-252])
    return compute_momentum_12_1m(price_now, price_1m, price_12m)


def fetch_momentum_for_ticker(ticker: str) -> float | None:
    """Fetch 12-1m momentum for a ticker using yfinance.

    Pulls ~14 months of daily close prices and computes the skip-month
    momentum factor. Returns None on any data failure (network, delisted,
    insufficient history, etc.).

    This is called from the signal logger -- it must never raise.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period="14mo", auto_adjust=True)
        if hist.empty or len(hist) < 252:
            return None
        closes = hist["Close"].values
        return compute_momentum_from_series(closes)
    except Exception as e:
        logger.debug(f"Failed to fetch momentum for {ticker}: {e}")
        return None

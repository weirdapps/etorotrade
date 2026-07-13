"""Robust batched price fetcher for the v3 pipeline.

Chunks a large ticker list into small batches, throttles between calls,
and retries on network/rate-limit errors with exponential back-off.
"""

from __future__ import annotations

import time

import pandas as pd


def _default_downloader(batch: list[str], period: str) -> pd.DataFrame:
    """Thin yfinance wrapper.

    Handles both column shapes that yfinance emits:
    - Multi-ticker (or new-style single-ticker): MultiIndex columns — extract
      the "Close" level, which gives a DataFrame with ticker names as columns.
    - Legacy single-ticker: flat columns ("Open", "High", …, "Close") —
      extract the "Close" column and rename to the ticker symbol.

    Imported at call time so the module is safe to import without yfinance.
    """
    import yfinance as yf  # noqa: PLC0415

    data = yf.download(
        batch,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Standard multi-ticker result: ("Close","AAPL"), ("Open","AAPL"), …
        close = data["Close"]
        if isinstance(close, pd.Series):
            # Shouldn't happen with MultiIndex, but guard anyway.
            close = close.to_frame(name=batch[0])
    elif "Close" in data.columns:
        # Legacy single-ticker flat layout: columns are metric names.
        close = data[["Close"]].rename(columns={"Close": batch[0]})
    else:
        close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=batch[0])

    # Safety net: if yfinance returns a 1-column frame but the name doesn't
    # match the requested ticker (e.g. version quirks), fix it up.
    if len(batch) == 1 and len(close.columns) == 1 and close.columns[0] != batch[0]:
        close = close.rename(columns={close.columns[0]: batch[0]})

    return close


def robust_fetch_prices(
    tickers: list[str],
    period: str = "5y",
    batch_size: int = 80,
    pause: float = 1.0,
    retries: int = 3,
    downloader=None,
) -> pd.DataFrame:
    """Fetch adjusted Close prices in batches with retry and exponential back-off.

    Parameters
    ----------
    tickers:
        Full list of ticker symbols to fetch.
    period:
        yfinance period string, e.g. "5y" or "2y".
    batch_size:
        Number of tickers per download call.
    pause:
        Seconds to sleep between successful batches; also the base for the
        exponential back-off on retries (``pause * 2 ** attempt``).
    retries:
        Maximum retry attempts per failing batch.  Total tries per batch =
        ``1 + retries``.
    downloader:
        Callable ``(batch: list[str], period: str) -> pd.DataFrame`` that
        returns a dates × tickers Close frame.  Injectable for testing.
        Defaults to :func:`_default_downloader` (yfinance).

    Returns
    -------
    pd.DataFrame
        Dates × tickers adjusted-Close frame, outer-joined across batches.
        All-NaN columns are dropped.  Returns an empty DataFrame if every
        batch fails.
    """
    if downloader is None:
        downloader = _default_downloader

    chunks = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    frames: list[pd.DataFrame] = []

    for chunk_idx, batch in enumerate(chunks):
        for attempt in range(retries + 1):  # initial try + up to `retries` retries
            try:
                df = downloader(batch, period)
                if isinstance(df, pd.Series):
                    df = df.to_frame(name=batch[0])
                frames.append(df)
                break
            except Exception:  # noqa: BLE001
                if attempt < retries:
                    backoff = pause * (2**attempt)
                    time.sleep(backoff)

        # Throttle between batches; skip sleep after the last chunk.
        if chunk_idx < len(chunks) - 1:
            time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1, join="outer")
    # Drop columns that are entirely NaN (unfetchable symbols).
    return result.dropna(axis=1, how="all")

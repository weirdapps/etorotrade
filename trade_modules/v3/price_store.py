"""BUILD ⑤ (2026-07-19): versioned, append-only price store (survivorship fix).

The v3 spine live-fetches prices every run with no disk backing, and a name that
leaves ``etoro.csv`` simply vanishes -> full survivorship bias (BUY/HOLD biased up,
SELL biased down). This store keeps every bar it has ever seen (delisting
retention) so a backtest can price names that have since left the universe, and it
refreshes same-``(date, ticker)`` closes to the newest fetch because split/dividend
adjustment rewrites history retroactively.

Storage: a single long-format parquet at ``~/.weirdapps-trading/v3_price_store.parquet``
with columns ``date, ticker, close`` (adjusted close, EUR or USD per the caller).
Append-only in that rows are never DROPPED; a conflicting ``(date, ticker)`` is
UPDATED to the newest close.

Deferred follow-ups (documented, not built here): git-history backward-seed of
already-delisted tickers from ``etoro.csv`` history; wiring the store as the spine's
read source; and the price-only sleeve re-validation (data-limited — Phase-1's naive
spine showed no standalone edge).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STORE_PATH: str = str(Path("~/.weirdapps-trading/v3_price_store.parquet").expanduser())
_COLS = ["date", "ticker", "close"]


def _to_long(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalise a wide (dates x tickers close) OR long (date, ticker, close) frame
    to canonical long form with string dates and numeric closes; NaN closes dropped."""
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=_COLS)
    if set(_COLS).issubset(prices.columns):
        out = prices[_COLS].copy()
    else:  # wide: index = dates, columns = tickers
        wide = prices.copy()
        wide.index = pd.to_datetime(wide.index)
        wide.index.name = "date"
        out = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="close")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["ticker"] = out["ticker"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["close"])[_COLS]


def _read_store(store_path: str) -> pd.DataFrame:
    p = Path(store_path).expanduser()
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    try:
        return pd.read_parquet(p)[_COLS]
    except Exception:  # noqa: BLE001 - a corrupt store must not crash the pipeline
        return pd.DataFrame(columns=_COLS)


def append_bars(prices: pd.DataFrame, *, store_path: str = STORE_PATH) -> int:
    """Merge ``prices`` (wide or long) into the append-only store.

    Existing ``(date, ticker)`` rows are refreshed to the newest close; no row is
    ever dropped (delisting retention). Returns the number of NEW ``(date, ticker)``
    pairs added (refreshes of existing pairs are not counted).
    """
    new = _to_long(prices)
    if new.empty:
        return 0
    existing = _read_store(store_path)
    existing_keys = (
        set(zip(existing["date"], existing["ticker"], strict=False))
        if not existing.empty
        else set()
    )
    added = len({(d, t) for d, t in zip(new["date"], new["ticker"], strict=False)} - existing_keys)

    combined = pd.concat([existing, new], ignore_index=True)
    # Newest-wins on (date, ticker): `new` is concatenated last, so keep="last".
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(
        ["date", "ticker"]
    )
    p = Path(store_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)
    return added


def read_close(tickers, *, store_path: str = STORE_PATH, start=None, end=None) -> pd.DataFrame:
    """Wide dates x tickers adjusted-close frame for ``tickers`` — delisted names
    included (their bars persist). Empty frame if the store or the selection is empty."""
    store = _read_store(store_path)
    if store.empty:
        return pd.DataFrame()
    want = {str(t) for t in tickers}
    sub = store[store["ticker"].isin(want)]
    if start is not None:
        sub = sub[sub["date"] >= str(start)]
    if end is not None:
        sub = sub[sub["date"] <= str(end)]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def store_coverage(store_path: str = STORE_PATH) -> dict:
    """Diagnostics: row / ticker / date counts and the stored date range."""
    store = _read_store(store_path)
    if store.empty:
        return {"n_rows": 0, "n_tickers": 0, "n_dates": 0, "first": None, "last": None}
    dates = store["date"]
    return {
        "n_rows": int(len(store)),
        "n_tickers": int(store["ticker"].nunique()),
        "n_dates": int(dates.nunique()),
        "first": str(dates.min()),
        "last": str(dates.max()),
    }

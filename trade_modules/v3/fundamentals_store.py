"""PIT fundamentals store (Sharadar SF1) — the point-in-time backbone for the
data-blocked durable premia (P/B, asset-growth/CMA, GP-assets, accruals, SUE/PEAD).

Mirrors ``v3/price_store.py``: a single append-only long parquet keyed by
``(ticker, datekey)``, where ``datekey`` is the date a filing became PUBLIC. The
core primitive is :func:`read_asof` — as of date T it returns, per ticker, the most
recent filing with ``datekey <= T`` only, so a factor computed "as of T" can never
see a report that had not yet been filed (no look-ahead). Rows are never dropped
(delisting / restatement history retained); a conflicting ``(ticker, datekey)`` is
refreshed to the newest values (restatements).

Storage: ``~/.weirdapps-trading/v3_fundamentals_store.parquet``. Fields are the
minimal set the derivations in ``v3/fundamentals.py`` need; extend NUM_FIELDS as
more factors are added. Ingestion (network) lives in ``scripts/v3_fundamentals_update.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STORE_PATH: str = str(Path("~/.weirdapps-trading/v3_fundamentals_store.parquet").expanduser())

KEYS = ["ticker", "datekey", "reportperiod"]
# Numeric as-reported fields we derive factors from (Sharadar SF1 names, ARQ dimension).
NUM_FIELDS = ["assets", "equity", "gp", "netinc", "ncfo", "revenue", "eps"]
STORE_COLS = KEYS + NUM_FIELDS


def _normalise(records: pd.DataFrame) -> pd.DataFrame:
    """Coerce an incoming frame to the canonical schema (missing fields -> NaN)."""
    if records is None or len(records) == 0:
        return pd.DataFrame(columns=STORE_COLS)
    out = records.reindex(columns=STORE_COLS).copy()
    out["ticker"] = out["ticker"].astype(str)
    out["datekey"] = pd.to_datetime(out["datekey"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["reportperiod"] = out["reportperiod"].astype("object")
    for f in NUM_FIELDS:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    return out.dropna(subset=["ticker", "datekey"])[STORE_COLS]


def _read_store(store_path: str) -> pd.DataFrame:
    p = Path(store_path).expanduser()
    if not p.exists():
        return pd.DataFrame(columns=STORE_COLS)
    try:
        return pd.read_parquet(p).reindex(columns=STORE_COLS)
    except Exception:  # noqa: BLE001 - a corrupt store must not crash the pipeline
        return pd.DataFrame(columns=STORE_COLS)


def append_records(records: pd.DataFrame, *, store_path: str = STORE_PATH) -> int:
    """Merge ``records`` into the append-only store.

    Existing ``(ticker, datekey)`` rows are refreshed to the newest values (a
    restatement filed under the same datekey); no row is ever dropped. Returns the
    number of NEW ``(ticker, datekey)`` pairs added.
    """
    new = _normalise(records)
    if new.empty:
        return 0
    existing = _read_store(store_path)
    existing_keys = (
        set(zip(existing["ticker"], existing["datekey"], strict=False))
        if not existing.empty
        else set()
    )
    added = len(
        {(t, d) for t, d in zip(new["ticker"], new["datekey"], strict=False)} - existing_keys
    )
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "datekey"], keep="last").sort_values(
        ["ticker", "datekey"]
    )
    p = Path(store_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)
    return added


def read_asof(tickers, as_of, *, store_path: str = STORE_PATH) -> pd.DataFrame:
    """Point-in-time cross-section: per ticker, the latest filing with datekey <= as_of.

    Returns a frame indexed by ticker (columns = NUM_FIELDS + datekey + reportperiod).
    Tickers with no visible filing as of ``as_of`` are omitted. Empty if the store is.
    """
    store = _read_store(store_path)
    if store.empty:
        return pd.DataFrame()
    want = {str(t) for t in tickers}
    sub = store[store["ticker"].isin(want) & (store["datekey"] <= str(as_of))]
    if sub.empty:
        return pd.DataFrame()
    latest = sub.sort_values("datekey").groupby("ticker", as_index=True).last()
    return latest[NUM_FIELDS + ["datekey", "reportperiod"]]


def read_history(tickers, as_of, *, store_path: str = STORE_PATH) -> pd.DataFrame:
    """All filings visible as of ``as_of`` for ``tickers``, oldest-first — for
    series-based factors (e.g. seasonal-random-walk SUE needs the EPS history)."""
    store = _read_store(store_path)
    if store.empty:
        return pd.DataFrame(columns=STORE_COLS)
    want = {str(t) for t in tickers}
    sub = store[store["ticker"].isin(want) & (store["datekey"] <= str(as_of))]
    return sub.sort_values(["ticker", "datekey"]).reset_index(drop=True)


def store_coverage(store_path: str = STORE_PATH) -> dict:
    """Diagnostics: row / ticker counts and the stored datekey range."""
    store = _read_store(store_path)
    if store.empty:
        return {"n_rows": 0, "n_tickers": 0, "first": None, "last": None}
    dk = store["datekey"]
    return {
        "n_rows": int(len(store)),
        "n_tickers": int(store["ticker"].nunique()),
        "first": str(dk.min()),
        "last": str(dk.max()),
    }

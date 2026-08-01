"""Raw yfinance ``.info`` snapshot — a byproduct of the nightly market build.

The etoro.csv build already fetches ``yticker.info`` once per ticker for the whole
universe. A downstream consumer (the plessas-trading-stack overlay) needs a handful of
those raw fields (P/B, trailing P/S, operating margin, ROA, targets, sector, …) that the
etoro.csv columns don't carry. Rather than re-fetch ``.info`` a second time there, this
module records those fields as each ticker is fetched and dumps them to a parquet at
process exit — so the SAME single ``.info`` sweep feeds both etoro.csv and the stack's
info store.

Design guarantees (this must NEVER affect the etoro.csv build):
  * ``record`` is a pure in-memory dict write wrapped by the caller in try/except.
  * the dump is registered via ``atexit`` (runs after etoro.csv is written) and is itself
    fully guarded — any failure is swallowed.
  * the consumer treats the snapshot as best-effort: a missing/stale snapshot just means
    it re-fetches, so nothing breaks if this module no-ops.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

import pandas as pd

# The yfinance ``.info`` keys the stack's info store consumes (kept in sync with
# pts_signals.v3.features._INFO_NUM / _INFO_STR). Projecting to these keeps the snapshot
# small (a raw .info is 100+ fields) and is the contract between the two repos.
INFO_SNAPSHOT_KEYS: tuple[str, ...] = (
    "priceToBook",
    "enterpriseToEbitda",
    "returnOnAssets",
    "grossMargins",
    "operatingMargins",
    "currentRatio",
    "targetHighPrice",
    "targetLowPrice",
    "averageVolume",
    "revenueGrowth",
    "priceToSalesTrailing12Months",
    "sector",
    "industry",
    "country",
    "quoteType",
    "longBusinessSummary",
)

# Default output path: alongside etoro.csv so the stack reads it like it reads etoro.csv.
SNAPSHOT_PATH: str = os.environ.get(
    "ETORO_INFO_SNAPSHOT",
    os.path.join(os.path.dirname(__file__), "..", "output", "etoro_info_snapshot.parquet"),
)

_ACCUM: dict[str, dict[str, Any]] = {}
_DUMP_REGISTERED = False


def record(ticker: str, raw_info: dict[str, Any] | None) -> None:
    """Accumulate the snapshot fields for one ticker from its raw ``.info`` dict.

    Non-fatal by contract — the caller wraps this in try/except, and the first call
    registers the atexit dump. A None/empty ``raw_info`` is ignored.
    """
    global _DUMP_REGISTERED
    if not raw_info:
        return
    proj = {k: raw_info.get(k) for k in INFO_SNAPSHOT_KEYS if raw_info.get(k) is not None}
    if proj:
        _ACCUM[str(ticker)] = proj
    if not _DUMP_REGISTERED:
        atexit.register(dump)
        _DUMP_REGISTERED = True


def merge_snapshots(
    snapshot_dir: str,
    output: str | None = None,
    pattern: str = "info_snapshot_shard_*.parquet",
) -> int:
    """Merge per-shard snapshot parquets (the CI market scan is sharded) into one file.

    Concatenate every ``info_snapshot_shard_*.parquet`` under ``snapshot_dir`` and dedupe
    on ticker (newest wins). Returns the merged ticker count; 0 (no file written) when no
    shard snapshots are found. Never raises.
    """
    import glob

    try:
        paths = sorted(glob.glob(os.path.join(snapshot_dir, pattern)))
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_parquet(p))
            except Exception:  # noqa: BLE001 - skip a corrupt shard, keep the rest
                continue
        if not frames:
            return 0
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker"], keep="last")
        out = os.path.abspath(output or SNAPSHOT_PATH)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_parquet(out, index=False)
        return len(df)
    except Exception:  # noqa: BLE001 - best-effort; a merge failure just means no snapshot
        return 0


def dump(path: str | None = None) -> int:
    """Write the accumulated snapshot to a tidy parquet (one row per ticker). Returns the
    number of tickers written; 0 (and no file) when nothing was recorded. Never raises."""
    if not _ACCUM:
        return 0
    try:
        rows = [{"ticker": t, **fields} for t, fields in _ACCUM.items()]
        df = pd.DataFrame(rows)
        out = os.path.abspath(path or SNAPSHOT_PATH)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_parquet(out, index=False)
        return len(df)
    except Exception:  # noqa: BLE001 - best-effort byproduct; must never break the build
        return 0

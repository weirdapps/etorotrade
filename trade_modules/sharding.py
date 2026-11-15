"""Sharding helpers for the daily market scan (``trade.py -o e``).

The eToro universe grew past ~12.5K tickers, pushing the single market-scan job
over GitHub's hard 6-hour hosted-runner cap. Splitting the universe across N
parallel jobs keeps each job well under the cap. Each shard writes its own
``etoro_shard_<k>.csv``; the merge step recombines them into
``yahoofinance/output/etoro.csv``.

When the SHARD_COUNT/SHARD_INDEX env vars are unset (local runs, the committee,
the weekly universe refresh), behaviour is identical to the unsharded pipeline.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping

import pandas as pd

from trade_modules.output_manager import sort_by_market_cap_descending


def shard_tickers(tickers: list[str], index: int, count: int) -> list[str]:
    """Return shard ``index`` of ``count`` via round-robin stride.

    ``count <= 1`` means "no sharding" and returns the full list unchanged.
    Round-robin (``tickers[index::count]``) guarantees every ticker is covered
    exactly once across the shards, with shard sizes differing by at most one and
    slow/fast tickers spread evenly across shards.

    Raises:
        ValueError: if ``count > 1`` and ``index`` is not in ``[0, count)``.
    """
    if count <= 1:
        return list(tickers)
    if not (0 <= index < count):
        raise ValueError(f"shard index {index} out of range for count {count}")
    return list(tickers)[index::count]


def apply_shard_from_env(tickers: list[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Slice ``tickers`` to the shard named by the SHARD_INDEX/SHARD_COUNT env vars.

    Defaults: SHARD_COUNT=1 (no sharding), SHARD_INDEX=0. Pass ``env`` explicitly
    in tests; production callers let it default to ``os.environ``.
    """
    if env is None:
        env = os.environ
    count = int(env.get("SHARD_COUNT", "1"))
    index = int(env.get("SHARD_INDEX", "0"))
    return shard_tickers(tickers, index, count)


def merge_shard_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-shard DataFrames, dedupe on TKR, and sort by market cap.

    Round-robin sharding guarantees disjoint ticker sets, so the dedupe is a
    safety net. The final sort reuses the canonical
    :func:`trade_modules.output_manager.sort_by_market_cap_descending` so the
    merged ``etoro.csv`` matches the ordering of the unsharded pipeline.
    """
    combined = pd.concat(frames, ignore_index=True)
    if "TKR" in combined.columns:
        combined = combined.drop_duplicates(subset="TKR", keep="first")
    combined = sort_by_market_cap_descending(combined)
    return combined.reset_index(drop=True)


def merge_shard_csvs(input_dir: str, output_path: str, pattern: str = "etoro_shard_*.csv") -> int:
    """Read every shard CSV in ``input_dir``, merge, and write ``output_path``.

    Returns the merged row count.

    Raises:
        FileNotFoundError: if no files match ``pattern`` in ``input_dir``.
    """
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No shard files matching {pattern!r} in {input_dir}")
    frames = [pd.read_csv(p) for p in paths]
    merged = merge_shard_frames(frames)
    merged.to_csv(output_path, index=False)
    return len(merged)


#: The largest share of a shard's rows that may carry no price before the shard is read as
#: DEGRADED rather than scanned.
#:
#: Measured on the committed ``etoro.csv`` vintages, 2026-10-21 to 2026-11-15. About 1,780 of
#: ~10,900 names never price (16%), and round-robin sharding spreads them evenly, so a healthy
#: shard sits near 0.16. On 2026-11-15 two of six shards lost their Yahoo session a third of the
#: way through (10,421 "Invalid Crumb" 401s), wrote a blank row for every later name and still
#: reported "1821/1824 succeeded": 3,874 blank rows that day against the usual 1,782, which
#: over two 1,824-row shards is about 0.74 each. Four smaller days (2026-10-27, 10-30, 11-05,
#: 11-14) added 372 to 1,187 blank names. 0.25 is about nine binomial standard deviations above
#: a healthy shard, and catches a shard that lost 160 names or more. Per-shard shares on those
#: days were not kept (artifacts live one day), so the limit is set from the totals.
MAX_BLANK_PRICE_SHARE = 0.25

#: How a row with no price is spelled in the scan's CSV.
_BLANK_PRICE = {"", "--", "nan", "none"}


def blank_price_counts(csv_path: str) -> tuple[int, int]:
    """``(rows, rows_with_no_price)`` for one scan CSV. A file with no ``PRC`` column is all blank.

    Read as text: the scan writes ``--`` for a missing figure, and letting pandas infer the
    column would turn a fully blank one into floats and a mixed one into objects.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "PRC" not in df.columns:
        return len(df), len(df)
    blank = df["PRC"].str.strip().str.lower().isin(_BLANK_PRICE)
    return len(df), int(blank.sum())


def shard_is_degraded(rows: int, blank: int, max_share: float = MAX_BLANK_PRICE_SHARE) -> bool:
    """True when a shard has no rows at all, or more blank prices than a healthy scan leaves."""
    return rows <= 0 or blank / rows > max_share

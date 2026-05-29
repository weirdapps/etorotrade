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

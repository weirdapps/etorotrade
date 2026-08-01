"""Tests for the raw yfinance .info snapshot byproduct (record → dump → merge shards)."""

from __future__ import annotations

import pandas as pd

from yahoofinance.data import info_snapshot as S


def test_record_projects_and_dump_roundtrips(tmp_path):
    S._ACCUM.clear()
    S.record("AAA", {"priceToBook": 2.5, "operatingMargins": 0.3, "junkField": 999})
    S.record("BBB", {"priceToBook": 1.1})
    S.record("CCC", None)  # ignored
    out = str(tmp_path / "snap.parquet")
    n = S.dump(out)
    assert n == 2
    df = pd.read_parquet(out)
    assert "junkField" not in df.columns  # projected to the snapshot keys only
    aaa = df[df.ticker == "AAA"].iloc[0].dropna().to_dict()
    assert aaa["priceToBook"] == 2.5 and aaa["operatingMargins"] == 0.3


def test_merge_snapshots_dedupes_shards(tmp_path):
    pd.DataFrame(
        [{"ticker": "AAA", "priceToBook": 2.5}, {"ticker": "BBB", "priceToBook": 1.0}]
    ).to_parquet(tmp_path / "info_snapshot_shard_0.parquet", index=False)
    pd.DataFrame(
        [{"ticker": "CCC", "priceToBook": 3.0}, {"ticker": "AAA", "priceToBook": 9.9}]
    ).to_parquet(tmp_path / "info_snapshot_shard_1.parquet", index=False)
    out = str(tmp_path / "merged.parquet")
    n = S.merge_snapshots(str(tmp_path), output=out)
    assert n == 3  # AAA, BBB, CCC
    df = pd.read_parquet(out).set_index("ticker")
    assert df.loc["AAA", "priceToBook"] == 9.9  # newest shard wins on dedupe


def test_merge_snapshots_no_files_is_noop(tmp_path):
    assert S.merge_snapshots(str(tmp_path), output=str(tmp_path / "x.parquet")) == 0

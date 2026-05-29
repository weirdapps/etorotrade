"""
Test suite for trade_modules.sharding module.

Covers the daily-signals sharding helpers:
- shard_tickers: deterministic round-robin slice of the universe
- apply_shard_from_env: env-gated wrapper used by the `trade -o e` path
- merge_shard_frames / merge_shard_csvs: recombine per-shard outputs into etoro.csv
"""

from pathlib import Path

import pandas as pd
import pytest

from trade_modules.sharding import (
    apply_shard_from_env,
    merge_shard_csvs,
    merge_shard_frames,
    shard_tickers,
)

# --- shard_tickers ---------------------------------------------------------


@pytest.fixture
def universe():
    # 20 tickers, not divisible by most shard counts -> exposes balance bugs
    return [f"T{i}" for i in range(20)]


def test_count_one_returns_identity(universe):
    assert shard_tickers(universe, 0, 1) == universe


def test_count_zero_or_negative_returns_full(universe):
    # count<=1 means "no sharding"
    assert shard_tickers(universe, 0, 0) == universe
    assert shard_tickers(universe, 0, -3) == universe


@pytest.mark.parametrize("count", [1, 3, 6, 8])
def test_full_coverage_no_overlap(universe, count):
    shards = [shard_tickers(universe, i, count) for i in range(count)]
    # every ticker covered exactly once
    flattened = [t for s in shards for t in s]
    assert sorted(flattened) == sorted(universe)
    assert len(flattened) == len(set(flattened))  # no duplicates


@pytest.mark.parametrize("count", [3, 6, 8])
def test_shards_balanced(universe, count):
    sizes = [len(shard_tickers(universe, i, count)) for i in range(count)]
    assert max(sizes) - min(sizes) <= 1


def test_index_out_of_range_raises(universe):
    with pytest.raises(ValueError):
        shard_tickers(universe, 6, 6)  # index must be < count
    with pytest.raises(ValueError):
        shard_tickers(universe, -1, 6)


def test_empty_universe():
    assert shard_tickers([], 2, 6) == []


def test_deterministic(universe):
    assert shard_tickers(universe, 2, 6) == shard_tickers(universe, 2, 6)


# --- apply_shard_from_env --------------------------------------------------


def test_env_unset_returns_full(universe):
    assert apply_shard_from_env(universe, env={}) == universe


def test_env_count_one_returns_full(universe):
    assert apply_shard_from_env(universe, env={"SHARD_COUNT": "1"}) == universe


def test_env_applies_shard(universe):
    got = apply_shard_from_env(universe, env={"SHARD_COUNT": "6", "SHARD_INDEX": "2"})
    assert got == shard_tickers(universe, 2, 6)


def test_env_index_defaults_to_zero(universe):
    got = apply_shard_from_env(universe, env={"SHARD_COUNT": "6"})
    assert got == shard_tickers(universe, 0, 6)


# --- merge -----------------------------------------------------------------

COLUMNS = ["TKR", "NAME", "CAP", "BS"]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_merge_concatenates_disjoint_shards():
    f0 = _frame([["AAPL", "Apple", "3T", "B"]])
    f1 = _frame([["MSFT", "Microsoft", "2T", "H"]])
    f2 = _frame([["SAP.DE", "SAP", "200B", "S"]])
    merged = merge_shard_frames([f0, f1, f2])
    assert len(merged) == 3
    assert set(merged["TKR"]) == {"AAPL", "MSFT", "SAP.DE"}
    assert list(merged.columns) == COLUMNS


def test_merge_dedupes_overlapping_ticker():
    f0 = _frame([["AAPL", "Apple", "3T", "B"]])
    f1 = _frame([["AAPL", "Apple", "3T", "B"], ["MSFT", "Microsoft", "2T", "H"]])
    merged = merge_shard_frames([f0, f1])
    assert len(merged) == 2
    assert sorted(merged["TKR"]) == ["AAPL", "MSFT"]


def test_merge_sorts_by_market_cap_desc():
    f0 = _frame([["SMALL", "Small", "100B", "H"]])
    f1 = _frame([["BIG", "Big", "2T", "B"], ["MID", "Mid", "500B", "H"]])
    merged = merge_shard_frames([f0, f1])
    assert list(merged["TKR"]) == ["BIG", "MID", "SMALL"]


def test_merge_csvs_roundtrip(tmp_path):
    _frame([["AAPL", "Apple", "3T", "B"]]).to_csv(tmp_path / "etoro_shard_0.csv", index=False)
    _frame([["MSFT", "Microsoft", "2T", "H"]]).to_csv(tmp_path / "etoro_shard_1.csv", index=False)
    out = tmp_path / "etoro.csv"
    count = merge_shard_csvs(str(tmp_path), str(out))
    assert count == 2
    result = pd.read_csv(out)
    assert sorted(result["TKR"]) == ["AAPL", "MSFT"]


def test_merge_csvs_errors_on_no_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_shard_csvs(str(tmp_path), str(tmp_path / "etoro.csv"))


def test_merge_shards_cli_main(tmp_path):
    from scripts.merge_shards import main

    _frame([["AAPL", "Apple", "3T", "B"]]).to_csv(tmp_path / "etoro_shard_0.csv", index=False)
    _frame([["MSFT", "Microsoft", "2T", "H"]]).to_csv(tmp_path / "etoro_shard_1.csv", index=False)
    out = tmp_path / "etoro.csv"
    rc = main(["--shard-dir", str(tmp_path), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert sorted(pd.read_csv(out)["TKR"]) == ["AAPL", "MSFT"]


def test_merge_shards_runs_as_script(tmp_path):
    """`python scripts/merge_shards.py` (how CI invokes it) must resolve the
    trade_modules import. Script mode puts scripts/ on sys.path, not the repo
    root, so the script must bootstrap the root itself. Importing main() (above)
    can't catch this regression — only executing the script as a subprocess does."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    _frame([["AAPL", "Apple", "3T", "B"]]).to_csv(tmp_path / "etoro_shard_0.csv", index=False)
    _frame([["MSFT", "Microsoft", "2T", "H"]]).to_csv(tmp_path / "etoro_shard_1.csv", index=False)
    out = tmp_path / "etoro.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "merge_shards.py"),
            "--shard-dir",
            str(tmp_path),
            "--output",
            str(out),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

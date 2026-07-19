"""TDD — PIT fundamentals store (Sharadar SF1). The core primitive is a strict
point-in-time read: as of date T, only filings with datekey <= T are visible."""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.fundamentals_store import (
    append_records,
    read_asof,
    read_history,
    store_coverage,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # AAA: Q1 filed 03-15, Q2 filed 06-14 (a filing becomes visible on its datekey)
            {
                "ticker": "AAA",
                "datekey": "2026-03-15",
                "reportperiod": "2026-03-31",
                "assets": 100.0,
                "equity": 40.0,
            },
            {
                "ticker": "AAA",
                "datekey": "2026-06-14",
                "reportperiod": "2026-06-30",
                "assets": 110.0,
                "equity": 44.0,
            },
            {
                "ticker": "BBB",
                "datekey": "2026-05-01",
                "reportperiod": "2026-03-31",
                "assets": 200.0,
                "equity": 80.0,
            },
        ]
    )


def test_read_asof_is_point_in_time(tmp_path):
    store = str(tmp_path / "f.parquet")
    append_records(_rows(), store_path=store)
    # As of 05-01, AAA's Q2 (datekey 06-14) is NOT yet public -> must see Q1 only.
    asof = read_asof(["AAA", "BBB"], "2026-05-01", store_path=store)
    assert asof.loc["AAA", "assets"] == 100.0  # no look-ahead
    assert asof.loc["BBB", "assets"] == 200.0
    # As of 06-30, AAA's Q2 is public.
    asof2 = read_asof(["AAA"], "2026-06-30", store_path=store)
    assert asof2.loc["AAA", "assets"] == 110.0


def test_read_asof_excludes_ticker_with_no_visible_filing(tmp_path):
    store = str(tmp_path / "f.parquet")
    append_records(_rows(), store_path=store)
    # Before any BBB filing (datekey 05-01), BBB has nothing visible.
    asof = read_asof(["AAA", "BBB"], "2026-04-01", store_path=store)
    assert "AAA" in asof.index
    assert "BBB" not in asof.index


def test_append_only_and_newest_wins_on_restatement(tmp_path):
    store = str(tmp_path / "f.parquet")
    append_records(_rows(), store_path=store)
    # A restated refresh of the SAME (ticker, datekey) updates the value...
    append_records(
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "datekey": "2026-03-15",
                    "reportperiod": "2026-03-31",
                    "assets": 101.0,
                }
            ]
        ),
        store_path=store,
    )
    # ...and a delisted name's rows persist (append-only, never dropped).
    append_records(
        pd.DataFrame(
            [
                {
                    "ticker": "ZZZ",
                    "datekey": "2026-01-10",
                    "reportperiod": "2025-12-31",
                    "assets": 50.0,
                }
            ]
        ),
        store_path=store,
    )
    asof = read_asof(["AAA"], "2026-04-01", store_path=store)
    assert asof.loc["AAA", "assets"] == 101.0  # newest-wins on same (ticker, datekey)
    cov = store_coverage(store)
    assert cov["n_tickers"] == 3  # AAA, BBB, ZZZ all retained


def test_read_history_returns_visible_rows_sorted(tmp_path):
    store = str(tmp_path / "f.parquet")
    append_records(_rows(), store_path=store)
    hist = read_history(["AAA"], "2026-12-31", store_path=store)
    assert list(hist["datekey"]) == ["2026-03-15", "2026-06-14"]  # both, oldest-first
    # bounded by as_of
    hist2 = read_history(["AAA"], "2026-05-01", store_path=store)
    assert list(hist2["datekey"]) == ["2026-03-15"]


def test_empty_store_is_safe(tmp_path):
    store = str(tmp_path / "none.parquet")
    assert read_asof(["AAA"], "2026-05-01", store_path=store).empty
    assert store_coverage(store)["n_rows"] == 0

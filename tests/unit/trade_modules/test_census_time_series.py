"""
Tests for Census Time-Series Analysis — CIO Review v4 Finding F12.

Covers snapshot loading, holder-trend computation, classification
thresholds, Fear & Greed trend tracking, the high-level get_census_context
wrapper, corrupt-file handling, per-day deduplication, and edge cases.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from trade_modules.census_time_series import (
    CENSUS_ARCHIVE_DIR,
    _classify_trend,
    _parse_filename,
    _select_latest_per_day,
    compute_fear_greed_trend,
    compute_holder_trends,
    get_census_context,
    load_census_snapshots,
)

# ============================================================
# Helpers
# ============================================================

# Dynamic date generation to prevent test drift outside 30-day window
TODAY = date.today()
DATE_TODAY = TODAY.strftime("%Y-%m-%d")
DATE_2D_AGO = (TODAY - timedelta(days=2)).strftime("%Y-%m-%d")
DATE_4D_AGO = (TODAY - timedelta(days=4)).strftime("%Y-%m-%d")
DATE_6D_AGO = (TODAY - timedelta(days=6)).strftime("%Y-%m-%d")
DATE_8D_AGO = (TODAY - timedelta(days=8)).strftime("%Y-%m-%d")
DATE_10D_AGO = (TODAY - timedelta(days=10)).strftime("%Y-%m-%d")
DATE_12D_AGO = (TODAY - timedelta(days=12)).strftime("%Y-%m-%d")
DATE_14D_AGO = (TODAY - timedelta(days=14)).strftime("%Y-%m-%d")
DATE_16D_AGO = (TODAY - timedelta(days=16)).strftime("%Y-%m-%d")
DATE_20D_AGO = (TODAY - timedelta(days=20)).strftime("%Y-%m-%d")


def _make_census_file(
    directory: Path,
    date: str,
    time: str,
    holdings_data: dict[str, int],
    fg_index: int = 50,
    investor_count: int = 100,
    extra_instruments: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a minimal census JSON file matching the real structure."""
    instruments = [
        {"instrumentId": 1001, "symbolFull": "AAPL"},
        {"instrumentId": 1002, "symbolFull": "GOOG"},
        {"instrumentId": 1003, "symbolFull": "META"},
        {"instrumentId": 1004, "symbolFull": "MSFT"},
        {"instrumentId": 1005, "symbolFull": "AMZN"},
        {"instrumentId": 1006, "symbolFull": "NVDA"},
        {"instrumentId": 1007, "symbolFull": "TSLA"},
    ]
    if extra_instruments:
        instruments.extend(extra_instruments)

    # Map ticker -> instrumentId
    ticker_to_id = {i["symbolFull"]: i["instrumentId"] for i in instruments}

    top_holdings = []
    for ticker, count in holdings_data.items():
        iid = ticker_to_id.get(ticker, hash(ticker) % 90000 + 10000)
        top_holdings.append(
            {
                "instrumentId": iid,
                "symbol": ticker,
                "holdersCount": count,
            }
        )

    data = {
        "instruments": {"details": instruments},
        "analyses": [
            {
                "investorCount": investor_count,
                "fearGreedIndex": fg_index,
                "topHoldings": top_holdings,
            },
        ],
    }

    filename = f"etoro-data-{date}-{time}.json"
    fpath = directory / filename
    fpath.write_text(json.dumps(data), encoding="utf-8")
    return fpath


def _make_standard_snapshots(
    directory: Path,
) -> list[Path]:
    """Create a set of 4 census files spanning ~12 days for reuse."""
    files = []
    files.append(
        _make_census_file(
            directory,
            DATE_12D_AGO,
            "02-30",
            {"AAPL": 40, "GOOG": 35, "NVDA": 30, "TSLA": 25, "META": 20},
            fg_index=45,
        )
    )
    files.append(
        _make_census_file(
            directory,
            DATE_8D_AGO,
            "02-30",
            {"AAPL": 42, "GOOG": 34, "NVDA": 33, "TSLA": 23, "META": 21},
            fg_index=50,
        )
    )
    files.append(
        _make_census_file(
            directory,
            DATE_4D_AGO,
            "02-30",
            {"AAPL": 43, "GOOG": 33, "NVDA": 37, "TSLA": 20, "META": 22},
            fg_index=55,
        )
    )
    files.append(
        _make_census_file(
            directory,
            DATE_TODAY,
            "02-30",
            {"AAPL": 44, "GOOG": 32, "NVDA": 42, "TSLA": 18, "META": 22},
            fg_index=60,
        )
    )
    return files


# ============================================================
# Tests: filename parsing
# ============================================================


class TestParseFilename:
    def test_valid_filename(self) -> None:
        result = _parse_filename(f"etoro-data-{DATE_10D_AGO}-02-30.json")
        assert result == (DATE_10D_AGO, "02-30")

    def test_invalid_filename_returns_none(self) -> None:
        assert _parse_filename("not-a-census-file.json") is None
        assert _parse_filename("etoro-data-bad.json") is None
        assert _parse_filename(f"etoro-data-{DATE_10D_AGO}.csv") is None


# ============================================================
# Tests: per-day deduplication
# ============================================================


class TestSelectLatestPerDay:
    def test_dedup_keeps_latest(self, tmp_path: Path) -> None:
        """When two files exist for the same day, keep the later one."""
        f1 = _make_census_file(tmp_path, DATE_12D_AGO, "00-07", {"AAPL": 10}, fg_index=40)
        f2 = _make_census_file(tmp_path, DATE_12D_AGO, "02-31", {"AAPL": 12}, fg_index=42)

        result = _select_latest_per_day([f1, f2])
        assert len(result) == 1
        assert result[0][0] == DATE_12D_AGO
        assert result[0][1] == f2  # later time wins

    def test_multiple_days_sorted(self, tmp_path: Path) -> None:
        f1 = _make_census_file(tmp_path, DATE_16D_AGO, "02-30", {"AAPL": 10})
        f2 = _make_census_file(tmp_path, DATE_14D_AGO, "02-30", {"AAPL": 11})
        f3 = _make_census_file(tmp_path, DATE_20D_AGO, "02-30", {"AAPL": 9})

        result = _select_latest_per_day([f1, f2, f3])
        dates = [r[0] for r in result]
        assert dates == [DATE_20D_AGO, DATE_16D_AGO, DATE_14D_AGO]

    def test_non_matching_files_skipped(self, tmp_path: Path) -> None:
        _make_census_file(tmp_path, DATE_16D_AGO, "02-30", {"AAPL": 10})
        junk = tmp_path / "random.json"
        junk.write_text("{}", encoding="utf-8")

        all_files = list(tmp_path.iterdir())
        result = _select_latest_per_day(all_files)
        assert len(result) == 1


# ============================================================
# Tests: snapshot loading
# ============================================================


class TestLoadCensusSnapshots:
    def test_loads_snapshots(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        assert len(snaps) == 4
        assert snaps[0]["date"] < snaps[-1]["date"]
        assert "AAPL" in snaps[0]["holdings"]
        assert snaps[0]["fear_greed"] == 45

    def test_days_back_filter(self, tmp_path: Path) -> None:
        """Only snapshots within days_back window are returned."""
        _make_standard_snapshots(tmp_path)
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=5)
        # Only DATE_TODAY and DATE_4D_AGO should be returned
        # All dates should be >= (now - 5 days)
        cutoff_date = (TODAY - timedelta(days=5)).strftime("%Y-%m-%d")
        for snap in snaps:
            assert snap["date"] >= cutoff_date

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        snaps = load_census_snapshots(archive_dir=tmp_path / "nonexistent", days_back=30)
        assert snaps == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        assert snaps == []

    def test_corrupt_file_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSON files are skipped, valid ones still load."""
        _make_census_file(
            tmp_path,
            DATE_16D_AGO,
            "02-30",
            {"AAPL": 40},
            fg_index=50,
        )
        corrupt = tmp_path / f"etoro-data-{DATE_14D_AGO}-02-30.json"
        corrupt.write_text("{bad json!!", encoding="utf-8")
        _make_census_file(
            tmp_path,
            DATE_12D_AGO,
            "02-30",
            {"AAPL": 42},
            fg_index=52,
        )

        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        assert len(snaps) == 2
        assert snaps[0]["date"] == DATE_16D_AGO
        assert snaps[1]["date"] == DATE_12D_AGO

    def test_dedup_in_loading(self, tmp_path: Path) -> None:
        """Two files for the same day: only latest one is used."""
        _make_census_file(tmp_path, DATE_12D_AGO, "00-07", {"AAPL": 10}, fg_index=40)
        _make_census_file(tmp_path, DATE_12D_AGO, "02-31", {"AAPL": 12}, fg_index=42)

        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        assert len(snaps) == 1
        assert snaps[0]["holdings"]["AAPL"] == 12
        assert snaps[0]["fear_greed"] == 42

    def test_investor_tier_selection(self, tmp_path: Path) -> None:
        """Only the matching investor tier analysis is extracted."""
        data = {
            "instruments": {
                "details": [
                    {"instrumentId": 1001, "symbolFull": "AAPL"},
                ]
            },
            "analyses": [
                {
                    "investorCount": 100,
                    "fearGreedIndex": 50,
                    "topHoldings": [{"instrumentId": 1001, "symbol": "AAPL", "holdersCount": 40}],
                },
                {
                    "investorCount": 500,
                    "fearGreedIndex": 55,
                    "topHoldings": [{"instrumentId": 1001, "symbol": "AAPL", "holdersCount": 200}],
                },
            ],
        }
        fpath = tmp_path / f"etoro-data-{DATE_10D_AGO}-02-30.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        snaps_100 = load_census_snapshots(archive_dir=tmp_path, days_back=30, investor_tier=100)
        assert snaps_100[0]["holdings"]["AAPL"] == 40

        snaps_500 = load_census_snapshots(archive_dir=tmp_path, days_back=30, investor_tier=500)
        assert snaps_500[0]["holdings"]["AAPL"] == 200

    def test_missing_tier_skips_file(self, tmp_path: Path) -> None:
        """If the requested tier is missing in a file, that file is skipped."""
        _make_census_file(
            tmp_path,
            DATE_10D_AGO,
            "02-30",
            {"AAPL": 40},
            fg_index=50,
            investor_count=100,
        )
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30, investor_tier=500)
        assert snaps == []

    def test_instrument_map_fallback(self, tmp_path: Path) -> None:
        """Holdings without 'symbol' fall back to instruments.details mapping."""
        data = {
            "instruments": {
                "details": [
                    {"instrumentId": 1001, "symbolFull": "AAPL"},
                ]
            },
            "analyses": [
                {
                    "investorCount": 100,
                    "fearGreedIndex": 50,
                    "topHoldings": [
                        {"instrumentId": 1001, "holdersCount": 40},
                    ],
                },
            ],
        }
        fpath = tmp_path / f"etoro-data-{DATE_10D_AGO}-02-30.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        assert snaps[0]["holdings"]["AAPL"] == 40


# ============================================================
# Tests: holder trend computation
# ============================================================


class TestComputeHolderTrends:
    def test_basic_trends(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        trends = compute_holder_trends(snaps)

        # NVDA went 30 -> 42 = +40% — strong accumulation
        assert trends["NVDA"]["current_holders"] == 42
        assert trends["NVDA"]["delta_30d"] == 12
        assert trends["NVDA"]["pct_change_30d"] == pytest.approx(40.0)
        assert trends["NVDA"]["classification"] == "strong_accumulation"

        # TSLA went 25 -> 18 = -28% — strong distribution
        assert trends["TSLA"]["current_holders"] == 18
        assert trends["TSLA"]["delta_30d"] == -7
        assert trends["TSLA"]["classification"] == "strong_distribution"

    def test_stable_classification(self, tmp_path: Path) -> None:
        """Small changes are classified as stable."""
        _make_census_file(tmp_path, DATE_12D_AGO, "02-30", {"AAPL": 40}, fg_index=50)
        _make_census_file(tmp_path, DATE_TODAY, "02-30", {"AAPL": 41}, fg_index=51)

        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        trends = compute_holder_trends(snaps)
        assert trends["AAPL"]["classification"] == "stable"

    def test_holder_pct_computed(self, tmp_path: Path) -> None:
        _make_census_file(tmp_path, DATE_12D_AGO, "02-30", {"AAPL": 40})
        _make_census_file(tmp_path, DATE_TODAY, "02-30", {"AAPL": 50})

        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        trends = compute_holder_trends(snaps)
        assert trends["AAPL"]["holder_pct"] == pytest.approx(50.0)  # 50/100 * 100

    def test_7d_delta(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        snaps = load_census_snapshots(archive_dir=tmp_path, days_back=30)
        trends = compute_holder_trends(snaps)

        # 7d ago from today => closest is DATE_8D_AGO
        # NVDA was 33 on DATE_8D_AGO, now 42 => delta_7d = 9
        assert trends["NVDA"]["delta_7d"] == 9

    def test_single_snapshot_returns_empty(self) -> None:
        snaps = [
            {
                "date": DATE_10D_AGO,
                "holdings": {"AAPL": 40},
                "fear_greed": 50,
                "investor_count": 100,
            }
        ]
        trends = compute_holder_trends(snaps)
        assert trends == {}

    def test_empty_snapshots_returns_empty(self) -> None:
        assert compute_holder_trends([]) == {}

    def test_ticker_only_in_earliest_excluded(self) -> None:
        """A ticker present in earliest but not latest is excluded."""
        snaps = [
            {
                "date": DATE_20D_AGO,
                "holdings": {"AAPL": 40, "DOGE": 5},
                "fear_greed": 50,
                "investor_count": 100,
            },
            {
                "date": DATE_10D_AGO,
                "holdings": {"AAPL": 42},
                "fear_greed": 55,
                "investor_count": 100,
            },
        ]
        trends = compute_holder_trends(snaps)
        assert "DOGE" not in trends
        assert "AAPL" in trends

    def test_ticker_appearing_only_once_excluded(self) -> None:
        """Ticker in only 1 snapshot is excluded from trends."""
        snaps = [
            {
                "date": DATE_20D_AGO,
                "holdings": {"AAPL": 40},
                "fear_greed": 50,
                "investor_count": 100,
            },
            {
                "date": DATE_10D_AGO,
                "holdings": {"AAPL": 42, "NEW": 5},
                "fear_greed": 55,
                "investor_count": 100,
            },
        ]
        trends = compute_holder_trends(snaps)
        assert "NEW" not in trends


# ============================================================
# Tests: classification thresholds
# ============================================================


class TestClassifyTrend:
    def test_strong_accumulation_pp(self) -> None:
        assert _classify_trend(11.0, 0.0) == "strong_accumulation"

    def test_strong_accumulation_pct(self) -> None:
        assert _classify_trend(0.0, 21.0) == "strong_accumulation"

    def test_accumulation_pp(self) -> None:
        assert _classify_trend(4.0, 0.0) == "accumulation"

    def test_accumulation_pct(self) -> None:
        assert _classify_trend(0.0, 6.0) == "accumulation"

    def test_strong_distribution_pp(self) -> None:
        assert _classify_trend(-11.0, 0.0) == "strong_distribution"

    def test_strong_distribution_pct(self) -> None:
        assert _classify_trend(0.0, -21.0) == "strong_distribution"

    def test_distribution_pp(self) -> None:
        assert _classify_trend(-4.0, 0.0) == "distribution"

    def test_distribution_pct(self) -> None:
        assert _classify_trend(0.0, -6.0) == "distribution"

    def test_stable(self) -> None:
        assert _classify_trend(1.0, 2.0) == "stable"
        assert _classify_trend(-1.0, -2.0) == "stable"
        assert _classify_trend(0.0, 0.0) == "stable"

    def test_boundary_values(self) -> None:
        """Exact threshold values fall into moderate, not strong."""
        assert _classify_trend(10.0, 0.0) == "accumulation"
        assert _classify_trend(0.0, 20.0) == "accumulation"
        assert _classify_trend(-10.0, 0.0) == "distribution"
        assert _classify_trend(0.0, -20.0) == "distribution"

    def test_exact_moderate_boundary(self) -> None:
        """Exact moderate threshold values fall into stable."""
        assert _classify_trend(3.0, 0.0) == "stable"
        assert _classify_trend(0.0, 5.0) == "stable"
        assert _classify_trend(-3.0, 0.0) == "stable"
        assert _classify_trend(0.0, -5.0) == "stable"


# ============================================================
# Tests: Fear & Greed trend
# ============================================================


class TestComputeFearGreedTrend:
    def test_rising_trend(self) -> None:
        snaps = [
            {"date": DATE_20D_AGO, "holdings": {}, "fear_greed": 45, "investor_count": 100},
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 50, "investor_count": 100},
            {"date": DATE_2D_AGO, "holdings": {}, "fear_greed": 60, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["current"] == 60
        assert result["7d_ago"] == 50
        assert result["delta_7d"] == 10
        assert result["trend"] == "rising"

    def test_falling_trend(self) -> None:
        snaps = [
            {"date": DATE_20D_AGO, "holdings": {}, "fear_greed": 70, "investor_count": 100},
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 65, "investor_count": 100},
            {"date": DATE_2D_AGO, "holdings": {}, "fear_greed": 55, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["current"] == 55
        assert result["trend"] == "falling"
        assert result["delta_7d"] == -10

    def test_flat_trend(self) -> None:
        snaps = [
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 50, "investor_count": 100},
            {"date": DATE_2D_AGO, "holdings": {}, "fear_greed": 51, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["trend"] == "flat"

    def test_empty_snapshots(self) -> None:
        result = compute_fear_greed_trend([])
        assert result["current"] is None
        assert result["trend"] == "unknown"
        assert result["delta_7d"] is None

    def test_single_snapshot(self) -> None:
        snaps = [
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 60, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["current"] == 60
        assert result["7d_ago"] is None
        assert result["trend"] == "unknown"

    def test_30d_delta(self) -> None:
        snaps = [
            {"date": "2026-02-10", "holdings": {}, "fear_greed": 30, "investor_count": 100},
            {"date": DATE_16D_AGO, "holdings": {}, "fear_greed": 50, "investor_count": 100},
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 60, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["30d_ago"] == 30
        assert result["delta_30d"] == 30

    def test_none_fear_greed_values(self) -> None:
        """Snapshots with None fear_greed are skipped for history."""
        snaps = [
            {"date": DATE_20D_AGO, "holdings": {}, "fear_greed": 40, "investor_count": 100},
            {"date": DATE_16D_AGO, "holdings": {}, "fear_greed": None, "investor_count": 100},
            {"date": DATE_10D_AGO, "holdings": {}, "fear_greed": 60, "investor_count": 100},
        ]
        result = compute_fear_greed_trend(snaps)
        assert result["current"] == 60
        assert result["7d_ago"] == 40


# ============================================================
# Tests: get_census_context integration
# ============================================================


class TestGetCensusContext:
    def test_full_context(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)

        assert ctx["data_available"] is True
        assert ctx["snapshots_loaded"] == 4
        assert ctx["date_range"]["start"] == DATE_12D_AGO
        assert ctx["date_range"]["end"] == DATE_TODAY
        assert ctx["fear_greed"]["current"] == 60
        assert len(ctx["top_accumulating"]) > 0
        assert len(ctx["top_distributing"]) > 0
        assert isinstance(ctx["ticker_trends"], dict)
        assert "Census:" in ctx["summary"]

    def test_no_data_available(self, tmp_path: Path) -> None:
        ctx = get_census_context(archive_dir=tmp_path / "nope", days_back=30)
        assert ctx["data_available"] is False
        assert ctx["snapshots_loaded"] == 0
        assert ctx["summary"] == "Census: No snapshots available."

    def test_top_accumulating_sorted(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)
        pct_changes = [t["pct_change_30d"] for t in ctx["top_accumulating"]]
        assert pct_changes == sorted(pct_changes, reverse=True)

    def test_top_distributing_sorted(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)
        if ctx["top_distributing"]:
            pct_changes = [t["pct_change_30d"] for t in ctx["top_distributing"]]
            assert pct_changes == sorted(pct_changes)

    def test_summary_includes_fg(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)
        assert "F&G:" in ctx["summary"]

    def test_summary_includes_accumulation(self, tmp_path: Path) -> None:
        _make_standard_snapshots(tmp_path)
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)
        assert "Top accumulation:" in ctx["summary"]

    def test_context_with_single_snapshot(self, tmp_path: Path) -> None:
        """Single snapshot loads but trends are empty."""
        _make_census_file(
            tmp_path,
            DATE_10D_AGO,
            "02-30",
            {"AAPL": 40},
            fg_index=55,
        )
        ctx = get_census_context(archive_dir=tmp_path, days_back=30)
        assert ctx["data_available"] is True
        assert ctx["snapshots_loaded"] == 1
        assert ctx["ticker_trends"] == {}
        assert ctx["top_accumulating"] == []

    def test_census_archive_dir_constant(self) -> None:
        """Verify the default constant points to the expected path."""
        assert str(CENSUS_ARCHIVE_DIR).endswith("SourceCode/etoro_census/archive/data")

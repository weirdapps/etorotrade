"""Tests for trade_modules.riskfirst.news_gate — TDD RED->GREEN.

Run: cd ~/SourceCode/etorotrade && python3 -m pytest tests/unit/trade_modules/test_riskfirst_news_gate.py -q
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from trade_modules.riskfirst.news_gate import (
    _to_date,
    apply_exclusions,
    earnings_blackout,
    load_event_risk,
)

# ---------------------------------------------------------------------------
# _to_date helpers
# ---------------------------------------------------------------------------


class TestToDate:
    def test_iso_string_parses(self):
        assert _to_date("2026-07-10") == date(2026, 7, 10)

    def test_date_passthrough(self):
        d = date(2026, 7, 10)
        assert _to_date(d) == d

    def test_datetime_converted(self):
        dt = datetime(2026, 7, 10, 9, 30)
        assert _to_date(dt) == date(2026, 7, 10)

    def test_iso_string_with_time(self):
        assert _to_date("2026-07-10T09:30:00") == date(2026, 7, 10)

    def test_none_returns_none(self):
        assert _to_date(None) is None

    def test_empty_string_returns_none(self):
        assert _to_date("") is None

    def test_garbage_string_returns_none(self):
        assert _to_date("not-a-date") is None

    def test_integer_returns_none(self):
        assert _to_date(12345) is None


# ---------------------------------------------------------------------------
# earnings_blackout
# ---------------------------------------------------------------------------


class TestEarningsBlackout:
    """Inclusive window: [as_of, as_of + blackout_days]."""

    def _as_of(self):
        return date(2026, 7, 10)

    def test_earnings_exactly_on_as_of_is_blacked_out(self):
        result = earnings_blackout({"AAPL": "2026-07-10"}, self._as_of(), blackout_days=7)
        assert "AAPL" in result

    def test_earnings_exactly_on_last_day_is_blacked_out(self):
        # as_of + 7 = 2026-07-17
        result = earnings_blackout({"AAPL": "2026-07-17"}, self._as_of(), blackout_days=7)
        assert "AAPL" in result

    def test_earnings_inside_window_is_blacked_out(self):
        result = earnings_blackout({"AAPL": "2026-07-14"}, self._as_of(), blackout_days=7)
        assert "AAPL" in result

    def test_earnings_one_day_past_window_not_blacked_out(self):
        # as_of + 8 = 2026-07-18
        result = earnings_blackout({"AAPL": "2026-07-18"}, self._as_of(), blackout_days=7)
        assert "AAPL" not in result

    def test_past_earnings_not_blacked_out(self):
        # Before as_of — already happened
        result = earnings_blackout({"AAPL": "2026-07-09"}, self._as_of(), blackout_days=7)
        assert "AAPL" not in result

    def test_none_date_not_blacked_out(self):
        result = earnings_blackout({"AAPL": None}, self._as_of(), blackout_days=7)
        assert "AAPL" not in result

    def test_missing_date_key_not_blacked_out(self):
        result = earnings_blackout({}, self._as_of(), blackout_days=7)
        assert len(result) == 0

    def test_unparseable_date_not_blacked_out(self):
        result = earnings_blackout({"AAPL": "garbage"}, self._as_of(), blackout_days=7)
        assert "AAPL" not in result

    def test_mixed_date_types_handled(self):
        earnings_map = {
            "AAPL": date(2026, 7, 14),  # date object — inside window
            "MSFT": datetime(2026, 7, 12),  # datetime object — inside window
            "GOOG": "2026-07-20",  # string outside window
            "TSLA": None,  # None
            "AMZN": "2026-07-17",  # string on boundary
        }
        result = earnings_blackout(earnings_map, self._as_of(), blackout_days=7)
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" not in result
        assert "TSLA" not in result
        assert "AMZN" in result

    def test_as_of_as_string_supported(self):
        # as_of can itself be a date-like string
        result = earnings_blackout({"AAPL": "2026-07-14"}, "2026-07-10", blackout_days=7)
        assert "AAPL" in result

    def test_returns_set(self):
        result = earnings_blackout({"AAPL": "2026-07-14"}, self._as_of())
        assert isinstance(result, set)

    def test_empty_map_returns_empty_set(self):
        result = earnings_blackout({}, self._as_of())
        assert result == set()

    def test_default_blackout_days_is_7(self):
        # On as_of + 7 should be blacked out by default
        result = earnings_blackout({"AAPL": "2026-07-17"}, self._as_of())
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# apply_exclusions
# ---------------------------------------------------------------------------


def _make_df(tickers):
    """Helper: DataFrame with tickers as index and a dummy 'score' column."""
    return pd.DataFrame({"score": range(len(tickers))}, index=tickers)


class TestApplyExclusions:
    def test_excluded_tickers_removed(self):
        df = _make_df(["AAPL", "MSFT", "GOOG"])
        result = apply_exclusions(df, {"AAPL", "GOOG"})
        assert list(result.index) == ["MSFT"]

    def test_non_excluded_tickers_kept(self):
        df = _make_df(["AAPL", "MSFT", "GOOG"])
        result = apply_exclusions(df, {"AAPL"})
        assert "MSFT" in result.index
        assert "GOOG" in result.index

    def test_empty_exclude_returns_df_unchanged(self):
        df = _make_df(["AAPL", "MSFT"])
        result = apply_exclusions(df, set())
        assert list(result.index) == list(df.index)

    def test_none_exclude_returns_df_unchanged(self):
        df = _make_df(["AAPL", "MSFT"])
        result = apply_exclusions(df, None)
        assert list(result.index) == list(df.index)

    def test_exclude_ticker_not_in_df_is_harmless(self):
        df = _make_df(["AAPL", "MSFT"])
        result = apply_exclusions(df, {"NVDA"})
        assert list(result.index) == ["AAPL", "MSFT"]

    def test_does_not_mutate_input_df(self):
        df = _make_df(["AAPL", "MSFT", "GOOG"])
        original_index = list(df.index)
        apply_exclusions(df, {"AAPL"})
        assert list(df.index) == original_index

    def test_all_excluded_returns_empty_df(self):
        df = _make_df(["AAPL", "MSFT"])
        result = apply_exclusions(df, {"AAPL", "MSFT"})
        assert len(result) == 0

    def test_exclude_accepts_list(self):
        df = _make_df(["AAPL", "MSFT", "GOOG"])
        result = apply_exclusions(df, ["AAPL", "GOOG"])
        assert list(result.index) == ["MSFT"]

    def test_exclude_accepts_empty_list(self):
        df = _make_df(["AAPL", "MSFT"])
        result = apply_exclusions(df, [])
        assert list(result.index) == ["AAPL", "MSFT"]

    def test_result_columns_preserved(self):
        df = _make_df(["AAPL", "MSFT"])
        df["extra"] = [10, 20]
        result = apply_exclusions(df, {"AAPL"})
        assert "score" in result.columns
        assert "extra" in result.columns


# ---------------------------------------------------------------------------
# load_event_risk
# ---------------------------------------------------------------------------


class TestLoadEventRisk:
    def test_missing_file_returns_empty_set(self):
        result = load_event_risk("/nonexistent/path/event_risk.json")
        assert result == set()

    def test_json_array_returns_set(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["AAPL", "MSFT", "GOOG"], f)
            path = f.name
        result = load_event_risk(path)
        assert result == {"AAPL", "MSFT", "GOOG"}

    def test_json_array_upper_cased(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["aapl", "Msft"], f)
            path = f.name
        result = load_event_risk(path)
        assert result == {"AAPL", "MSFT"}

    def test_json_object_keys_used(self):
        data = {"AAPL": {"reason": "scandal"}, "TSLA": {"reason": "litigation"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        result = load_event_risk(path)
        assert result == {"AAPL", "TSLA"}

    def test_json_object_keys_upper_cased(self):
        data = {"aapl": {}, "tsla": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        result = load_event_risk(path)
        assert result == {"AAPL", "TSLA"}

    def test_garbage_file_returns_empty_set(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not json {{{")
            path = f.name
        result = load_event_risk(path)
        assert result == set()

    def test_empty_file_returns_empty_set(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            path = f.name
        result = load_event_risk(path)
        assert result == set()

    def test_returns_set_type(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["AAPL"], f)
            path = f.name
        result = load_event_risk(path)
        assert isinstance(result, set)

    def test_empty_array_returns_empty_set(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            path = f.name
        result = load_event_risk(path)
        assert result == set()

    def test_path_object_supported(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["AAPL"], f)
            path = Path(f.name)
        result = load_event_risk(path)
        assert result == {"AAPL"}

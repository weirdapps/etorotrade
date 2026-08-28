"""Tests for the persistent IPO-date cache.

The load-bearing property in this file is negative: **a failed probe must never
be written to disk**. The pre-existing in-memory cache stored ``None`` both for
"the provider has no history for this ticker" and for "the fetch blew up".
Persisting that conflation would write a transient network failure down forever
and silently change ``sell_criteria`` for that name on every future run.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from trade_modules import ipo_cache
from trade_modules.analysis import signals
from trade_modules.analysis.signals import is_recent_ipo

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class FakeYamlConfig:
    def __init__(self, config):
        self._config = config

    def load_config(self):
        return self._config


def make_config(known_ipos=None, enabled=True, auto_detect=True):
    cfg = {"ipo_grace_period": {"enabled": enabled, "auto_detect": auto_detect}}
    if known_ipos:
        cfg["ipo_grace_period"]["known_ipos"] = known_ipos
    return cfg


def hist_frame(first_trade: datetime) -> pd.DataFrame:
    index = pd.DatetimeIndex([first_trade, first_trade + timedelta(days=1)])
    return pd.DataFrame({"Close": [100.0, 101.0]}, index=index)


@pytest.fixture
def cache_path(tmp_path) -> Path:
    return tmp_path / "ipo_dates.json"


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point the module singleton at a temp file and clear the in-memory cache."""
    path = tmp_path / "ipo_dates.json"
    monkeypatch.setenv(ipo_cache.ENV_CACHE_PATH, str(path))
    ipo_cache.reset_cache()
    signals._ipo_date_cache.clear()
    yield path
    ipo_cache.reset_cache()
    signals._ipo_date_cache.clear()


# --------------------------------------------------------------------------
# 1. the three-state probe, the prerequisite for everything else
# --------------------------------------------------------------------------


class TestThreeStateProbe:
    """``None`` used to mean two different things. Now it means neither."""

    @patch("yfinance.Ticker")
    def test_a_real_first_bar_is_FOUND(self, mock_ticker_cls):
        first = datetime(2019, 3, 14, 0, 0, 0)
        mock_ticker_cls.return_value.history.return_value = hist_frame(first)

        status, value = ipo_cache.probe_first_trade_date("REAL")

        assert status == ipo_cache.PROBE_FOUND
        assert value == first

    @patch("yfinance.Ticker")
    def test_an_empty_frame_is_NO_DATA_not_an_error(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        status, value = ipo_cache.probe_first_trade_date("EMPTY")

        assert status == ipo_cache.PROBE_NO_DATA
        assert value is None

    @patch("yfinance.Ticker")
    def test_an_exception_is_ERROR_not_no_data(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = RuntimeError("connection reset")

        status, value = ipo_cache.probe_first_trade_date("BOOM")

        assert status == ipo_cache.PROBE_ERROR
        assert value is None

    def test_the_three_states_are_distinct(self):
        states = {ipo_cache.PROBE_FOUND, ipo_cache.PROBE_NO_DATA, ipo_cache.PROBE_ERROR}
        assert len(states) == 3

    @patch("yfinance.Ticker")
    def test_the_probe_strips_tzinfo_exactly_as_the_old_inline_code_did(self, mock_ticker_cls):
        aware = pd.DatetimeIndex(["2021-06-01 09:30:00"], tz="America/New_York")
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame(
            {"Close": [1.0]}, index=aware
        )

        status, value = ipo_cache.probe_first_trade_date("TZ")

        assert status == ipo_cache.PROBE_FOUND
        assert value.tzinfo is None
        assert value == datetime(2021, 6, 1, 9, 30, 0)


# --------------------------------------------------------------------------
# 2. only confirmed positive knowledge is persisted
# --------------------------------------------------------------------------


class TestOnlyPositiveKnowledgePersists:
    def test_put_then_reload_round_trips_the_exact_datetime(self, cache_path):
        # Not date-only: the comparison in is_recent_ipo is datetime > datetime,
        # so dropping the time component could flip a verdict at the boundary.
        moment = datetime(2024, 11, 7, 14, 30, 15)
        c = ipo_cache.IpoDateCache(cache_path)
        c.put("EXACT", moment)
        assert c.save() is True

        reloaded = ipo_cache.IpoDateCache(cache_path)
        assert reloaded.get("EXACT") == moment

    @pytest.mark.parametrize("bad", [None, "2024-01-01", 1704067200, float("nan")])
    def test_put_REFUSES_anything_that_is_not_a_datetime(self, cache_path, bad):
        c = ipo_cache.IpoDateCache(cache_path)
        with pytest.raises(TypeError):
            c.put("BAD", bad)
        assert c.get("BAD") is None

    def test_save_is_a_NO_OP_when_nothing_new_was_learned(self, cache_path):
        c = ipo_cache.IpoDateCache(cache_path)
        assert c.save() is False
        assert not cache_path.exists()

    def test_a_clean_reload_does_not_rewrite_the_file(self, cache_path):
        c = ipo_cache.IpoDateCache(cache_path)
        c.put("A", datetime(2020, 1, 1))
        c.save()
        before = cache_path.read_bytes()

        again = ipo_cache.IpoDateCache(cache_path)
        assert again.save() is False
        assert cache_path.read_bytes() == before

    def test_the_file_carries_a_schema_version(self, cache_path):
        c = ipo_cache.IpoDateCache(cache_path)
        c.put("A", datetime(2020, 1, 1))
        c.save()

        payload = json.loads(cache_path.read_text())
        assert payload["schema_version"] == ipo_cache.SCHEMA_VERSION

    def test_the_write_is_atomic_and_leaves_no_temp_files(self, cache_path):
        c = ipo_cache.IpoDateCache(cache_path)
        c.put("A", datetime(2020, 1, 1))
        c.save()
        leftovers = [p.name for p in cache_path.parent.iterdir() if p.name != cache_path.name]
        assert leftovers == []


# --------------------------------------------------------------------------
# 3. a damaged cache fails OPEN: behaves as empty, never raises
# --------------------------------------------------------------------------


class TestFailsOpen:
    @pytest.mark.parametrize(
        "content",
        [
            "",  # empty file
            "{",  # truncated
            "not json at all",
            "[]",  # right JSON, wrong shape
            '{"dates": {"A": "2020-01-01T00:00:00"}}',  # no schema_version
            '{"schema_version": 1}',  # no dates
            '{"schema_version": 1, "dates": "nope"}',  # dates not a mapping
            '{"schema_version": 99, "dates": {"A": "2020-01-01T00:00:00"}}',  # future schema
        ],
        ids=[
            "empty",
            "truncated",
            "garbage",
            "wrong-shape",
            "no-version",
            "no-dates",
            "dates-not-a-map",
            "future-version",
        ],
    )
    def test_a_damaged_file_reads_as_EMPTY_and_never_raises(self, cache_path, content):
        cache_path.write_text(content)

        c = ipo_cache.IpoDateCache(cache_path)

        assert c.get("A") is None
        assert len(c) == 0

    def test_a_single_unparseable_entry_does_not_poison_the_good_ones(self, cache_path):
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": ipo_cache.SCHEMA_VERSION,
                    "dates": {
                        "GOOD": "2020-01-01T00:00:00",
                        "BAD": "the-fourteenth-of-never",
                        "ALSOBAD": None,
                        "NOTASTRING": 17,
                        "ALSOGOOD": "2021-02-03T04:05:06",
                    },
                }
            )
        )

        c = ipo_cache.IpoDateCache(cache_path)

        assert c.get("GOOD") == datetime(2020, 1, 1)
        assert c.get("ALSOGOOD") == datetime(2021, 2, 3, 4, 5, 6)
        assert c.get("BAD") is None
        assert c.get("ALSOBAD") is None
        assert c.get("NOTASTRING") is None

    def test_a_directory_where_the_file_should_be_is_not_fatal(self, tmp_path):
        weird = tmp_path / "ipo_dates.json"
        weird.mkdir()

        c = ipo_cache.IpoDateCache(weird)

        assert len(c) == 0
        c.put("A", datetime(2020, 1, 1))
        assert c.save() is False  # could not write, but did not raise

    def test_an_unwritable_directory_is_not_fatal(self, tmp_path):
        c = ipo_cache.IpoDateCache(tmp_path / "nope" / "\0bad" / "ipo_dates.json")
        c.put("A", datetime(2020, 1, 1))
        assert c.save() is False


# --------------------------------------------------------------------------
# 4. the cache is bounded
# --------------------------------------------------------------------------


class TestBounded:
    def test_entries_beyond_the_cap_are_dropped_deterministically(self, cache_path, monkeypatch):
        monkeypatch.setattr(ipo_cache, "MAX_ENTRIES", 5)
        c = ipo_cache.IpoDateCache(cache_path)
        for i in range(12):
            c.put(f"T{i:02d}", datetime(2020, 1, 1) + timedelta(days=i))
        c.save()

        reloaded = ipo_cache.IpoDateCache(cache_path)
        assert len(reloaded) == 5
        assert sorted(reloaded.tickers()) == ["T00", "T01", "T02", "T03", "T04"]

    def test_an_oversized_file_on_disk_is_truncated_on_read(self, cache_path, monkeypatch):
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": ipo_cache.SCHEMA_VERSION,
                    "dates": {f"T{i:02d}": "2020-01-01T00:00:00" for i in range(20)},
                }
            )
        )
        monkeypatch.setattr(ipo_cache, "MAX_ENTRIES", 3)

        c = ipo_cache.IpoDateCache(cache_path)

        assert len(c) == 3


# --------------------------------------------------------------------------
# 5. merging shard caches (how CI accrues the benefit)
# --------------------------------------------------------------------------


class TestMerge:
    def _write(self, path, dates):
        path.write_text(
            json.dumps(
                {
                    "schema_version": ipo_cache.SCHEMA_VERSION,
                    "dates": {k: v.isoformat() for k, v in dates.items()},
                }
            )
        )

    def test_merge_is_a_union(self, tmp_path):
        a, b, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "out.json"
        self._write(a, {"AAA": datetime(2020, 1, 1)})
        self._write(b, {"BBB": datetime(2021, 1, 1)})

        n = ipo_cache.merge_cache_files([a, b], out)

        assert n == 2
        merged = ipo_cache.IpoDateCache(out)
        assert merged.get("AAA") == datetime(2020, 1, 1)
        assert merged.get("BBB") == datetime(2021, 1, 1)

    def test_on_conflict_the_EARLIEST_date_wins_regardless_of_file_order(self, tmp_path):
        # An earlier first bar means a longer history was retrieved. It is also
        # the fail-closed direction: earlier means not a recent IPO, so strict criteria.
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        self._write(a, {"X": datetime(2015, 5, 5)})
        self._write(b, {"X": datetime(2024, 5, 5)})

        for i, order in enumerate(([a, b], [b, a])):
            out = tmp_path / f"out_{i}.json"
            ipo_cache.merge_cache_files(order, out)
            assert ipo_cache.IpoDateCache(out).get("X") == datetime(2015, 5, 5)

    def test_a_corrupt_shard_file_is_skipped_not_fatal(self, tmp_path):
        a, bad, out = tmp_path / "a.json", tmp_path / "bad.json", tmp_path / "out.json"
        self._write(a, {"AAA": datetime(2020, 1, 1)})
        bad.write_text("{{{")

        n = ipo_cache.merge_cache_files([a, bad, out], out)

        assert n == 1
        assert ipo_cache.IpoDateCache(out).get("AAA") == datetime(2020, 1, 1)

    def test_merging_nothing_writes_nothing(self, tmp_path):
        out = tmp_path / "out.json"
        assert ipo_cache.merge_cache_files([], out) == 0
        assert not out.exists()


# --------------------------------------------------------------------------
# 6. is_recent_ipo end to end: the behaviour that must not change
# --------------------------------------------------------------------------


class TestIsRecentIpoUsesTheCache:
    @patch("yfinance.Ticker")
    def test_POSITIVE_CONTROL_a_cold_cache_really_does_reach_the_network(
        self, mock_ticker_cls, wired
    ):
        """Without this, "a hit avoids the network" could pass vacuously."""
        mock_ticker_cls.return_value.history.return_value = hist_frame(
            datetime.now() - timedelta(days=90)
        )

        assert is_recent_ipo("COLD", FakeYamlConfig(make_config())) is True
        assert mock_ticker_cls.return_value.history.call_count == 1

    @patch("yfinance.Ticker")
    def test_a_warm_DISK_cache_avoids_the_network_in_a_fresh_process(self, mock_ticker_cls, wired):
        seed = ipo_cache.IpoDateCache(wired)
        seed.put("WARM", datetime.now() - timedelta(days=90))
        seed.save()
        # Simulate a brand-new process: singleton dropped, memory cleared.
        ipo_cache.reset_cache()
        signals._ipo_date_cache.clear()

        assert is_recent_ipo("WARM", FakeYamlConfig(make_config())) is True
        assert mock_ticker_cls.return_value.history.call_count == 0

    @patch("yfinance.Ticker")
    def test_a_miss_fetches_and_PERSISTS(self, mock_ticker_cls, wired):
        first = datetime(2018, 4, 2, 0, 0, 0)
        mock_ticker_cls.return_value.history.return_value = hist_frame(first)

        assert is_recent_ipo("MISS", FakeYamlConfig(make_config())) is False
        ipo_cache.get_cache().save()

        assert wired.exists()
        assert ipo_cache.IpoDateCache(wired).get("MISS") == first

    @patch("yfinance.Ticker")
    def test_A_FETCH_FAILURE_PERSISTS_NOTHING(self, mock_ticker_cls, wired):
        """THE load-bearing test.

        A transient network failure written to disk would be honoured forever,
        silently relaxing or tightening sell_criteria on that name for every
        future run. It must stay a per-run, in-memory fact.
        """
        mock_ticker_cls.return_value.history.side_effect = TimeoutError("read timed out")

        assert is_recent_ipo("FLAKY", FakeYamlConfig(make_config())) is False
        ipo_cache.get_cache().save()

        assert not wired.exists()
        assert ipo_cache.IpoDateCache(wired).get("FLAKY") is None

    @patch("yfinance.Ticker")
    def test_a_fetch_failure_does_not_corrupt_an_EXISTING_file(self, mock_ticker_cls, wired):
        seed = ipo_cache.IpoDateCache(wired)
        seed.put("KEEP", datetime(2001, 1, 1))
        seed.save()
        before = wired.read_bytes()
        ipo_cache.reset_cache()
        signals._ipo_date_cache.clear()
        mock_ticker_cls.return_value.history.side_effect = OSError("rate limited")

        assert is_recent_ipo("FLAKY", FakeYamlConfig(make_config())) is False
        ipo_cache.get_cache().save()

        assert wired.read_bytes() == before

    @patch("yfinance.Ticker")
    def test_NO_DATA_persists_nothing_either(self, mock_ticker_cls, wired):
        # yfinance answers a throttle with an empty frame and no exception, so
        # "no history" is not confirmed negative knowledge, it is ambiguous.
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        assert is_recent_ipo("NODATA", FakeYamlConfig(make_config())) is False
        ipo_cache.get_cache().save()

        assert not wired.exists()

    @patch("yfinance.Ticker")
    def test_a_failure_is_not_retried_within_the_same_run(self, mock_ticker_cls, wired):
        mock_ticker_cls.return_value.history.side_effect = TimeoutError("nope")

        is_recent_ipo("FLAKY", FakeYamlConfig(make_config()))
        is_recent_ipo("FLAKY", FakeYamlConfig(make_config()))

        assert mock_ticker_cls.return_value.history.call_count == 1

    @patch("yfinance.Ticker")
    def test_a_failure_IS_retried_on_the_next_run(self, mock_ticker_cls, wired):
        mock_ticker_cls.return_value.history.side_effect = TimeoutError("nope")
        is_recent_ipo("FLAKY", FakeYamlConfig(make_config()))
        ipo_cache.get_cache().save()

        # next run
        ipo_cache.reset_cache()
        signals._ipo_date_cache.clear()
        mock_ticker_cls.return_value.history.side_effect = None
        mock_ticker_cls.return_value.history.return_value = hist_frame(
            datetime.now() - timedelta(days=30)
        )

        assert is_recent_ipo("FLAKY", FakeYamlConfig(make_config())) is True

    @patch("yfinance.Ticker")
    def test_a_corrupt_cache_file_is_ignored_and_the_probe_still_runs(self, mock_ticker_cls, wired):
        wired.write_text('{"schema_version": 1, "dates": {"TRUNC')
        mock_ticker_cls.return_value.history.return_value = hist_frame(
            datetime.now() - timedelta(days=45)
        )

        assert is_recent_ipo("TRUNC", FakeYamlConfig(make_config())) is True
        assert mock_ticker_cls.return_value.history.call_count == 1

    @patch("yfinance.Ticker")
    def test_config_known_ipos_still_short_circuit_before_any_cache(self, mock_ticker_cls, wired):
        recent = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        cfg = FakeYamlConfig(make_config(known_ipos={"KNOWN": recent}))

        assert is_recent_ipo("KNOWN", cfg) is True
        assert mock_ticker_cls.return_value.history.call_count == 0
        assert not wired.exists()

    @patch("yfinance.Ticker")
    def test_auto_detect_disabled_never_touches_the_cache(self, mock_ticker_cls, wired):
        assert is_recent_ipo("X", FakeYamlConfig(make_config(auto_detect=False))) is False
        assert mock_ticker_cls.return_value.history.call_count == 0
        assert not wired.exists()


# --------------------------------------------------------------------------
# 7. verdict equivalence: cold and warm must agree, bit for bit
# --------------------------------------------------------------------------


class TestVerdictEquivalence:
    """``is_recent_ipo``'s bool is the ONLY channel this change can reach the
    model through (signals.py:1104-1123). If the bool is identical cold and
    warm, every downstream verdict is identical by construction."""

    @pytest.mark.parametrize("days_ago", [1, 30, 179, 359, 360, 361, 365, 400, 1000, 5000])
    @patch("yfinance.Ticker")
    def test_cold_and_warm_agree_across_the_grace_boundary(self, mock_ticker_cls, wired, days_ago):
        first = datetime.now() - timedelta(days=days_ago)
        mock_ticker_cls.return_value.history.return_value = hist_frame(first)

        cold = is_recent_ipo("BOUNDARY", FakeYamlConfig(make_config()))
        ipo_cache.get_cache().save()

        ipo_cache.reset_cache()
        signals._ipo_date_cache.clear()
        mock_ticker_cls.return_value.history.side_effect = AssertionError("warm run refetched")
        warm = is_recent_ipo("BOUNDARY", FakeYamlConfig(make_config()))

        assert warm is cold

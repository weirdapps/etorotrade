"""TDD — BUILD ③ (2026-07-18): offline, cache-backed sector resolution.

v3 sector data came ONLY from live yfinance ``.info["sector"]``, so when yfinance
throttled, ``sector`` went NaN -> ``_sector_demean`` collapsed to a single ``__NA__``
bucket (global demean = no-op) and the risk-gate sector cap became a single-bucket
no-op. The v2 static index map covers only ~14% of our 3,345-name universe, so the
fix is a 3-tier resolver: static index map (high trust) > persistent cache > live
yfinance (written back to the cache so coverage grows and runs are reproducible).
"""

from __future__ import annotations

from trade_modules.v3.sectors import (
    _read_cache,
    load_offline_sector_map,
    merge_offline_map,
    update_sector_cache,
)


def test_merge_static_wins_over_cache():
    m = merge_offline_map({"AAPL": "Technology"}, {"AAPL": "Energy", "XYZ": "Utilities"})
    assert m["AAPL"] == "Technology"  # static is authoritative
    assert m["XYZ"] == "Utilities"  # cache fills where static is silent


def test_merge_keys_are_uppercased():
    m = merge_offline_map({"aapl": "Technology"}, {"xyz": "Energy"})
    assert m["AAPL"] == "Technology"
    assert m["XYZ"] == "Energy"


def test_merge_drops_blank_values():
    m = merge_offline_map({"AAPL": ""}, {"XYZ": None})
    assert m == {}


def test_load_offline_map_merges_static_csv_and_cache(tmp_path):
    static = tmp_path / "market.csv"
    static.write_text("symbol,name,sector\nAAPL,Apple,Technology\n")
    cache = tmp_path / "cache.json"
    cache.write_text('{"XYZ": "Energy", "AAPL": "ShouldLoseToStatic"}')
    m = load_offline_sector_map(static_paths=(str(static),), cache_path=str(cache))
    assert m["AAPL"] == "Technology"  # static wins over cache
    assert m["XYZ"] == "Energy"


def test_update_cache_persists_only_non_static(tmp_path):
    static = tmp_path / "market.csv"
    static.write_text("symbol,name,sector\nAAPL,Apple,Technology\n")
    cache = tmp_path / "cache.json"
    added = update_sector_cache(
        {"AAPL": "Technology", "XYZ": "Energy"},
        cache_path=str(cache),
        static_paths=(str(static),),
    )
    assert added == 1  # AAPL is static -> not cached; only XYZ persisted
    c = _read_cache(str(cache))
    assert c["XYZ"] == "Energy"
    assert "AAPL" not in c


def test_update_cache_is_idempotent(tmp_path):
    static = tmp_path / "m.csv"
    static.write_text("symbol,name,sector\n")
    cache = tmp_path / "c.json"
    update_sector_cache({"XYZ": "Energy"}, cache_path=str(cache), static_paths=(str(static),))
    again = update_sector_cache(
        {"XYZ": "Energy"}, cache_path=str(cache), static_paths=(str(static),)
    )
    assert again == 0  # already cached, nothing new written


def test_update_cache_creates_missing_parent_dir(tmp_path):
    static = tmp_path / "m.csv"
    static.write_text("symbol,name,sector\n")
    cache = tmp_path / "deep" / "nested" / "c.json"
    added = update_sector_cache(
        {"XYZ": "Energy"}, cache_path=str(cache), static_paths=(str(static),)
    )
    assert added == 1
    assert cache.exists()


def test_update_cache_skips_blank_sectors(tmp_path):
    static = tmp_path / "m.csv"
    static.write_text("symbol,name,sector\n")
    cache = tmp_path / "c.json"
    added = update_sector_cache(
        {"XYZ": "", "ABC": None}, cache_path=str(cache), static_paths=(str(static),)
    )
    assert added == 0


def test_read_cache_missing_or_corrupt_returns_empty(tmp_path):
    assert _read_cache(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert _read_cache(str(bad)) == {}


def test_default_static_paths_point_at_repo_market_files():
    from trade_modules.v3.sectors import DEFAULT_STATIC_PATHS

    assert any(p.endswith("market.csv") for p in DEFAULT_STATIC_PATHS)
    assert any(p.endswith("usindex.csv") for p in DEFAULT_STATIC_PATHS)

"""TDD — BUILD ⑤ (2026-07-19): versioned, append-only price store (survivorship fix).

The v3 spine live-fetches prices every run with no disk backing, and delisted names
simply vanish from the universe -> full survivorship bias. This store keeps every
bar it has ever seen (delisting retention) and refreshes same-(date,ticker) closes
to the newest fetch (split/div adjustment changes history retroactively).
"""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.price_store import append_bars, read_close, store_coverage


def _wide(dates, data):
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def test_append_then_read_roundtrips(tmp_path):
    store = str(tmp_path / "s.parquet")
    px = _wide(["2026-01-01", "2026-01-02"], {"AAA": [10.0, 11.0], "BBB": [20.0, 21.0]})
    assert append_bars(px, store_path=store) == 4  # 4 new (date,ticker) pairs
    out = read_close(["AAA", "BBB"], store_path=store)
    assert out.loc["2026-01-02", "AAA"] == 11.0
    assert out.loc["2026-01-01", "BBB"] == 20.0


def test_append_is_additive_across_dates_and_tickers(tmp_path):
    store = str(tmp_path / "s.parquet")
    append_bars(_wide(["2026-01-01"], {"AAA": [10.0]}), store_path=store)
    append_bars(_wide(["2026-01-02"], {"AAA": [11.0], "CCC": [30.0]}), store_path=store)
    out = read_close(["AAA", "CCC"], store_path=store)
    assert out.loc["2026-01-01", "AAA"] == 10.0  # old date retained
    assert out.loc["2026-01-02", "CCC"] == 30.0  # new ticker added


def test_delisting_retention(tmp_path):
    store = str(tmp_path / "s.parquet")
    append_bars(_wide(["2026-01-01"], {"AAA": [10.0], "BBB": [20.0]}), store_path=store)
    # Next run: BBB has left the universe (delisted) — only AAA fetched.
    append_bars(_wide(["2026-01-02"], {"AAA": [11.0]}), store_path=store)
    out = read_close(["AAA", "BBB"], store_path=store)
    assert out.loc["2026-01-01", "BBB"] == 20.0  # delisted name's bar survives


def test_newest_wins_on_same_key(tmp_path):
    store = str(tmp_path / "s.parquet")
    append_bars(_wide(["2026-01-01"], {"AAA": [10.0]}), store_path=store)
    n = append_bars(_wide(["2026-01-01"], {"AAA": [10.5]}), store_path=store)  # adjustment refresh
    assert n == 0  # no NEW (date,ticker) pair
    out = read_close(["AAA"], store_path=store)
    assert out.loc["2026-01-01", "AAA"] == 10.5  # refreshed to newest


def test_long_format_input_accepted(tmp_path):
    store = str(tmp_path / "s.parquet")
    long = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-01"], "ticker": ["AAA", "BBB"], "close": [10.0, 20.0]}
    )
    assert append_bars(long, store_path=store) == 2
    assert read_close(["AAA"], store_path=store).loc["2026-01-01", "AAA"] == 10.0


def test_read_missing_ticker_and_empty_store(tmp_path):
    store = str(tmp_path / "s.parquet")
    assert read_close(["AAA"], store_path=store).empty  # store does not exist yet
    append_bars(_wide(["2026-01-01"], {"AAA": [10.0]}), store_path=store)
    assert read_close(["ZZZ"], store_path=store).empty  # ticker absent


def test_append_empty_is_noop(tmp_path):
    store = str(tmp_path / "s.parquet")
    assert append_bars(pd.DataFrame(), store_path=store) == 0
    assert store_coverage(store_path=store)["n_rows"] == 0


def test_store_coverage(tmp_path):
    store = str(tmp_path / "s.parquet")
    append_bars(
        _wide(["2026-01-01", "2026-01-02"], {"AAA": [10.0, 11.0], "BBB": [20.0, 21.0]}),
        store_path=store,
    )
    cov = store_coverage(store_path=store)
    assert cov["n_tickers"] == 2
    assert cov["n_dates"] == 2
    assert cov["n_rows"] == 4
    assert cov["first"] == "2026-01-01" and cov["last"] == "2026-01-02"


def test_nan_closes_dropped(tmp_path):
    store = str(tmp_path / "s.parquet")
    append_bars(_wide(["2026-01-01"], {"AAA": [10.0], "BBB": [float("nan")]}), store_path=store)
    assert read_close(["BBB"], store_path=store).empty  # NaN close not stored
    assert read_close(["AAA"], store_path=store).loc["2026-01-01", "AAA"] == 10.0

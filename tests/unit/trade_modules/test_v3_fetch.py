"""TDD tests for trade_modules.v3.fetch.robust_fetch_prices.

All tests use a FAKE downloader — no network calls, pause=0 for speed.
"""

import sys
import types

import numpy as np
import pandas as pd

from trade_modules.v3.fetch import _default_downloader, robust_fetch_prices


def test_default_downloader_maps_etoro_to_yahoo_and_renames_back(monkeypatch):
    """_default_downloader fetches Yahoo symbols (SBMO.NV -> SBMO.AS) but returns columns
    keyed by the original eToro tickers."""
    captured = {}

    def fake_download(batch, **kwargs):
        captured["batch"] = list(batch)
        idx = pd.date_range("2026-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["Close"], list(batch)])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    out = _default_downloader(["SBMO.NV", "7012.T"], "1mo")
    assert captured["batch"] == ["SBMO.AS", "7012.T"]  # fetched the mapped Yahoo symbols
    assert set(out.columns) == {"SBMO.NV", "7012.T"}  # renamed back to eToro tickers


def test_default_downloader_flat_single_ticker_layout(monkeypatch):
    """Legacy single-ticker flat 'Close' layout is handled and renamed back to eToro."""

    def fake_download(batch, **kwargs):
        idx = pd.date_range("2026-01-01", periods=3)
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 2.0}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    out = _default_downloader(["SBMO.NV"], "1mo")  # -> fetches SBMO.AS, renames back
    assert list(out.columns) == ["SBMO.NV"]
    assert float(out["SBMO.NV"].iloc[-1]) == 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_close(tickers: list[str], n: int = 5) -> pd.DataFrame:
    """Synthetic Close DataFrame for a batch of tickers."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    data = {t: rng.uniform(50, 150, n) for t in tickers}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# (a) Basic batching — returns synthetic frames, all tickers present
# ---------------------------------------------------------------------------


def test_single_batch_returns_all_tickers():
    """All tickers returned when they fit in one batch."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    calls: list[list[str]] = []

    def fake_dl(batch, period):
        calls.append(list(batch))
        return _make_close(batch)

    result = robust_fetch_prices(tickers, period="5y", batch_size=10, pause=0, downloader=fake_dl)

    assert set(result.columns) == set(tickers)
    assert len(calls) == 1
    assert calls[0] == tickers


def test_store_first_reads_store_and_live_fetches_only_missing(monkeypatch):
    """use_store=True: names in the price store come from it; only store-MISSING names
    are live-downloaded (the daily refresh keeps the store full -> no throttle)."""
    stored = _make_close(["A", "B"])

    def fake_read_close(tickers, **kw):
        cols = [t for t in ["A", "B"] if t in set(tickers)]
        return stored[cols] if cols else pd.DataFrame()

    monkeypatch.setattr("trade_modules.v3.price_store.read_close", fake_read_close)
    live_calls: list[list[str]] = []

    def fake_dl(batch, period):
        live_calls.append(list(batch))
        return _make_close(batch)

    out = robust_fetch_prices(["A", "B", "C"], pause=0, downloader=fake_dl, use_store=True)
    assert live_calls == [["C"]]  # only the store-missing name hit the network
    assert set(out.columns) == {"A", "B", "C"}  # store + live merged


def test_store_off_by_default_does_not_read_store(monkeypatch):
    """use_store=False (default): the store is never read; all names are live-fetched."""

    def boom(*a, **k):
        raise AssertionError("store must not be read when use_store is off")

    monkeypatch.setattr("trade_modules.v3.price_store.read_close", boom)
    out = robust_fetch_prices(
        ["A", "C"], pause=0, downloader=lambda b, p: _make_close(b), use_store=False
    )
    assert set(out.columns) == {"A", "C"}


def test_batching_correct_chunk_sizes():
    """7 tickers @ batch_size=3 → 3 calls with correct chunk sizes."""
    tickers = [f"T{i}" for i in range(7)]
    calls: list[list[str]] = []

    def fake_dl(batch, period):
        calls.append(list(batch))
        return _make_close(batch)

    result = robust_fetch_prices(tickers, period="5y", batch_size=3, pause=0, downloader=fake_dl)

    assert len(calls) == 3  # ceil(7/3) = 3
    assert len(calls[0]) == 3
    assert len(calls[1]) == 3
    assert len(calls[2]) == 1  # remainder
    assert set(result.columns) == set(tickers)


def test_result_is_dataframe_with_correct_index():
    """Output is a DataFrame with DatetimeIndex and ticker columns."""
    tickers = ["AAA", "BBB"]

    def fake_dl(batch, period):
        return _make_close(batch)

    result = robust_fetch_prices(tickers, period="5y", batch_size=10, pause=0, downloader=fake_dl)

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert set(result.columns) == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# (b) Retry — simulated rate-limit error on first attempt, success on retry
# ---------------------------------------------------------------------------


def test_retry_on_failure_then_success():
    """A batch that raises on attempt 0 succeeds on attempt 1; both tickers returned."""
    tickers = ["AAPL", "MSFT"]
    attempt_count = {"n": 0}

    def fake_dl(batch, period):
        attempt_count["n"] += 1
        if attempt_count["n"] == 1:
            raise RuntimeError("YFRateLimitError: rate limit exceeded")
        return _make_close(batch)

    result = robust_fetch_prices(
        tickers, period="5y", batch_size=10, pause=0, retries=3, downloader=fake_dl
    )

    assert set(result.columns) == set(tickers)
    assert attempt_count["n"] == 2  # failed once, succeeded on second attempt


def test_retry_exhausted_before_success_still_retries_n_times():
    """With retries=2, a failing batch is tried 3 times total (initial + 2 retries)."""
    tickers = ["X"]
    call_count = {"n": 0}

    def fake_dl(batch, period):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ValueError("transient")
        return _make_close(batch)  # succeeds on 3rd call

    result = robust_fetch_prices(
        tickers, period="5y", batch_size=10, pause=0, retries=2, downloader=fake_dl
    )

    assert "X" in result.columns
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Permanently failing batch — skipped, others present, no exception raised
# ---------------------------------------------------------------------------


def test_permanently_failing_batch_skipped_not_raised():
    """A batch that always fails is skipped; successful batches' tickers are returned."""
    tickers = ["GOOD1", "GOOD2", "BAD1", "BAD2"]

    def fake_dl(batch, period):
        if "BAD1" in batch:
            raise RuntimeError("YFRateLimitError: persistent failure")
        return _make_close(batch)

    # batch_size=2 → [GOOD1,GOOD2] then [BAD1,BAD2]
    result = robust_fetch_prices(
        tickers, period="5y", batch_size=2, pause=0, retries=2, downloader=fake_dl
    )

    assert "GOOD1" in result.columns
    assert "GOOD2" in result.columns
    assert "BAD1" not in result.columns
    assert "BAD2" not in result.columns


def test_all_batches_fail_returns_empty_dataframe():
    """If every batch fails, return empty DataFrame (no exception)."""
    tickers = ["A", "B"]

    def fake_dl(batch, period):
        raise RuntimeError("always fails")

    result = robust_fetch_prices(
        tickers, period="5y", batch_size=10, pause=0, retries=1, downloader=fake_dl
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# All-NaN columns dropped
# ---------------------------------------------------------------------------


def test_all_nan_columns_dropped():
    """Tickers whose prices are all NaN are dropped from the result."""
    tickers = ["AAPL", "EMPTY"]
    idx = pd.date_range("2025-01-01", periods=3, freq="B")

    def fake_dl(batch, period):
        df = pd.DataFrame(index=idx)
        for t in batch:
            df[t] = np.nan if t == "EMPTY" else [100.0, 101.0, 102.0]
        return df

    result = robust_fetch_prices(tickers, period="5y", batch_size=10, pause=0, downloader=fake_dl)

    assert "AAPL" in result.columns
    assert "EMPTY" not in result.columns


# ---------------------------------------------------------------------------
# Outer join across batches with different date ranges
# ---------------------------------------------------------------------------


def test_outer_join_across_batches():
    """Batches covering different date ranges are outer-joined on the date axis."""
    tickers = ["A", "B"]
    idx1 = pd.date_range("2025-01-01", periods=3, freq="B")
    idx2 = pd.date_range("2025-01-08", periods=3, freq="B")
    call_n = {"n": 0}

    def fake_dl(batch, period):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx1)
        return pd.DataFrame({"B": [200.0, 201.0, 202.0]}, index=idx2)

    result = robust_fetch_prices(tickers, period="5y", batch_size=1, pause=0, downloader=fake_dl)

    assert "A" in result.columns
    assert "B" in result.columns
    # Outer join includes dates from both batches (min 3, max 6 unique trading days)
    assert len(result) >= 3

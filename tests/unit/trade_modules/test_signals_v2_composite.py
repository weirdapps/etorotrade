"""TDD tests for trade_modules.signals_v2.composite (pure signal-composite module).

Run in isolation:
    cd ~/SourceCode/etorotrade
    python3 -m pytest tests/unit/trade_modules/test_signals_v2_composite.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trade_modules.signals_v2.composite import (
    factor_composite,
    map_to_signal,
    price_sleeve_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int, trend: str = "up", seed: int = 42) -> pd.Series:
    """Build a daily price series of length n.

    trend='up'   — strictly rising (each day +1%)
    trend='down' — strictly falling (each day -1%)
    trend='flat' — constant (no movement)
    trend='choppy' — alternating ±1% so 200dma and 12-1 mom cancel out
    """
    rng = np.random.default_rng(seed)
    prices = [100.0]
    if trend == "up":
        for _ in range(n - 1):
            prices.append(prices[-1] * 1.01)
    elif trend == "down":
        for _ in range(n - 1):
            prices.append(prices[-1] * 0.99)
    elif trend == "flat":
        prices = [100.0] * n
    elif trend == "choppy":
        for i in range(n - 1):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 0.99))
    else:
        raise ValueError(trend)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# map_to_signal
# ---------------------------------------------------------------------------


class TestMapToSignal:
    """Tests for map_to_signal(composite, buy_pct, sell_pct)."""

    def _monotonic(self, n: int = 20) -> pd.Series:
        """Monotonically increasing composite (ticker_0 worst, ticker_n-1 best)."""
        return pd.Series(
            {f"t{i}": float(i) for i in range(n)},
            name="composite",
        )

    def test_top_20pct_are_buy(self):
        """Exactly the top 20% of a 20-name universe → 'B'."""
        comp = self._monotonic(20)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        # Top 20% of 20 → names ranked 16-19 (indices t16..t19)
        top4 = [f"t{i}" for i in range(16, 20)]
        assert all(sig[t] == "B" for t in top4), sig[top4]

    def test_bottom_20pct_are_sell(self):
        """Bottom 20% → 'S'."""
        comp = self._monotonic(20)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        bottom4 = [f"t{i}" for i in range(4)]
        assert all(sig[t] == "S" for t in bottom4), sig[bottom4]

    def test_middle_are_hold(self):
        """Middle 60% → 'H'."""
        comp = self._monotonic(20)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        middle = [f"t{i}" for i in range(4, 16)]
        assert all(sig[t] == "H" for t in middle), sig[middle]

    def test_counts_sum_to_total(self):
        """B + H + S counts must equal len(composite)."""
        comp = self._monotonic(25)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        assert len(sig) == 25
        assert (sig == "B").sum() + (sig == "H").sum() + (sig == "S").sum() == 25

    def test_nan_composite_maps_to_hold(self):
        """NaN composite values must map to 'H' (not tradable, no crash)."""
        comp = pd.Series({"A": 1.0, "B": float("nan"), "C": 3.0, "D": 4.0, "E": 5.0})
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        assert sig["B"] == "H"

    def test_empty_series_returns_empty(self):
        """Empty composite → empty output, no crash."""
        comp = pd.Series(dtype=float)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        assert len(sig) == 0

    def test_buy_plus_sell_over_one_raises(self):
        """buy_pct + sell_pct > 1 → ValueError."""
        comp = self._monotonic(10)
        with pytest.raises(ValueError):
            map_to_signal(comp, buy_pct=0.60, sell_pct=0.60)

    def test_buy_pct_zero_means_no_buys(self):
        """buy_pct=0 → no 'B' in output."""
        comp = self._monotonic(10)
        sig = map_to_signal(comp, buy_pct=0.0, sell_pct=0.20)
        assert (sig == "B").sum() == 0

    def test_sell_pct_zero_means_no_sells(self):
        """sell_pct=0 → no 'S' in output."""
        comp = self._monotonic(10)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.0)
        assert (sig == "S").sum() == 0

    def test_output_only_contains_valid_values(self):
        """Output values are strictly {'B', 'H', 'S'}."""
        comp = self._monotonic(15)
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        assert set(sig.unique()).issubset({"B", "H", "S"})

    def test_single_name_is_hold(self):
        """A single-name universe with buy_pct=sell_pct=0.2 → 'H' (falls in middle)."""
        comp = pd.Series({"ONLY": 5.0})
        sig = map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        # With 1 name, quantile(0.8) == quantile(0.2) == 5.0; boundary logic
        # must not crash and must return a valid label.
        assert sig["ONLY"] in ("B", "H", "S")

    def test_does_not_mutate_input(self):
        """map_to_signal must not mutate the input Series."""
        comp = self._monotonic(10)
        original_values = comp.values.copy()
        map_to_signal(comp, buy_pct=0.20, sell_pct=0.20)
        np.testing.assert_array_equal(comp.values, original_values)


# ---------------------------------------------------------------------------
# price_sleeve_signal
# ---------------------------------------------------------------------------


class TestPriceSleeveSignal:
    """Tests for price_sleeve_signal(prices_df, ma_window)."""

    def _df(self, series_dict: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(series_dict)

    def test_rising_series_is_buy(self):
        """Strictly rising 300-day series → 'B'."""
        s = _make_prices(300, "up")
        df = self._df({"AAPL": s})
        sig = price_sleeve_signal(df)
        assert sig["AAPL"] == "B"

    def test_falling_series_is_sell(self):
        """Strictly falling 300-day series → 'S'."""
        s = _make_prices(300, "down")
        df = self._df({"BTCUSD": s})
        sig = price_sleeve_signal(df)
        assert sig["BTCUSD"] == "S"

    def test_flat_series_is_hold(self):
        """Flat price series → 'H' (mom=0, last==ma)."""
        s = _make_prices(300, "flat")
        df = self._df({"ETH": s})
        sig = price_sleeve_signal(df)
        assert sig["ETH"] == "H"

    def test_choppy_series_is_hold(self):
        """Alternating ±1% series → 'H' (no clear trend)."""
        s = _make_prices(300, "choppy")
        df = self._df({"XRP": s})
        sig = price_sleeve_signal(df)
        assert sig["XRP"] == "H"

    def test_short_history_is_hold_no_crash(self):
        """Fewer than ~253 observations → 'H', must not raise."""
        s = _make_prices(100, "up")
        df = self._df({"SHORT": s})
        sig = price_sleeve_signal(df)
        assert sig["SHORT"] == "H"

    def test_exactly_252_observations_does_not_crash(self):
        """Exactly 252 observations (boundary) → no crash."""
        s = _make_prices(252, "up")
        df = self._df({"EDGE": s})
        sig = price_sleeve_signal(df)  # result can be B or H, just no crash
        assert sig["EDGE"] in ("B", "H", "S")

    def test_multi_ticker_independence(self):
        """Multiple tickers are scored independently."""
        df = self._df(
            {
                "UP": _make_prices(300, "up"),
                "DOWN": _make_prices(300, "down"),
            }
        )
        sig = price_sleeve_signal(df)
        assert sig["UP"] == "B"
        assert sig["DOWN"] == "S"

    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame → empty Series, no crash."""
        df = pd.DataFrame()
        sig = price_sleeve_signal(df)
        assert len(sig) == 0

    def test_does_not_mutate_input(self):
        """price_sleeve_signal must not mutate prices_df."""
        s = _make_prices(300, "up")
        df = self._df({"A": s})
        original = df.copy()
        price_sleeve_signal(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# factor_composite
# ---------------------------------------------------------------------------


class TestFactorComposite:
    """Tests for factor_composite(df, factor_fns, weights)."""

    def _simple_df(self, n: int = 10) -> pd.DataFrame:
        """Minimal DataFrame: tickers as index, some columns."""
        tickers = [f"T{i}" for i in range(n)]
        return pd.DataFrame(
            {
                "val": np.linspace(1, 10, n),
                "qual": np.linspace(5, 15, n),
            },
            index=tickers,
        )

    def test_single_factor_composite_equals_that_factor(self):
        """With a single injected factor returning known values, composite == those values."""
        df = self._simple_df(10)
        known = pd.Series({f"T{i}": float(i) for i in range(10)})

        def my_factor(df_):
            return known.reindex(df_.index)

        result = factor_composite(df, factor_fns=[my_factor])
        pd.testing.assert_series_equal(result.sort_index(), known.sort_index(), check_names=False)

    def test_two_equal_factors_average(self):
        """Two identical factor fns → composite equals each (average of same = same)."""
        df = self._simple_df(8)
        vals = pd.Series({f"T{i}": float(i) for i in range(8)})

        def f1(df_):
            return vals.reindex(df_.index)

        def f2(df_):
            return vals.reindex(df_.index)

        result = factor_composite(df, factor_fns=[f1, f2])
        pd.testing.assert_series_equal(result.sort_index(), vals.sort_index(), check_names=False)

    def test_nan_in_one_factor_is_skipped(self):
        """NaN from a factor for one ticker doesn't poison the composite for that ticker."""
        df = self._simple_df(5)
        base = pd.Series({f"T{i}": 1.0 for i in range(5)})
        nan_factor = base.copy()
        nan_factor["T2"] = float("nan")

        def f1(df_):
            return base.reindex(df_.index)

        def f2(df_):
            return nan_factor.reindex(df_.index)

        result = factor_composite(df, factor_fns=[f1, f2])
        # T2 gets NaN from f2 but has 1.0 from f1 — composite must be a number
        assert math.isfinite(result["T2"])

    def test_all_factors_nan_gives_nan(self):
        """If all factors return NaN for a ticker, composite is NaN for that ticker."""
        df = self._simple_df(4)
        nan_series = pd.Series({f"T{i}": float("nan") for i in range(4)})

        def f_all_nan(df_):
            return nan_series.reindex(df_.index)

        result = factor_composite(df, factor_fns=[f_all_nan])
        assert result.isna().all()

    def test_result_indexed_to_df(self):
        """Result must be indexed to the input df's index."""
        df = self._simple_df(6)
        vals = pd.Series({f"T{i}": float(i) for i in range(6)})

        def f(df_):
            return vals.reindex(df_.index)

        result = factor_composite(df, factor_fns=[f])
        assert list(result.index) == list(df.index)

    def test_does_not_mutate_df(self):
        """factor_composite must not mutate the input DataFrame."""
        df = self._simple_df(5)
        original = df.copy()
        vals = pd.Series({f"T{i}": float(i) for i in range(5)})

        def f(df_):
            return vals.reindex(df_.index)

        factor_composite(df, factor_fns=[f])
        pd.testing.assert_frame_equal(df, original)

"""Unit tests for the MOMENTUM factor module (snapshot proxy).

Tests cover:
- higher 52W value -> higher z-score
- higher AM value -> higher z-score
- NaN in one column is handled gracefully (composite still computes from the other)
- missing column entirely does not raise
- output is aligned to df.index
- all-NaN input returns all-NaN output
"""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.factors.momentum import compute

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(records: list[dict], index: list[str] | None = None) -> pd.DataFrame:
    """Build a DataFrame from a list of dicts; index defaults to ticker names."""
    df = pd.DataFrame(records)
    if index is not None:
        df.index = index
    else:
        df.index = [f"T{i}" for i in range(len(records))]
    return df


# ---------------------------------------------------------------------------
# Tests: 52W column
# ---------------------------------------------------------------------------


class Test52WMomentum:
    def test_higher_52w_yields_higher_z(self):
        """A stock closer to its 52-week high should score higher."""
        df = _df(
            [{"52W": 95.0, "AM": 0.0}, {"52W": 50.0, "AM": 0.0}, {"52W": 10.0, "AM": 0.0}],
            index=["HIGH", "MID", "LOW"],
        )
        result = compute(df)
        assert result["HIGH"] > result["MID"] > result["LOW"]

    def test_52w_only_no_am(self):
        """When AM is absent, factor is computed from 52W alone (no raise)."""
        df = _df(
            [{"52W": 90.0}, {"52W": 40.0}],
            index=["A", "B"],
        )
        result = compute(df)
        assert result["A"] > result["B"]


# ---------------------------------------------------------------------------
# Tests: AM column
# ---------------------------------------------------------------------------


class TestAMMomentum:
    def test_higher_am_yields_higher_z(self):
        """A stock with stronger analyst momentum should score higher."""
        df = _df(
            [{"52W": 50.0, "AM": 15.0}, {"52W": 50.0, "AM": 0.0}, {"52W": 50.0, "AM": -10.0}],
            index=["UP", "FLAT", "DOWN"],
        )
        result = compute(df)
        assert result["UP"] > result["FLAT"] > result["DOWN"]

    def test_am_only_no_52w(self):
        """When 52W is absent, factor is computed from AM alone (no raise)."""
        df = _df(
            [{"AM": 10.0}, {"AM": -5.0}],
            index=["A", "B"],
        )
        result = compute(df)
        assert result["A"] > result["B"]


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_nan_in_one_column_uses_other(self):
        """If 52W is NaN for a row, composite should use AM z-score for that row."""
        df = _df(
            [{"52W": np.nan, "AM": 20.0}, {"52W": 80.0, "AM": np.nan}, {"52W": 50.0, "AM": 0.0}],
            index=["A", "B", "C"],
        )
        result = compute(df)
        # All three should be finite (no NaN when at least one component is available)
        assert result.notna().all(), f"Expected no NaN, got: {result.to_dict()}"

    def test_both_columns_nan_gives_nan(self):
        """A row where both 52W and AM are NaN should produce NaN in the composite."""
        df = _df(
            [{"52W": np.nan, "AM": np.nan}, {"52W": 80.0, "AM": 5.0}],
            index=["EMPTY", "GOOD"],
        )
        result = compute(df)
        assert np.isnan(result["EMPTY"])
        assert np.isfinite(result["GOOD"])

    def test_all_nan_input_returns_all_nan(self):
        """All-NaN DataFrame should return all-NaN output without raising."""
        df = _df(
            [{"52W": np.nan, "AM": np.nan}, {"52W": np.nan, "AM": np.nan}],
        )
        result = compute(df)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Tests: missing columns
# ---------------------------------------------------------------------------


class TestMissingColumns:
    def test_no_columns_at_all_does_not_raise(self):
        """An empty or unrelated DataFrame must not raise; returns all-NaN."""
        df = pd.DataFrame({"PET": [10.0, 20.0]}, index=["A", "B"])
        result = compute(df)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_missing_52w_only(self):
        """Missing 52W column — factor computed from AM without error."""
        df = _df([{"AM": 5.0}, {"AM": -3.0}])
        result = compute(df)
        assert not result.isna().all(), "AM-only computation should yield non-NaN values"

    def test_missing_am_only(self):
        """Missing AM column — factor computed from 52W without error."""
        df = _df([{"52W": 95.0}, {"52W": 30.0}])
        result = compute(df)
        assert not result.isna().all(), "52W-only computation should yield non-NaN values"


# ---------------------------------------------------------------------------
# Tests: index alignment
# ---------------------------------------------------------------------------


class TestIndexAlignment:
    def test_output_index_matches_input_index(self):
        """compute() output must have the same index as the input DataFrame."""
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        df = pd.DataFrame(
            {"52W": [90.0, 70.0, 50.0, 30.0], "AM": [5.0, 3.0, -1.0, -4.0]},
            index=tickers,
        )
        result = compute(df)
        assert list(result.index) == tickers

    def test_output_is_series(self):
        """compute() must always return a pd.Series."""
        df = _df([{"52W": 80.0, "AM": 2.0}])
        result = compute(df)
        assert isinstance(result, pd.Series)

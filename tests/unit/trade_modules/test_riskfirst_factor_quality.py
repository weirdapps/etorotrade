"""Tests for riskfirst.factors.quality — profitable, low-leverage, cash-generative.

QUALITY composite = mean of z-scored sub-metrics:
  zscore(ROE), zscore(-DE), zscore(FCF), zscore(EG)
Higher composite = higher quality.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.factors.quality import compute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _df(data: dict) -> pd.DataFrame:
    """Build a small ticker-indexed DataFrame from column dicts."""
    return pd.DataFrame(data, index=list(data[list(data.keys())[0]].keys()))


def _full_df():
    """Standard 4-ticker frame with all sub-metrics present."""
    return pd.DataFrame(
        {
            # HIGH quality: high ROE, LOW DE, high FCF, high EG
            "ROE": {"GOOD": 30.0, "BAD": 5.0, "MID1": 15.0, "MID2": 12.0},
            "DE": {"GOOD": 0.1, "BAD": 3.5, "MID1": 1.0, "MID2": 1.2},
            "FCF": {"GOOD": 0.12, "BAD": 0.01, "MID1": 0.06, "MID2": 0.05},
            "EG": {"GOOD": 25.0, "BAD": 2.0, "MID1": 10.0, "MID2": 8.0},
        }
    )


# ---------------------------------------------------------------------------
# Core ordering test
# ---------------------------------------------------------------------------


class TestQualityOrdering:
    def test_good_scores_higher_than_bad(self):
        """GOOD (high-ROE/low-DE/high-FCF/high-EG) must outrank BAD."""
        df = _full_df()
        result = compute(df)
        assert result["GOOD"] > result["BAD"]

    def test_returns_series_aligned_to_index(self):
        df = _full_df()
        result = compute(df)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(df.index)

    def test_all_tickers_present(self):
        df = _full_df()
        result = compute(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# DE inversion: lower DE must contribute a higher z-score
# ---------------------------------------------------------------------------


class TestDeInversion:
    def test_lower_de_gets_higher_composite(self):
        """When only DE varies, the firm with lower DE should rank higher."""
        df = pd.DataFrame(
            {
                "ROE": {"LOW_DE": 15.0, "HIGH_DE": 15.0},
                "DE": {"LOW_DE": 0.2, "HIGH_DE": 4.0},
                "FCF": {"LOW_DE": 0.08, "HIGH_DE": 0.08},
                "EG": {"LOW_DE": 10.0, "HIGH_DE": 10.0},
            }
        )
        result = compute(df)
        assert result["LOW_DE"] > result["HIGH_DE"]

    def test_de_column_negated_before_zscore(self):
        """Pure DE-only frame: the ticker with the LOWER DE gets a POSITIVE z."""
        df = pd.DataFrame(
            {
                "DE": {"A": 0.5, "B": 2.5, "C": 5.0},
            }
        )
        result = compute(df)
        # A has lowest DE -> negated DE is highest -> positive z
        assert result["A"] > 0
        # C has highest DE -> negated DE is lowest -> negative z
        assert result["C"] < 0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_nan_in_one_column_uses_remaining(self):
        """A ticker missing EG should still get a score from ROE/DE/FCF."""
        df = pd.DataFrame(
            {
                "ROE": {"A": 20.0, "B": 10.0},
                "DE": {"A": 0.5, "B": 1.5},
                "FCF": {"A": 0.10, "B": 0.05},
                "EG": {"A": np.nan, "B": np.nan},
            }
        )
        result = compute(df)
        assert not result.isna().any()

    def test_all_nan_ticker_returns_nan(self):
        """A ticker that has NaN in every sub-metric gets NaN output."""
        df = pd.DataFrame(
            {
                "ROE": {"A": 20.0, "B": np.nan},
                "DE": {"A": 0.5, "B": np.nan},
                "FCF": {"A": 0.10, "B": np.nan},
                "EG": {"A": 12.0, "B": np.nan},
            }
        )
        result = compute(df)
        assert not np.isnan(result["A"])
        assert np.isnan(result["B"])

    def test_partial_nan_row_uses_available_columns(self):
        """If only 2 of 4 columns available for a ticker, mean over those 2."""
        df = pd.DataFrame(
            {
                "ROE": {"X": 25.0, "Y": 5.0, "Z": np.nan},
                "DE": {"X": 0.2, "Y": 3.0, "Z": np.nan},
                "FCF": {"X": 0.15, "Y": 0.02, "Z": 0.08},
                "EG": {"X": 20.0, "Y": 3.0, "Z": np.nan},
            }
        )
        result = compute(df)
        # Z has only FCF; should still return a numeric value
        assert not np.isnan(result["Z"])


# ---------------------------------------------------------------------------
# Missing column resilience
# ---------------------------------------------------------------------------


class TestMissingColumns:
    def test_missing_one_column_does_not_raise(self):
        """Drop FCF entirely — compute must not raise."""
        df = pd.DataFrame(
            {
                "ROE": {"A": 20.0, "B": 10.0},
                "DE": {"A": 0.5, "B": 2.0},
                "EG": {"A": 15.0, "B": 5.0},
                # FCF absent
            }
        )
        result = compute(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_all_columns_missing_returns_all_nan(self):
        """If none of the sub-metric columns exist, every ticker gets NaN."""
        df = pd.DataFrame({"OTHER": {"A": 1.0, "B": 2.0}})
        result = compute(df)
        assert result.isna().all()

    def test_empty_dataframe_returns_empty_series(self):
        """Empty input -> empty output, no exception."""
        df = pd.DataFrame(columns=["ROE", "DE", "FCF", "EG"])
        result = compute(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_extra_columns_are_ignored(self):
        """Columns beyond the 4 sub-metrics must be silently ignored."""
        df = _full_df()
        df["EXTRA"] = 999.0
        result = compute(df)
        assert set(result.index) == set(df.index)


# ---------------------------------------------------------------------------
# Index alignment
# ---------------------------------------------------------------------------


class TestIndexAlignment:
    def test_output_index_matches_input(self):
        tickers = ["TSLA", "NVDA", "AAPL", "MSFT"]
        df = pd.DataFrame(
            {
                "ROE": {t: float(i * 5) for i, t in enumerate(tickers)},
                "DE": {t: float(i * 0.5) for i, t in enumerate(tickers)},
                "FCF": {t: float(i * 0.03) for i, t in enumerate(tickers)},
                "EG": {t: float(i * 8) for i, t in enumerate(tickers)},
            }
        )
        result = compute(df)
        assert list(result.index) == tickers

    def test_single_row_returns_zero(self):
        """Single ticker: z-scores are 0 (constant cross-section) -> composite 0."""
        df = pd.DataFrame(
            {
                "ROE": {"SOLO": 15.0},
                "DE": {"SOLO": 1.0},
                "FCF": {"SOLO": 0.05},
                "EG": {"SOLO": 10.0},
            }
        )
        result = compute(df)
        assert result["SOLO"] == pytest.approx(0.0)

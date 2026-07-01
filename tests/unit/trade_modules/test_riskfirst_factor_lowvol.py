"""Tests for riskfirst.factors.lowvol — low-volatility factor (beta proxy).

The low-vol anomaly: lower risk (beta) is rewarded.
Contract: compute(df) -> pd.Series aligned to df.index, HIGHER = MORE ATTRACTIVE.
Implementation: zscore(-B) so lower beta -> higher z-score.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.factors.lowvol import compute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _df(betas, index=None):
    """Build a minimal DataFrame with a 'B' (beta) column."""
    idx = index or [f"T{i}" for i in range(len(betas))]
    return pd.DataFrame({"B": betas}, index=idx)


# ---------------------------------------------------------------------------
# Core sign test: lower beta -> higher z-score
# ---------------------------------------------------------------------------


class TestCoreSign:
    def test_lower_beta_yields_higher_score(self):
        """The whole point of the factor: low-vol stocks score higher."""
        df = _df([0.5, 1.0, 1.5], index=["LOW", "MID", "HIGH"])
        z = compute(df)
        assert z["LOW"] > z["MID"] > z["HIGH"]

    def test_negative_beta_is_most_attractive(self):
        """Negative-beta asset (inverse fund, gold-like) ranks highest."""
        df = _df([-0.5, 0.5, 1.5], index=["NEG", "MID", "HIGH"])
        z = compute(df)
        assert z["NEG"] > z["MID"] > z["HIGH"]

    def test_scores_centred_near_zero(self):
        """Z-scores should average near zero (NaN-safe)."""
        df = _df([0.3, 0.7, 1.0, 1.3, 1.7])
        z = compute(df)
        assert z.mean() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NaN handling in 'B' column
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_nan_beta_stays_nan(self):
        df = _df([0.5, np.nan, 1.5], index=["A", "B", "C"])
        z = compute(df)
        assert np.isnan(z["B"])

    def test_valid_tickers_unaffected_by_nan_peer(self):
        df = _df([0.5, np.nan, 1.5], index=["A", "B", "C"])
        z = compute(df)
        assert z["A"] > z["C"]  # ordering holds despite NaN in B
        assert not np.isnan(z["A"])
        assert not np.isnan(z["C"])

    def test_all_nan_betas_returns_all_nan(self):
        df = _df([np.nan, np.nan, np.nan])
        z = compute(df)
        assert z.isna().all()


# ---------------------------------------------------------------------------
# Missing 'B' column — must not raise, return all-NaN aligned to index
# ---------------------------------------------------------------------------


class TestMissingColumn:
    def test_no_B_column_returns_all_nan(self):
        df = pd.DataFrame({"PRC": [10.0, 20.0, 30.0]}, index=["X", "Y", "Z"])
        z = compute(df)
        assert z.isna().all()

    def test_no_B_column_index_aligned(self):
        df = pd.DataFrame({"PRC": [1.0, 2.0]}, index=["X", "Y"])
        z = compute(df)
        assert list(z.index) == ["X", "Y"]

    def test_no_B_column_does_not_raise(self):
        df = pd.DataFrame({"CAP": ["1T", "500B"]}, index=["A", "B"])
        compute(df)  # should complete silently


# ---------------------------------------------------------------------------
# Index alignment
# ---------------------------------------------------------------------------


class TestIndexAlignment:
    def test_output_index_matches_input(self):
        idx = ["MSFT", "AAPL", "NVDA", "TSLA"]
        df = _df([0.9, 1.1, 1.5, 2.0], index=idx)
        z = compute(df)
        assert list(z.index) == idx

    def test_single_row_returns_zero(self):
        """A single-element series has zero variance; zscore returns 0.0."""
        df = _df([1.2], index=["SOLO"])
        z = compute(df)
        assert z["SOLO"] == pytest.approx(0.0)

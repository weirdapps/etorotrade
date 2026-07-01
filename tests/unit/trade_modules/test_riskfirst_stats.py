"""Tests for riskfirst.stats — winsorize + NaN-safe cross-sectional z-score."""

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.stats import winsorize, zscore


def test_winsorize_pulls_in_both_extremes():
    s = pd.Series([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 200.0])
    w = winsorize(s, lower=0.05, upper=0.95)
    assert w.min() > -100.0
    assert w.max() < 200.0
    # interior values are untouched
    assert w.iloc[5] == pytest.approx(5.0)


def test_zscore_standardises_clean_series():
    z = zscore(pd.Series([1.0, 2, 3, 4, 5]), winsor=False)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    assert z.iloc[2] == pytest.approx(0.0)  # the median maps to 0


def test_zscore_is_nan_safe():
    z = zscore(pd.Series([1.0, np.nan, 3.0, 5.0]), winsor=False)
    assert np.isnan(z.iloc[1])  # NaN in -> NaN out
    assert z.dropna().mean() == pytest.approx(0.0, abs=1e-9)


def test_zscore_constant_series_returns_zeros_not_nan():
    z = zscore(pd.Series([7.0, 7.0, 7.0]), winsor=False)
    assert (z == 0).all()  # zero std must not produce inf/nan


def test_zscore_preserves_index():
    s = pd.Series([1.0, 2, 3], index=["AAPL", "MSFT", "NVDA"])
    z = zscore(s, winsor=False)
    assert list(z.index) == ["AAPL", "MSFT", "NVDA"]

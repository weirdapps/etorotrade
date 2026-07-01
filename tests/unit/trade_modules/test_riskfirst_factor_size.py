"""Tests for the SIZE factor module (small-cap tilt).

TDD: tests written first, implementation follows.
"""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.factors.size import compute


def _df(caps):
    """Build a minimal DataFrame with a CAP column from a list of strings."""
    return pd.DataFrame({"CAP": caps}, index=[f"T{i}" for i in range(len(caps))])


# --- core sign property ---


def test_smaller_cap_higher_z():
    """SMALLER cap must produce a HIGHER z-score (size premium)."""
    df = _df(["500M", "5B", "2T"])
    s = compute(df)
    # T0 (500M) < T1 (5B) < T2 (2T) => z(T0) > z(T1) > z(T2)
    assert s["T0"] > s["T1"] > s["T2"]


# --- CAP string parsing ---


def test_trillion_suffix():
    """'2T' parses correctly (should yield the lowest z in a mixed series)."""
    df = _df(["100M", "1B", "2T"])
    s = compute(df)
    assert s["T0"] > s["T1"] > s["T2"]


def test_billion_suffix():
    """Billion suffix parses correctly."""
    df = _df(["200M", "1B"])
    s = compute(df)
    assert s.notna().all()
    assert s["T0"] > s["T1"]


def test_million_suffix():
    """Million suffix parses correctly."""
    df = _df(["50M", "500M"])
    s = compute(df)
    assert s.notna().all()
    assert s["T0"] > s["T1"]


# --- guard: non-positive / NaN cap -> NaN ---


def test_zero_cap_yields_nan():
    """_parse_market_cap returns 0.0 for invalid input; size.compute must map that to NaN."""
    df = _df(["0", "1B", "5B"])
    s = compute(df)
    assert pd.isna(s["T0"])
    assert s.notna()["T1"] and s.notna()["T2"]


def test_invalid_cap_string_yields_nan():
    """Unparseable strings (e.g. '--') return 0.0 from _parse_market_cap -> NaN in factor."""
    df = _df(["--", "2B"])
    s = compute(df)
    assert pd.isna(s["T0"])
    assert pd.notna(s["T1"])


def test_nan_cap_yields_nan():
    """Explicit NaN in the CAP column propagates to NaN z-score."""
    df = pd.DataFrame({"CAP": [np.nan, "3B"]}, index=["T0", "T1"])
    s = compute(df)
    assert pd.isna(s["T0"])
    assert pd.notna(s["T1"])


def test_negative_cap_yields_nan():
    """Negative market cap (pathological) -> NaN."""
    df = pd.DataFrame({"CAP": [-1e9, 5e9]}, index=["T0", "T1"])
    s = compute(df)
    assert pd.isna(s["T0"])
    assert pd.notna(s["T1"])


# --- missing 'CAP' column does not raise ---


def test_missing_cap_column_returns_all_nan():
    """When the 'CAP' column is absent, compute() returns an all-NaN Series aligned to df.index."""
    df = pd.DataFrame({"PRC": [10.0, 20.0]}, index=["A", "B"])
    s = compute(df)
    assert isinstance(s, pd.Series)
    assert list(s.index) == ["A", "B"]
    assert s.isna().all()


# --- index alignment ---


def test_index_alignment():
    """Output index matches input df.index exactly."""
    idx = ["AAPL", "MSFT", "TSLA", "NVDA"]
    df = pd.DataFrame({"CAP": ["3T", "2T", "1T", "500B"]}, index=idx)
    s = compute(df)
    assert list(s.index) == idx


def test_single_row_no_crash():
    """A single-row df with one valid cap doesn't crash (zscore of constant series = 0)."""
    df = _df(["5B"])
    s = compute(df)
    assert isinstance(s, pd.Series)
    assert len(s) == 1

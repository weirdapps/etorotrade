"""Tests for the VALUE factor module.

TDD: tests written BEFORE implementation.

Coverage:
- cheaper names score higher than expensive ones
- NaN handling: missing PET still works via other sub-metrics
- missing column doesn't raise
- output index equals df.index
"""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.factors.value import compute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _df(data: dict) -> pd.DataFrame:
    """Build a DataFrame indexed by ticker from column dicts."""
    return pd.DataFrame(data, index=["CHEAP", "MID", "EXPENSIVE"])


def _basic_df() -> pd.DataFrame:
    """Three tickers with all four value columns populated."""
    return _df(
        {
            "PET": [10.0, 20.0, 50.0],  # cheap has low P/E → high earnings yield
            "PEF": [8.0, 18.0, 45.0],
            "FCF": [8.0, 4.0, 1.0],  # cheap has high FCF yield
            "P/S": [1.0, 3.0, 8.0],  # cheap has low P/S → high sales yield
        }
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_cheap_scores_higher_than_expensive():
    """A cheaper stock (low PET, high FCF) must outscore an expensive one."""
    df = _basic_df()
    result = compute(df)
    assert result["CHEAP"] > result["MID"]
    assert result["MID"] > result["EXPENSIVE"]


def test_output_index_matches_input():
    """Output Series index must equal df.index exactly."""
    df = _basic_df()
    result = compute(df)
    pd.testing.assert_index_equal(result.index, df.index)


def test_returns_series():
    """compute() must return a pandas Series."""
    df = _basic_df()
    result = compute(df)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_missing_pet_still_works():
    """If PET is NaN for one name, the other sub-metrics fill in."""
    df = _df(
        {
            "PET": [np.nan, 20.0, 50.0],  # CHEAP has no trailing P/E
            "PEF": [8.0, 18.0, 45.0],
            "FCF": [8.0, 4.0, 1.0],
            "P/S": [1.0, 3.0, 8.0],
        }
    )
    result = compute(df)
    # CHEAP should still score better than EXPENSIVE despite missing PET
    assert result["CHEAP"] > result["EXPENSIVE"]
    # No NaN in result despite missing input
    assert result.notna().all()


def test_all_pet_nan_uses_remaining_columns():
    """If the entire PET column is NaN, result still uses PEF/FCF/P·S."""
    df = _df(
        {
            "PET": [np.nan, np.nan, np.nan],
            "PEF": [8.0, 18.0, 45.0],
            "FCF": [8.0, 4.0, 1.0],
            "P/S": [1.0, 3.0, 8.0],
        }
    )
    result = compute(df)
    assert isinstance(result, pd.Series)
    assert result["CHEAP"] > result["EXPENSIVE"]


def test_negative_pet_treated_as_nan():
    """Negative P/E (loss-making) must produce NaN earnings yield for that row."""
    df = _df(
        {
            "PET": [-5.0, 20.0, 50.0],  # CHEAP has negative P/E → NaN yield
            "PEF": [8.0, 18.0, 45.0],
            "FCF": [8.0, 4.0, 1.0],
            "P/S": [1.0, 3.0, 8.0],
        }
    )
    result = compute(df)
    # Should not raise; CHEAP still benefits from other sub-metrics
    assert isinstance(result, pd.Series)
    assert result["CHEAP"] > result["EXPENSIVE"]


def test_zero_pet_treated_as_nan():
    """Zero P/E must not cause divide-by-zero; treated as NaN."""
    df = _df(
        {
            "PET": [0.0, 20.0, 50.0],
            "PEF": [8.0, 18.0, 45.0],
            "FCF": [8.0, 4.0, 1.0],
            "P/S": [1.0, 3.0, 8.0],
        }
    )
    result = compute(df)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Missing column resilience
# ---------------------------------------------------------------------------


def test_missing_pet_column_does_not_raise():
    """DataFrame with no PET column must not raise."""
    df = pd.DataFrame(
        {"PEF": [8.0, 18.0, 45.0], "FCF": [8.0, 4.0, 1.0], "P/S": [1.0, 3.0, 8.0]},
        index=["CHEAP", "MID", "EXPENSIVE"],
    )
    result = compute(df)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == ["CHEAP", "MID", "EXPENSIVE"]


def test_missing_fcf_column_does_not_raise():
    """DataFrame with no FCF column must not raise."""
    df = pd.DataFrame(
        {"PET": [10.0, 20.0, 50.0], "PEF": [8.0, 18.0, 45.0], "P/S": [1.0, 3.0, 8.0]},
        index=["CHEAP", "MID", "EXPENSIVE"],
    )
    result = compute(df)
    assert isinstance(result, pd.Series)


def test_all_columns_missing_returns_nan_series():
    """DataFrame with none of the value columns returns all-NaN Series."""
    df = pd.DataFrame({"PRC": [100.0, 200.0, 300.0]}, index=["A", "B", "C"])
    result = compute(df)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.isna().all()


def test_missing_ps_column_does_not_raise():
    """DataFrame with no P/S column must not raise."""
    df = pd.DataFrame(
        {"PET": [10.0, 20.0, 50.0], "PEF": [8.0, 18.0, 45.0], "FCF": [8.0, 4.0, 1.0]},
        index=["CHEAP", "MID", "EXPENSIVE"],
    )
    result = compute(df)
    assert isinstance(result, pd.Series)


def test_single_row_no_raise():
    """Single-row DataFrame must not raise (std = 0 path in zscore)."""
    df = pd.DataFrame(
        {"PET": [15.0], "PEF": [12.0], "FCF": [5.0], "P/S": [2.0]},
        index=["ONLY"],
    )
    result = compute(df)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == ["ONLY"]

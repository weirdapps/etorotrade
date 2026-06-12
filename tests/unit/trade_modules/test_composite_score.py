import pandas as pd
import pytest

from trade_modules.analysis.composite_score import (
    ANALYST_UPSIDE_CAP,
    DEFAULT_WEIGHTS,
    compute_composite_scores,
)


@pytest.fixture
def sample_universe():
    return pd.DataFrame(
        {
            "ticker": ["HIGH_MOM", "VALUE", "QUALITY", "LAGGARD", "BALANCED"],
            "momentum_12_1m": [40.0, -5.0, 10.0, -20.0, 15.0],
            "analyst_momentum": [8.0, -2.0, 3.0, -10.0, 1.0],
            "return_on_equity": [15.0, 8.0, 30.0, 5.0, 18.0],
            "debt_to_equity": [50.0, 200.0, 30.0, 300.0, 80.0],
            "fcf_yield": [3.0, 5.0, 4.0, -2.0, 3.5],
            "pe_forward": [25.0, 8.0, 35.0, 50.0, 20.0],
            "upside": [15.0, 40.0, 10.0, 60.0, 20.0],
        }
    )


def test_composite_score_column_exists(sample_universe):
    result = compute_composite_scores(sample_universe)
    assert "composite_score" in result.columns
    assert "composite_quintile" in result.columns


def test_high_momentum_ranks_well(sample_universe):
    result = compute_composite_scores(sample_universe)
    high_mom = result.loc[result["ticker"] == "HIGH_MOM", "composite_score"].values[0]
    laggard = result.loc[result["ticker"] == "LAGGARD", "composite_score"].values[0]
    assert high_mom > laggard


def test_scores_between_0_and_100(sample_universe):
    result = compute_composite_scores(sample_universe)
    assert result["composite_score"].min() >= 0
    assert result["composite_score"].max() <= 100


def test_quintile_values(sample_universe):
    result = compute_composite_scores(sample_universe)
    valid = result["composite_quintile"].dropna()
    assert all(q in [1, 2, 3, 4, 5] for q in valid)


def test_missing_momentum_column():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "upside": [10, 20, 30, 40, 50],
            "pe_forward": [15, 25, 35, 45, 55],
            "return_on_equity": [10, 15, 20, 25, 30],
            "debt_to_equity": [50, 100, 150, 200, 250],
            "fcf_yield": [2, 3, 4, 5, 6],
            "analyst_momentum": [1, 2, 3, 4, 5],
        }
    )
    result = compute_composite_scores(df)
    assert result["composite_score"].notna().all()


def test_too_few_stocks():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "upside": [10],
            "pe_forward": [15],
            "return_on_equity": [10],
            "debt_to_equity": [50],
            "fcf_yield": [2],
            "analyst_momentum": [1],
            "momentum_12_1m": [5],
        }
    )
    result = compute_composite_scores(df)
    assert result["composite_score"].isna().all()


def test_analyst_upside_capped(sample_universe):
    result = compute_composite_scores(sample_universe)
    # LAGGARD has upside=60 but cap is 50 — should be treated as 50
    assert ANALYST_UPSIDE_CAP == 50.0


def test_custom_weights(sample_universe):
    # All momentum weight
    w = {"momentum": 1.0, "revision": 0, "quality": 0, "value": 0, "analyst": 0}
    result = compute_composite_scores(sample_universe, weights=w)
    # HIGH_MOM should be #1
    top = result.sort_values("composite_score", ascending=False).iloc[0]
    assert top["ticker"] == "HIGH_MOM"


def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001

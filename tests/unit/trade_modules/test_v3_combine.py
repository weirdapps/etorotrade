"""TDD tests for trade_modules.v3.combine.compute_scores."""

import numpy as np
import pandas as pd

from trade_modules.v3.combine import CLUSTERS, compute_scores


def _base(index):
    """A features frame with all metric columns NaN, so each test isolates one."""
    cols = set()
    for members in CLUSTERS.values():
        cols.update(members)
    df = pd.DataFrame({c: [np.nan] * len(index) for c in cols}, index=index)
    df["sector"] = ["__NA__"] * len(index)
    return df


def test_value_metric_negation_cheap_ranks_high():
    """Low P/E (cheap) must produce a HIGH value_z (value metrics are negated)."""
    df = _base(["CHEAP", "MID", "RICH"])
    df["pe_trailing"] = [5.0, 15.0, 50.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["CHEAP", "value_z"] > scores.loc["MID", "value_z"]
    assert scores.loc["MID", "value_z"] > scores.loc["RICH", "value_z"]
    # cheap should be clearly positive, rich clearly negative
    assert scores.loc["CHEAP", "value_z"] > 0 > scores.loc["RICH", "value_z"]


def test_quality_metric_kept_high_is_good():
    """High ROE must produce a HIGH quality_z (quality metrics are kept)."""
    df = _base(["A", "B", "C"])
    df["roe"] = [10.0, 20.0, 40.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["C", "quality_z"] > scores.loc["A", "quality_z"]
    assert scores.loc["C", "quality_z"] > 0 > scores.loc["A", "quality_z"]


def test_lowvol_metric_negated_low_vol_is_good():
    """Low realized_vol must produce a HIGH lowvol_z (negated)."""
    df = _base(["CALM", "WILD"])
    df["realized_vol"] = [0.10, 0.60]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["CALM", "lowvol_z"] > scores.loc["WILD", "lowvol_z"]


def test_sector_neutral_flips_a_sector_leader_positive():
    """A globally below-average name that LEADS its sector flips positive under
    sector-neutral demeaning."""
    df = _base(["A", "B", "C", "D"])
    df["sector"] = ["Tech", "Tech", "Fin", "Fin"]
    # Tech ROE high in absolute terms; Fin low. C leads Fin but is globally low.
    df["roe"] = [50.0, 40.0, 20.0, 10.0]

    glob = compute_scores(df, sector_neutral=False)
    sn = compute_scores(df, sector_neutral=True)

    # Globally, C (roe=20, below the 30 mean) is negative on quality.
    assert glob.loc["C", "quality_z"] < 0
    # Sector-neutral: C leads Fin (20 > 10) -> positive.
    assert sn.loc["C", "quality_z"] > 0


def test_conviction_and_rank_produced():
    df = _base(["A", "B", "C", "D"])
    df["pe_trailing"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["mom_12_1"] = [0.30, 0.10, -0.05, -0.20]
    scores = compute_scores(df, sector_neutral=False)

    assert "conviction" in scores.columns
    assert "rank" in scores.columns
    # A is best on every axis -> rank 1, highest conviction.
    assert scores.loc["A", "rank"] == 1
    assert scores["conviction"].idxmax() == "A"
    # ranks are a permutation of 1..N
    assert sorted(int(r) for r in scores["rank"]) == [1, 2, 3, 4]
    # re-z'd conviction is ~mean 0
    assert abs(float(scores["conviction"].mean())) < 1e-6


def test_nans_tolerated():
    """Rows/metrics with NaN don't crash; clusters use present metrics only."""
    df = _base(["A", "B", "C"])
    df["pe_trailing"] = [5.0, np.nan, 40.0]  # B missing value metric
    df["roe"] = [30.0, 20.0, np.nan]  # C missing quality metric
    scores = compute_scores(df, sector_neutral=False)

    # B has no value metric -> value_z NaN, but still gets conviction from quality.
    assert pd.isna(scores.loc["B", "value_z"])
    assert np.isfinite(scores.loc["B", "conviction"])
    # C has no quality metric -> quality_z NaN, still ranked.
    assert pd.isna(scores.loc["C", "quality_z"])
    assert scores["rank"].notna().all()


def test_value_quality_weight_cap_about_55pct():
    """Value + Quality jointly carry ~55% of conviction weight."""
    from trade_modules.v3.combine import CLUSTER_WEIGHTS

    total = sum(CLUSTER_WEIGHTS.values())
    vq = CLUSTER_WEIGHTS["value_z"] + CLUSTER_WEIGHTS["quality_z"]
    assert abs(vq / total - 0.55) < 0.01

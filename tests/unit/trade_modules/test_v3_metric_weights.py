"""TDD — per-metric weighting inside clusters (owner taxonomy 2026-07-23).

The engine historically applied an EQUAL mean of the member metric-z's inside each
cluster, so an unequal per-metric weight (e.g. ROE at 2x the other quality metrics)
was invisible. These tests pin the new contract: ``METRIC_WEIGHTS`` is the SSOT of
per-metric conviction weight, ``CLUSTER_WEIGHTS`` is DERIVED from it (cluster = sum of
its members' weights; value is the one composite cluster), and a cluster-z is the
``METRIC_WEIGHTS``-weighted mean of the members PRESENT for a row (renormalized over
present, so a missing metric is never a penalty — same rule as the value recipe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.combine import (
    CLUSTER_WEIGHTS,
    CLUSTERS,
    METRIC_WEIGHTS,
    compute_scores,
)


def _base(index):
    """Feature frame with every cluster metric NaN; tests set only what they need."""
    cols = set()
    for members in CLUSTERS.values():
        cols.update(members)
    df = pd.DataFrame({c: [np.nan] * len(index) for c in cols}, index=index)
    df["sector"] = ["__NA__"] * len(index)
    return df


def test_cluster_weights_are_derived_from_metric_weights():
    """Each non-value cluster weight == the sum of its members' METRIC_WEIGHTS; value is
    the single composite cluster (its internal split is VALUE_GROUP_RECIPES)."""
    assert CLUSTER_WEIGHTS["quality_z"] == pytest.approx(0.25)  # roe .10 + fcf/gp/ni .05
    assert CLUSTER_WEIGHTS["growth_z"] == pytest.approx(0.10)  # earn_growth + earn_stability
    assert CLUSTER_WEIGHTS["momentum_z"] == pytest.approx(0.10)  # pct_52w_high
    assert CLUSTER_WEIGHTS["pead_z"] == pytest.approx(0.15)  # sue
    assert CLUSTER_WEIGHTS["trajectory_z"] == pytest.approx(0.15)  # earn_trajectory (PEF/PET)
    assert CLUSTER_WEIGHTS["strength_z"] == pytest.approx(0.10)  # analyst_mom
    assert CLUSTER_WEIGHTS["value_z"] == pytest.approx(0.15)  # composite
    assert CLUSTER_WEIGHTS["lowvol_z"] == pytest.approx(0.0)  # sizing only


def test_cluster_weights_sum_to_one():
    assert sum(CLUSTER_WEIGHTS.values()) == pytest.approx(1.0)


def test_metric_weights_plus_value_sum_to_one():
    assert sum(METRIC_WEIGHTS.values()) + CLUSTER_WEIGHTS["value_z"] == pytest.approx(1.0)


def test_roe_carries_double_the_intra_quality_weight_of_fcf():
    """ROE (0.10) is weighted 2x FCF (0.05) inside quality. With ROE and FCF perfectly
    anti-correlated across three names (gp_assets / net_issuance absent), the ROE-strong
    name must win quality_z — under the OLD equal mean the two extremes would TIE at 0."""
    idx = ["HI_ROE", "MID", "HI_FCF"]
    df = _base(idx)
    df["roe"] = [3.0, 2.0, 1.0]  # HI_ROE best ROE, HI_FCF worst
    df["fcf"] = [1.0, 2.0, 3.0]  # ...and the reverse for FCF
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["HI_ROE", "quality_z"] > s.loc["HI_FCF", "quality_z"]


def test_equal_weight_members_reduce_to_plain_mean():
    """Growth's two members are equal-weight (0.05 each), so the weighted mean must
    still equal the plain mean — two symmetric anti-correlated names tie at 0."""
    idx = ["A", "B", "C"]
    df = _base(idx)
    df["earn_growth"] = [3.0, 2.0, 1.0]
    df["earn_stability"] = [1.0, 2.0, 3.0]
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["A", "growth_z"] == pytest.approx(s.loc["C", "growth_z"], abs=1e-9)


def test_single_present_quality_metric_reproduces_its_own_z():
    """A name with only ROE present (fcf/gp/ni NaN) gets quality_z == its ROE-z: the
    weighted mean renormalizes over present members, so a lone metric is not diluted."""
    idx = ["A", "B", "C"]
    df = _base(idx)
    df["roe"] = [40.0, 20.0, 5.0]
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["A", "quality_z"] == pytest.approx(s.loc["A", "roe_z"], abs=1e-12)
    assert s.loc["C", "quality_z"] == pytest.approx(s.loc["C", "roe_z"], abs=1e-12)

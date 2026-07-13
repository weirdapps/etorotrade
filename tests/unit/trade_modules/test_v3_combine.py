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


def test_no_quote_type_column_means_all_eligible():
    """Abstract combiner frames (no quote_type) keep the full ranked cross-section."""
    df = _base(["A", "B", "C"])
    df["pe_trailing"] = [5.0, 15.0, 40.0]
    df["roe"] = [40.0, 20.0, 10.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores["eligible"].all()
    assert scores["rank"].notna().all()


def test_accruals_in_quality_negated_low_accruals_scores_higher():
    """Low (more negative) accruals = higher earnings quality = higher quality_z.

    accruals direction is -1 (negated), consistent with the existing quality
    metrics that penalise leverage (de: -1).  CLEAN has negative accruals
    (cash-flow exceeds reported income) which is the gold standard for
    earnings quality and must rank above DIRTY (high accruals = inflated
    earnings, lower quality).
    """
    df = _base(["CLEAN", "DIRTY"])
    df["accruals"] = [-0.10, 0.15]  # CLEAN: quality earner; DIRTY: accrual-heavy
    scores = compute_scores(df, sector_neutral=False)
    # CLEAN (low/negative accruals) must outscore DIRTY on quality_z.
    assert scores.loc["CLEAN", "quality_z"] > scores.loc["DIRTY", "quality_z"]
    # With only one metric, the direction must produce the right sign.
    assert scores.loc["CLEAN", "quality_z"] > 0
    assert scores.loc["DIRTY", "quality_z"] < 0


def test_eligibility_excludes_non_equity_and_dataless():
    """With quote_type present, ETFs and dataless names drop out of the ranking."""
    idx = ["EQ1", "EQ2", "ETF1", "DATALESS"]
    df = _base(idx)
    df["quote_type"] = ["EQUITY", "EQUITY", "ETF", "EQUITY"]
    df["price"] = [100.0, 50.0, 200.0, 30.0]
    df["pe_trailing"] = [10.0, 20.0, 15.0, 12.0]
    df["roe"] = [30.0, 20.0, 25.0, 18.0]
    df["mom_12_1"] = [0.20, 0.10, 0.15, 0.05]
    df["realized_vol"] = [0.20, 0.25, 0.22, 0.30]
    # DATALESS: strip its core data (no price/mom/vol, and its cluster members go NaN).
    df.loc["DATALESS", ["price", "pe_trailing", "roe", "mom_12_1", "realized_vol"]] = np.nan

    scores = compute_scores(df, sector_neutral=False)

    # Two data-complete equities are eligible and ranked.
    assert bool(scores.loc["EQ1", "eligible"]) is True
    assert bool(scores.loc["EQ2", "eligible"]) is True
    assert scores.loc["EQ1", "rank"] in (1, 2)
    # The ETF is excluded despite having full data (non-equity).
    assert bool(scores.loc["ETF1", "eligible"]) is False
    assert pd.isna(scores.loc["ETF1", "conviction"])
    assert pd.isna(scores.loc["ETF1", "rank"])
    # The dataless equity is excluded (insufficient data).
    assert bool(scores.loc["DATALESS", "eligible"]) is False
    assert pd.isna(scores.loc["DATALESS", "rank"])
    # Ranking is dense over the two eligible names only.
    assert sorted(int(r) for r in scores["rank"].dropna()) == [1, 2]


# --------------------------------------------------------------------------- #
# PEG removal (honor the user's EXCLUDE decision)
# --------------------------------------------------------------------------- #


def test_peg_removed_from_direction_and_value_cluster():
    """PEG must not participate in scoring: gone from DIRECTION and value_z."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert "peg" not in DIRECTION
    assert "peg" not in CLUSTERS["value_z"]
    # No cluster references peg any more.
    for members in CLUSTERS.values():
        assert "peg" not in members


# --------------------------------------------------------------------------- #
# IC-weighting
# --------------------------------------------------------------------------- #


def test_derive_cluster_weights_none_uses_fixed_weights():
    from trade_modules.v3.combine import CLUSTER_WEIGHTS, derive_cluster_weights

    assert derive_cluster_weights(None) == CLUSTER_WEIGHTS


def test_derive_cluster_weights_proportional_to_positive_ic():
    """With no cap binding, cluster weights are ∝ max(IC, 0), renormalized."""
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.1,
        "quality_z": 0.1,
        "momentum_z": 0.2,
        "lowvol_z": 0.3,
        "strength_z": 0.3,
    }
    w = derive_cluster_weights(ic)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # ordering follows IC, and Value+Quality (0.2) is below the 0.55 cap so it is
    # left untouched (weights equal the normalized IC).
    assert w["strength_z"] > w["momentum_z"] > w["value_z"]
    assert abs(w["momentum_z"] - 0.2) < 1e-9
    assert abs((w["value_z"] + w["quality_z"]) - 0.2) < 1e-9


def test_derive_cluster_weights_preserves_value_quality_cap():
    """IC that would over-weight Value+Quality is capped at 0.55, excess to the rest."""
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.5,
        "quality_z": 0.5,
        "momentum_z": 0.05,
        "lowvol_z": 0.05,
        "strength_z": 0.05,
    }
    w = derive_cluster_weights(ic)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    vq = w["value_z"] + w["quality_z"]
    assert abs(vq - 0.55) < 1e-9  # capped exactly at the joint cap
    # equal IC -> equal split inside the VQ bucket
    assert abs(w["value_z"] - w["quality_z"]) < 1e-9
    # the freed 0.45 is spread over the other three (equal IC -> equal split)
    assert abs(w["momentum_z"] - 0.15) < 1e-9


def test_derive_cluster_weights_clamps_negative_ic():
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": -0.5,  # negative IC -> zero weight
        "quality_z": 0.3,
        "momentum_z": 0.3,
        "lowvol_z": 0.2,
        "strength_z": 0.2,
    }
    w = derive_cluster_weights(ic)
    assert w["value_z"] == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_derive_cluster_weights_all_nonpositive_falls_back_to_fixed():
    from trade_modules.v3.combine import CLUSTER_WEIGHTS, derive_cluster_weights

    ic = dict.fromkeys(CLUSTER_WEIGHTS, -1.0)
    assert derive_cluster_weights(ic) == CLUSTER_WEIGHTS


def test_derive_cluster_weights_equal_spread_when_rest_ic_nonpositive():
    """Freed weight spreads equally when non-VQ clusters all have IC ≤ 0 (else branch).

    Triggers the ``else`` branch at combine.py ~line 118: VQ IC is high enough
    to push sum above the 0.55 cap, and the three remaining clusters (momentum,
    lowvol, strength) have IC ≤ 0 so their raw weights are all zero.  The freed
    weight (1 - 0.55 = 0.45) must be spread equally across the three clusters,
    giving each 0.15, and the VQ pair is still exactly at the 0.55 cap.
    """
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.6,
        "quality_z": 0.6,
        "momentum_z": -0.1,  # negative IC -> clamped to 0 (else branch triggered)
        "lowvol_z": 0.0,  # zero IC -> clamped to 0
        "strength_z": -0.2,  # negative IC -> clamped to 0
    }
    w = derive_cluster_weights(ic)

    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights sum {sum(w.values())} != 1.0"
    assert abs(w["value_z"] + w["quality_z"] - 0.55) < 1e-9, "VQ cap not exactly 0.55"

    expected_each = (1.0 - 0.55) / 3  # 0.45 / 3 = 0.15
    for c in ("momentum_z", "lowvol_z", "strength_z"):
        assert abs(w[c] - expected_each) < 1e-9, (
            f"{c}: got {w[c]:.6f}, expected {expected_each:.6f}"
        )


def test_ic_weighting_shifts_conviction_toward_high_ic_cluster():
    """A momentum-favoring IC vector flips the value-name / momentum-name ranking."""
    df = _base(["MOM_STAR", "VAL_STAR", "MID1", "MID2"])
    df["pe_trailing"] = [50.0, 5.0, 20.0, 25.0]  # VAL_STAR cheapest -> best value
    df["mom_12_1"] = [0.50, -0.20, 0.10, 0.05]  # MOM_STAR strongest momentum

    default = compute_scores(df, sector_neutral=False)
    # Fixed weights lean on Value (Value+Quality ~55%) -> the value name wins.
    assert default.loc["VAL_STAR", "rank"] < default.loc["MOM_STAR", "rank"]

    ic = compute_scores(
        df,
        sector_neutral=False,
        ic_weights={
            "momentum_z": 1.0,
            "value_z": 0.05,
            "quality_z": 0.05,
            "lowvol_z": 0.05,
            "strength_z": 0.05,
        },
    )
    # Momentum-heavy IC -> the momentum name now out-ranks the value name.
    assert ic.loc["MOM_STAR", "rank"] < ic.loc["VAL_STAR", "rank"]
